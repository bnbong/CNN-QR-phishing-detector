"""실험 매트릭스 오케스트레이션 (스펙 7·8절).

순서 원칙(스펙 12절 #14): **P0(카운트·누출 진단·라운드트립 게이트)를 통과하기 전에는
모델을 학습하지 않는다.** ``run_matrix``는 ``reports/stratum_counts.json``이 없으면
``FileNotFoundError``로 하드 실패한다(우회 옵션 없음).
"""

from __future__ import annotations

import copy
import json
import subprocess
import traceback
from dataclasses import is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from qrphish import baselines as bl
from qrphish.campaign import MAJOR_THRESHOLD as CAMPAIGN_MAJOR_THRESHOLD
from qrphish.campaign import SESOI as CAMPAIGN_SESOI
from qrphish.campaign import VERDICT_DEFINITIONS as CAMPAIGN_VERDICT_DEFINITIONS
from qrphish.campaign import campaign_verdict
from qrphish.dataset import (
    VERSION_SPECS,
    QRGridDataset,
    build_stratum,
    ec_const,
    load_stratum_arrays,
    stratum_tier,
    version_in_spec,
)
from qrphish.evaluate import (
    auroc as auroc_fn,
)
from qrphish.evaluate import (
    bootstrap_p_value,
    cluster_bootstrap_by_seed,
    holm,
    metrics_from_probs,
    paired_cluster_bootstrap_by_seed,
    paired_cluster_permutation_test,
    percentile_ci,
    pick_threshold,
    predict_probs,
    unpaired_delta_bootstrap_by_seed,
)
from qrphish.models import build_model
from qrphish.train import pick_device, set_seed, train_model

__all__ = [
    "condition_id",
    "run_p0",
    "run_matrix",
    "run_template_split",
    "run_campaign_holdout",
    "run_explain",
    "run_probes",
    "run_occlusion",
    "compare_conditions",
    "run_hypothesis_tests",
    "aggregate_reports",
    "reaggregate",
    "recompute_campaign_verdicts",
    "recompute_probe_summaries",
    "run_motifs",
    "run_transfer",
    "load_matrix",
]

# v2: text_upper_bound* 필드 폐기(decoded_text_reference*·gap_to_decoded_text로 대체),
#     auroc_ci_pooled → auroc_pooled/auroc_pooled_ci(시드 통합 클러스터 부트스트랩).
# v2 이후 pooling 규약 변경: 시드 예측을 (seed,row)로 이어 붙여 AUROC 하나를 내던 방식은
#     시드마다 점수 척도가 달라 값이 체계적으로 낮았다. 이제 auroc_pooled/auroc_pooled_ci는
#     **시드 층화**(그룹 리샘플 후 시드별 지표를 평균)로 계산하고 POOLING_MODE로 기록한다.
SCHEMA_VERSION = 2
# results.json / 검정 산출물에 남기는 집계 규약 표식.
POOLING_MODE = "seed_stratified"
# 참조 기준(char n-gram LR)의 test 예측 파일. H2 쌍체 검정에 쓴다.
CHARNGRAM_PREDS = "preds_test_charngram.npz"
# P0가 훑는 (length_match, length_bucket) 격자. 스펙 1.1의 폭을 그대로 쓴다.
P0_LENGTH_GRID = (("exact", 1), ("quantile", 5), ("none", 1))


def _length_bucket(cond: Any) -> int | None:
    """``length_match`` + ``length_bucket`` → 매칭 버킷 폭.

    exact는 1로 고정(config 검증이 강제한다), quantile은 config의 ``length_bucket``,
    none은 매칭 없음(None)이다.
    """
    lm = str(_get(cond, "length_match", "exact"))
    if lm == "none":
        return None
    if lm == "exact":
        return 1
    bucket = int(_get(cond, "length_bucket", 5) or 5)
    if bucket < 1:
        raise ValueError(f"condition.length_bucket은 1 이상이어야 한다 (got {bucket})")
    return bucket


# --------------------------------------------------------------------------- utils
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _deep_set(node: Any, dotted: str, value: Any) -> None:
    """가변 객체/dict용 점 표기 설정. frozen dataclass에는 쓰지 않는다."""
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p] if isinstance(node, dict) else getattr(node, p)
    if isinstance(node, dict):
        node[parts[-1]] = value
    else:
        setattr(node, parts[-1], value)


def apply_overrides(cfg: Any, overrides: dict | None) -> Any:
    """``{"condition.length_match": "none"}`` 형태의 점 표기 오버라이드를 적용한 복사본.

    Config는 frozen dataclass이므로 dict로 내렸다가 다시 올린다. 그 덕에 오버라이드도
    config 검증(``optimize=0`` 등 재논의 불가 항목)을 그대로 통과해야 한다.
    """
    if not overrides:
        return copy.deepcopy(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        from qrphish.config import from_dict, to_dict

        raw = to_dict(cfg)
        for k, v in overrides.items():
            _deep_set(raw, k, v)
        return from_dict(raw)
    new = copy.deepcopy(cfg)
    for k, v in overrides.items():
        _deep_set(new, k, v)
    return new


def git_sha(repo: Path | None = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo or Path.cwd()),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _cfg_snapshot(cfg: Any) -> dict:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        from dataclasses import asdict

        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return {"repr": repr(cfg)}


def condition_id(cfg: Any, extra: str | None = None) -> str:
    """스펙 7절: ``{url_mode}-{length_match}-{features}-{mask_mode}-{arch}[-{extra}]``."""
    cond = _get(cfg, "condition")
    qr = _get(cfg, "qr")
    model = _get(cfg, "model")
    parts = [
        str(_get(cond, "url_mode", "norm")),
        str(_get(cond, "length_match", "exact")),
        str(_get(cond, "features", "data_only")),
        str(_get(qr, "mask_mode", "fixed")),
        str(_get(model, "arch", "small_cnn")),
    ]
    if extra:
        parts.append(str(extra))
    return "-".join(parts)


def _out_root(cfg: Any) -> Path:
    return Path(str(_get(cfg, "output_dir", "artifacts")))


def _reports_dir(cfg: Any) -> Path:
    """config의 ``reports_dir``. 폴백을 두지 않는다 — 필드가 없으면 config 오류다."""
    d = _get(cfg, "reports_dir")
    if not d:
        raise ValueError("config에 reports_dir이 없다 (configs/base.yaml에 추가할 것)")
    path = Path(str(d))
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------- data prep
def _prepare_frame(
    cfg: Any, stratum: str, seed: int, *, frame: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict]:
    """load → path filter → 층 필터 → **그룹 분할 → split별 길이 매칭**.

    매칭을 분할보다 **뒤에** 두는 것이 핵심이다. 층 전체에서 한 번 맞춰봐야
    :func:`group_split`의 그룹 배정과 5% 상한 다운샘플이 행을 통째로 옮기거나 지우면서
    split 안의 길이 주변분포를 다시 깨뜨린다. 매칭은 행 제거만 하므로 순서를 바꿔도
    split 간 그룹 교집합 0은 유지된다.

    ``frame``이 주어지면 ``load_webphish`` 단계만 건너뛰고 **이후 절차는 완전히 같다**.
    외부 검증(F)이 쓰는 주입 지점이다. 외부용으로 이 절차를 복제하면 두 경로가 조용히
    갈라져 F-a 대 in-domain 비교가 무의미해진다(설계 7.1).
    """
    from qrphish.splits import (
        assert_length_matched,
        group_split,
        match_by_length_within_splits,
    )
    from qrphish.urls import load_webphish

    data = _get(cfg, "data")
    cond = _get(cfg, "condition")
    qr = _get(cfg, "qr")
    ec = ec_const(_get(qr, "ec", "L"))

    url_mode = str(_get(cond, "url_mode", "norm"))
    if url_mode not in ("raw", "norm"):
        raise ValueError(f"condition.url_mode는 'raw'|'norm'이어야 한다 (got {url_mode!r})")
    if frame is not None:
        df = frame.reset_index(drop=True).copy()
        stats = {"injected_frame": True, "n_rows": int(len(df))}
    else:
        df, stats = load_webphish(
            Path(str(_get(data, "csv_path"))),
            url_mode,  # type: ignore[arg-type]
            category_col=str(_get(data, "category_col", "Category")),
            url_col=str(_get(data, "url_col", "Data")),
            positive_label=str(_get(data, "positive_label", "spam")),
        )
    diag: dict = {"load_stats": stats, "n_loaded": int(len(df))}

    if _get(cond, "path_filter", False):
        df = df[df["path_depth"] >= 1].reset_index(drop=True)
        diag["n_after_path_filter"] = int(len(df))

    from qrphish.qrgen import natural_version

    if "version" not in df.columns:
        df = df.assign(version=[natural_version(u, ec) for u in df["url"]])
    # 해당 EC의 v40 용량을 넘는 URL은 어느 층에도 배정할 수 없다(version=None).
    # 층 필터보다 먼저 떨어내야 int() 변환에서 터지지 않는다.
    n_before_capacity = int(len(df))
    # pandas가 None을 NaN으로 승격시키므로 notna로 거른다.
    df = df[df["version"].notna()].reset_index(drop=True)
    diag["n_dropped_capacity"] = n_before_capacity - int(len(df))
    df = df[[version_in_spec(int(v), stratum) for v in df["version"]]].reset_index(drop=True)
    diag["n_in_stratum"] = int(len(df))
    if len(df) == 0:
        diag["n_after_match"] = 0
        return df, diag

    # --- 그룹 키 선택 훅 (설계 5절 G) -------------------------------------
    # "etld1"이면 1차 실험과 완전히 동일하다. "etld1_template"이면 eTLD+1과
    # URL 경로 템플릿 클러스터를 union-find로 합친 키로 group 컬럼을 갈아끼운다.
    # splits.py는 손대지 않는다 — group 값만 바뀌고 group_split은 그대로 쓴다.
    split_cfg = _get(cfg, "split")
    group_key = str(_get(split_cfg, "group_key", "etld1"))
    if group_key not in ("etld1", "etld1_template"):
        raise ValueError(
            f"split.group_key는 'etld1'|'etld1_template'이어야 한다 (got {group_key!r})"
        )
    diag["group_key"] = group_key
    if group_key == "etld1_template":
        from qrphish.templates import (
            combined_group_key,
            template_diagnostics,
            template_groups,
        )

        df = df.copy()
        df["group_etld1"] = df["group"]  # 원래 eTLD+1 키를 보존한다
        tpl = template_groups(
            df["url"].tolist(),
            threshold=float(_get(split_cfg, "template_threshold", 0.7)),
            k=int(_get(split_cfg, "template_shingle", 3)),
            min_template_tokens=int(_get(split_cfg, "min_template_tokens", 3)),
        )
        df["group"] = combined_group_key(df["group_etld1"].tolist(), tpl.tolist())
        diag["template"] = {
            "clusters": template_diagnostics(tpl.tolist(), df["label"].tolist()),
            "combined": template_diagnostics(df["group"].tolist(), df["label"].tolist()),
            "n_groups_etld1": int(df["group_etld1"].nunique()),
            "n_groups_after_union": int(df["group"].nunique()),
            "threshold": float(_get(split_cfg, "template_threshold", 0.7)),
            "shingle_k": int(_get(split_cfg, "template_shingle", 3)),
            "min_template_tokens": int(_get(split_cfg, "min_template_tokens", 3)),
        }

    ratios = tuple(_get(_get(cfg, "split"), "ratios", (0.70, 0.15, 0.15)))
    df, sdiag = group_split(
        df,
        ratios=ratios,
        seed=seed,
        max_group_frac=float(_get(_get(cfg, "split"), "max_group_frac", 0.05)),
    )
    diag["split"] = sdiag
    diag["n_after_split"] = int(len(df))

    bucket = _length_bucket(cond)
    df, lreport = match_by_length_within_splits(df, bucket, seed)
    df = df.reset_index(drop=True)
    diag["n_after_match"] = int(len(df))
    # 진단 리포트를 남기고, 동일성이 깨졌으면 여기서 하드 실패한다.
    diag["length_match"] = lreport
    assert_length_matched(df, bucket)
    return df, diag


# ------------------------------------------------------------------------------ P0
def run_p0(cfg: Any) -> dict:
    """P0.1~P0.4 산출물. 모델은 하나도 학습하지 않는다."""
    reports = _reports_dir(cfg)
    strata = list(_get(cfg, "strata", VERSION_SPECS))
    rules = _get(cfg, "stratum_rules")
    primary_min = int(_get(rules, "primary_min_per_class", 1000))
    secondary_min = int(_get(rules, "secondary_min_per_class", 300))
    seed = int((_get(cfg, "seed_list", [0]) or [0])[0])

    counts: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rules": {"primary_min_per_class": primary_min, "secondary_min_per_class": secondary_min},
        "cells": [],
    }
    split_diag: dict[str, Any] = {}
    pad_frames: dict[str, pd.DataFrame] = {}

    for length_match, bucket in P0_LENGTH_GRID:
        # length_bucket도 함께 내려야 한다. exact는 1로 고정이라 base.yaml이 1을 들고
        # 있는데, 그대로 quantile로 바꾸면 버킷 폭이 1인 "exact와 같은" 조건이 된다.
        cfg_lm = apply_overrides(
            cfg,
            {"condition.length_match": length_match, "condition.length_bucket": bucket},
        )
        for stratum in strata:
            try:
                df, diag = _prepare_frame(cfg_lm, stratum, seed)
            except Exception as exc:  # 층이 비었거나 데이터 문제 → 셀만 비우고 계속
                counts["cells"].append(
                    {"length_match": length_match, "stratum": stratum, "error": repr(exc)}
                )
                continue
            per_class = df["label"].value_counts().to_dict() if len(df) else {}
            per_class_min = int(min(per_class.values())) if per_class else 0
            counts["cells"].append(
                {
                    "length_match": length_match,
                    "stratum": stratum,
                    "n_total": int(len(df)),
                    "n_by_class": {str(k): int(v) for k, v in per_class.items()},
                    "per_class_min": per_class_min,
                    "tier": stratum_tier(per_class_min, primary_min, secondary_min),
                    "n_in_stratum_before_match": diag.get("n_in_stratum"),
                }
            )
            key = f"{length_match}/{stratum}"
            split_diag[key] = diag.get("split")
            if len(df):
                pad_frames[key] = _first_pad_frame(cfg_lm, df)

    (reports / "stratum_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (reports / "split_diagnostics.json").write_text(
        json.dumps(split_diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    png = _plot_first_pad(pad_frames, reports / "first_pad_codeword.png")
    gate = run_roundtrip_gate(cfg)
    (reports / "roundtrip_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "stratum_counts": counts,
        "split_diagnostics": split_diag,
        "first_pad_png": str(png) if png else None,
        "roundtrip_gate": gate,
    }


def _first_pad_frame(cfg: Any, df: pd.DataFrame, cap: int = 3000) -> pd.DataFrame:
    """누출 진단용 ``first_pad_codeword`` 표본(스펙 5절 주석)."""
    from qrphish.qrgen import encode

    qr = _get(cfg, "qr")
    ec = ec_const(_get(qr, "ec", "L"))
    mp = None if _get(qr, "mask_mode", "fixed") == "auto" else int(_get(qr, "mask_pattern", 0))
    sub = df.sample(n=min(cap, len(df)), random_state=0) if len(df) > cap else df
    rows = []
    for url, label in zip(sub["url"], sub["label"], strict=True):
        art = encode(url, ec=ec, mask_pattern=mp)
        rows.append({"label": int(label), "first_pad_codeword": int(getattr(art, "first_pad_codeword", -1))})
    return pd.DataFrame(rows)


def _plot_first_pad(frames: dict[str, pd.DataFrame], out: Path) -> Path | None:
    """P0.2 — 클래스별 first_pad_codeword 분포가 겹치는지 눈으로 확인."""
    if not frames:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(frames)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, key in zip(axes.ravel(), keys, strict=False):
        d = frames[key]
        for lab, name in ((0, "benign"), (1, "phishing")):
            v = d.loc[d["label"] == lab, "first_pad_codeword"].to_numpy()
            if v.size:
                ax.hist(v, bins=30, alpha=0.55, label=name, density=True)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("first_pad_codeword")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def run_roundtrip_gate(cfg: Any, n_urls: int = 200, versions=(1, 2, 3, 4, 5, 6)) -> dict:
    """P0.3 — 역매핑 신뢰성 게이트(스펙 2.3). 하나라도 실패하면 ``passed=False``."""
    import random as _random

    from qrphish.mapping import verify_roundtrip
    from qrphish.qrgen import encode

    ec = ec_const(_get(_get(cfg, "qr"), "ec", "L"))
    rng = _random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-./?=&_%"
    caps = {1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134}
    checked = 0
    failures: list[dict] = []
    for v in versions:
        cap = caps.get(v, 106)
        low = caps.get(v - 1, 0) + 1
        for _ in range(max(1, n_urls // len(versions))):
            length = rng.randint(low, cap)
            url = "".join(rng.choice(alphabet) for _ in range(length))
            for mask in range(8):
                art = encode(url, ec=ec, mask_pattern=mask, version=v)
                checked += 1
                if not verify_roundtrip(url, art):
                    failures.append({"url": url, "version": v, "mask": mask})
    return {"passed": not failures, "n_checked": checked, "n_failures": len(failures),
            "failures": failures[:20]}


# ------------------------------------------------------------------------- matrix
def load_matrix(path: Path | str = "configs/matrix.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _surviving_strata(cfg: Any) -> list[str]:
    """P0 카운트 표에서 dropped가 아닌 층만 고른다. 표가 없으면 config의 strata 전체."""
    f = _reports_dir(cfg) / "stratum_counts.json"
    strata = list(_get(cfg, "strata", VERSION_SPECS))
    if not f.exists():
        return strata
    counts = json.loads(f.read_text(encoding="utf-8"))
    lm = str(_get(_get(cfg, "condition"), "length_match", "exact"))
    keep = {
        c["stratum"]
        for c in counts.get("cells", [])
        if c.get("length_match") == lm and c.get("tier") in ("primary", "secondary")
    }
    return [s for s in strata if s in keep] or strata


def _loaders(ds_all: QRGridDataset, split: np.ndarray, batch_size: int):
    def mk(code: int, shuffle: bool):
        idx = np.nonzero(split == code)[0]
        sub = copy.copy(ds_all)
        sub.indices = idx
        return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0), idx

    tr, itr = mk(0, True)
    va, iva = mk(1, False)
    te, ite = mk(2, False)
    return (tr, itr), (va, iva), (te, ite)


def _decoded_text_reference(entry: dict) -> bool:
    """matrix 항목의 ``decoded_text_reference`` 플래그.

    옛 이름 ``text_upper_bound``는 "CNN이 넘으면 누출"이라는 잘못된 함의를 담고 있어
    폐기했다. char n-gram LR은 강한 텍스트 베이스라인일 뿐 Bayes 최적 분류기가 아니다.
    하위 호환은 두지 않고 하드 실패시킨다.
    """
    if "text_upper_bound" in entry:
        raise ValueError(
            "matrix 항목의 'text_upper_bound' 키는 폐기되었다. "
            "'decoded_text_reference'로 개명할 것 (하위 호환 없음)."
        )
    return bool(entry.get("decoded_text_reference", True))


def _load_seed_preds(
    cfg: Any, cid: str, stratum: str, seed: int, filename: str = "preds_test.npz"
) -> dict[str, np.ndarray] | None:
    """``preds_test.npz``(test 예측 y/p/group/url)를 읽는다. 없으면 None.

    ``filename``으로 참조 기준의 예측(``preds_test_charngram.npz``)도 같은 경로에서 읽는다.
    ``seed`` 배열이 파일에 없으면(옛 산출물) 파일 경로의 시드 번호로 채운다. 시드 층화
    집계가 어느 행이 어느 시드에서 나왔는지 알아야 하기 때문이다.
    """
    path = _out_root(cfg) / cid / stratum / f"seed{seed}" / filename
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in ("y", "p", "group", "url")}
        out["seed"] = (
            np.asarray(z["seed"], dtype=np.int64)
            if "seed" in z.files
            else np.full(out["y"].size, int(seed), dtype=np.int64)
        )
    return out


def _pool_preds(
    cfg: Any, cid: str, stratum: str, seeds, filename: str = "preds_test.npz"
) -> dict[str, np.ndarray] | None:
    """5시드의 test 예측을 하나로 잇는다.

    이어 붙이되 ``seed`` 배열을 함께 들고 다닌다. 집계는 시드 층화
    (:func:`qrphish.evaluate.cluster_bootstrap_by_seed`)로 하므로 어느 행이 어느 시드에서
    나왔는지가 필요하다. 리샘플 단위는 시드와 무관한 eTLD+1 그룹 id다.
    """
    loaded = [
        (int(s), _load_seed_preds(cfg, cid, stratum, int(s), filename)) for s in seeds
    ]
    parts: list[tuple[int, dict[str, np.ndarray]]] = [
        (s, d) for s, d in loaded if d is not None and d["y"].size
    ]
    if not parts:
        return None
    return {
        "y": np.concatenate([d["y"] for _, d in parts]),
        "p": np.concatenate([d["p"] for _, d in parts]),
        "group": np.concatenate([d["group"].astype(str) for _, d in parts]),
        "url": np.concatenate([d["url"].astype(str) for _, d in parts]),
        "seed": np.concatenate([np.asarray(d["seed"], dtype=np.int64) for _, d in parts]),
    }


def _run_one_seed(
    cfg: Any,
    stratum: str,
    seed: int,
    cond_dir: Path,
    entry: dict,
    *,
    frame: pd.DataFrame | None = None,
) -> dict:
    """한 (층, 시드) 실행 → per_seed 항목.

    ``frame``은 :func:`_prepare_frame`으로 그대로 넘어간다. F-b(외부 자체 학습)가
    학습 경로를 복제하지 않고 이 함수를 그대로 재사용하기 위한 주입 지점이다.
    """
    cond = _get(cfg, "condition")
    qr = _get(cfg, "qr")
    mcfg = _get(cfg, "model")
    ecfg = _get(cfg, "eval")

    df, diag = _prepare_frame(cfg, stratum, seed, frame=frame)
    if len(df) == 0:
        return {"seed": seed, "error": "empty stratum after matching"}

    seed_dir = cond_dir / stratum / f"seed{seed}"
    sm = build_stratum(
        df,
        stratum,
        cond,
        seed_dir,
        qr=qr,
        condition_id=cond_dir.name,
        rules=_get(cfg, "stratum_rules"),
        git_sha=git_sha(),
        length_match=diag.get("length_match"),
    )
    arr = load_stratum_arrays(seed_dir)
    meta = pd.read_parquet(seed_dir / "meta.parquet")

    label_shuffle = bool(entry.get("label_shuffle", False))
    split = arr["split"].astype(int)
    # 영가설 바닥(스펙 1.8 H1): train+val 라벨만 섞고 **test는 원 라벨로 둔다.**
    # test까지 섞으면 "실제 라벨을 맞추지 못한다"가 아니라 "섞인 라벨을 맞추지
    # 못한다"를 보게 되어 검정이 의미를 잃는다.
    ds = QRGridDataset(
        arr["X_packed"],
        arr["y"],
        int(arr["n"]),
        arr["versions"],
        int(arr["ec"]),
        features=str(_get(cond, "features", "data_only")),
        mask_mode=str(_get(qr, "mask_mode", "fixed")),
        shuffle_positions=bool(_get(cond, "shuffle_positions", False)),
        perm_seed=seed,
        label_shuffle_seed=(1000 + seed) if label_shuffle else None,
        label_shuffle_rows=(split != 2) if label_shuffle else None,
    )
    groups = meta["group"].to_numpy()

    (tr_loader, _), (va_loader, iva), (te_loader, ite) = _loaders(
        ds, split, int(_get(mcfg, "batch_size", 256))
    )

    set_seed(seed)
    model = build_model(str(_get(mcfg, "arch", "small_cnn")), ds.n, ds.canonical_data_mask)
    device = pick_device()
    tres = train_model(
        model,
        tr_loader,
        va_loader,
        lr=float(_get(mcfg, "lr", 1e-3)),
        weight_decay=float(_get(mcfg, "weight_decay", 1e-4)),
        max_epochs=int(_get(mcfg, "max_epochs", 60)),
        patience=int(_get(mcfg, "patience", 8)),
        class_weight=_get(mcfg, "class_weight", "balanced"),
        amp=bool(_get(mcfg, "amp", True)),
        seed=seed,
        device=device,
    )
    # 조건 스냅샷을 함께 남긴다. 로드할 때 현재 cfg의 features/mask_mode로 데이터셋을
    # 재구성하면 다른 조건의 체크포인트를 조용히 잘못 읽을 수 있다.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n": ds.n,
            "arch": str(_get(mcfg, "arch")),
            "condition": _cfg_snapshot(cond),
            "qr": _cfg_snapshot(qr),
        },
        seed_dir / "model.pt",
    )

    n_boot = int(_get(ecfg, "n_bootstrap", 2000))
    y_te, p_te = predict_probs(model, te_loader, device=device)
    test = metrics_from_probs(y_te, p_te, groups[ite], n_boot=n_boot, seed=seed,
                              threshold=tres.threshold)
    y_va, p_va = predict_probs(model, va_loader, device=device)

    # 쌍체 부트스트랩·풀링 CI를 위해 test 예측을 남긴다. url은 두 조건이 같은 split의
    # 같은 행을 보고 있는지(쌍체 가능 여부) 검증하는 키다.
    np.savez_compressed(
        seed_dir / "preds_test.npz",
        y=np.asarray(y_te, dtype=np.int64),
        p=np.asarray(p_te, dtype=np.float64),
        group=np.asarray(groups[ite], dtype=object).astype(str),
        url=meta["url"].to_numpy()[ite].astype(str),
        seed=np.full(len(ite), int(seed), dtype=np.int64),
    )

    out = {
        "seed": seed,
        "best_epoch": tres.best_epoch,
        "preds_test": f"seed{seed}/preds_test.npz",
        "threshold": tres.threshold,
        "test": {k: test[k] for k in ("auroc", "auprc", "f1", "acc", "auroc_ci", "f1_ci")},
        "val": {"auroc": tres.best_val_auroc},
        "group_perf_iqr": test["group_perf_iqr"],
        "stratum_tier": sm.tier,
        "n_after_match": diag.get("n_after_match"),
        "length_match": diag.get("length_match"),
    }

    # 베이스라인(같은 split 인덱스에서 실행)
    base_names = list(entry.get("baselines", []))
    if _decoded_text_reference(entry) and "charngram_lr" not in base_names:
        base_names = ["charngram_lr", *base_names]
    urls = meta["url"].to_numpy()
    # 베이스라인도 CNN과 같은 라벨을 봐야 비교가 성립한다(label_shuffle 포함).
    y_all = np.asarray(ds.y, dtype=np.int64)
    base_out: dict[str, Any] = {}
    for name in base_names:
        try:
            pv, pt = bl.get_baseline(name)(urls, y_all, split, meta=meta, seed=seed)
            thr = pick_threshold(y_all[iva], pv)
            base_out[name] = metrics_from_probs(
                y_all[ite], pt, groups[ite], n_boot=min(n_boot, 500), seed=seed, threshold=thr
            )
            if name == "charngram_lr":
                # H2(참조 기준 − CNN)를 쌍체로 검정하려면 참조 기준의 test 예측도
                # 필요하다. CNN과 같은 split·같은 행이므로 url 키로 대응이 확인된다.
                np.savez_compressed(
                    seed_dir / CHARNGRAM_PREDS,
                    y=np.asarray(y_all[ite], dtype=np.int64),
                    p=np.asarray(pt, dtype=np.float64),
                    group=np.asarray(groups[ite], dtype=object).astype(str),
                    url=meta["url"].to_numpy()[ite].astype(str),
                    seed=np.full(len(ite), int(seed), dtype=np.int64),
                )
        except NotImplementedError as exc:
            base_out[name] = {"skipped": str(exc)}
    out["baselines"] = base_out
    return out


def run_matrix(cfg: Any, phase: str, matrix_path: Path | str = "configs/matrix.yaml") -> list[dict]:
    """P1/P2 매트릭스 실행. ``results.json``이 이미 있으면 건너뛴다(중단·재개)."""
    matrix = load_matrix(matrix_path)
    entries = matrix.get(phase) or []
    if not entries:
        raise ValueError(f"matrix에 phase '{phase}'가 없다")
    counts_path = _reports_dir(cfg) / "stratum_counts.json"
    if not counts_path.exists():
        # 스펙 12절 #14: P0(카운트 표)를 통과하기 전에는 모델을 학습하지 않는다.
        # 우회 옵션은 두지 않는다.
        raise FileNotFoundError(
            f"{counts_path}가 없다. 학습 전에 `qrphish p0`를 먼저 실행해야 한다 (스펙 12절 #14)."
        )

    seeds = list(_get(cfg, "seed_list", [0]))
    results: list[dict] = []
    for entry in entries:
        cfg_e = apply_overrides(cfg, entry.get("overrides"))
        cid = condition_id(cfg_e, entry.get("extra"))
        cond_dir = _out_root(cfg_e) / cid
        strata = entry.get("strata") or _surviving_strata(cfg_e)
        for stratum in strata:
            rpath = cond_dir / stratum / "results.json"
            done = _completed_result(rpath, f"{cid}/{stratum}")
            if done is not None:
                print(f"[skip] {cid}/{stratum} (results.json 존재)")
                results.append(done)
                continue
            print(f"[run ] {phase} {cid}/{stratum}")
            # 한 조건·층이 터져도 매트릭스 전체가 죽으면 안 된다(3~4시간 실행에서 가장
            # 나쁜 결과다). 시드 단위로 잡고, 층 조립까지 실패하면 error results.json을
            # 남기고 다음 층으로 간다.
            per_seed = []
            for s in seeds:
                try:
                    per_seed.append(_run_one_seed(cfg_e, stratum, s, cond_dir, entry))
                except Exception as exc:
                    print(f"[ERROR] {phase} {cid}/{stratum} seed{s}: {exc!r}")
                    traceback.print_exc()
                    per_seed.append({"seed": s, "error": repr(exc)})
            try:
                res = _assemble_results(cfg_e, cid, stratum, phase, entry, per_seed)
            except Exception as exc:
                print(f"[ERROR] {phase} {cid}/{stratum} 집계 실패: {exc!r}")
                traceback.print_exc()
                res = _error_results(cfg_e, cid, stratum, phase, per_seed, repr(exc))
            if all("error" in s for s in per_seed):
                res["error"] = "모든 시드 실패"
                print(f"[ERROR] {phase} {cid}/{stratum}: 모든 시드 실패 — results.json에 기록하고 진행")
            rpath.parent.mkdir(parents=True, exist_ok=True)
            rpath.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
            results.append(res)
    return results


def _test_urls(df: pd.DataFrame) -> set[str]:
    return set(df.loc[df["split"] == "test", "url"].astype(str))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_template_split(cfg: Any, strata: list[str] | None = None) -> dict:
    """G(템플릿 단위 분할)의 **사전 진단**만 수행한다. 학습은 하지 않는다.

    P0와 같은 성격이다. 층마다 eTLD+1 분할과 템플릿 합집합 분할을 각각 돌려
    (1) 템플릿 클러스터 크기 분포, (2) 분할 후 클래스비·최대 그룹 점유율,
    (3) **두 분할의 test 집합 자카드**를 남긴다. (3)이 곧 "1차 test 행 중 몇 %가
    템플릿 분할에서 train으로 이동했는가"의 직접 측정이다.

    실제 재학습은 ``run_matrix(cfg, "P2")``의 ``template_split`` 항목이 담당한다
    (조건 id가 달라 1차 아티팩트를 덮어쓰지 않는다).

    산출물: ``reports/template_split/{stratum}/diagnostics.json``
    """
    out_root = _reports_dir(cfg) / "template_split"
    seeds = list(_get(cfg, "seed_list", [0]))
    targets = list(strata) if strata else _surviving_strata(cfg)
    cfg_etld1 = apply_overrides(cfg, {"split.group_key": "etld1"})
    cfg_tpl = apply_overrides(cfg, {"split.group_key": "etld1_template"})

    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "condition_id": condition_id(cfg),
        "seeds": [int(s) for s in seeds],
        "strata": {},
    }
    for stratum in targets:
        per_seed: list[dict] = []
        template_diag: dict | None = None
        for seed in seeds:
            try:
                df_e, diag_e = _prepare_frame(cfg_etld1, stratum, seed)
                df_t, diag_t = _prepare_frame(cfg_tpl, stratum, seed)
            except Exception as exc:  # 한 층이 터져도 나머지 층은 계속 본다
                print(f"[ERROR] template_split {stratum} seed{seed}: {exc!r}")
                traceback.print_exc()
                per_seed.append({"seed": int(seed), "error": repr(exc)})
                continue
            if template_diag is None:
                template_diag = diag_t.get("template")
            te, tt = _test_urls(df_e), _test_urls(df_t)
            per_seed.append(
                {
                    "seed": int(seed),
                    "etld1": _split_summary(df_e, diag_e),
                    "etld1_template": _split_summary(df_t, diag_t),
                    "test_set_jaccard": float(_jaccard(te, tt)),
                    "n_test_etld1": len(te),
                    "n_test_template": len(tt),
                    # 1차 test 행 중 템플릿 분할에서 test에 남지 않은 비율
                    "frac_stage1_test_rows_left_test": (
                        float(len(te - tt) / len(te)) if te else 0.0
                    ),
                }
            )
        ok = [r for r in per_seed if "error" not in r]
        entry: dict[str, Any] = {
            "template": template_diag,
            "per_seed": per_seed,
            "n_seeds_ok": len(ok),
        }
        if ok:
            entry["mean"] = {
                "test_set_jaccard": float(np.mean([r["test_set_jaccard"] for r in ok])),
                "frac_stage1_test_rows_left_test": float(
                    np.mean([r["frac_stage1_test_rows_left_test"] for r in ok])
                ),
                "n_after_match_etld1": float(np.mean([r["etld1"]["n"] for r in ok])),
                "n_after_match_template": float(
                    np.mean([r["etld1_template"]["n"] for r in ok])
                ),
                "max_group_frac_observed_template": float(
                    np.mean([r["etld1_template"]["max_group_frac_observed"] for r in ok])
                ),
            }
        summary["strata"][stratum] = entry
        d = out_root / stratum
        d.mkdir(parents=True, exist_ok=True)
        (d / "diagnostics.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _split_summary(df: pd.DataFrame, diag: dict) -> dict:
    """진단 JSON에 담을 분할 요약(전체 diag는 크므로 필요한 것만 뽑는다)."""
    sd = diag.get("split", {})
    return {
        "n": int(len(df)),
        "n_in_stratum": int(diag.get("n_in_stratum", 0)),
        "n_after_split": int(diag.get("n_after_split", 0)),
        "n_groups": int(sd.get("n_groups", 0)),
        "groups_disjoint": bool(sd.get("groups_disjoint", False)),
        "max_group_frac_observed": float(sd.get("max_group_frac_observed", 0.0)),
        "n_downsampled_groups": int(len(sd.get("downsampled_groups", []))),
        "top5_test_group_frac": float(sd.get("top5_test_group_frac", 0.0)),
        "pos_ratio": {
            name: float(info.get("pos_ratio", 0.0))
            for name, info in (sd.get("per_split") or {}).items()
        },
        "length_match_ok": bool(diag.get("length_match", {}).get("all_identical", True)),
    }


def _pooled_auroc(cfg, cid, stratum, seeds, n_boot: int) -> tuple[float, list[float]]:
    """시드 층화 그룹 클러스터 부트스트랩으로 ``auroc_pooled``/``auroc_pooled_ci``를 낸다.

    리샘플 단위는 eTLD+1 그룹이고, 지표는 시드별로 계산한 뒤 평균낸다. 그래서 점추정은
    ``auroc_mean``(시드별 AUROC의 평균)과 같은 정의다. 시드 예측을 그냥 이어 붙여
    AUROC 하나를 내던 옛 방식은 시드마다 점수 척도가 달라 값이 체계적으로 낮았다.
    """
    pooled = _pool_preds(cfg, cid, stratum, seeds)
    if pooled is None:
        return float("nan"), [float("nan"), float("nan")]
    cb = cluster_bootstrap_by_seed(
        pooled["y"], pooled["p"], pooled["group"], pooled["seed"],
        auroc_fn, n_boot=n_boot, seed=0,
    )
    return float(cb["point"]), [float(cb["lo"]), float(cb["hi"])]


def _assemble_results(cfg, cid, stratum, phase, entry, per_seed) -> dict:
    """스펙 8절 results.json 스키마."""
    ok = [s for s in per_seed if "error" not in s]
    aur = np.array([s["test"]["auroc"] for s in ok], dtype=float) if ok else np.array([])
    f1 = np.array([s["test"]["f1"] for s in ok], dtype=float) if ok else np.array([])
    acc = np.array([s["test"]["acc"] for s in ok], dtype=float) if ok else np.array([])

    # 디코딩 텍스트 참조 기준(강한 텍스트 베이스라인). 상한선이 아니므로 넘어도 누출이
    # 아니다. 부호 있는 갭(참조 − CNN)만 보고한다.
    text_ref = [
        s["baselines"]["charngram_lr"]["auroc"]
        for s in ok
        if "auroc" in s.get("baselines", {}).get("charngram_lr", {})
    ]
    text_mean = float(np.nanmean(text_ref)) if text_ref else float("nan")
    auroc_mean = float(np.nanmean(aur)) if aur.size else float("nan")

    # 5시드 test 예측을 모아 **시드 층화** 그룹 클러스터 부트스트랩으로 CI 하나를 만든다.
    # (시드별 CI의 하한/상한을 평균내던 옛 auroc_ci_pooled는 정식 CI가 아니라 폐기했다.)
    seeds = list(_get(cfg, "seed_list", [0]))
    n_boot = int(_get(_get(cfg, "eval"), "n_bootstrap", 2000))
    auroc_pooled, pooled_ci = _pooled_auroc(cfg, cid, stratum, seeds, n_boot)

    first = ok[0] if ok else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "condition_id": cid,
        "phase": phase,
        "stratum": stratum,
        "tier": first.get("stratum_tier", "unknown"),
        "config": _cfg_snapshot(cfg),
        "provenance": {
            "git_sha": git_sha(),
            "qrphish_version": _pkg_version_safe(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "data": _data_block(cfg, cid, stratum),
        "per_seed": per_seed,
        "aggregate": {
            "auroc_mean": auroc_mean,
            "auroc_sd_across_seeds": float(np.nanstd(aur, ddof=1)) if aur.size > 1 else 0.0,
            "auroc_pooled": auroc_pooled,
            "auroc_pooled_ci": pooled_ci,
            "pooling": POOLING_MODE,
            "f1_mean": float(np.nanmean(f1)) if f1.size else float("nan"),
            "acc_mean": float(np.nanmean(acc)) if acc.size else float("nan"),
            "decoded_text_reference_auroc": text_mean,
            "gap_to_decoded_text": float(text_mean - auroc_mean),
        },
        "sanity": {
            "label_shuffle_auroc": auroc_mean if entry.get("label_shuffle") else None,
        },
        "explain": {"cam_mass_by_kind": {}},
    }


def _error_results(cfg, cid, stratum, phase, per_seed, error: str) -> dict:
    """집계까지 실패한 조건·층의 최소 results.json. 스키마 키를 비워서라도 유지한다."""
    return {
        "schema_version": SCHEMA_VERSION,
        "condition_id": cid,
        "phase": phase,
        "stratum": stratum,
        "tier": "unknown",
        "error": error,
        "config": _cfg_snapshot(cfg),
        "provenance": {
            "git_sha": git_sha(),
            "qrphish_version": _pkg_version_safe(),
            "torch": torch.__version__,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "data": {},
        "per_seed": per_seed,
        "aggregate": {},
        "sanity": {"label_shuffle_auroc": None},
        "explain": {"cam_mass_by_kind": {}},
    }


def _pkg_version_safe() -> str:
    from qrphish.dataset import _pkg_version

    return _pkg_version()


def _data_block(cfg, cid, stratum) -> dict:
    """가장 낮은 시드의 stratum_meta에서 데이터 요약을 끌어온다."""
    base = _out_root(cfg) / cid / stratum
    cands = sorted(base.glob("seed*/stratum_meta.json"))
    if not cands:
        return {}
    sm = json.loads(cands[0].read_text(encoding="utf-8"))
    return {
        "n_total": sm["n_total"],
        "n_train": sm["n_by_split"]["train"],
        "n_val": sm["n_by_split"]["val"],
        "n_test": sm["n_by_split"]["test"],
        "class_ratio": sm["class_ratio"],
        "n_groups": sm["n_groups"],
        "top5_test_group_frac": sm.get("extra", {}).get("top5_test_group_frac", 0.0),
    }


# ------------------------------------------------------------------------ explain
# 2차 실험(review_01)의 새 phase들이 읽는 주 조건. 체크포인트는 Colab 실행 산출물이다.
MAIN_CONDITION = "norm-exact-data_only-fixed-small_cnn"


def _seed_dir(cfg: Any, cid: str, stratum: str, seed: int) -> Path:
    return _out_root(cfg) / cid / stratum / f"seed{seed}"


def _ckpt_dataset_spec(cfg: Any, ckpt: dict, path: Path) -> tuple[str, str]:
    """체크포인트의 조건 스냅샷에서 ``(features, mask_mode)``를 읽는다.

    스냅샷이 있으면 그것이 진실이다. 현재 cfg로 데이터셋을 재구성하면 다른 조건의
    체크포인트를 조용히 잘못 읽는다. 현재 cfg와 어긋나면 하드 실패시킨다. 스냅샷이 없는
    옛 체크포인트는 경고 없이 현재 cfg를 쓰되, 그 사실이 여기 한 곳에만 있게 한다.
    """
    cond, qr = _get(cfg, "condition"), _get(cfg, "qr")
    cfg_features = str(_get(cond, "features", "data_only"))
    cfg_mask = str(_get(qr, "mask_mode", "fixed"))
    snap_c, snap_q = ckpt.get("condition"), ckpt.get("qr")
    if not isinstance(snap_c, dict) or not isinstance(snap_q, dict):
        return cfg_features, cfg_mask
    features = str(snap_c.get("features", cfg_features))
    mask_mode = str(snap_q.get("mask_mode", cfg_mask))
    if (features, mask_mode) != (cfg_features, cfg_mask):
        raise ValueError(
            f"체크포인트의 조건 스냅샷이 현재 config와 다르다: {path}\n"
            f"  체크포인트: features={features}, mask_mode={mask_mode}\n"
            f"  현재 config: features={cfg_features}, mask_mode={cfg_mask}\n"
            "같은 조건의 config로 다시 부르거나, 해당 조건을 다시 학습할 것."
        )
    return features, mask_mode


def _load_trained(cfg: Any, cid: str, stratum: str, seed: int):
    """저장된 체크포인트 -> ``(model, dataset, arr, meta)``. 없으면 명확한 에러.

    체크포인트는 :func:`_run_one_seed`가 남긴 ``model.pt``
    (``{"state_dict", "n", "arch"}``)다. 새 phase들은 절대 새로 학습하지 않는다.
    """
    sd = _seed_dir(cfg, cid, stratum, seed)
    ckpt_path = sd / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt_path}\n"
            "P1 매트릭스(run_matrix)를 먼저 돌리거나, Colab에서 만든 artifacts를 "
            f"cfg.output_dir({_out_root(cfg)}) 아래로 가져와라."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arr = load_stratum_arrays(sd)
    meta = pd.read_parquet(sd / "meta.parquet")
    features, mask_mode = _ckpt_dataset_spec(cfg, ckpt, ckpt_path)
    ds = QRGridDataset(
        arr["X_packed"], arr["y"], int(arr["n"]), arr["versions"], int(arr["ec"]),
        features=features,
        mask_mode=mask_mode,
    )
    model = build_model(ckpt["arch"], ds.n, ds.canonical_data_mask)
    model.load_state_dict(ckpt["state_dict"])
    # 아래 phase들(explain/probes/occlusion)은 모델 파라미터의 장치를 보고 입력을 옮긴다.
    # 여기서 학습 때와 같은 장치로 올려두면 그 규약이 GPU에서도 그대로 성립한다.
    model.to(pick_device())
    model.eval()
    return model, ds, arr, meta


def _ckpt_arch(cfg: Any, cid: str, stratum: str, seed: int) -> str:
    """체크포인트에 기록된 arch 이름. 무작위 초기화 대조가 같은 구조를 쓰게 한다."""
    ckpt = torch.load(
        _seed_dir(cfg, cid, stratum, seed) / "model.pt", map_location="cpu", weights_only=False
    )
    return str(ckpt["arch"])


def _available_seeds(cfg: Any, cid: str, stratum: str) -> list[int]:
    root = _out_root(cfg) / cid / stratum
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("seed*")):
        if (p / "model.pt").exists():
            out.append(int(p.name[4:]))
    return out


def _explain_one_seed(
    cfg: Any, cid: str, stratum: str, seed: int, per_class: int, use_ig: bool, ig_steps: int,
) -> dict:
    """한 시드의 Grad-CAM 요약. 클래스 균형 표본 + 부호 있는 타깃 + enrichment."""
    from qrphish.explain import (
        attribute_by_kind,
        attribute_to_chars,
        balanced_sample,
        cam_correlation,
        enrichment_by_kind,
        gradcam_signed,
        ig_logit_gap,
        integrated_gradients,
        mass_fraction_by_kind,
        random_attribution_enrichment,
        randomized_model_cam,
        signed_target,
    )
    from qrphish.mapping import provenance
    from qrphish.qrgen import encode

    model, ds, arr, meta = _load_trained(cfg, cid, stratum, seed)
    if not hasattr(model, "last_conv_block"):
        return {"seed": seed, "error": "Grad-CAM은 conv 모델(small_cnn)에만 적용한다"}
    target = model.last_conv_block

    ec = int(arr["ec"])
    y_all = np.asarray(arr["y"], dtype=np.int64)
    test_rows = np.nonzero(arr["split"].astype(int) == 2)[0]
    # 앞쪽 N개 고정 표본은 클래스 구성이 통제되지 않는다(review_01 핵심 문제 3).
    idx = balanced_sample(y_all, test_rows, per_class=per_class, seed=seed)

    mass: dict[str, list[float]] = {}
    enr: dict[str, list[float]] = {}
    enr_rand: dict[str, list[float]] = {}
    enr_ig: dict[str, list[float]] = {}
    legacy: dict[str, list[float]] = {}
    char_curves: dict[int, list[np.ndarray]] = {0: [], 1: []}
    sanity: list[float] = []
    ig_err: list[float] = []
    cam_sum = None
    n_by_class = {0: 0, 1: 0}

    for k, i in enumerate(idx):
        i = int(i)
        x, _ = ds[i]
        label = int(y_all[i])
        n_by_class[label] += 1
        sgn = signed_target(label)
        cam = gradcam_signed(model, x, target, sign=sgn)
        cam_sum = cam if cam_sum is None else cam_sum + cam
        url = str(meta["url"].iloc[i])
        art = encode(url, ec=ec, mask_pattern=int(meta["mask_used"].iloc[i]))
        prov = provenance(url, art)
        n_art = art.modules.shape[0]
        off = (ds.n - n_art) // 2
        cam_c = cam[off : off + n_art, off : off + n_art]

        for kk, v in attribute_by_kind(cam_c, prov).items():
            legacy.setdefault(kk, []).append(v)
        for kk, v in mass_fraction_by_kind(cam_c, prov).items():
            mass.setdefault(kk, []).append(v)
        for kk, v in enrichment_by_kind(cam_c, prov).items():
            enr.setdefault(kk, []).append(v)
        for kk, v in random_attribution_enrichment(cam_c.shape, prov, seed=seed * 97 + k).items():
            enr_rand.setdefault(kk, []).append(v)

        if use_ig:
            attr = integrated_gradients(model, x, baseline=0.0, steps=ig_steps, sign=sgn)
            gap = ig_logit_gap(model, x, baseline=0.0, sign=sgn)
            denom = max(abs(gap), 1e-9)
            ig_err.append(abs(float(attr.sum()) - gap) / denom)
            # 값 채널 기여도만 격자로 접는다(마스크 채널은 층 내 상수).
            ig_map = attr[0][off : off + n_art, off : off + n_art]
            for kk, v in enrichment_by_kind(ig_map, prov).items():
                enr_ig.setdefault(kk, []).append(v)

        # sanity check는 비싸므로 앞쪽 8개 표본에서만 한다.
        if k < 8:
            rnd_cam = randomized_model_cam(model, x, sign=sgn, seed=seed * 31 + k)
            sanity.append(cam_correlation(cam, rnd_cam))

        ch = attribute_to_chars(cam_c, prov, url)
        if ch.size:
            grid = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, ch.size), ch)
            char_curves[label].append(grid)

    def _mean(d: dict[str, list[float]]) -> dict[str, float]:
        return {k: float(np.nanmean(v)) for k, v in d.items() if v}

    out: dict[str, Any] = {
        "seed": seed,
        "n_samples": int(idx.size),
        "n_by_class": {str(k): int(v) for k, v in n_by_class.items()},
        "cam_mass_by_kind": _mean(legacy),
        "cam_mass_fraction": _mean(mass),
        "cam_enrichment": _mean(enr),
        "random_attribution_enrichment": _mean(enr_rand),
        "sanity_randomized_cam_corr": float(np.mean(sanity)) if sanity else float("nan"),
        "char_position_curve": {
            str(k): (np.mean(v, axis=0).tolist() if v else []) for k, v in char_curves.items()
        },
    }
    if use_ig:
        out["ig_enrichment"] = _mean(enr_ig)
        out["ig_completeness_rel_error"] = float(np.mean(ig_err)) if ig_err else float("nan")
    out["_mean_cam"] = (cam_sum / max(idx.size, 1)) if cam_sum is not None else None
    return out


def _avg_dicts(rows: list[dict], key: str) -> dict[str, float]:
    """``rows``의 ``row[key]`` dict들을 키별 NaN 무시 평균으로 합친다."""
    keys: set[str] = set()
    for d in rows:
        keys |= set(d.get(key, {}))
    return {
        k: float(np.nanmean([d[key][k] for d in rows if k in d.get(key, {})]))
        for k in sorted(keys)
    }


def run_explain(
    cfg: Any,
    condition_id_str: str = MAIN_CONDITION,
    stratum: str | None = None,
    max_samples: int = 128,
    per_class: int = 64,
    use_ig: bool = False,
    ig_steps: int = 32,
) -> dict:
    """P3 — Grad-CAM을 **보조 분석**으로 재구성한다 (review_01 핵심 문제 3·4).

    바뀐 점:
      - 5시드 전부, test에서 **클래스별 균형 무작위 표본**(기본 클래스당 64).
      - phishing은 ``+logit``, benign은 ``-logit``을 타깃으로 잡는다.
      - CAM 질량 분수를 영역 면적 분수로 나눈 **enrichment**를 보고한다.
        균일 난수 attribution 기준선(기대값 1)을 함께 낸다.
      - 모델 파라미터 무작위화 sanity check와 (옵션) signed Integrated Gradients.

    ``max_samples``는 하위 호환을 위해 남긴 인자로, ``per_class``가 주어지지 않은
    호출에서 클래스당 ``max_samples // 2``로 해석된다. 출력 스키마는 기존
    ``cam_mass_by_kind``/``char_position_curve``/``n_samples``를 그대로 유지한 채
    키를 추가하기만 한다.
    """
    import time

    if per_class is None:
        per_class = max(int(max_samples) // 2, 1)
    cond_dir = _out_root(cfg) / condition_id_str
    if not cond_dir.exists():
        raise FileNotFoundError(f"조건 디렉터리가 없다: {cond_dir}")
    strata = [stratum] if stratum else [p.name for p in sorted(cond_dir.iterdir()) if p.is_dir()]
    seeds_cfg = [int(s) for s in _get(cfg, "seed_list", [0])]

    out: dict[str, Any] = {}
    for st in strata:
        avail = [s for s in _available_seeds(cfg, condition_id_str, st) if s in set(seeds_cfg)]
        if not avail:
            continue
        # 층 단위 재개: explain.json이 이미 있으면 건너뛴다.
        done_path = cond_dir / st / "explain.json"
        done = _completed_result(done_path, f"explain {st}")
        if done is not None:
            print(f"[skip] explain {st} (explain.json 존재)", flush=True)
            out[st] = done
            continue
        t_st = time.time()
        print(f"[explain] {st} 시작 — seeds={avail}, per_class={per_class}", flush=True)
        per_seed = []
        for s in avail:
            t_seed = time.time()
            try:
                per_seed.append(
                    _explain_one_seed(cfg, condition_id_str, st, s, per_class, use_ig, ig_steps)
                )
                print(f"  [explain] {st} seed{s} 종료 ({time.time() - t_seed:.1f}s)", flush=True)
            except Exception as exc:
                print(f"[ERROR] explain {st} seed{s}: {exc!r}")
                traceback.print_exc()
                per_seed.append({"seed": s, "error": repr(exc)})
        ok = [d for d in per_seed if "error" not in d]
        if not ok:
            out[st] = {"error": per_seed[0].get("error", "no usable seed")}
            continue

        cams = [d.pop("_mean_cam") for d in per_seed if d.get("_mean_cam") is not None]

        res: dict[str, Any] = {
            # --- 기존 키(삭제 금지) ---
            "n_samples": int(sum(d["n_samples"] for d in ok)),
            "cam_mass_by_kind": _avg_dicts(ok, "cam_mass_by_kind"),
            "char_position_curve": {
                str(c): (
                    np.mean(
                        [d["char_position_curve"][str(c)] for d in ok
                         if d["char_position_curve"].get(str(c))],
                        axis=0,
                    ).tolist()
                    if any(d["char_position_curve"].get(str(c)) for d in ok)
                    else []
                )
                for c in (0, 1)
            },
            # --- 2차 실험에서 추가된 키 ---
            "schema_version": SCHEMA_VERSION,
            "seeds": [d["seed"] for d in ok],
            "per_class_sample": int(per_class),
            "signed_target": True,
            "cam_mass_fraction": _avg_dicts(ok, "cam_mass_fraction"),
            "cam_enrichment": _avg_dicts(ok, "cam_enrichment"),
            "random_attribution_enrichment": _avg_dicts(ok, "random_attribution_enrichment"),
            "sanity_randomized_cam_corr": float(
                np.nanmean([d["sanity_randomized_cam_corr"] for d in ok])
            ),
            "per_seed": [{k: v for k, v in d.items() if k != "char_position_curve"} for d in ok],
            "notes": [
                "phishing=+logit, benign=-logit(부호 있는 클래스별 근거).",
                "enrichment = CAM 질량 분수 / 영역 면적 분수. 1이면 면적만큼만 본 것.",
                "Grad-CAM은 보조 시각화다. 인과 주장은 occlusion 결과로만 한다.",
            ],
        }
        if use_ig:
            res["ig_enrichment"] = _avg_dicts(ok, "ig_enrichment")
            res["ig_completeness_rel_error"] = float(
                np.nanmean([d.get("ig_completeness_rel_error", np.nan) for d in ok])
            )

        (cond_dir / st / "explain.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        rp = cond_dir / st / "results.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            r["explain"] = {
                "cam_mass_by_kind": res["cam_mass_by_kind"],
                "cam_enrichment": res["cam_enrichment"],
                "sanity_randomized_cam_corr": res["sanity_randomized_cam_corr"],
            }
            rp.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if cams:
            np.save(cond_dir / st / "mean_cam.npy", np.mean(cams, axis=0))
        out[st] = res
        print(f"[explain] {st} 종료 ({time.time() - t_st:.1f}s)", flush=True)

    # scripts/make_results_tables.py가 읽는 요약본. 층별로 병합해 덮어쓴다.
    summary_path = _reports_dir(cfg) / "explain_summary.json"
    merged = {}
    if summary_path.exists():
        merged = json.loads(summary_path.read_text(encoding="utf-8"))
    merged.update(out)
    summary_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return out


# ------------------------------------------------------------------- comparisons
def _paired_dir(cfg: Any) -> Path:
    d = _reports_dir(cfg) / "paired"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _matrix_condition_ids(cfg: Any, matrix_path: Path | str) -> dict[str, str]:
    """matrix 항목 이름 → 조건 id. 가설 검정이 조건 id를 문자열로 박지 않게 한다."""
    matrix = load_matrix(matrix_path)
    out: dict[str, str] = {}
    for phase_entries in matrix.values():
        for entry in phase_entries or []:
            cfg_e = apply_overrides(cfg, entry.get("overrides"))
            out[str(entry.get("name"))] = condition_id(cfg_e, entry.get("extra"))
    return out


def compare_conditions(
    cfg: Any,
    cond_a: str,
    cond_b: str,
    stratum: str,
    n_boot: int | None = None,
    alpha: float = 0.05,
    alternative: str = "greater",
    save: bool = True,
) -> dict:
    """두 조건의 ΔAUROC(a − b)를 부트스트랩으로 비교한다.

    집계는 시드 층화다: 그룹을 리샘플하되 ΔAUROC는 시드별로 계산한 뒤 평균낸다
    (시드마다 모델이 달라 점수 척도가 다르므로 시드를 섞어 순위를 매기면 안 된다).

    두 조건이 **같은 split을 공유**하면(같은 url_mode·length_match·seed → 같은 분할과
    같은 test 행) 시드별 test 예측을 행 단위로 맞춰 **쌍체** 클러스터 부트스트랩을 쓴다.
    공유하지 않으면(예: L-none vs L-exact) 표본 집합 자체가 다르므로 쌍체가 성립하지
    않는다. 그 경우 ``paired=False``로 표시하고 비쌍체 차이만 보고한다.
    """
    seeds = list(_get(cfg, "seed_list", [0]))
    n_boot = int(n_boot or _get(_get(cfg, "eval"), "n_bootstrap", 2000))
    pa = _pool_preds(cfg, cond_a, stratum, seeds)
    pb = _pool_preds(cfg, cond_b, stratum, seeds)
    if pa is None or pb is None:
        missing = [c for c, d in ((cond_a, pa), (cond_b, pb)) if d is None]
        raise FileNotFoundError(
            f"preds_test.npz가 없다: {missing} (stratum={stratum}). 해당 조건을 먼저 실행할 것."
        )

    paired = (
        pa["url"].shape == pb["url"].shape
        and bool(np.array_equal(pa["url"], pb["url"]))
        and bool(np.array_equal(pa["seed"], pb["seed"]))
        and bool(np.array_equal(pa["y"], pb["y"]))
    )
    if paired:
        r = paired_cluster_bootstrap_by_seed(
            pa["y"], pa["p"], pb["p"], pa["group"], pa["seed"],
            n_boot=n_boot, seed=0, alpha=alpha,
        )
        caveat = "같은 split의 같은 test 행을 공유하므로 쌍체 비교가 성립한다."
        auroc_a, auroc_b = float(r["point_a"]), float(r["point_b"])
    else:
        r = unpaired_delta_bootstrap_by_seed(
            pa["y"], pa["p"], pa["group"], pa["seed"],
            pb["y"], pb["p"], pb["group"], pb["seed"],
            n_boot=n_boot, seed=0, alpha=alpha,
        )
        caveat = (
            "두 조건의 test 표본 집합이 다르다(표본 수·클래스 구성·URL 집합이 함께 변한다). "
            "따라서 이 차이는 단일 개입의 효과가 아니며 쌍체 비교도 불가능하다."
        )
        auroc_a, auroc_b = float(r["point_a"]), float(r["point_b"])

    out = {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "stratum": stratum,
        "paired": bool(paired),
        "n_pooled": int(pa["y"].size),
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "delta_auroc": float(r["point"]),
        "delta_ci": [float(r["lo"]), float(r["hi"])],
        "pseudo_p": bootstrap_p_value(r["samples"], alternative=alternative),
        "alternative": alternative,
        "n_boot": n_boot,
        "n_valid": int(r["n_valid"]),
        "pooling": POOLING_MODE,
        "caveat": caveat,
    }
    if save:
        path = _paired_dir(cfg) / f"{cond_a}__vs__{cond_b}__{stratum}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["path"] = str(path)
    return out


# 순열 검정이 만드는 null이 무엇인지 한 줄로 못 박는다. "엄밀한 순열 검정을
# 통과했다"가 아니라 "정의한 그룹 블록 null 기준을 넘었다"가 맞는 서술이다 —
# 반복 수를 늘려도 교환 가능성 가정 자체가 검증되지는 않는다(review_04).
PERM_NULL_NOTE = (
    "정의한 그룹 블록 null 기준. 같은 group(도메인 클러스터) 안의 행을 한 블록으로 묶어 "
    "두 조건 라벨을 교환해 만든 영가설 분포이며, 블록 단위 교환 가능성을 가정한다. "
    "반복 수를 늘리면 몬테카를로 오차만 줄고 그 가정이 검증되지는 않는다."
)


# H1~H4. matrix 항목 이름으로 조건을 지목한다(조건 id 문자열을 박지 않는다).
HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "id": "H1",
        "statement": "길이·버전을 통제한 뒤에도 CNN AUROC가 라벨 셔플 바닥보다 높다.",
        "kind": "compare",
        "a": "main_cnn",
        "b": "label_shuffle",
    },
    {
        "id": "H2",
        "statement": "디코딩 텍스트 참조 기준이 CNN보다 높다(갭이 남는다).",
        "kind": "reference_gap",
        "a": "main_cnn",
    },
    {
        "id": "H3",
        "statement": "길이 매칭을 풀면(L-none) CNN AUROC가 L-exact보다 높다.",
        "kind": "compare",
        "a": "length_none",
        "b": "main_cnn",
    },
    {
        "id": "H4",
        "statement": "원래 모듈 배치가 위치 셔플보다 CNN에게 유리하다.",
        "kind": "compare",
        "a": "main_cnn",
        "b": "shuffle_pos",
    },
)


def run_hypothesis_tests(
    cfg: Any,
    strata: list[str] | None = None,
    matrix_path: Path | str = "configs/matrix.yaml",
    n_boot: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """H1~H4를 층별로 검정하고 Holm 보정을 적용해 ``reports/hypotheses.json``에 쓴다."""
    ids = _matrix_condition_ids(cfg, matrix_path)
    strata = strata or _surviving_strata(cfg)
    n_boot = int(n_boot or _get(_get(cfg, "eval"), "n_bootstrap", 2000))
    seeds = list(_get(cfg, "seed_list", [0]))

    tests: list[dict[str, Any]] = []
    for st in strata:
        for h in HYPOTHESES:
            rec: dict[str, Any] = {
                "hypothesis": h["id"],
                "stratum": st,
                "statement": h["statement"],
            }
            try:
                if h["kind"] == "compare":
                    cmp = compare_conditions(
                        cfg, ids[h["a"]], ids[h["b"]], st, n_boot=n_boot, alpha=alpha
                    )
                    rec.update(
                        {
                            "cond_a": cmp["cond_a"],
                            "cond_b": cmp["cond_b"],
                            "estimate": cmp["delta_auroc"],
                            "ci": cmp["delta_ci"],
                            "pseudo_p": cmp["pseudo_p"],
                            "paired": cmp["paired"],
                            "pooling": cmp["pooling"],
                            "caveat": cmp["caveat"],
                        }
                    )
                    if cmp["paired"]:
                        # 쌍체 예측이 있으면 정식 영가설 분포를 만드는 클러스터 인식
                        # 순열 검정도 함께 낸다(review_02 6절). pseudo-p와 달리 그룹
                        # 단위로 a/b 라벨을 교환해 영가설 분포를 직접 생성한다.
                        rec["perm_p"] = _paired_permutation_p(
                            cfg, ids[h["a"]], ids[h["b"]], st, seeds
                        )
                else:
                    gap = _reference_gap_test(cfg, ids[h["a"]], st, n_boot, alpha, seeds)
                    rec.update(gap)
            except Exception as exc:
                # 어떤 조건의 preds가 없거나(dropped 층·error 조건) 데이터가 모자라도
                # 나머지 검정은 계속 돌려야 한다.
                print(f"[ERROR] hypothesis {h['id']} {st}: {exc!r}")
                rec["error"] = repr(exc)
            tests.append(rec)

    # 판정은 "95% 양측 백분위 CI가 0을 배제하는가"로 하고, Holm은 그와 동치인
    # CI 역전 p값에 건다. descriptive_only(H2 폴백)는 family에서 뺀다.
    for t in tests:
        ci = t.get("ci")
        t["ci_excludes_null"] = bool(
            ci is not None
            and np.isfinite(ci[0])
            and np.isfinite(ci[1])
            and (ci[0] > 0 or ci[1] < 0)
        )

    # 정식 검정과 기술 추정을 필드로 갈라 둔다. 정식 = 영가설 분포를 실제로 만든
    # 클러스터 순열 검정(perm_p)뿐이고, pseudo_p는 부트스트랩 CI를 역전시킨 기술
    # 통계다(review_04 "통계 표 정리"). 표·본문이 둘을 섞어 쓰지 못하게 한다.
    for t in tests:
        if "error" in t:
            t["formal_test"] = None
            t["descriptive"] = None
        elif t.get("perm_p") is not None:
            t["formal_test"] = {
                "method": "paired_cluster_permutation",
                "p": float(t["perm_p"]),
                "null": PERM_NULL_NOTE,
                "alternative": "greater",
            }
            t["descriptive"] = None
        else:
            t["formal_test"] = None
            t["descriptive"] = {
                "method": (
                    "paired_cluster_bootstrap_by_seed"
                    if t.get("paired")
                    else "unpaired_cluster_bootstrap_by_seed"
                ),
                "estimate": t.get("estimate"),
                "ci": t.get("ci"),
                "pseudo_p": t.get("pseudo_p"),
                "note": (
                    "정식 검정 없음 — 효과 추정치와 95% CI만 보고한다. pseudo_p는 CI를 "
                    "역전시켜 정의한 값이지 영가설 분포에서 나온 p값이 아니다."
                ),
            }

    runnable = [
        t
        for t in tests
        if "pseudo_p" in t
        and not t.get("descriptive_only")
        and not np.isnan(t.get("pseudo_p", float("nan")))
    ]
    adj = holm([t["pseudo_p"] for t in runnable])
    for t, a in zip(runnable, adj, strict=True):
        t["pseudo_p_holm"] = float(a)
        t["reject"] = bool(a < alpha)
    for t in tests:
        t.setdefault("pseudo_p_holm", None)
        t.setdefault("perm_p", None)
        t.setdefault("reject", None)

    out = {
        "alpha": alpha,
        "n_boot": n_boot,
        "n_tests_in_family": len(runnable),
        "correction": "holm",
        "pooling": POOLING_MODE,
        "p_definition": (
            "bootstrap-tail pseudo-p (CI inversion); not a formal null-distribution p-value"
        ),
        "decision_rule": (
            "정식 가설 검정은 쌍체 예측이 있는 비교(H1·H4)의 클러스터 순열 검정 하나뿐이며 "
            "formal_test 필드에 담는다 — 귀무가설은 'ΔAUROC ≤ 0'이고, 그룹 단위로 두 조건 "
            "라벨을 교환해 영가설 분포를 직접 만든다. 나머지(H2·H3)는 descriptive 필드에 "
            "효과 추정치와 95% 클러스터 부트스트랩 CI만 싣는 기술 추정이다. pseudo_p와 "
            "Holm 보정값(pseudo_p_holm)은 CI를 역전시켜 정의한 참고 수치이므로 정식 표에 "
            "싣지 않고 부록으로만 보고한다. H3는 L-none과 L-exact의 평가 표본이 달라 쌍체 "
            "해석이 성립하지 않는다."
        ),
        "permutation_null": PERM_NULL_NOTE,
        "generated_at": datetime.now(UTC).isoformat(),
        "tests": tests,
    }
    path = _reports_dir(cfg) / "hypotheses.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out["path"] = str(path)
    return out


def _paired_permutation_p(cfg, cond_a: str, cond_b: str, stratum: str, seeds) -> float | None:
    """쌍체 두 조건의 클러스터 인식 순열 p값. 예측을 못 맞추면 ``None``."""
    pa = _pool_preds(cfg, cond_a, stratum, list(seeds))
    pb = _pool_preds(cfg, cond_b, stratum, list(seeds))
    if pa is None or pb is None:
        return None
    if not (
        pa["url"].shape == pb["url"].shape
        and bool(np.array_equal(pa["url"], pb["url"]))
        and bool(np.array_equal(pa["seed"], pb["seed"]))
        and bool(np.array_equal(pa["y"], pb["y"]))
    ):
        return None
    r = paired_cluster_permutation_test(
        pa["y"], pa["p"], pb["p"], pa["group"],
        n_perm=2000, seeds=pa["seed"], alternative="greater", seed=0,
    )
    return float(r["perm_p"])


def _reference_gap_test(cfg, cid, stratum, n_boot, alpha, seeds) -> dict:
    """H2 — 디코딩 텍스트 참조 기준 − CNN > 0.

    참조 기준의 test 예측(``preds_test_charngram.npz``)이 있으면 CNN과 **같은 split의 같은
    행**을 공유하므로 쌍체 클러스터 부트스트랩으로 검정한다. 옛 산출물처럼 예측이 저장되지
    않았으면 참조 기준 AUROC를 상수로 두고 CNN 쪽만 흔드는 옛 방식으로 되돌아가되,
    ``descriptive_only=True``로 표시해 **Holm family에서 제외**한다(참조 기준 자체의 표본
    변동이 빠져 CI가 좁게 나오므로 검정으로 쓸 수 없다).
    """
    pooled = _pool_preds(cfg, cid, stratum, seeds)
    if pooled is None:
        raise FileNotFoundError(f"preds_test.npz가 없다: {cid}/{stratum}")
    ref_pred = _pool_preds(cfg, cid, stratum, seeds, CHARNGRAM_PREDS)
    caveat_base = (
        "참조 기준은 강한 텍스트 베이스라인이지 Bayes 최적 분류기가 아니다. "
        "CNN이 이를 넘더라도 누출의 증거가 아니다."
    )
    if (
        ref_pred is not None
        and ref_pred["y"].shape == pooled["y"].shape
        and bool(np.array_equal(ref_pred["url"], pooled["url"]))
        and bool(np.array_equal(ref_pred["y"], pooled["y"]))
    ):
        r = paired_cluster_bootstrap_by_seed(
            pooled["y"], ref_pred["p"], pooled["p"], pooled["group"], pooled["seed"],
            n_boot=n_boot, seed=0, alpha=alpha,
        )
        return {
            "cond_a": "charngram_lr (decoded-text reference)",
            "cond_b": cid,
            "estimate": float(r["point"]),
            "ci": [float(r["lo"]), float(r["hi"])],
            "pseudo_p": bootstrap_p_value(r["samples"], alternative="greater"),
            "paired": True,
            "descriptive_only": False,
            "pooling": POOLING_MODE,
            "caveat": caveat_base + " 참조 기준의 test 예측을 함께 흔든 쌍체 비교다.",
        }

    rp = _out_root(cfg) / cid / stratum / "results.json"
    if not rp.exists():
        raise FileNotFoundError(f"results.json이 없다: {rp}")
    ref = json.loads(rp.read_text(encoding="utf-8")).get("aggregate", {}).get(
        "decoded_text_reference_auroc"
    )
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        raise KeyError(f"{cid}/{stratum}에 decoded_text_reference_auroc이 없다")
    cb = cluster_bootstrap_by_seed(
        pooled["y"], pooled["p"], pooled["group"], pooled["seed"],
        auroc_fn, n_boot=n_boot, seed=0, alpha=alpha,
    )
    samples = float(ref) - cb["samples"]
    lo, hi, _ = percentile_ci(samples, alpha)
    return {
        "cond_a": "charngram_lr (decoded-text reference)",
        "cond_b": cid,
        "estimate": float(ref) - cb["point"],
        "ci": [lo, hi],
        "pseudo_p": float("nan"),
        "paired": False,
        "descriptive_only": True,
        "pooling": POOLING_MODE,
        "caveat": (
            caveat_base
            + " 참조 기준의 test 예측이 저장되지 않아 AUROC를 상수로 두었다. "
            "참조 기준 자체의 표본 변동이 빠져 CI가 좁으므로 기술 통계(descriptive gap)로만 "
            "읽고 Holm family에서 제외했다. 재실행하면 쌍체 검정으로 승격된다."
        ),
    }


MOTIF_PREDS_TMPL = "seed{seed}_preds.npz"


def _save_motif_preds(sdir: Path, seed: int, preds: dict) -> Path:
    """motif 표현별 test 예측(y/p/group)을 한 파일에 담는다."""
    payload: dict[str, np.ndarray] = {}
    for key, d in preds.items():
        payload[f"{key}__y"] = np.asarray(d["y"], dtype=np.int64)
        payload[f"{key}__p"] = np.asarray(d["p"], dtype=np.float64)
        payload[f"{key}__group"] = np.asarray(d["group"], dtype=object).astype(str)
    path = sdir / MOTIF_PREDS_TMPL.format(seed=seed)
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    return path


def _load_motif_preds(sdir: Path, seed: int) -> dict[str, dict[str, np.ndarray]] | None:
    path = sdir / MOTIF_PREDS_TMPL.format(seed=seed)
    if not path.exists():
        return None
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            key, _, field = name.rpartition("__")
            out.setdefault(key, {})[field] = z[name]
    return out


def reaggregate(cfg: Any, strata: list[str] | None = None) -> dict[str, Any]:
    """저장된 test 예측에서 ``results.json``의 시드 통합 CI만 다시 계산해 덮어쓴다.

    ``run_matrix``/``run_motifs``는 ``results.json``이 있으면 그 조합을 건너뛰므로, 집계
    규약이 바뀌어도 재실행만으로는 새 CI가 나오지 않는다. 이 함수는 **학습을 다시 하지
    않고** ``preds_test.npz``(CNN)와 ``seed{k}_preds.npz``(motif)만 읽어 ``auroc_pooled``·
    ``auroc_pooled_ci``·``pooling``을 시드 층화 방식으로 다시 쓴다.

    ``aggregate_reports``는 ``results.json``의 aggregate 블록을 읽기만 하므로 반드시
    **이 함수를 먼저** 부른 뒤에 호출해야 새 CI가 표에 반영된다.
    """
    n_boot = int(_get(_get(cfg, "eval"), "n_bootstrap", 2000))
    want = set(strata) if strata else None
    updated: list[str] = []
    skipped: list[dict[str, str]] = []

    for rp in sorted(_out_root(cfg).glob("*/*/results.json")):
        cid, stratum = rp.parent.parent.name, rp.parent.name
        if want and stratum not in want:
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        seeds = [
            int(ps["seed"]) for ps in r.get("per_seed", []) if "error" not in ps and "seed" in ps
        ]
        if not seeds:
            skipped.append({"path": str(rp), "reason": "쓸 수 있는 시드가 없다"})
            continue
        point, ci = _pooled_auroc(cfg, cid, stratum, seeds, n_boot)
        if not np.isfinite(point):
            skipped.append({"path": str(rp), "reason": "preds_test.npz가 없다"})
            continue
        agg = r.setdefault("aggregate", {})
        agg["auroc_pooled"] = point
        agg["auroc_pooled_ci"] = ci
        agg["pooling"] = POOLING_MODE
        rp.write_text(
            json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        updated.append(str(rp))
        print(f"[reagg] {cid}/{stratum} auroc_pooled={point:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    motif_updated: list[str] = []
    for rp in sorted((_reports_dir(cfg) / "motifs").glob("*/*/results.json")):
        stratum = rp.parent.name
        if want and stratum not in want:
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        ok = []
        for ps in r.get("per_seed", []):
            if "error" in ps or "representations" not in ps:
                continue
            preds = _load_motif_preds(rp.parent, int(ps["seed"]))
            if preds is None:
                continue
            ok.append({**ps, "preds": preds})
        if not ok:
            skipped.append({"path": str(rp), "reason": "seed*_preds.npz가 없다"})
            continue
        r["pooling"] = POOLING_MODE
        r["representations"] = {
            key: _motif_aggregate(ok, key, n_boot=n_boot) for key in sorted(ok[0]["representations"])
        }
        rp.write_text(
            json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        motif_updated.append(str(rp))
        print(f"[reagg] motifs {stratum} 재집계 완료")

    return {
        "pooling": POOLING_MODE,
        "updated": updated,
        "motifs_updated": motif_updated,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------- aggregate
def aggregate_reports(cfg: Any) -> dict[str, Path]:
    """모든 results.json → ``reports/table_*.csv`` (스펙 8절 마지막 문단)."""
    root = _out_root(cfg)
    reports = _reports_dir(cfg)
    rows, brows = [], []
    for rp in sorted(root.glob("*/*/results.json")):
        r = json.loads(rp.read_text(encoding="utf-8"))
        agg = r.get("aggregate", {})
        rows.append(
            {
                "condition_id": r.get("condition_id"),
                "phase": r.get("phase"),
                "stratum": r.get("stratum"),
                "tier": r.get("tier"),
                "n_total": r.get("data", {}).get("n_total"),
                "auroc_mean": agg.get("auroc_mean"),
                "auroc_sd": agg.get("auroc_sd_across_seeds"),
                "auroc_pooled": agg.get("auroc_pooled"),
                "auroc_ci_lo": (agg.get("auroc_pooled_ci") or [None, None])[0],
                "auroc_ci_hi": (agg.get("auroc_pooled_ci") or [None, None])[1],
                "pooling": agg.get("pooling"),
                "f1_mean": agg.get("f1_mean"),
                "acc_mean": agg.get("acc_mean"),
                "decoded_text_reference_auroc": agg.get("decoded_text_reference_auroc"),
                "gap_to_decoded_text": agg.get("gap_to_decoded_text"),
                "label_shuffle_auroc": r.get("sanity", {}).get("label_shuffle_auroc"),
            }
        )
        for ps in r.get("per_seed", []):
            for bname, b in (ps.get("baselines") or {}).items():
                if "auroc" not in b:
                    continue
                brows.append(
                    {
                        "condition_id": r.get("condition_id"),
                        "stratum": r.get("stratum"),
                        "seed": ps.get("seed"),
                        "baseline": bname,
                        "auroc": b["auroc"],
                        "f1": b["f1"],
                        "acc": b["acc"],
                    }
                )

    out: dict[str, Path] = {}
    main = pd.DataFrame(rows)
    p = reports / "table_main.csv"
    main.to_csv(p, index=False)
    out["main"] = p
    if brows:
        bdf = pd.DataFrame(brows).groupby(
            ["condition_id", "stratum", "baseline"], as_index=False
        )[["auroc", "f1", "acc"]].mean()
        p = reports / "table_baselines.csv"
        bdf.to_csv(p, index=False)
        out["baselines"] = p
    if not main.empty:
        for phase, sub in main.groupby("phase", dropna=False):
            p = reports / f"table_{str(phase).lower()}.csv"
            sub.to_csv(p, index=False)
            out[str(phase)] = p
    return out


# --------------------------------------------------------------------------- motifs
# 리뷰 review_01 "B. Visual motif 실험" — Q1(공통 국소 시각 motif가 있는가)을 직접 측정한다.
# 주 조건(norm-exact-data_only-fixed)의 split을 그대로 재현하되 CNN은 학습하지 않는다.
MOTIF_STRATA = ("v3", "v2", "v4")
# (키, 창 크기, 피라미드 여부)
MOTIF_REPRESENTATIONS = (
    ("patch2", 2, False),
    ("patch3", 3, False),
    ("pyramid3", 3, True),
    # within-QR null (review_02 2절). 3x3 히스토그램을 그대로 쓰되 입력 격자를 QR 안에서만
    # 섞는다. patch3 대비 얼마나 떨어지는지가 "국소 공간 의존성"의 크기다.
    #   patch3_shuffle_module    데이터 모듈 값 완전 순열 — 검은 모듈 비율만 보존.
    #   patch3_shuffle_codeword  8비트 코드워드 순서만 순열 — 바이트 조성까지 보존.
    ("patch3_shuffle_module", 3, False),
    ("patch3_shuffle_codeword", 3, False),
)
# 위 두 표현은 시드마다 셔플이 달라지므로 URL 캐시에 넣지 않고 매 시드 다시 만든다.
MOTIF_SHUFFLE_KEYS = ("patch3_shuffle_module", "patch3_shuffle_codeword")
# enrichment CI 부트스트랩 횟수. motif 축(3x3 = 512종) 전체를 흔들어야 해서 지표 CI(2000)
# 보다 낮춰 잡았다. enrichment.json의 ci_n_boot에 그대로 기록된다.
ENRICHMENT_CI_BOOT = 500


def _top_list_overlap(per_seed_tops: dict[int, list[int]]) -> dict:
    """시드별 top motif 목록 사이의 Jaccard 겹침 요약.

    선정을 시드별 train으로 분리하면 목록이 시드마다 달라진다. 겹침이 낮다면 "안정
    motif"라는 표현 자체가 약해지므로, 절제 결과를 읽을 때 함께 봐야 하는 진단값이다.
    """
    seeds = sorted(per_seed_tops)
    sets = {k: set(int(v) for v in per_seed_tops[k]) for k in seeds}
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            u = sets[a] | sets[b]
            pairs.append(
                {
                    "seeds": [a, b],
                    "jaccard": (len(sets[a] & sets[b]) / len(u)) if u else float("nan"),
                }
            )
    js = [d["jaccard"] for d in pairs if not np.isnan(d["jaccard"])]
    inter = set.intersection(*sets.values()) if sets else set()
    union = set.union(*sets.values()) if sets else set()
    return {
        "n_seeds": len(seeds),
        "sizes": {str(k): len(sets[k]) for k in seeds},
        "pairwise_jaccard": pairs,
        "jaccard_mean": float(np.mean(js)) if js else float("nan"),
        "jaccard_min": float(np.min(js)) if js else float("nan"),
        "n_in_all_seeds": len(inter),
        "n_in_any_seed": len(union),
    }



def _phase_dir(cfg: Any, phase: str, cid: str, stratum: str) -> Path:
    """``reports/{phase}/{condition_id}/{stratum}/``.

    조건 id를 경로에 넣지 않으면 서로 다른 조건(예: raw vs norm, mask-auto)의 산출물이
    같은 층 폴더에서 덮어써진다. 옛 ``reports/{phase}/{stratum}/`` 경로는 쓰지 않는다.
    """
    d = _reports_dir(cfg) / phase / cid / stratum
    d.mkdir(parents=True, exist_ok=True)
    return d


def _motif_condition_id(cfg: Any) -> str:
    """motif 실험이 재현하는 조건 이름. arch는 붙지 않는다(모델을 학습하지 않는다)."""
    cond = _get(cfg, "condition")
    qr = _get(cfg, "qr")
    return "-".join(
        [
            str(_get(cond, "url_mode", "norm")),
            str(_get(cond, "length_match", "exact")),
            str(_get(cond, "features", "data_only")),
            str(_get(qr, "mask_mode", "fixed")),
        ]
    )


def _motif_grid(url: str, cfg: Any, n_max: int) -> tuple[np.ndarray, int, int]:
    """URL -> (중앙 정렬된 (n_max,n_max) bool 격자, version, mask_used)."""
    from qrphish.dataset import _center_pad
    from qrphish.qrgen import encode

    qr = _get(cfg, "qr")
    ec = ec_const(_get(qr, "ec", "L"))
    mask_mode = str(_get(qr, "mask_mode", "fixed"))
    mp = None if mask_mode == "auto" else int(_get(qr, "mask_pattern", 0) or 0)
    art = encode(url, ec=ec, mask_pattern=mp, version=None)
    mods = np.asarray(art.modules, dtype=bool)
    return _center_pad(mods, n_max), int(art.version), int(art.mask_pattern)


def _motif_null_geometry(version: int, mask_used: int, n_max: int, ec: int) -> dict:
    """within-QR null에 필요한 (버전, 마스크) 고정 기하 정보.

    ``order``는 데이터 비트 배치 좌표를 **패딩된 좌표계**로 옮긴 것이고, ``mask_flip``은
    그 마스크 패턴이 뒤집는 모듈 위치다(기능 패턴 제외). 둘 다 URL과 무관하므로 캐시한다.
    """
    from qrphish.dataset import _center_pad
    from qrphish.mapping import bit_placement_order
    from qrphish.mapping import unmask as _unmask

    n = 4 * int(version) + 17
    off = (int(n_max) - n) // 2
    order = np.asarray(bit_placement_order(int(version), ec), dtype=np.int64) + off
    # zeros를 unmask하면 XOR 뒤집기 위치 자체가 나온다(기능 패턴은 건드리지 않는다).
    flip = np.asarray(_unmask(np.zeros((n, n), dtype=bool), int(mask_used)), dtype=bool)
    return {"order": order, "mask_flip": _center_pad(flip, int(n_max))}


def _motif_features(
    cfg: Any, df: pd.DataFrame, cache: dict, n_max: int, seed: int = 0
) -> tuple[dict[str, np.ndarray], float]:
    """df의 URL들에 대해 표현별 히스토그램 행렬과 평균 창 개수를 만든다.

    ``cache``는 URL -> 표현별 벡터 dict. 시드마다 split만 달라지고 격자는 같으므로
    5시드 전체에서 URL당 한 번만 인코딩·히스토그램 계산을 한다. 다만
    ``MOTIF_SHUFFLE_KEYS``의 within-QR null 표현은 시드마다 셔플이 달라야 하므로
    캐시된 격자에서 매 시드 다시 만든다. 셔플 난수는 ``(seed, URL)``로만 정해지므로
    행 순서가 바뀌어도 같은 값이 나온다(결정적).
    """
    from zlib import crc32

    from qrphish.dataset import data_mask_for
    from qrphish.motifs import (
        extract_patch_ids,
        patch_histogram,
        shuffle_within_qr,
        shuffle_within_qr_bytes,
        spatial_pyramid_histogram,
    )

    ec = ec_const(_get(_get(cfg, "qr"), "ec", "L"))
    dm_cache: dict[int, np.ndarray] = {}
    geo_cache: dict[tuple[int, int], dict] = {}
    rows: dict[str, list[np.ndarray]] = {k: [] for k, _, _ in MOTIF_REPRESENTATIONS}
    n_windows: list[int] = []
    for url in df["url"].tolist():
        got = cache.get(url)
        if got is None:
            grid, version, mask_used = _motif_grid(url, cfg, n_max)
            dm = dm_cache.get(version)
            if dm is None:
                dm = data_mask_for(version, ec, n_max)
                dm_cache[version] = dm
            got = {
                "patch2": patch_histogram(grid, dm, 2, mask_pattern=mask_used),
                "patch3": patch_histogram(grid, dm, 3, mask_pattern=mask_used),
                "pyramid3": spatial_pyramid_histogram(grid, dm, 3, mask_pattern=mask_used),
                "n_windows3": np.int32(extract_patch_ids(grid, dm, 3).size),
                # null 표현을 시드마다 다시 만들기 위한 원본. bool 격자를 비트로 눌러 담는다.
                "grid_packed": np.packbits(np.asarray(grid, dtype=bool)),
                "version": int(version),
                "mask_used": int(mask_used),
            }
            cache[url] = got
        version = int(got["version"])
        dm = dm_cache.get(version)
        if dm is None:
            dm = data_mask_for(version, ec, n_max)
            dm_cache[version] = dm
        grid = (
            np.unpackbits(got["grid_packed"], count=n_max * n_max)
            .reshape(n_max, n_max)
            .astype(bool)
        )
        gkey = (version, int(got["mask_used"]))
        geo = geo_cache.get(gkey)
        if geo is None:
            geo = _motif_null_geometry(version, int(got["mask_used"]), n_max, ec)
            geo_cache[gkey] = geo
        base = int(crc32(url.encode("utf-8")))
        g_mod = shuffle_within_qr(grid, dm, seed=(int(seed), base, 1))
        g_cw = shuffle_within_qr_bytes(
            grid, geo["order"], seed=(int(seed), base, 2), mask_flip=geo["mask_flip"]
        )
        shuffled = {
            "patch3_shuffle_module": patch_histogram(g_mod, dm, 3),
            "patch3_shuffle_codeword": patch_histogram(g_cw, dm, 3),
        }
        for k, _, _ in MOTIF_REPRESENTATIONS:
            rows[k].append(shuffled[k] if k in shuffled else got[k])
        n_windows.append(int(got["n_windows3"]))
    X = {k: np.stack(v).astype(np.float32) for k, v in rows.items()}
    return X, (float(np.mean(n_windows)) if n_windows else 0.0)


def _motif_one_seed(cfg: Any, stratum: str, seed: int, cache: dict) -> dict:
    """한 (층, 시드): split 재현 -> 표현별 LR -> test 지표 + train/test enrichment."""
    from qrphish.dataset import SPLIT_CODE
    from qrphish.motifs import bag_of_patches_lr, motif_enrichment

    df, diag = _prepare_frame(cfg, stratum, seed)
    if len(df) == 0:
        return {"seed": seed, "error": "empty stratum after matching"}
    if "version" not in df.columns:
        raise ValueError("_prepare_frame이 version 컬럼을 주지 않았다")
    n_max = int(4 * int(df["version"].max()) + 17)

    X, n_windows_mean = _motif_features(cfg, df, cache, n_max, seed)
    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["group"].astype(str).to_numpy()
    split = np.asarray(
        [SPLIT_CODE[s] if isinstance(s, str) else int(s) for s in df["split"].tolist()],
        dtype=np.int64,
    )
    tr, va, te = split == 0, split == 1, split == 2

    # 라벨 셔플 대조: test는 원 라벨로 두고 train/val 라벨만 섞는다. 러너의 기존 규약
    # (`label_shuffle_rows=(split != 2)`)과 같다 — val까지 섞어야 C 선택으로 신호가
    # 새어 들어오지 않는다.
    rng = np.random.default_rng(1000 + seed)
    y_shuf = y.copy()
    nz = np.flatnonzero(~te)
    y_shuf[nz] = y_shuf[nz][rng.permutation(nz.size)]

    n_boot = min(int(_get(_get(cfg, "eval"), "n_bootstrap", 2000)), 1000)
    out: dict[str, Any] = {
        "seed": seed,
        "n_total": int(len(df)),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "n_windows_mean": n_windows_mean,
        "n_after_match": diag.get("n_after_match"),
        "representations": {},
        "preds": {},
        "enrichment_inputs": {},
    }
    for key, _size, _pyr in MOTIF_REPRESENTATIONS:
        Xk = X[key]
        for suffix, labels in (("", y), ("_labelshuffle", y_shuf)):
            fit = bag_of_patches_lr(
                Xk[tr], labels[tr], Xk[te], seed, X_hist_val=Xk[va], y_val=labels[va]
            )
            thr = pick_threshold(labels[va], fit["p_val"])
            met = metrics_from_probs(
                y[te], fit["p_test"], groups[te], n_boot=n_boot, seed=seed, threshold=thr
            )
            out["representations"][key + suffix] = {
                "auroc": met["auroc"],
                "auprc": met["auprc"],
                "f1": met["f1"],
                "acc": met["acc"],
                "auroc_ci": met["auroc_ci"],
                "C": fit["C"],
                "dim": int(Xk.shape[1]),
            }
            # 시드 통합 CI는 5시드 test 예측을 풀링해서 만든다(CNN 쪽 auroc_pooled와 같은 방식).
            # 라벨 셔플 행도 예측을 남겨 같은 절차로 바닥선 CI를 낸다.
            out["preds"][key + suffix] = {
                "y": y[te],
                "p": np.asarray(fit["p_test"], dtype=np.float64),
                "group": groups[te],
            }

    out["enrichment"] = {}
    for size, key in ((3, "patch3"), (2, "patch2")):
        tr_enr = motif_enrichment(
            X[key][tr], y[tr], groups[tr], size=size, n_boot=n_boot, seed=seed
        )
        te_enr = motif_enrichment(X[key][te], y[te], groups[te], size=size, n_boot=0, seed=seed)
        out["enrichment"][size] = {"train": tr_enr, "test": te_enr["log_odds"]}
        # 시드 통합 enrichment CI용 train 원자료. run_motifs가 (seed, row)로 이어 붙인다.
        out["enrichment_inputs"][size] = {
            "H": X[key][tr],
            "y": y[tr],
            "groups": groups[tr],
        }
    return out


def _motif_aggregate(per_seed: list[dict], key: str, n_boot: int = 1000) -> dict:
    """시드 평균 + **시드 층화** 그룹 클러스터 부트스트랩 CI.

    ``_assemble_results``의 CNN 쪽 ``auroc_pooled``/``auroc_pooled_ci``와 같은 절차다.
    리샘플 단위는 시드와 무관한 eTLD+1 그룹이고, AUROC는 시드별로 계산한 뒤 평균낸다
    (시드마다 LR 모델이 달라 점수 척도가 다르므로 시드를 섞어 순위를 매기면 안 된다).
    시드별 CI 경계를 평균하던 옛 ``auroc_ci_pooled``는 정식 CI가 아니라 폐기했다.
    """
    ok = [s for s in per_seed if "error" not in s]
    a = np.array([s["representations"][key]["auroc"] for s in ok], dtype=float)
    parts = [(s, s["preds"][key]) for s in ok if key in s.get("preds", {})]
    if parts:
        cb = cluster_bootstrap_by_seed(
            np.concatenate([d["y"] for _, d in parts]),
            np.concatenate([d["p"] for _, d in parts]),
            np.concatenate([d["group"] for _, d in parts]),
            np.concatenate(
                [np.full(np.asarray(d["y"]).size, int(m["seed"]), dtype=np.int64)
                 for m, d in parts]
            ),
            auroc_fn,
            n_boot=n_boot,
            seed=0,
        )
        auroc_pooled, pooled_ci = cb["point"], [cb["lo"], cb["hi"]]
        n_pooled = int(sum(np.asarray(d["y"]).size for _, d in parts))
    else:
        auroc_pooled, pooled_ci, n_pooled = float("nan"), [float("nan"), float("nan")], 0
    return {
        "auroc_mean": float(np.nanmean(a)) if a.size else float("nan"),
        "auroc_sd_across_seeds": float(np.nanstd(a, ddof=1)) if a.size > 1 else 0.0,
        "auroc_pooled": auroc_pooled,
        "auroc_pooled_ci": pooled_ci,
        "pooling": POOLING_MODE,
        "n_pooled": n_pooled,
        "f1_mean": float(np.nanmean([s["representations"][key]["f1"] for s in ok])) if ok else float("nan"),
        "acc_mean": float(np.nanmean([s["representations"][key]["acc"] for s in ok])) if ok else float("nan"),
        "dim": int(ok[0]["representations"][key]["dim"]) if ok else 0,
    }


def run_motifs(cfg: Any, strata: list[str] | None = None) -> dict:
    """Bag-of-QR-patches 실험 (리뷰 review_01 B). 모델 학습 없이 CPU에서 돈다.

    층별로 2x2 / 3x3 / 3x3 spatial pyramid 세 표현 x LR을 5시드로 돌려
    ``reports/motifs/{stratum}/results.json``에, motif 오즈비를
    ``reports/motifs/{stratum}/enrichment.json``에 저장한다. split은 주 조건
    ``norm-exact-data_only-fixed``의 :func:`_prepare_frame`을 같은 시드로 다시 불러
    재현하므로 CNN 실험과 정확히 같은 train/val/test 분할을 쓴다.
    """
    import time

    from qrphish.motifs import combine_seed_enrichments, motif_enrichment

    seeds = list(_get(cfg, "seed_list", [0]))
    n_boot_pool = int(_get(_get(cfg, "eval"), "n_bootstrap", 2000))
    cid = _motif_condition_id(cfg)
    todo = list(strata) if strata else [s for s in MOTIF_STRATA if s in set(_get(cfg, "strata", MOTIF_STRATA))]
    out: dict[str, Any] = {}

    for stratum in todo:
        done_path = _phase_dir(cfg, "motifs", cid, stratum) / "results.json"
        done = _completed_result(done_path, f"motifs {stratum}")
        if done is not None:
            print(f"[skip] motifs {stratum} (results.json 존재)", flush=True)
            out[stratum] = done
            continue
        print(f"[motifs] {stratum} 시작 — seeds={seeds}", flush=True)
        t0 = time.time()
        cache: dict[str, Any] = {}
        per_seed = []
        for s in seeds:
            try:
                t_seed = time.time()
                per_seed.append(_motif_one_seed(cfg, stratum, s, cache))
                print(f"  [motifs] {stratum} seed{s} 종료 ({time.time() - t_seed:.1f}s)",
                      flush=True)
            except Exception as exc:
                print(f"[ERROR] motifs {stratum} seed{s}: {exc!r}")
                traceback.print_exc()
                per_seed.append({"seed": s, "error": repr(exc)})
        ok = [s for s in per_seed if "error" not in s]
        if not ok:
            out[stratum] = {"error": "no usable seed"}
            continue
        sdir = _phase_dir(cfg, "motifs", cid, stratum)
        # 재집계(:func:`reaggregate`)가 학습을 다시 돌리지 않고 CI만 다시 낼 수 있도록
        # 시드별 test 예측을 남긴다. CNN 쪽 preds_test.npz와 같은 역할이다.
        for blk in ok:
            _save_motif_preds(sdir, int(blk["seed"]), blk["preds"])

        res = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "bag_of_patches",
            "condition_id": cid,
            "stratum": stratum,
            "seeds": [s["seed"] for s in ok],
            "config": _cfg_snapshot(cfg),
            "provenance": {
                "git_sha": git_sha(),
                "qrphish_version": _pkg_version_safe(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "data": {
                "n_total": ok[0]["n_total"],
                "n_train": ok[0]["n_train"],
                "n_val": ok[0]["n_val"],
                "n_test": ok[0]["n_test"],
                "n_windows_mean": ok[0]["n_windows_mean"],
            },
            "pooling": POOLING_MODE,
            "representations": {
                key: _motif_aggregate(ok, key, n_boot=n_boot_pool)
                for key in sorted(ok[0]["representations"])
            },
            "per_seed": [
                {k: v for k, v in s.items()
                 if k not in ("enrichment", "enrichment_inputs", "preds")}
                for s in per_seed
            ],
            "runtime_sec": None,
        }

        enr: dict[str, Any] = {"stratum": stratum}
        for size, sub_key in ((3, None), (2, "size2")):
            # CI는 5시드 train을 (seed, row)로 풀링해 한 번만 돌린다. 부트스트랩 횟수는
            # motif 축이 512개라 비용이 커서 ENRICHMENT_CI_BOOT(500)로 낮췄다.
            src = [s["enrichment_inputs"][size] for s in ok]
            pooled_enr = motif_enrichment(
                np.concatenate([d["H"] for d in src]),
                np.concatenate([d["y"] for d in src]),
                np.concatenate([d["groups"] for d in src]),
                size=size,
                n_boot=ENRICHMENT_CI_BOOT,
                seed=0,
            )
            block = combine_seed_enrichments(
                [s["enrichment"][size]["train"] for s in ok],
                [s["enrichment"][size]["test"] for s in ok],
                size=size,
                n_windows_mean=ok[0]["n_windows_mean"],
                pooled=pooled_enr,
                ci_n_boot=ENRICHMENT_CI_BOOT,
            )
            block["stratum"] = stratum
            if sub_key is None:
                enr.update(block)
            else:
                enr[sub_key] = block

        # ---- 시드별 enrichment (선정·평가 분리, review_03 2절) -----------------
        # 절제(run_occlusion)가 시드 k에서 쓸 목록은 **그 시드의 train만**으로 골라야 한다.
        # 시드를 합쳐 고르면 시드 0의 test URL이 시드 1의 train에 들어가 선정이 평가 라벨에
        # 오염된다. combine_seed_enrichments를 시드 하나짜리 목록으로 불러 같은 필터
        # (존재율·CI가 0 제외·ubiquitous 아님)를 그대로 적용한다.
        per_seed_tops: dict[int, list[int]] = {}
        # _motif_one_seed가 시드별 train enrichment CI를 만들 때 쓴 부트스트랩 횟수.
        seed_ci_boot = min(int(_get(_get(cfg, "eval"), "n_bootstrap", 2000)), 1000)
        for blk in ok:
            k = int(blk["seed"])
            seed_enr: dict[str, Any] = {
                "stratum": stratum,
                "seed": k,
                "usage": "occlusion_per_seed",
                "selection_split": "train_only",
                "note": (
                    "이 파일의 top 목록은 시드 "
                    f"{k}의 train split만으로 골랐다. 절제는 같은 시드의 test에서만 한다."
                ),
            }
            for size, sub_key in ((3, None), (2, "size2")):
                sblock = combine_seed_enrichments(
                    [blk["enrichment"][size]["train"]],
                    [blk["enrichment"][size]["test"]],
                    size=size,
                    n_windows_mean=blk["n_windows_mean"],
                    ci_n_boot=seed_ci_boot,
                )
                sblock["stratum"] = stratum
                sblock["seed"] = k
                sblock["ci_source"] = f"seed{k}_train"
                if sub_key is None:
                    seed_enr.update(sblock)
                else:
                    seed_enr[sub_key] = sblock
            per_seed_tops[k] = list(seed_enr.get("top_phishing", []))
            (sdir / f"enrichment_seed{k}.json").write_text(
                json.dumps(seed_enr, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        enr["usage"] = "reporting_only"
        enr["note"] = (
            "여러 시드의 train을 합쳐 고른 '시드 간 안정 motif' 목록이다. 시드마다 train/test "
            "분할이 달라 한 시드의 test URL이 다른 시드의 train에 들어가므로, 절제 평가에는 "
            "쓰지 않는다(enrichment_seed{k}.json을 쓴다)."
        )
        enr["per_seed_top_overlap"] = _top_list_overlap(per_seed_tops)

        res["runtime_sec"] = round(time.time() - t0, 1)
        (sdir / "results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (sdir / "enrichment.json").write_text(
            json.dumps(enr, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        out[stratum] = res
        print(f"[motifs] {stratum} 종료 ({res['runtime_sec']}s)", flush=True)
    return out


# --------------------------------------------------------------------------- probes
# review_01 "D. Lexical probe" — CNN 임베딩에서 URL 어휘 속성을 선형으로 읽어낼 수
# 있는지 본다. "복원(recover)"이 아니라 "선형 접근 가능(linearly accessible)"의 증거다.


def _probe_one_seed(cfg: Any, cid: str, stratum: str, seed: int, ngrams: list[str],
                    n_boot: int) -> dict:
    """한 (층, 시드)의 프로브 결과. ``ngrams``는 층에서 **한 번 고정한** 목표 유니버스다."""
    from qrphish.probes import build_targets, embed, run_probe_suite

    model, ds, arr, meta = _load_trained(cfg, cid, stratum, seed)
    if not hasattr(model, "feature_maps"):
        return {"seed": seed, "error": "프로브는 conv 모델(small_cnn) 임베딩에만 적용한다"}

    split = arr["split"].astype(int)
    tr_rows = np.nonzero(split == 0)[0]
    te_rows = np.nonzero(split == 2)[0]
    if tr_rows.size == 0 or te_rows.size == 0:
        return {"seed": seed, "error": "train/test가 비어 있다"}

    urls = meta["url"].astype(str).to_numpy()
    groups = meta["group"].astype(str).to_numpy()
    y = np.asarray(arr["y"], dtype=np.int64)
    targets = build_targets(urls, y, list(ngrams))

    z_tr = embed(model, ds, tr_rows)
    z_te = embed(model, ds, te_rows)

    # 대조: 같은 구조의 **미학습** CNN 임베딩. 구조/입력만으로 얻어지는 몫을 뺀다.
    # arch는 체크포인트에 기록된 값을 쓴다(문자열을 박으면 다른 조건에서 조용히 어긋난다).
    set_seed(50_000 + seed)
    rnd = build_model(_ckpt_arch(cfg, cid, stratum, seed), ds.n, ds.canonical_data_mask)
    # 학습 모델과 같은 장치여야 embed()가 같은 장치로 입력을 옮긴다.
    rnd.to(next(model.parameters()).device)
    rnd.eval()
    z_tr_r = embed(rnd, ds, tr_rows)
    z_te_r = embed(rnd, ds, te_rows)

    res = run_probe_suite(
        z_tr, z_te, z_tr_r, z_te_r, targets, tr_rows, te_rows, groups,
        seed=seed, n_boot=n_boot, progress=True,
    )
    return {
        "seed": seed,
        "n_train": int(tr_rows.size),
        "n_test": int(te_rows.size),
        "n_ngram_targets": len(ngrams),
        "targets": res,
    }


def _probe_ngram_universe(
    cfg: Any, cid: str, stratum: str, seed: int, n_ngram: int
) -> list[str]:
    """층 전체가 공유하는 char 3-gram 목표 목록. 라벨을 보지 않고 빈도로만 고른다."""
    from qrphish.probes import top_char_ngrams

    sd = _seed_dir(cfg, cid, stratum, seed)
    arr = load_stratum_arrays(sd)
    meta = pd.read_parquet(sd / "meta.parquet")
    tr_rows = np.nonzero(arr["split"].astype(int) == 0)[0]
    return top_char_ngrams(meta["url"].astype(str).to_numpy()[tr_rows], size=3, top=n_ngram)


# 결과 JSON에 그대로 싣는 해석 주의문. 세 질문을 섞어 읽지 못하게 못을 박는다.
PROBE_NOTES = (
    "프로브가 유의해도 'CNN이 문자를 복원했다'는 뜻은 아니다. "
    "임베딩에서 그 속성이 선형으로 접근 가능하다는 뜻이다.",
    "세 질문은 서로 다르다 — (1) accessible: 표현에 어휘 정보가 선형으로 접근 가능한가 "
    "(전 시드에서 학습 표현 CI 하한 > 셔플 CI 상한), (2) learned_gain: 피싱 분류 학습이 "
    "그 접근성을 높였는가 (전 시드에서 학습 표현 CI 하한 > 무작위 초기화 CNN CI 상한), "
    "(3) used_in_decision: 그 정보가 실제 분류 결정에 쓰이는가.",
    "used_in_decision은 현재 실험(동결 표현 위의 선형 프로브)으로 답할 수 없다. "
    "프로브는 표현을 읽을 뿐 결정 경로에 개입하지 않으므로 값을 null로 남긴다.",
    "accessible_and_gained(= 옛 significant)는 (1)과 (2)의 AND다. 이 기준을 통과하지 "
    "못했다는 것이 '해당 정보를 전혀 읽어낼 수 없다'는 뜻은 아니다 — accessible_only "
    "목록이 그 구분을 보여준다.",
    "셔플 기준선은 그룹 단위로 목표를 갈아끼운다(행 단위 셔플은 기준선을 부풀린다).",
    "프로브 입력은 GAP **이후** 128차원 벡터(분류 헤드의 입력)다. 'GAP 직전'이 아니다.",
    "targets[*].per_seed에 시드별 점수·CI·두 판정을 그대로 싣는다. 이 블록이 있어야 "
    "나중에 판정 규약이 바뀌어도 재학습 없이 전 시드 판정을 정확히 재집계할 수 있다.",
)
# 요약에 싣는 세 질문의 키. ``used_in_decision``은 판정 자체가 불가능하므로 뺀다.
PROBE_QUESTIONS = ("accessible", "learned_gain", "accessible_and_gained")


def _probe_top(lex: dict, names, k: int = 10) -> list[dict]:
    """점수 내림차순 상위 k개를 표에 실을 최소 필드로 줄인다."""
    ranked = sorted(names, key=lambda n: -float(lex[n]["score"]))[:k]
    return [
        {
            "target": n,
            "metric": lex[n]["metric"],
            "score": lex[n]["score"],
            "shuffle": lex[n]["shuffle"],
            "random_init": lex[n]["random_init"],
        }
        for n in ranked
    ]


def _probe_n(summary: dict, question: str) -> str:
    """요약 블록의 통과 목표 수. 미집계면 개수 대신 그렇게 적는다."""
    blk = summary.get(question)
    return str(blk["n"]) if isinstance(blk, dict) else "미집계"


def _probe_summary(lex: dict) -> dict:
    """목표별 집계 -> 세 질문 요약 (review_03 6절).

    ``accessible_only``는 **셔플 기준선은 넘지만 무작위 초기화 기준선은 넘지 못한** 목표다.
    옛 단일 ``significant`` 기준에서는 이 목표들이 통째로 "증거 없음"으로 사라졌는데,
    그건 "정보를 읽어낼 수 없다"가 아니라 "학습이 접근성을 **더** 높였다고는 말할 수
    없다"는 뜻이다. 그 구분을 요약에서 바로 보이게 따로 싣는다.
    """
    n_targets = len(lex)
    out: dict[str, Any] = {"n_targets": n_targets}
    for q in PROBE_QUESTIONS:
        # 어떤 목표라도 그 질문의 판정이 미정(None)이면 **개수 자체를 내지 않는다**.
        # 미정을 거짓으로 접어 세면 "0개 통과"와 "재집계 못 함"이 같은 수로 보인다 —
        # review_04가 지적한 오류가 정확히 그 종류였다.
        if any(v.get(q) is None for v in lex.values()):
            out[q] = None
            continue
        names = [k for k, v in lex.items() if v.get(q)]
        out[q] = {
            "n": len(names),
            "frac": (len(names) / n_targets) if n_targets else 0.0,
            "targets": sorted(names),
            "top10": _probe_top(lex, names),
        }
    if out["accessible"] is None or out["learned_gain"] is None:
        out["accessible_only"] = None
    else:
        only = [k for k, v in lex.items() if v.get("accessible") and not v.get("learned_gain")]
        out["accessible_only"] = {
            "n": len(only),
            "note": (
                "셔플 기준선은 CI 수준에서 넘지만 무작위 초기화 CNN 기준선은 넘지 못한 목표. "
                "정보는 선형으로 접근 가능하되 '학습이 접근성을 높였다'고는 말할 수 없다."
            ),
            "top10": _probe_top(lex, only),
        }
    out["used_in_decision"] = None
    # 하위 호환 별칭(표·옛 스크립트가 읽던 키). 미집계면 옛 키도 만들지 않는다.
    both = out["accessible_and_gained"]
    if isinstance(both, dict):
        out["n_significant"] = both["n"]
        out["frac_significant"] = both["frac"]
        out["top10"] = both["top10"]
    return out


# results.json의 ``targets[name].per_seed``에 싣는 시드별 원값. 표·재집계에 필요한
# 최소 집합이며, 여기에 CI가 남아야 ``accessible``의 전 시드 판정을 재실행 없이
# 정확히 복원할 수 있다.
PROBE_SEED_FIELDS = ("score", "ci", "shuffle", "shuffle_ci", "random_init", "random_init_ci")


def _probe_seed_record(seed: int, row: dict) -> dict:
    """시드 한 개의 프로브 원값 + 그 시드에서의 두 판정."""
    rec: dict[str, Any] = {"seed": int(seed)}
    for f in PROBE_SEED_FIELDS:
        v = row.get(f)
        if isinstance(v, list | tuple):
            rec[f] = [float(x) for x in v]
        elif v is not None:
            rec[f] = float(v)
    rec["accessible"] = _seed_accessible(row)
    rec["learned_gain"] = bool(row.get("above_random_init", row.get("learned_gain")))
    return rec


def _probe_aggregate(per_seed: list[dict], n_seeds: int) -> dict:
    """시드별 프로브 결과 -> 목표별 요약.

    ``ci``는 시드별 그룹 부트스트랩 CI의 하한/상한 평균이다. 정식 집계 CI가 아니므로
    판정은 **시드별로** 내린 뒤 ``n_seeds`` 전부에서 성립할 것을 요구한다. 일부 시드에서
    표본 부족으로 건너뛴 목표는 전 시드 일치를 확인할 수 없으므로 세지 않는다.

    세 질문을 분리해 각각 보고한다(review_03 6절, :mod:`qrphish.probes` 참조):
    ``accessible``(셔플 대비), ``learned_gain``(무작위 초기화 대비), 둘의 AND인
    ``accessible_and_gained``(옛 ``significant``). ``used_in_decision``은 현재 실험으로
    답할 수 없어 ``None``이다.
    """
    names: set[str] = set()
    for d in per_seed:
        names |= set(d["targets"])
    out: dict[str, Any] = {}
    for name in sorted(names):
        pairs = [
            (int(d.get("seed", i)), d["targets"][name])
            for i, d in enumerate(per_seed)
            if name in d["targets"]
        ]
        rows = [r for _, r in pairs]
        usable_pairs = [(sd, r) for sd, r in pairs if "skipped" not in r]
        usable = [r for _, r in usable_pairs]
        if not usable:
            out[name] = {"skipped": rows[0].get("skipped", "unknown"), "n_seeds": 0}
            continue
        kind = usable[0]["kind"]
        out[name] = {
            "kind": kind,
            "metric": "auroc" if kind == "binary" else "r2",
            "family": usable[0]["family"],
            "n_seeds": len(usable),
            "score": float(np.nanmean([r["score"] for r in usable])),
            "score_sd": float(np.nanstd([r["score"] for r in usable], ddof=0)),
            "ci": [
                float(np.nanmean([r["ci"][0] for r in usable])),
                float(np.nanmean([r["ci"][1] for r in usable])),
            ],
            "shuffle": float(np.nanmean([r["shuffle"] for r in usable])),
            "shuffle_ci": [
                float(np.nanmean([r["shuffle_ci"][0] for r in usable])),
                float(np.nanmean([r["shuffle_ci"][1] for r in usable])),
            ],
            "random_init": float(np.nanmean([r["random_init"] for r in usable])),
            "random_init_ci": [
                float(np.nanmean([r["random_init_ci"][0] for r in usable])),
                float(np.nanmean([r["random_init_ci"][1] for r in usable])),
            ],
            # 시드별 원값. 이게 results.json에 남아야 나중에 규약이 바뀌어도 재학습 없이
            # **정확히** 재집계할 수 있다. 이전 스키마에는 이 블록이 없어서 목표별·시드별
            # CI가 사라졌고, 그 결과 accessible의 전 시드 판정을 복원할 수 없었다
            # (review_04 "가장 중요한 문제").
            "per_seed": [_probe_seed_record(sd, r) for sd, r in usable_pairs],
            # 옛 결과(accessible 필드가 없는 seed json)도 읽을 수 있게 fallback을 둔다.
            # accessible의 fallback은 저장된 CI로 직접 재검산한다.
            "n_seeds_accessible": int(sum(_seed_accessible(r) for r in usable)),
            "n_seeds_learned_gain": int(sum(bool(r["above_random_init"]) for r in usable)),
            "n_seeds_significant": int(sum(bool(r["significant"]) for r in usable)),
            "accessible": bool(
                len(usable) == n_seeds and all(_seed_accessible(r) for r in usable)
            ),
            "learned_gain": bool(
                len(usable) == n_seeds and all(r["above_random_init"] for r in usable)
            ),
            "accessible_and_gained": bool(
                len(usable) == n_seeds and all(r["significant"] for r in usable)
            ),
            # 프로브는 표현을 읽을 뿐 결정 경로를 건드리지 않는다 — 답할 수 없음을 명시.
            "used_in_decision": None,
            # 하위 호환 별칭.
            "significant": bool(
                len(usable) == n_seeds and all(r["significant"] for r in usable)
            ),
            "above_random_init": bool(
                len(usable) == n_seeds and all(r["above_random_init"] for r in usable)
            ),
        }
    return out


def _seed_accessible(row: dict) -> bool:
    """시드 한 개의 ``accessible`` — 없으면 저장된 CI로 재검산한다.

    "학습 표현 CI 하한 > 셔플 CI 상한". 옛 seed json에는 이 필드가 없지만 ``ci``와
    ``shuffle_ci``는 남아 있으므로 재학습 없이 정확히 복원된다.
    """
    if "accessible" in row:
        return bool(row["accessible"])
    lo = float(row["ci"][0])
    hi = float(row["shuffle_ci"][1])
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo > hi)


def run_probes(cfg: Any, strata: list[str] | None = None) -> dict:
    """어휘 프로브 실험 (review_01 D). 저장된 체크포인트만 읽고 학습하지 않는다.

    층별 결과를 ``reports/probes/{stratum}/results.json``에 쓴다.
    """
    import time

    from qrphish.probes import PROBE_N_BOOT

    cid = MAIN_CONDITION
    seeds = [int(s) for s in _get(cfg, "seed_list", [0])]
    n_boot = min(int(_get(_get(cfg, "eval"), "n_bootstrap", 2000)), PROBE_N_BOOT)
    todo = list(strata) if strata else list(_get(cfg, "strata", ["v2", "v3", "v4"]))

    out: dict[str, Any] = {}
    for stratum in todo:
        # 층 단위 재개: 이미 끝난 층은 다시 돌지 않는다(중단·재개).
        done_path = _phase_dir(cfg, "probes", cid, stratum) / "results.json"
        done = _completed_result(done_path, f"probes {stratum}")
        if done is not None:
            print(f"[skip] probes {stratum} (results.json 존재)", flush=True)
            out[stratum] = done
            continue
        avail = [s for s in _available_seeds(cfg, cid, stratum) if s in set(seeds)]
        if not avail:
            # dropped 층(P0에서 표본 부족으로 학습을 돌리지 않은 층)은 체크포인트가 없다.
            # 전체 실행을 죽이지 말고 건너뛴다.
            msg = (
                f"{_out_root(cfg) / cid / stratum} 아래에 model.pt를 가진 seed 디렉터리가 없다 "
                "— 층을 건너뛴다 (P0에서 dropped 되었거나 P1이 아직 안 돌았다)."
            )
            print(f"[skip] probes {stratum}: {msg}")
            out[stratum] = {"skipped": msg}
            continue
        t0 = time.time()
        # n-gram 목표 유니버스는 층에서 **한 번만** 고른다(가장 낮은 시드의 train URL,
        # 라벨 무관 빈도). 시드마다 다시 고르면 시드별로 목표 집합이 달라져 "전 시드에서
        # 유의"라는 판정이 같은 대상을 두고 내린 판정이 아니게 된다.
        try:
            ngrams = _probe_ngram_universe(cfg, cid, stratum, avail[0], 100)
        except Exception as exc:
            print(f"[ERROR] probes {stratum}: n-gram 유니버스 구성 실패 {exc!r}")
            traceback.print_exc()
            out[stratum] = {"error": repr(exc)}
            continue
        print(
            f"[probes] {stratum} 시작 — seeds={avail}, 목표 {len(ngrams)}개 n-gram + 고정 목표, "
            f"n_boot={n_boot}",
            flush=True,
        )
        sdir = _phase_dir(cfg, "probes", cid, stratum)
        per_seed = []
        for s in avail:
            # 시드 단위 재개: seed{k}.json이 있으면 그 시드는 다시 계산하지 않는다.
            cache_path = sdir / f"seed{s}.json"
            cached = _completed_result(cache_path, f"probes {stratum} seed{s}")
            if cached is not None:
                print(f"[skip] probes {stratum} seed{s} (seed{s}.json 존재)", flush=True)
                per_seed.append(cached)
                continue
            t_seed = time.time()
            print(f"  [probes] {stratum} seed{s} 시작", flush=True)
            try:
                block = _probe_one_seed(cfg, cid, stratum, s, ngrams, n_boot)
            except Exception as exc:
                print(f"[ERROR] probes {stratum} seed{s}: {exc!r}")
                traceback.print_exc()
                block = {"seed": s, "error": repr(exc)}
            if "error" not in block:
                cache_path.write_text(
                    json.dumps(block, ensure_ascii=False, default=str), encoding="utf-8"
                )
            per_seed.append(block)
            print(
                f"  [probes] {stratum} seed{s} 종료 ({time.time() - t_seed:.1f}s)", flush=True
            )
        ok = [d for d in per_seed if "error" not in d]
        if not ok:
            out[stratum] = {"error": per_seed[0].get("error", "no usable seed")}
            continue

        agg = _probe_aggregate(ok, len(avail))
        # 라벨 프로브는 어휘 증거가 아니라 상한 참고선이므로 요약 카운트에서 뺀다.
        lex = {k: v for k, v in agg.items() if v.get("family") not in (None, "label")}
        summary = _probe_summary(lex)

        res = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "lexical_probe",
            "condition_id": cid,
            "stratum": stratum,
            "seeds": [d["seed"] for d in ok],
            "config": _cfg_snapshot(cfg),
            "provenance": {
                "git_sha": git_sha(),
                "qrphish_version": _pkg_version_safe(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "data": {"n_train": ok[0]["n_train"], "n_test": ok[0]["n_test"]},
            "n_boot": int(n_boot),
            "ngrams": list(ngrams),
            "targets": agg,
            "summary": {**summary, "label_probe": agg.get("label:phishing")},
            "notes": list(PROBE_NOTES),
            "per_seed": [
                {"seed": d["seed"], "n_train": d["n_train"], "n_test": d["n_test"]} for d in ok
            ],
            "runtime_sec": round(time.time() - t0, 1),
        }
        (sdir / "results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        out[stratum] = res
        print(
            f"[probes] {stratum} 종료 ({res['runtime_sec']}s) — "
            f"접근 가능 {_probe_n(summary, 'accessible')}/{summary['n_targets']}, "
            f"학습 이득 {_probe_n(summary, 'learned_gain')}/{summary['n_targets']}, "
            f"둘 다 {_probe_n(summary, 'accessible_and_gained')}/{summary['n_targets']}",
            flush=True,
        )
    return out


# ------------------------------------------------------------------------ occlusion
# review_01 "C. Causal motif ablation" — 상위 motif를 뒤집으면 CNN 예측이 무너지는가.


def _occlusion_aggregate(
    per_seed: list[dict],
    arrays: list[dict] | None = None,
    n_boot: int = 500,
) -> dict:
    """조건별 시드 평균 + **시드 층화 통합 CI**.

    시드별 CI 경계를 평균하는 것은 통합 효과의 95% CI가 아니다(review_03 3절). 여기서는
    시드별 값의 평균·표준편차만 요약으로 남기고, 구간 추정은
    :func:`qrphish.evaluate.paired_cluster_bootstrap_by_seed`로 5시드 test를 (seed, row)로
    풀링해 한 번만 만든다. 무작위 조건은 반복 r마다 같은 부트스트랩 시드로 돌린 뒤
    리샘플 표본을 반복 평균한다 — 리샘플 규약이 같아 반복 간 표본이 정확히 짝지어지므로,
    평균한 표본은 "반복 평균 ΔAUROC"의 부트스트랩 분포가 된다.
    """
    conds = sorted({c for d in per_seed for c in d["conditions"]})
    keys = ("auroc", "d_auroc", "mean_logit_delta",
            "mean_logit_delta_phishing", "mean_logit_delta_benign")
    out: dict[str, Any] = {
        "auroc_original": float(np.nanmean([d["auroc_original"] for d in per_seed])),
    }
    for c in conds:
        rows = [d["conditions"][c] for d in per_seed if c in d["conditions"]]
        block: dict[str, Any] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        block["d_auroc_sd_across_seeds"] = float(
            np.nanstd([r["d_auroc"] for r in rows], ddof=0)
        )
        block["n_flipped_mean"] = float(np.nanmean([r["n_flipped"]["mean"] for r in rows]))
        block["frac_samples_no_match"] = float(
            np.nanmean([r["n_flipped"]["frac_zero"] for r in rows])
        )
        # 무작위 조건: 반복 간 변동(반복별 AUROC의 표준편차·범위)을 그대로 싣는다.
        if all("d_auroc_sd_across_reps" in r for r in rows) and rows:
            block["n_repeat"] = int(rows[0]["n_repeat"])
            block["d_auroc_sd_across_reps"] = float(
                np.nanmean([r["d_auroc_sd_across_reps"] for r in rows])
            )
            block["d_auroc_range"] = [
                float(np.nanmin([r["d_auroc_range"][0] for r in rows])),
                float(np.nanmax([r["d_auroc_range"][1] for r in rows])),
            ]
            block["aggregation"] = "per_repeat_auroc_then_mean"
        # topology-matched 대조에만 있는 진단값: 후보 부족으로 밀도 제약을 푼 비율.
        relaxed = [r["frac_relaxed"] for r in rows if "frac_relaxed" in r]
        if relaxed:
            block["frac_relaxed"] = float(np.nanmean(relaxed))
        out[c] = block

    # ---- ΔΔ(표적 − 무작위) 시드 평균 --------------------------------------------
    contrast_names = sorted({c for d in per_seed for c in d.get("contrasts", {})})
    contrasts: dict[str, Any] = {}
    for name in contrast_names:
        rows = [d["contrasts"][name] for d in per_seed if name in d.get("contrasts", {})]
        contrasts[name] = {
            "target": rows[0].get("target"),
            "control": rows[0].get("control"),
            "estimate_mean_across_seeds": float(np.nanmean([r["estimate"] for r in rows])),
            "sd_across_seeds": float(np.nanstd([r["estimate"] for r in rows], ddof=0)),
            "sd_across_reps_mean": float(np.nanmean([r["sd_across_reps"] for r in rows])),
            "perm_p_per_seed": [float(r["perm_p"]) for r in rows],
        }
    if contrasts:
        out["contrasts"] = contrasts

    if arrays:
        out.update(_occlusion_pooled(arrays, conds, contrast_names, n_boot=n_boot))
    return out


def _occlusion_pooled(
    arrays: list[dict], conds: list[str], contrast_names: list[str], n_boot: int = 500
) -> dict:
    """5시드 test를 (seed, row)로 풀링한 시드 층화 CI 묶음.

    ``arrays``의 각 원소는 :func:`qrphish.occlusion.occlude_stratum`이 남긴 ``_arrays``에
    ``seed``를 붙인 것이다.
    """
    from qrphish.occlusion import OCCLUSION_CONTRASTS

    y = np.concatenate([a["y"] for a in arrays])
    g = np.concatenate([np.asarray(a["groups"]).astype(str) for a in arrays])
    sd = np.concatenate(
        [np.full(np.asarray(a["y"]).size, int(a["seed"]), dtype=np.int64) for a in arrays]
    )
    base = np.concatenate([a["base"] for a in arrays])

    def _cat_cond(name: str) -> np.ndarray | None:
        """단일 개입 조건(반복 없음)의 로짓을 (seed, row)로 이어 붙인다."""
        if all(name in a["cond"] for a in arrays):
            return np.concatenate([a["cond"][name] for a in arrays])
        return None

    def _rep_count(name: str) -> int:
        return min(len(a["reps"][name]) for a in arrays)

    def _cat_rep(name: str, r: int) -> np.ndarray:
        return np.concatenate([a["reps"][name][r] for a in arrays])

    def _pooled_delta(p_a_list: list[np.ndarray], p_b_list: list[np.ndarray],
                      seed: int) -> dict:
        """반복 쌍 목록의 Δ를 같은 리샘플로 돌린 뒤 표본을 평균해 CI를 만든다."""
        pts, samples = [], []
        for pa, pb in zip(p_a_list, p_b_list, strict=True):
            cb = paired_cluster_bootstrap_by_seed(
                y, pa, pb, g, sd, auroc_fn, n_boot=n_boot, seed=seed
            )
            pts.append(cb["point"])
            samples.append(cb["samples"])
        vals = np.mean(np.stack(samples), axis=0)
        lo, hi, n_valid = percentile_ci(vals, 0.05)
        return {
            "estimate": float(np.mean(pts)),
            "ci": [lo, hi],
            "n_boot": int(n_boot),
            "n_valid": int(n_valid),
            "pooling": "seed_stratified",
        }

    out: dict[str, Any] = {"n_pooled": int(y.size), "pooling": "seed_stratified"}
    d_pooled: dict[str, Any] = {}
    for c in conds:
        cat = _cat_cond(c)
        if cat is not None:
            d_pooled[c] = _pooled_delta([cat], [base], seed=0)
        elif all(c in a["reps"] for a in arrays):
            n_r = _rep_count(c)
            d_pooled[c] = _pooled_delta(
                [_cat_rep(c, r) for r in range(n_r)], [base] * n_r, seed=0
            )
    out["d_auroc_pooled"] = d_pooled

    dd: dict[str, Any] = {}
    for i, (name, tgt, rnd) in enumerate(OCCLUSION_CONTRASTS):
        if name not in contrast_names:
            continue
        cat = _cat_cond(tgt)
        if cat is None or not all(rnd in a["reps"] for a in arrays):
            continue
        n_r = _rep_count(rnd)
        blk = _pooled_delta([cat] * n_r, [_cat_rep(rnd, r) for r in range(n_r)], seed=i + 1)
        blk["target"] = tgt
        blk["control"] = rnd
        blk["n_repeat"] = int(n_r)
        dd[name] = blk
    if dd:
        out["delta_vs_random"] = dd
    return out


def run_occlusion(
    cfg: Any,
    strata: list[str] | None = None,
    top_k: int = 10,
    size: int = 3,
    n_random_rep: int = 5,
    n_perm: int = 500,
) -> dict:
    """인과 motif 절제 (review_01 C). ``run_motifs`` 결과가 먼저 있어야 한다.

    시드 k는 ``reports/motifs/{motif_cid}/{stratum}/enrichment_seed{k}.json``의 상위
    phishing motif만 쓴다. 그 파일은 **시드 k의 train만**으로 골랐으므로 선정이 평가 라벨에
    오염되지 않는다(review_03 2절). 시드를 합친 ``enrichment.json``으로 폴백하지 않는다 —
    폴백하면 오염된 옛 경로가 조용히 되살아난다.

    매칭된 창의 중심 모듈을 뒤집고, 같은 개수의 무작위 데이터 모듈 뒤집기 및 상위 benign
    motif 뒤집기와 원본 대비 쌍체 비교한다. 결과는
    ``reports/occlusion/{cid}/{stratum}/results.json``.
    """
    import time

    from qrphish.occlusion import load_enrichment, occlude_stratum

    cid = MAIN_CONDITION
    motif_cid = _motif_condition_id(cfg)
    seeds = [int(s) for s in _get(cfg, "seed_list", [0])]
    n_boot = min(int(_get(_get(cfg, "eval"), "n_bootstrap", 2000)), 500)
    todo = list(strata) if strata else list(_get(cfg, "strata", ["v2", "v3", "v4"]))

    out: dict[str, Any] = {}
    for stratum in todo:
        done_path = _phase_dir(cfg, "occlusion", cid, stratum) / "results.json"
        done = _completed_result(done_path, f"occlusion {stratum}")
        if done is not None:
            print(f"[skip] occlusion {stratum} (results.json 존재)", flush=True)
            out[stratum] = done
            continue
        mdir = _reports_dir(cfg) / "motifs" / motif_cid / stratum
        if not list(mdir.glob("enrichment_seed*.json")):
            # run_motifs가 돌지 않은 층(MOTIF_STRATA 밖이거나 dropped)은 건너뛴다.
            msg = (
                f"enrichment_seed*.json이 없다: {mdir} — 층을 건너뛴다 "
                "([9] motif를 먼저 돌려라. 옛 통합 enrichment.json은 쓰지 않는다)."
            )
            print(f"[skip] occlusion {stratum}: {msg}")
            out[stratum] = {"skipped": msg}
            continue
        avail = [s for s in _available_seeds(cfg, cid, stratum) if s in set(seeds)]
        if not avail:
            msg = (
                f"{_out_root(cfg) / cid / stratum} 아래에 model.pt를 가진 seed 디렉터리가 없다 "
                "— 층을 건너뛴다."
            )
            print(f"[skip] occlusion {stratum}: {msg}")
            out[stratum] = {"skipped": msg}
            continue
        t0 = time.time()
        print(f"[occlusion] {stratum} 시작 — seeds={avail}, n_boot={n_boot}", flush=True)
        per_seed = []
        arrays: list[dict] = []
        enr_paths: dict[str, str] = {}
        for s in avail:
            t_seed = time.time()
            try:
                enr_path = mdir / f"enrichment_seed{s}.json"
                if not enr_path.exists():
                    raise FileNotFoundError(
                        f"시드별 enrichment가 없다: {enr_path}\n"
                        "[9] run_motifs를 다시 돌려 enrichment_seed{k}.json을 만들어라. "
                        "시드를 합친 enrichment.json으로 대체하면 motif 선정이 평가 라벨에 "
                        "오염된다(review_03 2절)."
                    )
                enr = load_enrichment(enr_path)
                if int(enr.get("size", size)) != int(size):
                    raise ValueError(
                        f"{enr_path.name}의 size={enr.get('size')}가 "
                        f"요청한 size={size}와 다르다"
                    )
                if int(enr.get("seed", s)) != int(s):
                    raise ValueError(
                        f"{enr_path.name}의 seed={enr.get('seed')}가 {s}와 다르다"
                    )
                enr_paths[str(s)] = str(enr_path)
                model, ds, arr, meta = _load_trained(cfg, cid, stratum, s)
                te_rows = np.nonzero(arr["split"].astype(int) == 2)[0]
                groups = meta["group"].astype(str).to_numpy()[te_rows]
                block = occlude_stratum(
                    model, ds, te_rows, groups, enr,
                    top_k=top_k, size=size, n_random_rep=n_random_rep, seed=s,
                    n_boot=n_boot, n_perm=n_perm,
                )
            except Exception as exc:
                print(f"[ERROR] occlusion {stratum} seed{s}: {exc!r}")
                traceback.print_exc()
                per_seed.append({"seed": int(s), "error": repr(exc), "conditions": {}})
                continue
            arr_blk = block.pop("_arrays")
            arr_blk["seed"] = int(s)
            arrays.append(arr_blk)
            block["seed"] = int(s)
            block["motif_source"] = "per_seed_train"
            block["motif_source_path"] = str(enr_path)
            per_seed.append(block)
            print(f"  [occlusion] {stratum} seed{s} 종료 ({time.time() - t_seed:.1f}s)",
                  flush=True)
        if not any("error" not in b for b in per_seed):
            out[stratum] = {"error": "no usable seed", "per_seed": per_seed}
            print(f"[ERROR] occlusion {stratum}: 모든 시드 실패")
            continue

        top_by_seed = {
            int(a["seed"]): [
                int(v) for v in load_enrichment(enr_paths[str(a["seed"])]).get(
                    "top_phishing", []
                )[:top_k]
            ]
            for a in arrays
        }
        agg = _occlusion_aggregate(
            [b for b in per_seed if "error" not in b], arrays, n_boot=n_boot
        )
        agg["motif_top_overlap"] = _top_list_overlap(top_by_seed)

        res = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "causal_motif_occlusion",
            "condition_id": cid,
            "stratum": stratum,
            "seeds": avail,
            "config": _cfg_snapshot(cfg),
            "provenance": {
                "git_sha": git_sha(),
                "qrphish_version": _pkg_version_safe(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "motif_source": "per_seed_train",
            "motif_source_paths": enr_paths,
            "top_k": int(top_k),
            "size": int(size),
            "n_random_rep": int(n_random_rep),
            "n_perm": int(n_perm),
            "aggregate": agg,
            "per_seed": per_seed,
            "notes": [
                "뒤집은 격자는 더 이상 유효한 QR이 아니다. RS 오류정정 덕에 실제 스캐너는 "
                "여전히 디코딩할 수 있지만, 이것은 모델 입력에 대한 개입일 뿐 실제 QR 변형이 아니다.",
                "시드 k의 motif 목록은 시드 k의 train만으로 골랐다(enrichment_seed{k}.json). "
                "시드를 합친 enrichment.json은 보고용이며 절제에 쓰지 않는다(review_03 2절).",
                "무작위 대조는 각 샘플에서 짝이 되는 motif가 맞은 개수와 같은 수를 뒤집는다.",
                "무작위 조건은 반복마다 AUROC를 따로 계산해 평균한다. 반복별 로짓을 평균하면 "
                "무작위 훼손 효과가 상쇄되어 대조군 하락이 과소평가된다(review_03 3절).",
                "delta_vs_random은 같은 test에서 (표적 개입 − 무작위 개입)의 AUROC 차 ΔΔ이고, "
                "CI는 시드 층화 클러스터 부트스트랩이다. 음수면 표적 개입이 더 크게 무너뜨렸다.",
                "random_matched/random_matched_benign은 개수에 더해 사분면과 국소 흑색 밀도(3x3 창 "
                "안 1의 개수 ±1)까지 맞춘 topology-matched 대조다(review_02 4절). 후보가 모자라 밀도 "
                "제약을 푼 비율은 frac_relaxed에 남는다.",
                "매칭은 원본 격자에서 한 번에 계산한 뒤 동시에 뒤집는다(순서 의존 제거).",
            ],
            "runtime_sec": round(time.time() - t0, 1),
        }
        sdir = _phase_dir(cfg, "occlusion", cid, stratum)
        (sdir / "results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        out[stratum] = res
        print(f"[occlusion] {stratum} 종료 ({res['runtime_sec']}s)", flush=True)
    return out


# ------------------------------------------------------------------------ transfer
# 외부 검증(F). 설계 스펙 `EXTERNAL_VALIDATION_DESIGN.md` 3·4·6·7절.
# 세부 계산은 qrphish/transfer.py에 있고, 여기서는 층·시드 루프와 산출물 경로만 다룬다.
TRANSFER_MODES = ("a", "b", "c", "d")
# F-a 베이스라인·motif를 fit할 WebPhish train 행 수 상한. 전이 판정의 주 지표는 CNN AUROC라
# 베이스라인은 "함께 무너지는가"만 보면 되고, char n-gram TF-IDF를 수십만 행에서 fit하면
# 로컬 CPU 실행 예산을 혼자 다 쓴다.
TRANSFER_FIT_CAP = 30000
# motif 히스토그램은 URL당 QR 인코딩 + sliding window라 훨씬 비싸다. 따로 더 낮게 잡는다.
TRANSFER_MOTIF_FIT_CAP = 4000
TRANSFER_MOTIF_EVAL_CAP = 8000


def _transfer_mode_dir(mode: str, cohort: str, match: str = "length") -> str:
    """산출물 경로에 쓰는 모드 이름. F-a는 cohort와 매칭 조건에 따라 갈린다.

    고정 cohort와 시드별 cohort는 평가 행 집합이 다르므로 같은 경로에 쓰면 서로 덮어쓴다.
    ``match="length_path"``(경로 유무까지 맞춘 민감도 조건)도 평가 행이 달라지므로
    접미사로 갈라 둔다.
    """
    name = "a_fixed" if (mode == "a" and cohort == "fixed") else mode
    if match == "length_path":
        name = f"{name}_lenpath"
    return name


def _transfer_report_dir(cfg: Any, source_tag: str, mode: str, cid: str, stratum: str) -> Path:
    """``reports/transfer/{source_tag}/{mode}/{condition_id}/{stratum}/`` (설계 7.5).

    mode와 조건 id를 경로에 넣지 않으면 F-a/F-b/F-c 산출물이 서로 덮어쓴다.
    """
    d = _reports_dir(cfg) / "transfer" / source_tag / mode / cid / stratum
    d.mkdir(parents=True, exist_ok=True)
    return d


def _external_cond_dir(cfg: Any, source_tag: str, mode: str, cid: str) -> Path:
    """``artifacts/external/{source_tag}/{mode}/{condition_id}/``.

    1차 체크포인트가 있는 ``artifacts/{condition_id}/``를 절대 침범하면 안 된다
    (설계 9절 #19). 외부 층은 전부 이 아래로 간다.

    ``mode``를 경로에 넣지 않으면 F-a/F-b/F-c가 같은 층에서 ``grids.npz``·``meta.parquet``·
    ``results.json``을 서로 덮어쓴다(F-b는 외부 학습 분할, F-a는 평가 전용 분할이라 내용이
    다르다). 리포트 경로(:func:`_transfer_report_dir`)와 같은 층위를 쓴다.
    """
    return _out_root(cfg) / "external" / source_tag / mode / cid


def _webphish_results(cfg: Any, cid: str, stratum: str) -> dict | None:
    return _read_json_safe(_out_root(cfg) / cid / stratum / "results.json")


def _completed_result(path: Path, label: str) -> dict | None:
    """층 단위 재개용 캐시 읽기. **성공한** 결과만 돌려준다.

    실패 기록(최상위 ``error``, 또는 ``per_seed``가 전부 error)까지 "이미 있으니 건너뛴다"로
    처리하면, 버그를 고치고 다시 돌려도 그 층은 영영 실패 상태로 남는다. 그런 파일은
    ``None``을 돌려주고 ``[rerun]``을 찍어 덮어쓰게 한다.
    """
    path = Path(path)
    if not path.exists():
        return None
    res = _read_json_safe(path)
    if res is None:
        print(f"[rerun] {label}: {path.name}을 읽을 수 없다 — 다시 돌린다", flush=True)
        return None
    if "error" in res:
        print(f"[rerun] {label}: 이전 실행이 실패로 기록돼 있다 ({res['error']!r}) — 다시 돌린다",
              flush=True)
        return None
    per_seed = res.get("per_seed")
    if isinstance(per_seed, list) and per_seed and all(
        isinstance(d, dict) and "error" in d for d in per_seed
    ):
        print(f"[rerun] {label}: 성공한 시드가 없다 — 다시 돌린다", flush=True)
        return None
    return res


def _read_json_safe(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _seed_thresholds(results: dict | None) -> dict[int, float]:
    """WebPhish ``results.json``의 시드별 val 임계값. zero-shot은 이 값을 재사용한다."""
    if not results:
        return {}
    out = {}
    for s in results.get("per_seed", []):
        if "error" in s or s.get("threshold") is None:
            continue
        out[int(s["seed"])] = float(s["threshold"])
    return out


def _subsample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df.reset_index(drop=True)
    return df.sample(n=cap, random_state=seed).reset_index(drop=True)


def _split_mask(df: pd.DataFrame, code: int) -> np.ndarray:
    from qrphish.dataset import SPLIT_CODE

    vals = [SPLIT_CODE[s] if isinstance(s, str) else int(s) for s in df["split"].tolist()]
    return np.asarray(vals, dtype=np.int64) == code


def _eval_rows_mask(df: pd.DataFrame, eval_rows: str) -> np.ndarray:
    """F-a 평가 행. 주 결과는 ``all``(외부는 학습에 전혀 쓰지 않으므로 전체가 평가 대상),
    F-b와 같은 행으로 쌍체 비교하고 싶으면 ``test``."""
    if eval_rows == "all":
        return np.ones(len(df), dtype=bool)
    if eval_rows == "test":
        return _split_mask(df, 2)
    raise ValueError(f"eval_rows는 'all'|'test'여야 한다 (got {eval_rows!r})")


def _predict_with_checkpoint(cfg: Any, ckpt_path: Path, seed_dir: Path, rows: np.ndarray):
    """저장된 체크포인트로 ``seed_dir``의 층 아티팩트 일부 행을 평가한다.

    ``ds.n != ckpt["n"]``이면 하드 실패다. v5plus처럼 버전 합집합 층에서 외부 데이터의
    최대 버전이 다르면 격자 크기가 달라져 같은 CNN을 쓸 수 없다(설계 3.4 #4).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    features, mask_mode = _ckpt_dataset_spec(cfg, ckpt, ckpt_path)
    arr = load_stratum_arrays(seed_dir)
    meta = pd.read_parquet(seed_dir / "meta.parquet")
    ds = QRGridDataset(
        arr["X_packed"], arr["y"], int(arr["n"]), arr["versions"], int(arr["ec"]),
        features=features, mask_mode=mask_mode,
    )
    if int(ds.n) != int(ckpt["n"]):
        raise ValueError(
            f"격자 크기가 체크포인트와 다르다: ds.n={ds.n}, ckpt['n']={ckpt['n']} "
            f"({ckpt_path}). 같은 층이라도 버전 분포가 다르면 같은 CNN을 쓸 수 없다 "
            "(설계 3.4 #4)."
        )
    device = pick_device()
    model = build_model(ckpt["arch"], ds.n, ds.canonical_data_mask)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    # 체크포인트는 map_location="cpu"로 읽으므로, 여기서 옮기지 않으면 입력만 CUDA로 가서
    # "Input type ... and weight type ... should be the same"로 죽는다.
    assert next(model.parameters()).device.type == device.type, (
        f"모델 장치({next(model.parameters()).device})와 추론 장치({device})가 다르다"
    )

    sub = copy.copy(ds)
    sub.indices = np.nonzero(np.asarray(rows))[0]
    loader = DataLoader(sub, batch_size=256, shuffle=False, num_workers=0)
    y, p = predict_probs(model, loader, device=device)
    return y, p, meta, arr


def _motif_hists(cfg: Any, frames: dict[str, pd.DataFrame], cache: dict) -> dict:
    """여러 프레임에 대해 같은 ``n_max``로 motif 히스토그램을 만든다.

    ``n_max``가 프레임마다 다르면 ``data_mask_for``가 달라져 창 개수가 달라지고, 두
    히스토그램이 비교 불가능해진다. 그래서 모든 프레임의 최대 버전에서 한 번에 정한다.
    """
    vmax = max(int(df["version"].max()) for df in frames.values() if len(df))
    n_max = 4 * vmax + 17
    return {k: _motif_features(cfg, df, cache, n_max)[0] for k, df in frames.items()}


def _transfer_baseline_block(
    cfg: Any,
    stratum: str,
    seed: int,
    ext_eval: pd.DataFrame,
    *,
    fit_frame: pd.DataFrame | None,
    fit_on: str,
    with_motif: bool,
    preds_out: dict | None = None,
    motif_eval_cap: int = TRANSFER_MOTIF_EVAL_CAP,
    eval_hist_cache: dict | None = None,
) -> dict:
    """F-a/F-c 베이스라인: ``fit_frame``(train)에서 fit → 외부 평가 행에 apply.

    설계 3.7 — CNN만 무너지는지, 텍스트 기준선도 함께 무너지는지가 해석 매트릭스를 가른다.

    ``preds_out``을 주면 기준선별 예측 점수를 담아 준다(CNN과의 쌍체 ΔAUROC용).
    ``motif_eval_cap<=0``이면 motif 히스토그램을 평가 cohort **전체**에서 만든다 — 이때만
    motif 기준선이 CNN과 같은 행을 보므로 쌍체 비교가 성립한다. 고정 cohort에서는
    ``eval_hist_cache``로 평가 쪽 히스토그램을 시드 간에 재사용한다(F-a 비용의 대부분).
    """
    from qrphish.transfer import transfer_baselines

    if fit_frame is None or len(fit_frame) == 0:
        return {}
    base = transfer_baselines(
        _subsample(fit_frame, TRANSFER_FIT_CAP, seed), ext_eval, seed, fit_on=fit_on,
        preds_out=preds_out,
    )
    for v in base.values():
        v["n_fit"] = int(min(len(fit_frame), TRANSFER_FIT_CAP) if TRANSFER_FIT_CAP > 0
                         else len(fit_frame))
    if not with_motif:
        return base
    # motif 히스토그램은 URL당 QR 인코딩이 필요해 훨씬 비싸다. fit에는 따로 캡을 걸고,
    # 실패하더라도 전이 판정(CNN AUROC 대 순열 바닥선)은 그대로 진행한다.
    try:
        fit_m = _subsample(fit_frame, TRANSFER_MOTIF_FIT_CAP, seed)
        ev_m = ext_eval.reset_index(drop=True) if motif_eval_cap <= 0 else _subsample(
            ext_eval, motif_eval_cap, seed
        )
        # n_max는 fit·eval 두 프레임의 최대 버전에서 한 번에 정해야 창 개수가 같아진다.
        # 고정 cohort에서는 eval 히스토그램이 시드와 무관하므로 캐시해 재계산을 피한다
        # (F-a 비용의 대부분이 여기다).
        vmax = max(int(fit_m["version"].max()), int(ev_m["version"].max()))
        n_max = 4 * vmax + 17
        cache_key = ("eval", int(len(ev_m)), int(motif_eval_cap), int(n_max))
        if eval_hist_cache is not None and cache_key in eval_hist_cache:
            H_eval = eval_hist_cache[cache_key]
        else:
            H_eval = _motif_features(cfg, ev_m, {}, n_max)[0]
            if eval_hist_cache is not None:
                eval_hist_cache[cache_key] = H_eval
        H = {"fit": _motif_features(cfg, fit_m, {}, n_max)[0], "eval": H_eval}
        motif = {
            key: {"H_fit": H["fit"][key], "H_eval": H["eval"][key]}
            for key in ("patch3", "pyramid3")
        }
        motif_preds: dict | None = {} if preds_out is not None else None
        got = transfer_baselines(
            fit_m, ev_m, seed, fit_on=fit_on, motif=motif, preds_out=motif_preds
        )
        paired = bool(len(ev_m) == len(ext_eval))
        for k, v in got.items():
            if not k.startswith("motif_"):
                continue
            v["n_fit"] = int(len(fit_m))
            v["eval_rows_same_as_cnn"] = paired
            base[k] = v
            if preds_out is not None and motif_preds is not None and paired:
                preds_out[k] = motif_preds[k]
    except Exception as exc:  # motif는 보조 지표다
        print(f"[WARN] transfer motif 베이스라인 실패 {stratum} seed{seed}: {exc!r}")
    return base


def _aggregate_baselines(per_seed_base: list[dict]) -> dict:
    """시드별 베이스라인 결과 → 시드 평균 + 시드별 값. 모든 시드에 있는 이름만 남긴다."""
    ok = [b for b in per_seed_base if b]
    if not ok:
        return {}
    names = set(ok[0])
    for b in ok[1:]:
        names &= set(b)
    out: dict[str, Any] = {}
    for name in sorted(names):
        vals = [float(b[name]["auroc"]) for b in ok]
        first = ok[0][name]
        out[name] = {
            "auroc": float(np.mean(vals)),
            "auroc_per_seed": vals,
            "auroc_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            # CI는 첫 시드의 그룹 클러스터 부트스트랩이다. 시드 층화 CI는
            # comparison_to_baselines의 쌍체 Δ 쪽에서 낸다.
            "ci": first.get("ci"),
            "ci_note": "첫 시드의 그룹 클러스터 CI. 시드 층화 불확실성은 쌍체 Δ를 볼 것",
            "fit_on": first.get("fit_on"),
            "n": first.get("n"),
            "n_fit": first.get("n_fit"),
            "n_seeds": len(vals),
            "eval_rows_same_as_cnn": first.get("eval_rows_same_as_cnn", True),
        }
    dropped = sorted((set().union(*(set(b) for b in ok))) - names)
    if dropped:
        out["_incomplete_across_seeds"] = dropped  # type: ignore[assignment]
    return out


def _paired_baseline_deltas(
    per_seed: list[dict], per_seed_preds: list[dict], baselines: dict, n_boot: int
) -> dict:
    """CNN − 각 기준선의 **쌍체** ΔAUROC와 시드 층화 CI (리뷰 03 항목 8)."""
    from qrphish.transfer import paired_delta_auroc

    ok = [(d, pr) for d, pr in zip(per_seed, per_seed_preds, strict=True)
          if "error" not in d and pr]
    if not ok:
        return {}
    names = set(ok[0][1])
    for _, pr in ok[1:]:
        names &= set(pr)
    y = np.concatenate([d["y"] for d, _ in ok])
    p_cnn = np.concatenate([d["p"] for d, _ in ok])
    g = np.concatenate([np.asarray(d["group"]).astype(str) for d, _ in ok])
    sd = np.concatenate([np.full(d["y"].size, int(d["seed"]), dtype=np.int64) for d, _ in ok])
    out: dict[str, Any] = {}
    for name in sorted(names):
        p_b = np.concatenate([pr[name] for _, pr in ok])
        if p_b.size != y.size:
            continue
        res = paired_delta_auroc(y, p_cnn, p_b, g, sd, n_boot=n_boot, seed=0)
        res["baseline"] = name
        res["n_seeds"] = len(ok)
        res["n_rows"] = int(y.size)
        out[name] = res
    skipped = sorted(set(baselines) - set(out) - {"_incomplete_across_seeds"})
    if skipped:
        out["_skipped"] = {  # type: ignore[assignment]
            "names": skipped,
            "reason": "모든 시드에서 같은 평가 행의 예측을 얻지 못했다(캡 적용 등)",
        }
    return out


def _transfer_data_block(df: pd.DataFrame, sm: Any, eval_rows: str, tier: str) -> dict:
    ev = df
    return {
        "n_total": int(len(ev)),
        "n_benign": int((ev["label"] == 0).sum()),
        "n_phishing": int((ev["label"] == 1).sum()),
        "base_rate": float(ev["label"].mean()) if len(ev) else float("nan"),
        "n_groups": int(ev["group"].nunique()),
        "tier": tier,
        "eval_rows": eval_rows,
        "split_diagnostics": {
            "top5_test_group_frac": float(
                getattr(sm, "extra", {}).get("top5_test_group_frac", 0.0)
            )
            if sm is not None
            else None
        },
    }


def _transfer_model_block(
    per_seed: list[dict], n_boot: int, threshold_source: str
) -> dict:
    """시드별 예측을 모아 **시드 층화** 그룹 클러스터 부트스트랩으로 CI를 낸다."""
    from qrphish.evaluate import acc_at, auprc, f1_at

    ok = [d for d in per_seed if "error" not in d]
    if not ok:
        return {"auroc_mean": float("nan"), "auroc_per_seed": [], "error": "no usable seed"}
    aur = [float(auroc_fn(d["y"], d["p"])) for d in ok]
    y = np.concatenate([d["y"] for d in ok])
    p = np.concatenate([d["p"] for d in ok])
    g = np.concatenate([np.asarray(d["group"]).astype(str) for d in ok])
    s = np.concatenate([np.full(d["y"].size, int(d["seed"]), dtype=np.int64) for d in ok])
    cb = cluster_bootstrap_by_seed(y, p, g, s, auroc_fn, n_boot=n_boot, seed=0)
    f1s = [float(f1_at(d["y"], d["p"], float(d["threshold"]))) for d in ok if d.get("threshold")]
    accs = [float(acc_at(d["y"], d["p"], float(d["threshold"]))) for d in ok if d.get("threshold")]
    return {
        "auroc_mean": float(np.mean(aur)),
        "auroc_sd": float(np.std(aur, ddof=1)) if len(aur) > 1 else 0.0,
        "auroc_per_seed": aur,
        "auroc_pooled": float(cb["point"]),
        "auroc_pooled_ci": [float(cb["lo"]), float(cb["hi"])],
        "pooling": POOLING_MODE,
        "auprc_mean": float(np.mean([auprc(d["y"], d["p"]) for d in ok])),
        "f1_at_val_threshold": float(np.mean(f1s)) if f1s else float("nan"),
        "acc_at_val_threshold": float(np.mean(accs)) if accs else float("nan"),
        "threshold_source": threshold_source,
        "thresholds": [d.get("threshold") for d in ok],
        "n_boot": int(n_boot),
        "seeds": [int(d["seed"]) for d in ok],
    }


def _pooled_arrays(per_seed: list[dict]) -> tuple[np.ndarray, ...]:
    ok = [d for d in per_seed if "error" not in d]
    return (
        np.concatenate([d["y"] for d in ok]),
        np.concatenate([d["p"] for d in ok]),
        np.concatenate([np.asarray(d["group"]).astype(str) for d in ok]),
        np.concatenate([np.full(d["y"].size, int(d["seed"]), dtype=np.int64) for d in ok]),
    )


def _fa_cohort(
    cfg: Any,
    stratum: str,
    cohort_seed: int,
    ext_df: pd.DataFrame,
    ext_cond_dir: Path,
    eval_rows: str,
    match: str = "length",
) -> dict:
    """F-a 평가 cohort 하나(외부 층 프레임 + 격자 + 평가 행 마스크)를 만든다.

    ``cohort="fixed"``에서는 이 결과를 다섯 모델 시드가 **공유**한다. 그러면 시드 간
    변동에 모델만 남고 평가 분포는 완전히 같아진다(리뷰 02). 격자 생성이 F-a 비용의
    대부분이므로 재사용은 속도에도 이득이다.
    """
    from qrphish.transfer import build_external_stratum, prepare_external_frame

    df, diag = prepare_external_frame(cfg, ext_df, stratum, cohort_seed, match=match)
    if len(df) == 0:
        raise RuntimeError(f"F-a {stratum}: 길이 매칭 후 외부 층이 비었다 (cohort seed {cohort_seed})")
    seed_dir = ext_cond_dir / stratum / f"seed{cohort_seed}"
    sm = build_external_stratum(cfg, df, stratum, cohort_seed, seed_dir)
    meta = pd.read_parquet(seed_dir / "meta.parquet")
    return {
        "cohort_seed": int(cohort_seed),
        "frame": df,
        "seed_dir": seed_dir,
        "stratum_meta": sm,
        "rows": _eval_rows_mask(meta, eval_rows),
        "n_after_match": diag.get("n_after_match"),
        "match": str(match),
        "path_match": diag.get("path_match"),
    }


def _fa_one_seed(
    cfg: Any,
    cid: str,
    stratum: str,
    seed: int,
    ext_df: pd.DataFrame,
    ext_cond_dir: Path,
    eval_rows: str,
    thresholds: dict[int, float],
    cohort: dict | None = None,
    match: str = "length",
) -> dict:
    """F-a 한 시드: 외부 프레임 준비 → 층 조립 → WebPhish 체크포인트로 추론.

    ``cohort``를 주면 그 고정 cohort의 격자·평가 행을 그대로 쓰고, 시드에 따라 바뀌는 것은
    체크포인트(모델)와 임계값뿐이다.
    """
    co = cohort or _fa_cohort(cfg, stratum, seed, ext_df, ext_cond_dir, eval_rows, match)
    rows = co["rows"]
    ckpt_path = _seed_dir(cfg, cid, stratum, seed) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"WebPhish 체크포인트가 없다: {ckpt_path}. F-a는 1차 학습 산출물을 재사용한다."
        )
    y, p, meta, _arr = _predict_with_checkpoint(cfg, ckpt_path, co["seed_dir"], rows)
    idx = np.nonzero(rows)[0]
    return {
        "seed": seed,
        "y": np.asarray(y, dtype=np.int64),
        "p": np.asarray(p, dtype=np.float64),
        "group": meta["group"].to_numpy()[idx].astype(str),
        "threshold": thresholds.get(int(seed)),
        "frame": co["frame"],
        "eval_frame": meta.iloc[idx].reset_index(drop=True),
        "stratum_meta": co["stratum_meta"],
        "n_after_match": co["n_after_match"],
        "cohort_seed": int(co["cohort_seed"]),
        "path_match": co.get("path_match"),
    }


def _fc_one_seed(cfg: Any, cid: str, stratum: str, seed: int, ext_cond_dir: Path) -> dict:
    """F-c 한 시드: 외부에서 학습한 체크포인트로 **WebPhish test 행**을 평가한다."""
    ckpt_path = ext_cond_dir / stratum / f"seed{seed}" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"외부 학습 체크포인트가 없다: {ckpt_path} (F-b를 먼저 돌려야 한다)")
    wp_dir = _seed_dir(cfg, cid, stratum, seed)
    arr = load_stratum_arrays(wp_dir)
    rows = arr["split"].astype(int) == 2
    y, p, meta, _ = _predict_with_checkpoint(cfg, ckpt_path, wp_dir, rows)
    idx = np.nonzero(rows)[0]
    ext_res = _read_json_safe(ext_cond_dir / stratum / "results.json")
    return {
        "seed": seed,
        "y": np.asarray(y, dtype=np.int64),
        "p": np.asarray(p, dtype=np.float64),
        "group": meta["group"].to_numpy()[idx].astype(str),
        "threshold": _seed_thresholds(ext_res).get(int(seed)),
        "eval_frame": meta.iloc[idx].reset_index(drop=True),
    }


def _transfer_results(
    cfg: Any,
    *,
    mode: str,
    source_tag: str,
    cid: str,
    stratum: str,
    data: dict,
    model: dict,
    null_perm: dict,
    baselines: dict,
    in_domain: float | None,
    gates: dict,
    extra: dict | None = None,
    common_direction_bias: bool = False,
    path_shortcut_dominant: bool = False,
    match: str = "length",
) -> dict:
    """설계 7.5 결과 스키마. 키를 비워서라도 유지한다(표 생성기가 이 키를 읽는다)."""
    from qrphish.transfer import verdict as _verdict

    auroc = float(model.get("auroc_mean", float("nan")))
    ci = list(model.get("auroc_pooled_ci", [float("nan"), float("nan")]))
    text = (baselines.get("charngram_lr") or {}).get("auroc")
    res = {
        "schema_version": SCHEMA_VERSION,
        "phase": "transfer",
        "mode": mode,
        "source_tag": source_tag,
        "condition_id": cid,
        "stratum": stratum,
        "git_sha": git_sha(),
        "qrphish_version": _pkg_version_safe(),
        "created_at": datetime.now(UTC).isoformat(),
        "config": _cfg_snapshot(cfg),
        "data": data,
        "model": model,
        "null_permutation": null_perm,
        "baselines": baselines,
        "comparison_to_in_domain": {
            "in_domain_auroc": in_domain,
            "delta": (float(auroc - in_domain) if in_domain is not None else None),
            "paired": False,
            "delta_ci": [None, None],
            "transfer_gap_to_text": (
                float(text - auroc) if text is not None and text == text else None
            ),
        },
        "verdict": _verdict(
            auroc, ci, float(null_perm.get("ci_upper", float("nan"))), in_domain, gates,
            common_direction_bias=common_direction_bias,
            path_shortcut_dominant=path_shortcut_dominant,
            match=match,
        ),
    }
    if extra:
        res.update(extra)
    return res


def run_transfer(
    cfg: Any,
    external_csv: str | Path,
    sources: list[str] | None = None,
    strata: list[str] | None = None,
    *,
    modes: tuple[str, ...] | list[str] = ("a",),
    cohort: str = "fixed",
    source_tag: str | None = None,
    cid: str = MAIN_CONDITION,
    eval_rows: str = "all",
    n_perm: int = 2000,
    dedup: str = "etld1",
    motif: bool = True,
    match: str = "length",
    motif_eval_cap: int = TRANSFER_MOTIF_EVAL_CAP,
    entry: dict | None = None,
) -> dict:
    """외부 검증(F). 설계 3·4·6·7절.

    Args:
        external_csv: 병합·중복제거가 끝난 외부 평가 세트 CSV.
        sources: ``source`` 컬럼으로 거를 소스 이름들(예: ``["openphish"]``). None이면 전부.
        strata: 대상 층. None이면 config의 ``strata``.
        modes: 실행할 F 모드. ``"a"``(zero-shot, 학습 없음) / ``"b"``(외부 자체 학습) /
            ``"c"``(역방향) / ``"d"``(혼합 — 미구현).
        cohort: F-a 평가 cohort. ``"fixed"``(기본, 주 결과)는 외부 층 프레임을 시드 0의
            분할·매칭으로 한 번 만들어 다섯 모델이 같은 행을 본다. ``"per_seed"``는
            시드마다 다시 매칭하며 F-b와의 쌍체 비교(secondary)용이다. 두 결과는 서로 다른
            경로에 쓴다: ``reports/transfer/{tag}/a_fixed/…`` 대 ``…/a/…``.
        eval_rows: F-a 평가 행. 주 결과는 ``"all"``.
        n_perm: 라벨 순열 횟수(바닥선). 결과 JSON의 ``null_permutation.n_perm``에 그대로
            기록되며 그 값이 정본이다. 기본 2,000회.
        match: F-a 외부 층의 매칭 조건. ``"length"``(기본, 1차 실험과 동일) 또는
            ``"length_path"``(길이 + 경로 유무 동시 매칭, 리뷰 03 항목 4 민감도).
            ``"length_path"`` 산출물은 ``…/a_fixed_lenpath/…`` 경로로 따로 나간다.
        motif_eval_cap: motif 히스토그램 평가 행 상한. ``0`` 이하면 cohort 전체를 쓰고,
            그때만 motif 기준선이 CNN과 같은 행을 보므로 쌍체 Δ가 계산된다.

    Returns:
        ``{mode: {stratum: results}}``. 부작용으로
        ``reports/transfer/{source_tag}/{mode_dir}/{cid}/{stratum}/results.json``을 쓴다
        (``mode_dir``은 F-a 고정 cohort일 때 ``a_fixed``).

    층·시드 단위로 예외를 잡는다. 한 층이 터져도 나머지 층은 끝까지 간다 — 여러 시간짜리
    실행에서 가장 나쁜 결과는 마지막 층에서 죽는 것이다.
    """
    import time

    from qrphish.transfer import (
        collection_bias_gate,
        ensure_version_column,
        filter_sources,
        load_external_frame,
    )

    modes = tuple(str(m) for m in modes)
    if match not in ("length", "length_path"):
        raise ValueError(f"match는 'length'|'length_path'여야 한다 (got {match!r})")
    if cohort not in ("fixed", "per_seed"):
        raise ValueError(f"cohort는 'fixed'|'per_seed' 중 하나여야 한다 (got {cohort!r})")
    for m in modes:
        if m not in TRANSFER_MODES:
            raise ValueError(f"transfer mode는 {TRANSFER_MODES} 중 하나여야 한다 (got {m!r})")
    if "d" in modes:
        raise NotImplementedError(
            "F-d(혼합 학습)는 이번 회차 범위 밖이다. 그룹 분할을 두 데이터셋 합집합 위에서 "
            "한 번에 해야 하고 origin_lr 베이스라인이 필수다 (설계 3.2)."
        )

    url_mode = str(_get(_get(cfg, "condition"), "url_mode", "norm"))
    ext_all, ext_stats = load_external_frame(
        external_csv, mode=url_mode, dedup=dedup,
        webphish_csv=_get(_get(cfg, "data"), "csv_path"),
    )
    ext_all = filter_sources(ext_all, sources)
    # QR 버전은 층·시드와 무관하므로 **여기서 한 번만** 계산해 캐시한다. 외부 세트는
    # 60만 행 규모라 층 × 시드마다 다시 계산하면 같은 일을 20번 반복하게 된다.
    _t_ver = time.time()
    ensure_version_column(cfg, ext_all)
    print(f"[transfer] natural_version 캐시 {len(ext_all)}행 "
          f"({time.time() - _t_ver:.1f}s)", flush=True)
    tag = source_tag or Path(str(external_csv)).stem
    todo = list(strata) if strata else list(_get(cfg, "strata", VERSION_SPECS))
    seeds = [int(s) for s in _get(cfg, "seed_list", [0])]
    n_boot = int(_get(_get(cfg, "eval"), "n_bootstrap", 2000))
    rules = _get(cfg, "stratum_rules")
    secondary_min = int(_get(rules, "secondary_min_per_class", 300))
    entry = dict(entry or {})
    # 수집 단계가 이미 내린 편향 게이트를 승계한다. 길이 매칭된 프레임으로 재계산하면
    # 수집 단계에서 fail이던 세트가 조용히 통과한다(설계 2.3·6.1).
    coll_gate = collection_bias_gate(external_csv, _reports_dir(cfg).parent)
    if coll_gate is not None:
        print(f"[transfer] 수집 단계 편향 게이트 승계: gate_passed={coll_gate['gate_passed']} "
              f"({coll_gate['source']}#{coll_gate['block']})", flush=True)
    else:
        print("[transfer] 수집 단계 편향 게이트를 못 찾았다 — 층별로 재계산한다.", flush=True)

    out: dict[str, Any] = {m: {} for m in modes}
    print(f"[transfer] source={tag} n_external={len(ext_all)} modes={modes} strata={todo}",
          flush=True)

    for stratum in todo:
        for mode in modes:
            mode_dir = _transfer_mode_dir(mode, cohort, match if mode == "a" else "length")
            rpath = _transfer_report_dir(cfg, tag, mode_dir, cid, stratum) / "results.json"
            done = _completed_result(rpath, f"transfer F-{mode} {stratum}")
            if done is not None:
                print(f"[skip] transfer {mode} {stratum} (results.json 존재)", flush=True)
                out[mode][stratum] = done
                continue
            t0 = time.time()
            # F-c는 F-b가 외부에서 학습한 체크포인트를 읽는다 — 그래서 "b" 디렉터리를 본다.
            ext_cond_dir = _external_cond_dir(
                cfg, tag, "b" if mode == "c" else mode_dir, cid
            )
            print(f"[transfer] F-{mode} {stratum} 시작 — seeds={seeds}", flush=True)
            try:
                if mode == "a":
                    res = _run_transfer_a(
                        cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir,
                        eval_rows=eval_rows, n_perm=n_perm, n_boot=n_boot,
                        secondary_min=secondary_min, motif=motif, ext_stats=ext_stats,
                        coll_gate=coll_gate, cohort=cohort, match=match,
                        motif_eval_cap=motif_eval_cap,
                    )
                elif mode == "b":
                    res = _run_transfer_b(
                        cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir,
                        n_perm=n_perm, n_boot=n_boot, entry=entry, coll_gate=coll_gate,
                    )
                else:
                    res = _run_transfer_c(
                        cfg, cid, tag, stratum, seeds, ext_cond_dir,
                        n_perm=n_perm, n_boot=n_boot, coll_gate=coll_gate,
                    )
            except Exception as exc:
                print(f"[ERROR] transfer F-{mode} {stratum}: {exc!r}")
                traceback.print_exc()
                res = {
                    "schema_version": SCHEMA_VERSION, "phase": "transfer", "mode": mode,
                    "source_tag": tag, "condition_id": cid, "stratum": stratum,
                    "error": repr(exc), "data": {}, "model": {}, "null_permutation": {},
                    "baselines": {}, "comparison_to_in_domain": {},
                    "verdict": {"label": "descriptive_only", "gates_passed": {},
                                "criteria_version": "prereg_v1"},
                }
            res["runtime_sec"] = round(time.time() - t0, 1)
            rpath.write_text(
                json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            out[mode][stratum] = res
            print(
                f"[transfer] F-{mode} {stratum} 종료 ({res['runtime_sec']}s) "
                f"verdict={res.get('verdict', {}).get('label')}",
                flush=True,
            )
    # 게이트 판단에 쓴 외부 로더 통계는 층과 무관하므로 한 번만 남긴다.
    (_reports_dir(cfg) / "transfer" / tag / "external_stats.json").write_text(
        json.dumps(ext_stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return out


def _apply_collection_gate(data: dict, coll_gate: dict | None, recomputed_flag: bool) -> bool:
    """공통 방향 편향 **플래그**를 결정한다 — 수집 단계 기록이 있으면 그것을 쓴다.

    반환값은 차단 게이트가 아니라 :func:`qrphish.transfer.verdict`에 넘길 플래그다.
    길이 매칭된 프레임에서 재계산하면 수집 단계와 값이 달라질 수 있으므로(매칭이 경로
    보유율 분포를 바꾼다) 수집 단계 판정을 우선 승계하고, 어느 쪽을 썼는지 남긴다.
    """
    if coll_gate is not None:
        flag = bool(coll_gate.get("common_direction_bias", not coll_gate["gate_passed"]))
        data["bias_gate"] = {
            "source": "collection",
            "common_direction_bias": flag,
            "path": coll_gate.get("source"),
            "block": coll_gate.get("block"),
            "note": coll_gate.get("gate_note"),
            "recomputed_transfer_stage": bool(recomputed_flag),
            "blocking": False,
        }
        return flag
    data["bias_gate"] = {
        "source": "recomputed_transfer_stage",
        "common_direction_bias": bool(recomputed_flag),
        "note": "수집 단계 bias_diagnostics.json을 못 찾아 층 프레임에서 재계산했다.",
        "blocking": False,
    }
    return bool(recomputed_flag)


def _run_transfer_a(
    cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir, *,
    eval_rows, n_perm, n_boot, secondary_min, motif, ext_stats, coll_gate=None,
    cohort="fixed", match="length", motif_eval_cap=TRANSFER_MOTIF_EVAL_CAP,
) -> dict:
    """F-a — zero-shot. 학습하지 않는다. 임계값은 WebPhish val 값을 재사용한다.

    ``cohort``(리뷰 02):

    - ``"fixed"`` (주 결과): 외부 층 프레임을 **시드 0의 분할·매칭으로 한 번만** 만들고
      WebPhish 다섯 모델을 전부 같은 행에 평가한다. 시드 간에 바뀌는 것은 모델뿐이라
      전이 성능의 시드 변동이 평가 분포 변동과 섞이지 않는다.
    - ``"per_seed"``: 시드마다 분할·매칭을 다시 한다. F-b와 같은 행 집합을 쓰므로
      F-a 대 F-b 쌍체 비교(secondary)에만 쓴다.
    """
    from qrphish.dataset import stratum_tier
    from qrphish.transfer import (
        bias_direction_ok,
        gates_from,
        path_shortcut_flag,
        permutation_null,
        row_permutation_null,
    )

    if cohort not in ("fixed", "per_seed"):
        raise ValueError(f"cohort는 'fixed'|'per_seed' 중 하나여야 한다 (got {cohort!r})")
    wp_res = _webphish_results(cfg, cid, stratum)
    thresholds = _seed_thresholds(wp_res)
    fixed_cohort = None
    if cohort == "fixed":
        cohort_seed = int(seeds[0]) if seeds else 0
        fixed_cohort = _fa_cohort(
            cfg, stratum, cohort_seed, ext_all, ext_cond_dir, eval_rows, match
        )
        print(f"[transfer] F-a {stratum}: 고정 cohort (seed{cohort_seed} 분할·매칭, "
              f"match={match}) {int(np.asarray(fixed_cohort['rows']).sum())}행을 "
              "모든 모델 시드가 공유", flush=True)
    per_seed: list[dict] = []
    n_failed = 0
    for s in seeds:
        try:
            per_seed.append(
                _fa_one_seed(cfg, cid, stratum, s, ext_all, ext_cond_dir, eval_rows,
                             thresholds, cohort=fixed_cohort, match=match)
            )
        except Exception as exc:
            # 시드마다 같은 traceback을 반복해 찍으면 로그에서 원인을 찾기 어렵다.
            # 첫 실패만 전체 traceback, 나머지는 한 줄 요약.
            print(f"[ERROR] transfer F-a {stratum} seed{s}: {exc!r}")
            if n_failed == 0:
                traceback.print_exc()
            n_failed += 1
            per_seed.append({"seed": s, "error": repr(exc)})
    ok = [d for d in per_seed if "error" not in d]
    if not ok:
        raise RuntimeError(
            f"F-a {stratum}: 모든 시드 실패 ({n_failed}/{len(per_seed)}) "
            "— run_transfer가 이 층을 건너뛰고 다음 단계(F-b)로 진행한다"
        )

    first = ok[0]
    ev = first["eval_frame"]
    per_class_min = int(ev["label"].value_counts().min()) if len(ev) else 0
    tier = stratum_tier(per_class_min, secondary_min=secondary_min)
    if per_class_min < secondary_min:
        # 사전 등록된 층 채택 규칙을 외부 데이터에 맞춰 완화하지 않는다(설계 3.4 #1).
        print(f"[transfer] F-a {stratum}: 외부 표본 부족 (클래스당 {per_class_min}) — 서술만",
              flush=True)

    model = _transfer_model_block(per_seed, n_boot, "webphish_val")
    y, p, g, sd = _pooled_arrays(per_seed)
    null_perm = permutation_null(y, p, g, sd, n_perm=n_perm, seed=0)
    null_row = row_permutation_null(y, p, sd, n_perm=n_perm, seed=0)

    # 베이스라인은 **모든 성공 시드**에서 낸다(리뷰 03 항목 8). 시드 s의 WebPhish train에서
    # fit해 같은 (고정) cohort에 apply하므로, CNN과 시드가 정확히 짝지어진다.
    wp_frame = None
    per_seed_base: list[dict] = []
    per_seed_preds: list[dict] = []
    eval_hist_cache: dict = {}
    for d in ok:
        preds: dict = {}
        wp_s, _ = _prepare_frame(cfg, stratum, int(d["seed"]))
        if wp_frame is None:
            wp_frame = wp_s
        wp_train_s = wp_s[_split_mask(wp_s, 0)].reset_index(drop=True)
        per_seed_base.append(
            _transfer_baseline_block(
                cfg, stratum, int(d["seed"]), d["eval_frame"],
                fit_frame=wp_train_s, fit_on="webphish_train", with_motif=motif,
                preds_out=preds, motif_eval_cap=motif_eval_cap,
                eval_hist_cache=(eval_hist_cache if cohort == "fixed" else None),
            )
        )
        per_seed_preds.append(preds)
    baselines = _aggregate_baselines(per_seed_base)
    paired_deltas = _paired_baseline_deltas(per_seed, per_seed_preds, baselines, n_boot)

    bias = bias_direction_ok(ev, wp_frame)
    sm = first.get("stratum_meta")
    data = _transfer_data_block(ev, sm, eval_rows, tier)
    data["length_match"] = {"bucket": _length_bucket(_get(cfg, "condition")),
                            "all_identical": True}
    data["external_loader"] = ext_stats.get("loader")
    data["benign_cleaning"] = ext_stats.get("benign_cleaning")
    data["cohort"] = str(cohort)
    data["cohort_seed"] = (
        int(fixed_cohort["cohort_seed"]) if fixed_cohort is not None else None
    )
    data["bias_diagnostics"] = bias
    data["baseline_seeds"] = [int(d["seed"]) for d in ok]
    data["match"] = str(match)
    data["path_match"] = first.get("path_match")
    gate_notes: dict[str, str] = {}
    common_bias = _apply_collection_gate(
        data, coll_gate, bool(bias.get("common_direction_bias", False))
    )
    shortcut = path_shortcut_flag(baselines)
    data["path_shortcut"] = shortcut
    gates = gates_from(baselines, data["split_diagnostics"].get("top5_test_group_frac"),
                       notes=gate_notes)
    data["gate_notes"] = gate_notes
    if tier == "dropped":
        gates["stratum_tier_ok"] = False

    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    res = _transfer_results(
        cfg, mode="a", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
        common_direction_bias=common_bias,
        path_shortcut_dominant=bool(shortcut["path_shortcut_dominant"]),
        match=str(match),
        extra={
            "cohort": str(cohort),
            "match": str(match),
            "null_row_permutation": null_row,
            "comparison_to_baselines": paired_deltas,
        },
    )
    if motif:
        res["motif_replication"] = _transfer_motif_replication(
            cfg, tag, stratum, first, ok, cohort=str(cohort)
        )
    return res


def _transfer_motif_replication(cfg, tag, stratum, first, ok, *, cohort="per_seed") -> dict:
    """설계 4절 — WebPhish에서 찾은 motif가 외부 train에서 다시 보이는가.

    motif 선택은 **WebPhish train에서만** 한다. 외부를 보고 고르면 순환 논증이다.
    """
    from qrphish.transfer import motif_replication

    enr_path = _phase_dir(cfg, "motifs", _motif_condition_id(cfg), stratum) / "enrichment.json"
    enr = _read_json_safe(enr_path)
    if not enr:
        return {"skipped": f"WebPhish enrichment가 없다: {enr_path} (run_motifs 먼저)"}
    try:
        df = first["frame"]
        tr = df[_split_mask(df, 0)].reset_index(drop=True)
        tr = _subsample(tr, TRANSFER_MOTIF_EVAL_CAP, int(first["seed"]))
        H = _motif_hists(cfg, {"train": tr}, {})["train"]["patch3"]
        rep = motif_replication(
            enr, H, tr["label"].to_numpy(dtype=np.int64),
            tr["group"].astype(str).to_numpy(), size=3, k=20, seed=0,
        )
        rep["n_rows"] = int(len(tr))
        rep["webphish_enrichment"] = str(enr_path)
        # cohort가 다르면 train 행 집합도 달라진다. 같은 경로에 쓰면 서로 덮어쓴다.
        sub = "motif_replication_fixed" if cohort == "fixed" else "motif_replication"
        d = _reports_dir(cfg) / "transfer" / tag / sub / stratum
        d.mkdir(parents=True, exist_ok=True)
        (d / "replication.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return rep
    except Exception as exc:
        print(f"[WARN] motif 재현성 {stratum}: {exc!r}")
        traceback.print_exc()
        return {"error": repr(exc)}


def _run_transfer_b(cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir, *,
                    n_perm, n_boot, entry, coll_gate=None) -> dict:
    """F-b — 외부 데이터 자체 학습·평가. ``_run_one_seed``를 그대로 재사용한다.

    전이가 실패했을 때 "외부에 신호가 없다"와 "전이가 안 된다"를 가르는 실행이다.
    """
    from qrphish.dataset import stratum_tier
    from qrphish.transfer import (
        gates_from,
        path_shortcut_flag,
        permutation_null,
        row_permutation_null,
        transfer_baselines,
    )

    per_seed_raw = []
    for s in seeds:
        try:
            per_seed_raw.append(
                _run_one_seed(cfg, stratum, int(s), ext_cond_dir, entry, frame=ext_all)
            )
        except Exception as exc:
            print(f"[ERROR] transfer F-b {stratum} seed{s}: {exc!r}")
            traceback.print_exc()
            per_seed_raw.append({"seed": s, "error": repr(exc)})
    ok = [d for d in per_seed_raw if "error" not in d]
    if not ok:
        raise RuntimeError(f"F-b {stratum}: 모든 시드 실패")
    # 층 재개용으로 아티팩트 쪽에도 results.json을 남긴다(F-c가 임계값을 여기서 읽는다).
    (ext_cond_dir / stratum).mkdir(parents=True, exist_ok=True)
    (ext_cond_dir / stratum / "results.json").write_text(
        json.dumps({"per_seed": per_seed_raw, "condition_id": cid, "stratum": stratum},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    per_seed = []
    for d in per_seed_raw:
        if "error" in d:
            per_seed.append(d)
            continue
        pr = _load_seed_preds_at(ext_cond_dir / stratum / f"seed{d['seed']}", int(d["seed"]))
        if pr is None:
            per_seed.append({"seed": d["seed"], "error": "preds_test.npz 없음"})
            continue
        pr["threshold"] = d.get("threshold")
        per_seed.append(pr)
    model = _transfer_model_block(per_seed, n_boot, "external_val")
    y, p, g, sd = _pooled_arrays(per_seed)
    null_perm = permutation_null(y, p, g, sd, n_perm=n_perm, seed=0)
    null_row = row_permutation_null(y, p, sd, n_perm=n_perm, seed=0)

    seed0 = int(ok[0]["seed"])
    meta = pd.read_parquet(ext_cond_dir / stratum / f"seed{seed0}" / "meta.parquet")
    tr = meta[_split_mask(meta, 0)].reset_index(drop=True)
    te = meta[_split_mask(meta, 2)].reset_index(drop=True)
    baselines = transfer_baselines(tr, te, seed0, fit_on="external_train")
    per_class_min = int(te["label"].value_counts().min()) if len(te) else 0
    data = _transfer_data_block(
        te, None, "test",
        stratum_tier(per_class_min,
                     secondary_min=int(_get(_get(cfg, "stratum_rules"), "secondary_min_per_class", 300))),
    )
    data["n_train"] = int(len(tr))
    gate_notes: dict[str, str] = {}
    # F-b는 외부에서 학습·평가한다. "두 데이터셋 공통 방향" 질문은 전이(F-a)의 것이므로
    # 플래그를 판정에 넣지 않고 기록만 한다. 경로 지름길은 평가 cohort 자체의 성질이라
    # 그대로 적용한다.
    _apply_collection_gate(data, coll_gate, False)
    shortcut = path_shortcut_flag(baselines)
    data["path_shortcut"] = shortcut
    gates = gates_from(baselines, None, notes=gate_notes)
    data["gate_notes"] = gate_notes
    wp_res = _webphish_results(cfg, cid, stratum)
    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    return _transfer_results(
        cfg, mode="b", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
        path_shortcut_dominant=bool(shortcut["path_shortcut_dominant"]),
        extra={"null_row_permutation": null_row,
               "per_seed_train": [{k: v for k, v in d.items() if k != "baselines"}
                                  for d in per_seed_raw]},
    )


def _load_seed_preds_at(seed_dir: Path, seed: int) -> dict | None:
    path = Path(seed_dir) / "preds_test.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {
            "seed": int(seed),
            "y": np.asarray(z["y"], dtype=np.int64),
            "p": np.asarray(z["p"], dtype=np.float64),
            "group": np.asarray(z["group"]).astype(str),
        }


def _run_transfer_c(cfg, cid, tag, stratum, seeds, ext_cond_dir, *, n_perm, n_boot,
                    coll_gate=None) -> dict:
    """F-c — 외부에서 학습한 모델로 WebPhish test를 평가한다(역방향 전이).

    F-a와 비대칭이면 어느 쪽 신호가 더 일반적인지 알려준다.
    ``ext_cond_dir``는 **F-b가 쓴** 디렉터리다(체크포인트가 거기 있다).
    """
    from qrphish.transfer import (
        gates_from,
        path_shortcut_flag,
        permutation_null,
        row_permutation_null,
        transfer_baselines,
    )

    per_seed = []
    for s in seeds:
        try:
            per_seed.append(_fc_one_seed(cfg, cid, stratum, int(s), ext_cond_dir))
        except Exception as exc:
            print(f"[ERROR] transfer F-c {stratum} seed{s}: {exc!r}")
            traceback.print_exc()
            per_seed.append({"seed": s, "error": repr(exc)})
    ok = [d for d in per_seed if "error" not in d]
    if not ok:
        raise RuntimeError(f"F-c {stratum}: 모든 시드 실패")
    model = _transfer_model_block(per_seed, n_boot, "external_val")
    y, p, g, sd = _pooled_arrays(per_seed)
    null_perm = permutation_null(y, p, g, sd, n_perm=n_perm, seed=0)
    null_row = row_permutation_null(y, p, sd, n_perm=n_perm, seed=0)

    seed0 = int(ok[0]["seed"])
    ext_meta = pd.read_parquet(ext_cond_dir / stratum / f"seed{seed0}" / "meta.parquet")
    ext_train = ext_meta[_split_mask(ext_meta, 0)].reset_index(drop=True)
    wp_test = ok[0]["eval_frame"]
    baselines = transfer_baselines(
        _subsample(ext_train, TRANSFER_FIT_CAP, seed0), wp_test, seed0, fit_on="external_train"
    )
    data = _transfer_data_block(wp_test, None, "test", "webphish_test")
    # 평가는 WebPhish지만 **학습**이 외부 데이터라, 수집 단계 게이트가 깨졌으면 F-c도
    # 결론에 쓰지 않는다.
    gate_notes: dict[str, str] = {}
    _apply_collection_gate(data, coll_gate, False)
    shortcut = path_shortcut_flag(baselines)
    data["path_shortcut"] = shortcut
    gates = gates_from(baselines, None, notes=gate_notes)
    data["gate_notes"] = gate_notes
    wp_res = _webphish_results(cfg, cid, stratum)
    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    return _transfer_results(
        cfg, mode="c", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
        path_shortcut_dominant=bool(shortcut["path_shortcut_dominant"]),
        extra={"null_row_permutation": null_row},
    )


# ------------------------------------------------------- G 재설계: 캠페인 홀드아웃
# review_02 3절. 고정 캠페인 test 집합 T 위에서 Model A(eTLD+1만 격리) 대 
# Model B(eTLD+1 + 템플릿 격리)를 **쌍체**로 비교한다. 옛 template_split 조건은
# 평가 대상의 88%가 함께 바뀌어 causal interpretation이 불가능했다(폐기).
# ``A_sm``은 A-sizematched(설계 5.5) — 형제는 전부 두고 비형제를 덜어 |train| = |train_B|로
# 맞춘 A다. 주 지표는 A_sm − B이고, 학습 표본 수 효과가 섞인 A − B는 보조로 함께 낸다.
CAMPAIGN_MODELS = ("A", "A_sm", "B")
# T에서 함께 재는 텍스트 기준선. 텍스트 기준선도 템플릿 누출을 타는지 보기 위해
# A/B train으로 각각 fit해 같은 T에서 평가한다(설계 5.5의 마지막 문단).
CAMPAIGN_BASELINES = ("charngram_lr", "bytehist_lr")
# 사전 등록 판정 경계(설계 6.4). 부호만 뒤집어 쓴다 — 여기서는 Δ = A − B가 양수일 때
# "템플릿 누출 이득"이므로, 하락폭 대신 이득 크기를 같은 0.05와 비교한다.
# SESOI(0.02)는 이것과 **다른 질문**의 임계다: 0.05는 "결론을 고쳐야 하는가",
# 0.02는 "무시해도 되는가". 정의는 :mod:`qrphish.campaign` 상단에 있다.
CAMPAIGN_EFFECT_THRESHOLD = CAMPAIGN_MAJOR_THRESHOLD
# 판정을 붙이는 태그. 주 지표(cnn_sm)와 보조 A−B, 그리고 두 누출 대조 부집합까지 모두
# 같은 규칙으로 판정한다(review_03 5절: "부집합에도 같은 규칙을 명시"). 텍스트 기준선
# 태그에는 붙이지 않는다 — 사전 등록 가설이 아니다.
CAMPAIGN_VERDICT_TAGS = (
    "cnn_sm", "cnn", "cnn_sm_leaky_vs_contrast", "cnn_leaky_vs_contrast",
)


def _campaign_pool_frame(cfg: Any, stratum: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """층 전체 프레임(길이 매칭·그룹 다운샘플 **이전**)을 :func:`_prepare_frame`로 얻는다.

    캠페인 홀드아웃은 test 성분 안에서 eTLD+1을 다시 가르므로 **행이 하나도 빠지지 않은**
    층 프레임이 필요하다. 그래서 ``length_match=none``(매칭은 행 제거다)과
    ``max_group_frac=1.0``(다운샘플도 행 제거다)로 눌러 둔 config로 부른다. 로드·정규화·
    경로 필터·용량 필터·층 필터는 1차 경로와 완전히 같은 코드를 지난다.
    """
    cfg_pool = apply_overrides(
        cfg,
        {
            "condition.length_match": "none",
            "condition.length_bucket": 1,
            "split.group_key": "etld1",
            "split.max_group_frac": 1.0,
        },
    )
    df, diag = _prepare_frame(cfg_pool, stratum, seed)
    return df.drop(columns=[c for c in ("split",) if c in df.columns]), diag


def _campaign_split(cfg: Any, stratum: str, seed: int, sibling_frac: float):
    """층 프레임 → 길이 매칭까지 끝난 :class:`qrphish.campaign.CampaignSplit`."""
    from qrphish.campaign import build_campaign_split, length_match_campaign

    pool, pdiag = _campaign_pool_frame(cfg, stratum, seed)
    scfg = _get(cfg, "split")
    cs = build_campaign_split(
        pool,
        seed=seed,
        ratios=tuple(_get(scfg, "ratios", (0.70, 0.15, 0.15))),
        max_group_frac=float(_get(scfg, "max_group_frac", 0.05)),
        sibling_frac=float(sibling_frac),
        template_params={
            "template_threshold": float(_get(scfg, "template_threshold", 0.7)),
            "template_shingle": int(_get(scfg, "template_shingle", 3)),
            "min_template_tokens": int(_get(scfg, "min_template_tokens", 3)),
        },
    )
    cs = length_match_campaign(cs, _length_bucket(_get(cfg, "condition")), seed)
    cs.diagnostics["pool"] = {
        "n_loaded": pdiag.get("n_loaded"),
        "n_in_stratum": pdiag.get("n_in_stratum"),
        "n_dropped_capacity": pdiag.get("n_dropped_capacity"),
    }
    return cs


def _campaign_union_frame(cs) -> pd.DataFrame:
    """A train ∪ B train, val, T를 한 프레임으로 합친다.

    격자를 **한 번만** 만들기 위해서다. 그래야 (1) ``n_max``가 A와 B에서 같고,
    (2) T의 격자가 두 모델에서 바이트 단위로 동일하다. 길이 매칭을 A/B 각각 걸면
    ``train_b``가 ``train_a``의 부분집합이 아닐 수 있으므로 합집합을 쓴다.
    """
    train = (
        pd.concat([cs.train_a, cs.train_b], ignore_index=True)
        .drop_duplicates(subset="url", keep="first")
        .assign(split="train")
    )
    return pd.concat(
        [train, cs.val.assign(split="val"), cs.test.assign(split="test")],
        ignore_index=True,
    ).reset_index(drop=True)


def _campaign_rows(meta: pd.DataFrame, frame: pd.DataFrame, label: str) -> np.ndarray:
    """``meta``(build_stratum 산출) 안에서 ``frame``의 행 위치. 하나라도 없으면 실패."""
    pos = {u: i for i, u in enumerate(meta["url"].astype(str).tolist())}
    want = frame["url"].astype(str).tolist()
    missing = [u for u in want if u not in pos]
    if missing:
        raise ValueError(f"{label}: build_stratum이 {len(missing)}행을 떨어뜨렸다 (예: {missing[0]!r})")
    return np.asarray([pos[u] for u in want], dtype=np.int64)


def _campaign_loader(ds: QRGridDataset, idx: np.ndarray, batch_size: int, shuffle: bool):
    sub = copy.copy(ds)
    sub.indices = np.asarray(idx, dtype=np.int64)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _campaign_train_one(
    cfg: Any,
    ds: QRGridDataset,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    out_dir: Path,
):
    """모델 하나(A 또는 B)를 학습하고 T 예측을 낸다. ``train_model``을 그대로 쓴다."""
    mcfg = _get(cfg, "model")
    bs = int(_get(mcfg, "batch_size", 256))
    set_seed(seed)
    model = build_model(str(_get(mcfg, "arch", "small_cnn")), ds.n, ds.canonical_data_mask)
    device = pick_device()
    tres = train_model(
        model,
        _campaign_loader(ds, idx_tr, bs, True),
        _campaign_loader(ds, idx_va, bs, False),
        lr=float(_get(mcfg, "lr", 1e-3)),
        weight_decay=float(_get(mcfg, "weight_decay", 1e-4)),
        max_epochs=int(_get(mcfg, "max_epochs", 60)),
        patience=int(_get(mcfg, "patience", 8)),
        class_weight=_get(mcfg, "class_weight", "balanced"),
        amp=bool(_get(mcfg, "amp", True)),
        seed=seed,
        device=device,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n": ds.n,
            "arch": str(_get(mcfg, "arch")),
            "condition": _cfg_snapshot(_get(cfg, "condition")),
            "qr": _cfg_snapshot(_get(cfg, "qr")),
        },
        out_dir / "model.pt",
    )
    y_te, p_te = predict_probs(model, _campaign_loader(ds, idx_te, bs, False), device=device)
    return tres, np.asarray(y_te), np.asarray(p_te, dtype=np.float64)


def _campaign_baselines(
    train_urls, train_y, val_urls, val_y, test_urls, test_y, groups_te, seed: int, n_boot: int
) -> dict:
    """char n-gram LR / byte-hist LR을 그 모델의 train으로 fit해 같은 T에서 평가한다."""
    out: dict[str, Any] = {}
    fitters = {
        "charngram_lr": (bl.fit_charngram_lr, bl.apply_charngram_lr),
        "bytehist_lr": (bl.fit_bytehist_lr, bl.apply_bytehist_lr),
    }
    for name in CAMPAIGN_BASELINES:
        fit, apply = fitters[name]
        try:
            m = fit(train_urls, train_y, seed=seed)
            thr = pick_threshold(np.asarray(val_y), apply(m, val_urls))
            p_te = apply(m, test_urls)
            out[name] = metrics_from_probs(
                np.asarray(test_y), p_te, groups_te, n_boot=n_boot, seed=seed, threshold=thr
            )
            out[name]["p_test"] = p_te
        except Exception as exc:  # 기준선 하나가 터져도 CNN 결과는 살린다
            out[name] = {"error": repr(exc)}
    return out


def _campaign_one_seed(
    cfg: Any, stratum: str, seed: int, out_dir: Path, *, sibling_frac: float
) -> dict:
    """한 (층, 시드): 분할 고정 → A·B 학습 → 같은 T에서 평가."""
    cond, qr, ecfg = _get(cfg, "condition"), _get(cfg, "qr"), _get(cfg, "eval")
    n_boot = int(_get(ecfg, "n_bootstrap", 2000))

    cs = _campaign_split(cfg, stratum, seed, sibling_frac)
    if not len(cs.test) or not len(cs.train_b) or not len(cs.val):
        return {"seed": int(seed), "error": "빈 T/val/train (분할 실패)", "split": cs.diagnostics}

    seed_dir = out_dir / stratum / f"seed{seed}"
    frame = _campaign_union_frame(cs)
    sm = build_stratum(
        frame,
        stratum,
        cond,
        seed_dir,
        qr=qr,
        condition_id="campaign",
        rules=_get(cfg, "stratum_rules"),
        git_sha=git_sha(),
        length_match=cs.diagnostics.get("length_match"),
    )
    arr = load_stratum_arrays(seed_dir)
    meta = pd.read_parquet(seed_dir / "meta.parquet")
    ds = QRGridDataset(
        arr["X_packed"],
        arr["y"],
        int(arr["n"]),
        arr["versions"],
        int(arr["ec"]),
        features=str(_get(cond, "features", "data_only")),
        mask_mode=str(_get(qr, "mask_mode", "fixed")),
        shuffle_positions=bool(_get(cond, "shuffle_positions", False)),
        perm_seed=seed,
    )
    from qrphish.campaign import sizematched_train_a

    train_a_sm = sizematched_train_a(cs, seed)
    idx = {
        "A": _campaign_rows(meta, cs.train_a, "train_a"),
        "A_sm": _campaign_rows(meta, train_a_sm, "train_a_sm"),
        "B": _campaign_rows(meta, cs.train_b, "train_b"),
    }
    idx_va = _campaign_rows(meta, cs.val, "val")
    idx_te = _campaign_rows(meta, cs.test, "test")
    # npz는 allow_pickle=False로 다시 읽으므로 object dtype이 남으면 안 된다.
    groups_te = meta["group"].astype(str).to_numpy().astype("U")[idx_te]
    urls = meta["url"].astype(str).to_numpy().astype("U")
    y_all = np.asarray(ds.y, dtype=np.int64)

    # T 행 중 A의 train에 같은 캠페인 형제가 실제로 있는 행. 전체 T에서는 개입이
    # 크게 희석되므로(실측 v2에서 2.6%), 이 부분집합 위의 Δ를 부차 지표로 함께 낸다.
    leaky = _campaign_leaky_rows(cs)
    entry: dict[str, Any] = {
        "seed": int(seed),
        "n_test_leaky": int(leaky.sum()),
        "split": cs.diagnostics,
        "stratum_tier": sm.tier,
        "n_test": int(len(idx_te)),
        "models": {},
    }
    preds: dict[str, np.ndarray] = {}
    for name in CAMPAIGN_MODELS:
        mdir = seed_dir / name
        tres, y_te, p_te = _campaign_train_one(
            cfg, ds, idx[name], idx_va, idx_te, seed, mdir
        )
        base = _campaign_baselines(
            urls[idx[name]], y_all[idx[name]],
            urls[idx_va], y_all[idx_va],
            urls[idx_te], y_all[idx_te],
            groups_te, seed, min(n_boot, 500),
        )
        arrays: dict[str, np.ndarray] = {
            "y": np.asarray(y_te, dtype=np.int64),
            "p": p_te,
            "group": groups_te,
            "url": urls[idx_te],
            "leaky": leaky,
            "seed": np.full(len(idx_te), int(seed), dtype=np.int64),
        }
        for b in CAMPAIGN_BASELINES:
            # 예측 벡터는 npz로만 내보내고 results.json에는 남기지 않는다.
            if "p_test" in base.get(b, {}):
                arrays[f"p_{b}"] = np.asarray(base[b].pop("p_test"), dtype=np.float64)
        np.savez_compressed(mdir / "preds_test.npz", **arrays)  # type: ignore[arg-type]
        preds[name] = p_te
        test_m = metrics_from_probs(
            np.asarray(y_te), p_te, groups_te, n_boot=n_boot, seed=seed, threshold=tres.threshold
        )
        entry["models"][name] = {
            "n_train": int(len(idx[name])),
            "best_epoch": tres.best_epoch,
            "threshold": tres.threshold,
            "val": {"auroc": tres.best_val_auroc},
            "test": {k: test_m[k] for k in ("auroc", "auprc", "f1", "acc", "auroc_ci", "f1_ci")},
            "baselines": base,
            "model": f"seed{seed}/{name}/model.pt",
            "preds_test": f"seed{seed}/{name}/preds_test.npz",
        }
    entry["delta_auroc_seed"] = float(
        entry["models"]["A_sm"]["test"]["auroc"] - entry["models"]["B"]["test"]["auroc"]
    )
    entry["delta_auroc_seed_unmatched"] = float(
        entry["models"]["A"]["test"]["auroc"] - entry["models"]["B"]["test"]["auroc"]
    )
    entry["n_train_sizematch_gap"] = int(len(idx["A_sm"]) - len(idx["B"]))
    # 모든 모델이 정말 같은 행을 평가했는지 확인한다(쌍체 비교의 전제).
    entry["paired_ok"] = bool(
        preds["A"].shape == preds["B"].shape == preds["A_sm"].shape
    )
    return entry


def _campaign_leaky_rows(cs) -> np.ndarray:
    """T 행별로 "A의 train에 같은 템플릿 클러스터 행이 있는가" bool 배열."""
    from qrphish.campaign import SOLO_PREFIX

    tpl = cs.train_a["template_id"].astype(str)
    pool = set(tpl[~tpl.str.startswith(SOLO_PREFIX)])
    t = cs.test["template_id"].astype(str)
    return (t.isin(pool) & ~t.str.startswith(SOLO_PREFIX)).to_numpy(dtype=bool)


def _campaign_pooled(
    per_seed: list[dict],
    out_dir: Path,
    stratum: str,
    key: str,
    *,
    leaky_only: bool = False,
    model_a: str = "A_sm",
    model_b: str = "B",
):
    """시드별 ``preds_test.npz``를 두 모델 정렬 상태로 이어 붙인다.

    ``model_a``/``model_b``는 :data:`CAMPAIGN_MODELS`의 이름이다(기본은 주 지표인
    ``A_sm`` 대 ``B``).

    ``leaky_only``면 개입이 닿을 수 있는 T 행만 남긴다: A의 train에 같은 캠페인 형제가
    있는 **누출 행** + **누출 행이 하나도 없는 클래스의 행 전부(대조)**. 형제 행은 거의
    전부 피싱이라 누출 행만 남기면 단일 클래스가 되어 AUROC가 NaN이 된다 — 대조군으로
    반대 클래스를 통째로 남겨야 지표가 정의된다. 그래서 이 지표의 이름은
    ``*_leaky_vs_contrast``이지 "누출 부분집합"이 아니다. 두 집단의 크기·구성이 다르므로
    같은 층의 전체 T Δ와 직접 크기를 비교하면 안 된다.
    """
    ys, pa, pb, gs, ss = [], [], [], [], []
    for row in per_seed:
        if "error" in row:
            continue
        seed = int(row["seed"])
        sdir = out_dir / stratum / f"seed{seed}"
        try:
            with np.load(sdir / model_a / "preds_test.npz", allow_pickle=False) as za, \
                 np.load(sdir / model_b / "preds_test.npz", allow_pickle=False) as zb:
                if not np.array_equal(za["url"].astype(str), zb["url"].astype(str)):
                    raise ValueError(
                        f"{model_a}/{model_b}의 test 행이 다르다 — 쌍체 비교 불가"
                    )
                m = np.ones(za["y"].size, dtype=bool)
                if leaky_only:
                    lk, yy = za["leaky"].astype(bool), za["y"]
                    m = lk.copy()
                    for c in (0, 1):
                        if not (lk & (yy == c)).any():
                            m |= yy == c
                if not m.any():
                    continue
                ys.append(za["y"][m])
                pa.append(za[key][m])
                pb.append(zb[key][m])
                gs.append(za["group"].astype(str)[m])
                ss.append(za["seed"][m])
        except (FileNotFoundError, KeyError):
            continue
    if not ys:
        return None
    return {
        "y": np.concatenate(ys), "p_a": np.concatenate(pa), "p_b": np.concatenate(pb),
        "group": np.concatenate(gs), "seed": np.concatenate(ss),
    }


def _campaign_verdict(lo: float, hi: float, point: float) -> dict[str, Any]:
    """사전 등록 판정 — :func:`qrphish.campaign.campaign_verdict`의 얇은 래퍼.

    옛 규칙(``lo <= 0`` → "누출 무시 가능")은 CI ``[-0.10, +0.30]``에도 같은 판정을 내려
    "검출하지 못했다"와 "차이가 없다"를 섞었다(review_03 5절). 새 규칙은 SESOI
    (:data:`CAMPAIGN_SESOI`)로 둘을 가른다. 판정 코드와 정의는 campaign 모듈에 있다.
    """
    return campaign_verdict(float(lo), float(hi), float(point))


def _campaign_leaky_sample_warning(per_seed: list[dict]) -> dict[str, Any]:
    """누출 대조 부집합의 표본 수 경고 블록.

    ``*_leaky_vs_contrast``의 Δ는 "A train에 형제가 있는 T 행"이 몇 개냐에 검정력이
    통째로 걸려 있다. 실측은 층당 평균 15 ~ 38행뿐이라 CI가 넓고, 그 CI로는
    ``negligible``이 나올 수 없다. 그 사실을 결과 JSON에 명시적으로 남긴다.
    """
    vals = [
        int(r["n_test_leaky"])
        for r in per_seed
        if "error" not in r and r.get("n_test_leaky") is not None
    ]
    if not vals:
        return {"n_leaky_rows_per_seed": [], "warning": "누출 행 수를 알 수 없다"}
    mean = float(np.mean(vals))
    return {
        "n_leaky_rows_per_seed": vals,
        "n_leaky_rows_mean": mean,
        "warning": (
            f"누출 기회가 있는 T 행은 시드당 평균 {mean:.0f}개뿐이다(시드별 {vals}). "
            "이 부집합의 CI는 구조적으로 넓어 등가성(negligible) 판정이 사실상 불가능하며, "
            "'phishing kit 암기 가능성을 배제했다'는 결론의 근거로 쓸 수 없다. "
            "표본 구성이 다르므로 전체 T의 Δ와 크기를 직접 비교해서도 안 된다."
        ),
    }


def _campaign_aggregate(cfg, per_seed, out_dir, stratum) -> dict:
    """A_sm−B(주) · A−B(보조) 쌍체 CI(시드 층화 클러스터 부트스트랩)와 클러스터 순열 p값."""
    ecfg = _get(cfg, "eval")
    n_boot = int(_get(ecfg, "n_bootstrap", 2000))
    alpha = float(_get(ecfg, "alpha", 0.05))
    n_perm = int(_get(ecfg, "n_permutation", 2000) or 2000)
    ok = [r for r in per_seed if "error" not in r]
    agg: dict[str, Any] = {
        "pooling": POOLING_MODE,
        "n_seeds_ok": len(ok),
        "auroc_mean": {
            m: (float(np.mean([r["models"][m]["test"]["auroc"] for r in ok])) if ok else float("nan"))
            for m in CAMPAIGN_MODELS
        },
    }
    # (npz 키, 태그, leaky_only, model_a). 주 지표는 A_sm − B다(설계 5.5).
    specs = [
        ("p", "cnn_sm", False, "A_sm"),
        ("p", "cnn_sm_leaky_vs_contrast", True, "A_sm"),
        ("p", "cnn", False, "A"),
        ("p", "cnn_leaky_vs_contrast", True, "A"),
    ]
    specs += [(f"p_{b}", f"{b}_sm", False, "A_sm") for b in CAMPAIGN_BASELINES]
    specs += [(f"p_{b}", b, False, "A") for b in CAMPAIGN_BASELINES]
    for key, tag, leaky_only, model_a in specs:
        pooled = _campaign_pooled(
            per_seed, out_dir, stratum, key, leaky_only=leaky_only, model_a=model_a
        )
        if pooled is None:
            continue
        pb_ = paired_cluster_bootstrap_by_seed(
            pooled["y"], pooled["p_a"], pooled["p_b"], pooled["group"], pooled["seed"],
            auroc_fn, n_boot=n_boot, seed=0, alpha=alpha,
        )
        perm = paired_cluster_permutation_test(
            pooled["y"], pooled["p_a"], pooled["p_b"], pooled["group"],
            n_perm=n_perm, seeds=pooled["seed"], alternative="two-sided", seed=0,
        )
        block = {
            "delta_auroc": float(pb_["point"]),
            "delta_ci": [float(pb_["lo"]), float(pb_["hi"])],
            "auroc_A": float(pb_["point_a"]),
            "auroc_B": float(pb_["point_b"]),
            "per_seed_delta": [float(v) for v in pb_["per_seed"]],
            "paired": True,
            "pooling": POOLING_MODE,
            "perm_p": float(perm["perm_p"]),
            "n_perm": int(perm["n_perm"]),
            "n_rows": int(pooled["y"].size),
        }
        block["model_a"] = model_a
        block["model_b"] = "B"
        if tag in CAMPAIGN_VERDICT_TAGS:
            # 사전 등록 판정. 주 지표는 여전히 cnn_sm(A_sm − B)이지만, 누출 대조
            # 부집합에도 **같은 규칙**을 적용해 판정을 명시한다(review_03 5절).
            block["verdict"] = _campaign_verdict(pb_["lo"], pb_["hi"], pb_["point"])
            block["primary"] = tag == "cnn_sm"
            if leaky_only:
                block["sample_size_warning"] = _campaign_leaky_sample_warning(per_seed)
        agg[tag] = block
    agg["verdict_rule"] = {
        "sesoi": CAMPAIGN_SESOI,
        "major_threshold": CAMPAIGN_EFFECT_THRESHOLD,
        "primary_tag": "cnn_sm",
        "definitions": dict(CAMPAIGN_VERDICT_DEFINITIONS),
    }
    return agg


def run_campaign_holdout(cfg: Any, strata: list[str] | None = None) -> dict:
    """G 재설계: 고정 캠페인 test 집합 위의 쌍체 A/B 비교 (review_02 3절).

    시드마다 (1) 템플릿∪eTLD+1 성분으로 분할해 test 성분을 잡고, (2) 그 성분 안에서
    eTLD+1을 갈라 고정 test 집합 T와 형제(같은 캠페인·다른 도메인) 행을 나눈 뒤,
    (3) Model B(완전 격리)와 Model A(형제 허용)를 학습해 **같은 T**에서 평가한다.
    ``ΔAUROC = A − B``의 시드 층화 쌍체 클러스터 부트스트랩 CI와 클러스터 순열 p값이
    주 산출물이다. 자세한 구성 근거는 :mod:`qrphish.campaign` 도크스트링에 있다.

    Args:
        cfg: 주 조건 config. ``split.group_key``는 여기서 무시된다(항상 합집합 키를 쓴다).
        strata: 대상 층. 기본은 P0에서 살아남은 층.

    Returns:
        층별 결과 dict. 파일은 ``reports/campaign/{stratum}/results.json``과
        ``artifacts/campaign/{stratum}/seed{k}/{A,B}/model.pt``.
    """
    sibling_frac = float(_get(_get(cfg, "split"), "campaign_sibling_frac", 0.5) or 0.5)
    out_dir = _out_root(cfg) / "campaign"
    rep_root = _reports_dir(cfg) / "campaign"
    seeds = [int(s) for s in _get(cfg, "seed_list", [0])]
    targets = list(strata) if strata else _surviving_strata(cfg)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "condition_id": condition_id(cfg, "campaign"),
        "sibling_frac": sibling_frac,
        "seeds": seeds,
        "strata": {},
    }
    for stratum in targets:
        rpath = rep_root / stratum / "results.json"
        done = _completed_result(rpath, f"campaign/{stratum}")
        if done is not None:
            print(f"[skip] campaign/{stratum} (results.json 존재)", flush=True)
            summary["strata"][stratum] = done
            continue
        print(f"[run ] campaign/{stratum}", flush=True)
        per_seed: list[dict] = []
        for s in seeds:
            try:
                per_seed.append(
                    _campaign_one_seed(cfg, stratum, s, out_dir, sibling_frac=sibling_frac)
                )
                d = per_seed[-1].get("delta_auroc_seed")
                print(f"  [seed{s}] ΔAUROC(A_sm−B) = {d if d is None else round(d, 4)}", flush=True)
            except Exception as exc:
                print(f"[ERROR] campaign/{stratum} seed{s}: {exc!r}", flush=True)
                traceback.print_exc()
                per_seed.append({"seed": int(s), "error": repr(exc)})
        try:
            agg = _campaign_aggregate(cfg, per_seed, out_dir, stratum)
        except Exception as exc:
            print(f"[ERROR] campaign/{stratum} 집계 실패: {exc!r}", flush=True)
            traceback.print_exc()
            agg = {"error": repr(exc)}
        res: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "phase": "campaign",
            "condition_id": condition_id(cfg, "campaign"),
            "stratum": stratum,
            "sibling_frac": sibling_frac,
            "config": _cfg_snapshot(cfg),
            "provenance": {
                "git_sha": git_sha(),
                "qrphish_version": _pkg_version_safe(),
                "torch": torch.__version__,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "per_seed": per_seed,
            "aggregate": agg,
        }
        if all("error" in r for r in per_seed):
            res["error"] = "모든 시드 실패"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        summary["strata"][stratum] = res
    rep_root.mkdir(parents=True, exist_ok=True)
    (rep_root / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def recompute_campaign_verdicts(
    cfg: Any, strata: list[str] | None = None
) -> dict[str, Any]:
    """저장된 캠페인 결과에 **새 판정 규칙만** 다시 적용한다 (재학습 없음).

    ``run_campaign_holdout``은 ``results.json``이 있으면 층을 통째로 건너뛰므로, 판정
    규칙이 바뀌어도 재실행으로는 새 verdict가 나오지 않는다. 이 함수는 이미 저장된
    ``delta_auroc``·``delta_ci``만 읽어 :func:`_campaign_verdict`를 다시 돌리고, 옛 판정과
    새 판정을 나란히 돌려준다. 부트스트랩·순열은 다시 하지 않는다(값이 그대로이므로
    판정만 바뀐다).

    Args:
        cfg: 결과 경로만 쓴다(``reports_dir``).
        strata: 대상 층. 기본은 ``reports/campaign`` 아래의 모든 층.

    Returns:
        ``{"strata": {stratum: {tag: {"old", "new", "delta_auroc", "delta_ci"}}},
        "sesoi": ..., "updated": [경로]}``
    """
    rep_root = _reports_dir(cfg) / "campaign"
    want = set(strata) if strata else None
    changes: dict[str, Any] = {}
    updated: list[str] = []
    for rpath in sorted(rep_root.glob("*/results.json")):
        stratum = rpath.parent.name
        if want and stratum not in want:
            continue
        data = json.loads(rpath.read_text(encoding="utf-8"))
        agg = data.get("aggregate")
        if not isinstance(agg, dict) or "error" in agg:
            continue
        per_seed = [r for r in data.get("per_seed", []) if isinstance(r, dict)]
        rows: dict[str, Any] = {}
        for tag in CAMPAIGN_VERDICT_TAGS:
            block = agg.get(tag)
            if not isinstance(block, dict) or "delta_ci" not in block:
                continue
            lo, hi = (float(v) for v in block["delta_ci"][:2])
            point = float(block["delta_auroc"])
            old = block.get("verdict")
            old_code = old.get("code") if isinstance(old, dict) else old
            new = _campaign_verdict(lo, hi, point)
            block["verdict"] = new
            block["primary"] = tag == "cnn_sm"
            if tag.endswith("leaky_vs_contrast"):
                block["sample_size_warning"] = _campaign_leaky_sample_warning(per_seed)
            rows[tag] = {
                "old": old_code,
                "new": new["code"],
                "delta_auroc": point,
                "delta_ci": [lo, hi],
            }
        if not rows:
            continue
        agg["verdict_rule"] = {
            "sesoi": CAMPAIGN_SESOI,
            "major_threshold": CAMPAIGN_EFFECT_THRESHOLD,
            "primary_tag": "cnn_sm",
            "definitions": dict(CAMPAIGN_VERDICT_DEFINITIONS),
        }
        rpath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        updated.append(str(rpath))
        changes[stratum] = rows
        for tag, row in rows.items():
            print(
                f"[verdict] campaign/{stratum} {tag}: {row['old']} -> {row['new']} "
                f"(Δ={row['delta_auroc']:+.4f} CI=[{row['delta_ci'][0]:+.4f}, "
                f"{row['delta_ci'][1]:+.4f}])",
                flush=True,
            )

    # 층 요약(reports/campaign/results.json)도 같은 층 파일에서 다시 만든다.
    summary_path = rep_root / "results.json"
    if summary_path.exists() and changes:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for stratum in list(summary.get("strata", {})):
            sp = rep_root / stratum / "results.json"
            if sp.exists():
                summary["strata"][stratum] = json.loads(sp.read_text(encoding="utf-8"))
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        updated.append(str(summary_path))
    return {
        "sesoi": CAMPAIGN_SESOI,
        "major_threshold": CAMPAIGN_EFFECT_THRESHOLD,
        "definitions": dict(CAMPAIGN_VERDICT_DEFINITIONS),
        "strata": changes,
        "updated": updated,
    }


# 저장된 프로브 results.json에서 세 질문을 다시 만들 때, **정확히** 복원되는 것과
# 아예 복원할 수 없는 것을 구분해 적는다. 근사(시드 평균 CI) 경로는 두지 않는다 —
# 평균 CI로는 "전 시드에서 성립"을 판정할 수 없는데도 그 값이 정확한 집계인 양
# 표와 본문에 실렸던 것이 review_04가 지적한 오류다.
PROBE_RECOMPUTE_BASIS = {
    "learned_gain": "exact:above_random_init (전 시드 AND가 그대로 저장돼 있다)",
    "accessible_and_gained": "exact:significant (전 시드 AND가 그대로 저장돼 있다)",
    "accessible": (
        "exact:per_seed — seed{k}.json 캐시 또는 targets[*].per_seed가 있을 때만 "
        "'전 시드에서 CI 하한 > 셔플 CI 상한'을 판정한다. 둘 다 없으면 미집계(None)로 "
        "남기고 재실행이 필요하다고 표시한다. 시드 평균 CI로 근사하지 않는다."
    ),
}
# 정확 재계산에 필요한데 옛 results.json에는 없는 필드.
PROBE_MISSING_FIELDS = (
    "targets[*].per_seed[*].ci / shuffle_ci — 목표별·시드별 CI. "
    "reports/probes/{cid}/{stratum}/seed{k}.json 캐시가 남아 있으면 재학습 없이 정확 "
    "재집계가 가능하지만, 커밋된 결과에는 층 results.json만 있고 그 안에도 시드별 CI가 "
    "없다. 이 경우 저장된 CNN 체크포인트에서 프로브만 다시 돌려야 한다(노트북 [11]).",
)
PROBE_RERUN_NOTE = (
    "accessible(질문 ①)은 미집계다. 저장된 자료에 목표별·시드별 CI가 없어 전 시드 판정을 "
    "복원할 수 없다. 옛 표는 이 칸에 accessible_and_gained(옛 n_significant)를 대신 "
    "채웠는데, 그건 서로 다른 질문의 답이다."
)


def _probe_seed_blocks(sdir: Path, data: dict) -> tuple[list[dict], str] | None:
    """정확 재집계에 쓸 시드별 블록을 찾는다. 없으면 ``None``.

    두 출처를 본다. (1) ``seed{k}.json`` 캐시 — ``_probe_one_seed``의 원출력이라 목표별
    시드 CI가 모두 들어 있다. (2) 새 스키마의 ``targets[*].per_seed`` — 층 results.json
    안에 시드별 점수·CI·판정을 되심은 블록. 근사 경로는 없다.
    """
    caches = []
    for cpath in sorted(sdir.glob("seed*.json")):
        try:
            blk = json.loads(cpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(blk, dict) and isinstance(blk.get("targets"), dict):
            caches.append(blk)
    if caches:
        return caches, "exact:seed_cache"

    targets = data.get("targets") or {}
    by_seed: dict[int, dict[str, dict]] = {}
    found = False
    for name, row in targets.items():
        if not isinstance(row, dict):
            continue
        for rec in row.get("per_seed") or []:
            if not isinstance(rec, dict) or "ci" not in rec:
                continue
            found = True
            by_seed.setdefault(int(rec["seed"]), {})[name] = {
                "kind": row.get("kind"),
                "family": row.get("family"),
                **{k: v for k, v in rec.items() if k != "seed"},
                "above_random_init": bool(rec.get("learned_gain")),
                "significant": bool(rec.get("accessible") and rec.get("learned_gain")),
            }
    if found:
        blocks = [{"seed": sd, "targets": t} for sd, t in sorted(by_seed.items())]
        return blocks, "exact:results_per_seed"
    return None


def recompute_probe_summaries(
    cfg: Any, strata: list[str] | None = None, condition_id_: str | None = None
) -> dict[str, Any]:
    """저장된 프로브 결과에 **세 질문 분리 보고**를 적용한다 (재학습 없음).

    ``run_probes``는 ``results.json``이 있으면 층을 건너뛰므로, 보고 규약이 바뀌어도
    재실행으로는 새 요약이 나오지 않는다. 이 함수는 저장된 결과만 읽어
    ``accessible`` / ``learned_gain`` / ``accessible_and_gained`` / ``used_in_decision``을
    채우고 요약을 다시 만든다.

    **정확 재집계만 한다.** ``learned_gain``과 ``accessible_and_gained``는 저장된 전 시드
    AND(``above_random_init`` / ``significant``)에서 정확히 복원된다. ``accessible``은
    ``seed{k}.json`` 캐시나 새 스키마의 ``targets[*].per_seed``가 있을 때만 전 시드로
    판정하고, 없으면 ``None``(미집계)으로 남긴 뒤 그 층을 ``needs_rerun``에 넣는다.
    시드 평균 CI로 근사하던 옛 경로는 제거했다 — 평균 CI는 전 시드 일치 여부를 담지
    못하는데도 정확한 집계처럼 보고돼 결론을 틀리게 만들었다(review_04).

    Returns:
        ``{"strata": {stratum: {요약}}, "basis": ..., "needs_rerun": [...],
        "missing_fields": [...], "updated": [경로]}``
    """
    cid = condition_id_ or MAIN_CONDITION
    root = _reports_dir(cfg) / "probes" / cid
    want = set(strata) if strata else None
    per_stratum: dict[str, Any] = {}
    updated: list[str] = []
    needs_rerun: list[str] = []
    for rpath in sorted(root.glob("*/results.json")):
        stratum = rpath.parent.name
        if want and stratum not in want:
            continue
        data = json.loads(rpath.read_text(encoding="utf-8"))
        targets = data.get("targets")
        if not isinstance(targets, dict):
            continue

        exact = _probe_seed_blocks(rpath.parent, data)
        if exact is not None:
            blocks, basis = exact
            n_seeds = len(data.get("seeds") or blocks)
            targets = _probe_aggregate(blocks, n_seeds)
            for row in targets.values():
                if isinstance(row, dict) and "ci" in row:
                    row["accessible_basis"] = basis
            data["targets"] = targets
        else:
            basis = "unrecomputable:no_per_seed_ci"
            needs_rerun.append(stratum)
            for row in targets.values():
                if not isinstance(row, dict) or "ci" not in row:
                    continue
                row["accessible"] = None
                row["accessible_basis"] = basis
                row["learned_gain"] = bool(row.get("above_random_init"))
                row["accessible_and_gained"] = bool(row.get("significant"))
                row["used_in_decision"] = None

        lex = {
            k: v
            for k, v in targets.items()
            if isinstance(v, dict) and v.get("family") not in (None, "label")
        }
        summary = _probe_summary(lex)
        summary["label_probe"] = targets.get("label:phishing")
        summary["recompute_basis"] = dict(PROBE_RECOMPUTE_BASIS)
        summary["accessible_basis"] = basis
        if exact is None:
            summary["needs_rerun"] = PROBE_RERUN_NOTE
            summary["missing_fields"] = list(PROBE_MISSING_FIELDS)
        data["summary"] = summary
        data["notes"] = list(PROBE_NOTES)
        rpath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        updated.append(str(rpath))
        per_stratum[stratum] = {q: summary[q] for q in PROBE_QUESTIONS} | {
            "n_targets": summary["n_targets"],
            "accessible_only": (
                summary["accessible_only"]["n"]
                if isinstance(summary["accessible_only"], dict)
                else None
            ),
            "accessible_basis": basis,
            "used_in_decision": None,
        }
        print(
            f"[probes] {stratum} 재요약({basis}) — "
            f"accessible {_probe_n(summary, 'accessible')}/{summary['n_targets']}, "
            f"learned_gain {_probe_n(summary, 'learned_gain')}/{summary['n_targets']}, "
            f"둘 다 {_probe_n(summary, 'accessible_and_gained')}/{summary['n_targets']}",
            flush=True,
        )
    return {
        "condition_id": cid,
        "strata": per_stratum,
        "basis": dict(PROBE_RECOMPUTE_BASIS),
        "needs_rerun": needs_rerun,
        "missing_fields": list(PROBE_MISSING_FIELDS) if needs_rerun else [],
        "updated": updated,
    }
