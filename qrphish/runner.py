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
    "run_explain",
    "run_probes",
    "run_occlusion",
    "compare_conditions",
    "run_hypothesis_tests",
    "aggregate_reports",
    "reaggregate",
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
            if rpath.exists():
                print(f"[skip] {cid}/{stratum} (results.json 존재)")
                results.append(json.loads(rpath.read_text(encoding="utf-8")))
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
        if done_path.exists():
            print(f"[skip] explain {st} (explain.json 존재)", flush=True)
            out[st] = json.loads(done_path.read_text(encoding="utf-8"))
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
        "p_value": bootstrap_p_value(r["samples"], alternative=alternative),
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
                            "p_value": cmp["p_value"],
                            "paired": cmp["paired"],
                            "pooling": cmp["pooling"],
                            "caveat": cmp["caveat"],
                        }
                    )
                else:
                    rec.update(_reference_gap_test(cfg, ids[h["a"]], st, n_boot, alpha, seeds))
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
    runnable = [
        t
        for t in tests
        if "p_value" in t
        and not t.get("descriptive_only")
        and not np.isnan(t.get("p_value", float("nan")))
    ]
    adj = holm([t["p_value"] for t in runnable])
    for t, a in zip(runnable, adj, strict=True):
        t["p_holm"] = float(a)
        t["reject"] = bool(a < alpha)
    for t in tests:
        t.setdefault("p_holm", None)
        t.setdefault("reject", None)

    out = {
        "alpha": alpha,
        "n_boot": n_boot,
        "n_tests_in_family": len(runnable),
        "correction": "holm",
        "pooling": POOLING_MODE,
        "decision_rule": (
            "판정은 쌍체 ΔAUROC의 95% 양측 백분위 CI가 0을 배제하는지로 한다. "
            "p값은 그 CI를 역전시켜 정의했고(가장 작은 alpha에서 CI가 0을 배제), "
            "Holm 보정은 이 p값에 건다. descriptive_only 항목은 family에서 제외한다."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "tests": tests,
    }
    path = _reports_dir(cfg) / "hypotheses.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out["path"] = str(path)
    return out


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
            "p_value": bootstrap_p_value(r["samples"], alternative="greater"),
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
        "p_value": float("nan"),
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
MOTIF_REPRESENTATIONS = (("patch2", 2, False), ("patch3", 3, False), ("pyramid3", 3, True))
# enrichment CI 부트스트랩 횟수. motif 축(3x3 = 512종) 전체를 흔들어야 해서 지표 CI(2000)
# 보다 낮춰 잡았다. enrichment.json의 ci_n_boot에 그대로 기록된다.
ENRICHMENT_CI_BOOT = 500


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


def _motif_features(
    cfg: Any, df: pd.DataFrame, cache: dict, n_max: int
) -> tuple[dict[str, np.ndarray], float]:
    """df의 URL들에 대해 표현별 히스토그램 행렬과 평균 창 개수를 만든다.

    ``cache``는 URL -> 표현별 벡터 dict. 시드마다 split만 달라지고 격자는 같으므로
    5시드 전체에서 URL당 한 번만 인코딩·히스토그램 계산을 한다.
    """
    from qrphish.dataset import data_mask_for
    from qrphish.motifs import (
        extract_patch_ids,
        patch_histogram,
        spatial_pyramid_histogram,
    )

    ec = ec_const(_get(_get(cfg, "qr"), "ec", "L"))
    dm_cache: dict[int, np.ndarray] = {}
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
            }
            cache[url] = got
        for k, _, _ in MOTIF_REPRESENTATIONS:
            rows[k].append(got[k])
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

    X, n_windows_mean = _motif_features(cfg, df, cache, n_max)
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
        if done_path.exists():
            print(f"[skip] motifs {stratum} (results.json 존재)", flush=True)
            out[stratum] = json.loads(done_path.read_text(encoding="utf-8"))
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


def _probe_aggregate(per_seed: list[dict], n_seeds: int) -> dict:
    """시드별 프로브 결과 -> 목표별 요약.

    ``ci``는 시드별 그룹 부트스트랩 CI의 하한/상한 평균이다. 정식 집계 CI가 아니므로
    유의 판정은 **시드별로** (CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한))를 따진 뒤
    ``n_seeds`` 전부에서 성립할 것을 요구한다. 일부 시드에서 표본 부족으로 건너뛴 목표는
    전 시드 일치를 확인할 수 없으므로 유의로 세지 않는다.
    """
    names: set[str] = set()
    for d in per_seed:
        names |= set(d["targets"])
    out: dict[str, Any] = {}
    for name in sorted(names):
        rows = [d["targets"][name] for d in per_seed if name in d["targets"]]
        usable = [r for r in rows if "skipped" not in r]
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
            "n_seeds_significant": int(sum(bool(r["significant"]) for r in usable)),
            "significant": bool(
                len(usable) == n_seeds and all(r["significant"] for r in usable)
            ),
            "above_random_init": bool(
                len(usable) == n_seeds and all(r["above_random_init"] for r in usable)
            ),
        }
    return out


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
        if done_path.exists():
            print(f"[skip] probes {stratum} (results.json 존재)", flush=True)
            out[stratum] = json.loads(done_path.read_text(encoding="utf-8"))
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
            if cache_path.exists():
                print(f"[skip] probes {stratum} seed{s} (seed{s}.json 존재)", flush=True)
                per_seed.append(json.loads(cache_path.read_text(encoding="utf-8")))
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
        sig = [k for k, v in lex.items() if v.get("significant")]
        ranked = sorted(
            (k for k in sig), key=lambda k: -lex[k]["score"]
        )[:10]

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
            "summary": {
                "n_targets": len(lex),
                "n_significant": len(sig),
                "frac_significant": (len(sig) / len(lex)) if lex else 0.0,
                "top10": [
                    {"target": k, "metric": lex[k]["metric"], "score": lex[k]["score"],
                     "shuffle": lex[k]["shuffle"], "random_init": lex[k]["random_init"]}
                    for k in ranked
                ],
                "label_probe": agg.get("label:phishing"),
            },
            "notes": [
                "프로브가 유의해도 '문자를 복원했다'는 뜻은 아니다. "
                "임베딩에서 그 속성이 선형으로 접근 가능하다는 뜻이다.",
                "유의 = 모든 시드에서 (그룹 부트스트랩 CI 하한 > 셔플 CI 상한).",
                "셔플 기준선은 그룹 단위로 목표를 갈아끼운다(행 단위 셔플은 기준선을 부풀린다).",
            ],
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
            f"유의 {len(sig)}/{len(lex)}",
            flush=True,
        )
    return out


# ------------------------------------------------------------------------ occlusion
# review_01 "C. Causal motif ablation" — 상위 motif를 뒤집으면 CNN 예측이 무너지는가.


def _occlusion_aggregate(per_seed: list[dict]) -> dict:
    """조건별 시드 평균. CI는 시드별 값을 그대로 per_seed에 남기고 여기서는 평균만."""
    conds = sorted({c for d in per_seed for c in d["conditions"]})
    keys = ("auroc", "d_auroc", "mean_logit_delta",
            "mean_logit_delta_phishing", "mean_logit_delta_benign")
    out: dict[str, Any] = {
        "auroc_original": float(np.nanmean([d["auroc_original"] for d in per_seed])),
    }
    for c in conds:
        rows = [d["conditions"][c] for d in per_seed if c in d["conditions"]]
        block = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        block["d_auroc_sd_across_seeds"] = float(
            np.nanstd([r["d_auroc"] for r in rows], ddof=0)
        )
        block["n_seeds_d_auroc_ci_excludes_0"] = int(
            sum(1 for r in rows if r["d_auroc_ci"][1] < 0 or r["d_auroc_ci"][0] > 0)
        )
        block["n_flipped_mean"] = float(np.nanmean([r["n_flipped"]["mean"] for r in rows]))
        block["frac_samples_no_match"] = float(
            np.nanmean([r["n_flipped"]["frac_zero"] for r in rows])
        )
        out[c] = block
    return out


def run_occlusion(
    cfg: Any,
    strata: list[str] | None = None,
    top_k: int = 10,
    size: int = 3,
    n_random_rep: int = 5,
) -> dict:
    """인과 motif 절제 (review_01 C). ``run_motifs`` 결과가 먼저 있어야 한다.

    ``reports/motifs/{stratum}/enrichment.json``의 상위 phishing motif가 매칭된 창의
    중심 모듈을 뒤집고, 같은 개수의 무작위 데이터 모듈 뒤집기 및 상위 benign motif
    뒤집기와 원본 대비 쌍체 비교한다. 결과는
    ``reports/occlusion/{stratum}/results.json``.
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
        if done_path.exists():
            print(f"[skip] occlusion {stratum} (results.json 존재)", flush=True)
            out[stratum] = json.loads(done_path.read_text(encoding="utf-8"))
            continue
        enr_path = _reports_dir(cfg) / "motifs" / motif_cid / stratum / "enrichment.json"
        if not enr_path.exists():
            # run_motifs가 돌지 않은 층(MOTIF_STRATA 밖이거나 dropped)은 건너뛴다.
            msg = f"enrichment.json이 없다: {enr_path} — 층을 건너뛴다 ([9] motif를 먼저 돌려라)."
            print(f"[skip] occlusion {stratum}: {msg}")
            out[stratum] = {"skipped": msg}
            continue
        enr = load_enrichment(enr_path)
        if int(enr.get("size", size)) != int(size):
            raise ValueError(
                f"enrichment.json의 size={enr.get('size')}가 요청한 size={size}와 다르다"
            )
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
        for s in avail:
            t_seed = time.time()
            try:
                model, ds, arr, meta = _load_trained(cfg, cid, stratum, s)
                te_rows = np.nonzero(arr["split"].astype(int) == 2)[0]
                groups = meta["group"].astype(str).to_numpy()[te_rows]
                block = occlude_stratum(
                    model, ds, te_rows, groups, enr,
                    top_k=top_k, size=size, n_random_rep=n_random_rep, seed=s, n_boot=n_boot,
                )
            except Exception as exc:
                print(f"[ERROR] occlusion {stratum} seed{s}: {exc!r}")
                traceback.print_exc()
                per_seed.append({"seed": int(s), "error": repr(exc), "conditions": {}})
                continue
            block["seed"] = int(s)
            per_seed.append(block)
            print(f"  [occlusion] {stratum} seed{s} 종료 ({time.time() - t_seed:.1f}s)",
                  flush=True)
        if not any("error" not in b for b in per_seed):
            out[stratum] = {"error": "no usable seed", "per_seed": per_seed}
            print(f"[ERROR] occlusion {stratum}: 모든 시드 실패")
            continue

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
            "motif_source": str(enr_path),
            "top_k": int(top_k),
            "size": int(size),
            "n_random_rep": int(n_random_rep),
            "aggregate": _occlusion_aggregate([b for b in per_seed if "error" not in b]),
            "per_seed": per_seed,
            "notes": [
                "뒤집은 격자는 더 이상 유효한 QR이 아니다. RS 오류정정 덕에 실제 스캐너는 "
                "여전히 디코딩할 수 있지만, 이것은 모델 입력에 대한 개입일 뿐 실제 QR 변형이 아니다.",
                "무작위 대조는 각 샘플에서 phishing motif가 맞은 개수와 같은 수를 뒤집는다.",
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
    model = build_model(ckpt["arch"], ds.n, ds.canonical_data_mask)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sub = copy.copy(ds)
    sub.indices = np.nonzero(np.asarray(rows))[0]
    loader = DataLoader(sub, batch_size=256, shuffle=False, num_workers=0)
    y, p = predict_probs(model, loader, device=pick_device())
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
) -> dict:
    """F-a/F-c 베이스라인: ``fit_frame``(train)에서 fit → 외부 평가 행에 apply.

    설계 3.7 — CNN만 무너지는지, 텍스트 기준선도 함께 무너지는지가 해석 매트릭스를 가른다.
    """
    from qrphish.transfer import transfer_baselines

    if fit_frame is None or len(fit_frame) == 0:
        return {}
    base = transfer_baselines(
        _subsample(fit_frame, TRANSFER_FIT_CAP, seed), ext_eval, seed, fit_on=fit_on
    )
    if not with_motif:
        return base
    # motif 히스토그램은 URL당 QR 인코딩이 필요해 훨씬 비싸다. fit/eval 모두 따로 캡을 걸고,
    # 실패하더라도 전이 판정(CNN AUROC 대 순열 바닥선)은 그대로 진행한다.
    try:
        fit_m = _subsample(fit_frame, TRANSFER_MOTIF_FIT_CAP, seed)
        ev_m = _subsample(ext_eval, TRANSFER_MOTIF_EVAL_CAP, seed)
        H = _motif_hists(cfg, {"fit": fit_m, "eval": ev_m}, {})
        motif = {
            key: {"H_fit": H["fit"][key], "H_eval": H["eval"][key]}
            for key in ("patch3", "pyramid3")
        }
        got = transfer_baselines(fit_m, ev_m, seed, fit_on=fit_on, motif=motif)
        base.update({k: v for k, v in got.items() if k.startswith("motif_")})
    except Exception as exc:  # motif는 보조 지표다
        print(f"[WARN] transfer motif 베이스라인 실패 {stratum} seed{seed}: {exc!r}")
    return base


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


def _fa_one_seed(
    cfg: Any,
    cid: str,
    stratum: str,
    seed: int,
    ext_df: pd.DataFrame,
    ext_cond_dir: Path,
    eval_rows: str,
    thresholds: dict[int, float],
) -> dict:
    """F-a 한 시드: 외부 프레임 준비 → 층 조립 → WebPhish 체크포인트로 추론."""
    from qrphish.transfer import build_external_stratum, prepare_external_frame

    df, diag = prepare_external_frame(cfg, ext_df, stratum, seed)
    if len(df) == 0:
        return {"seed": seed, "error": "empty external stratum after matching"}
    seed_dir = ext_cond_dir / stratum / f"seed{seed}"
    sm = build_external_stratum(cfg, df, stratum, seed, seed_dir)
    meta = pd.read_parquet(seed_dir / "meta.parquet")
    rows = _eval_rows_mask(meta, eval_rows)
    ckpt_path = _seed_dir(cfg, cid, stratum, seed) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"WebPhish 체크포인트가 없다: {ckpt_path}. F-a는 1차 학습 산출물을 재사용한다."
        )
    y, p, meta, _arr = _predict_with_checkpoint(cfg, ckpt_path, seed_dir, rows)
    idx = np.nonzero(rows)[0]
    return {
        "seed": seed,
        "y": np.asarray(y, dtype=np.int64),
        "p": np.asarray(p, dtype=np.float64),
        "group": meta["group"].to_numpy()[idx].astype(str),
        "threshold": thresholds.get(int(seed)),
        "frame": df,
        "eval_frame": meta.iloc[idx].reset_index(drop=True),
        "stratum_meta": sm,
        "n_after_match": diag.get("n_after_match"),
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
        "verdict": _verdict(auroc, ci, float(null_perm.get("ci_upper", float("nan"))),
                            in_domain, gates),
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
    source_tag: str | None = None,
    cid: str = MAIN_CONDITION,
    eval_rows: str = "all",
    n_perm: int = 200,
    dedup: str = "etld1",
    motif: bool = True,
    entry: dict | None = None,
) -> dict:
    """외부 검증(F). 설계 3·4·6·7절.

    Args:
        external_csv: 병합·중복제거가 끝난 외부 평가 세트 CSV.
        sources: ``source`` 컬럼으로 거를 소스 이름들(예: ``["openphish"]``). None이면 전부.
        strata: 대상 층. None이면 config의 ``strata``.
        modes: 실행할 F 모드. ``"a"``(zero-shot, 학습 없음) / ``"b"``(외부 자체 학습) /
            ``"c"``(역방향) / ``"d"``(혼합 — 미구현).
        eval_rows: F-a 평가 행. 주 결과는 ``"all"``.
        n_perm: 그룹 단위 라벨 순열 횟수(바닥선).

    Returns:
        ``{mode: {stratum: results}}``. 부작용으로
        ``reports/transfer/{source_tag}/{mode}/{cid}/{stratum}/results.json``을 쓴다.

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
            rpath = _transfer_report_dir(cfg, tag, mode, cid, stratum) / "results.json"
            if rpath.exists():
                print(f"[skip] transfer {mode} {stratum} (results.json 존재)", flush=True)
                out[mode][stratum] = _read_json_safe(rpath)
                continue
            t0 = time.time()
            # F-c는 F-b가 외부에서 학습한 체크포인트를 읽는다 — 그래서 "b" 디렉터리를 본다.
            ext_cond_dir = _external_cond_dir(cfg, tag, "b" if mode == "c" else mode, cid)
            print(f"[transfer] F-{mode} {stratum} 시작 — seeds={seeds}", flush=True)
            try:
                if mode == "a":
                    res = _run_transfer_a(
                        cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir,
                        eval_rows=eval_rows, n_perm=n_perm, n_boot=n_boot,
                        secondary_min=secondary_min, motif=motif, ext_stats=ext_stats,
                        coll_gate=coll_gate,
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


def _apply_collection_gate(data: dict, coll_gate: dict | None, recomputed_ok: bool) -> bool:
    """편향 게이트를 결정한다 — 수집 단계 판정이 있으면 그것을 쓴다.

    반환값이 ``gates["bias_direction_ok"]``에 들어간다. 어느 쪽을 썼는지는 ``data``에
    남겨 결과 JSON만 보고도 추적할 수 있게 한다.
    """
    if coll_gate is not None:
        data["bias_gate"] = {
            "source": "collection",
            "gate_passed": bool(coll_gate["gate_passed"]),
            "path": coll_gate.get("source"),
            "block": coll_gate.get("block"),
            "note": coll_gate.get("gate_note"),
            "recomputed_transfer_stage": bool(recomputed_ok),
        }
        return bool(coll_gate["gate_passed"])
    data["bias_gate"] = {
        "source": "recomputed_transfer_stage",
        "gate_passed": bool(recomputed_ok),
        "note": "수집 단계 bias_diagnostics.json을 못 찾아 층 프레임에서 재계산했다.",
    }
    return bool(recomputed_ok)


def _run_transfer_a(
    cfg, cid, tag, stratum, seeds, ext_all, ext_cond_dir, *,
    eval_rows, n_perm, n_boot, secondary_min, motif, ext_stats, coll_gate=None,
) -> dict:
    """F-a — zero-shot. 학습하지 않는다. 임계값은 WebPhish val 값을 재사용한다."""
    from qrphish.dataset import stratum_tier
    from qrphish.transfer import bias_direction_ok, gates_from, permutation_null

    wp_res = _webphish_results(cfg, cid, stratum)
    thresholds = _seed_thresholds(wp_res)
    per_seed: list[dict] = []
    for s in seeds:
        try:
            per_seed.append(
                _fa_one_seed(cfg, cid, stratum, s, ext_all, ext_cond_dir, eval_rows, thresholds)
            )
        except Exception as exc:
            print(f"[ERROR] transfer F-a {stratum} seed{s}: {exc!r}")
            traceback.print_exc()
            per_seed.append({"seed": s, "error": repr(exc)})
    ok = [d for d in per_seed if "error" not in d]
    if not ok:
        raise RuntimeError(f"F-a {stratum}: 모든 시드 실패")

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

    # 베이스라인은 첫 성공 시드에서만 낸다(WebPhish train에서 fit하는 비용이 크고,
    # 판정의 주 지표는 CNN AUROC다). 어느 시드였는지 결과에 남긴다.
    wp_frame, _ = _prepare_frame(cfg, stratum, int(first["seed"]))
    wp_train = wp_frame[_split_mask(wp_frame, 0)].reset_index(drop=True)
    baselines = _transfer_baseline_block(
        cfg, stratum, int(first["seed"]), ev,
        fit_frame=wp_train, fit_on="webphish_train", with_motif=motif,
    )

    bias = bias_direction_ok(ev, wp_frame)
    sm = first.get("stratum_meta")
    data = _transfer_data_block(ev, sm, eval_rows, tier)
    data["length_match"] = {"bucket": _length_bucket(_get(cfg, "condition")),
                            "all_identical": True}
    data["external_loader"] = ext_stats.get("loader")
    data["bias_diagnostics"] = bias
    data["baseline_seed"] = int(first["seed"])
    gate_notes: dict[str, str] = {}
    bias_ok = _apply_collection_gate(data, coll_gate, bool(bias["ok"]))
    gates = gates_from(baselines, data["split_diagnostics"].get("top5_test_group_frac"),
                       bias_ok, notes=gate_notes)
    data["gate_notes"] = gate_notes
    if tier == "dropped":
        gates["stratum_tier_ok"] = False

    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    res = _transfer_results(
        cfg, mode="a", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
    )
    if motif:
        res["motif_replication"] = _transfer_motif_replication(
            cfg, tag, stratum, first, ok
        )
    return res


def _transfer_motif_replication(cfg, tag, stratum, first, ok) -> dict:
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
        d = _reports_dir(cfg) / "transfer" / tag / "motif_replication" / stratum
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
    from qrphish.transfer import gates_from, permutation_null, transfer_baselines

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
    bias_ok = _apply_collection_gate(data, coll_gate, True)
    gates = gates_from(baselines, None, bias_ok, notes=gate_notes)
    data["gate_notes"] = gate_notes
    wp_res = _webphish_results(cfg, cid, stratum)
    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    return _transfer_results(
        cfg, mode="b", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
        extra={"per_seed_train": [{k: v for k, v in d.items() if k != "baselines"}
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
    from qrphish.transfer import gates_from, permutation_null, transfer_baselines

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
    bias_ok = _apply_collection_gate(data, coll_gate, True)
    gates = gates_from(baselines, None, bias_ok, notes=gate_notes)
    data["gate_notes"] = gate_notes
    wp_res = _webphish_results(cfg, cid, stratum)
    in_domain = ((wp_res or {}).get("aggregate") or {}).get("auroc_mean")
    return _transfer_results(
        cfg, mode="c", source_tag=tag, cid=cid, stratum=stratum, data=data, model=model,
        null_perm=null_perm, baselines=baselines,
        in_domain=(float(in_domain) if in_domain is not None else None), gates=gates,
    )
