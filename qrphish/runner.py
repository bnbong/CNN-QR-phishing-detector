"""실험 매트릭스 오케스트레이션 (스펙 7·8절).

순서 원칙(스펙 12절 #14): **P0(카운트·누출 진단·라운드트립 게이트)를 통과하기 전에는
모델을 학습하지 않는다.** ``run_matrix``는 ``reports/stratum_counts.json``이 없으면
``FileNotFoundError``로 하드 실패한다(우회 옵션 없음).
"""

from __future__ import annotations

import copy
import json
import subprocess
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
from qrphish.evaluate import metrics_from_probs, pick_threshold, predict_probs
from qrphish.models import build_model
from qrphish.train import pick_device, set_seed, train_model

__all__ = [
    "condition_id",
    "run_p0",
    "run_matrix",
    "run_explain",
    "aggregate_reports",
    "load_matrix",
]

SCHEMA_VERSION = 1
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
def _prepare_frame(cfg: Any, stratum: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """load → path filter → 층 필터 → **그룹 분할 → split별 길이 매칭**.

    매칭을 분할보다 **뒤에** 두는 것이 핵심이다. 층 전체에서 한 번 맞춰봐야
    :func:`group_split`의 그룹 배정과 5% 상한 다운샘플이 행을 통째로 옮기거나 지우면서
    split 안의 길이 주변분포를 다시 깨뜨린다. 매칭은 행 제거만 하므로 순서를 바꿔도
    split 간 그룹 교집합 0은 유지된다.
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
    df = df[[version_in_spec(int(v), stratum) for v in df["version"]]].reset_index(drop=True)
    diag["n_in_stratum"] = int(len(df))
    if len(df) == 0:
        diag["n_after_match"] = 0
        return df, diag

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


def _run_one_seed(cfg: Any, stratum: str, seed: int, cond_dir: Path, entry: dict) -> dict:
    """한 (층, 시드) 실행 → per_seed 항목."""
    cond = _get(cfg, "condition")
    qr = _get(cfg, "qr")
    mcfg = _get(cfg, "model")
    ecfg = _get(cfg, "eval")

    df, diag = _prepare_frame(cfg, stratum, seed)
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
    torch.save({"state_dict": model.state_dict(), "n": ds.n, "arch": str(_get(mcfg, "arch"))},
               seed_dir / "model.pt")

    n_boot = int(_get(ecfg, "n_bootstrap", 2000))
    y_te, p_te = predict_probs(model, te_loader, device=device)
    test = metrics_from_probs(y_te, p_te, groups[ite], n_boot=n_boot, seed=seed,
                              threshold=tres.threshold)
    y_va, p_va = predict_probs(model, va_loader, device=device)

    out = {
        "seed": seed,
        "best_epoch": tres.best_epoch,
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
    if entry.get("text_upper_bound", True) and "charngram_lr" not in base_names:
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
            per_seed = [_run_one_seed(cfg_e, stratum, s, cond_dir, entry) for s in seeds]
            res = _assemble_results(cfg_e, cid, stratum, phase, entry, per_seed)
            rpath.parent.mkdir(parents=True, exist_ok=True)
            rpath.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
            results.append(res)
    return results


def _assemble_results(cfg, cid, stratum, phase, entry, per_seed) -> dict:
    """스펙 8절 results.json 스키마."""
    ok = [s for s in per_seed if "error" not in s]
    aur = np.array([s["test"]["auroc"] for s in ok], dtype=float) if ok else np.array([])
    f1 = np.array([s["test"]["f1"] for s in ok], dtype=float) if ok else np.array([])
    acc = np.array([s["test"]["acc"] for s in ok], dtype=float) if ok else np.array([])
    los = [s["test"]["auroc_ci"][0] for s in ok]
    his = [s["test"]["auroc_ci"][1] for s in ok]

    # sanity: CNN이 텍스트 상한선을 넘으면 누출이다(스펙 0절)
    text_ub = [
        s["baselines"]["charngram_lr"]["auroc"]
        for s in ok
        if "auroc" in s.get("baselines", {}).get("charngram_lr", {})
    ]
    text_mean = float(np.nanmean(text_ub)) if text_ub else float("nan")
    auroc_mean = float(np.nanmean(aur)) if aur.size else float("nan")

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
            "auroc_ci_pooled": [
                float(np.nanmean(los)) if los else float("nan"),
                float(np.nanmean(his)) if his else float("nan"),
            ],
            "f1_mean": float(np.nanmean(f1)) if f1.size else float("nan"),
            "acc_mean": float(np.nanmean(acc)) if acc.size else float("nan"),
            "text_upper_bound_auroc": text_mean,
        },
        "sanity": {
            "below_text_upper_bound": bool(np.isnan(text_mean) or auroc_mean <= text_mean + 1e-9),
            "label_shuffle_auroc": auroc_mean if entry.get("label_shuffle") else None,
        },
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
def run_explain(cfg: Any, condition_id_str: str, stratum: str | None = None,
                max_samples: int = 128) -> dict:
    """P3 — 저장된 best 모델에 Grad-CAM을 돌려 kind별 CAM 질량과 문자 기여도를 낸다."""
    from qrphish.explain import attribute_by_kind, attribute_to_chars, gradcam
    from qrphish.mapping import provenance
    from qrphish.qrgen import encode

    cond_dir = _out_root(cfg) / condition_id_str
    strata = [stratum] if stratum else [p.name for p in sorted(cond_dir.iterdir()) if p.is_dir()]
    out: dict[str, Any] = {}
    for st in strata:
        seed_dirs = sorted((cond_dir / st).glob("seed*"))
        if not seed_dirs:
            continue
        sd = seed_dirs[0]
        ckpt = torch.load(sd / "model.pt", map_location="cpu", weights_only=False)
        arr = load_stratum_arrays(sd)
        meta = pd.read_parquet(sd / "meta.parquet")
        cond = _get(cfg, "condition")
        qr = _get(cfg, "qr")
        ds = QRGridDataset(
            arr["X_packed"], arr["y"], int(arr["n"]), arr["versions"], int(arr["ec"]),
            features=str(_get(cond, "features", "data_only")),
            mask_mode=str(_get(qr, "mask_mode", "fixed")),
        )
        model = build_model(ckpt["arch"], ds.n, ds.canonical_data_mask)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        target = getattr(model, "last_conv_block", None)
        if target is None:
            out[st] = {"error": "Grad-CAM은 conv 모델(small_cnn)에만 적용한다"}
            continue

        ec = int(arr["ec"])
        test_idx = np.nonzero(arr["split"].astype(int) == 2)[0][:max_samples]
        acc_kind: dict[str, list[float]] = {}
        char_curves: dict[int, list[np.ndarray]] = {0: [], 1: []}
        cam_sum = None
        for i in test_idx:
            x, _ = ds[int(i)]
            cam = gradcam(model, x, target)
            cam_sum = cam if cam_sum is None else cam_sum + cam
            url = str(meta["url"].iloc[int(i)])
            art = encode(url, ec=ec, mask_pattern=int(meta["mask_used"].iloc[int(i)]))
            prov = provenance(url, art)
            n_art = art.modules.shape[0]
            off = (ds.n - n_art) // 2
            cam_c = cam[off : off + n_art, off : off + n_art]
            for k, v in attribute_by_kind(cam_c, prov).items():
                acc_kind.setdefault(k, []).append(v)
            ch = attribute_to_chars(cam_c, prov, url)
            if ch.size:
                # 상대 위치로 정규화해 길이가 다른 URL을 평균낼 수 있게 한다
                grid = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, ch.size), ch)
                char_curves[int(meta["label"].iloc[int(i)])].append(grid)
        res = {
            "n_samples": int(len(test_idx)),
            "cam_mass_by_kind": {k: float(np.mean(v)) for k, v in acc_kind.items()},
            "char_position_curve": {
                str(k): (np.mean(v, axis=0).tolist() if v else []) for k, v in char_curves.items()
            },
        }
        (cond_dir / st / "explain.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rp = cond_dir / st / "results.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            r["explain"] = {"cam_mass_by_kind": res["cam_mass_by_kind"]}
            rp.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if cam_sum is not None:
            np.save(cond_dir / st / "mean_cam.npy", cam_sum / max(len(test_idx), 1))
        out[st] = res
    return out


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
                "auroc_ci_lo": (agg.get("auroc_ci_pooled") or [None, None])[0],
                "auroc_ci_hi": (agg.get("auroc_ci_pooled") or [None, None])[1],
                "f1_mean": agg.get("f1_mean"),
                "acc_mean": agg.get("acc_mean"),
                "text_upper_bound_auroc": agg.get("text_upper_bound_auroc"),
                "below_text_upper_bound": r.get("sanity", {}).get("below_text_upper_bound"),
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
