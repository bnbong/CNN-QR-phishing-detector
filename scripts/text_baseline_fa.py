"""외부 세트의 **텍스트 기준선 F-a**만 뽑는다 (CNN 없이, 학습 없이).

설계 3.7 — CNN을 돌리기 전에 "WebPhish train에서 fit한 텍스트 기준선이 외부에서 얼마나
버티는가"를 먼저 본다. 층 준비(``prepare_external_frame``)와 게이트 진단까지 같은 경로를
쓰므로, 수집·정규화를 고친 뒤 값이 어떻게 움직였는지 확인하는 회귀 도구이기도 하다.

사용 예::

    uv run python scripts/text_baseline_fa.py \
        --external-csv data/external/external_2026-09-08.csv --stratum v3

산출물은 ``reports/transfer/{tag}/text_baseline_fa_{stratum}.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrphish.config import load_config  # noqa: E402
from qrphish.runner import (  # noqa: E402
    TRANSFER_FIT_CAP,
    _prepare_frame,
    _split_mask,
    _subsample,
)
from qrphish.transfer import (  # noqa: E402
    bias_direction_ok,
    collection_bias_gate,
    ensure_version_column,
    load_external_frame,
    permutation_null,
    prepare_external_frame,
    transfer_baselines,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="외부 세트 텍스트 기준선 F-a")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--external-csv", default="data/external/external_2026-09-08.csv")
    ap.add_argument("--stratum", default="v3", help="베이스라인·순열을 낼 층")
    ap.add_argument("--strata", default="", help="층 준비 통계만 낼 층 목록(쉼표). 비면 config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tag = Path(args.external_csv).stem

    t0 = time.time()
    ext_all, load_stats = load_external_frame(args.external_csv, mode=cfg.condition.url_mode)
    out: dict = {
        "external_csv": args.external_csv,
        "tag": tag,
        "load_sec": round(time.time() - t0, 1),
        "load_stats": load_stats,
    }
    t0 = time.time()
    ensure_version_column(cfg, ext_all)
    out["version_cache_sec"] = round(time.time() - t0, 1)

    strata = [s for s in args.strata.split(",") if s.strip()] or list(cfg.strata)
    frames = {}
    out["strata"] = {}
    for st in strata:
        t0 = time.time()
        ev, diag = prepare_external_frame(cfg, ext_all, st, args.seed)
        frames[st] = ev
        n_ben = int((ev["label"] == 0).sum())
        n_phi = int((ev["label"] == 1).sum())
        per_class_min = min(n_ben, n_phi)
        te = ev[_split_mask(ev, 2)]
        top5 = (
            float(te["group"].value_counts().head(5).sum() / len(te)) if len(te) else float("nan")
        )
        out["strata"][st] = {
            "sec": round(time.time() - t0, 1),
            "n_in_stratum": int(diag.get("n_in_stratum", 0)),
            "n_after_match": int(diag.get("n_after_match", len(ev))),
            "n_benign": n_ben,
            "n_phishing": n_phi,
            "per_class_min": per_class_min,
            "primary_ok": per_class_min >= cfg.stratum_rules.primary_min_per_class,
            "secondary_ok": per_class_min >= cfg.stratum_rules.secondary_min_per_class,
            "n_groups": int(ev["group"].nunique()),
            "top5_test_group_frac": top5,
        }

    st = args.stratum
    wp_frame, _ = _prepare_frame(cfg, st, args.seed)
    wp_train = wp_frame[_split_mask(wp_frame, 0)].reset_index(drop=True)
    ev = frames[st]

    out["bias"] = bias_direction_ok(ev, wp_frame)
    gate = collection_bias_gate(args.external_csv, Path(cfg.reports_dir).parent)
    out["collection_bias_gate"] = gate

    t0 = time.time()
    fit = _subsample(wp_train, TRANSFER_FIT_CAP, args.seed)
    out["baselines"] = transfer_baselines(fit, ev, args.seed, fit_on="webphish_train")
    out["baseline_sec"] = round(time.time() - t0, 1)

    # 바닥선은 CNN이 없으므로 가장 강한 텍스트 기준선(charngram)의 확률로 낸다.
    t0 = time.time()
    from qrphish import baselines as bl

    m = bl.fit_charngram_lr(
        np.asarray(fit["url"], dtype=object), np.asarray(fit["label"], dtype=np.int64), args.seed
    )
    p = bl.apply_charngram_lr(m, np.asarray(ev["url"], dtype=object))
    out["null_permutation_charngram"] = permutation_null(
        np.asarray(ev["label"], dtype=np.int64),
        p,
        np.asarray(ev["group"], dtype=object).astype(str),
        n_perm=args.n_perm,
        seed=0,
    )
    out["null_sec"] = round(time.time() - t0, 1)
    out["n_webphish_train"] = int(len(wp_train))
    out["stratum"] = st

    path = Path(args.out) if args.out else (
        Path(cfg.reports_dir) / "transfer" / tag / f"text_baseline_fa_{st}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[text-baseline] {path}")
    for k, v in out["baselines"].items():
        print(f"  {k:16s} AUROC {v['auroc']:.4f}  CI [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    print(f"  null(charngram) ci_upper {out['null_permutation_charngram']['ci_upper']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
