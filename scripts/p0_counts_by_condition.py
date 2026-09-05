"""P0 카운트 표를 조건(url_mode × length_match) × 층으로 확장한다.

``qrphish p0``는 config의 ``condition.url_mode``를 고정한 채 length_match 세 가지만
훑고, config의 ``strata``에 없는 v1은 빼놓는다. 통합 보고에는 ``raw`` 조건과 v1까지
필요해서 이 스크립트로 카운트만 추가로 뽑는다. 모델은 학습하지 않는다.

first_pad_codeword의 클래스 간 총변동거리(TVD)도 함께 낸다. 길이 매칭이 의도대로
작동했다면 ``L-exact``에서 두 클래스의 패딩 경계 분포가 완전히 겹쳐 TVD = 0 이어야
한다(스펙 1.1). 층 전체를 쓰고 서브샘플링하지 않는다.

사용법:
    uv run python scripts/p0_counts_by_condition.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qrphish.config import load_config
from qrphish.dataset import ec_const, stratum_tier
from qrphish.qrgen import encode
from qrphish.runner import _prepare_frame, apply_overrides

STRATA = ("v1", "v2", "v3", "v4", "v5plus")
URL_MODES = ("norm", "raw")
# (length_match, length_bucket) — 스펙 1.1의 버킷 폭.
LENGTH_GRID = (("exact", 1), ("quantile", 5), ("none", 1))


def _tvd(a: np.ndarray, b: np.ndarray) -> float:
    """두 이산 표본의 총변동거리. 0이면 분포가 완전히 겹친다."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    support = np.union1d(a, b)
    pa = np.array([(a == v).mean() for v in support])
    pb = np.array([(b == v).mean() for v in support])
    return float(0.5 * np.abs(pa - pb).sum())


def _first_pad_tvd(cfg, df: pd.DataFrame) -> dict:
    ec = ec_const(cfg.qr.ec)
    mp = None if cfg.qr.mask_mode == "auto" else int(cfg.qr.mask_pattern)
    # 서브샘플링하지 않는다. 클래스별로 수백~수천이면 유한표본 잡음만으로 TVD가
    # 0.05 안팎까지 뜨는데, L-exact에서 우리가 보려는 값이 바로 "0인가"이기 때문이다.
    sub = df
    vals: dict[int, list[int]] = {0: [], 1: []}
    for url, label in zip(sub["url"], sub["label"], strict=True):
        art = encode(url, ec=ec, mask_pattern=mp)
        vals[int(label)].append(int(art.first_pad_codeword))
    ben, phi = np.array(vals[0]), np.array(vals[1])
    return {
        "n_sampled": int(len(sub)),
        "tvd_first_pad_codeword": _tvd(ben, phi),
        "mean_benign": float(ben.mean()) if ben.size else float("nan"),
        "mean_phishing": float(phi.mean()) if phi.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.seed_list[0])
    primary_min = cfg.stratum_rules.primary_min_per_class
    secondary_min = cfg.stratum_rules.secondary_min_per_class

    rows: list[dict] = []
    for url_mode in URL_MODES:
        for length_match, bucket in LENGTH_GRID:
            cfg_c = apply_overrides(
                cfg,
                {
                    "condition.url_mode": url_mode,
                    "condition.length_match": length_match,
                    "condition.length_bucket": bucket,
                },
            )
            for stratum in STRATA:
                try:
                    df, diag = _prepare_frame(cfg_c, stratum, seed)
                except Exception as exc:
                    rows.append(
                        {
                            "url_mode": url_mode,
                            "length_match": length_match,
                            "stratum": stratum,
                            "error": repr(exc),
                        }
                    )
                    continue
                counts = df["label"].value_counts().to_dict() if len(df) else {}
                n_ben, n_phi = int(counts.get(0, 0)), int(counts.get(1, 0))
                per_class_min = min(n_ben, n_phi)
                row = {
                    "url_mode": url_mode,
                    "length_match": length_match,
                    "stratum": stratum,
                    "n_before_match": diag.get("n_in_stratum"),
                    "n_total": int(len(df)),
                    "n_benign": n_ben,
                    "n_phishing": n_phi,
                    "per_class_min": per_class_min,
                    "tier": stratum_tier(per_class_min, primary_min, secondary_min),
                }
                # 패딩 누출 진단은 주 조건(norm)에서만 낸다.
                if url_mode == "norm" and len(df):
                    row.update(_first_pad_tvd(cfg_c, df))
                rows.append(row)
                print(
                    f"{url_mode}-{length_match}/{stratum}: "
                    f"n={row['n_total']} ben={n_ben} phi={n_phi} tier={row['tier']} "
                    f"tvd={row.get('tvd_first_pad_codeword')}"
                )

    out_json = reports / "p0_counts_by_condition.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    out_csv = reports / "p0_counts_by_condition.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n{out_json}\n{out_csv}")


if __name__ == "__main__":
    main()
