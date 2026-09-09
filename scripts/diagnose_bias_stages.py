"""단계별 경로·길이 편향 진단 (리뷰 03 항목 4).

`docs/REPORT.md`는 "WebPhish에서는 benign에 경로가 적고 phishing에 많은데 외부에서는 방향이
뒤집힌다"고 설명해 왔다. 그런데 저장된 매칭 후 v2 진단값은 WebPhish benign 99.75% /
phishing 44.31%, 외부 benign 86.83% / phishing 33.73%로 **둘 다 benign 쪽이 높다**.
이 스크립트는 그 값이 어느 단계에서 만들어지는지를 원자료로 따라간다.

각 데이터셋(WebPhish, 외부 세트)에 대해 다섯 단계를 찍는다::

    raw          로더 직전 원본 행 (라벨 매핑 전)
    norm         load_webphish / load_external 통과 직후 (정규화·중복·충돌 제거 끝)
    stratum      QR 버전 층 필터 후
    split        group_split 후
    matched      split별 길이 매칭 후 = **실제 평가 표본**

단계마다 클래스별로 n, path_depth>=1 비율, 바이트 길이 중앙값, `www.` 비율, `/` 개수 평균을
낸다. 층·시드는 `_prepare_frame` / `prepare_external_frame`을 그대로 호출하므로 F-a 결과
JSON의 `bias_diagnostics`와 같은 프레임을 본다.

사용::

    uv run python scripts/diagnose_bias_stages.py
    uv run python scripts/diagnose_bias_stages.py --strata v2 v3 --seed 0

산출: `reports/external/bias_stages.json` + 표준출력 표.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrphish.config import load_config  # noqa: E402
from qrphish.urls import load_webphish, path_depth  # noqa: E402

STAGES = ("raw", "norm", "stratum", "split", "matched")


def _class_stats(df: pd.DataFrame) -> dict[str, Any]:
    """클래스별 편향 지표. 입력은 url/label 컬럼을 가진 프레임."""
    out: dict[str, Any] = {}
    for lab, name in ((0, "benign"), (1, "phishing")):
        sub = df[df["label"] == lab]
        if not len(sub):
            out[name] = {"n": 0}
            continue
        urls = sub["url"].astype(str)
        depth = (
            sub["path_depth"].astype(int)
            if "path_depth" in sub.columns
            else pd.Series([path_depth(u) for u in urls], index=sub.index)
        )
        blen = (
            sub["url_bytes_len"].astype(int)
            if "url_bytes_len" in sub.columns
            else pd.Series([len(u.encode("utf-8")) for u in urls], index=sub.index)
        )
        out[name] = {
            "n": int(len(sub)),
            "path_ge1_frac": float((depth >= 1).mean()),
            "path_depth_mean": float(depth.mean()),
            "bytes_median": float(blen.median()),
            "www_frac": float(urls.str.contains(r"(?:^|//)www\.", regex=True).mean()),
            "slash_mean": float(urls.str.count("/").mean()),
        }
    ben = out.get("benign", {}).get("path_ge1_frac")
    phi = out.get("phishing", {}).get("path_ge1_frac")
    out["path_diff_benign_minus_phishing"] = (
        float(ben - phi) if ben is not None and phi is not None else None
    )
    return out


def _raw_webphish(cfg: Any) -> pd.DataFrame:
    """라벨 매핑만 한 원본. 정규화·중복제거 이전 상태를 본다."""
    data = cfg.data
    raw = pd.read_csv(Path(str(data.csv_path)), dtype=str)
    cat = raw[str(data.category_col)].astype(str).str.strip().str.lower()
    pos = str(data.positive_label).strip().lower()
    df = pd.DataFrame({"url": raw[str(data.url_col)].astype(str), "cat": cat})
    df = df[df["cat"].isin({pos, "ham", "benign", "legitimate", "good", "0"})].copy()
    df["label"] = (df["cat"] == pos).astype(int)
    return df.drop(columns=["cat"]).reset_index(drop=True)


def _raw_external(path: Path) -> pd.DataFrame:
    """조립된 외부 세트 CSV의 원본 행. 이 파일은 이미 정규화가 끝나 있다."""
    df = pd.read_csv(path)
    df["url"] = df["url"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def _label_map_note(cfg: Any) -> dict[str, Any]:
    """`spam`/`ham` → phishing/benign 매핑과 원본 컬럼 정의를 결과에 박아 둔다."""
    return {
        "webphish": {
            "csv": str(cfg.data.csv_path),
            "category_col": str(cfg.data.category_col),
            "url_col": str(cfg.data.url_col),
            "positive_label": str(cfg.data.positive_label),
            "mapping": {str(cfg.data.positive_label): "phishing(1)",
                        "ham|benign|legitimate|good|0": "benign(0)"},
        },
        "external": {
            "category_col": "Category",
            "url_col": "Data",
            "mapping": {"spam": "phishing(1)", "ham|benign|legitimate|good|0": "benign(0)"},
            "note": (
                "조립이 끝난 external_*.csv는 Category/Data가 아니라 url/label 컬럼을 갖는다. "
                "이 스크립트의 raw 단계는 그 파일을 읽으므로 '수집 파이프라인 출력'이 원본이다."
            ),
        },
    }


def diagnose_frame(
    cfg: Any,
    name: str,
    raw: pd.DataFrame,
    loaded: pd.DataFrame,
    strata: list[str],
    seed: int,
    *,
    external: bool,
) -> dict[str, Any]:
    """한 데이터셋에 대해 층별로 raw → matched 다섯 단계를 낸다."""
    from qrphish.runner import _prepare_frame
    from qrphish.transfer import prepare_external_frame

    out: dict[str, Any] = {
        "raw": _class_stats(raw),
        "norm": _class_stats(loaded),
        "strata": {},
    }
    for stratum in strata:
        if external:
            df, diag = prepare_external_frame(cfg, loaded, stratum, seed)
        else:
            df, diag = _prepare_frame(cfg, stratum, seed)
        # `_prepare_frame`은 중간 프레임을 돌려주지 않는다. 같은 단계를 다시 만들지 않고
        # 진단에 필요한 만큼만 재현한다 — 층 필터와 분할은 순수 필터라 재현이 정확하다.
        stage: dict[str, Any] = {
            "matched": _class_stats(df),
            "counts": {
                "n_loaded": diag.get("n_loaded"),
                "n_in_stratum": diag.get("n_in_stratum"),
                "n_after_split": diag.get("n_after_split"),
                "n_after_match": diag.get("n_after_match"),
                "n_dropped_capacity": diag.get("n_dropped_capacity"),
            },
            "length_match": diag.get("length_match"),
        }
        # 층·분할 단계 프레임은 길이 매칭 리포트가 들고 있지 않으므로 직접 만든다.
        base = loaded if external else loaded
        from qrphish.qrgen import natural_version
        from qrphish.runner import _get, ec_const, version_in_spec

        ec = ec_const(_get(_get(cfg, "qr"), "ec", "L"))
        work = base.copy()
        if "version" not in work.columns:
            work["version"] = [natural_version(u, ec) for u in work["url"]]
        work = work[work["version"].notna()]
        in_str = work[[version_in_spec(int(v), stratum) for v in work["version"]]]
        stage["stratum"] = _class_stats(in_str.reset_index(drop=True))
        # split 단계: 매칭 전 프레임은 group_split 직후 상태다. `_prepare_frame`이 되돌려
        # 주지 않으므로 같은 시드로 group_split만 다시 돌린다(결정적).
        from qrphish.splits import group_split

        ratios = tuple(_get(_get(cfg, "split"), "ratios", (0.70, 0.15, 0.15)))
        after_split, _ = group_split(
            in_str.reset_index(drop=True),
            ratios=ratios,
            seed=seed,
            max_group_frac=float(_get(_get(cfg, "split"), "max_group_frac", 0.05)),
        )
        stage["split"] = _class_stats(after_split)
        # 매칭이 무엇을 떨어뜨렸는지: split 단계의 (길이, 경로 유무) 셀별 클래스 수.
        stage["match_effect"] = _match_effect(after_split, df)
        out["strata"][stratum] = stage
    return out


def _match_effect(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    """길이 매칭이 경로 보유율을 어떻게 움직였는지 수치로 남긴다.

    길이 버킷 1:1 매칭은 **길이별로** 두 클래스를 같은 수로 자른다. 어느 클래스가 그 길이에서
    더 많았는지에 따라 경로 있는 행이 남기도 하고 빠지기도 한다. 여기서는 매칭 전후의
    경로 보유율과, 클래스별로 몇 행이 빠졌는지를 낸다.
    """
    out: dict[str, Any] = {}
    for lab, name in ((0, "benign"), (1, "phishing")):
        b = before[before["label"] == lab]
        a = after[after["label"] == lab]
        out[name] = {
            "n_before": int(len(b)),
            "n_after": int(len(a)),
            "n_dropped": int(len(b) - len(a)),
            "path_ge1_before": float((b["path_depth"] >= 1).mean()) if len(b) else None,
            "path_ge1_after": float((a["path_depth"] >= 1).mean()) if len(a) else None,
            "bytes_median_before": float(b["url_bytes_len"].median()) if len(b) else None,
            "bytes_median_after": float(a["url_bytes_len"].median()) if len(a) else None,
        }
    return out


def _print_table(title: str, stats: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(
        f"  {'class':10s} {'n':>8s} {'path>=1':>8s} {'depth':>7s} "
        f"{'bytes_md':>9s} {'www':>7s} {'slash':>7s}"
    )
    for name in ("benign", "phishing"):
        d = stats.get(name, {})
        if not d.get("n"):
            print(f"  {name:10s} {'0':>8s}")
            continue
        print(
            f"  {name:10s} {d['n']:>8,} {d['path_ge1_frac']:>8.4f} "
            f"{d['path_depth_mean']:>7.2f} {d['bytes_median']:>9.1f} "
            f"{d['www_frac']:>7.4f} {d['slash_mean']:>7.2f}"
        )
    diff = stats.get("path_diff_benign_minus_phishing")
    if diff is not None:
        print(f"  benign - phishing 경로 보유율 차이: {diff:+.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="단계별 경로·길이 편향 진단")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--external", default="data/external/external_primary_2026-09-08.csv")
    ap.add_argument("--strata", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/external/bias_stages.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    strata = list(args.strata) if args.strata else list(cfg.strata)

    wp_raw = _raw_webphish(cfg)
    wp_loaded, wp_stats = load_webphish(
        Path(str(cfg.data.csv_path)),
        str(cfg.condition.url_mode),  # type: ignore[arg-type]
        category_col=str(cfg.data.category_col),
        url_col=str(cfg.data.url_col),
        positive_label=str(cfg.data.positive_label),
    )
    ext_path = Path(args.external)
    ext_raw = _raw_external(ext_path)
    from qrphish.transfer import load_external_frame

    ext_loaded, ext_stats = load_external_frame(
        ext_path, mode=str(cfg.condition.url_mode), webphish_csv=cfg.data.csv_path
    )

    result: dict[str, Any] = {
        "config": args.config,
        "external_csv": str(ext_path),
        "seed": int(args.seed),
        "strata": strata,
        "label_definitions": _label_map_note(cfg),
        "loader_stats": {"webphish": wp_stats, "external": ext_stats},
        "datasets": {},
    }
    result["datasets"]["webphish"] = diagnose_frame(
        cfg, "webphish", wp_raw, wp_loaded, strata, args.seed, external=False
    )
    result["datasets"]["external"] = diagnose_frame(
        cfg, "external", ext_raw, ext_loaded, strata, args.seed, external=True
    )

    for ds in ("webphish", "external"):
        block = result["datasets"][ds]
        print(f"\n{'=' * 78}\n{ds.upper()}\n{'=' * 78}")
        _print_table(f"[{ds}] raw (라벨 매핑만)", block["raw"])
        _print_table(f"[{ds}] norm (정규화·중복·충돌 제거 후)", block["norm"])
        for stratum, st in block["strata"].items():
            for stage in ("stratum", "split", "matched"):
                _print_table(f"[{ds}] {stratum} / {stage}", st[stage])
            me = st["match_effect"]
            print(
                f"  매칭 제거: benign {me['benign']['n_dropped']:,} / "
                f"phishing {me['phishing']['n_dropped']:,}  "
                f"(경로 보유율 benign {me['benign']['path_ge1_before']:.4f}→"
                f"{me['benign']['path_ge1_after']:.4f}, "
                f"phishing {me['phishing']['path_ge1_before']:.4f}→"
                f"{me['phishing']['path_ge1_after']:.4f})"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
