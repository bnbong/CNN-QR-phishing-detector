"""Colab 노트북 사전 점검 — `notebooks/colab_run.ipynb`를 축소 데이터로 실제 실행한다.

노트북의 파이썬 코드를 **한 글자도 바꾸지 않고** 그대로 exec 한다. 제거하는 것은
Colab 전용 줄(`!`/`%` 매직, `drive.mount`, `google.colab` import)과 `DATA_CSV`/`OUT_DIR`
재대입뿐이다. 두 변수는 실행 전 네임스페이스에 주입한다.

축소는 config 쪽에서만 한다. `qrphish.config.load_config`를 monkeypatch 해서 반환
config에 `model.max_epochs` / `eval.n_bootstrap` / `seed_list` 오버라이드를 얹는다.
노트북 코드에는 손대지 않는다.

사용:
    python scripts/preflight_colab.py --out-dir /tmp/pf --per-class 1500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Colab 전용이라 로컬에서 제거하는 줄. 이 외의 파이썬 줄은 그대로 실행한다.
_DROP_PREFIXES = ("!", "%")
_DROP_PATTERNS = (
    re.compile(r"^\s*from\s+google\.colab\b"),
    re.compile(r"^\s*drive\.mount\("),
    re.compile(r"^\s*(DATA_CSV|OUT_DIR)\s*="),
)


def strip_cell(src: str) -> str:
    """셀 소스에서 Colab 전용 줄만 걷어낸다."""
    kept = []
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith(_DROP_PREFIXES):
            continue
        if any(p.match(line) for p in _DROP_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def load_cells(nb_path: Path) -> list[tuple[int, str]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = strip_cell("".join(cell["source"]))
        out.append((i, code))
    return out


def make_subsample(src_csv: Path, dst: Path, per_class: int, seed: int) -> dict:
    """클래스별 단순 무작위 추출. 길이 분포는 원본 그대로 유지된다."""
    import pandas as pd

    df = pd.read_csv(src_csv, encoding="utf-8")
    parts = []
    for _label, sub in df.groupby(df.columns[0]):
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8")
    return {"n": int(len(out)), "by_class": out[out.columns[0]].value_counts().to_dict()}


def install_shims() -> list[str]:
    """Colab에만 있는 모듈의 최소 스텁. 노트북 코드가 그대로 돌게만 한다."""
    shimmed = []
    try:
        import IPython.display  # noqa: F401
    except ImportError:
        ip = types.ModuleType("IPython")
        disp = types.ModuleType("IPython.display")

        def Image(path=None, **kw):  # noqa: N802 - IPython API 이름 그대로
            return {"_shim_image": str(path)}

        disp.Image = Image  # type: ignore[attr-defined]
        ip.display = disp  # type: ignore[attr-defined]
        # matplotlib.pyplot이 백엔드를 고를 때 IPython.get_ipython()/version_info를 본다.
        ip.get_ipython = lambda: None  # type: ignore[attr-defined]
        ip.version_info = (8, 30, 0, "")  # type: ignore[attr-defined]
        ip.__version__ = "8.30.0"  # type: ignore[attr-defined]
        sys.modules["IPython"] = ip
        sys.modules["IPython.display"] = disp
        shimmed.append("IPython.display.Image")
    return shimmed


def patch_load_config(overrides: dict) -> None:
    """`load_config`가 축소 오버라이드를 얹은 config를 돌려주게 한다."""
    import qrphish.config as qcfg
    from qrphish.runner import apply_overrides

    orig = qcfg.load_config

    def patched(path):
        return apply_overrides(orig(path), overrides)

    qcfg.load_config = patched  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="OUT_DIR (Drive 폴더 흉내)")
    ap.add_argument("--data-csv", default=None, help="미지정 시 서브샘플을 만든다")
    ap.add_argument("--source-csv", default=str(REPO / "data" / "webphish.csv"))
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=2)
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--report", default=None, help="preflight_report.md 경로")
    ap.add_argument("--notebook", default=str(REPO / "notebooks" / "colab_run.ipynb"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    if args.data_csv:
        data_csv = Path(args.data_csv).resolve()
        sample_info = {"reused": str(data_csv)}
    else:
        data_csv = out_dir / "webphish_sub.csv"
        sample_info = make_subsample(
            Path(args.source_csv), data_csv, args.per_class, args.sample_seed
        )

    shimmed = install_shims()
    patch_load_config(
        {
            "model.max_epochs": args.max_epochs,
            "eval.n_bootstrap": args.n_bootstrap,
            "seed_list": [int(s) for s in args.seeds.split(",") if s.strip()],
        }
    )

    ns: dict = {
        "__name__": "__main__",
        "DATA_CSV": str(data_csv),
        "OUT_DIR": str(out_dir),
    }
    cells = load_cells(Path(args.notebook))
    rows: list[dict] = []
    failed_names: set[str] = set()

    for idx, code in cells:
        if not code.strip():
            rows.append({"cell": idx, "status": "empty", "sec": 0.0})
            print(f"[cell {idx:>2}] empty (Colab 전용 줄만 있었음)")
            continue
        t0 = time.time()
        try:
            exec(compile(code, f"<cell {idx}>", "exec"), ns)  # noqa: S102
            dt = time.time() - t0
            rows.append({"cell": idx, "status": "ok", "sec": dt})
            print(f"[cell {idx:>2}] OK   {dt:8.2f}s")
        except Exception as exc:
            dt = time.time() - t0
            tb = traceback.format_exc()
            # 앞선 실패 셀이 정의하지 못한 이름을 쓰면 연쇄 실패로 표기한다.
            cascade = isinstance(exc, NameError) and any(
                n in str(exc) for n in failed_names
            )
            missing = re.findall(r"name '([^']+)' is not defined", str(exc))
            failed_names.update(missing)
            rows.append(
                {
                    "cell": idx,
                    "status": "cascade" if cascade else "FAIL",
                    "sec": dt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                }
            )
            print(f"[cell {idx:>2}] {'CASC' if cascade else 'FAIL'} {dt:8.2f}s  {exc!r}")
            print(tb)

    report = Path(args.report or (out_dir / "preflight_report.md"))
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    lines = [
        "# Colab 사전 점검 리포트",
        "",
        f"- notebook: `{args.notebook}`",
        f"- OUT_DIR: `{out_dir}`",
        f"- DATA_CSV: `{data_csv}` ({sample_info})",
        f"- overrides: max_epochs={args.max_epochs}, n_bootstrap={args.n_bootstrap}, "
        f"seeds={args.seeds}",
        f"- shims: {shimmed or '없음'}",
        f"- 실패 셀 수: {n_fail}",
        "",
        "| cell | status | sec | error |",
        "|---|---|---|---|",
    ]
    for r in rows:
        err = (r.get("error") or "").replace("|", "\\|")[:200]
        lines.append(f"| {r['cell']} | {r['status']} | {r['sec']:.2f} | {err} |")
    for r in rows:
        if r.get("traceback"):
            lines += ["", f"## cell {r['cell']} traceback", "", "```", r["traceback"], "```"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n리포트: {report}  (실패 {n_fail}개)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
