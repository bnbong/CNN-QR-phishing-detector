"""backups/data/URL.xlsx -> data/webphish.csv 변환.

원본 컬럼(Category, Data)을 그대로 유지한다. 결과 CSV는 .gitignore 대상이다.

사용법:
    uv run python scripts/convert_webphish.py [SRC_XLSX] [DST_CSV]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_SRC = Path("backups/data/URL.xlsx")
DEFAULT_DST = Path("data/webphish.csv")
COLUMNS = ["Category", "Data"]


def convert(src: Path, dst: Path) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"원본 엑셀이 없다: {src}")
    df = pd.read_excel(src, dtype=str)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing} (있는 컬럼: {list(df.columns)})")
    out = df[COLUMNS]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return out


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else DEFAULT_SRC
    dst = Path(argv[2]) if len(argv) > 2 else DEFAULT_DST
    out = convert(src, dst)
    print(f"{src} -> {dst}: {len(out)} rows")
    print(out["Category"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
