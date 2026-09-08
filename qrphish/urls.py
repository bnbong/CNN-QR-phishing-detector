"""URL 정규화, eTLD+1 추출, WebPhish 로딩(스펙 4절 / 1.5 / 1.9)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import pandas as pd
import tldextract

__all__ = [
    "has_leading_scheme",
    "ensure_scheme",
    "strip_leading_scheme",
    "normalize_url",
    "etld1",
    "path_depth",
    "load_webphish",
    "default_extractor",
    "load_external",
]

# **선행** 스킴만 인식한다. ``"://" in s``로 판정하면
# ``good.com/redirect?url=https://evil.com``처럼 쿼리 안에 URL을 품은 주소가 스킴 있는 URL로
# 오인돼 호스트가 빈 값(`<empty>` 그룹)이 된다. 스킴 문법은 RFC 3986 3.1.
_LEADING_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def has_leading_scheme(url: str) -> bool:
    """문자열이 **선행** 스킴(`http://`, `ftp://` 등)으로 시작하는가."""
    return _LEADING_SCHEME_RE.match(str(url).strip()) is not None


def ensure_scheme(url: str, default: str = "http://") -> str:
    """선행 스킴이 없으면 ``default``를 붙인다. 있으면 그대로."""
    s = str(url).strip()
    return s if has_leading_scheme(s) else default + s


def strip_leading_scheme(url: str) -> str:
    """선행 스킴을 1회 제거한다. 없으면 그대로."""
    s = str(url).strip()
    m = _LEADING_SCHEME_RE.match(s)
    return s[m.end():] if m else s


def default_extractor() -> tldextract.TLDExtract:
    """네트워크 접근 없이 번들 PSL 스냅샷만 쓰는 추출기.

    실험 재현성을 위해 원격 suffix list를 가져오지 않는다.
    """
    return tldextract.TLDExtract(suffix_list_urls=())


_EXTRACTOR: tldextract.TLDExtract | None = None


def _shared_extractor() -> tldextract.TLDExtract:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = default_extractor()
    return _EXTRACTOR


def normalize_url(url: str, mode: Literal["raw", "norm"]) -> str:
    """`raw`는 공백 strip만, `norm`은 소문자화 + 선행 `www.` 1회 제거 + 말미 `/` 제거.

    스펙 13절 결정 6에 따라 `m.`, `www2.` 같은 변형은 건드리지 않는다.
    """
    if mode not in ("raw", "norm"):
        raise ValueError(f"mode는 'raw'|'norm'이어야 한다 (got {mode!r})")
    s = str(url).strip()
    if mode == "raw":
        return s

    s = s.lower()
    # 선행 스킴이 있으면 스킴은 보존한 채 호스트 앞의 www.만 제거한다.
    m = _LEADING_SCHEME_RE.match(s)
    if m:
        head, rest = s[: m.end()], s[m.end() :]
    else:
        head, rest = "", s
    if rest.startswith("www."):
        rest = rest[4:]
    s = head + rest
    if s.endswith("/"):
        s = s[:-1]
    return s


def _host(url: str) -> str:
    s = ensure_scheme(url)
    try:
        return urlsplit(s).hostname or ""
    except ValueError:
        return ""


def etld1(url: str, extractor: tldextract.TLDExtract | None = None) -> str:
    """eTLD+1(registered_domain). 비면 호스트 문자열, 그것도 비면 `<empty>`."""
    ex = extractor if extractor is not None else _shared_extractor()
    host = _host(url)
    if not host:
        return "<empty>"
    res = ex(host)
    # tldextract 5.3+는 registered_domain을 top_domain_under_public_suffix로 개명했다.
    reg = getattr(res, "top_domain_under_public_suffix", None)
    if reg is None:
        reg = res.registered_domain
    if reg:
        return reg
    return host or "<empty>"


def path_depth(url: str) -> int:
    """호스트 뒤 경로의 비어 있지 않은 세그먼트 개수."""
    s = ensure_scheme(url)
    try:
        path = urlsplit(s).path
    except ValueError:
        return 0
    return sum(1 for seg in path.split("/") if seg)


def _read_table(path: Path, category_col: str, url_col: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, dtype=str)
    elif suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, dtype=str, sep=sep)
    else:
        raise ValueError(f"지원하지 않는 확장자: {path.suffix} ({path})")
    missing = [c for c in (category_col, url_col) if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing} (있는 컬럼: {list(df.columns)})")
    return df


def load_webphish(
    csv_path: str | Path,
    mode: Literal["raw", "norm"] = "norm",
    *,
    category_col: str = "Category",
    url_col: str = "Data",
    positive_label: str = "spam",
    extractor: tldextract.TLDExtract | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """WebPhish CSV/XLSX를 읽어 정규화·dedup·라벨충돌 제거를 마친 DataFrame을 반환한다.

    컬럼: url, label(phishing=1), group, url_len, url_bytes_len, path_depth.
    두 번째 반환값은 제거 통계 dict.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없다: {path}")
    raw = _read_table(path, category_col, url_col)

    stats: dict[str, int] = {"n_raw": int(len(raw))}

    df = pd.DataFrame(
        {
            "url_src": raw[url_col].astype("string"),
            "category": raw[category_col].astype("string").str.strip().str.lower(),
        }
    )

    # 결측/빈 URL 제거
    before = len(df)
    df = df[df["url_src"].notna() & (df["url_src"].str.strip().str.len() > 0)]
    stats["n_dropped_empty_url"] = before - len(df)

    # 라벨 매핑: positive_label -> 1, 그 외 알려진 값 -> 0, 미지 값은 제거
    pos = positive_label.strip().lower()
    known_neg = {"ham", "benign", "legitimate", "0", "good"}
    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label[df["category"] == pos] = 1
    label[df["category"].isin(known_neg) | (df["category"] == "0")] = 0
    before = len(df)
    keep = label.notna()
    stats["n_dropped_unknown_label"] = int((~keep).sum())
    df = df[keep].copy()
    df["label"] = label[keep].astype(int).to_numpy()

    # 정규화
    df["url"] = [normalize_url(u, mode) for u in df["url_src"]]
    before = len(df)
    df = df[df["url"].str.len() > 0]
    stats["n_dropped_empty_after_norm"] = before - len(df)

    # 라벨 충돌: 같은 URL이 양쪽 클래스 → 양쪽 모두 제거 (스펙 1.9)
    nunique = df.groupby("url")["label"].transform("nunique")
    conflict = nunique > 1
    stats["n_conflicting_urls"] = int(df.loc[conflict, "url"].nunique())
    stats["n_dropped_label_conflict"] = int(conflict.sum())
    df = df[~conflict].copy()

    # 완전 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").copy()
    stats["n_dropped_duplicate"] = before - len(df)

    ex = extractor if extractor is not None else _shared_extractor()
    df["group"] = [etld1(u, ex) for u in df["url"]]
    df["url_len"] = df["url"].str.len().astype(int)
    df["url_bytes_len"] = [len(u.encode("utf-8")) for u in df["url"]]
    df["path_depth"] = [path_depth(u) for u in df["url"]]

    df = df.drop(columns=["url_src", "category"])
    df = df.reset_index(drop=True)

    stats["n_final"] = int(len(df))
    stats["n_final_benign"] = int((df["label"] == 0).sum())
    stats["n_final_phishing"] = int((df["label"] == 1).sum())
    stats["n_groups"] = int(df["group"].nunique())
    return df, stats


def load_external(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict]:
    """2단계(외부 수집) 인터페이스. 구현은 `qrphish.external.load_external`에 위임한다.

    순환 import를 피하려고 함수 안에서 import한다(external.py가 이 모듈을 쓴다).
    """
    from qrphish.external import load_external as _impl

    return _impl(*args, **kwargs)  # type: ignore[arg-type]
