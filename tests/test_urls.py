"""urls.py 테스트 — 정규화, eTLD+1, dedup/라벨충돌(스펙 9절 9번)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qrphish.urls import (
    default_extractor,
    etld1,
    load_external,
    load_webphish,
    normalize_url,
    path_depth,
)


def test_normalize_raw_only_strips():
    assert normalize_url("  HTTP://WWW.Example.com/  ", "raw") == "HTTP://WWW.Example.com/"


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("HTTP://WWW.Example.com/", "http://example.com"),
        ("www.example.com/", "example.com"),
        ("https://www.a.www.b.com", "https://a.www.b.com"),  # 선행 www.만 1회
        ("http://m.example.com/", "http://m.example.com"),  # m. 은 보존
        ("http://www2.example.com", "http://www2.example.com"),  # www2. 는 보존
        ("http://example.com//", "http://example.com/"),  # 말미 / 1회만
    ],
)
def test_normalize_norm(src, expected):
    assert normalize_url(src, "norm") == expected


def test_normalize_invalid_mode():
    with pytest.raises(ValueError):
        normalize_url("http://x.com", "lower")  # type: ignore[arg-type]


def test_etld1_variants():
    ex = default_extractor()
    assert etld1("http://sub.foo.co.uk/x", ex) == "foo.co.uk"
    assert etld1("https://www.example.com", ex) == "example.com"
    # IP 주소 URL은 registered_domain이 비므로 호스트 문자열이 그룹 키가 된다
    assert etld1("http://192.168.10.24/a", ex) == "192.168.10.24"
    assert etld1("", ex) == "<empty>"


def test_path_depth():
    assert path_depth("http://example.com") == 0
    assert path_depth("http://example.com/") == 0
    assert path_depth("example.com/a/b/c") == 3
    assert path_depth("http://example.com/a?x=1/2") == 1


def test_load_webphish_columns_and_labels(sample_csv: Path):
    df, stats = load_webphish(sample_csv, "norm")
    for col in ("url", "label", "group", "url_len", "url_bytes_len", "path_depth"):
        assert col in df.columns
    assert set(df["label"].unique()) == {0, 1}
    assert stats["n_raw"] == 20
    assert stats["n_final"] == len(df)
    assert stats["n_final_benign"] + stats["n_final_phishing"] == len(df)


def test_url_bytes_len_is_utf8_bytes(tmp_path: Path):
    src = tmp_path / "u.csv"
    pd.DataFrame(
        [
            {"Category": "ham", "Data": "http://한글.com"},
            {"Category": "spam", "Data": "http://ascii.com"},
        ]
    ).to_csv(src, index=False)
    df, _ = load_webphish(src, "norm")
    row = df[df["url"].str.contains("한글")].iloc[0]
    assert row["url_bytes_len"] == len(row["url"].encode("utf-8"))
    assert row["url_bytes_len"] > row["url_len"]


def test_load_webphish_dedup_and_conflict(tmp_path: Path):
    """스펙 9절 9번: 중복 URL 제거 + 라벨 충돌 시 양쪽 모두 제거."""
    src = tmp_path / "dup.csv"
    pd.DataFrame(
        [
            {"Category": "ham", "Data": "http://a.com"},
            {"Category": "ham", "Data": "http://www.a.com/"},  # norm 후 중복
            {"Category": "ham", "Data": "http://c.com"},
            {"Category": "spam", "Data": "http://c.com"},  # 라벨 충돌
            {"Category": "spam", "Data": "http://b.com"},
        ]
    ).to_csv(src, index=False)
    df, stats = load_webphish(src, "norm")
    assert set(df["url"]) == {"http://a.com", "http://b.com"}
    assert df["url"].duplicated().sum() == 0
    assert stats["n_conflicting_urls"] == 1
    assert stats["n_dropped_label_conflict"] == 2
    assert stats["n_dropped_duplicate"] == 1


def test_raw_and_norm_dedup_independently(tmp_path: Path):
    """스펙 1.9: raw/norm 각각 별도 dedup — 공통 부분집합 강제 금지."""
    src = tmp_path / "rn.csv"
    pd.DataFrame(
        [
            {"Category": "ham", "Data": "http://a.com"},
            {"Category": "ham", "Data": "http://www.a.com/"},
            {"Category": "spam", "Data": "http://b.com"},
        ]
    ).to_csv(src, index=False)
    raw_df, _ = load_webphish(src, "raw")
    norm_df, _ = load_webphish(src, "norm")
    assert len(raw_df) == 3
    assert len(norm_df) == 2


def test_unknown_label_dropped(tmp_path: Path):
    src = tmp_path / "unk.csv"
    pd.DataFrame(
        [
            {"Category": "ham", "Data": "http://a.com"},
            {"Category": "mystery", "Data": "http://b.com"},
        ]
    ).to_csv(src, index=False)
    df, stats = load_webphish(src, "norm")
    assert stats["n_dropped_unknown_label"] == 1
    assert len(df) == 1


def test_load_webphish_xlsx(tmp_path: Path, sample_frame: pd.DataFrame):
    """확장자 분기: xlsx도 csv와 동일한 결과."""
    xlsx = tmp_path / "s.xlsx"
    csv = tmp_path / "s.csv"
    sample_frame.to_excel(xlsx, index=False)
    sample_frame.to_csv(csv, index=False)
    a, _ = load_webphish(xlsx, "norm")
    b, _ = load_webphish(csv, "norm")
    pd.testing.assert_frame_equal(a, b)


def test_missing_file_and_bad_suffix(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_webphish(tmp_path / "nope.csv", "norm")
    bad = tmp_path / "x.parquetish"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_webphish(bad, "norm")


def test_load_external_not_implemented(tmp_path: Path):
    with pytest.raises(NotImplementedError):
        load_external("tranco", tmp_path)
