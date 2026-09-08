"""qrphish.external 테스트 — 정규화·dedup·라벨충돌·편향진단·fetch 스모크."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from qrphish import external as ext
from qrphish.urls import load_external as urls_load_external
from qrphish.urls import load_webphish

NET = pytest.mark.skipif(
    not os.environ.get("QRPHISH_NET"), reason="네트워크 테스트는 QRPHISH_NET=1일 때만"
)


def _write(path, rows):
    pd.DataFrame(rows, columns=["Category", "Data", "source", "first_seen"]).to_csv(
        path, index=False
    )
    return path


# ---------------------------------------------------------------- 정규화


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://example.com/a", "example.com/a"),
        ("https://example.com/a", "example.com/a"),
        ("ftp://example.com/a", "example.com/a"),
        ("  https://example.com/a  ", "example.com/a"),
        ("example.com/a", "example.com/a"),
        ("mailto:x@example.com", "mailto:x@example.com"),
    ],
)
def test_strip_scheme(raw, expected):
    assert ext.strip_scheme(raw) == expected


def test_normalize_external_url_matches_webphish_convention():
    # 스킴 제거 + 소문자 + 선행 www. 제거 + 말미 / 제거
    assert ext.normalize_external_url("HTTPS://WWW.Example.COM/Path/") == "example.com/path"
    assert ext.normalize_external_url("http://www.a.com", "raw") == "www.a.com"


def test_normalize_does_not_touch_m_prefix():
    assert ext.normalize_external_url("https://m.example.com/x") == "m.example.com/x"


# ---------------------------------------------------------------- parse_source


def test_parse_source_text_lines(tmp_path):
    p = tmp_path / "phishdb.txt"
    p.write_text("http://a.com/x\n\n# comment\nhttp://b.com/y\n", encoding="utf-8")
    df = ext.parse_source(ext.SOURCES["phishdb"], p)
    assert list(df["url_src"]) == ["http://a.com/x", "http://b.com/y"]
    assert set(df["label"]) == {1}


def test_parse_source_dated(tmp_path):
    p = tmp_path / "op.txt"
    p.write_text("2026-08-01T00:00:00Z\thttp://a.com/x\nhttp://b.com/y\n", encoding="utf-8")
    df = ext.parse_source(ext.SOURCES["openphish"], p)
    assert list(df["first_seen"]) == ["2026-08-01T00:00:00Z", ""]
    assert list(df["url_src"]) == ["http://a.com/x", "http://b.com/y"]


def test_parse_source_csv(tmp_path):
    p = tmp_path / "cc.csv"
    pd.DataFrame({"url": ["https://a.com/1"]}).to_csv(p, index=False)
    df = ext.parse_source(ext.SOURCES["commoncrawl"], p)
    assert list(df["url_src"]) == ["https://a.com/1"]
    assert set(df["label"]) == {0}


def test_build_external_dataset_roundtrip(tmp_path):
    src = tmp_path / "phishdb.txt"
    src.write_text("http://a.com/x\n", encoding="utf-8")
    parts = {"phishdb": ext.parse_source(ext.SOURCES["phishdb"], src)}
    written = ext.build_external_dataset(tmp_path / "out", parts, date="2026-09-08")
    df = pd.read_csv(written["phishdb"])
    assert list(df.columns) == ["Category", "Data", "source", "first_seen"]
    assert df["Category"].iloc[0] == "spam"


# ---------------------------------------------------------------- load_external


def test_load_external_columns_match_webphish(tmp_path):
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://bad.example/login.php", "phishdb", ""),
            ("ham", "https://www.good.example/docs/a/", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p)
    for col in ("url", "label", "group", "url_len", "url_bytes_len", "path_depth", "source"):
        assert col in df.columns
    assert set(df["url"]) == {"bad.example/login.php", "good.example/docs/a"}
    assert stats["n_final"] == 2
    assert df.loc[df["label"] == 0, "path_depth"].iloc[0] == 2


def test_load_external_label_conflict_drops_both(tmp_path):
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://x.example/a", "phishdb", ""),
            ("ham", "https://x.example/a", "commoncrawl", ""),
            ("ham", "https://y.example/b", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p)
    assert stats["n_dropped_label_conflict"] == 2
    assert stats["n_conflicting_urls"] == 1
    assert list(df["url"]) == ["y.example/b"]


def test_load_external_exact_duplicate_removed(tmp_path):
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://x.example/a", "phishdb", ""),
            ("spam", "https://x.example/a", "openphish", ""),
        ],
    )
    df, stats = ext.load_external(p)
    assert stats["n_dropped_duplicate"] == 1
    assert len(df) == 1


def test_load_external_multiple_paths(tmp_path):
    a = _write(tmp_path / "a.csv", [("spam", "http://p.example/x", "phishdb", "")])
    b = _write(tmp_path / "b.csv", [("ham", "http://b.example/y", "commoncrawl", "")])
    df, stats = ext.load_external([a, b])
    assert stats["n_final"] == 2
    assert stats["by_source"] == {"phishdb": 1, "commoncrawl": 1}


def test_load_external_drops_hosting_blocklist_from_benign(tmp_path):
    p = _write(
        tmp_path / "e.csv",
        [
            ("ham", "http://foo.000webhostapp.com/a", "commoncrawl", ""),
            ("ham", "http://ok.example/a", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p)
    assert stats["n_dropped_hosting_blocklist"] == 1
    assert list(df["url"]) == ["ok.example/a"]


def test_load_external_drops_phish_domains_from_benign(tmp_path):
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://shared.example/evil", "phishdb", ""),
            ("ham", "http://shared.example/nice", "commoncrawl", ""),
            ("ham", "http://clean.example/nice", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p)
    assert stats["n_dropped_phish_domain_from_benign"] == 1
    assert set(df["url"]) == {"shared.example/evil", "clean.example/nice"}


# ---------------------------------------------------------------- dedup 양방향


def _webphish_csv(tmp_path, rows):
    p = tmp_path / "wp.csv"
    pd.DataFrame(rows, columns=["Category", "Data"]).to_csv(p, index=False)
    return p


def test_dedup_url_only_removes_exact(tmp_path):
    wp = _webphish_csv(tmp_path, [("spam", "shared.example/a"), ("ham", "good.example")])
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://shared.example/a", "phishdb", ""),  # 정확 일치
            ("spam", "http://shared.example/b", "phishdb", ""),  # 도메인만 일치
            ("ham", "http://new.example/c", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p, dedup="url", webphish_csv=wp)
    assert stats["n_overlap_exact_url_vs_webphish"] == 1
    assert stats["n_overlap_etld1_vs_webphish"] == 2
    assert stats["n_dropped_vs_webphish"] == 1
    assert set(df["url"]) == {"shared.example/b", "new.example/c"}


def test_dedup_etld1_removes_domain(tmp_path):
    wp = _webphish_csv(tmp_path, [("spam", "shared.example/a"), ("ham", "good.example")])
    p = _write(
        tmp_path / "e.csv",
        [
            ("spam", "http://shared.example/b", "phishdb", ""),
            ("ham", "http://www.good.example/x", "commoncrawl", ""),
            ("ham", "http://new.example/c", "commoncrawl", ""),
        ],
    )
    df, stats = ext.load_external(p, dedup="etld1", webphish_csv=wp)
    assert stats["n_dropped_vs_webphish"] == 2
    assert list(df["url"]) == ["new.example/c"]


def test_dedup_against_reports_both_directions(tmp_path):
    wp, _ = load_webphish(_webphish_csv(tmp_path, [("spam", "a.example/x")]))
    ext_df = pd.DataFrame(
        {"url": ["a.example/y", "b.example/z"], "group": ["a.example", "b.example"]}
    )
    kept, stats = ext.dedup_against(ext_df, wp, dedup="etld1")
    assert stats["n_webphish_groups_seen_in_external"] == 1
    assert list(kept["url"]) == ["b.example/z"]


# ---------------------------------------------------------------- 편향 진단


def _diag_frame(benign_paths, phish_paths):
    rows = []
    for i, d in enumerate(benign_paths):
        url = f"b{i}.example" + ("/x" * d)
        rows.append((url, 0, f"b{i}.example", d))
    for i, d in enumerate(phish_paths):
        url = f"p{i}.example" + ("/y" * d)
        rows.append((url, 1, f"p{i}.example", d))
    df = pd.DataFrame(rows, columns=["url", "label", "group", "path_depth"])
    df["url_len"] = df["url"].str.len()
    df["url_bytes_len"] = [len(u.encode()) for u in df["url"]]
    return df


def test_bias_diagnostics_gate_passes_when_benign_has_paths():
    df = _diag_frame([1, 2, 1, 3], [1, 0, 1, 1])
    d = ext.bias_diagnostics(df)
    assert d["benign"]["path_depth_ge1_frac"] == 1.0
    assert d["phishing"]["path_depth_ge1_frac"] == 0.75
    assert d["gate_passed"] is True


def test_bias_diagnostics_gate_fails_on_webphish_like_bias():
    df = _diag_frame([0, 0, 0, 1], [2, 3, 1, 2])
    d = ext.bias_diagnostics(df)
    assert d["gate_passed"] is False
    assert "경고" in d["gate_note"]


def test_bias_diagnostics_reports_length_and_groups():
    df = _diag_frame([1, 1], [1, 1])
    d = ext.bias_diagnostics(df)
    assert set(d["benign"]["url_bytes_len"]) == {"mean", "p10", "median", "p90", "max"}
    assert d["benign"]["n_groups"] == 2
    assert 0.0 <= d["benign"]["top5_group_frac"] <= 1.0
    assert d["benign"]["scheme_frac"] == 0.0


def test_bias_diagnostics_empty_class():
    df = _diag_frame([1, 2], [])
    d = ext.bias_diagnostics(df)
    assert d["gate_passed"] is False


# ---------------------------------------------------------------- urls.py 위임


def test_urls_load_external_delegates(tmp_path):
    p = _write(tmp_path / "e.csv", [("spam", "http://a.example/x", "phishdb", "")])
    df, stats = urls_load_external(p)
    assert stats["n_final"] == 1
    assert list(df["url"]) == ["a.example/x"]


# ---------------------------------------------------------------- fetch 스모크 (monkeypatch)


class _FakeResp:
    def __init__(self, content=b"", payload=None, headers=None):
        self.content = content
        self.text = content.decode("utf-8", "replace")
        self._payload = payload
        self.headers = headers or {}
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    def close(self):
        return None


def test_fetch_phishdb_smoke(tmp_path, monkeypatch):
    def fake_get(session, url, **kw):
        if "api.github.com" in url:
            return _FakeResp(
                payload=[{"sha": "abc123", "commit": {"committer": {"date": "2025-12-22T20:39:14Z"}}}]
            )
        return _FakeResp(b"http://a.com/x\nhttp://b.com/y\n")

    monkeypatch.setattr(ext, "_get", fake_get)
    meta = ext.fetch_phishdb(tmp_path, session=object())
    assert meta["n_raw"] == 2
    assert meta["pinned_id"] == "abc123"
    assert len(meta["sha256"]) == 64
    assert "아카이브" in meta["freshness_note"]


def test_fetch_openphish_history_accumulates(tmp_path, monkeypatch):
    commits = [
        {"sha": "s2", "commit": {"committer": {"date": "2026-09-02T00:00:00Z"}}},
        {"sha": "s1", "commit": {"committer": {"date": "2026-09-01T00:00:00Z"}}},
    ]
    feeds = {"s1": b"http://a.com/1\n", "s2": b"http://a.com/1\nhttp://b.com/2\n"}

    def fake_get(session, url, **kw):
        if "api.github.com" in url:
            return _FakeResp(payload=commits)
        ref = url.split("/public_feed/")[1].split("/")[0]
        return _FakeResp(feeds[ref])

    monkeypatch.setattr(ext, "_get", fake_get)
    meta = ext.fetch_openphish_history(tmp_path, days=90, session=object(), sleep=0.0)
    assert meta["n_raw"] == 2
    assert meta["n_commits_fetched"] == 2
    text = (tmp_path / f"openphish_history_{ext.COLLECT_DATE}.txt").read_text()
    # 가장 이른 커밋 날짜가 first_seen이 되어야 한다.
    assert "2026-09-01T00:00:00Z\thttp://a.com/1" in text
    assert "2026-09-02T00:00:00Z\thttp://b.com/2" in text


def test_fetch_tranco_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ext, "_get", lambda s, u, **kw: _FakeResp(b"1,google.com\n2,cloudflare.com\n")
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    meta = ext.fetch_tranco(raw_dir, session=object(), n_domains=1)
    assert meta["n_raw"] == 2
    assert meta["pinned_id"] == ext.TRANCO_LIST_ID
    assert pd.read_csv(meta["parsed_path"])["url"].tolist() == ["google.com"]


def test_sources_registry_shape():
    assert set(ext.SOURCES) == {
        "phishdb",
        "openphish",
        "commoncrawl",
        "commoncrawl_unranked",
        "tranco",
    }
    assert ext.SOURCES["commoncrawl_unranked"].label == 0
    assert ext.SOURCES["phishdb"].label == 1
    assert ext.SOURCES["commoncrawl"].label == 0
    assert "MIT" in ext.SOURCES["phishdb"].license_note


# ---------------------------------------------------------------- 실제 네트워크 (opt-in)


@NET
def test_net_phishdb_head(tmp_path):
    meta = ext.fetch_phishdb(tmp_path)
    assert meta["n_raw"] > 100_000


@NET
def test_net_tranco(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    meta = ext.fetch_tranco(raw, n_domains=100)
    assert meta["n_raw"] == 1_000_000


# ------------------------------------------- benign에서 피싱 도메인 제거(선택 편향 옵션)
def _phish_domain_csv(tmp_path):
    return _write(
        tmp_path / "pd.csv",
        [
            ("spam", "http://mixed.example/login.php", "phishdb", ""),
            ("ham", "http://mixed.example/about/us", "commoncrawl", ""),
            ("ham", "http://clean.example/docs/a", "commoncrawl", ""),
        ],
    )


def test_drop_benign_on_phish_domains_default_on(tmp_path):
    df, stats = ext.load_external(_phish_domain_csv(tmp_path))
    assert stats["drop_benign_on_phish_domains"] is True
    assert stats["n_benign_on_phish_domains"] == 1
    assert stats["n_dropped_phish_domain_from_benign"] == 1
    assert set(df.loc[df["label"] == 0, "group"]) == {"clean.example"}


def test_drop_benign_on_phish_domains_can_be_disabled(tmp_path):
    """민감도 분석용으로 끌 수 있고, 건수는 끈 상태에서도 기록된다."""
    df, stats = ext.load_external(
        _phish_domain_csv(tmp_path), drop_benign_on_phish_domains=False
    )
    assert stats["drop_benign_on_phish_domains"] is False
    assert stats["n_benign_on_phish_domains"] == 1
    assert stats["n_dropped_phish_domain_from_benign"] == 0
    assert set(df.loc[df["label"] == 0, "group"]) == {"clean.example", "mixed.example"}


def test_external_nested_scheme_url_keeps_group(tmp_path):
    """쿼리 안에 URL을 품은 benign이 ``<empty>`` 그룹으로 뭉치면 안 된다(회귀)."""
    p = _write(
        tmp_path / "n.csv",
        [
            ("ham", "http://good.example/redirect?url=https://evil.example", "commoncrawl", ""),
            ("spam", "http://bad.example/a/b", "phishdb", ""),
        ],
    )
    df, _ = ext.load_external(p)
    assert "<empty>" not in set(df["group"])
    assert set(df.loc[df["label"] == 0, "group"]) == {"good.example"}


# --------------------------------------------------- benign 정제 절제 3조건 (리뷰 02)
def _cleaning_csv(tmp_path):
    return _write(
        tmp_path / "clean.csv",
        [
            ("spam", "http://shared.example/evil", "openphish", ""),
            ("ham", "http://shared.example/nice", "commoncrawl", ""),
            ("ham", "http://foo.000webhostapp.com/a", "commoncrawl", ""),
            ("ham", "http://clean.example/nice", "commoncrawl", ""),
        ],
    )


def test_benign_cleaning_clean_drops_both(tmp_path):
    df, stats = ext.load_external(_cleaning_csv(tmp_path), benign_cleaning="clean")
    assert stats["benign_cleaning"] == "clean"
    assert set(df["url"]) == {"shared.example/evil", "clean.example/nice"}


def test_benign_cleaning_keep_phish_domains(tmp_path):
    df, stats = ext.load_external(
        _cleaning_csv(tmp_path), benign_cleaning="keep_phish_domains"
    )
    assert stats["n_dropped_phish_domain_from_benign"] == 0
    assert stats["n_benign_on_phish_domains"] == 1  # 건수는 조건과 무관하게 남는다
    assert stats["n_dropped_hosting_blocklist"] == 1
    assert "shared.example/nice" in set(df["url"])


def test_benign_cleaning_no_hosting_blocklist(tmp_path):
    df, stats = ext.load_external(
        _cleaning_csv(tmp_path), benign_cleaning="no_hosting_blocklist"
    )
    assert stats["n_dropped_hosting_blocklist"] == 0
    assert stats["drop_hosting_from_benign"] is False
    assert "foo.000webhostapp.com/a" in set(df["url"])
    assert "shared.example/nice" not in set(df["url"])  # 이쪽 정제는 그대로 적용


def test_benign_cleaning_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="benign_cleaning"):
        ext.load_external(_cleaning_csv(tmp_path), benign_cleaning="nope")


def test_benign_cleaning_modes_are_the_three_required_conditions():
    assert ext.BENIGN_CLEANING_MODES == (
        "clean",
        "keep_phish_domains",
        "no_hosting_blocklist",
    )


# ------------------------------------------- primary/secondary는 같은 benign 행 집합
def test_split_by_source_keeps_benign_identical(tmp_path):
    """phishing 소스가 달라도 benign 행 집합은 같아야 한다 (리뷰 02 — OpenPhish primary)."""
    p = _write(
        tmp_path / "u.csv",
        [
            ("spam", "http://op.example/a", "openphish", ""),
            ("spam", "http://db.example/b", "phishdb", ""),
            ("ham", "http://benign1.example/x", "commoncrawl", ""),
            ("ham", "http://benign2.example/y", "commoncrawl", ""),
        ],
    )
    union, _ = ext.load_external(p)
    primary, st_p = ext.split_by_source(union, ["openphish", "commoncrawl"])
    secondary, st_s = ext.split_by_source(union, ["phishdb", "commoncrawl"])
    ben_p = set(primary.loc[primary["label"] == 0, "url"])
    ben_s = set(secondary.loc[secondary["label"] == 0, "url"])
    assert ben_p == ben_s == {"benign1.example/x", "benign2.example/y"}
    assert set(primary.loc[primary["label"] == 1, "url"]) == {"op.example/a"}
    assert set(secondary.loc[secondary["label"] == 1, "url"]) == {"db.example/b"}
    assert st_p["n_final"] == st_s["n_final"] == 3
    assert "gate_passed" in st_p["bias_diagnostics"]
