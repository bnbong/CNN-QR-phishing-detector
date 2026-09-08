"""templates.py 테스트 — 설계 5절(G, 템플릿 단위 분할)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrphish.config import from_dict, load_config
from qrphish.templates import (
    UnionFind,
    combined_group_key,
    is_clusterable_template,
    lsh_candidates,
    minhash_signature,
    minhash_signatures,
    template_diagnostics,
    template_groups,
    template_shingles,
    url_template,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- 골격 치환 규칙 --------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # 설계 5.2의 예시
        (
            "http://secure-login.example.com/wp-admin/user/verify2.php?id=8a3f1c&ref=44",
            "/wp-admin/user/verify#.php?id&ref",
        ),
        # 순수 숫자 세그먼트 -> "#"
        ("http://a.com/news/12345", "/news/#"),
        # 16진 8자 이상 -> "H" (숫자 규칙보다 우선순위가 앞선다)
        ("http://a.com/0dfa1b53b835500696e9a1f02cd1f464/", "/H"),
        # 숫자만으로 이루어진 16진은 "#"이 이긴다(순수 숫자 규칙이 먼저다)
        ("http://a.com/12345678", "/#"),
        # 난수스러운 영숫자(길이>=12, 대소문자 혼합) -> "R"
        ("http://a.com/AbCdEfGhIjKlMn", "/R"),
        # 16진 문자만으로 이루어진 12자는 "H"가 먼저 잡는다
        ("http://a.com/ab12cd34ef56", "/H"),
        # 난수스러운 영숫자(길이>=12, 숫자 비율 >= 0.3) -> "R"
        ("http://a.com/ab12xy34zz56", "/R"),
        # 확장자는 보존한다
        ("http://a.com/login.php", "/login.php"),
        ("http://a.com/0dfa1b53b835500696e9a1f02cd1f464.html", "/H.html"),
        # 세그먼트 내부의 숫자런/16진런만 치환
        ("http://a.com/user12/step3.html", "/user#/step#.html"),
        # 쿼리는 키 집합만 정렬해서 붙인다(값은 버린다)
        ("http://a.com/x?z=1&a=2&z=3", "/x?a&z"),
        # 호스트는 골격에 들어가지 않는다 — 도메인이 달라도 같은 골격
        ("http://other.co.uk/x?a=9", "/x?a"),
        # 대소문자 정규화
        ("http://a.com/Login.PHP", "/login.php"),
        # 스킴이 없어도 동작한다(WebPhish 원본 형태)
        ("a.com/wp-admin/login.php", "/wp-admin/login.php"),
    ],
)
def test_url_template_rules(url: str, expected: str) -> None:
    assert url_template(url) == expected


def test_url_template_empty_for_bare_domain() -> None:
    """path_depth == 0이면 골격은 빈 문자열이다 (설계 5.2의 최대 함정)."""
    for u in ("http://example.com", "example.com", "example.com/", "http://example.com/?a=1"):
        assert url_template(u) == ""


def test_bare_domains_never_share_a_cluster() -> None:
    """경로 없는 URL은 클러스터링에서 제외되어 각자 단독 클러스터여야 한다.

    이 처리를 빠뜨리면 WebPhish benign 절반이 한 그룹이 되어 group_split의
    5% 상한 다운샘플이 폭발한다.
    """
    urls = [f"http://site{i}.com" for i in range(50)]
    groups = template_groups(urls)
    assert len(set(groups)) == 50


# --- shingle / minhash ------------------------------------------------------
def test_template_shingles_tokenization() -> None:
    assert template_shingles("/wp-admin/user/verify#.php?id&ref", k=3) == {
        "wp-admin\x00user\x00verify#",
        "user\x00verify#\x00php",
        "verify#\x00php\x00id",
        "php\x00id\x00ref",
    }


def test_template_shingles_short_template_kept_whole() -> None:
    assert template_shingles("/login.php", k=3) == {"login\x00php"}
    assert template_shingles("", k=3) == set()


def test_minhash_signature_deterministic_and_jaccard_like() -> None:
    a = template_shingles("/a/b/c/d/e/f", k=2)
    b = template_shingles("/a/b/c/d/e/g", k=2)
    s1 = minhash_signature(a, n_perm=256, seed=0)
    s2 = minhash_signature(a, n_perm=256, seed=0)
    np.testing.assert_array_equal(s1, s2)
    sb = minhash_signature(b, n_perm=256, seed=0)
    est = float(np.mean(s1 == sb))
    true = len(a & b) / len(a | b)
    assert abs(est - true) < 0.15
    # 동일 집합은 서명도 동일 -> 추정 Jaccard 1.0
    assert float(np.mean(s1 == minhash_signature(a, n_perm=256, seed=0))) == 1.0


def test_minhash_signatures_matches_single() -> None:
    sets = [template_shingles("/a/b/c", 2), template_shingles("/x/y/z/w", 2), set()]
    batch = minhash_signatures(sets, n_perm=64, seed=3)
    for i, sh in enumerate(sets):
        np.testing.assert_array_equal(batch[i], minhash_signature(sh, n_perm=64, seed=3))


def test_lsh_candidates_finds_identical_and_validates_args() -> None:
    sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [5, 6, 7, 8]], dtype=np.int64)
    assert lsh_candidates(sig, bands=2, rows=2) == {(0, 1)}
    assert lsh_candidates(sig, bands=1, rows=4) == set()
    with pytest.raises(ValueError):
        lsh_candidates(sig, bands=3, rows=2)  # bands*rows > 서명 길이


# --- 클러스터링 -------------------------------------------------------------
def test_same_template_across_domains_clusters_together() -> None:
    urls = [f"http://kit{i}.com/wp-admin/secure/login.php?id={i}" for i in range(8)]
    groups = template_groups(urls)
    assert len(set(groups)) == 1


def test_different_templates_stay_separate() -> None:
    urls = [
        "http://a.com/wp-admin/secure/login.php?id=1",
        "http://b.com/totally/other/path/structure/here/deep",
        "http://c.com/news/sports/soccer/report",
    ]
    groups = template_groups(urls)
    assert len(set(groups)) == 3


def test_near_duplicate_templates_merge_under_threshold() -> None:
    """토큰 하나만 다른 긴 골격은 Jaccard가 높아 같은 클러스터가 된다."""
    base = "http://{h}.com/wp/admin/secure/account/verify/step/{tail}"
    urls = [base.format(h="a", tail="one"), base.format(h="b", tail="two")]
    merged = template_groups(urls, threshold=0.5)
    assert len(set(merged)) == 1
    # 임계값을 1.0으로 올리면 정확 일치만 묶이므로 분리된다.
    strict = template_groups(urls, threshold=1.0)
    assert len(set(strict)) == 2


# --- 과병합 게이트 (설계 5.2.1) ---------------------------------------------
def test_generic_skeletons_are_not_clusterable() -> None:
    """``/index.html`` 같은 범용 골격은 캠페인이 아니다."""
    for tpl in ("", "/index.html", "/index.htm", "/about", "/contact.php"):
        assert not is_clusterable_template(tpl)


def test_structural_skeletons_stay_clusterable() -> None:
    """자리표시자·쿼리 키·토큰 3개는 각각 단독으로 통과 조건이다."""
    assert is_clusterable_template("/html/rfc#")  # 자리표시자
    assert is_clusterable_template("/login.php?id")  # 쿼리 키
    assert is_clusterable_template("/wp/admin/login.php")  # 토큰 4개
    # min_tokens를 1로 낮추면 게이트가 사실상 꺼진다.
    assert is_clusterable_template("/about", min_tokens=1)
    with pytest.raises(ValueError):
        is_clusterable_template("/about", min_tokens=0)


def test_index_html_does_not_merge_unrelated_domains() -> None:
    """게이트가 없으면 이 8개가 한 클러스터가 된다(v2 실측 276건의 축소판)."""
    urls = [f"http://unrelated{i}.com/index.html" for i in range(8)]
    assert len(set(template_groups(urls))) == 8
    # 게이트를 끄면 (min_template_tokens=1) 예전처럼 한 덩어리다.
    assert len(set(template_groups(urls, min_template_tokens=1))) == 1


def test_placeholder_skeleton_still_clusters_across_domains() -> None:
    """``/html/rfc#``처럼 자리표시자를 가진 짧은 골격은 게이트 뒤에도 묶인다."""
    urls = [f"http://mirror{i}.example/html/rfc{2000 + i}" for i in range(6)]
    assert len(set(template_groups(urls))) == 1


def test_template_groups_deterministic_and_order_invariant() -> None:
    urls = [
        "http://a.com/wp-admin/login.php?id=1",
        "http://b.com/wp-admin/login.php?id=2",
        "http://c.com",
        "http://d.com/news/12",
        "http://e.com/news/99",
    ]
    g1 = template_groups(urls)
    g2 = template_groups(urls)
    np.testing.assert_array_equal(g1, g2)
    # 순서를 뒤집어도 같은 분할(id 문자열까지 동일)이어야 한다.
    rev = template_groups(urls[::-1])
    assert [g for g in rev][::-1] == list(g1)


def test_template_groups_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        template_groups(["http://a.com/x"], threshold=0.0)
    assert template_groups([]).size == 0


# --- eTLD+1 합집합 ----------------------------------------------------------
def test_combined_group_key_unions_domain_and_template() -> None:
    etld1 = ["a.com", "b.com", "b.com", "c.com"]
    tpl = ["t1", "t1", "t2", "t3"]
    g = combined_group_key(etld1, tpl)
    # a.com—b.com은 템플릿 t1로, b.com 두 행은 도메인으로 이어져 한 성분이 된다.
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]


def test_combined_group_key_is_deterministic_regardless_of_order() -> None:
    etld1 = ["z.com", "a.com", "a.com", "m.com"]
    tpl = ["t1", "t1", "t2", "t2"]
    g = combined_group_key(etld1, tpl)
    assert len(set(g)) == 1
    # 성분 id는 성분 내 사전순 최소 eTLD+1에서 나온다 -> 입력 순서와 무관
    assert set(g) == {"g:a.com"}
    idx = [3, 0, 2, 1]
    g2 = combined_group_key([etld1[i] for i in idx], [tpl[i] for i in idx])
    assert set(g2) == set(g)


def test_combined_group_key_length_mismatch() -> None:
    with pytest.raises(ValueError):
        combined_group_key(["a.com"], ["t1", "t2"])


def test_union_find_components_stable() -> None:
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(5, 4)
    uf.union(1, 2)
    comps = uf.components()
    assert comps[0] == comps[1] == comps[2]
    assert comps[4] == comps[5]
    assert len({int(c) for c in comps}) == 3


def test_template_diagnostics_shape() -> None:
    groups = ["a", "a", "a", "b", "c"]
    labels = [1, 1, 0, 0, 0]
    d = template_diagnostics(groups, labels)
    assert d["n_rows"] == 5
    assert d["n_clusters"] == 3
    assert d["n_singletons"] == 2
    assert d["largest_cluster_frac"] == pytest.approx(0.6)
    assert d["cluster_size_quantiles"]["max"] == 3
    assert d["n_mixed_class_clusters"] == 1
    assert template_diagnostics([])["n_clusters"] == 0
    with pytest.raises(ValueError):
        template_diagnostics(groups, [0, 1])


# --- config 하위 호환 -------------------------------------------------------
def test_base_config_defaults_group_key() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert cfg.split.group_key == "etld1"
    assert cfg.split.template_threshold == 0.7
    assert cfg.split.template_shingle == 3


def test_config_validates_template_fields() -> None:
    ok = from_dict({"split": {"group_key": "etld1_template", "template_threshold": 0.8}})
    assert ok.split.group_key == "etld1_template"
    with pytest.raises(ValueError):
        from_dict({"split": {"group_key": "template_only"}})
    with pytest.raises(ValueError):
        from_dict({"split": {"template_threshold": 0.0}})
    with pytest.raises(ValueError):
        from_dict({"split": {"template_shingle": 0}})
    with pytest.raises(ValueError):
        from_dict({"split": {"min_template_tokens": 0}})
    assert from_dict({"split": {"min_template_tokens": 4}}).split.min_template_tokens == 4


def test_matrix_has_template_split_entry() -> None:
    from qrphish.runner import apply_overrides as ao
    from qrphish.runner import condition_id, load_matrix

    matrix = load_matrix(REPO_ROOT / "configs" / "matrix.yaml")
    entries = [e for e in matrix["P2"] if e["name"] == "template_split"]
    assert len(entries) == 1
    entry = entries[0]
    cfg = ao(load_config(REPO_ROOT / "configs" / "base.yaml"), entry["overrides"])
    assert cfg.split.group_key == "etld1_template"
    # 1차 주 조건 아티팩트를 덮어쓰지 않도록 조건 id가 달라야 한다.
    assert condition_id(cfg, entry.get("extra")).endswith("-templatesplit")


# --- _prepare_frame 통합 ----------------------------------------------------
# 10개 키트 패밀리 × 20 도메인. 도메인은 다 다르지만 경로 골격은 10종뿐이다.
_KITS = [
    "wp-admin/secure/login",
    "panel/user/signin",
    "auth/step/verify",
    "secure/pay/confirm",
    "app/account/reset",
    "office/mail/inbox",
    "bank/portal/entry",
    "cloud/drive/open",
    "shop/order/pay",
    "id/profile/update",
]
_SECTIONS = [
    "articles",
    "stories",
    "posts",
    "topics",
    "guides",
    "reviews",
    "notes",
    "pages",
    "docs",
    "blog",
]


def _pad(spec: str, target: int) -> str:
    """``호스트|나머지`` 형태를 받아 호스트를 늘려 전체 길이를 target으로 맞춘다.

    L-exact 매칭은 바이트 길이가 정확히 같은 benign/phishing 쌍만 남기므로,
    합성 데이터에서도 두 클래스의 길이를 맞춰 두지 않으면 프레임이 비어버린다.
    """
    host, _, rest = spec.partition("|")
    return host + "x" * (target - (len(host) + len(rest))) + rest


def _tiny_csv(tmp_path: Path) -> Path:
    """소형 합성 데이터. 키트 골격을 여러 도메인에 복제해 템플릿 누출을 만든다.

    네 종류 모두 45바이트로 맞춰 v3 층에 들어가고 L-exact 매칭에서 살아남는다.
    benign 절반은 경로가 없는 맨 도메인이다(설계 5.2의 함정 재현).
    """
    rows = []
    for i in range(200):
        tag = f"{i:03d}"
        kit, sec = _KITS[i % 10], _SECTIONS[i % 10]
        rows.append(("spam", _pad(f"kit{tag}|.com/{kit}.php?id={tag}", 45)))
        rows.append(("ham", _pad(f"benign{tag}|.org", 45)))
        rows.append(("ham", _pad(f"news{tag}|.org/{sec}/{tag}/read.html", 45)))
    path = tmp_path / "tiny.csv"
    pd.DataFrame(rows, columns=["Category", "Data"]).to_csv(path, index=False)
    return path


def _tiny_cfg(tmp_path: Path, group_key: str):
    return from_dict(
        {
            "data": {"csv_path": str(_tiny_csv(tmp_path))},
            "split": {"group_key": group_key, "max_group_frac": 0.2},
            "strata": ["v3"],
        }
    )


@pytest.mark.parametrize("group_key", ["etld1", "etld1_template"])
def test_prepare_frame_groups_disjoint_for_both_keys(tmp_path: Path, group_key: str) -> None:
    """test_splits.py의 test_group_disjoint 유형 — 그룹 키를 바꿔도 교집합은 0이다."""
    from qrphish.runner import _prepare_frame

    cfg = _tiny_cfg(tmp_path, group_key)
    df, diag = _prepare_frame(cfg, "v3", seed=0)
    assert diag["group_key"] == group_key
    assert len(df) > 0
    sets = {name: set(df.loc[df["split"] == name, "group"]) for name in ("train", "val", "test")}
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["val"]
    assert not sets["val"] & sets["test"]
    assert diag["split"]["groups_disjoint"] is True


def test_prepare_frame_template_key_merges_kit_domains(tmp_path: Path) -> None:
    """같은 키트 템플릿을 쓰는 서로 다른 도메인이 한 그룹으로 합쳐진다."""
    from qrphish.runner import _prepare_frame

    df_e, _ = _prepare_frame(_tiny_cfg(tmp_path, "etld1"), "v3", seed=0)
    df_t, diag_t = _prepare_frame(_tiny_cfg(tmp_path, "etld1_template"), "v3", seed=0)

    assert "group_etld1" not in df_e.columns
    # 원래 eTLD+1 키는 보존된다
    assert "group_etld1" in df_t.columns
    assert df_t["group"].nunique() < df_t["group_etld1"].nunique()
    tdiag = diag_t["template"]
    assert tdiag["n_groups_after_union"] < tdiag["n_groups_etld1"]
    assert tdiag["threshold"] == 0.7
    assert tdiag["clusters"]["n_clusters"] >= 1
    # 경로 없는 benign은 단독 클러스터로 남아야 한다(거대 그룹이 생기면 안 된다)
    assert tdiag["clusters"]["largest_cluster_frac"] < 0.5


def test_run_template_split_writes_diagnostics(tmp_path: Path) -> None:
    from qrphish.runner import run_template_split

    cfg_raw = {
        "data": {"csv_path": str(_tiny_csv(tmp_path))},
        "seed_list": [0, 1],
        "strata": ["v3"],
        "split": {"max_group_frac": 0.2},
        "reports_dir": str(tmp_path / "reports"),
        "output_dir": str(tmp_path / "artifacts"),
    }
    summary = run_template_split(from_dict(cfg_raw), strata=["v3"])
    entry = summary["strata"]["v3"]
    assert entry["n_seeds_ok"] == 2
    assert 0.0 <= entry["mean"]["test_set_jaccard"] <= 1.0
    assert entry["template"]["n_groups_after_union"] > 0
    written = tmp_path / "reports" / "template_split" / "v3" / "diagnostics.json"
    assert written.exists()
    assert (tmp_path / "reports" / "template_split" / "diagnostics.json").exists()
    for row in entry["per_seed"]:
        assert row["etld1"]["groups_disjoint"] and row["etld1_template"]["groups_disjoint"]
        assert row["etld1"]["length_match_ok"] and row["etld1_template"]["length_match_ok"]
