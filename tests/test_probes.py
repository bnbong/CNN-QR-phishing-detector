"""probes — 임베딩 추출, 목표 생성, 셔플 기준선, 프로브 유의성(약한 검정)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qrphish.models import SmallCNN
from qrphish.probes import (
    KEYWORDS,
    PROBE_N_BOOT,
    build_targets,
    embed,
    fit_binary_probe,
    group_shuffle,
    r2_score_fn,
    run_probe_suite,
    standardize,
    top_char_ngrams,
)
from tests.test_explain import Setup, build_trained


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> Setup:
    return build_trained(tmp_path_factory.mktemp("probes"), n_per_class=40, epochs=2)


# ------------------------------------------------------------------- 임베딩
def test_embed_shape_and_matches_head_input(setup: Setup) -> None:
    """임베딩은 GAP 출력이므로 ``head``에 넣으면 forward 로짓과 같아야 한다."""
    z = embed(setup.model, setup.ds, np.arange(8))
    assert z.shape == (8, 128)
    with torch.no_grad():
        xs = torch.stack([setup.ds[i][0] for i in range(8)])
        logit = setup.model(xs).numpy()
        from_embed = (
            setup.model.head(torch.from_numpy(z).float()).squeeze(-1).numpy()
        )
    assert np.allclose(logit, from_embed, atol=1e-4)


def test_embed_rejects_non_conv() -> None:
    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Linear(1, 1)

    with pytest.raises(TypeError):
        embed(Dummy(), [], np.arange(0))


# --------------------------------------------------------------------- 목표
def test_top_char_ngrams_is_label_independent(setup: Setup) -> None:
    grams = top_char_ngrams(setup.urls, size=3, top=20)
    assert len(grams) == 20
    assert all(len(g) == 3 for g in grams)
    assert "htt" in grams
    # 결정적: 같은 입력이면 같은 목록
    assert grams == top_char_ngrams(setup.urls, size=3, top=20)


def test_build_targets_covers_spec(setup: Setup) -> None:
    grams = top_char_ngrams(setup.urls, size=3, top=5)
    targets = build_targets(setup.urls, setup.y, grams)
    names = {t.name for t in targets}
    for kw in KEYWORDS:
        assert f"keyword:{kw}" in names
    for cont in ("digit_ratio", "special_ratio", "dot_count", "path_depth", "subdomain_count"):
        assert cont in names
        t = next(t for t in targets if t.name == cont)
        assert t.kind == "continuous"
    for b in ("has_path", "has_query", "has_digit", "has_hyphen", "ip_host", "label:phishing"):
        assert b in names
    # 합성 데이터 설계상 phishing만 숫자를 가진다.
    hd = next(t for t in targets if t.name == "has_digit")
    assert np.array_equal(hd.values, setup.y)


def test_ip_host_target() -> None:
    urls = ["http://192.168.10.24/wp-admin/login.php", "http://example.com/a"]
    t = {x.name: x for x in build_targets(urls, np.array([1, 0]), [])}
    assert list(t["ip_host"].values) == [1, 0]
    assert list(t["has_query"].values) == [0, 0]
    assert list(t["path_depth"].values) == [2, 1]


# ----------------------------------------------------------------- 기준선/지표
def test_group_shuffle_preserves_length_and_breaks_link() -> None:
    groups = np.repeat(np.arange(10), 5)
    values = np.repeat(np.arange(10), 5)  # 그룹당 상수 목표
    out = group_shuffle(values, groups, seed=0)
    assert out.shape == values.shape
    # 그룹째 갈아끼우므로 그룹 안은 여전히 상수여야 한다.
    for g in range(10):
        assert len(set(out[groups == g])) == 1
    assert not np.array_equal(out, values)


def test_r2_score_fn() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pytest.approx(1.0) == r2_score_fn(y, y)
    assert r2_score_fn(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert np.isnan(r2_score_fn(np.ones(4), np.ones(4)))


def test_fit_binary_probe_constant_target() -> None:
    z = np.random.default_rng(0).normal(size=(20, 4))
    pred = fit_binary_probe(z[:10], np.zeros(10), z[10:])
    assert pred.shape == (10,)
    assert np.allclose(pred, 0.0)


# ------------------------------------------------------------------- 통합(약한 검정)
def test_probe_suite_digit_target_beats_shuffle(setup: Setup) -> None:
    """학습된 CNN 임베딩에서 '숫자 포함'이 셔플 기준선보다 잘 예측돼야 한다.

    합성 데이터에서 숫자 포함 = phishing 라벨이라 학습된 임베딩이 이 목표를 선형으로
    담고 있어야 한다. 소표본이므로 유의성이 아니라 **점추정 우위**만 요구하는
    약한 검정이다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    grams = top_char_ngrams(setup.urls[tr], size=3, top=3)
    targets = [
        t for t in build_targets(setup.urls, setup.y, grams)
        if t.name in ("has_digit", "digit_ratio", "label:phishing")
    ]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(123)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    res = run_probe_suite(
        z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
        seed=0, n_boot=100, min_positive=5,
    )
    hd = res["has_digit"]
    assert "skipped" not in hd
    assert hd["kind"] == "binary"
    assert hd["score"] > hd["shuffle"]
    assert set(hd) >= {"score", "ci", "shuffle", "shuffle_ci", "random_init", "significant"}
    assert res["digit_ratio"]["kind"] == "continuous"


def test_probe_suite_skips_degenerate_targets(setup: Setup) -> None:
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = [t for t in build_targets(setup.urls, setup.y, []) if t.name == "ip_host"]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    res = run_probe_suite(z_tr, z_te, z_tr, z_te, targets, tr, te, setup.groups, n_boot=10)
    # 합성 URL에 IP 호스트가 없으므로 건너뛰어야 한다.
    assert "skipped" in res["ip_host"]


def test_probe_significance_requires_both_baselines(setup: Setup) -> None:
    """유의 = CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한) (코덱스 리뷰 3).

    학습된 임베딩 자리에 **무작위 초기화 임베딩을 그대로** 넣으면 두 쪽이 같으므로
    어떤 목표도 무작위 초기화 기준선을 넘을 수 없고, 따라서 유의로 서면 안 된다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = [t for t in build_targets(setup.urls, setup.y, []) if t.name == "has_digit"]
    torch.manual_seed(321)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    res = run_probe_suite(
        zr_tr, zr_te, zr_tr, zr_te, targets, tr, te, setup.groups,
        seed=0, n_boot=100, min_positive=5,
    )
    row = res["has_digit"]
    assert "random_init_ci" in row
    assert row["significant"] is False
    assert row["above_random_init"] is False


def test_probe_ngram_universe_is_fixed_across_seeds(setup: Setup) -> None:
    """목표 유니버스는 시드와 무관하게 같아야 한다 (코덱스 리뷰 3).

    ``top_char_ngrams``는 라벨을 보지 않고 빈도·사전순으로만 정렬하므로, 같은 URL
    집합에서는 몇 번을 불러도 같은 목록을 낸다. 러너는 이 목록을 층에서 한 번만 만들어
    모든 시드에 넘긴다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    a = top_char_ngrams(setup.urls[tr], size=3, top=10)
    b = top_char_ngrams(setup.urls[tr], size=3, top=10)
    assert a == b and len(a) == len(set(a))


# ------------------------------------------------------- 속도 최적화 경로(의미 보존)
def test_standardize_matches_pipeline_scaler(setup: Setup) -> None:
    """시드당 한 번 적합한 스케일러가 파이프라인 안 StandardScaler와 같은 값을 낸다."""
    from sklearn.preprocessing import StandardScaler

    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    a_tr, a_te = standardize(z_tr, z_te)
    sc = StandardScaler().fit(z_tr)
    assert np.allclose(a_tr, sc.transform(z_tr))
    assert np.allclose(a_te, sc.transform(z_te))


def test_fit_binary_probe_scale_flag_equivalent(setup: Setup) -> None:
    """미리 표준화한 뒤 ``scale=False``로 적합해도 파이프라인과 같은 결정값(순위)이 나온다."""
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    t = setup.y[tr]
    a = fit_binary_probe(z_tr, t, z_te)
    s_tr, s_te = standardize(z_tr, z_te)
    b = fit_binary_probe(s_tr, t, s_te, scale=False)
    assert np.allclose(a, b, atol=1e-8)


def test_probe_n_boot_default_is_200() -> None:
    """부트스트랩 기본 반복 수는 200이다(결과 JSON에도 이 값이 기록된다)."""
    assert PROBE_N_BOOT == 200


def test_probe_suite_parallel_matches_serial(setup: Setup) -> None:
    """joblib 병렬 경로는 직렬 경로와 **같은 결과**를 낸다(결정성).

    시드는 목표 인덱스에서 파생하므로 실행 순서·워커 수와 무관하다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    grams = top_char_ngrams(setup.urls[tr], size=3, top=6)
    targets = build_targets(setup.urls, setup.y, grams)
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(7)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    kw = dict(seed=0, n_boot=40, min_positive=5)
    a = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                        n_jobs=1, **kw)
    b = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                        n_jobs=2, **kw)
    assert set(a) == set(b)
    for name, row in a.items():
        assert row == b[name]


def test_probe_significance_rule_is_unchanged(setup: Setup) -> None:
    """유의 = CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한) — 저장된 값으로 재검산."""
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = build_targets(setup.urls, setup.y, top_char_ngrams(setup.urls[tr], size=3, top=4))
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(11)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)
    res = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                          seed=0, n_boot=40, min_positive=5, n_jobs=1)
    checked = 0
    for row in res.values():
        if "skipped" in row:
            continue
        bars = [v for v in (row["shuffle_ci"][1], row["random_init_ci"][1]) if np.isfinite(v)]
        bar = max(bars) if bars else float("nan")
        expect = bool(np.isfinite(row["ci"][0]) and np.isfinite(bar) and row["ci"][0] > bar)
        assert row["significant"] is expect
        assert row["above_random_init"] is bool(
            np.isfinite(row["random_init_ci"][1]) and row["ci"][0] > row["random_init_ci"][1]
        )
        checked += 1
    assert checked > 0
