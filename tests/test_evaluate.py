"""지표와 클러스터 부트스트랩 테스트 (스펙 1.4-5, 9절 #14)."""

from __future__ import annotations

import numpy as np
import pytest

from qrphish.evaluate import (
    acc_at,
    auroc,
    bootstrap_p_value,
    bootstrap_with_resamples,
    cluster_bootstrap,
    cluster_bootstrap_by_seed,
    f1_at,
    group_perf_iqr,
    holm,
    metrics_from_probs,
    paired_cluster_bootstrap,
    paired_cluster_bootstrap_by_seed,
    percentile_ci,
    pick_threshold,
    precompute_cluster_resamples,
    unpaired_delta_bootstrap,
    unpaired_delta_bootstrap_by_seed,
)


def _toy(n_groups: int = 40, per_group: int = 25, signal: float = 1.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = rng.integers(0, 2, size=n_groups * per_group)
    p = rng.normal(0, 1, size=y.size) + signal * y
    return y, p, groups


def test_auroc_matches_known_ordering() -> None:
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_f1_and_acc_at_threshold() -> None:
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.4, 0.6])
    assert f1_at(y, p, 0.5) == pytest.approx(0.5)
    assert acc_at(y, p, 0.5) == pytest.approx(0.5)


def test_pick_threshold_maximizes_f1() -> None:
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    thr = pick_threshold(y, p)
    assert f1_at(y, p, thr) == pytest.approx(1.0)


def test_cluster_bootstrap_is_deterministic() -> None:
    y, p, g = _toy()
    a = cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=7)
    b = cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=7)
    assert a["lo"] == b["lo"] and a["hi"] == b["hi"]
    assert not np.array_equal(
        a["samples"], cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=8)["samples"]
    )


def test_cluster_bootstrap_ci_contains_point_estimate() -> None:
    y, p, g = _toy()
    r = cluster_bootstrap(y, p, g, auroc, n_boot=300, seed=0)
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["n_valid"] > 0


def test_cluster_bootstrap_wider_than_naive_iid() -> None:
    """그룹 내 상관이 강하면 클러스터 CI가 표본 부트스트랩보다 넓어야 한다."""
    rng = np.random.default_rng(3)
    n_groups, per_group = 20, 60
    gy = rng.integers(0, 2, size=n_groups)  # 라벨이 그룹 단위로 완전 상관
    y = np.repeat(gy, per_group)
    groups = np.repeat(np.arange(n_groups), per_group)
    p = rng.normal(0, 1, size=y.size) + 0.8 * y
    clustered = cluster_bootstrap(y, p, groups, auroc, n_boot=400, seed=0)
    iid = cluster_bootstrap(y, p, np.arange(y.size), auroc, n_boot=400, seed=0)
    assert (clustered["hi"] - clustered["lo"]) > (iid["hi"] - iid["lo"])


def test_label_shuffle_ci_contains_half() -> None:
    """라벨을 셔플하면 AUROC의 95% CI가 0.5를 포함해야 한다(스펙 9절 #14의 통계 부분)."""
    y, p, g = _toy(n_groups=30, per_group=20, signal=1.5, seed=1)
    rng = np.random.default_rng(123)
    y_shuf = y[rng.permutation(y.size)]
    r = cluster_bootstrap(y_shuf, p, g, auroc, n_boot=500, seed=0)
    assert r["lo"] <= 0.5 <= r["hi"]


def test_group_perf_iqr_zero_when_uniform() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.1, 0.9])
    g = np.array(["a", "a", "b", "b"])
    assert group_perf_iqr(y, p, g, 0.5) == pytest.approx(0.0)


def test_metrics_from_probs_schema() -> None:
    y, p, g = _toy(n_groups=10, per_group=10)
    p = 1 / (1 + np.exp(-p))
    m = metrics_from_probs(y, p, g, n_boot=50, seed=0, threshold=0.5)
    for k in ("auroc", "auprc", "f1", "acc", "auroc_ci", "f1_ci", "group_perf_iqr", "n"):
        assert k in m
    assert len(m["auroc_ci"]) == 2 and m["auroc_ci"][0] <= m["auroc_ci"][1]


def test_percentile_ci_is_not_widened_to_include_point() -> None:
    """순수 백분위 CI: 점추정이 CI 밖이어도 인위적으로 넓히지 않는다."""
    vals = np.linspace(0.0, 1.0, 1001)
    lo, hi, n = percentile_ci(vals, 0.05)
    assert n == 1001
    assert lo == pytest.approx(0.025, abs=1e-3)
    assert hi == pytest.approx(0.975, abs=1e-3)
    # 점추정이 2.0이더라도 hi는 그대로여야 한다.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    g = np.array(["a", "b", "c", "d"])
    r = cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=0)
    assert r["hi"] <= 1.0  # 강제 확장이 있었다면 point=1.0을 넣으려 hi를 밀었을 것


def test_pooled_bootstrap_is_deterministic() -> None:
    """여러 시드의 예측을 이어붙인 풀링 CI도 같은 시드에서 재현된다."""
    ys, ps, gs = [], [], []
    for s in range(3):
        y, p, g = _toy(n_groups=15, per_group=8, seed=s)
        ys.append(y)
        ps.append(p)
        gs.append(g)  # 그룹 id는 시드와 무관한 eTLD+1 키라 그대로 이어붙인다
    y, p, g = np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)
    a = cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=0)
    b = cluster_bootstrap(y, p, g, auroc, n_boot=200, seed=0)
    assert (a["lo"], a["hi"], a["point"]) == (b["lo"], b["hi"], b["point"])
    assert a["n_valid"] > 0


def test_paired_bootstrap_zero_delta_for_identical_predictions() -> None:
    y, p, g = _toy(n_groups=25, per_group=12, seed=2)
    r = paired_cluster_bootstrap(y, p, p.copy(), g, n_boot=300, seed=0)
    assert r["point"] == pytest.approx(0.0)
    assert r["lo"] <= 0.0 <= r["hi"]
    assert np.allclose(r["samples"][~np.isnan(r["samples"])], 0.0)
    assert bootstrap_p_value(r["samples"]) > 0.5


def test_paired_bootstrap_detects_better_predictor() -> None:
    y, p, g = _toy(n_groups=40, per_group=20, signal=1.5, seed=5)
    rng = np.random.default_rng(0)
    weak = p + rng.normal(0, 3.0, size=p.size)  # 잡음을 얹어 성능을 떨어뜨린다
    r = paired_cluster_bootstrap(y, p, weak, g, n_boot=300, seed=0)
    assert r["point"] > 0
    assert r["lo"] > 0
    assert bootstrap_p_value(r["samples"], alternative="greater") < 0.05


def test_paired_bootstrap_rejects_length_mismatch() -> None:
    y, p, g = _toy(n_groups=5, per_group=4)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(y, p, p[:-1], g, n_boot=10)


def test_unpaired_delta_bootstrap_is_wider_than_paired() -> None:
    y, p, g = _toy(n_groups=40, per_group=20, signal=1.5, seed=5)
    rng = np.random.default_rng(1)
    weak = p + rng.normal(0, 2.0, size=p.size)
    pr = paired_cluster_bootstrap(y, p, weak, g, n_boot=400, seed=0)
    ur = unpaired_delta_bootstrap(y, p, g, y, weak, g, n_boot=400, seed=0)
    assert (ur["hi"] - ur["lo"]) > (pr["hi"] - pr["lo"])
    assert ur["paired"] is False


def test_holm_known_example() -> None:
    # R: p.adjust(c(0.01, 0.02, 0.03, 0.04), method="holm")
    assert holm([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert holm([0.005]) == pytest.approx([0.005])
    assert holm([]) == []
    # 순서 무관성: 입력 순서를 바꿔도 각 p값의 조정 결과는 같다
    got = holm([0.04, 0.01, 0.03, 0.02])
    assert got == pytest.approx([0.06, 0.04, 0.06, 0.06])
    # 1.0으로 잘린다
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_bootstrap_p_value_bounds() -> None:
    # CI 역전 정의: p = 2 * (1 + #{v <= 0}) / (1 + n). 전부 양수면 2 * 1/4 = 0.5.
    assert bootstrap_p_value(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.5)
    assert bootstrap_p_value(np.array([-1.0, -2.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bootstrap_p_value(np.array([1.0]), alternative="two-sided")


def test_bootstrap_p_value_matches_ci_inversion() -> None:
    """p < alpha ⟺ (1-alpha) 양측 백분위 CI가 null을 배제한다 (코덱스 리뷰 2)."""
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.05, 0.2, -0.2):
        v = rng.normal(shift, 0.1, size=2000)
        p = bootstrap_p_value(v, alternative="greater")
        for alpha in (0.01, 0.05, 0.10):
            lo, _, _ = percentile_ci(v, alpha)
            assert (p < alpha) == (lo > 0.0), (shift, alpha, p, lo)


# ------------------------------------------- 사전 계산 리샘플(프로브 고속 경로)
def test_precompute_resamples_matches_cluster_bootstrap() -> None:
    """리샘플을 미리 뽑아 재사용해도 ``cluster_bootstrap``과 같은 CI가 나와야 한다.

    프로브 스위트는 목표마다 리샘플을 다시 뽑는 대신 시드·층당 한 번 뽑아 공유한다.
    같은 시드·같은 그룹 배열이면 두 경로의 리샘플 규약이 같으므로 결과도 같다.
    """
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(25), 8)
    y = rng.integers(0, 2, size=groups.size)
    p = y * 0.6 + rng.normal(size=groups.size) * 0.5

    ref = cluster_bootstrap(y, p, groups, auroc, n_boot=64, seed=7)
    rs = precompute_cluster_resamples(groups, n_boot=64, seed=7)
    got = bootstrap_with_resamples(y, p, rs, auroc)
    assert got["point"] == pytest.approx(ref["point"])
    assert got["lo"] == pytest.approx(ref["lo"])
    assert got["hi"] == pytest.approx(ref["hi"])
    assert np.allclose(got["samples"], ref["samples"], equal_nan=True)


def test_fast_auroc_path_matches_sklearn_with_ties() -> None:
    """AUROC 고속 경로(순위 공식)는 동점이 있어도 ``roc_auc_score``와 같은 값이다."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), 5)
    y = rng.integers(0, 2, size=groups.size)
    # 값을 일부러 굵게 반올림해 동점을 많이 만든다.
    p = np.round(rng.normal(size=groups.size), 1)
    rs = precompute_cluster_resamples(groups, n_boot=32, seed=1)
    fast = bootstrap_with_resamples(y, p, rs, auroc)["samples"]
    slow = np.array([auroc(y[i], p[i]) for i in rs])
    assert np.allclose(fast, slow, equal_nan=True)


def test_bootstrap_with_resamples_accepts_packed_form() -> None:
    """``(flat, offsets)``로 이어 붙인 형태도 목록과 같은 결과를 낸다(병렬 전달용)."""
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(10), 6)
    y = rng.integers(0, 2, size=groups.size)
    p = rng.normal(size=groups.size)
    rs = precompute_cluster_resamples(groups, n_boot=16, seed=2)
    flat = np.concatenate(rs)
    offsets = np.concatenate(([0], np.cumsum([r.size for r in rs])))
    a = bootstrap_with_resamples(y, p, rs, auroc)
    b = bootstrap_with_resamples(y, p, (flat, offsets), auroc)
    assert np.allclose(a["samples"], b["samples"], equal_nan=True)


# ------------------------------------------------- 시드 층화 클러스터 부트스트랩
def _two_seed_offset_preds():
    """순위는 같고 로짓 오프셋만 5 차이 나는 두 시드의 합성 test 예측."""
    rng = np.random.default_rng(11)
    n = 400
    y1 = np.repeat([0, 1], n // 2)
    score = rng.normal(size=n) + y1 * 1.5
    groups1 = np.array([f"g{i % 40}" for i in range(n)])
    y = np.concatenate([y1, y1])
    p = np.concatenate([score, score + 5.0])  # 시드1은 척도만 밀린 같은 순위
    groups = np.concatenate([groups1, groups1])
    seeds = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return y, p, groups, seeds, y1, score


def test_seed_stratified_point_equals_mean_of_per_seed_auroc() -> None:
    """시드 간 로짓 오프셋이 있으면 옛 풀링은 AUROC가 떨어지고, 시드 층화는 시드 평균과 같다."""
    y, p, groups, seeds, y1, score = _two_seed_offset_preds()
    per_seed = [auroc(y1, score), auroc(y1, score + 5.0)]
    assert per_seed[0] == pytest.approx(per_seed[1])  # 순위가 같으니 시드별 AUROC도 같다

    old = auroc(y, p)  # (seed, row)를 그냥 이어 붙인 옛 방식
    new = cluster_bootstrap_by_seed(y, p, groups, seeds, auroc, n_boot=64, seed=0)

    assert new["point"] == pytest.approx(float(np.mean(per_seed)))
    assert old < new["point"] - 0.05  # 옛 방식은 순위가 섞여 체계적으로 낮다
    assert new["per_seed"] == pytest.approx(per_seed)
    assert new["pooling"] == "seed_stratified"
    # CI가 점추정 근처에 오고, 옛 풀링 값(허수)을 포함하지 않는다.
    assert new["lo"] <= new["point"] <= new["hi"]
    assert not (new["lo"] <= old <= new["hi"])


def test_seed_stratified_is_deterministic() -> None:
    y, p, groups, seeds, _, _ = _two_seed_offset_preds()
    a = cluster_bootstrap_by_seed(y, p, groups, seeds, auroc, n_boot=48, seed=3)
    b = cluster_bootstrap_by_seed(y, p, groups, seeds, auroc, n_boot=48, seed=3)
    assert np.allclose(a["samples"], b["samples"], equal_nan=True)
    assert a["lo"] == pytest.approx(b["lo"])


def test_seed_stratified_matches_cluster_bootstrap_with_single_seed() -> None:
    """시드가 하나면 시드 층화는 그냥 그룹 클러스터 부트스트랩과 같은 점추정을 낸다."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(20), 10)
    y = rng.integers(0, 2, size=groups.size)
    p = y * 0.5 + rng.normal(size=groups.size)
    seeds = np.zeros(groups.size, dtype=int)
    a = cluster_bootstrap(y, p, groups, auroc, n_boot=64, seed=1)
    b = cluster_bootstrap_by_seed(y, p, groups, seeds, auroc, n_boot=64, seed=1)
    assert b["point"] == pytest.approx(a["point"])
    assert np.allclose(b["samples"], a["samples"], equal_nan=True)


def test_paired_seed_stratified_zero_for_identical_predictions() -> None:
    """같은 예측 두 벌의 쌍체 Δ는 점추정도 CI도 정확히 0이다."""
    y, p, groups, seeds, _, _ = _two_seed_offset_preds()
    r = paired_cluster_bootstrap_by_seed(y, p, p, groups, seeds, auroc, n_boot=32, seed=0)
    assert r["point"] == pytest.approx(0.0)
    assert r["lo"] == pytest.approx(0.0)
    assert r["hi"] == pytest.approx(0.0)
    assert r["paired"] is True


def test_paired_seed_stratified_equals_mean_of_per_seed_delta() -> None:
    """쌍체 Δ의 점추정은 시드별 ΔAUROC의 평균이다(시드를 섞은 값이 아니다)."""
    y, p_a, groups, seeds, y1, score = _two_seed_offset_preds()
    rng = np.random.default_rng(12)
    p_b = p_a + rng.normal(scale=0.4, size=p_a.size)
    r = paired_cluster_bootstrap_by_seed(y, p_a, p_b, groups, seeds, auroc, n_boot=32, seed=0)
    expected = np.mean(
        [
            auroc(y[seeds == s], p_a[seeds == s]) - auroc(y[seeds == s], p_b[seeds == s])
            for s in (0, 1)
        ]
    )
    assert r["point"] == pytest.approx(float(expected))


def test_unpaired_seed_stratified_delta_of_seed_means() -> None:
    """비쌍체 Δ의 점추정은 두 조건의 시드 평균 AUROC 차이다."""
    y, p_a, groups, seeds, _, _ = _two_seed_offset_preds()
    rng = np.random.default_rng(13)
    p_b = rng.normal(size=p_a.size)
    r = unpaired_delta_bootstrap_by_seed(
        y, p_a, groups, seeds, y, p_b, groups, seeds, auroc, n_boot=32, seed=0
    )
    mean_a = np.mean([auroc(y[seeds == s], p_a[seeds == s]) for s in (0, 1)])
    mean_b = np.mean([auroc(y[seeds == s], p_b[seeds == s]) for s in (0, 1)])
    assert r["point"] == pytest.approx(float(mean_a - mean_b))
    assert r["paired"] is False
    assert r["pooling"] == "seed_stratified"


def test_seed_stratified_resamples_groups_across_seeds_together() -> None:
    """리샘플 단위는 그룹이다 — 한 그룹이 여러 시드에 걸쳐 있으면 함께 뽑힌다."""
    from qrphish.evaluate import _block_indices, _seed_blocks

    groups = np.array(["a", "a", "b", "b", "a", "b"])
    seeds = np.array([0, 0, 0, 0, 1, 1])
    n_codes, blocks = _seed_blocks(groups, seeds)
    assert n_codes == 2
    assert [int(b["seed"]) for b in blocks] == [0, 1]
    pick = np.array([0, 0])  # 그룹 "a"를 두 번
    assert sorted(blocks[0]["rows"][_block_indices(blocks[0], pick)]) == [0, 0, 1, 1]
    assert sorted(blocks[1]["rows"][_block_indices(blocks[1], pick)]) == [4, 4]


# ------------------------------ review_02 6: 클러스터 인식 순열 검정 (정식 p값)
def test_paired_cluster_permutation_test_detects_real_difference():
    from qrphish.evaluate import paired_cluster_permutation_test

    rng = np.random.default_rng(0)
    n_g, per_g = 40, 6
    groups = np.repeat([f"g{i}" for i in range(n_g)], per_g)
    y = rng.integers(0, 2, size=n_g * per_g)
    noise = rng.normal(0, 1, size=y.size)
    p_a = 1 / (1 + np.exp(-(2.5 * y + noise)))  # 강한 예측
    p_b = 1 / (1 + np.exp(-(0.2 * y + noise)))  # 약한 예측
    r = paired_cluster_permutation_test(y, p_a, p_b, groups, n_perm=500, seed=0)
    assert r["estimate"] > 0
    assert r["perm_p"] < 0.05
    assert r["n_valid"] == 500
    assert r["alternative"] == "greater"


def test_paired_cluster_permutation_test_null_is_not_significant():
    from qrphish.evaluate import paired_cluster_permutation_test

    rng = np.random.default_rng(1)
    groups = np.repeat([f"g{i}" for i in range(30)], 6)
    y = rng.integers(0, 2, size=groups.size)
    p = 1 / (1 + np.exp(-(1.0 * y + rng.normal(0, 1, size=y.size))))
    # 두 예측이 동일하면 Δ=0이고 순열 p는 크게 나온다.
    r = paired_cluster_permutation_test(y, p, p.copy(), groups, n_perm=300, seed=0)
    assert r["estimate"] == pytest.approx(0.0)
    assert r["perm_p"] > 0.5


def test_paired_cluster_permutation_test_is_deterministic():
    from qrphish.evaluate import paired_cluster_permutation_test

    rng = np.random.default_rng(2)
    groups = np.repeat([f"g{i}" for i in range(20)], 5)
    y = rng.integers(0, 2, size=groups.size)
    p_a = rng.random(y.size)
    p_b = rng.random(y.size)
    kw = dict(n_perm=200, seed=3)
    r1 = paired_cluster_permutation_test(y, p_a, p_b, groups, **kw)
    r2 = paired_cluster_permutation_test(y, p_a, p_b, groups, **kw)
    assert r1 == r2
    r3 = paired_cluster_permutation_test(y, p_a, p_b, groups, n_perm=200, seed=4)
    assert r3["estimate"] == r1["estimate"]  # 관측치는 시드와 무관


def test_paired_cluster_permutation_test_seed_stratified_and_validation():
    from qrphish.evaluate import paired_cluster_permutation_test

    rng = np.random.default_rng(5)
    groups = np.tile(np.repeat([f"g{i}" for i in range(20)], 4), 2)
    seeds = np.repeat([0, 1], groups.size // 2)
    y = rng.integers(0, 2, size=groups.size)
    p_a = 1 / (1 + np.exp(-(2.0 * y + rng.normal(0, 1, size=y.size))))
    p_b = rng.random(y.size)
    r = paired_cluster_permutation_test(
        y, p_a, p_b, groups, n_perm=300, seeds=seeds, seed=0
    )
    assert r["estimate"] > 0 and r["perm_p"] < 0.05
    with pytest.raises(ValueError):
        paired_cluster_permutation_test(y, p_a, p_b, groups[:-1], n_perm=10)
    with pytest.raises(ValueError):
        paired_cluster_permutation_test(y, p_a, p_b, groups, n_perm=10, alternative="x")
