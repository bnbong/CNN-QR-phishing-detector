"""지표와 클러스터 부트스트랩 테스트 (스펙 1.4-5, 9절 #14)."""

from __future__ import annotations

import numpy as np
import pytest

from qrphish.evaluate import (
    acc_at,
    auroc,
    bootstrap_p_value,
    cluster_bootstrap,
    f1_at,
    group_perf_iqr,
    holm,
    metrics_from_probs,
    paired_cluster_bootstrap,
    percentile_ci,
    pick_threshold,
    unpaired_delta_bootstrap,
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
