"""지표와 클러스터 부트스트랩 테스트 (스펙 1.4-5, 9절 #14)."""

from __future__ import annotations

import numpy as np
import pytest

from qrphish.evaluate import (
    acc_at,
    auroc,
    cluster_bootstrap,
    f1_at,
    group_perf_iqr,
    metrics_from_probs,
    pick_threshold,
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
