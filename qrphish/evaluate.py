"""지표 계산과 그룹 클러스터 부트스트랩 CI(스펙 1.4-5, 1.8).

표본 단위 부트스트랩은 eTLD+1 그룹 내 상관 때문에 CI를 과소평가한다. 그래서 여기서는
**그룹을 리샘플링**한다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "auroc",
    "auprc",
    "f1_at",
    "acc_at",
    "cluster_bootstrap",
    "group_bootstrap",
    "percentile_ci",
    "paired_cluster_bootstrap",
    "unpaired_delta_bootstrap",
    "bootstrap_p_value",
    "holm",
    "pick_threshold",
    "group_perf_iqr",
    "predict_probs",
    "evaluate",
]


def _safe(fn: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, p: np.ndarray) -> float:
    """한 클래스만 남은 리샘플에서는 지표가 정의되지 않으므로 NaN으로 흘린다."""
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, p))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return _safe(roc_auc_score, y, p)


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    return _safe(average_precision_score, y, p)


def f1_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    yhat = (np.asarray(p) >= threshold).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    tp = float(((yhat == 1) & (y == 1)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def acc_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    if len(y) == 0:
        return float("nan")
    return float(((np.asarray(p) >= threshold).astype(np.int64) == np.asarray(y)).mean())


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """validation에서 F1을 최대화하는 임계값을 고른다(스펙 1.8)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    if p.size == 0:
        return 0.5
    cands = np.unique(np.concatenate([p, np.array([0.5])]))
    if cands.size > 512:
        cands = np.quantile(p, np.linspace(0.0, 1.0, 512))
        cands = np.unique(np.concatenate([cands, np.array([0.5])]))
    scores = [f1_at(y, p, float(t)) for t in cands]
    return float(cands[int(np.argmax(scores))])


def percentile_ci(vals: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """부트스트랩 표본의 순수 백분위 CI. 보정·강제 확장은 하지 않는다."""
    valid = np.asarray(vals, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return lo, hi, int(valid.size)


def _group_slices(groups: np.ndarray) -> list[np.ndarray]:
    """그룹 id → 그 그룹에 속한 행 인덱스 배열의 리스트."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    starts = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="right")
    return [order[s:e] for s, e in zip(starts, ends, strict=True)]


def bootstrap_p_value(samples: np.ndarray, alternative: str = "greater", null: float = 0.0) -> float:
    """부트스트랩 **백분위 CI 역전**으로 정의한 단측 p값.

    양측 백분위 CI ``[q(alpha/2), q(1-alpha/2)]``가 ``null``을 배제하는 가장 작은
    ``alpha``를 p값으로 쓴다. ``alternative="greater"``라면 하한이 null을 넘어야 하므로
    ``q(alpha/2) > null`` ⟺ ``alpha > 2 * F(null)``이 되고, 따라서
    ``p = 2 * (1 + #{반증 방향}) / (1 + n_valid)``다(``+1``은 p=0을 피하는 평활).
    이 정의 아래에서 "p < 0.05"와 "95% 양측 백분위 CI가 0을 배제한다"는 같은 판정이다.

    .. warning::
       이것은 영가설 아래에서 재표집한 정식 검정 통계량의 p값이 아니라, 관측치를
       중심으로 한 부트스트랩 분포에서 CI를 역전시킨 값이다. 명목 수준을 정확히 지키지
       않을 수 있으므로 절대적인 유의 판정보다는 CI와 함께 방향·크기를 읽는 데 쓴다.
       판정 자체는 CI가 0을 배제하는지로 하고, Holm 보정은 이 p값에 건다.
    """
    v = np.asarray(samples, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    if alternative == "greater":
        k = int(np.sum(v <= null))
    elif alternative == "less":
        k = int(np.sum(v >= null))
    else:
        raise ValueError(f"alternative는 'greater'|'less' (got {alternative!r})")
    return float(min(2.0 * (1 + k) / (1 + v.size), 1.0))


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni 조정 p값(입력 순서 유지, 단조 증가 보장)."""
    p = np.asarray(list(pvalues), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    return [float(x) for x in adj]


def cluster_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """그룹 단위 복원추출로 metric의 (1-alpha) 백분위 CI를 만든다.

    순수 백분위 CI다. 점추정이 CI 밖으로 나가더라도 인위적으로 넓히지 않는다.

    반환: ``{"point", "lo", "hi", "n_valid", "samples"}``. 동일 시드에서 결정적이다.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups)
    point = float(metric_fn(y, p))

    # 그룹별 인덱스 슬라이스(부트스트랩 루프에서 재계산하지 않도록 미리 구성)
    per_group = _group_slices(groups)

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    n_g = len(per_group)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([per_group[g] for g in pick]) if n_g else np.array([], dtype=int)
        vals[b] = metric_fn(y[idx], p[idx]) if idx.size else float("nan")

    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {"point": point, "lo": lo, "hi": hi, "n_valid": n_valid, "samples": vals}


def group_bootstrap(
    groups: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """행 인덱스를 그룹 단위로 복원추출해 **임의 통계량**의 백분위 CI를 만든다.

    ``stat_fn(idx) -> float``. :func:`cluster_bootstrap`은 ``metric_fn(y, p)`` 하나만
    다루므로, 예측 두 벌의 차이나 여러 조건을 동시에 흔드는 통계량은 여기로 온다.
    리샘플 규약(``rng.integers(0, n_g, size=n_g)``)은 :func:`cluster_bootstrap`과 같다.
    ``qrphish.probes``와 ``qrphish.occlusion``이 공유한다.
    """
    groups = np.asarray(groups)
    per_group = _group_slices(groups)
    n_g = len(per_group)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g) if n_g else np.array([], dtype=int)
        idx = np.concatenate([per_group[g] for g in pick]) if n_g else np.array([], dtype=int)
        vals[b] = stat_fn(idx) if idx.size else float("nan")
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {"lo": lo, "hi": hi, "n_valid": n_valid, "samples": vals}


def group_perf_iqr(y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float) -> float:
    """그룹별 정확도의 IQR(스펙 1.4-4). 한 그룹이 성능을 끌고 가는지 진단한다."""
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    accs = []
    for g in np.unique(groups):
        m = groups == g
        accs.append(acc_at(y[m], p[m], threshold))
    if len(accs) < 2:
        return 0.0
    a = np.asarray(accs, dtype=np.float64)
    return float(np.quantile(a, 0.75) - np.quantile(a, 0.25))


def paired_cluster_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """같은 표본에 대한 두 예측의 Δmetric(a − b)와 그룹 클러스터 부트스트랩 CI.

    두 조건이 **동일한 split의 동일한 행**을 공유할 때만 의미가 있다. 리샘플된 그룹을
    두 예측에 똑같이 적용하므로 표본 변동이 상쇄되어 비쌍체 비교보다 CI가 좁다.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    if not (y.size == p_a.size == p_b.size == np.asarray(groups).size):
        raise ValueError("paired_cluster_bootstrap: y/p_a/p_b/groups 길이가 다르다")
    point = float(metric_fn(y, p_a)) - float(metric_fn(y, p_b))

    per_group = _group_slices(groups)
    n_g = len(per_group)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([per_group[g] for g in pick]) if n_g else np.array([], dtype=int)
        if idx.size == 0:
            vals[b] = float("nan")
            continue
        vals[b] = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_valid": n_valid,
        "samples": vals,
        "paired": True,
    }


def unpaired_delta_bootstrap(
    y_a: np.ndarray,
    p_a: np.ndarray,
    groups_a: np.ndarray,
    y_b: np.ndarray,
    p_b: np.ndarray,
    groups_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """표본 집합이 다른 두 조건의 Δmetric(a − b).

    두 조건을 **독립적으로** 리샘플한 뒤 차이를 취한다. 표본이 겹치지 않으므로 쌍체
    비교보다 CI가 넓고, 개입(intervention) 하나만 다른 비교로 해석해서는 안 된다.
    """
    ra = cluster_bootstrap(y_a, p_a, groups_a, metric_fn, n_boot=n_boot, seed=seed, alpha=alpha)
    rb = cluster_bootstrap(
        y_b, p_b, groups_b, metric_fn, n_boot=n_boot, seed=seed + 10_000, alpha=alpha
    )
    vals = ra["samples"] - rb["samples"]
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {
        "point": ra["point"] - rb["point"],
        "lo": lo,
        "hi": hi,
        "n_valid": n_valid,
        "samples": vals,
        "paired": False,
        "point_a": ra["point"],
        "point_b": rb["point"],
    }


def predict_probs(model, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, prob) 반환. loader는 ``(x, y)``를 낸다."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logit = model(xb.to(device))
            ps.append(torch.sigmoid(logit.float()).cpu().numpy().reshape(-1))
            ys.append(np.asarray(yb).reshape(-1))
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(
    model,
    loader,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    threshold: float = 0.5,
    device=None,
) -> dict:
    """AUROC/AUPRC/F1/Acc + 그룹 클러스터 부트스트랩 95% CI.

    ``threshold``는 validation에서 선택된 값을 넘겨받는다(스펙 1.8).
    """
    y, p = predict_probs(model, loader, device=device)
    return metrics_from_probs(y, p, groups, n_boot=n_boot, seed=seed, threshold=threshold)


def metrics_from_probs(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict:
    """확률 배열에서 직접 지표를 낸다(베이스라인도 같은 경로를 쓴다)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups)
    auroc_ci = cluster_bootstrap(y, p, groups, auroc, n_boot=n_boot, seed=seed)
    f1_ci = cluster_bootstrap(
        y, p, groups, lambda a, b: f1_at(a, b, threshold), n_boot=n_boot, seed=seed + 1
    )
    return {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "f1": f1_at(y, p, threshold),
        "acc": acc_at(y, p, threshold),
        "threshold": float(threshold),
        "auroc_ci": [auroc_ci["lo"], auroc_ci["hi"]],
        "f1_ci": [f1_ci["lo"], f1_ci["hi"]],
        "group_perf_iqr": group_perf_iqr(y, p, groups, threshold),
        "n": int(y.size),
    }
