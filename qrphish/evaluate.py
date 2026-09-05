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

    반환: ``{"point", "lo", "hi", "n_valid", "samples"}``. 동일 시드에서 결정적이다.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups)
    point = float(metric_fn(y, p))

    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    # 그룹별 인덱스 슬라이스(부트스트랩 루프에서 재계산하지 않도록 미리 구성)
    starts = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="right")
    per_group = [order[s:e] for s, e in zip(starts, ends, strict=True)]

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    n_g = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([per_group[g] for g in pick]) if n_g else np.array([], dtype=int)
        vals[b] = metric_fn(y[idx], p[idx]) if idx.size else float("nan")

    valid = vals[~np.isnan(vals)]
    if valid.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_valid": 0, "samples": vals}
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    # 점추정이 CI 밖으로 나가면(치우친 소표본) CI를 넓혀 점추정을 포함시킨다.
    lo = min(lo, point)
    hi = max(hi, point)
    return {"point": point, "lo": lo, "hi": hi, "n_valid": int(valid.size), "samples": vals}


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
