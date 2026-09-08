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
    "cluster_bootstrap_by_seed",
    "precompute_cluster_resamples",
    "bootstrap_with_resamples",
    "group_bootstrap",
    "percentile_ci",
    "paired_cluster_bootstrap",
    "paired_cluster_bootstrap_by_seed",
    "unpaired_delta_bootstrap",
    "unpaired_delta_bootstrap_by_seed",
    "bootstrap_p_value",
    "paired_cluster_permutation_test",
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
    """부트스트랩 **백분위 CI 역전**으로 정의한 단측 **pseudo-p**.

    산출물에서는 이 값을 ``pseudo_p``(Holm 보정본은 ``pseudo_p_holm``)로 부른다.
    정식 영가설 분포에서 나온 p값이 아니라는 것을 이름에서부터 드러내기 위해서다.

    양측 백분위 CI ``[q(alpha/2), q(1-alpha/2)]``가 ``null``을 배제하는 가장 작은
    ``alpha``를 p값으로 쓴다. ``alternative="greater"``라면 하한이 null을 넘어야 하므로
    ``q(alpha/2) > null`` ⟺ ``alpha > 2 * F(null)``이 되고, 따라서
    ``p = 2 * (1 + #{반증 방향}) / (1 + n_valid)``다(``+1``은 p=0을 피하는 평활).
    이 정의 아래에서 "p < 0.05"와 "95% 양측 백분위 CI가 0을 배제한다"는 같은 판정이다.

    .. warning::
       이것은 영가설 아래에서 재표집한 정식 검정 통계량의 p값이 아니라, 관측치를
       중심으로 한 부트스트랩 분포에서 CI를 역전시킨 값이다. 명목 수준을 정확히 지키지
       않을 수 있으므로 절대적인 유의 판정보다는 CI와 함께 방향·크기를 읽는 데 쓴다.
       판정 자체는 CI가 0을 배제하는지로 하고, Holm 보정은 이 p값에 건다. 쌍체 예측이
       있는 비교라면 :func:`paired_cluster_permutation_test`의 순열 p값을 함께 본다.
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


def paired_cluster_permutation_test(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 2000,
    *,
    seeds: np.ndarray | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    alternative: str = "greater",
    seed: int = 0,
) -> dict:
    """클러스터 인식 순열 검정 — 같은 행을 공유하는 두 예측의 Δmetric에 대한 **정식 p값**.

    영가설은 "두 조건의 라벨(a/b)이 교환 가능하다"다. 부트스트랩 CI 역전 pseudo-p와 달리
    영가설 분포를 실제로 만들어 낸다. 표본이 eTLD+1 그룹 안에서 상관되어 있으므로 교환은
    **행 단위가 아니라 그룹 단위**로 한다: 각 순열에서 그룹마다 동전을 던져, 앞면이면 그
    그룹의 모든 행에서 ``p_a``와 ``p_b``를 통째로 맞바꾼다.

    Args:
        y: 0/1 라벨. ``p_a``/``p_b``: 같은 행에 대한 두 조건의 예측.
        groups: 클러스터 키(eTLD+1 등). ``seeds``: 주면 지표를 **시드별로 계산해 평균**한다
            (:func:`paired_cluster_bootstrap_by_seed`와 같은 정의). 시드마다 점수 척도가
            다를 수 있으므로 여러 시드를 풀링했다면 넘기는 편이 맞다.
        n_perm: 순열 수. ``alternative``: ``"greater"``(Δ>0) / ``"less"`` / ``"two-sided"``.

    Returns:
        ``{"estimate", "perm_p", "n_perm", "n_valid", "alternative", "null_sd"}``.
        같은 ``seed``에서 결정적이다.
    """
    if metric_fn is None:
        metric_fn = auroc
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    groups = np.asarray(groups)
    if not (y.size == p_a.size == p_b.size == groups.size):
        raise ValueError("paired_cluster_permutation_test: y/p_a/p_b/groups 길이가 다르다")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"alternative는 'greater'|'less'|'two-sided' (got {alternative!r})")

    if seeds is None:
        blocks = [np.arange(y.size, dtype=np.int64)]
    else:
        sarr = np.asarray(seeds)
        if sarr.size != y.size:
            raise ValueError("paired_cluster_permutation_test: seeds 길이가 y와 다르다")
        blocks = [np.flatnonzero(sarr == u) for u in np.unique(sarr)]

    _, ginv = np.unique(groups, return_inverse=True)
    ginv = ginv.astype(np.int64)
    n_g = int(ginv.max()) + 1 if ginv.size else 0

    def delta(pa: np.ndarray, pb: np.ndarray) -> float:
        vals = [
            metric_fn(y[b], pa[b]) - metric_fn(y[b], pb[b]) for b in blocks if b.size
        ]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    obs = delta(p_a, p_b)
    rng = np.random.default_rng(seed)
    stat = np.empty(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        if n_g == 0:
            stat[i] = float("nan")
            continue
        swap = rng.integers(0, 2, size=n_g).astype(bool)[ginv]
        pa = np.where(swap, p_b, p_a)
        pb = np.where(swap, p_a, p_b)
        stat[i] = delta(pa, pb)
    valid = stat[~np.isnan(stat)]
    if valid.size == 0 or np.isnan(obs):
        p = float("nan")
    elif alternative == "greater":
        p = float((1 + np.sum(valid >= obs)) / (1 + valid.size))
    elif alternative == "less":
        p = float((1 + np.sum(valid <= obs)) / (1 + valid.size))
    else:
        p = float((1 + np.sum(np.abs(valid) >= abs(obs))) / (1 + valid.size))
    return {
        "estimate": obs,
        "perm_p": p,
        "n_perm": int(n_perm),
        "n_valid": int(valid.size),
        "alternative": alternative,
        "null_sd": float(np.std(valid, ddof=0)) if valid.size else float("nan"),
    }


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


def precompute_cluster_resamples(
    groups: np.ndarray, n_boot: int = 200, seed: int = 0
) -> list[np.ndarray]:
    """그룹 단위 복원추출 인덱스를 **한 번만** 만들어 재사용하도록 돌려준다.

    :func:`cluster_bootstrap`은 호출될 때마다 같은 규약으로 리샘플 인덱스를 다시
    만든다. 같은 test 행·같은 그룹 배열 위에서 지표만 바꿔 수십·수백 번 부트스트랩할
    때(프로브 스위트: 목표 x {실제, 셔플, 무작위 초기화}) 이 인덱스 생성이 비용의
    대부분을 차지한다. 인덱스를 공유해도 각 부트스트랩 표본은 여전히 그룹 단위
    복원추출이고, 조건 간에 **같은 리샘플**을 쓰므로 오히려 비교가 쌍체가 된다.

    리샘플 규약은 :func:`cluster_bootstrap`과 동일하다
    (``rng.integers(0, n_g, size=n_g)`` 후 그룹 슬라이스 연결).
    """
    per_group = _group_slices(groups)
    n_g = len(per_group)
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    empty = np.array([], dtype=np.int64)
    for _ in range(int(n_boot)):
        if n_g == 0:
            out.append(empty)
            continue
        pick = rng.integers(0, n_g, size=n_g)
        out.append(np.concatenate([per_group[g] for g in pick]))
    return out


def _auroc_resampler(y: np.ndarray, p: np.ndarray) -> Callable[[np.ndarray], float]:
    """리샘플 인덱스 → AUROC. 전체 test에 대해 값 순위를 **한 번만** 계산해 재사용한다.

    ``roc_auc_score``를 리샘플마다 부르면 정렬·검증 오버헤드가 리샘플 수만큼 붙는다.
    대신 test 예측값을 한 번 유일값으로 묶어(``inv``) 리샘플마다 ``bincount``로
    값별 개수/양성 개수를 세고, 동점 평균 순위 공식으로 AUROC를 직접 만든다.
    동점 처리까지 ``roc_auc_score``와 같은 값을 낸다(테스트로 고정).
    """
    y = np.asarray(y).astype(np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    _, inv = np.unique(p, return_inverse=True)
    inv = inv.astype(np.int64)
    k = int(inv.max()) + 1 if inv.size else 0

    def fn(idx: np.ndarray) -> float:
        if idx.size == 0 or k == 0:
            return float("nan")
        codes = inv[idx]
        cnt = np.bincount(codes, minlength=k).astype(np.float64)
        pos = np.bincount(codes, weights=y[idx], minlength=k).astype(np.float64)
        n_pos = float(pos.sum())
        n_neg = float(cnt.sum() - n_pos)
        if n_pos <= 0 or n_neg <= 0:
            return float("nan")
        below = np.concatenate(([0.0], np.cumsum(cnt)[:-1]))
        rank_sum = float(np.sum(pos * (below + (cnt + 1.0) / 2.0)))
        return float((rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))

    return fn


def bootstrap_with_resamples(
    y: np.ndarray,
    p: np.ndarray,
    resamples,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    alpha: float = 0.05,
) -> dict:
    """미리 만들어 둔 리샘플 인덱스로 백분위 CI를 만든다.

    ``resamples``는 :func:`precompute_cluster_resamples`의 반환값(인덱스 배열의 목록)
    이거나 ``(flat, offsets)`` 형태로 이어 붙인 쌍이다(프로세스 간 전달 비용을 줄이려고
    프로브가 쓰는 형태). 결과 스키마는 :func:`cluster_bootstrap`과 같다.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if isinstance(resamples, tuple) and len(resamples) == 2:
        flat, offsets = resamples
        resamples = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    point = float(metric_fn(y, p))
    fast = _auroc_resampler(y, p) if metric_fn is auroc else None
    vals = np.empty(len(resamples), dtype=np.float64)
    for b, idx in enumerate(resamples):
        if idx.size == 0:
            vals[b] = float("nan")
        elif fast is not None:
            vals[b] = fast(idx)
        else:
            vals[b] = metric_fn(y[idx], p[idx])
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



# ------------------------------------------------------- 시드 층화 클러스터 부트스트랩
# 시드마다 **다른 모델**이 학습되므로 점수의 척도(로짓 오프셋·스케일)가 시드마다 다르다.
# 5시드의 test 예측을 (seed, row)로 이어 붙여 AUROC를 하나 계산하면 시드 간 순위가
# 섞이면서 값이 체계적으로 낮아진다(시드 평균 AUROC 0.877 vs 풀링 0.73 같은 괴리).
# 그래서 여기서는 **리샘플 단위는 그룹(eTLD+1), 지표는 시드별로 계산한 뒤 시드 평균**으로
# 정의한다. 같은 그룹이 여러 시드의 test에 걸쳐 있으면 함께 뽑힌다(그룹 id로 리샘플한 뒤
# 시드별로 분리). 점추정도 같은 정의(시드별 지표의 평균)라 표의 ``auroc_mean``과 일치한다.

_EMPTY_IDX = np.array([], dtype=np.int64)


def _mean_over_seeds(vals) -> float:
    """시드별 지표의 평균. 한 클래스만 남은 시드(NaN)는 빼고 평균낸다."""
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _seed_blocks(groups: np.ndarray, seeds: np.ndarray) -> tuple[int, list[dict]]:
    """(그룹 수, 시드별 블록). 블록은 그룹 코드 → 그 시드 안의 지역 인덱스를 들고 있다.

    그룹 코드는 **전 시드 공통**이다(그래야 한 번의 리샘플이 모든 시드에 같은 그룹
    집합을 적용한다). 시드에 없는 그룹은 ``code_map``에서 -1이라 자동으로 빠진다.
    """
    groups = np.asarray(groups)
    seeds = np.asarray(seeds)
    if groups.size != seeds.size:
        raise ValueError("groups와 seeds 길이가 다르다")
    _, gcode = np.unique(groups, return_inverse=True)
    n_codes = int(gcode.max()) + 1 if gcode.size else 0
    blocks: list[dict] = []
    for s in np.unique(seeds):
        rows = np.nonzero(seeds == s)[0]
        codes = gcode[rows]
        order = np.argsort(codes, kind="stable")
        csorted = codes[order]
        present = np.unique(csorted)
        starts = np.searchsorted(csorted, present, side="left")
        ends = np.searchsorted(csorted, present, side="right")
        code_map = np.full(n_codes, -1, dtype=np.int64)
        code_map[present] = np.arange(present.size)
        blocks.append(
            {
                "seed": s,
                "rows": rows,
                "slices": [order[a:b] for a, b in zip(starts, ends, strict=True)],
                "code_map": code_map,
            }
        )
    return n_codes, blocks


def _block_indices(block: dict, pick: np.ndarray) -> np.ndarray:
    """리샘플된 그룹 코드 → 그 시드 안의 지역 행 인덱스."""
    sel = block["code_map"][pick]
    sel = sel[sel >= 0]
    if sel.size == 0:
        return _EMPTY_IDX
    slices = block["slices"]
    return np.concatenate([slices[j] for j in sel])


def _block_metric_fns(y, p, blocks, metric_fn) -> list:
    """시드별 (지역 인덱스 → 지표) 콜러블. AUROC는 순위 재사용 고속 경로를 탄다."""
    fns = []
    for blk in blocks:
        rows = blk["rows"]
        ys, ps = y[rows], p[rows]
        if metric_fn is auroc:
            fns.append(_auroc_resampler(ys, ps))
        else:

            def fn(idx, ys=ys, ps=ps):
                return metric_fn(ys[idx], ps[idx]) if idx.size else float("nan")

            fns.append(fn)
    return fns


def cluster_bootstrap_by_seed(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    seeds: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """시드 층화 그룹 클러스터 부트스트랩.

    리샘플마다 그룹(eTLD+1)을 복원추출하되, 지표는 **시드별로 계산한 뒤 평균**낸다.
    점추정도 시드별 지표의 평균이라 ``auroc_mean``과 같은 정의다.

    반환: ``{"point", "lo", "hi", "n_valid", "samples", "per_seed", "seeds", "pooling"}``.
    동일 ``seed``에서 결정적이다.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    n_codes, blocks = _seed_blocks(groups, seeds)
    fns = _block_metric_fns(y, p, blocks, metric_fn)
    per_seed = [float(metric_fn(y[b["rows"]], p[b["rows"]])) for b in blocks]
    point = _mean_over_seeds(per_seed)

    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        if n_codes == 0:
            vals[b] = float("nan")
            continue
        pick = rng.integers(0, n_codes, size=n_codes)
        vals[b] = _mean_over_seeds(fn(_block_indices(blk, pick)) for blk, fn in zip(blocks, fns, strict=True))
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_valid": n_valid,
        "samples": vals,
        "per_seed": per_seed,
        "seeds": [int(b["seed"]) for b in blocks],
        "pooling": "seed_stratified",
    }


def paired_cluster_bootstrap_by_seed(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    seeds: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """같은 행을 공유하는 두 예측의 Δmetric(a − b), 시드 층화.

    시드별로 Δ를 낸 뒤 시드 평균을 취한다. 두 예측이 동일하면 Δ는 정확히 0이다.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    if not (y.size == p_a.size == p_b.size == np.asarray(groups).size == np.asarray(seeds).size):
        raise ValueError("paired_cluster_bootstrap_by_seed: y/p_a/p_b/groups/seeds 길이가 다르다")
    n_codes, blocks = _seed_blocks(groups, seeds)
    fns_a = _block_metric_fns(y, p_a, blocks, metric_fn)
    fns_b = _block_metric_fns(y, p_b, blocks, metric_fn)
    per_seed_a = [float(metric_fn(y[b["rows"]], p_a[b["rows"]])) for b in blocks]
    per_seed_b = [float(metric_fn(y[b["rows"]], p_b[b["rows"]])) for b in blocks]
    per_seed = [a - b for a, b in zip(per_seed_a, per_seed_b, strict=True)]
    point = _mean_over_seeds(per_seed)

    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        if n_codes == 0:
            vals[b] = float("nan")
            continue
        pick = rng.integers(0, n_codes, size=n_codes)
        deltas = []
        for blk, fa, fb in zip(blocks, fns_a, fns_b, strict=True):
            idx = _block_indices(blk, pick)
            deltas.append(fa(idx) - fb(idx))
        vals[b] = _mean_over_seeds(deltas)
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_valid": n_valid,
        "samples": vals,
        "paired": True,
        "per_seed": per_seed,
        "point_a": _mean_over_seeds(per_seed_a),
        "point_b": _mean_over_seeds(per_seed_b),
        "pooling": "seed_stratified",
    }


def unpaired_delta_bootstrap_by_seed(
    y_a: np.ndarray,
    p_a: np.ndarray,
    groups_a: np.ndarray,
    seeds_a: np.ndarray,
    y_b: np.ndarray,
    p_b: np.ndarray,
    groups_b: np.ndarray,
    seeds_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """표본 집합이 다른 두 조건의 Δmetric(a − b), 시드 층화. 두 조건을 독립 리샘플한다."""
    ra = cluster_bootstrap_by_seed(
        y_a, p_a, groups_a, seeds_a, metric_fn, n_boot=n_boot, seed=seed, alpha=alpha
    )
    rb = cluster_bootstrap_by_seed(
        y_b, p_b, groups_b, seeds_b, metric_fn, n_boot=n_boot, seed=seed + 10_000, alpha=alpha
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
        "pooling": "seed_stratified",
    }


def predict_probs(model, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, prob) 반환. loader는 ``(x, y)``를 낸다."""
    import torch

    device = device or next(model.parameters()).device
    # 방어적: 체크포인트를 CPU로 로드한 뒤 device만 넘기는 호출자가 있으면 입력/가중치
    # 장치가 어긋난다. 이미 같은 장치면 no-op이다.
    model.to(device)
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
