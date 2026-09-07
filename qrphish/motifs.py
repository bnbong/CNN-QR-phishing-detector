"""Bag-of-QR-patches: 국소 시각 motif를 직접 측정한다 (리뷰 review_01 "B. Visual motif 실험").

기존 H4(shuffle-pos)는 "원래 공간 배치가 CNN에 유용한가"만 보여줄 뿐, "여러 피싱 QR에
반복되는 구체적 motif가 있는가"(Q1)는 답하지 못한다. 여기서는 데이터 모듈 위를 stride 1
sliding window로 훑어 2x2/3x3 이진 패치의 출현 빈도 히스토그램을 만들고, 그 히스토그램만
으로 LR을 학습한다. 학습 없이 CPU에서 돌아가며, motif별 오즈비까지 낼 수 있어 Q1에 대한
직접 증거가 된다.

용어:
    patch id  창 안의 비트를 **row-major**로 읽은 이진수. 2x2 -> 0..15, 3x3 -> 0..511.
    valid     창 전체가 데이터 모듈(data_mask=1)인 경우에만 센다. 기능 패턴(파인더/타이밍
              /정렬)은 URL과 무관하게 고정이라 포함시키면 상수 motif가 히스토그램을 지배한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = [
    "n_motifs",
    "bits_string",
    "extract_patch_ids",
    "patch_histogram",
    "spatial_pyramid_histogram",
    "histogram_matrix",
    "bag_of_patches_lr",
    "group_bootstrap_weights",
    "motif_enrichment",
    "combine_seed_enrichments",
]

C_GRID = (0.01, 0.1, 1.0, 10.0)


def n_motifs(size: int) -> int:
    """``size x size`` 창에서 가능한 patch id 개수."""
    return 1 << (size * size)


def bits_string(motif_id: int, size: int) -> str:
    """patch id -> row-major 비트 문자열 (``"101010111"``)."""
    return format(int(motif_id), f"0{size * size}b")


def _as_bool_grid(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}은 정사각 2차원 배열이어야 한다 (got {arr.shape})")
    return arr.astype(bool)


def extract_patch_ids(
    grid_values: np.ndarray,
    data_mask: np.ndarray,
    size: int,
    *,
    unmask: bool = False,
    mask_pattern: int = 0,
    return_positions: bool = False,
):
    """데이터 모듈로만 이루어진 ``size x size`` 창을 stride 1로 훑어 patch id를 낸다.

    Args:
        grid_values: ``(n, n)`` 모듈 값(0/1). 기본은 마스크 고정(mask_pattern=0) 격자.
        data_mask: ``(n, n)`` 데이터 모듈 마스크. 창의 **모든** 칸이 1이어야 센다.
        size: 창 한 변 길이.
        unmask: True면 :func:`qrphish.mapping.unmask`로 마스크를 되돌린 격자를 쓴다
            (순수 비트 배치 motif). 기본 False = 고정 마스크 격자 그대로.
        mask_pattern: ``unmask=True``일 때 되돌릴 마스크 패턴.
        return_positions: True면 ``(ids, rows, cols)``를 낸다. ``rows/cols``는 창의
            **좌상단** 좌표다(피라미드 사분면 배정은 중심 = 좌상단 + size//2로 한다).

    Returns:
        ``ids`` (int32 1차원) 또는 ``(ids, rows, cols)``.
    """
    if size < 1:
        raise ValueError(f"size는 1 이상이어야 한다 (got {size})")
    vals = _as_bool_grid(grid_values, "grid_values")
    mask = _as_bool_grid(data_mask, "data_mask")
    if vals.shape != mask.shape:
        raise ValueError(f"grid_values{vals.shape}와 data_mask{mask.shape}의 모양이 다르다")
    if unmask:
        from qrphish.mapping import unmask as _unmask

        vals = np.asarray(_unmask(vals, int(mask_pattern)), dtype=bool)

    n = vals.shape[0]
    if n < size:
        empty = np.zeros(0, dtype=np.int32)
        return (empty, empty, empty) if return_positions else empty

    win_v = sliding_window_view(vals, (size, size)).reshape(n - size + 1, n - size + 1, -1)
    win_m = sliding_window_view(mask, (size, size)).reshape(n - size + 1, n - size + 1, -1)
    valid = win_m.all(axis=2)
    # row-major 비트열: 첫 칸이 최상위 비트
    weights = (1 << np.arange(size * size - 1, -1, -1)).astype(np.int64)
    ids = win_v.astype(np.int64) @ weights
    sel = np.nonzero(valid)
    out = ids[sel].astype(np.int32)
    if not return_positions:
        return out
    return out, sel[0].astype(np.int32), sel[1].astype(np.int32)


def patch_histogram(
    grid_values: np.ndarray,
    data_mask: np.ndarray,
    size: int,
    *,
    unmask: bool = False,
    mask_pattern: int = 0,
) -> np.ndarray:
    """QR 하나 -> ``2**(size*size)`` 차원 정규화 빈도 벡터(창 개수로 나눈다).

    유효 창이 하나도 없으면 영벡터를 낸다(합=0). 그 외에는 합=1이다.
    """
    ids = extract_patch_ids(
        grid_values, data_mask, size, unmask=unmask, mask_pattern=mask_pattern
    )
    hist = np.bincount(ids, minlength=n_motifs(size)).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def spatial_pyramid_histogram(
    grid_values: np.ndarray,
    data_mask: np.ndarray,
    size: int,
    *,
    unmask: bool = False,
    mask_pattern: int = 0,
) -> np.ndarray:
    """전역 히스토그램 + 2x2 사분면 히스토그램 4개를 이어 붙인다.

    차원은 ``5 * 2**(size*size)``. 사분면은 창의 **중심**(좌상단 + size//2) 좌표를
    격자 중앙선과 비교해 정한다(경계에 걸친 창도 한 사분면에만 들어간다). 각 블록은
    자기 블록의 창 개수로 나눠 정규화하므로, 창이 없는 사분면은 영벡터가 된다.
    """
    ids, rows, cols = extract_patch_ids(
        grid_values,
        data_mask,
        size,
        unmask=unmask,
        mask_pattern=mask_pattern,
        return_positions=True,
    )
    m = n_motifs(size)
    out = np.zeros(5 * m, dtype=np.float32)

    def _fill(block: int, sel_ids: np.ndarray) -> None:
        h = np.bincount(sel_ids, minlength=m).astype(np.float32)
        s = h.sum()
        if s > 0:
            h /= s
        out[block * m : (block + 1) * m] = h

    _fill(0, ids)
    n = np.asarray(grid_values).shape[0]
    half = n / 2.0
    cr = rows + size // 2
    cc = cols + size // 2
    top, left = cr < half, cc < half
    for block, quad in enumerate(
        (top & left, top & ~left, ~top & left, ~top & ~left), start=1
    ):
        _fill(block, ids[quad])
    return out


def histogram_matrix(
    grids: np.ndarray,
    data_masks: np.ndarray,
    size: int,
    *,
    pyramid: bool = False,
    unmask: bool = False,
    mask_patterns: np.ndarray | int = 0,
) -> np.ndarray:
    """``(N, n, n)`` 격자 묶음 -> ``(N, D)`` 히스토그램 행렬 (float32).

    ``data_masks``는 ``(N, n, n)`` 또는 층 전체 공통 ``(n, n)`` 둘 다 받는다.
    """
    grids = np.asarray(grids)
    if grids.ndim != 3:
        raise ValueError(f"grids는 (N, n, n)이어야 한다 (got {grids.shape})")
    masks = np.asarray(data_masks)
    shared = masks.ndim == 2
    mps = np.asarray(mask_patterns)
    fn = spatial_pyramid_histogram if pyramid else patch_histogram
    rows = [
        fn(
            grids[i],
            masks if shared else masks[i],
            size,
            unmask=unmask,
            mask_pattern=int(mps if mps.ndim == 0 else mps[i]),
        )
        for i in range(len(grids))
    ]
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)


def bag_of_patches_lr(
    X_hist_train: np.ndarray,
    y_train: np.ndarray,
    X_hist_test: np.ndarray,
    seed: int = 0,
    *,
    X_hist_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    C_grid: tuple[float, ...] = C_GRID,
) -> dict:
    """표준화 + LogisticRegression. C는 val AUROC로 고른다(val이 없으면 1.0 고정).

    Returns:
        ``{"p_val", "p_test", "C"}``. 확률은 클래스 1(phishing)의 확률이다.
        train에 한 클래스만 있으면 상수 확률을 낸다.
    """
    from qrphish.evaluate import auroc

    Xtr = np.asarray(X_hist_train, dtype=np.float64)
    ytr = np.asarray(y_train).astype(np.int64).reshape(-1)
    Xte = np.asarray(X_hist_test, dtype=np.float64)
    Xva = None if X_hist_val is None else np.asarray(X_hist_val, dtype=np.float64)

    if len(np.unique(ytr)) < 2:
        const = float(ytr.mean()) if ytr.size else 0.5
        return {
            "p_val": np.full(0 if Xva is None else Xva.shape[0], const),
            "p_test": np.full(Xte.shape[0], const),
            "C": float("nan"),
        }

    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zte = scaler.transform(Xte)
    Zva = None if Xva is None else scaler.transform(Xva)

    can_select = Zva is not None and y_val is not None and len(np.unique(np.asarray(y_val))) > 1
    grid = tuple(C_grid) if can_select else (1.0,)
    best = None
    for C in grid:
        clf = LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed
        )
        clf.fit(Ztr, ytr)
        pv = clf.predict_proba(Zva)[:, 1] if Zva is not None else np.zeros(0)
        score = auroc(np.asarray(y_val), pv) if can_select else 0.0
        if best is None or (score == score and score > best[0]):  # NaN은 후보에서 제외
            best = (score, C, clf, pv)
    assert best is not None
    _, C, clf, pv = best
    return {"p_val": pv, "p_test": clf.predict_proba(Zte)[:, 1], "C": float(C)}


# --------------------------------------------------------------------- enrichment
def group_bootstrap_weights(
    groups: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """그룹 클러스터 부트스트랩의 **리샘플 가중치** 행렬.

    :func:`qrphish.evaluate.cluster_bootstrap`과 동일한 난수 사용(``rng.integers(0, n_g,
    size=n_g)``)이라 같은 시드에서 같은 리샘플을 만든다. 다만 motif가 수백 개라 인덱스를
    매번 concat하는 대신, 그룹이 몇 번 뽑혔는지를 세어 그룹별 집계량에 곱한다(수학적으로
    동일하다). ``tests/test_motifs.py``가 두 경로의 동치를 검증한다.

    Returns:
        ``(group_index, W)`` — ``group_index``는 표본 -> 그룹 인덱스, ``W``는
        ``(n_boot, n_groups)`` 정수 가중치.
    """
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    n_g = int(inv.max()) + 1 if inv.size else 0
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, n_g), dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g) if n_g else np.zeros(0, dtype=np.int64)
        if n_g:
            W[b] = np.bincount(pick, minlength=n_g)
    return inv.astype(np.int64), W


def _group_counts(
    presence: np.ndarray, y: np.ndarray, gidx: np.ndarray, n_g: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """그룹 x motif 단위 2x2 표의 구성요소를 미리 집계한다."""
    m = presence.shape[1]
    a = np.zeros((n_g, m), dtype=np.float64)  # present & phishing
    c = np.zeros((n_g, m), dtype=np.float64)  # present & benign
    n1 = np.zeros(n_g, dtype=np.float64)
    n0 = np.zeros(n_g, dtype=np.float64)
    ph = y == 1
    np.add.at(a, gidx[ph], presence[ph].astype(np.float64))
    np.add.at(c, gidx[~ph], presence[~ph].astype(np.float64))
    np.add.at(n1, gidx[ph], 1.0)
    np.add.at(n0, gidx[~ph], 1.0)
    return a, c, n1, n0


def _log_odds(a: np.ndarray, c: np.ndarray, n1: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """Haldane-Anscombe(+0.5) 보정 log odds ratio. 0셀에서도 유한하다."""
    b = n1[..., None] - a
    d = n0[..., None] - c
    return np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))


def motif_enrichment(
    H: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    size: int,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """motif별 클래스 평균 빈도 + 존재율 오즈비 + 그룹 클러스터 부트스트랩 CI.

    Args:
        H: ``(N, 2**(size*size))`` 정규화 빈도 행렬. **train split만** 넣는다
            (발견은 train에서만 한다는 규약).
        y: 0/1 라벨, ``groups``: eTLD+1 등 클러스터 키.

    Returns:
        ``{"rate_phishing", "rate_benign", "log_odds", "ci_lo", "ci_hi",
        "present_phishing", "present_benign"}`` — 모두 motif 축 길이 배열.
    """
    H = np.asarray(H, dtype=np.float32)
    y = np.asarray(y).astype(np.int64).reshape(-1)
    m = n_motifs(size)
    if H.ndim != 2 or H.shape[1] != m:
        raise ValueError(f"H는 (N, {m}) 이어야 한다 (got {H.shape})")

    ph = y == 1
    rate_ph = H[ph].mean(axis=0) if ph.any() else np.zeros(m, dtype=np.float32)
    rate_bn = H[~ph].mean(axis=0) if (~ph).any() else np.zeros(m, dtype=np.float32)
    presence = (H > 0).astype(np.float64)

    gidx, W = group_bootstrap_weights(groups, n_boot=n_boot, seed=seed)
    n_g = W.shape[1]
    a, c, n1, n0 = _group_counts(presence, y, gidx, n_g)
    point = _log_odds(a.sum(0), c.sum(0), np.array(n1.sum()), np.array(n0.sum()))

    if n_g and n_boot:
        A = W @ a
        Cc = W @ c
        N1 = W @ n1
        N0 = W @ n0
        vals = _log_odds(A, Cc, N1, N0)  # (n_boot, m)
        lo = np.quantile(vals, alpha / 2, axis=0)
        hi = np.quantile(vals, 1 - alpha / 2, axis=0)
        # evaluate.cluster_bootstrap과 같은 관례: 점추정이 CI 밖이면 CI를 넓힌다.
        lo = np.minimum(lo, point)
        hi = np.maximum(hi, point)
    else:
        lo = hi = np.full(m, np.nan)

    return {
        "rate_phishing": rate_ph.astype(np.float64),
        "rate_benign": rate_bn.astype(np.float64),
        "log_odds": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "present_phishing": (presence[ph].mean(axis=0) if ph.any() else np.zeros(m)),
        "present_benign": (presence[~ph].mean(axis=0) if (~ph).any() else np.zeros(m)),
    }


def combine_seed_enrichments(
    per_seed_train: list[dict],
    per_seed_test_log_odds: list[np.ndarray],
    *,
    size: int,
    n_windows_mean: float,
    top_k: int = 20,
    min_rate: float = 0.0,
    pooled: dict | None = None,
    ci_n_boot: int | None = None,
    min_presence: float = 0.01,
) -> dict:
    """시드별 train enrichment를 합쳐 ``enrichment.json``의 본문을 만든다.

    - ``log_odds``/``rate_*``: 시드 평균.
    - ``ci``: ``pooled``가 주어지면 **5시드 train을 (seed, row)로 풀링해** 한 번 돌린
      그룹 클러스터 부트스트랩 CI다. 시드마다 split이 달라 시드별 CI 경계를 평균하는
      것은 정당한 절차가 아니므로 풀링한 CI 하나만 싣는다. 클러스터 키는 시드와 무관한
      eTLD+1 그룹이라 같은 그룹은 어느 시드에서 나왔든 함께 리샘플된다.
      ``pooled``가 없으면(단위 테스트 등) 시드 0의 CI로 되돌아간다.
    - ``seed_sign_agreement``: 시드별 train log OR 부호가 평균 부호와 같은 시드 수.
      전 시드가 일치(=시드 수)하면 "안정 motif"다.
    - ``test_sign_match``: held-out test에서 계산한 log OR 부호가 train 평균 부호와
      같은 시드가 과반인지 여부(검증은 test에서만 한다는 규약).
    """
    if not per_seed_train:
        raise ValueError("per_seed_train이 비어 있다")
    n_seeds = len(per_seed_train)
    m = n_motifs(size)
    lo_tr = np.stack([d["log_odds"] for d in per_seed_train])
    mean_lo = lo_tr.mean(axis=0)
    sign = np.sign(mean_lo)
    agree = (np.sign(lo_tr) == sign[None, :]).sum(axis=0)
    if per_seed_test_log_odds:
        lo_te = np.stack(per_seed_test_log_odds)
        test_match = (np.sign(lo_te) == sign[None, :]).sum(axis=0) * 2 > len(per_seed_test_log_odds)
    else:
        test_match = np.zeros(m, dtype=bool)

    rate_ph = np.stack([d["rate_phishing"] for d in per_seed_train]).mean(axis=0)
    rate_bn = np.stack([d["rate_benign"] for d in per_seed_train]).mean(axis=0)
    ci_src = pooled if pooled is not None else per_seed_train[0]
    ci_lo, ci_hi = ci_src["ci_lo"], ci_src["ci_hi"]

    seen = (rate_ph + rate_bn) > min_rate
    motifs: list[dict[str, Any]] = [
        {
            "id": int(i),
            "bits": bits_string(int(i), size),
            "rate_phishing": float(rate_ph[i]),
            "rate_benign": float(rate_bn[i]),
            "log_odds": float(mean_lo[i]),
            "ci": [float(ci_lo[i]), float(ci_hi[i])],
            "seed_sign_agreement": int(agree[i]),
            "test_sign_match": bool(test_match[i]),
        }
        for i in np.nonzero(seen)[0]
    ]
    pres_ph, pres_bn = ci_src["present_phishing"], ci_src["present_benign"]
    ubiq_mask = np.minimum(pres_ph, pres_bn) > 0.99
    for d in motifs:
        i = d["id"]
        d["present_phishing"] = float(pres_ph[i])
        d["present_benign"] = float(pres_bn[i])
        d["ci_excludes_zero"] = bool(d["ci"][0] > 0 or d["ci"][1] < 0)
        d["ubiquitous"] = bool(ubiq_mask[i])
    stable = [
        d
        for d in motifs
        if d["seed_sign_agreement"] == n_seeds
        and max(d["present_phishing"], d["present_benign"]) >= min_presence
        and d["ci_excludes_zero"]
        and not d["ubiquitous"]
    ]
    top_ph = [
        d["id"] for d in sorted(stable, key=lambda d: -float(d["log_odds"])) if d["log_odds"] > 0
    ]
    top_bn = [
        d["id"] for d in sorted(stable, key=lambda d: float(d["log_odds"])) if d["log_odds"] < 0
    ]
    # 2x2(16종)는 창이 수백 개라 사실상 모든 QR에 모든 motif가 한 번은 나온다.
    # 그러면 존재율 기반 오즈비가 포화되어 방향이 불안정해진다(=안정 motif 없음).
    # 이 진단값이 크면 top_* 목록이 비는 것이 정상이며, 빈도 기반 LR AUROC를 봐야 한다.
    p0 = ci_src
    ubiq = int(
        np.sum(np.minimum(p0["present_phishing"], p0["present_benign"]) > 0.99)
    )
    return {
        "size": int(size),
        "n_seeds": int(n_seeds),
        "n_windows_mean": float(n_windows_mean),
        "n_motifs_ubiquitous": ubiq,
        "ci_source": "pooled_5seed_train" if pooled is not None else "seed0_train",
        "min_presence": float(min_presence),
        "n_stable_after_filter": len(stable),
        "ci_n_boot": (int(ci_n_boot) if ci_n_boot is not None else None),
        "motifs": motifs,
        "top_phishing": top_ph[:top_k],
        "top_benign": top_bn[:top_k],
    }
