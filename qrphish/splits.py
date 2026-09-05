"""길이 매칭 서브샘플링과 eTLD+1 그룹 분할(스펙 1.1 / 1.4 / 4절).

두 함수 모두 시드가 같으면 바이트 단위로 동일한 결과를 낸다.
길이는 항상 UTF-8 바이트 길이(`url_bytes_len`) 기준이다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "match_by_length",
    "match_by_length_within_splits",
    "length_match_report",
    "assert_length_matched",
    "group_split",
    "LENGTH_COL",
]

LENGTH_COL = "url_bytes_len"
_SPLIT_NAMES = ("train", "val", "test")


def _require(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing} (있는 컬럼: {list(df.columns)})")


def match_by_length(df: pd.DataFrame, bucket: int | None, seed: int) -> pd.DataFrame:
    """클래스 간 길이 주변분포를 맞추는 서브샘플링.

    bucket=1 -> `L-exact`, 5 -> `L-quantile`, None -> `L-none`(항등).
    각 버킷에서 `min(n_benign, n_phishing)`만큼 양쪽에서 시드 고정 샘플링한다.
    """
    _require(df, (LENGTH_COL, "label"))
    if bucket is None:
        return df.reset_index(drop=True).copy()
    if bucket < 1:
        raise ValueError(f"bucket은 1 이상이거나 None이어야 한다 (got {bucket})")

    work = df.reset_index(drop=True)
    key = (work[LENGTH_COL].to_numpy(dtype=np.int64) // bucket).astype(np.int64)
    labels = work["label"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)

    picked: list[np.ndarray] = []
    for b in np.unique(key):  # np.unique는 정렬 순서를 보장 -> 결정적
        in_bucket = key == b
        idx0 = np.flatnonzero(in_bucket & (labels == 0))
        idx1 = np.flatnonzero(in_bucket & (labels == 1))
        k = min(idx0.size, idx1.size)
        if k == 0:
            continue
        picked.append(np.sort(rng.choice(idx0, size=k, replace=False)))
        picked.append(np.sort(rng.choice(idx1, size=k, replace=False)))

    if not picked:
        return work.iloc[[]].reset_index(drop=True).copy()
    sel = np.sort(np.concatenate(picked))
    return work.iloc[sel].reset_index(drop=True).copy()


def _downsample_large_groups(
    df: pd.DataFrame, max_group_frac: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """단일 그룹이 층 전체의 max_group_frac을 넘지 않을 때까지 반복 다운샘플링한다.

    그룹 내부에서 층 전체 클래스 비율을 최대한 따라가도록 뽑는다.
    """
    work = df.reset_index(drop=True).copy()
    log: list[dict[str, Any]] = []
    for _ in range(100):
        n_total = len(work)
        if n_total == 0:
            break
        cap = int(np.floor(max_group_frac * n_total))
        sizes = work["group"].value_counts()
        over = sizes[sizes > cap]
        if over.empty:
            break
        p1 = float((work["label"] == 1).mean())
        keep_mask = np.ones(n_total, dtype=bool)
        for group in sorted(over.index):  # 정렬로 결정성 확보
            gidx = np.flatnonzero(work["group"].to_numpy() == group)
            glabels = work["label"].to_numpy()[gidx]
            i1 = gidx[glabels == 1]
            i0 = gidx[glabels == 0]
            want1 = min(int(round(cap * p1)), i1.size)
            want0 = min(cap - want1, i0.size)
            want1 = min(cap - want0, i1.size)  # 한쪽이 모자라면 다른 쪽으로 채운다
            take = np.concatenate(
                [
                    rng.choice(i1, size=want1, replace=False),
                    rng.choice(i0, size=want0, replace=False),
                ]
            )
            drop = np.setdiff1d(gidx, take, assume_unique=False)
            keep_mask[drop] = False
            log.append(
                {
                    "group": str(group),
                    "n_before": int(gidx.size),
                    "n_after": int(take.size),
                    "cap": cap,
                }
            )
        work = work[keep_mask].reset_index(drop=True)
    return work, log


def group_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    max_group_frac: float = 0.05,
    *,
    require_valid: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """eTLD+1 그룹 단위 그리디 빈패킹 분할.

    `split` 컬럼(train/val/test)을 추가한 DataFrame과 진단 dict를 반환한다.

    ``require_valid=True``(기본)면 세 split이 모두 비어 있지 않고 두 클래스를 모두
    포함하는지 검사해 아니면 ``ValueError``를 낸다. 한 split에 클래스가 하나뿐이면
    val AUROC가 NaN이 되고 임계값 선택도 무의미해지므로, 조용히 흘려보내면 안 된다.
    """
    _require(df, ("group", "label"))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9 or any(r <= 0 for r in ratios):
        raise ValueError(f"ratios는 합이 1인 양수 3개여야 한다 (got {ratios})")

    rng = np.random.default_rng(seed)
    n_before = len(df)
    work, downsample_log = _downsample_large_groups(df, max_group_frac, rng)
    n_after = len(work)

    if n_after == 0:
        out = work.copy()
        out["split"] = pd.Series([], dtype="object")
        return out, {
            "n_before_downsample": n_before,
            "n_after_downsample": 0,
            "downsampled_groups": downsample_log,
            "empty": True,
        }

    labels = work["label"].to_numpy(dtype=np.int64)
    groups = work["group"].to_numpy()
    uniq, inverse = np.unique(groups, return_inverse=True)
    n_groups = uniq.size

    g_total = np.bincount(inverse, minlength=n_groups).astype(np.int64)
    g_pos = np.bincount(inverse, weights=(labels == 1), minlength=n_groups).astype(np.int64)
    g_neg = g_total - g_pos

    ratio_arr = np.asarray(ratios, dtype=float)
    target_total = ratio_arr * n_after
    target_pos = ratio_arr * float(g_pos.sum())
    target_neg = ratio_arr * float(g_neg.sum())
    eps = 1.0

    # 크기 내림차순 + 시드 지터(동점 처리)
    jitter = rng.random(n_groups)
    order = np.lexsort((jitter, -g_total))

    cur_total = np.zeros(3)
    cur_pos = np.zeros(3)
    cur_neg = np.zeros(3)
    assign = np.empty(n_groups, dtype=np.int64)
    for gi in order:
        # 각 split에 넣었을 때의 "상대 충족률" 최대값(총량/양성/음성 중 가장 넘치는 축)이
        # 가장 작은 split에 배정한다. 여유가 큰 split이 우선 채워지므로 대형 그룹이
        # 작은 split에 몰리지 않고, 클래스별 축을 함께 보므로 class 비도 유지된다.
        fill = np.maximum.reduce(
            [
                (cur_total + g_total[gi]) / np.maximum(target_total, eps),
                (cur_pos + g_pos[gi]) / np.maximum(target_pos, eps),
                (cur_neg + g_neg[gi]) / np.maximum(target_neg, eps),
            ]
        )
        best = int(np.argmin(fill + rng.random(3) * 1e-12))
        assign[gi] = best
        cur_total[best] += g_total[gi]
        cur_pos[best] += g_pos[gi]
        cur_neg[best] += g_neg[gi]

    split_idx = assign[inverse]
    out = work.copy()
    out["split"] = np.asarray(_SPLIT_NAMES, dtype=object)[split_idx]

    # ---- 진단 ----
    group_sets = {name: set(out.loc[out["split"] == name, "group"]) for name in _SPLIT_NAMES}
    overlaps = {
        f"{a}&{b}": len(group_sets[a] & group_sets[b])
        for i, a in enumerate(_SPLIT_NAMES)
        for b in _SPLIT_NAMES[i + 1 :]
    }
    per_split: dict[str, Any] = {}
    for name in _SPLIT_NAMES:
        sub = out[out["split"] == name]
        per_split[name] = {
            "n": int(len(sub)),
            "frac": float(len(sub) / n_after),
            "n_pos": int((sub["label"] == 1).sum()),
            "pos_ratio": float((sub["label"] == 1).mean()) if len(sub) else 0.0,
            "n_groups": int(sub["group"].nunique()),
        }
    test = out[out["split"] == "test"]
    top5 = test["group"].value_counts().head(5)
    diagnostics: dict[str, Any] = {
        "seed": int(seed),
        "ratios": [float(r) for r in ratios],
        "max_group_frac": float(max_group_frac),
        "n_before_downsample": int(n_before),
        "n_after_downsample": int(n_after),
        "downsampled_groups": downsample_log,
        "n_groups": int(n_groups),
        "group_overlaps": overlaps,
        "groups_disjoint": all(v == 0 for v in overlaps.values()),
        "per_split": per_split,
        "top5_test_groups": {str(k): int(v) for k, v in top5.items()},
        "top5_test_group_frac": float(top5.sum() / len(test)) if len(test) else 0.0,
        "max_group_frac_observed": float(g_total.max() / n_after),
        "empty": False,
    }
    if require_valid:
        _assert_splits_usable(per_split)
    return out, diagnostics


def _assert_splits_usable(per_split: dict[str, Any]) -> None:
    """세 split이 모두 비어 있지 않고 두 클래스를 포함하는지 확인한다."""
    problems = []
    for name in _SPLIT_NAMES:
        info = per_split[name]
        if info["n"] == 0:
            problems.append(f"{name}: 비어 있음")
        elif info["n_pos"] == 0 or info["n_pos"] == info["n"]:
            problems.append(f"{name}: 클래스가 하나뿐 (n={info['n']}, n_pos={info['n_pos']})")
    if problems:
        raise ValueError("group_split 결과를 학습에 쓸 수 없다 — " + "; ".join(problems))


def match_by_length_within_splits(
    df: pd.DataFrame, bucket: int | None, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """**split별로 각각** 길이 매칭을 적용한다 (스펙 1.1 + 1.4 결합).

    층 전체에서 한 번만 매칭하고 그 뒤에 :func:`group_split`을 돌리면, 그룹 단위로
    행이 통째로 한 split에 몰리면서 **split 안의 길이 주변분포가 다시 어긋난다.**
    (실제로 재현된다: 어떤 길이 버킷의 benign이 전부 한 도메인 소속이면 그 버킷은
    train에만 남고 test에서는 phishing만 남는다.) 5% 상한 다운샘플도 매칭 이후에
    행을 지우므로 같은 방식으로 매칭을 깬다.

    그래서 순서를 **층 필터 → group_split(다운샘플 포함) → split별 매칭**으로 둔다.
    매칭은 행 제거만 하므로 split 간 그룹 교집합 0은 그대로 유지된다.

    Returns:
        ``split`` 컬럼이 유지된 매칭 후 DataFrame과 split별 진단 dict.
    """
    _require(df, (LENGTH_COL, "label", "split"))
    if bucket is None:
        out = df.reset_index(drop=True).copy()
        return out, length_match_report(out, bucket)

    parts: list[pd.DataFrame] = []
    for code, name in enumerate(_SPLIT_NAMES):
        sub = df[df["split"] == name]
        if len(sub) == 0:
            continue
        # split마다 다른 시드를 써도 seed가 같으면 전체가 결정적이다.
        parts.append(match_by_length(sub, bucket, seed * len(_SPLIT_NAMES) + code))
    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else df.iloc[[]].reset_index(drop=True).copy()
    )
    return out, length_match_report(out, bucket)


def length_match_report(df: pd.DataFrame, bucket: int | None) -> dict[str, Any]:
    """split별 (label × length_bucket) 히스토그램이 클래스 간 동일한지 진단한다."""
    report: dict[str, Any] = {"bucket": bucket, "per_split": {}}
    if "split" not in df.columns:
        return report
    for name in _SPLIT_NAMES:
        sub = df[df["split"] == name]
        entry: dict[str, Any] = {
            "n": int(len(sub)),
            "n_benign": int((sub["label"] == 0).sum()),
            "n_phishing": int((sub["label"] == 1).sum()),
        }
        if bucket is not None and len(sub):
            key = (sub[LENGTH_COL].to_numpy(dtype=np.int64) // bucket).astype(np.int64)
            lab = sub["label"].to_numpy(dtype=np.int64)
            h0 = pd.Series(key[lab == 0]).value_counts().sort_index()
            h1 = pd.Series(key[lab == 1]).value_counts().sort_index()
            entry["histograms_identical"] = bool(h0.equals(h1))
        elif bucket is not None:
            entry["histograms_identical"] = True
        report["per_split"][name] = entry
    if bucket is not None:
        report["all_identical"] = all(
            v.get("histograms_identical", True) for v in report["per_split"].values()
        )
    return report


def assert_length_matched(df: pd.DataFrame, bucket: int | None) -> dict[str, Any]:
    """split별 길이 히스토그램 동일성을 강제한다. 깨져 있으면 ``ValueError``."""
    report = length_match_report(df, bucket)
    if bucket is not None and not report.get("all_identical", True):
        bad = [k for k, v in report["per_split"].items() if not v.get("histograms_identical", True)]
        raise ValueError(f"split별 길이 매칭이 깨졌다: {bad} (bucket={bucket})")
    return report
