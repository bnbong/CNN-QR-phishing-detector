"""splits.py 테스트 — 스펙 9절 6, 8, 10, 11번 + 실데이터 스모크."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrphish.splits import (
    LENGTH_COL,
    assert_length_matched,
    group_split,
    length_match_report,
    match_by_length,
    match_by_length_within_splits,
)
from qrphish.urls import load_webphish

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synth(n_groups: int = 60, seed: int = 7) -> pd.DataFrame:
    """길이 분포가 클래스별로 어긋나고 거대 그룹이 하나 있는 합성 층."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        group = f"g{g:03d}.com"
        size = 400 if g == 0 else int(rng.integers(5, 60))
        for _ in range(size):
            label = int(rng.integers(0, 2))
            length = int(rng.integers(20, 40)) if label == 0 else int(rng.integers(28, 52))
            rows.append(
                {
                    "url": f"http://{group}/{rng.integers(0, 10**9)}",
                    "label": label,
                    "group": group,
                    LENGTH_COL: length,
                }
            )
    df = pd.DataFrame(rows)
    df["url"] = [f"{u}-{i}" for i, u in enumerate(df["url"])]
    return df


# --- 9절 6번: 길이 매칭 ---------------------------------------------------


def test_length_matching_exact_histograms_identical():
    df = _synth()
    matched = match_by_length(df, bucket=1, seed=0)
    h0 = matched[matched["label"] == 0][LENGTH_COL].value_counts().sort_index()
    h1 = matched[matched["label"] == 1][LENGTH_COL].value_counts().sort_index()
    pd.testing.assert_series_equal(h0, h1, check_names=False)
    assert len(matched) > 0
    assert (matched["label"] == 0).sum() == (matched["label"] == 1).sum()


def test_length_matching_quantile_bucket5():
    df = _synth()
    matched = match_by_length(df, bucket=5, seed=0)
    key = matched[LENGTH_COL] // 5
    h0 = key[matched["label"] == 0].value_counts().sort_index()
    h1 = key[matched["label"] == 1].value_counts().sort_index()
    pd.testing.assert_series_equal(h0, h1, check_names=False)
    # 완화 매칭은 엄격 매칭보다 표본이 많거나 같다
    assert len(matched) >= len(match_by_length(df, bucket=1, seed=0))


def test_length_matching_none_is_identity():
    df = _synth()
    out = match_by_length(df, bucket=None, seed=3)
    pd.testing.assert_frame_equal(out, df.reset_index(drop=True))


def test_length_matching_is_subset_of_input():
    df = _synth()
    matched = match_by_length(df, bucket=1, seed=1)
    assert set(matched["url"]).issubset(set(df["url"]))
    assert matched["url"].duplicated().sum() == 0


def test_length_matching_invalid_bucket():
    with pytest.raises(ValueError):
        match_by_length(_synth(), bucket=0, seed=0)


# --- 9절 8번: 그룹 분할 교집합 = 0 ---------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_group_disjoint(seed):
    df = match_by_length(_synth(), bucket=1, seed=seed)
    out, diag = group_split(df, seed=seed)
    sets = {name: set(out.loc[out["split"] == name, "group"]) for name in ("train", "val", "test")}
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    assert diag["groups_disjoint"] is True
    assert all(v == 0 for v in diag["group_overlaps"].values())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_ratios_and_class_balance(seed):
    df = match_by_length(_synth(), bucket=1, seed=seed)
    out, diag = group_split(df, ratios=(0.70, 0.15, 0.15), seed=seed)
    for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(diag["per_split"][name]["frac"] - target) < 0.12
        assert diag["per_split"][name]["n"] > 0
    overall = float((out["label"] == 1).mean())
    for name in ("train", "val", "test"):
        assert abs(diag["per_split"][name]["pos_ratio"] - overall) < 0.15


# --- 9절 9번(분할 단): 중복 URL 없음 -------------------------------------


def test_no_dup_url_across_splits():
    df = match_by_length(_synth(), bucket=1, seed=0)
    out, _ = group_split(df, seed=0)
    assert out["url"].duplicated().sum() == 0


# --- 9절 10번: 5% 상한 ----------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_max_group_frac(seed):
    df = match_by_length(_synth(), bucket=1, seed=seed)
    out, diag = group_split(df, seed=seed, max_group_frac=0.05)
    sizes = out["group"].value_counts()
    assert sizes.max() <= 0.05 * len(out) + 1e-9
    assert diag["max_group_frac_observed"] <= 0.05 + 1e-9
    assert diag["downsampled_groups"], "거대 그룹이 있는 합성 데이터인데 다운샘플 기록이 비었다"
    assert diag["n_after_downsample"] < diag["n_before_downsample"]


def test_downsample_log_shape():
    df = match_by_length(_synth(), bucket=1, seed=0)
    _, diag = group_split(df, seed=0)
    for rec in diag["downsampled_groups"]:
        assert set(rec) == {"group", "n_before", "n_after", "cap"}
        assert rec["n_after"] <= rec["n_before"]


# --- 9절 11번: 결정성 -----------------------------------------------------


def test_deterministic_pipeline():
    df = _synth()
    a_m = match_by_length(df, bucket=1, seed=2)
    b_m = match_by_length(df, bucket=1, seed=2)
    pd.testing.assert_frame_equal(a_m, b_m)

    a, da = group_split(a_m, seed=2)
    b, db = group_split(b_m, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert json.dumps(da, sort_keys=True) == json.dumps(db, sort_keys=True)
    # 바이트 동일성 (split/group 인코딩 포함)
    assert a["split"].to_numpy(dtype="U8").tobytes() == b["split"].to_numpy(dtype="U8").tobytes()
    assert (
        a["group"].astype("category").cat.codes.to_numpy().tobytes()
        == b["group"].astype("category").cat.codes.to_numpy().tobytes()
    )


def test_different_seeds_differ():
    df = _synth()
    a = match_by_length(df, bucket=1, seed=0)
    b = match_by_length(df, bucket=1, seed=1)
    assert a["url"].tolist() != b["url"].tolist()


def test_group_split_invalid_ratios():
    df = match_by_length(_synth(), bucket=1, seed=0)
    with pytest.raises(ValueError):
        group_split(df, ratios=(0.5, 0.3, 0.3), seed=0)


# --- 실데이터 스모크 ------------------------------------------------------


def test_real_pipeline_smoke(webphish_csv: Path):
    """load_webphish -> L-exact 매칭 -> 그룹 분할을 한 번 돌려 통계를 저장한다."""
    df, load_stats = load_webphish(webphish_csv, "norm")
    matched = match_by_length(df, bucket=1, seed=0)
    out, diag = group_split(matched, ratios=(0.70, 0.15, 0.15), seed=0, max_group_frac=0.05)

    assert (matched["label"] == 0).sum() == (matched["label"] == 1).sum()
    h0 = matched[matched["label"] == 0][LENGTH_COL].value_counts().sort_index()
    h1 = matched[matched["label"] == 1][LENGTH_COL].value_counts().sort_index()
    pd.testing.assert_series_equal(h0, h1, check_names=False)
    assert diag["groups_disjoint"] is True
    assert out["group"].value_counts().max() <= 0.05 * len(out) + 1e-9

    report = {
        "load_webphish": load_stats,
        "length_match_exact": {
            "n_before": int(len(df)),
            "n_after": int(len(matched)),
            "n_per_class": int((matched["label"] == 1).sum()),
            "n_length_buckets": int(matched[LENGTH_COL].nunique()),
            "bytes_len_min": int(matched[LENGTH_COL].min()),
            "bytes_len_max": int(matched[LENGTH_COL].max()),
        },
        "group_split": diag,
    }
    dest = REPO_ROOT / "reports" / "p0_urls_splits_smoke.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# --- 길이 매칭 × 그룹 분할 상호작용 (Codex 리뷰 #1 회귀) -------------------


def _split_hist(df: pd.DataFrame, split: str, label: int, bucket: int) -> pd.Series:
    sub = df[(df["split"] == split) & (df["label"] == label)]
    return (sub[LENGTH_COL] // bucket).value_counts().sort_index()


def test_match_then_split_breaks_length_matching():
    """**재현 케이스**: 매칭을 먼저 하면 split 안에서 길이 분포가 어긋난다.

    길이 10인 benign이 전부 한 도메인에 몰려 있으면, 그 그룹이 통째로 train에 가면서
    test에는 길이 10짜리 phishing만 남는다. 이 테스트는 "예전 순서는 깨진다"는 사실을
    고정해 두는 것이며, 아래 ``test_split_then_match_keeps_matching``이 새 순서를 지킨다.
    """
    rows = []
    # 길이 10 benign 200개가 전부 한 도메인 소속.
    for i in range(200):
        rows.append(
            {"url": f"http://benignhub.com/{i}", "label": 0, "group": "benignhub.com",
             LENGTH_COL: 10}
        )
    # 길이 10 phishing은 도메인이 200개로 흩어져 있다.
    for i in range(200):
        rows.append(
            {"url": f"http://phish{i:03d}.com/x", "label": 1, "group": f"phish{i:03d}.com",
             LENGTH_COL: 10}
        )
    # 두 클래스가 고르게 섞인 길이 20 블록(그래야 split이 비지 않는다).
    for i in range(400):
        lab = i % 2
        rows.append(
            {"url": f"http://mixed{i:03d}.com/y", "label": lab, "group": f"mixed{i:03d}.com",
             LENGTH_COL: 20}
        )
    df = pd.DataFrame(rows)

    # 옛 순서: 층 전체 매칭 → 분할.
    matched_first = match_by_length(df, bucket=1, seed=0)
    # 층 전체로 보면 완벽히 맞아 있다.
    assert (matched_first["label"] == 0).sum() == (matched_first["label"] == 1).sum()
    old, _ = group_split(matched_first, seed=0, max_group_frac=1.0)
    broken = [
        name
        for name in ("train", "val", "test")
        if not _split_hist(old, name, 0, 1).equals(_split_hist(old, name, 1, 1))
    ]
    assert broken, "재현 케이스가 더 이상 매칭을 깨지 않는다 — 테스트를 다시 설계해야 한다"

    # 새 순서: 분할 → split별 매칭. 같은 데이터에서 깨지지 않아야 한다.
    split_first, _ = group_split(df, seed=0, max_group_frac=1.0)
    fixed, report = match_by_length_within_splits(split_first, bucket=1, seed=0)
    assert report["all_identical"]
    for name in ("train", "val", "test"):
        pd.testing.assert_series_equal(
            _split_hist(fixed, name, 0, 1),
            _split_hist(fixed, name, 1, 1),
            check_names=False,
        )


@pytest.mark.parametrize("bucket", [1, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_then_match_keeps_matching(bucket, seed):
    """새 순서에서는 **split마다** 클래스별 길이 히스토그램이 동일하다."""
    df = _synth(seed=seed)
    split_df, _ = group_split(df, seed=seed)
    matched, report = match_by_length_within_splits(split_df, bucket=bucket, seed=seed)
    assert report["all_identical"]
    assert_length_matched(matched, bucket)
    for name in ("train", "val", "test"):
        pd.testing.assert_series_equal(
            _split_hist(matched, name, 0, bucket),
            _split_hist(matched, name, 1, bucket),
            check_names=False,
        )
    # 매칭은 행 제거만 하므로 그룹 교집합 0이 유지된다.
    sets = {n: set(matched.loc[matched["split"] == n, "group"]) for n in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


def test_match_within_splits_none_is_identity():
    df, _ = group_split(_synth(), seed=0)
    out, report = match_by_length_within_splits(df, bucket=None, seed=0)
    assert len(out) == len(df)
    assert report["bucket"] is None


def test_assert_length_matched_raises_on_broken_frame():
    df = pd.DataFrame(
        {
            "url": ["a", "b", "c"],
            "label": [0, 1, 1],
            "group": ["g1", "g2", "g3"],
            "split": ["test", "test", "test"],
            LENGTH_COL: [10, 10, 11],
        }
    )
    assert not length_match_report(df, 1)["all_identical"]
    with pytest.raises(ValueError, match="길이 매칭이 깨졌다"):
        assert_length_matched(df, 1)


# --- group_split 유효성 하드 실패 (Codex 리뷰 #2) --------------------------


def test_group_split_rejects_single_class_split():
    """한 클래스가 그룹 하나에 전부 몰리면 어떤 split은 클래스가 하나뿐이 된다."""
    rows = [
        {"url": f"http://one.com/{i}", "label": 0, "group": "one.com", LENGTH_COL: 10}
        for i in range(50)
    ]
    rows += [
        {"url": f"http://p{i:03d}.com/", "label": 1, "group": f"p{i:03d}.com", LENGTH_COL: 10}
        for i in range(50)
    ]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="학습에 쓸 수 없다"):
        group_split(df, seed=0, max_group_frac=1.0)
    # 진단 목적으로 검사를 끌 수 있다.
    out, diag = group_split(df, seed=0, max_group_frac=1.0, require_valid=False)
    assert len(out) == len(df)
    assert diag["groups_disjoint"]
