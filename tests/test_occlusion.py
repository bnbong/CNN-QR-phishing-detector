"""occlusion — 뒤집기 정확성, 무작위 대조 개수 일치, motif id 규약, 통합 실행."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from qrphish.motifs import extract_patch_ids
from qrphish.occlusion import (
    batch_logits,
    flip_centers,
    load_enrichment,
    match_centers,
    occlude_stratum,
    random_centers,
    window_centers,
    window_ids,
)
from tests.test_explain import Setup, build_trained


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> Setup:
    return build_trained(tmp_path_factory.mktemp("occl"), n_per_class=24, epochs=2)


def _sample(setup: Setup, i: int = 0):
    x, _ = setup.ds[i]
    return x, x[0].numpy(), x[1].numpy() > 0.5


# ------------------------------------------------------------------ 창 열거·id
def test_window_centers_inside_data_mask(setup: Setup) -> None:
    _, _, dm = _sample(setup)
    centers = window_centers(dm, 3)
    assert centers.ndim == 2 and centers.shape[1] == 2
    for r, c in centers:
        assert dm[r - 1 : r + 2, c - 1 : c + 2].all()


def test_window_ids_match_motifs_convention(setup: Setup) -> None:
    """occlusion의 id 규약이 enrichment를 만든 motifs.extract_patch_ids와 같아야 한다."""
    _, val, dm = _sample(setup)
    ids, rows, cols = extract_patch_ids(val > 0.5, dm, 3, return_positions=True)
    centers = np.stack([rows + 1, cols + 1], axis=1)
    assert np.array_equal(window_ids(val, centers, 3), ids.astype(np.int64))


def test_window_id_row_major_msb_first() -> None:
    grid = np.zeros((3, 3))
    grid[0, 0] = 1  # 최상위 비트
    ids = window_ids(grid, np.array([[1, 1]]), 3)
    assert int(ids[0]) == 1 << 8


def test_match_centers_finds_only_requested_motifs(setup: Setup) -> None:
    _, val, dm = _sample(setup)
    ids = extract_patch_ids(val > 0.5, dm, 3)
    target = int(np.bincount(ids.astype(np.int64)).argmax())
    centers = match_centers(val, dm, [target], 3)
    assert len(centers) == int((ids == target).sum())
    assert np.all(window_ids(val, centers, 3) == target)


def test_match_centers_empty_for_absent_motif(setup: Setup) -> None:
    _, val, dm = _sample(setup)
    present = set(extract_patch_ids(val > 0.5, dm, 3).tolist())
    absent = next(i for i in range(512) if i not in present)
    assert match_centers(val, dm, [absent], 3).shape == (0, 2)


# --------------------------------------------------------------------- 뒤집기
def test_flip_changes_only_specified_centers(setup: Setup) -> None:
    x, val, dm = _sample(setup)
    centers = window_centers(dm, 3)[:5]
    out = flip_centers(x, centers)
    diff = np.nonzero(out[0].numpy() != val)
    assert set(zip(diff[0].tolist(), diff[1].tolist(), strict=True)) == {
        (int(r), int(c)) for r, c in centers
    }
    for r, c in centers:
        assert out[0, r, c].item() == pytest.approx(1.0 - val[r, c])


def test_flip_preserves_mask_channel(setup: Setup) -> None:
    x, _, dm = _sample(setup)
    out = flip_centers(x, window_centers(dm, 3)[:7])
    assert torch.equal(out[1], x[1])
    assert not torch.equal(out[0], x[0])


def test_flip_empty_is_identity(setup: Setup) -> None:
    x, _, _ = _sample(setup)
    assert torch.equal(flip_centers(x, np.zeros((0, 2), dtype=np.int64)), x)


def test_random_centers_count_and_pool(setup: Setup) -> None:
    _, _, dm = _sample(setup)
    pool = {(int(r), int(c)) for r, c in window_centers(dm, 3)}
    rng = np.random.default_rng(0)
    c = random_centers(dm, 12, rng, 3)
    assert len(c) == 12
    assert len({(int(r), int(cc)) for r, cc in c}) == 12  # 비복원
    assert {(int(r), int(cc)) for r, cc in c} <= pool


def test_random_centers_capped_by_pool_size() -> None:
    dm = np.zeros((9, 9), dtype=bool)
    dm[3:6, 3:6] = True  # 유효 창 중심 1개
    c = random_centers(dm, 10, np.random.default_rng(0), 3)
    assert len(c) == 1


def test_random_control_matches_motif_flip_count(setup: Setup) -> None:
    """무작위 대조는 motif 매칭 개수와 정확히 같은 수를 뒤집어야 한다."""
    x, val, dm = _sample(setup)
    ids = extract_patch_ids(val > 0.5, dm, 3)
    target = int(np.bincount(ids.astype(np.int64)).argmax())
    k = len(match_centers(val, dm, [target], 3))
    rc = random_centers(dm, k, np.random.default_rng(1), 3)
    assert len(rc) == k
    flipped = (flip_centers(x, rc)[0].numpy() != val).sum()
    assert flipped == k


# ------------------------------------------------------------------- 통합 실행
def _write_enrichment(path, top_ph, top_bn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stratum": "v2", "size": 3, "n_seeds": 1, "motifs": [],
                "top_phishing": top_ph, "top_benign": top_bn,
            }
        ),
        encoding="utf-8",
    )


def test_load_enrichment_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="motif enrichment"):
        load_enrichment(tmp_path / "nope.json")


def test_load_enrichment_bad_schema(tmp_path) -> None:
    p = tmp_path / "enrichment.json"
    p.write_text(json.dumps({"stratum": "v2", "size": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="top_phishing"):
        load_enrichment(p)


def test_occlude_stratum_end_to_end(setup: Setup, tmp_path) -> None:
    _, val, dm = _sample(setup)
    ids = np.bincount(extract_patch_ids(val > 0.5, dm, 3).astype(np.int64), minlength=512)
    top_ph = [int(i) for i in np.argsort(-ids)[:3]]
    top_bn = [int(i) for i in np.argsort(-ids)[3:6]]
    p = tmp_path / "motifs" / "v2" / "enrichment.json"
    _write_enrichment(p, top_ph, top_bn)

    te = np.nonzero(setup.split == 2)[0]
    res = occlude_stratum(
        setup.model, setup.ds, te, setup.groups[te], load_enrichment(p),
        top_k=3, size=3, n_random_rep=2, seed=0, n_boot=20,
    )
    assert res["n_test"] == te.size
    assert set(res["conditions"]) == {
        "phishing_motif", "benign_motif", "random", "random_benign_matched",
    }
    for name, block in res["conditions"].items():
        assert set(block) >= {
            "auroc", "d_auroc", "d_auroc_ci", "mean_logit_delta",
            "mean_logit_delta_ci", "n_flipped",
        }, name
        assert block["d_auroc_ci"][0] <= block["d_auroc_ci"][1]
    # 무작위 대조는 짝이 되는 motif 조건과 같은 개수를 뒤집는다(개수 효과 상쇄).
    assert (
        res["conditions"]["random"]["n_flipped"]
        == res["conditions"]["phishing_motif"]["n_flipped"]
    )
    assert (
        res["conditions"]["random_benign_matched"]["n_flipped"]
        == res["conditions"]["benign_motif"]["n_flipped"]
    )
    assert res["conditions"]["phishing_motif"]["n_flipped"]["mean"] > 0


def test_occlusion_changes_logits(setup: Setup) -> None:
    """상위 motif를 뒤집으면 로짓이 실제로 움직여야 한다(개입이 작동하는지 확인)."""
    te = np.nonzero(setup.split == 2)[0][:8]
    xs = [setup.ds[int(i)][0] for i in te]
    base = batch_logits(setup.model, xs)
    flipped = []
    for x in xs:
        val, dm = x[0].numpy(), x[1].numpy() > 0.5
        ids = extract_patch_ids(val > 0.5, dm, 3)
        top = int(np.bincount(ids.astype(np.int64)).argmax())
        flipped.append(flip_centers(x, match_centers(val, dm, [top], 3)))
    assert not np.allclose(base, batch_logits(setup.model, flipped))
