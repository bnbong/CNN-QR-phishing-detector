"""occlusion — 뒤집기 정확성, 무작위 대조 개수 일치, motif id 규약, 통합 실행."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from qrphish.motifs import extract_patch_ids
from qrphish.occlusion import (
    OCCLUSION_CONTRASTS,
    batch_logits,
    delta_vs_random,
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
        "random_matched", "random_matched_benign",
    }
    for name, block in res["conditions"].items():
        assert set(block) >= {
            "auroc", "d_auroc", "d_auroc_ci", "mean_logit_delta",
            "mean_logit_delta_ci", "n_flipped",
        }, name
        assert block["d_auroc_ci"][0] <= block["d_auroc_ci"][1]
    # 무작위 조건은 반복별 AUROC를 계산해 평균한다(로짓 평균 금지, review_03 3절).
    for name in ("random", "random_benign_matched", "random_matched",
                 "random_matched_benign"):
        b = res["conditions"][name]
        assert b["aggregation"] == "per_repeat_auroc_then_mean"
        assert b["n_repeat"] == 2 and len(b["per_rep"]) == 2
        assert b["d_auroc"] == pytest.approx(
            float(np.mean([d["d_auroc"] for d in b["per_rep"]]))
        )
        assert b["auroc"] == pytest.approx(
            float(np.mean([d["auroc"] for d in b["per_rep"]]))
        )
        assert b["d_auroc_range"][0] <= b["d_auroc"] <= b["d_auroc_range"][1]
    # ΔΔ(표적 − 무작위) 대비 블록.
    assert set(res["contrasts"]) == {n for n, _, _ in OCCLUSION_CONTRASTS}
    for blk in res["contrasts"].values():
        assert blk["ci"][0] <= blk["estimate"] + 1e-9
        assert blk["estimate"] <= blk["ci"][1] + 1e-9
        assert 0.0 < blk["perm_p"] <= 1.0
        assert len(blk["per_rep"]) == 2
    assert "_arrays" in res and set(res["_arrays"]["reps"]) == {
        "random", "random_benign_matched", "random_matched", "random_matched_benign",
    }
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


# --------------------------------------- review_02 4: topology-matched random control
def _dm_and_values(n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    dm = np.zeros((n, n), dtype=bool)
    dm[1 : n - 1, 1 : n - 1] = True
    vals = rng.integers(0, 2, size=(n, n)).astype(np.float64)
    return vals, dm


def test_matched_random_centers_matches_quadrant_and_density():
    from qrphish.occlusion import _quadrant, local_black_density, matched_random_centers

    vals, dm = _dm_and_values()
    pool = window_centers(dm, 3)
    rng = np.random.default_rng(0)
    tgt = pool[rng.choice(len(pool), size=20, replace=False)]

    got, n_relaxed = matched_random_centers(vals, dm, tgt, np.random.default_rng(1), 3)
    assert got.shape == tgt.shape
    assert n_relaxed == 0
    # 중복 없음
    assert len({tuple(c) for c in got.tolist()}) == len(got)
    # 사분면 일치 + 국소 흑색 밀도 ±1
    n = dm.shape[0]
    assert np.array_equal(_quadrant(got, n), _quadrant(tgt, n))
    d_got = local_black_density(vals, got, 3)
    d_tgt = local_black_density(vals, tgt, 3)
    assert np.all(np.abs(d_got - d_tgt) <= 1)
    # 모두 유효한 창 중심(데이터 모듈로만 이루어진 창)이어야 한다
    pool_set = {tuple(c) for c in pool.tolist()}
    assert all(tuple(c) in pool_set for c in got.tolist())


def test_matched_random_centers_is_deterministic_and_handles_empty():
    from qrphish.occlusion import matched_random_centers

    vals, dm = _dm_and_values(seed=2)
    tgt = window_centers(dm, 3)[:8]
    a, _ = matched_random_centers(vals, dm, tgt, np.random.default_rng(9), 3)
    b, _ = matched_random_centers(vals, dm, tgt, np.random.default_rng(9), 3)
    assert np.array_equal(a, b)
    empty, relaxed = matched_random_centers(
        vals, dm, np.zeros((0, 2), dtype=np.int64), np.random.default_rng(0), 3
    )
    assert empty.shape == (0, 2) and relaxed == 0


def test_matched_random_centers_relaxes_when_pool_is_small():
    """후보가 모자라면 사분면 제약만 남기고 그 횟수를 기록한다."""
    from qrphish.occlusion import matched_random_centers

    n = 11
    dm = np.zeros((n, n), dtype=bool)
    dm[0:5, 0:5] = True  # 좌상단 사분면에만 창이 생긴다
    vals = np.zeros((n, n), dtype=np.float64)
    vals[0:3, 0:3] = 1.0  # (1,1) 창만 밀도 9, 나머지는 훨씬 낮다
    pool = window_centers(dm, 3)
    assert len(pool) >= 4
    # 밀도 9인 창은 하나뿐이므로 두 번째 요청부터는 밀도 제약을 풀 수밖에 없다.
    tgt = np.repeat(np.array([[1, 1]], dtype=np.int64), len(pool) + 3, axis=0)
    got, relaxed = matched_random_centers(vals, dm, tgt, np.random.default_rng(0), 3)
    # target center (1,1)은 후보 풀에서 배제되므로 최대 len(pool) - 1개만 고를 수 있다.
    assert len(got) == len(pool) - 1
    assert (1, 1) not in {tuple(c) for c in got.tolist()}
    assert len({tuple(c) for c in got.tolist()}) == len(got)
    assert relaxed > 0


def test_matched_random_centers_excludes_motif_centers():
    """대조군은 실제 motif center(자기 자신도, exclude로 넘긴 반대 클래스도) 고르지 않는다."""
    from qrphish.occlusion import matched_random_centers

    rng = np.random.default_rng(0)
    vals = (rng.random((21, 21)) > 0.5).astype(np.float64)
    dm = np.ones((21, 21), dtype=bool)
    pool = window_centers(dm, 3)
    tgt = pool[:15]          # phishing motif center
    other = pool[15:30]      # benign motif center

    got, _ = matched_random_centers(
        vals, dm, tgt, np.random.default_rng(3), 3, exclude=np.concatenate([tgt, other])
    )
    picked = {tuple(c) for c in got.tolist()}
    assert picked.isdisjoint({tuple(c) for c in tgt.tolist()})
    assert picked.isdisjoint({tuple(c) for c in other.tolist()})
    assert len(got) == len(tgt)


# --------------------------------- review_03 3: 반복별 AUROC와 ΔΔ 쌍체 통계 ------
def _dd_fixture(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int64)
    groups = np.array([f"g{i % 12}" for i in range(n)])
    p = y * 1.5 + rng.normal(size=n)
    return y, groups, p


def test_delta_vs_random_is_zero_for_identical_predictions() -> None:
    """표적과 무작위 예측이 같으면 ΔΔ와 CI가 정확히 0이어야 한다."""
    y, groups, p = _dd_fixture()
    res = delta_vs_random(y, groups, p, [p.copy(), p.copy()], n_boot=50, seed=0, n_perm=50)
    assert res["estimate"] == 0.0
    assert res["ci"] == [0.0, 0.0]
    assert res["per_rep"] == [0.0, 0.0]
    assert res["sd_across_reps"] == 0.0
    assert res["perm_p"] == 1.0


def test_delta_vs_random_sign_and_determinism() -> None:
    """표적이 더 크게 무너뜨리면 ΔΔ는 음수. 같은 시드에서 결정적이어야 한다."""
    y, groups, p = _dd_fixture()
    weak = p + np.random.default_rng(1).normal(scale=0.3, size=p.size)
    strong = np.random.default_rng(2).normal(size=p.size)  # 신호를 완전히 지운다
    a = delta_vs_random(y, groups, strong, [weak, weak.copy()],
                        n_boot=100, seed=0, n_perm=100)
    b = delta_vs_random(y, groups, strong, [weak, weak.copy()],
                        n_boot=100, seed=0, n_perm=100)
    assert a == b
    assert a["estimate"] < 0
    assert a["ci"][0] <= a["estimate"] <= a["ci"][1]


def test_occlude_stratum_is_deterministic(setup: Setup, tmp_path) -> None:
    _, val, dm = _sample(setup)
    ids = np.bincount(extract_patch_ids(val > 0.5, dm, 3).astype(np.int64), minlength=512)
    p = tmp_path / "det" / "enrichment_seed0.json"
    _write_enrichment(p, [int(i) for i in np.argsort(-ids)[:2]],
                      [int(i) for i in np.argsort(-ids)[2:4]])
    te = np.nonzero(setup.split == 2)[0]

    def run():
        r = occlude_stratum(
            setup.model, setup.ds, te, setup.groups[te], load_enrichment(p),
            top_k=2, size=3, n_random_rep=2, seed=0, n_boot=20, n_perm=20,
        )
        r.pop("_arrays")
        return r

    assert run() == run()


# ------------------------ review_03 2: 러너 경로 — 시드별 enrichment만 쓴다 -------
@pytest.fixture(scope="module")
def _occl_runner(tmp_path_factory):
    """합성 WebPhish로 1차 학습 → run_motifs → run_occlusion 전 경로."""
    from qrphish.runner import MAIN_CONDITION, _run_one_seed, run_motifs
    from tests.test_transfer import _cfg, _frame, _write_webphish_csv

    tmp = tmp_path_factory.mktemp("occl_runner")
    csv = tmp / "webphish.csv"
    _write_webphish_csv(csv, _frame(tag="wp"))
    cfg = _cfg(tmp, csv)
    res = _run_one_seed(cfg, "v3", 0, Path(cfg.output_dir) / MAIN_CONDITION, {"baselines": []})
    assert "error" not in res, res
    run_motifs(cfg, strata=["v3"])
    mdir = Path(cfg.reports_dir) / "motifs" / "norm-exact-data_only-fixed" / "v3"
    return cfg, mdir


def test_run_occlusion_uses_per_seed_enrichment(_occl_runner) -> None:
    from qrphish.runner import run_occlusion

    cfg, mdir = _occl_runner
    assert (mdir / "enrichment_seed0.json").exists()
    # 시드 통합 파일은 보고용으로만 남는다.
    assert json.loads((mdir / "enrichment.json").read_text())["usage"] == "reporting_only"

    res = run_occlusion(cfg, strata=["v3"], top_k=3, n_random_rep=2, n_perm=10)["v3"]
    assert res["motif_source"] == "per_seed_train"
    assert res["motif_source_paths"]["0"].endswith("enrichment_seed0.json")
    assert res["per_seed"][0]["motif_source"] == "per_seed_train"
    agg = res["aggregate"]
    assert agg["random"]["aggregation"] == "per_repeat_auroc_then_mean"
    assert set(agg["delta_vs_random"]) == {n for n, _, _ in OCCLUSION_CONTRASTS}
    for blk in agg["d_auroc_pooled"].values():
        assert blk["pooling"] == "seed_stratified"
        assert blk["ci"][0] <= blk["ci"][1]
    # 결과 JSON은 직렬화 가능해야 한다(_arrays가 새어 나가면 실패한다).
    json.dumps(res, default=str)
    assert "_arrays" not in res["per_seed"][0]


def test_run_occlusion_does_not_fall_back_to_pooled_enrichment(
    _occl_runner, tmp_path
) -> None:
    """시드별 파일이 없으면 옛 통합 enrichment.json으로 되돌아가지 않고 실패해야 한다."""
    import dataclasses
    import shutil

    from qrphish.runner import run_occlusion

    cfg, _mdir = _occl_runner
    dst = tmp_path / "reports"
    shutil.copytree(cfg.reports_dir, dst)
    shutil.rmtree(dst / "occlusion", ignore_errors=True)  # 앞 테스트의 결과 캐시 제거
    d2 = dst / "motifs" / "norm-exact-data_only-fixed" / "v3"
    (d2 / "enrichment_seed0.json").rename(d2 / "enrichment_seed9.json")  # glob은 통과
    cfg2 = dataclasses.replace(cfg, reports_dir=str(dst))
    out = run_occlusion(cfg2, strata=["v3"], top_k=3, n_random_rep=2, n_perm=10)["v3"]
    assert out["error"] == "no usable seed"
    assert "enrichment_seed0.json" in out["per_seed"][0]["error"]
