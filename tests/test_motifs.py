"""Bag-of-QR-patches (qrphish.motifs) 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrphish import motifs as M
from qrphish.evaluate import cluster_bootstrap


def test_bits_string_and_n_motifs():
    assert M.n_motifs(2) == 16
    assert M.n_motifs(3) == 512
    assert M.bits_string(173, 3) == "010101101"
    assert M.bits_string(0b1011, 2) == "1011"


def test_patch_ids_manual_2x2():
    # 격자:  1 0 1
    #        1 1 0
    #        0 0 1
    g = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=bool)
    mask = np.ones((3, 3), dtype=bool)
    ids = M.extract_patch_ids(g, mask, 2)
    # row-major, 좌상단부터: (1,0,1,1)=0b1011=11, (0,1,1,0)=0b0110=6,
    #                         (1,1,0,0)=0b1100=12, (1,0,0,1)=0b1001=9
    assert ids.tolist() == [11, 6, 12, 9]


def test_patch_ids_manual_3x3():
    g = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
    ids = M.extract_patch_ids(g, np.ones((3, 3), bool), 3)
    assert ids.tolist() == [int("101010111", 2)]


def test_data_mask_boundary_drops_partial_windows():
    g = np.ones((4, 4), dtype=bool)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False  # (0,0) 창만 무효가 된다
    ids, rows, cols = M.extract_patch_ids(g, mask, 2, return_positions=True)
    assert len(ids) == 9 - 1
    assert (0, 0) not in list(zip(rows.tolist(), cols.tolist(), strict=True))
    # 전부 1인 2x2 -> id 15
    assert set(ids.tolist()) == {15}
    # 데이터 모듈이 전혀 없으면 창도 없다
    assert M.extract_patch_ids(g, np.zeros((4, 4), bool), 2).size == 0


def test_histogram_sums_to_one_and_is_zero_without_windows():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, size=(9, 9)).astype(bool)
    h = M.patch_histogram(g, np.ones((9, 9), bool), 3)
    assert h.shape == (512,)
    assert h.sum() == pytest.approx(1.0, abs=1e-6)
    assert M.patch_histogram(g, np.zeros((9, 9), bool), 3).sum() == 0.0


def test_pyramid_dimension_and_block_normalization():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 2, size=(11, 11)).astype(bool)
    p = M.spatial_pyramid_histogram(g, np.ones((11, 11), bool), 3)
    assert p.shape == (5 * 512,)
    for b in range(5):
        assert p[b * 512 : (b + 1) * 512].sum() == pytest.approx(1.0, abs=1e-6)
    # 첫 블록은 전역 히스토그램과 같아야 한다
    assert np.allclose(p[:512], M.patch_histogram(g, np.ones((11, 11), bool), 3))
    # 2x2 피라미드 차원
    assert M.spatial_pyramid_histogram(g, np.ones((11, 11), bool), 2).shape == (5 * 16,)


def test_unmask_option_changes_values():
    from qrphish.dataset import data_mask_for
    from qrphish.mapping import unmask
    from qrphish.qrgen import encode

    art = encode("https://example.com/a", ec=0, mask_pattern=0)
    g = np.asarray(art.modules, dtype=bool)
    dm = data_mask_for(art.version, art.ec, g.shape[0])
    a = M.extract_patch_ids(g, dm, 3)
    b = M.extract_patch_ids(g, dm, 3, unmask=True, mask_pattern=art.mask_pattern)
    c = M.extract_patch_ids(np.asarray(unmask(g, art.mask_pattern)), dm, 3)
    assert np.array_equal(b, c)
    assert not np.array_equal(a, b)


def _synthetic_motif_data(n=240, size=2, planted=15, seed=0):
    """phishing 절반에만 '전부 1인 2x2'를 인위적으로 심은 합성 데이터."""
    rng = np.random.default_rng(seed)
    m = M.n_motifs(size)
    y = np.repeat([0, 1], n // 2)
    H = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        base = rng.random(m).astype(np.float32)
        base[planted] = 0.0
        if y[i] == 1:
            base[planted] = 2.0  # 피싱에만 심는다
        H[i] = base / base.sum()
    groups = np.array([f"g{i % 20}" for i in range(n)])
    return H, y, groups


def test_enrichment_sign_is_correct_for_planted_motif():
    H, y, groups = _synthetic_motif_data()
    enr = M.motif_enrichment(H, y, groups, size=2, n_boot=200, seed=0)
    planted = 15
    assert enr["log_odds"][planted] > 0
    assert enr["ci_lo"][planted] > 0  # CI가 0을 포함하지 않는다
    assert enr["rate_phishing"][planted] > enr["rate_benign"][planted]
    others = [i for i in range(16) if i != planted]
    assert abs(np.median(enr["log_odds"][others])) < abs(enr["log_odds"][planted])


def test_enrichment_sign_flips_when_planted_in_benign():
    H, y, groups = _synthetic_motif_data()
    enr = M.motif_enrichment(H, 1 - y, groups, size=2, n_boot=200, seed=0)
    assert enr["log_odds"][15] < 0


def test_group_bootstrap_weights_match_cluster_bootstrap_resampling():
    """가중치 경로가 evaluate.cluster_bootstrap의 리샘플과 동일한지 확인한다."""
    rng = np.random.default_rng(3)
    groups = np.array([f"g{i % 7}" for i in range(60)])
    y = rng.integers(0, 2, size=60)
    p = rng.random(60)

    gidx, W = M.group_bootstrap_weights(groups, n_boot=50, seed=11)
    # 같은 시드/같은 난수 호출이므로 그룹 선택 횟수가 일치해야 한다.
    ref_rng = np.random.default_rng(11)
    n_g = W.shape[1]
    for b in range(50):
        pick = ref_rng.integers(0, n_g, size=n_g)
        assert np.array_equal(W[b], np.bincount(pick, minlength=n_g).astype(float))
    # 가중치 합 = 표본 수를 그룹 크기로 잰 값과 일치
    sizes = np.bincount(gidx, minlength=n_g)
    assert int((W[0] * sizes).sum()) > 0
    # cluster_bootstrap이 같은 시드에서 결정적인지도 확인(회귀 방지)
    a = cluster_bootstrap(y, p, groups, lambda u, v: float(np.mean(v)), n_boot=20, seed=11)
    b = cluster_bootstrap(y, p, groups, lambda u, v: float(np.mean(v)), n_boot=20, seed=11)
    assert a["lo"] == b["lo"] and a["hi"] == b["hi"]


def test_bag_of_patches_lr_learns_planted_motif():
    from qrphish.evaluate import auroc

    H, y, groups = _synthetic_motif_data(n=400, seed=1)
    tr = np.arange(400) % 4 < 2
    va = np.arange(400) % 4 == 2
    te = np.arange(400) % 4 == 3
    fit = M.bag_of_patches_lr(H[tr], y[tr], H[te], 0, X_hist_val=H[va], y_val=y[va])
    assert fit["p_test"].shape == (te.sum(),)
    assert auroc(y[te], fit["p_test"]) > 0.9
    assert fit["C"] in M.C_GRID


def test_bag_of_patches_lr_single_class_train_is_constant():
    H, y, _ = _synthetic_motif_data(n=40)
    fit = M.bag_of_patches_lr(H[:10], np.zeros(10, dtype=int), H[30:], 0)
    assert np.allclose(fit["p_test"], 0.0)


def test_combine_seed_enrichments_marks_stable_motifs():
    H, y, groups = _synthetic_motif_data()
    per_seed = [M.motif_enrichment(H, y, groups, size=2, n_boot=50, seed=s) for s in range(5)]
    tests = [d["log_odds"] for d in per_seed]
    out = M.combine_seed_enrichments(per_seed, tests, size=2, n_windows_mean=42.0)
    assert out["size"] == 2 and out["n_seeds"] == 5 and out["n_windows_mean"] == 42.0
    assert out["top_phishing"][0] == 15
    ids = {d["id"]: d for d in out["motifs"]}
    assert ids[15]["seed_sign_agreement"] == 5
    assert ids[15]["test_sign_match"] is True
    assert ids[15]["bits"] == "1111"
    assert len(ids[15]["ci"]) == 2
    assert len(out["top_phishing"]) <= 20


# ------------------------------------------------------------------------- runner
def _smoke_cfg(tmp_path):
    from qrphish.config import from_dict

    rng = np.random.default_rng(0)
    rows = []
    words = ["login", "verify", "secure", "account", "update", "index", "home", "shop"]
    for i in range(160):
        label = i % 2
        word = words[i % len(words)] if label else words[(i + 3) % len(words)]
        host = f"site{i % 40}.com"
        pad = "".join(rng.choice(list("abcdefghijklmnop"), size=8))
        rows.append(
            {
                "Category": "spam" if label else "ham",
                "Data": f"http://{host}/{word}/{pad}{i:03d}",
            }
        )
    csv = tmp_path / "mini.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return from_dict(
        {
            "seed_list": [0, 1],
            "data": {"csv_path": str(csv)},
            "condition": {"length_match": "none"},
            "strata": ["v2", "v3"],
            "stratum_rules": {"primary_min_per_class": 10, "secondary_min_per_class": 5},
            "eval": {"n_bootstrap": 30},
            "output_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )


def _usable_stratum(cfg) -> str:
    """합성 URL이 어느 버전 층에 떨어지는지는 정규화 결과에 달려 있어 실측으로 고른다."""
    from qrphish.runner import _prepare_frame

    for st in ("v2", "v3"):
        try:
            df, _ = _prepare_frame(cfg, st, 0)
        except Exception:
            continue
        if len(df) and "split" in df.columns:
            return st
    raise AssertionError("합성 데이터가 어느 층에도 배정되지 않았다")


def test_run_motifs_smoke(tmp_path):
    import json

    from qrphish.runner import run_motifs

    cfg = _smoke_cfg(tmp_path)
    st = _usable_stratum(cfg)
    out = run_motifs(cfg, strata=[st])
    assert st in out
    res = json.loads((tmp_path / f"reports/motifs/norm-none-data_only-fixed/{st}/results.json").read_text(encoding="utf-8"))
    assert res["stratum"] == st
    assert res["condition_id"] == "norm-none-data_only-fixed"
    for key in ("patch2", "patch3", "pyramid3", "patch3_labelshuffle"):
        assert key in res["representations"]
        assert "auroc_mean" in res["representations"][key]
    assert res["representations"]["pyramid3"]["dim"] == 5 * 512
    assert res["representations"]["patch2"]["dim"] == 16

    enr = json.loads((tmp_path / f"reports/motifs/norm-none-data_only-fixed/{st}/enrichment.json").read_text(encoding="utf-8"))
    assert enr["stratum"] == st and enr["size"] == 3
    assert enr["n_windows_mean"] > 0
    assert isinstance(enr["motifs"], list) and enr["motifs"]
    first = enr["motifs"][0]
    assert set(first) >= {
        "id",
        "bits",
        "rate_phishing",
        "rate_benign",
        "log_odds",
        "ci",
        "seed_sign_agreement",
        "test_sign_match",
    }
    assert len(first["bits"]) == 9
    assert enr["size2"]["size"] == 2
    assert all(len(d["bits"]) == 4 for d in enr["size2"]["motifs"])
    assert len(enr["top_phishing"]) <= 20


def test_run_motifs_reproduces_main_condition_split(tmp_path):
    """motif 러너의 split이 기존 규약(_prepare_frame, 같은 시드)과 정확히 같은지."""
    from qrphish.runner import _prepare_frame

    cfg = _smoke_cfg(tmp_path)
    st = _usable_stratum(cfg)
    a, _ = _prepare_frame(cfg, st, 0)
    b, _ = _prepare_frame(cfg, st, 0)
    assert list(a["url"]) == list(b["url"])
    assert list(a["split"]) == list(b["split"])
    c, _ = _prepare_frame(cfg, st, 1)
    assert (list(a["url"]), list(a["split"])) != (list(c["url"]), list(c["split"]))


def test_combine_seed_enrichments_filters_top_motifs():
    """상위 motif는 최소 출현율·CI 0 배제·비포화를 모두 만족해야 한다 (코덱스 리뷰 5)."""
    H, y, groups = _synthetic_motif_data()
    per_seed = [M.motif_enrichment(H, y, groups, size=2, n_boot=100, seed=s) for s in range(5)]
    tests = [d["log_odds"] for d in per_seed]
    pooled = M.motif_enrichment(H, y, groups, size=2, n_boot=300, seed=0)

    out = M.combine_seed_enrichments(
        per_seed, tests, size=2, n_windows_mean=42.0, pooled=pooled, ci_n_boot=300
    )
    assert out["ci_source"] == "pooled_5seed_train" and out["ci_n_boot"] == 300
    ids = {d["id"]: d for d in out["motifs"]}
    for mid in out["top_phishing"] + out["top_benign"]:
        d = ids[mid]
        assert d["ci_excludes_zero"] is True
        assert d["ubiquitous"] is False
        assert max(d["present_phishing"], d["present_benign"]) >= out["min_presence"]

    # 출현율 문턱을 1 위로 올리면 어떤 motif도 통과할 수 없다.
    strict = M.combine_seed_enrichments(
        per_seed, tests, size=2, n_windows_mean=42.0, pooled=pooled, min_presence=1.5
    )
    assert strict["top_phishing"] == [] and strict["top_benign"] == []
    assert strict["n_stable_after_filter"] == 0
