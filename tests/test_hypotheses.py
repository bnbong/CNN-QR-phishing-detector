"""가설 검정 러너(H1~H4 + Holm)와 조건 간 쌍체 비교 테스트 (리뷰 A항목 3·4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qrphish.runner import (
    _decoded_text_reference,
    compare_conditions,
    run_hypothesis_tests,
)

SEEDS = [0, 1, 2]
BASE = "norm-exact-data_only-fixed-small_cnn"
SHUF = f"{BASE}-labelshuffle"
POS = f"{BASE}-shufflepos"
NONE = "norm-none-data_only-fixed-small_cnn"

MATRIX = """
P1:
  - name: main_cnn
    extra: null
    overrides: {condition.length_match: exact, model.arch: small_cnn}
  - name: label_shuffle
    extra: labelshuffle
    overrides: {condition.length_match: exact, model.arch: small_cnn}
  - name: shuffle_pos
    extra: shufflepos
    overrides: {condition.length_match: exact, model.arch: small_cnn}
P2:
  - name: length_none
    extra: null
    overrides: {condition.length_match: none, model.arch: small_cnn}
"""


def _cfg(tmp_path: Path) -> dict:
    return {
        "output_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
        "seed_list": SEEDS,
        "eval": {"n_bootstrap": 200},
        "condition": {"url_mode": "norm", "length_match": "exact", "features": "data_only"},
        "qr": {"mask_mode": "fixed"},
        "model": {"arch": "small_cnn"},
    }


def _write_preds(root: Path, cid: str, stratum: str, seed: int, p, y, urls, groups) -> None:
    d = root / cid / stratum / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "preds_test.npz",
        y=np.asarray(y, dtype=np.int64),
        p=np.asarray(p, dtype=np.float64),
        group=np.asarray(groups).astype(str),
        url=np.asarray(urls).astype(str),
    )


def _write_results(root: Path, cid: str, stratum: str, ref: float) -> None:
    d = root / cid / stratum
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps({"aggregate": {"decoded_text_reference_auroc": ref}}), encoding="utf-8"
    )


@pytest.fixture
def synthetic(tmp_path: Path) -> dict:
    """같은 split을 공유하는 3조건 + split이 다른 1조건의 합성 test 예측."""
    cfg = _cfg(tmp_path)
    root = Path(cfg["output_dir"])
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        n_g, per_g = 30, 10
        groups = np.repeat([f"g{seed}_{i}" for i in range(n_g)], per_g)
        urls = np.array([f"http://s{seed}/{i}" for i in range(n_g * per_g)])
        y_size = n_g * per_g
        y = rng.integers(0, 2, size=y_size)
        score = rng.normal(0, 1, size=y_size)
        strong = 1 / (1 + np.exp(-(score + 1.6 * y)))
        mid = 1 / (1 + np.exp(-(score + 0.7 * y)))
        chance = 1 / (1 + np.exp(-score))
        _write_preds(root, BASE, "v2", seed, strong, y, urls, groups)
        _write_preds(root, POS, "v2", seed, mid, y, urls, groups)
        _write_preds(root, SHUF, "v2", seed, chance, y, urls, groups)
        # L-none은 표본 집합 자체가 다르다(행 수·URL이 다름) → 쌍체 불가
        m = y_size // 2
        _write_preds(root, NONE, "v2", seed, strong[:m], y[:m], urls[:m] + "x", groups[:m])
    for cid in (BASE, POS, SHUF, NONE):
        _write_results(root, cid, "v2", 0.97)
    (tmp_path / "matrix.yaml").write_text(MATRIX, encoding="utf-8")
    return {"cfg": cfg, "matrix": tmp_path / "matrix.yaml"}


def test_decoded_text_reference_flag_rejects_old_key() -> None:
    assert _decoded_text_reference({"decoded_text_reference": False}) is False
    assert _decoded_text_reference({}) is True
    with pytest.raises(ValueError, match="text_upper_bound"):
        _decoded_text_reference({"text_upper_bound": True})


def test_compare_conditions_paired_when_split_shared(synthetic) -> None:
    r = compare_conditions(synthetic["cfg"], BASE, POS, "v2", n_boot=200)
    assert r["paired"] is True
    assert r["delta_auroc"] > 0
    assert r["delta_ci"][0] <= r["delta_auroc"] <= r["delta_ci"][1]
    assert 0.0 < r["pseudo_p"] <= 1.0
    assert Path(r["path"]).exists()


def test_compare_conditions_unpaired_when_samples_differ(synthetic) -> None:
    r = compare_conditions(synthetic["cfg"], NONE, BASE, "v2", n_boot=200)
    assert r["paired"] is False
    assert "표본 집합이 다르다" in r["caveat"]


def test_compare_conditions_identical_predictions_give_zero_delta(synthetic) -> None:
    r = compare_conditions(synthetic["cfg"], BASE, BASE, "v2", n_boot=100, save=False)
    assert r["paired"] is True
    assert r["delta_auroc"] == pytest.approx(0.0)
    assert r["delta_ci"][0] <= 0.0 <= r["delta_ci"][1]


def test_compare_conditions_missing_preds_raises(synthetic) -> None:
    with pytest.raises(FileNotFoundError):
        compare_conditions(synthetic["cfg"], BASE, "does-not-exist", "v2", n_boot=10)


def test_run_hypothesis_tests_smoke(synthetic) -> None:
    out = run_hypothesis_tests(
        synthetic["cfg"], strata=["v2"], matrix_path=synthetic["matrix"], n_boot=200
    )
    ids = [t["hypothesis"] for t in out["tests"]]
    assert ids == ["H1", "H2", "H3", "H4"]
    by_id = {t["hypothesis"]: t for t in out["tests"]}
    assert by_id["H1"]["paired"] is True
    assert by_id["H1"]["estimate"] > 0
    # 참조 기준 예측(preds_test_charngram.npz)이 없는 옛 산출물이므로 H2는 기술 통계다.
    assert by_id["H2"]["paired"] is False
    assert by_id["H2"]["descriptive_only"] is True
    assert np.isnan(by_id["H2"]["pseudo_p"])
    assert "Bayes 최적 분류기가 아니다" in by_id["H2"]["caveat"]
    assert by_id["H3"]["paired"] is False  # L-none vs L-exact는 표본이 다르다
    assert by_id["H4"]["paired"] is True
    for t in out["tests"]:
        assert isinstance(t["ci_excludes_null"], bool)
        if t.get("descriptive_only"):
            # Holm family에서 제외된다.
            assert t["pseudo_p_holm"] is None and t["reject"] is None
            continue
        assert 0.0 < t["pseudo_p"] <= 1.0
        assert t["pseudo_p_holm"] >= t["pseudo_p"] - 1e-12
        assert isinstance(t["reject"], bool)
        # 판정과 CI가 일치해야 한다(p < alpha ⟺ 95% CI가 0을 배제).
        assert (t["pseudo_p"] < out["alpha"]) == t["ci_excludes_null"]
    assert out["n_tests_in_family"] == 3
    assert out["correction"] == "holm"
    assert json.loads(Path(out["path"]).read_text(encoding="utf-8"))["tests"]

    # 정식 검정(순열)과 기술 추정(부트스트랩)이 필드로 갈려 있어야 한다 — 표·본문이
    # pseudo-p를 정식 p값처럼 쓰지 못하게 하는 장치다(review_04 "통계 표 정리").
    for h in ("H1", "H4"):
        ft = by_id[h]["formal_test"]
        assert ft["method"] == "paired_cluster_permutation"
        assert 0.0 < ft["p"] <= 1.0
        assert "그룹 블록 null" in ft["null"]
        assert by_id[h]["descriptive"] is None
    for h in ("H2", "H3"):
        assert by_id[h]["formal_test"] is None
        assert by_id[h]["descriptive"]["ci"] == by_id[h]["ci"]
        assert "정식 검정 없음" in by_id[h]["descriptive"]["note"]
    assert "그룹 블록 null" in out["permutation_null"]
    assert "pseudo_p" in out["decision_rule"] and "부록" in out["decision_rule"]


def test_hypotheses_table_separates_formal_and_descriptive(synthetic) -> None:
    """정식 표에는 pseudo-p 열이 없고, 판정 문구가 귀무가설을 밝힌다."""
    import importlib.util

    run_hypothesis_tests(
        synthetic["cfg"], strata=["v2"], matrix_path=synthetic["matrix"], n_boot=200
    )
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_results_tables.py"
    spec = importlib.util.spec_from_file_location("make_results_tables", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPORTS = Path(synthetic["cfg"]["reports_dir"])
    mod.STRATA = ["v2"]

    formal = mod.hypotheses_table()
    header = formal.splitlines()[0]
    assert "pseudo-p" not in header
    assert "순열 p" in header and "근거 종류" in header
    assert "H0: ΔAUROC ≤ 0" in formal
    assert "기각" not in formal
    assert "쌍체 해석" in formal  # H3 각주
    assert "그룹 블록 null" in formal

    appendix = mod.pseudo_p_table()
    assert "pseudo-p (Holm)" in appendix.splitlines()[0]
    assert "정식 p값이 아니다" in appendix



def test_h2_is_paired_when_reference_predictions_exist(synthetic, tmp_path) -> None:
    """참조 기준의 test 예측이 있으면 H2가 쌍체 검정으로 승격된다 (코덱스 리뷰 1)."""
    from qrphish.runner import CHARNGRAM_PREDS

    root = Path(synthetic["cfg"]["output_dir"])
    rng = np.random.default_rng(7)
    for seed in SEEDS:
        with np.load(root / BASE / "v2" / f"seed{seed}" / "preds_test.npz") as z:
            y, urls, groups = z["y"], z["url"], z["group"]
        # CNN보다 확실히 강한 참조 기준(같은 행·같은 라벨).
        ref = 1 / (1 + np.exp(-(rng.normal(0, 1, size=y.size) + 3.0 * y)))
        np.savez_compressed(
            root / BASE / "v2" / f"seed{seed}" / CHARNGRAM_PREDS,
            y=y, p=ref, group=groups, url=urls,
        )
    out = run_hypothesis_tests(
        synthetic["cfg"], strata=["v2"], matrix_path=synthetic["matrix"], n_boot=200
    )
    h2 = {t["hypothesis"]: t for t in out["tests"]}["H2"]
    assert h2["paired"] is True
    assert h2["descriptive_only"] is False
    assert h2["estimate"] > 0 and h2["ci"][0] > 0
    assert 0.0 < h2["pseudo_p"] <= 1.0
    assert out["n_tests_in_family"] == 4  # H2가 family에 들어온다


# ------------------------------------------------------------- 시드 층화 집계
def test_load_seed_preds_fills_seed_from_path(synthetic) -> None:
    """``seed`` 배열이 없는 옛 preds도 파일 경로의 시드 번호로 채워 읽는다."""
    from qrphish.runner import _load_seed_preds, _pool_preds

    d = _load_seed_preds(synthetic["cfg"], BASE, "v2", 1)
    assert d is not None
    assert np.array_equal(d["seed"], np.full(d["y"].size, 1))
    pooled = _pool_preds(synthetic["cfg"], BASE, "v2", SEEDS)
    assert pooled is not None
    assert sorted(set(pooled["seed"].tolist())) == SEEDS


def test_compare_conditions_is_immune_to_per_seed_score_offsets(synthetic, tmp_path) -> None:
    """시드마다 점수 오프셋이 달라도 시드 층화 집계는 값이 흔들리지 않는다.

    시드를 섞어 하나의 AUROC를 내던 옛 방식이라면 오프셋만으로 추정치가 크게 떨어진다.
    """
    from qrphish.runner import _pool_preds

    cfg = synthetic["cfg"]
    root = Path(cfg["output_dir"])
    base = compare_conditions(cfg, BASE, POS, "v2", n_boot=200, save=False)

    # 두 조건 모두에 시드별 단조 증가 변환(로짓 오프셋)을 건다 — 시드 내 순위는 불변.
    for cid in (BASE, POS):
        for k, seed in enumerate(SEEDS):
            path = root / cid / "v2" / f"seed{seed}" / "preds_test.npz"
            with np.load(path) as z:
                d = {k2: z[k2] for k2 in z.files}
            logit = np.log(d["p"] / (1 - d["p"])) + 5.0 * k
            d["p"] = 1 / (1 + np.exp(-logit))
            np.savez_compressed(path, **d)

    shifted = compare_conditions(cfg, BASE, POS, "v2", n_boot=200, save=False)
    assert shifted["delta_auroc"] == pytest.approx(base["delta_auroc"], abs=1e-9)
    assert shifted["auroc_a"] == pytest.approx(base["auroc_a"], abs=1e-9)
    assert shifted["pooling"] == "seed_stratified"

    # 옛 방식(시드를 섞은 하나의 AUROC)이라면 같은 데이터에서 값이 크게 떨어졌을 것이다.
    from qrphish.evaluate import auroc

    pooled = _pool_preds(cfg, BASE, "v2", SEEDS)
    assert pooled is not None
    assert auroc(pooled["y"], pooled["p"]) < shifted["auroc_a"] - 0.05


def test_reaggregate_rewrites_pooled_ci_in_results_json(synthetic) -> None:
    """``reaggregate``는 저장된 preds로 auroc_pooled/CI/pooling을 덮어쓴다."""
    from qrphish.runner import reaggregate

    cfg = synthetic["cfg"]
    rp = Path(cfg["output_dir"]) / BASE / "v2" / "results.json"
    r = json.loads(rp.read_text(encoding="utf-8"))
    r["per_seed"] = [{"seed": s} for s in SEEDS]
    r["aggregate"]["auroc_pooled"] = 0.123
    r["aggregate"]["auroc_pooled_ci"] = [0.0, 1.0]
    rp.write_text(json.dumps(r), encoding="utf-8")

    out = reaggregate(cfg, strata=["v2"])
    assert str(rp) in out["updated"]
    agg = json.loads(rp.read_text(encoding="utf-8"))["aggregate"]
    assert agg["pooling"] == "seed_stratified"
    assert agg["auroc_pooled"] != 0.123
    assert agg["auroc_pooled_ci"] != [0.0, 1.0]
    # 점추정은 시드별 AUROC의 평균과 같아야 한다(표의 auroc_mean과 같은 정의).
    from qrphish.evaluate import auroc
    from qrphish.runner import _load_seed_preds

    per_seed = [
        auroc(d["y"], d["p"])
        for d in (_load_seed_preds(cfg, BASE, "v2", s) for s in SEEDS)
        if d is not None
    ]
    assert agg["auroc_pooled"] == pytest.approx(float(np.mean(per_seed)))
