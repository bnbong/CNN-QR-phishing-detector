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
    assert 0.0 < r["p_value"] <= 1.0
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
    assert np.isnan(by_id["H2"]["p_value"])
    assert "Bayes 최적 분류기가 아니다" in by_id["H2"]["caveat"]
    assert by_id["H3"]["paired"] is False  # L-none vs L-exact는 표본이 다르다
    assert by_id["H4"]["paired"] is True
    for t in out["tests"]:
        assert isinstance(t["ci_excludes_null"], bool)
        if t.get("descriptive_only"):
            # Holm family에서 제외된다.
            assert t["p_holm"] is None and t["reject"] is None
            continue
        assert 0.0 < t["p_value"] <= 1.0
        assert t["p_holm"] >= t["p_value"] - 1e-12
        assert isinstance(t["reject"], bool)
        # 판정과 CI가 일치해야 한다(p < alpha ⟺ 95% CI가 0을 배제).
        assert (t["p_value"] < out["alpha"]) == t["ci_excludes_null"]
    assert out["n_tests_in_family"] == 3
    assert out["correction"] == "holm"
    assert json.loads(Path(out["path"]).read_text(encoding="utf-8"))["tests"]


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
    assert 0.0 < h2["p_value"] <= 1.0
    assert out["n_tests_in_family"] == 4  # H2가 family에 들어온다
