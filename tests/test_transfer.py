"""외부 검증(F) 전이 평가 테스트 — `EXTERNAL_VALIDATION_DESIGN.md` 3·4·6·7절.

검증하는 계약 네 가지.

1. **fit/apply 분리 경로가 기존 함수와 같은 확률을 낸다.** 두 경로가 갈라지면 F-a의
   베이스라인 전이가 1차 결과와 비교 불가능해진다.
2. **순열 바닥선이 그룹 단위**다. 행 단위 순열보다 넓은 귀무 분포를 내야 하고, 같은 시드에서
   결정적이어야 한다.
3. **motif 재현성 지표(부호 일치율)가 정확하다.**
4. **F-a/F-b/F-c 전 경로가 합성 데이터에서 끝까지 돌고**, 결과 JSON이 설계 7.5 스키마 키를
   전부 들고 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrphish import baselines as bl
from qrphish.transfer import (
    REQUIRED_TRANSFER_KEYS,
    jaccard,
    motif_replication,
    permutation_null,
    sign_match_rate,
    verdict,
)

# 합성 데이터 규격: 모든 URL을 같은 바이트 길이로 만들어 L-exact 매칭이 행을 버리지 않게 한다.
URL_LEN = 40


def _make_urls(n_groups: int, per_group: int, label: int, tld_tag: str) -> list[str]:
    """길이가 정확히 ``URL_LEN``인 합성 URL. 클래스 신호는 채움 문자로만 준다.

    benign은 ``a``, phishing은 ``z``로 채운다 — 디코딩 텍스트에서도 QR 격자에서도 분리
    가능한 신호라 소형 CNN이 1 epoch로도 학습한다.
    """
    fill = "z" if label == 1 else "a"
    out = []
    for g in range(n_groups):
        for i in range(per_group):
            head = f"http://{tld_tag}{g:03d}.com/{i:03d}/"
            out.append(head + fill * (URL_LEN - len(head)))
    assert all(len(u) == URL_LEN for u in out)
    return out


def _frame(n_groups: int = 30, per_group: int = 8, tag: str = "wp") -> pd.DataFrame:
    """``load_webphish``와 같은 컬럼의 합성 프레임."""
    from qrphish.urls import etld1, path_depth

    urls = _make_urls(n_groups, per_group, 0, f"{tag}b") + _make_urls(
        n_groups, per_group, 1, f"{tag}p"
    )
    labels = [0] * (n_groups * per_group) + [1] * (n_groups * per_group)
    df = pd.DataFrame({"url": urls, "label": labels})
    df["group"] = [etld1(u) for u in df["url"]]
    df["url_len"] = df["url"].str.len().astype(int)
    df["url_bytes_len"] = [len(u.encode("utf-8")) for u in df["url"]]
    df["path_depth"] = [path_depth(u) for u in df["url"]]
    df["source"] = "synthetic"
    return df


def _write_webphish_csv(path: Path, df: pd.DataFrame) -> None:
    pd.DataFrame(
        {"Category": ["spam" if v else "ham" for v in df["label"]], "Data": df["url"]}
    ).to_csv(path, index=False, encoding="utf-8")


def _cfg(tmp_path: Path, csv: Path):
    from qrphish.config import from_dict

    return from_dict(
        {
            "seed_list": [0],
            "data": {"csv_path": str(csv)},
            "strata": ["v3"],
            # 합성 표본은 작다. 층 채택 규칙만 낮추고 나머지 조건은 주 조건 그대로 둔다.
            "stratum_rules": {"primary_min_per_class": 50, "secondary_min_per_class": 10},
            "split": {"max_group_frac": 0.2},
            "model": {"max_epochs": 1, "patience": 1, "batch_size": 64},
            "eval": {"n_bootstrap": 20},
            "output_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )


# ------------------------------------------------------------------ fit/apply 동치
def test_fit_apply_matches_legacy_charngram():
    """F-a가 쓰는 fit/apply 경로가 기존 ``charngram_lr``과 **같은 확률**을 낸다."""
    df = _frame(n_groups=10, per_group=4)
    urls = df["url"].to_numpy(dtype=object)
    y = df["label"].to_numpy()
    split = np.array([0, 1, 2] * (len(df) // 3) + [0] * (len(df) % 3))
    p_val, p_test = bl.charngram_lr(urls, y, split, seed=0)
    m = bl.fit_charngram_lr(urls[split == 0], y[split == 0], 0)
    np.testing.assert_allclose(bl.apply_charngram_lr(m, urls[split == 1]), p_val)
    np.testing.assert_allclose(bl.apply_charngram_lr(m, urls[split == 2]), p_test)


@pytest.mark.parametrize(
    ("legacy", "fit", "apply"),
    [
        (bl.bytehist_lr, bl.fit_bytehist_lr, bl.apply_bytehist_lr),
        (bl.length_lr, bl.fit_length_lr, bl.apply_length_lr),
    ],
)
def test_fit_apply_matches_legacy(legacy, fit, apply):
    df = _frame(n_groups=10, per_group=4)
    urls = df["url"].to_numpy(dtype=object)
    y = df["label"].to_numpy()
    split = np.array([0, 1, 2] * (len(df) // 3) + [0] * (len(df) % 3))
    p_val, p_test = legacy(urls, y, split, seed=0)
    m = fit(urls[split == 0], y[split == 0], 0)
    np.testing.assert_allclose(apply(m, urls[split == 1]), p_val)
    np.testing.assert_allclose(apply(m, urls[split == 2]), p_test)


def test_fit_apply_motifhist_matches_bag_of_patches():
    """motif 히스토그램 LR도 ``motifs.bag_of_patches_lr``(C=1.0)와 수치가 같다."""
    from qrphish.motifs import bag_of_patches_lr

    rng = np.random.default_rng(0)
    Xtr = rng.random((60, 16))
    Xte = rng.random((20, 16))
    y = np.array([0, 1] * 30)
    ref = bag_of_patches_lr(Xtr, y, Xte, seed=0)
    m = bl.fit_motifhist_lr(Xtr, y, seed=0)
    np.testing.assert_allclose(bl.apply_motifhist_lr(m, Xte), ref["p_test"])


def test_fit_apply_single_class_is_constant():
    """train에 한 클래스만 있으면 상수 확률(기존 ``_fit_lr`` 규약)."""
    urls = np.array(["http://a.com/x", "http://b.com/y"], dtype=object)
    m = bl.fit_bytehist_lr(urls, np.array([1, 1]), 0)
    assert np.allclose(bl.apply_bytehist_lr(m, urls), 1.0)


# ---------------------------------------------------------------------- 순열 바닥선
def _perm_inputs(n_groups: int = 40, per_group: int = 10):
    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    # 그룹이 라벨을 거의 결정한다 — 행 단위 순열이 이 구조를 깨는 것을 보이기 위한 설정.
    gy = rng.integers(0, 2, size=n_groups)
    y = np.repeat(gy, per_group)
    p = y * 0.6 + rng.random(y.size) * 0.4
    return y, p, groups


def test_permutation_null_is_deterministic():
    y, p, groups = _perm_inputs()
    a = permutation_null(y, p, groups, n_perm=30, seed=7)
    b = permutation_null(y, p, groups, n_perm=30, seed=7)
    assert a["auroc_mean"] == b["auroc_mean"]
    assert a["ci"] == b["ci"]
    assert a["unit"] == "etld1_group"


def test_permutation_null_group_unit_is_wider_than_row_unit():
    """그룹 단위 순열이 행 단위보다 **넓은** 귀무 분포를 낸다(설계 3.1 #6).

    행 단위 순열은 그룹 안의 라벨 상관을 깨서 바닥선을 너무 낮게 잡는다. 바닥선이 낮으면
    "우연보다 낫다"가 너무 쉽게 참이 되어 전이 성공 판정이 부풀려진다.
    """
    from qrphish.evaluate import auroc

    y, p, groups = _perm_inputs()
    grp = permutation_null(y, p, groups, n_perm=200, seed=0)
    rng = np.random.default_rng(0)
    row = np.array([auroc(rng.permutation(y), p) for _ in range(200)])
    grp_width = grp["ci"][1] - grp["ci"][0]
    assert grp_width > float(np.percentile(row, 97.5) - np.percentile(row, 2.5))
    assert grp["ci_upper"] == grp["ci"][1]


# ------------------------------------------------------------------ motif 재현성
def test_sign_match_rate_and_jaccard():
    wp = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
    ext = np.array([1.0, 1.0, 3.0, -0.1, -0.5])  # 0,2,3번만 부호 일치
    assert sign_match_rate(wp, ext, [0, 1, 2, 3]) == pytest.approx(0.75)
    assert sign_match_rate(wp, ext, [0, 1, 2, 3, 4]) == pytest.approx(0.6)
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)


def test_motif_replication_recovers_known_signal():
    """외부에서 같은 방향의 motif가 보이면 부호 일치율이 1.0, Spearman ρ가 양수다."""
    rng = np.random.default_rng(0)
    size, m = 2, 16
    n = 200
    y = np.array([0, 1] * (n // 2))
    groups = np.array([f"g{i // 4}" for i in range(n)])
    # enrichment는 **존재율** 오즈비라 0이 섞여 있어야 한다(전부 양수면 포화되어 방향이 없다).
    H = (rng.random((n, m)) < 0.3) * 0.01
    # motif 3은 phishing에서만, motif 7은 benign에서만 존재하게 만든다.
    H[:, 3] = np.where(y == 1, 0.5, 0.0)
    H[:, 7] = np.where(y == 0, 0.5, 0.0)
    enr_wp = {
        "motifs": [
            {"id": 3, "log_odds": 2.0, "present_phishing": 1.0, "present_benign": 0.1},
            {"id": 7, "log_odds": -2.0, "present_phishing": 0.1, "present_benign": 1.0},
        ],
        "top_phishing": [3],
        "top_benign": [7],
    }
    rep = motif_replication(enr_wp, H, y, groups, size=size, k=2, n_boot=50, seed=0)
    assert rep["sign_match_rate"]["value"] == pytest.approx(1.0)
    assert rep["spearman_logodds"]["rho"] > 0
    assert rep["K"] == 2
    assert set(rep) >= {"spearman_logodds", "sign_match_rate", "jaccard_topk", "ablation"}


# ---------------------------------------------------------------------- 사전 등록 판정
def test_verdict_prereg_labels():
    gates = {"length_lr_neutral": True, "version_lr_neutral": True,
             "bias_direction_ok": True, "group_concentration_ok": True}
    assert verdict(0.85, [0.80, 0.90], 0.60, 0.88, gates)["label"] == "reproduced_strong"
    assert verdict(0.72, [0.68, 0.76], 0.60, 0.88, gates)["label"] == "reproduced_weak"
    assert verdict(0.65, [0.62, 0.70], 0.60, 0.90, gates)["label"] == "partial_collapse"
    assert verdict(0.60, [0.55, 0.65], 0.60, 0.88, gates)["label"] == "collapse"
    bad = dict(gates, length_lr_neutral=False)
    assert verdict(0.85, [0.80, 0.90], 0.60, 0.88, bad)["label"] == "descriptive_only"


# ------------------------------------------------------------------------ 전 경로 스모크
@pytest.fixture(scope="module")
def _smoke(tmp_path_factory):
    """합성 WebPhish로 1차 학습 → 합성 외부 CSV로 F-a/F-b/F-c를 모두 돈다."""
    from qrphish.runner import MAIN_CONDITION, _run_one_seed, run_transfer

    tmp = tmp_path_factory.mktemp("transfer_smoke")
    wp = _frame(tag="wp")
    csv = tmp / "webphish.csv"
    _write_webphish_csv(csv, wp)
    cfg = _cfg(tmp, csv)

    # 1차 체크포인트(F-a가 재사용하고 F-c의 평가 대상이 된다).
    cond_dir = Path(cfg.output_dir) / MAIN_CONDITION
    res = _run_one_seed(cfg, "v3", 0, cond_dir, {"baselines": []})
    assert "error" not in res, res
    (cond_dir / "v3").mkdir(parents=True, exist_ok=True)
    (cond_dir / "v3" / "results.json").write_text(
        json.dumps({"per_seed": [res], "aggregate": {"auroc_mean": res["test"]["auroc"]}},
                   default=str),
        encoding="utf-8",
    )

    ext_csv = tmp / "external.csv"
    _frame(tag="ex").to_csv(ext_csv, index=False, encoding="utf-8")
    out = run_transfer(
        cfg, ext_csv, strata=["v3"], modes=("a", "b", "c"),
        source_tag="synth", n_perm=20, motif=False,
    )
    return cfg, out, tmp


@pytest.mark.parametrize("mode", ["a", "b", "c"])
def test_transfer_modes_run_and_write_schema(_smoke, mode):
    cfg, out, tmp = _smoke
    res = out[mode]["v3"]
    assert "error" not in res, res.get("error")
    missing = [k for k in REQUIRED_TRANSFER_KEYS if k not in res]
    assert not missing, f"결과 스키마에 없는 키: {missing}"
    assert res["mode"] == mode
    assert res["model"]["auroc_mean"] == res["model"]["auroc_mean"]  # NaN 아님
    assert res["null_permutation"]["unit"] == "etld1_group"
    assert "charngram_lr" in res["baselines"]
    assert res["verdict"]["criteria_version"] == "prereg_v1"

    path = (
        Path(cfg.reports_dir) / "transfer" / "synth" / mode
        / res["condition_id"] / "v3" / "results.json"
    )
    assert path.exists(), f"결과 JSON이 설계 7.5 경로에 없다: {path}"


def test_fa_reuses_webphish_threshold(_smoke):
    """zero-shot 임계값은 WebPhish val 값이어야 한다. 외부에서 다시 고르면 F-a가 아니다."""
    cfg, out, _ = _smoke
    fa = out["a"]["v3"]
    assert fa["model"]["threshold_source"] == "webphish_val"
    wp = json.loads(
        (Path(cfg.output_dir) / "norm-exact-data_only-fixed-small_cnn" / "v3" / "results.json")
        .read_text(encoding="utf-8")
    )
    assert fa["model"]["thresholds"][0] == pytest.approx(wp["per_seed"][0]["threshold"])


def test_external_artifacts_do_not_touch_stage1_paths(_smoke):
    """외부 층은 ``artifacts/external/`` 아래에만 쓴다(설계 9절 #19)."""
    cfg, _, _ = _smoke
    ext_root = Path(cfg.output_dir) / "external" / "synth"
    assert ext_root.exists()
    stage1 = Path(cfg.output_dir) / "norm-exact-data_only-fixed-small_cnn"
    assert sorted(p.name for p in stage1.iterdir()) == ["v3"]


def test_transfer_shape_guard(_smoke):
    """``ds.n != ckpt['n']``이면 명확한 ValueError (설계 3.4 #4)."""
    import torch

    from qrphish.runner import _predict_with_checkpoint

    cfg, _, tmp = _smoke
    sd = Path(cfg.output_dir) / "norm-exact-data_only-fixed-small_cnn" / "v3" / "seed0"
    ckpt = torch.load(sd / "model.pt", map_location="cpu", weights_only=False)
    ckpt["n"] = int(ckpt["n"]) + 4
    bad = tmp / "bad.pt"
    torch.save(ckpt, bad)
    with pytest.raises(ValueError, match="격자 크기가 체크포인트와 다르다"):
        _predict_with_checkpoint(cfg, bad, sd, np.ones(4, dtype=bool))


def test_prepare_frame_injection_parity(tmp_path):
    """같은 프레임을 주입 경로와 CSV 경로에 넣으면 결과가 같다.

    주입 리팩터가 두 경로를 갈라놓지 않았는지 보는 계약 테스트다.
    """
    from qrphish.runner import _prepare_frame

    df = _frame(tag="wp")
    csv = tmp_path / "webphish.csv"
    _write_webphish_csv(csv, df)
    cfg = _cfg(tmp_path, csv)
    a, _ = _prepare_frame(cfg, "v3", 0)
    b, _ = _prepare_frame(cfg, "v3", 0, frame=df)
    pd.testing.assert_frame_equal(
        a[["url", "label", "group", "split"]], b[["url", "label", "group", "split"]]
    )


# ------------------------------------------------------- 게이트: fail-closed / 승계
def test_gates_from_fails_closed_when_value_missing():
    """값을 못 낸 게이트는 통과가 아니라 **실패**다(fail-open 금지)."""
    from qrphish.transfer import gates_from

    notes: dict[str, str] = {}
    g = gates_from({"length_lr": {"auroc": 0.50}}, 0.1, True, notes=notes)
    assert g["length_lr_neutral"] is True
    assert g["version_lr_neutral"] is False  # 값 자체가 없다
    assert "version_lr_neutral" in notes
    g2 = gates_from({"length_lr": {"auroc": float("nan")}}, 0.1, True)
    assert g2["length_lr_neutral"] is False
    assert verdict(0.85, [0.80, 0.90], 0.60, 0.88, g2)["label"] == "descriptive_only"


def _write_bias_diag(path: Path, *, main_csv: str, b2_csv: str, main_ok: bool, b2_ok: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "main": {"gate_passed": main_ok, "gate_note": "main", "output_csv": main_csv},
                "ext_b2": {"gate_passed": b2_ok, "gate_note": "b2", "output_csv": b2_csv},
            }
        ),
        encoding="utf-8",
    )


def test_collection_bias_gate_matches_block_by_output_csv(tmp_path):
    from qrphish.transfer import collection_bias_gate

    main_csv = tmp_path / "external_2026-09-08.csv"
    b2_csv = tmp_path / "external_b2_2026-09-08.csv"
    _write_bias_diag(tmp_path / "bias_diagnostics.json", main_csv=str(main_csv),
                     b2_csv=str(b2_csv), main_ok=True, b2_ok=False)
    assert collection_bias_gate(main_csv)["gate_passed"] is True
    got = collection_bias_gate(b2_csv)
    assert got["gate_passed"] is False and got["block"] == "ext_b2"
    assert collection_bias_gate(tmp_path / "없는파일.csv") is not None  # main으로 폴백


def test_collection_bias_gate_reads_reports_external(tmp_path):
    """CSV 옆에 없으면 ``{reports}/external/bias_diagnostics.json``을 본다."""
    from qrphish.transfer import collection_bias_gate

    csv = tmp_path / "data" / "external_b2_x.csv"
    csv.parent.mkdir(parents=True)
    _write_bias_diag(tmp_path / "reports" / "external" / "bias_diagnostics.json",
                     main_csv="a.csv", b2_csv="b.csv", main_ok=True, b2_ok=False)
    got = collection_bias_gate(csv, tmp_path / "reports")
    assert got is not None and got["gate_passed"] is False and got["block"] == "ext_b2"


def test_collection_gate_fail_forces_descriptive_only(_smoke, tmp_path):
    """EXT-B2처럼 수집 단계 게이트가 fail이면 전이 단계에서 재계산하지 않고 승계한다."""
    from qrphish.runner import run_transfer

    cfg, _, tmp = _smoke
    cfg2 = _cfg(tmp_path, Path(cfg.data.csv_path))
    # 1차 체크포인트는 스모크가 만든 것을 그대로 쓴다(F-a는 학습하지 않는다).
    import shutil

    shutil.copytree(Path(cfg.output_dir), Path(cfg2.output_dir), dirs_exist_ok=True)
    ext_csv = tmp_path / "external_b2_synth.csv"
    _frame(tag="ex").to_csv(ext_csv, index=False, encoding="utf-8")
    _write_bias_diag(tmp_path / "bias_diagnostics.json", main_csv="other.csv",
                     b2_csv=str(ext_csv), main_ok=True, b2_ok=False)

    out = run_transfer(cfg2, ext_csv, strata=["v3"], modes=("a",), source_tag="b2synth",
                       n_perm=20, motif=False)
    res = out["a"]["v3"]
    assert res["data"]["bias_gate"]["source"] == "collection"
    assert res["data"]["bias_gate"]["gate_passed"] is False
    assert res["verdict"]["gates_passed"]["bias_direction_ok"] is False
    assert res["verdict"]["label"] == "descriptive_only"


def test_external_artifact_dirs_are_separated_by_mode(_smoke):
    """F-a/F-b/F-c가 같은 층의 ``grids.npz``를 덮어쓰면 안 된다."""
    cfg, _, _ = _smoke
    root = Path(cfg.output_dir) / "external" / "synth"
    modes = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert modes == ["a", "b"]  # F-c는 F-b의 체크포인트를 읽기만 한다
    for m in modes:
        assert (root / m / "norm-exact-data_only-fixed-small_cnn" / "v3").exists()
