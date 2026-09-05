"""dataset 빌드와 end-to-end 스모크.

워커 A(config/urls/splits) · 워커 B(qrgen/mapping) 모듈이 아직 없으면 importorskip으로
건너뛴다. 통합 패스 이후에는 전부 실행되어야 한다.
"""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pandas as pd
import pytest
import qrcode
import torch

qrgen = pytest.importorskip("qrphish.qrgen", reason="워커 B의 qrgen 미구현")
mapping = pytest.importorskip("qrphish.mapping", reason="워커 B의 mapping 미구현")

from qrphish.dataset import (  # noqa: E402
    QRGridDataset,
    build_stratum,
    ec_const,
    load_stratum,
    load_stratum_arrays,
    stratum_tier,
    version_in_spec,
)


class _Cond:
    """워커 A의 Condition 대역. build_stratum이 읽는 속성만 갖는다."""

    url_mode = "norm"
    length_match = "exact"
    length_bucket = 1
    features = "data_only"
    path_filter = False
    shuffle_positions = False


class _QR:
    ec = "L"
    mask_mode = "fixed"
    mask_pattern = 0
    pad_mode = "spec"


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """두 클래스 × 랜덤 URL. 길이는 v3(33~53바이트) 안에 들어가게 만든다."""
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz0123456789-"
    rows = []
    for i in range(n):
        label = i % 2
        host = "".join(rng.choice(alpha) for _ in range(rng.randint(6, 10)))
        path = "".join(rng.choice(alpha) for _ in range(rng.randint(14, 24)))
        url = f"{host}.com/{path}"
        url = url[:45].ljust(40, "x")
        rows.append(
            {
                "url": url,
                "label": label,
                "group": f"g{i % 20}",
                "url_len": len(url),
                "path_depth": 1,
                "split": 0 if i % 10 < 6 else (1 if i % 10 < 8 else 2),
            }
        )
    return pd.DataFrame(rows)


def test_version_in_spec() -> None:
    assert version_in_spec(3, "v3") and not version_in_spec(3, "v2")
    assert version_in_spec(5, "v5plus") and version_in_spec(9, "v5plus")
    assert not version_in_spec(4, "v5plus")


def test_stratum_tier_rules() -> None:
    assert stratum_tier(1500) == "primary"
    assert stratum_tier(500) == "secondary"
    assert stratum_tier(120) == "dropped"


def _build(tmp_path, df=None, cond=None, qr=None):
    df = _synthetic_df() if df is None else df
    ec = ec_const("L")
    df = df.assign(version=[qrgen.natural_version(u, ec) for u in df["url"]])
    spec = f"v{int(df['version'].mode().iloc[0])}"
    df = df[df["version"] == int(spec[1:])].reset_index(drop=True)
    out = tmp_path / "st"
    sm = build_stratum(df, spec, cond or _Cond(), out, qr=qr or _QR(), condition_id="test")
    return df, spec, out, sm


def test_build_stratum_artifacts(tmp_path) -> None:
    df, spec, out, sm = _build(tmp_path)
    assert (out / "grids.npz").exists()
    assert (out / "meta.parquet").exists()
    assert json.loads((out / "stratum_meta.json").read_text())["stratum"] == spec

    X_packed, y, meta = load_stratum(out)
    assert len(meta) == len(df) == len(y)
    for col in (
        "row_id", "url", "label", "group", "url_len", "path_depth", "split",
        "version", "mask_used", "n_pad_bytes", "first_pad_codeword",
    ):
        assert col in meta.columns
    assert sm.n == 4 * int(spec[1:]) + 17


def test_npz_packbits_roundtrip(tmp_path) -> None:
    """스펙 9절 #13 — packbits/unpackbits 무손실."""
    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    n = int(arr["n"])
    bits = np.unpackbits(arr["X_packed"], axis=1)[:, : n * n]
    ec = int(arr["ec"])
    for i in range(0, len(df), 37):
        art = qrgen.encode(df["url"].iloc[i], ec=ec, mask_pattern=0)
        assert np.array_equal(bits[i].reshape(n, n).astype(bool), np.asarray(art.modules, bool))


def test_tensor_shape_and_channels(tmp_path) -> None:
    """스펙 9절 #12 — 모든 조건에서 (2, n, n), n == 4*version+17."""
    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    n = int(arr["n"])
    for features in ("data_only", "all"):
        ds = QRGridDataset(
            arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]), features=features
        )
        x, y = ds[0]
        assert x.shape == (2, n, n) and x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert set(np.unique(x[1].numpy())) <= {0.0, 1.0}
        if features == "data_only":
            # 기능 패턴 위치의 값 채널은 0이어야 한다
            assert float((x[0] * (1 - x[1])).abs().sum()) == 0.0


def test_shuffle_positions_permutes_but_preserves_multiset(tmp_path) -> None:
    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    n = int(arr["n"])
    base = QRGridDataset(arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]))
    shuf = QRGridDataset(
        arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]),
        shuffle_positions=True, perm_seed=0,
    )
    x0 = base[0][0].numpy()
    x1 = shuf[0][0].numpy()
    assert x0.shape == x1.shape
    assert not np.array_equal(x0, x1)
    assert np.array_equal(np.sort(x0.reshape(2, -1), axis=1), np.sort(x1.reshape(2, -1), axis=1))


def test_label_shuffle_changes_labels_only(tmp_path) -> None:
    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    n = int(arr["n"])
    base = QRGridDataset(arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]))
    shuf = QRGridDataset(
        arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]), label_shuffle_seed=1
    )
    assert np.array_equal(np.sort(base.y), np.sort(shuf.y))
    assert not np.array_equal(base.y, shuf.y)
    assert np.array_equal(base[0][0].numpy(), shuf[0][0].numpy())


def test_deterministic_build(tmp_path) -> None:
    """스펙 9절 #11 — 같은 입력 2회 빌드 시 X_packed/split/group_id 바이트 동일."""
    df = _synthetic_df()
    a = _build(tmp_path / "a", df=df)[2]
    b = _build(tmp_path / "b", df=df)[2]
    A, B = load_stratum_arrays(a), load_stratum_arrays(b)
    for k in ("X_packed", "split", "group_id", "y"):
        assert np.array_equal(A[k], B[k])


def test_end_to_end_smoke(tmp_path) -> None:
    """dataset build → 2 epoch 학습(CPU) → evaluate → results.json 스키마."""
    from torch.utils.data import DataLoader

    from qrphish.evaluate import metrics_from_probs, predict_probs
    from qrphish.models import build_model
    from qrphish.train import train_model

    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    meta = pd.read_parquet(out / "meta.parquet")
    n = int(arr["n"])
    ds = QRGridDataset(arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]))
    split = arr["split"].astype(int)

    import copy as _copy

    def loader(code, shuffle):
        sub = _copy.copy(ds)
        sub.indices = np.nonzero(split == code)[0]
        return DataLoader(sub, batch_size=32, shuffle=shuffle), sub.indices

    tr, _ = loader(0, True)
    va, iva = loader(1, False)
    te, ite = loader(2, False)

    model = build_model("small_cnn", n, ds.canonical_data_mask)
    res = train_model(
        model, tr, va, max_epochs=2, patience=2, amp=False, seed=0, device="cpu"
    )
    assert res.epochs_run <= 2
    assert 0.0 <= res.threshold <= 1.0

    y_te, p_te = predict_probs(model, te, device=torch.device("cpu"))
    assert y_te.size == len(ite)
    m = metrics_from_probs(
        y_te, p_te, meta["group"].to_numpy()[ite], n_boot=30, seed=0, threshold=res.threshold
    )
    assert 0.0 <= m["auroc"] <= 1.0 or np.isnan(m["auroc"])

    results = {
        "schema_version": 1,
        "condition_id": "test",
        "stratum": spec,
        "tier": sm.tier,
        "per_seed": [{"seed": 0, "best_epoch": res.best_epoch, "threshold": res.threshold,
                      "test": {k: m[k] for k in ("auroc", "auprc", "f1", "acc",
                                                 "auroc_ci", "f1_ci")},
                      "val": {"auroc": res.best_val_auroc},
                      "group_perf_iqr": m["group_perf_iqr"]}],
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results, default=str))
    loaded = json.loads(p.read_text())
    assert loaded["per_seed"][0]["test"]["auroc_ci"][0] <= loaded["per_seed"][0]["test"]["auroc_ci"][1]


def test_baselines_on_split(tmp_path) -> None:
    from qrphish.baselines import get_baseline

    df, spec, out, sm = _build(tmp_path)
    meta = pd.read_parquet(out / "meta.parquet")
    arr = load_stratum_arrays(out)
    split = arr["split"].astype(int)
    urls, y = meta["url"].to_numpy(), meta["label"].to_numpy()
    for name in ("charngram_lr", "version_lr", "length_lr", "bytehist_lr", "maskindex_lr"):
        pv, pt = get_baseline(name)(urls, y, split, meta=meta, seed=0)
        assert pv.shape[0] == int((split == 1).sum())
        assert pt.shape[0] == int((split == 2).sum())
        assert np.all((pt >= 0) & (pt <= 1))
    with pytest.raises(NotImplementedError):
        get_baseline("charcnn")(urls, y, split, meta=meta)


def test_gradcam_shape(tmp_path) -> None:
    from qrphish.explain import attribute_by_kind, gradcam
    from qrphish.models import SmallCNN

    df, spec, out, sm = _build(tmp_path)
    arr = load_stratum_arrays(out)
    n = int(arr["n"])
    ds = QRGridDataset(arr["X_packed"], arr["y"], n, arr["versions"], int(arr["ec"]))
    model = SmallCNN()
    cam = gradcam(model, ds[0][0], model.last_conv_block)
    assert cam.shape == (n, n)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6

    art = qrgen.encode(str(df["url"].iloc[0]), ec=int(arr["ec"]), mask_pattern=0)
    prov = mapping.provenance(str(df["url"].iloc[0]), art)
    frac = attribute_by_kind(cam, prov)
    total = sum(frac.values())
    # 학습되지 않은 모델은 ReLU 후 CAM이 전부 0일 수 있다. 그 경우 비율은 0으로 정의된다.
    assert total == pytest.approx(0.0) or total == pytest.approx(1.0)
    assert set(frac) >= {"function", "char", "pad", "ec", "remainder"}


def test_runner_end_to_end(tmp_path, monkeypatch) -> None:
    """runner 레벨 스모크: P0 → P1 1조건 → results.json → reports/table_*.csv."""
    import csv
    import shutil
    from pathlib import Path

    pytest.importorskip("qrphish.config", reason="워커 A의 config 미구현")
    pytest.importorskip("qrphish.urls", reason="워커 A의 urls 미구현")
    pytest.importorskip("qrphish.splits", reason="워커 A의 splits 미구현")

    from qrphish.config import load_config
    from qrphish.runner import aggregate_reports, apply_overrides, run_matrix, run_p0

    rng = random.Random(0)
    alpha = "abcdefghijklmnopqrstuvwxyz0123456789"
    rows = [["Category", "Data"]]
    for i in range(300):
        host = "".join(rng.choice(alpha) for _ in range(rng.randint(5, 9)))
        path = "".join(rng.choice(alpha) for _ in range(rng.randint(15, 25)))
        rows.append(["spam" if i % 2 else "ham", f"http://www.{host}.com/{path}"[:50]])
    csv_path = tmp_path / "webphish.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "configs", tmp_path / "configs", dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)

    cfg = apply_overrides(
        load_config("configs/base.yaml"),
        {
            "data.csv_path": str(csv_path),
            "seed_list": [0],
            "strata": ["v3"],
            "stratum_rules.primary_min_per_class": 40,
            "stratum_rules.secondary_min_per_class": 10,
            "model.max_epochs": 1,
            "model.patience": 1,
            "model.batch_size": 32,
            "model.amp": False,
            "eval.n_bootstrap": 20,
        },
    )

    p0 = run_p0(cfg)
    assert p0["roundtrip_gate"]["passed"], p0["roundtrip_gate"]["failures"]
    assert (tmp_path / "reports" / "stratum_counts.json").exists()
    assert (tmp_path / "reports" / "split_diagnostics.json").exists()
    assert Path(p0["first_pad_png"]).exists()

    matrix = {"P1": [
        {"name": "main_cnn", "overrides": {"model.arch": "small_cnn"},
         "text_upper_bound": True, "baselines": ["length_lr"]}
    ]}
    mpath = tmp_path / "mini_matrix.yaml"
    mpath.write_text(json.dumps(matrix))  # JSON은 YAML의 부분집합이다

    res = run_matrix(cfg, "P1", mpath)
    assert len(res) == 1
    r = res[0]
    assert r["schema_version"] == 1
    for key in ("condition_id", "stratum", "tier", "config", "provenance", "data",
                "per_seed", "aggregate", "sanity", "explain"):
        assert key in r
    assert r["data"]["n_total"] == sum(
        r["data"][k] for k in ("n_train", "n_val", "n_test")
    )
    assert "length_lr" in r["per_seed"][0]["baselines"]
    assert "charngram_lr" in r["per_seed"][0]["baselines"]

    # 재실행하면 results.json이 있으므로 건너뛴다
    assert run_matrix(cfg, "P1", mpath)[0]["condition_id"] == r["condition_id"]

    tables = aggregate_reports(cfg)
    assert tables["main"].exists()
    df = pd.read_csv(tables["main"])
    assert len(df) == 1 and df.loc[0, "condition_id"] == r["condition_id"]


# --- Codex 리뷰 후속 회귀 테스트 -------------------------------------------


def test_canonical_data_mask_is_union_over_versions():
    """v5plus처럼 버전이 섞인 층에서 최빈 버전 마스크만 쓰면 위치가 누락된다."""
    import numpy as np

    from qrphish.dataset import QRGridDataset, data_mask_for

    ec = qrcode.constants.ERROR_CORRECT_L
    n_max = 4 * 6 + 17  # v6 격자에 v5를 중앙 패딩
    versions = np.array([5, 5, 5, 6], dtype=np.int32)  # 최빈은 v5
    y = np.zeros(len(versions), dtype=np.uint8)
    X_packed = np.zeros((len(versions), (n_max * n_max + 7) // 8), dtype=np.uint8)
    ds = QRGridDataset(X_packed, y, n_max, versions, ec)

    got = ds.canonical_data_mask
    want = data_mask_for(5, ec, n_max).astype(bool) | data_mask_for(6, ec, n_max).astype(bool)
    assert np.array_equal(got, want)
    # 최빈 버전(v5) 마스크만 쓰면 실제로 위치가 빠진다 = 이 테스트가 자명하지 않다.
    assert got.sum() > data_mask_for(5, ec, n_max).astype(bool).sum()


def test_label_shuffle_leaves_test_labels_intact():
    """영가설 바닥은 train+val만 섞고 test는 원 라벨로 둔다(스펙 1.8 H1)."""
    import numpy as np

    from qrphish.dataset import QRGridDataset

    n = 21
    y = np.array([0, 1] * 50, dtype=np.uint8)
    split = np.array([0] * 60 + [1] * 20 + [2] * 20, dtype=np.uint8)
    X_packed = np.zeros((len(y), (n * n + 7) // 8), dtype=np.uint8)
    ds = QRGridDataset(
        X_packed, y, n, 1, qrcode.constants.ERROR_CORRECT_L,
        label_shuffle_seed=123, label_shuffle_rows=(split != 2),
    )
    te = split == 2
    assert np.array_equal(ds.y[te], y[te].astype(np.float32))
    # train+val 안에서만 섞이므로 그 구간의 라벨 합은 보존되고 순서는 바뀐다.
    trva = ~te
    assert ds.y[trva].sum() == y[trva].sum()
    assert not np.array_equal(ds.y[trva], y[trva].astype(np.float32))


def test_attribute_to_chars_handles_non_ascii_url():
    """provenance.char_index는 UTF-8 바이트 인덱스다. 문자 인덱스로 환산해야 한다."""
    import numpy as np

    from qrphish.explain import attribute_to_chars, byte_to_char_index
    from qrphish.mapping import KIND_CHAR, provenance

    url = "http://에이비.kr/가나다"  # 멀티바이트 문자 포함
    assert len(url.encode("utf-8")) > len(url)

    art = qrgen.encode(url, ec=qrcode.constants.ERROR_CORRECT_L, mask_pattern=0)
    prov = provenance(url, art)

    b2c = byte_to_char_index(url)
    assert b2c.size == len(url.encode("utf-8"))
    assert b2c[0] == 0 and b2c[-1] == len(url) - 1

    # CAM 전체를 1로 두면 char 모듈이 하나라도 있는 문자는 1.0이 되어야 한다.
    cam = np.ones_like(prov.kind, dtype=np.float64)
    out = attribute_to_chars(cam, prov, url)
    assert out.shape == (len(url),)
    assert out.max() == pytest.approx(1.0)
    # 모든 문자가 최소 한 모듈을 갖는다(짧은 URL이라 전 바이트가 배치된다).
    assert (out > 0).all()

    # 귀속된 모듈 수로 바이트/문자 구분을 확인한다. 3바이트 한글 문자에는 ASCII
    # 문자의 3배(≈24개) 모듈이 걸려야 한다. 바이트 인덱스를 그대로 썼다면 모든
    # 인덱스가 8개씩 균일하게 나온다.
    sel = prov.kind == KIND_CHAR
    keys = byte_to_char_index(url)[prov.char_index[sel].astype(np.int64)]
    per_char = np.bincount(keys, minlength=len(url))
    ascii_idx = [i for i, ch in enumerate(url) if len(ch.encode("utf-8")) == 1]
    multi_idx = [i for i, ch in enumerate(url) if len(ch.encode("utf-8")) == 3]
    assert multi_idx, "멀티바이트 문자가 있어야 의미 있는 테스트다"
    assert per_char[ascii_idx].max() == 8
    assert per_char[multi_idx].min() == 24


def test_train_raises_when_val_has_single_class():
    """val AUROC가 NaN이면 조기종료 기준이 무의미하므로 예외로 끊는다."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from qrphish.dataset import QRGridDataset
    from qrphish.models import SmallCNN
    from qrphish.train import train_model

    n = 21
    y = np.array([0, 1, 0, 1, 1, 1], dtype=np.uint8)
    X_packed = np.zeros((len(y), (n * n + 7) // 8), dtype=np.uint8)
    ds = QRGridDataset(X_packed, y, n, 1, qrcode.constants.ERROR_CORRECT_L)
    tr = copy.copy(ds)
    tr.indices = np.array([0, 1, 2, 3])
    va = copy.copy(ds)
    va.indices = np.array([4, 5])  # 전부 phishing
    with pytest.raises(ValueError, match="val AUROC가 NaN"):
        train_model(
            SmallCNN(),
            DataLoader(tr, batch_size=2),
            DataLoader(va, batch_size=2),
            max_epochs=1,
            patience=1,
            amp=False,
            device=torch.device("cpu"),
        )
