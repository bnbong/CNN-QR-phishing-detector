"""explain — enrichment 계산, 부호 있는 타깃, IG 완결성, sanity check.

여기서 만든 소형 합성 학습 setup(:func:`build_trained`)은 probes/occlusion 테스트도
공유한다. 실제 Colab 체크포인트 없이 CPU에서 2에폭 학습해 같은 형식의 ``model.pt``를
남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from qrphish.dataset import QRGridDataset, _center_pad
from qrphish.explain import (
    area_fraction_by_kind,
    balanced_sample,
    cam_correlation,
    enrichment_by_kind,
    gradcam_signed,
    ig_logit_gap,
    integrated_gradients,
    mass_fraction_by_kind,
    randomized_model_cam,
    signed_target,
)
from qrphish.mapping import provenance
from qrphish.models import SmallCNN
from qrphish.qrgen import encode

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


@dataclass
class Setup:
    """합성 층 하나 + 2에폭 학습된 SmallCNN."""

    model: SmallCNN
    ds: QRGridDataset
    urls: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    split: np.ndarray
    ec: int
    n: int
    ckpt_path: Path


def _synthetic_urls(n_per_class: int) -> tuple[list[str], list[int], list[str]]:
    """benign은 숫자 없음, phishing은 숫자 + 피싱 키워드. 어휘 신호를 일부러 심는다."""
    urls: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for i in range(n_per_class):
        w = WORDS[i % len(WORDS)]
        v = WORDS[(i + 3) % len(WORDS)]
        urls.append(f"http://{w}shop.com/{v}/page")
        labels.append(0)
        groups.append(f"b{i % 6}")
        urls.append(f"http://secure-login{i:02d}.com/verify/{i:03d}/account.php?id={i}")
        labels.append(1)
        groups.append(f"p{i % 6}")
    return urls, labels, groups


def build_trained(tmp_path: Path, n_per_class: int = 24, epochs: int = 2, seed: int = 0) -> Setup:
    """합성 URL -> QR 격자 -> QRGridDataset -> 2에폭 학습 -> ``model.pt`` 저장."""
    import qrcode

    ec = qrcode.constants.ERROR_CORRECT_L
    urls, labels, groups = _synthetic_urls(n_per_class)
    arts = [encode(u, ec=ec, mask_pattern=0) for u in urls]
    n_max = max(a.modules.shape[0] for a in arts)
    grids = [_center_pad(np.asarray(a.modules, dtype=bool), n_max) for a in arts]
    X = np.stack(grids).reshape(len(grids), -1)
    X_packed = np.packbits(X, axis=1)
    y = np.asarray(labels, dtype=np.uint8)
    versions = np.asarray([a.version for a in arts], dtype=np.int32)

    rng = np.random.default_rng(seed)
    split = np.where(rng.random(len(y)) < 0.5, 0, 2).astype(np.uint8)
    # 두 split 모두 양·음성이 있어야 AUROC가 정의된다.
    split[:4] = 0
    split[4:8] = 2

    ds = QRGridDataset(X_packed, y, n_max, versions, ec, features="data_only")

    torch.manual_seed(seed)
    model = SmallCNN(in_ch=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    tr = np.nonzero(split == 0)[0]
    model.train()
    for _ in range(epochs):
        for s in range(0, tr.size, 16):
            chunk = tr[s : s + 16]
            xb = torch.stack([ds[int(i)][0] for i in chunk])
            yb = torch.stack([ds[int(i)][1] for i in chunk])
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()

    ckpt = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "n": n_max, "arch": "small_cnn"}, ckpt)
    return Setup(
        model=model, ds=ds, urls=np.asarray(urls), y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups), split=split, ec=ec, n=n_max, ckpt_path=ckpt,
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> Setup:
    return build_trained(tmp_path_factory.mktemp("explain"))


def _prov(url: str, ec: int):
    art = encode(url, ec=ec, mask_pattern=0)
    return provenance(url, art), art.modules.shape[0]


# ------------------------------------------------------------------ enrichment
def test_area_fraction_sums_to_one(setup: Setup) -> None:
    prov, _ = _prov(str(setup.urls[0]), setup.ec)
    area = area_fraction_by_kind(prov)
    assert pytest.approx(1.0, abs=1e-12) == sum(area.values())
    assert area["function"] > 0 and area["char"] > 0


def test_mass_fraction_sums_to_one(setup: Setup) -> None:
    prov, n_art = _prov(str(setup.urls[0]), setup.ec)
    rng = np.random.default_rng(0)
    mass = mass_fraction_by_kind(rng.random((n_art, n_art)), prov)
    assert pytest.approx(1.0, abs=1e-12) == sum(mass.values())


def test_uniform_cam_enrichment_is_one(setup: Setup) -> None:
    """균일 attribution이면 모든 kind에서 enrichment가 정확히 1이어야 한다."""
    prov, n_art = _prov(str(setup.urls[0]), setup.ec)
    e = enrichment_by_kind(np.ones((n_art, n_art)), prov)
    for k, v in e.items():
        if np.isnan(v):
            continue
        assert pytest.approx(1.0, abs=1e-9) == v, k


def test_enrichment_detects_concentration(setup: Setup) -> None:
    """char 영역에만 질량을 몰아주면 char enrichment > 1, 나머지는 0."""
    prov, n_art = _prov(str(setup.urls[0]), setup.ec)
    cam = np.zeros((n_art, n_art))
    cam[np.asarray(prov.kind) == 1] = 1.0
    e = enrichment_by_kind(cam, prov)
    assert e["char"] > 1.0
    assert e["function"] == 0.0


def test_random_attribution_baseline_near_one(setup: Setup) -> None:
    from qrphish.explain import random_attribution_enrichment

    prov, n_art = _prov(str(setup.urls[0]), setup.ec)
    e = random_attribution_enrichment((n_art, n_art), prov, seed=1, n_rep=30)
    for k, v in e.items():
        assert abs(v - 1.0) < 0.25, (k, v)


# ------------------------------------------------------------------ 부호·표본
def test_signed_target() -> None:
    assert signed_target(1) == 1.0
    assert signed_target(0) == -1.0


def test_gradcam_signed_differs_by_sign(setup: Setup) -> None:
    x, _ = setup.ds[0]
    pos = gradcam_signed(setup.model, x, setup.model.target_layer, sign=1.0)
    neg = gradcam_signed(setup.model, x, setup.model.target_layer, sign=-1.0)
    assert pos.shape == (setup.n, setup.n)
    # 부호를 뒤집으면 ReLU가 반대쪽을 살리므로 같은 맵이 나오면 안 된다.
    assert not np.allclose(pos, neg)


def test_balanced_sample_is_class_balanced(setup: Setup) -> None:
    rows = np.nonzero(setup.split == 2)[0]
    idx = balanced_sample(setup.y, rows, per_class=3, seed=0)
    assert set(idx).issubset(set(rows))
    counts = np.bincount(setup.y[idx], minlength=2)
    assert counts[0] == counts[1] == 3
    # 같은 시드면 같은 표본
    assert np.array_equal(idx, balanced_sample(setup.y, rows, per_class=3, seed=0))


def test_sanity_randomized_cam(setup: Setup) -> None:
    """파라미터를 무작위화한 CAM은 원 CAM과 완전히 같지 않아야 한다."""
    x, _ = setup.ds[0]
    cam = gradcam_signed(setup.model, x, setup.model.target_layer, sign=1.0)
    rnd = randomized_model_cam(setup.model, x, sign=1.0, seed=3)
    assert rnd.shape == cam.shape
    corr = cam_correlation(cam, rnd)
    assert -1.0 <= corr <= 1.0
    assert not np.allclose(cam, rnd)


# ------------------------------------------------------------------ IG 완결성
def test_integrated_gradients_completeness(setup: Setup) -> None:
    """IG 기여도 합 ≈ f(x) - f(baseline)."""
    x, _ = setup.ds[0]
    attr = integrated_gradients(setup.model, x, baseline=0.0, steps=256, sign=1.0)
    gap = ig_logit_gap(setup.model, x, baseline=0.0, sign=1.0)
    assert attr.shape == (2, setup.n, setup.n)
    assert abs(float(attr.sum()) - gap) <= 0.05 * max(abs(gap), 1e-3)


def test_integrated_gradients_sign_flips(setup: Setup) -> None:
    x, _ = setup.ds[0]
    a = integrated_gradients(setup.model, x, steps=16, sign=1.0)
    b = integrated_gradients(setup.model, x, steps=16, sign=-1.0)
    assert np.allclose(a, -b, atol=1e-8)
