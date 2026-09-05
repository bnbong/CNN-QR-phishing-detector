"""모델 형상 테스트 (스펙 9절 #12)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qrphish.models import BitMLP, LinearProbe, SmallCNN, build_model


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 8])
def test_smallcnn_shapes(version: int) -> None:
    n = 4 * version + 17
    x = torch.rand(4, 2, n, n)
    out = SmallCNN()(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_smallcnn_exposes_gradcam_layer() -> None:
    m = SmallCNN()
    assert m.last_conv_block is m.block3
    assert m.target_layer is m.block3
    n = 29
    fm = m.feature_maps(torch.rand(2, 2, n, n))
    # 풀링 2회 → 공간 해상도 n//4 수준, 채널 128
    assert fm.shape[0] == 2 and fm.shape[1] == 128
    assert fm.shape[2] == n // 4 and fm.shape[3] == n // 4


def test_bitmlp_full_grid_and_masked_index() -> None:
    n = 29
    x = torch.rand(3, 2, n, n)
    full = BitMLP(n)
    assert full.flatten.out_dim == n * n
    assert full(x).shape == (3,)

    mask = np.zeros((n, n), dtype=bool)
    mask[5:20, 5:20] = True
    masked = BitMLP.from_data_mask(mask)
    assert masked.flatten.out_dim == int(mask.sum())
    assert masked(x).shape == (3,)


def test_bitmlp_ignores_function_pattern_values() -> None:
    """데이터 마스크가 0인 위치의 값 채널은 출력에 영향을 주면 안 된다."""
    n = 21
    m = BitMLP(n).eval()
    x = torch.rand(1, 2, n, n)
    x[:, 1] = 0.0
    x[:, 1, 3:10, 3:10] = 1.0
    y1 = m(x)
    x2 = x.clone()
    x2[:, 0, 15, 15] = 1.0 - x2[:, 0, 15, 15]  # 마스크 밖 위치를 뒤집는다
    assert torch.allclose(y1, m(x2))


def test_linear_probe_shape() -> None:
    n = 25
    assert LinearProbe(n)(torch.rand(2, 2, n, n)).shape == (2,)


def test_build_model_dispatch() -> None:
    n = 25
    mask = np.ones((n, n), dtype=bool)
    assert isinstance(build_model("small_cnn", n, mask), SmallCNN)
    assert isinstance(build_model("bit_mlp", n, mask), BitMLP)
    assert isinstance(build_model("linear_probe", n, mask), LinearProbe)
    with pytest.raises(ValueError):
        build_model("nope", n, mask)
