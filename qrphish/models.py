"""모델 정의: SmallCNN, BitMLP, LinearProbe.

입력 텐서 규약(스펙 1.7): 항상 ``(2, n, n)`` float32.
  - 채널 0: 모듈 값(0/1). ``feat-data`` 조건에서는 기능 패턴 위치가 0.
  - 채널 1: 데이터 모듈 마스크(1=데이터, 0=기능 패턴).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SmallCNN", "BitMLP", "LinearProbe", "build_model"]


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Conv(32,3)x2 -> Pool -> Conv(64,3)x2 -> Pool -> Conv(128,3) -> GAP -> Dropout -> Linear(1).

    n이 21~57로 작으므로 풀링은 2회까지만 한다. ``last_conv_block``/``target_layer``는
    Grad-CAM 훅 부착 지점으로 공개한다.
    """

    def __init__(self, in_ch: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = nn.Sequential(_conv_bn_relu(in_ch, 32), _conv_bn_relu(32, 32))
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(_conv_bn_relu(32, 64), _conv_bn_relu(64, 64))
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = _conv_bn_relu(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)

    @property
    def last_conv_block(self) -> nn.Module:
        """Grad-CAM 대상 레이어(마지막 conv 블록)."""
        return self.block3

    @property
    def target_layer(self) -> nn.Module:
        return self.block3

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool1(self.block1(x))
        h = self.pool2(self.block2(h))
        return self.block3(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_maps(x)
        h = self.gap(h).flatten(1)
        return self.head(self.dropout(h)).squeeze(-1)


class _FlattenBits(nn.Module):
    """(B,2,n,n) -> (B, D). 데이터 모듈 비트만 배치 순서대로 편다.

    ``data_index``가 등록되어 있으면 그 위치만 뽑고, 없으면 값 채널에 데이터 마스크를
    곱해(기능 패턴 = 0) 전체를 편다. 두 경우 모두 D는 고정이라 층 내에서 형상이 같다.
    """

    def __init__(self, n: int, data_index: torch.Tensor | None = None) -> None:
        super().__init__()
        self.n = n
        if data_index is None:
            self.register_buffer("data_index", None)
            self.out_dim = n * n
        else:
            idx = torch.as_tensor(data_index, dtype=torch.long).reshape(-1)
            self.register_buffer("data_index", idx)
            self.out_dim = int(idx.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[:, 0] * x[:, 1]
        v = v.reshape(v.shape[0], -1)
        idx = self.data_index
        if idx is not None:
            v = v.index_select(1, torch.as_tensor(idx))
        return v


class BitMLP(nn.Module):
    """flatten(데이터 모듈 비트) -> 512 -> 256 -> 1, Dropout 0.3."""

    def __init__(self, n: int, data_index: torch.Tensor | None = None, dropout: float = 0.3) -> None:
        super().__init__()
        self.flatten = _FlattenBits(n, data_index)
        d = self.flatten.out_dim
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @classmethod
    def from_data_mask(cls, data_mask, dropout: float = 0.3) -> BitMLP:
        """(n,n) bool 데이터 마스크에서 인덱스를 뽑아 생성한다."""
        m = torch.as_tensor(data_mask).bool().reshape(-1)
        n = int(round(float(m.numel()) ** 0.5))
        return cls(n, data_index=torch.nonzero(m, as_tuple=False).squeeze(-1), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x)).squeeze(-1)


class LinearProbe(nn.Module):
    """flatten(데이터 모듈 비트) -> Linear(1). 선형 분리 가능성 하한 측정용."""

    def __init__(self, n: int, data_index: torch.Tensor | None = None) -> None:
        super().__init__()
        self.flatten = _FlattenBits(n, data_index)
        self.head = nn.Linear(self.flatten.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(x)).squeeze(-1)


def build_model(arch: str, n: int, data_mask=None) -> nn.Module:
    """config의 ``model.arch`` 문자열로 모델을 만든다."""
    if arch == "small_cnn":
        return SmallCNN(in_ch=2)
    idx = None
    if data_mask is not None:
        m = torch.as_tensor(data_mask).bool().reshape(-1)
        idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
    if arch == "bit_mlp":
        return BitMLP(n, data_index=idx)
    if arch == "linear_probe":
        return LinearProbe(n, data_index=idx)
    raise ValueError(f"unknown arch: {arch}")
