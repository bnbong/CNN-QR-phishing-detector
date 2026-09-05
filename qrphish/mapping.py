"""mapping — 모듈 격자 ↔ 비트 ↔ 코드워드 ↔ URL 문자 역매핑 (스펙 2장).

자체 구현 범위는 스펙 2.2 대로 세 가지뿐이다.

1. :func:`bit_placement_order` — ``QRCode.map_data`` 의 지그재그 순회를
   값 대신 좌표만 기록하도록 복제(약 25줄).
2. :func:`interleave_inverse` — 인터리브 코드워드 인덱스 → 프리인터리브 위치.
3. ``create_data_randpad`` — :mod:`qrphish.qrgen` 에 있다.

마스크 함수, RS 인코딩, ``rs_blocks`` 표, 기능 패턴 배치는 전부 라이브러리 것을
그대로 쓴다. 기능 패턴 마스크는 :class:`_CaptureQRCode` 로 ``map_data`` 진입
시점의 non-None 집합을 캡처해 얻는다(순회 복제와 마스크 획득의 분리).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
import qrcode
from qrcode import base, util

from qrphish.qrgen import QRArtifact, url_bytes

__all__ = [
    "function_mask",
    "function_values",
    "bit_placement_order",
    "interleave_inverse",
    "ModuleProvenance",
    "KIND_FUNCTION",
    "KIND_CHAR",
    "KIND_LENGTH_HEADER",
    "KIND_MODE_HEADER",
    "KIND_PAD",
    "KIND_EC",
    "KIND_REMAINDER",
    "KIND_NAMES",
    "provenance",
    "unmask",
    "verify_roundtrip",
    "reconstruct_modules",
    "fixed_permutation",
    "to_tensor_channels",
]

KIND_FUNCTION = 0
KIND_CHAR = 1
KIND_LENGTH_HEADER = 2
KIND_MODE_HEADER = 3
KIND_PAD = 4
KIND_EC = 5
KIND_REMAINDER = 6

KIND_NAMES = {
    KIND_FUNCTION: "function",
    KIND_CHAR: "char",
    KIND_LENGTH_HEADER: "length_header",
    KIND_MODE_HEADER: "mode_header",
    KIND_PAD: "pad",
    KIND_EC: "ec",
    KIND_REMAINDER: "remainder",
}


# --------------------------------------------------------------------------
# 기능 패턴 마스크 / 값
# --------------------------------------------------------------------------


class _CaptureQRCode(qrcode.QRCode):
    """``map_data`` 진입 시점의 격자 상태를 가로채는 서브클래스."""

    captured: list[list[bool | None]]

    def map_data(self, data, mask_pattern):  # type: ignore[override]
        self.captured = [row[:] for row in self.modules]
        super().map_data(data, mask_pattern)


@cache
def _blank(version: int, ec: int, mask_pattern: int) -> tuple[np.ndarray, np.ndarray]:
    """``(function_mask, function_values)`` — 데이터와 무관한 격자 골격.

    파인더/타이밍/정렬 패턴 + 포맷 정보 + (v≥7) 버전 정보 + 다크 모듈을 포함한다.
    포맷 정보 **값** 은 ec·mask 에 의존하지만 **위치** 는 의존하지 않는다.
    """
    qr = _CaptureQRCode(
        version=version, error_correction=ec, border=0, mask_pattern=mask_pattern
    )
    qr.data_list = [util.QRData(b"\x00", mode=util.MODE_8BIT_BYTE, check_data=False)]
    qr.data_cache = None
    qr.make(fit=False)

    cap = qr.captured
    fmask = np.array([[v is not None for v in row] for row in cap], dtype=bool)
    fvals = np.array([[bool(v) for v in row] for row in cap], dtype=bool)
    fvals &= fmask
    return fmask, fvals


@cache
def function_mask(version: int, ec: int) -> np.ndarray:
    """``(n, n)`` bool. True = 기능 패턴(데이터가 들어가지 않는 모듈)."""
    fmask, _ = _blank(version, ec, 0)
    out = fmask.copy()
    out.setflags(write=False)
    return out


@cache
def function_values(version: int, ec: int, mask_pattern: int) -> np.ndarray:
    """``(n, n)`` bool. 기능 패턴 모듈의 값(비기능 위치는 False)."""
    _, fvals = _blank(version, ec, mask_pattern)
    out = fvals.copy()
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------
# 비트 배치 순서 (map_data 순회 복제)
# --------------------------------------------------------------------------


@cache
def bit_placement_order(version: int, ec: int) -> np.ndarray:
    """``(n_data_bits, 2) int32``. i번째 데이터 비트가 놓이는 ``(row, col)``.

    ``QRCode.map_data`` 의 순회를 값 대신 좌표만 기록하도록 복제한 것이다.
    전역 비트 인덱스 i ↔ 코드워드 ``i // 8``, 비트 ``7 - (i % 8)`` (MSB 우선).
    """
    fmask = function_mask(version, ec)
    count = fmask.shape[0]

    order: list[tuple[int, int]] = []
    inc = -1
    row = count - 1

    for col0 in range(count - 1, 0, -2):
        col = col0 - 1 if col0 <= 6 else col0
        col_range = (col, col - 1)
        while True:
            for c in col_range:
                if not fmask[row][c]:
                    order.append((row, c))
            row += inc
            if row < 0 or count <= row:
                row -= inc
                inc = -inc
                break

    arr = np.array(order, dtype=np.int32)
    arr.setflags(write=False)
    return arr


# --------------------------------------------------------------------------
# 인터리빙 역산
# --------------------------------------------------------------------------


@cache
def interleave_inverse(version: int, ec: int) -> list[tuple[str, int | None, int]]:
    """인터리브 코드워드 인덱스 → 출처.

    Returns:
        인덱스 j 에 대해 ``('data', buf_byte_index, block)`` 또는
        ``('ec', None, block)``. ``buf_byte_index`` 는 ``util.create_bytes``
        입력 버퍼(프리인터리브 BitBuffer)의 바이트 오프셋이다.

    ``create_bytes`` 는 블록별 데이터 코드워드를 라운드로빈으로 쌓고, 이어서
    EC 코드워드를 라운드로빈으로 쌓는다. 단일 블록(EC=L 기준 v1~v5)에서는 항등에
    가깝지만 v6/L 부터 블록이 둘로 갈리므로 일반형으로 구현한다.
    """
    blocks = base.rs_blocks(version, ec)
    dc = [b.data_count for b in blocks]
    ecc = [b.total_count - b.data_count for b in blocks]

    # 프리인터리브 버퍼에서 각 블록 데이터의 시작 바이트 오프셋.
    starts: list[int] = []
    off = 0
    for c in dc:
        starts.append(off)
        off += c

    out: list[tuple[str, int | None, int]] = []
    for i in range(max(dc)):
        for bi, c in enumerate(dc):
            if i < c:
                out.append(("data", starts[bi] + i, bi))
    for i in range(max(ecc)):
        for bi, c in enumerate(ecc):
            if i < c:
                out.append(("ec", None, bi))
    return out


@cache
def _buffer_block_of_byte(version: int, ec: int) -> np.ndarray:
    """프리인터리브 버퍼 바이트 인덱스 → 소속 RS 블록 인덱스."""
    blocks = base.rs_blocks(version, ec)
    out: list[int] = []
    for bi, b in enumerate(blocks):
        out.extend([bi] * b.data_count)
    arr = np.array(out, dtype=np.int32)
    arr.setflags(write=False)
    return arr


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleProvenance:
    """모듈별 출처 태깅 결과.

    Attributes:
        kind: ``(n, n) uint8``. 0=function 1=char 2=length_header 3=mode_header
            4=pad 5=ec 6=remainder.
            **종단자 0000 비트와 바이트 정렬용 0비트는 ``pad`` (4)로 분류한다.**
            둘 다 URL 본문 밖이고 길이의 결정적 함수라 패딩과 성질이 같다.
            ``remainder`` (6)는 데이터 코드워드를 다 쓰고 남는 잔여 비트로,
            항상 0이 배치되므로 클래스 정보가 없다(스펙 1.10).
        char_index: ``(n, n) int32``. kind==1 일 때 URL의 UTF-8 바이트 인덱스,
            아니면 -1.
        block_index: ``(n, n) int32``. kind==5 일 때 RS 블록 인덱스, 아니면 -1.
    """

    kind: np.ndarray
    char_index: np.ndarray
    block_index: np.ndarray


def provenance(url: str, art: QRArtifact) -> ModuleProvenance:
    """URL 문자 ↔ 모듈 좌표 역매핑을 만든다.

    헤더는 모드 4비트 + 길이 필드(v1~9 byte 모드 8비트, v10 이상 16비트)이므로
    v1~9 는 12비트, v10 이상은 20비트다. URL의 k번째 바이트는 프리인터리브
    버퍼의 비트 ``header + 8k … header + 8k + 7`` 을 차지하며 코드워드 경계에
    걸치므로 매핑은 반드시 비트 단위로 한다.
    """
    version, ec = art.version, art.ec
    n = art.n
    order = bit_placement_order(version, ec)
    inv = interleave_inverse(version, ec)

    kind = np.zeros((n, n), dtype=np.uint8)
    char_index = np.full((n, n), -1, dtype=np.int32)
    block_index = np.full((n, n), -1, dtype=np.int32)

    header_bits = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, version)
    n_url_bytes = len(url_bytes(url))
    n_codewords = len(art.data_cache)

    for i in range(order.shape[0]):
        r, c = int(order[i, 0]), int(order[i, 1])
        cw = i // 8
        if cw >= n_codewords:
            kind[r, c] = KIND_REMAINDER
            continue
        src, buf_byte, blk = inv[cw]
        if src == "ec":
            kind[r, c] = KIND_EC
            block_index[r, c] = blk
            continue
        assert buf_byte is not None  # src == "data"면 항상 버퍼 오프셋이 있다
        gbit = 8 * int(buf_byte) + (i % 8)
        if gbit < 4:
            kind[r, c] = KIND_MODE_HEADER
        elif gbit < header_bits:
            kind[r, c] = KIND_LENGTH_HEADER
        else:
            k = (gbit - header_bits) // 8
            if k < n_url_bytes:
                kind[r, c] = KIND_CHAR
                char_index[r, c] = k
            else:
                kind[r, c] = KIND_PAD

    return ModuleProvenance(kind=kind, char_index=char_index, block_index=block_index)


def char_block(url: str, art: QRArtifact, k: int) -> set[int]:
    """URL의 k번째 바이트가 걸쳐 있는 RS 블록 인덱스 집합.

    한 바이트는 코드워드 경계를 넘을 수 있고, 블록 경계와도 겹칠 수 있어
    집합으로 돌려준다. 인과 검증(스펙 9장 4번)에서 EC 오염 범위 계산에 쓴다.
    """
    header_bits = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, art.version)
    b2blk = _buffer_block_of_byte(art.version, art.ec)
    lo = header_bits + 8 * k
    return {int(b2blk[b]) for b in (lo // 8, (lo + 7) // 8) if b < len(b2blk)}


# --------------------------------------------------------------------------
# 마스크 / 재구성 / 검증
# --------------------------------------------------------------------------


@cache
def _mask_grid(n: int, mask_pattern: int) -> np.ndarray:
    f = util.mask_func(mask_pattern)
    g = np.array([[bool(f(r, c)) for c in range(n)] for r in range(n)], dtype=bool)
    g.setflags(write=False)
    return g


def unmask(modules: np.ndarray, mask_pattern: int) -> np.ndarray:
    """마스크를 XOR로 되돌린다 (``mask-off`` 조건, 스펙 1.6).

    기능 패턴 모듈은 애초에 마스크가 걸리지 않으므로 건드리지 않는다.
    버전은 ``n = 4v + 17`` 로부터 복원하고, 기능 패턴 **위치** 는 EC 레벨과
    무관하므로 L 기준 마스크를 쓴다. 자기 자신에 대해 involution 이다.
    """
    n = int(modules.shape[0])
    version = (n - 17) // 4
    fm = function_mask(version, qrcode.constants.ERROR_CORRECT_L)
    out = np.asarray(modules, dtype=bool).copy()
    flip = _mask_grid(n, mask_pattern) & ~fm
    out[flip] = ~out[flip]
    return out


def reconstruct_modules(art: QRArtifact) -> np.ndarray:
    """``data_cache`` + 비트 순서 + 마스크 + 기능 패턴으로 격자를 재구성한다."""
    version, ec, mp = art.version, art.ec, art.mask_pattern
    grid = np.array(function_values(version, ec, mp), dtype=bool)

    order = bit_placement_order(version, ec)
    mgrid = _mask_grid(art.n, mp)
    data = art.data_cache
    n_bits = 8 * len(data)

    for i in range(order.shape[0]):
        r, c = int(order[i, 0]), int(order[i, 1])
        bit = ((data[i // 8] >> (7 - (i % 8))) & 1) == 1 if i < n_bits else False
        grid[r, c] = (not bit) if mgrid[r, c] else bit
    return grid


def verify_roundtrip(url: str, art: QRArtifact) -> bool:
    """스펙 2.3 검증 게이트. 재구성 격자가 ``art.modules`` 와 전 모듈 일치하는가."""
    return bool(np.array_equal(reconstruct_modules(art), art.modules))


# --------------------------------------------------------------------------
# 텐서화 유틸
# --------------------------------------------------------------------------


@cache
def fixed_permutation(n: int, seed: int) -> np.ndarray:
    """길이 ``n`` 의 고정 순열 (``shuffle-pos`` 조건, 스펙 1.7).

    ``n`` 은 **순열의 길이**, 즉 셔플 대상 위치의 개수다. :class:`~qrphish.dataset.QRGridDataset`
    는 평탄화한 격자 전체를 셔플하므로 ``n = grid_n * grid_n`` 을 넣는다(값 채널과
    데이터 마스크 채널에 같은 순열을 적용하므로, 기능 패턴 위치를 따로 뺄 필요가 없다).
    층 내 **전체 샘플에 동일한** 순열이 쓰인다. 시드가 같으면 플랫폼과 무관하게 같은
    순열이 나온다(``np.random.default_rng``).
    """
    perm = np.random.default_rng(seed).permutation(n).astype(np.int32)
    perm.setflags(write=False)
    return perm


def to_tensor_channels(
    art: QRArtifact,
    features: Literal["all", "data_only"] = "data_only",
    mask_mode: Literal["fixed", "auto", "off"] = "fixed",
) -> np.ndarray:
    """``(2, n, n) float32`` 입력 텐서 (스펙 1.7).

    - 채널 0: 모듈 값. ``features="data_only"`` 면 기능 패턴 위치를 0으로.
    - 채널 1: 데이터 모듈 마스크 (1=데이터, 0=기능 패턴). 층 내 상수 채널이지만
      채널 0의 0이 "값 0"인지 "기능 패턴"인지 구분시키려고 명시적으로 둔다.

    ``mask_mode="off"`` 면 :func:`unmask` 로 마스크를 되돌린다.
    ``"fixed"`` / ``"auto"`` 는 ``art.modules`` 를 그대로 쓴다(이미 마스크 적용됨).

    .. note::
       **단건 시각화·테스트 전용이다.** 학습 파이프라인은 이 함수를 쓰지 않는다.
       ``mask_mode`` 의 정본 적용 시점은 층 빌드
       (:func:`qrphish.dataset.build_stratum`)이며, 거기서 unmask 된 격자가
       ``grids.npz`` 에 저장된다. :class:`~qrphish.dataset.QRGridDataset` 은
       저장된 격자를 그대로 읽으므로 마스크를 두 번 되돌리는 일이 없다.
       두 경로가 같은 결과를 내는지는 ``tests/test_mapping.py`` 의
       ``test_to_tensor_channels_matches_build_stratum`` 이 지킨다.
    """
    if features not in ("all", "data_only"):
        raise ValueError(f"Unknown features: {features!r}")
    if mask_mode not in ("fixed", "auto", "off"):
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")

    mods = unmask(art.modules, art.mask_pattern) if mask_mode == "off" else art.modules
    ch0 = np.asarray(mods, dtype=np.float32).copy()
    data_mask = ~function_mask(art.version, art.ec)
    if features == "data_only":
        ch0[~data_mask] = 0.0
    return np.stack([ch0, data_mask.astype(np.float32)], axis=0)
