"""qrgen — URL을 QR 모듈 격자로 인코딩한다 (스펙 4절).

설계 결정 (스펙 1.10, 12절):
- ``optimize=0`` 고정: URL 전체를 단일 8비트 바이트 청크로 인코딩한다.
- ``border=0``: 흰 여백은 상수라 격자만 키운다.
- ``mask_pattern`` 기본 0 고정. ``None``이면 라이브러리 자동 선택이며,
  실제로 사용된 패턴을 :class:`QRArtifact` 에 기록한다.
- ``randpad_seed`` 가 주어지면 ``PAD-rand`` 조건(스펙 1.1-4)으로,
  0xEC/0x11 교대 패딩 대신 시드 기반 의사난수 바이트를 채운다.

라이브러리 monkeypatch는 하지 않는다. ``util.create_data`` 를 복제한
:func:`create_data_randpad` / :func:`create_data_spec` 를 만들고 결과를
``QRCode.data_cache`` 에 주입하는 방식으로 격리한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache

import numpy as np
import qrcode
from qrcode import base, exceptions, util
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

__all__ = [
    "ERROR_CORRECT_L",
    "ERROR_CORRECT_M",
    "ERROR_CORRECT_Q",
    "ERROR_CORRECT_H",
    "QRArtifact",
    "encode",
    "natural_version",
    "fits_in_qr",
    "build_data",
    "create_data_spec",
    "create_data_randpad",
    "url_bytes",
    "data_bit_capacity",
]


def url_bytes(url: str) -> bytes:
    """URL의 바이트 표현. 길이 계산·문자 인덱싱은 모두 UTF-8 바이트 기준이다."""
    return url.encode("utf-8")


@dataclass(frozen=True)
class QRArtifact:
    """QR 인코딩 결과 하나.

    Attributes:
        version: QR 버전 (1~40).
        ec: 오류정정 레벨 상수 (``qrcode.constants.ERROR_CORRECT_*``).
        mask_pattern: 실제로 적용된 마스크 패턴 (0~7). 자동 선택이었어도 실측값.
        n: 격자 한 변 길이 = ``4 * version + 17``.
        modules: ``(n, n)`` bool 배열. border=0, 마스크 적용 상태.
        data_cache: 인터리브된 코드워드 리스트 (데이터 + EC).
        n_data_bits: 데이터 모듈(=기능 패턴이 아닌 모듈) 개수. 잔여 비트 포함.
        n_pad_bytes: 인터리브 전 버퍼에 채워진 패딩 코드워드 개수.
        first_pad_codeword: 인터리브 전 버퍼에서 첫 패딩 코드워드의 인덱스.
            패딩이 없으면 ``-1``. 길이 누출 진단용(스펙 5절).
    """

    version: int
    ec: int
    mask_pattern: int
    n: int
    modules: np.ndarray
    data_cache: list[int]
    n_data_bits: int
    n_pad_bytes: int
    first_pad_codeword: int


def _byte_qrdata(url: str) -> util.QRData:
    """URL을 단일 8비트 바이트 모드 청크로 감싼다 (스펙 1.10 모드 선택)."""
    return util.QRData(url, mode=util.MODE_8BIT_BYTE)


@cache
def data_bit_capacity(version: int, ec: int) -> int:
    """해당 버전·EC의 데이터 코드워드 비트 수 (헤더 + 본문 + 종단자 + 패딩)."""
    return sum(b.data_count * 8 for b in base.rs_blocks(version, ec))


def build_data(
    version: int,
    ec: int,
    url: str,
    randpad_seed: int | None = None,
) -> tuple[list[int], int, int]:
    """``util.create_data`` 복제본.

    Returns:
        ``(data_cache, n_pad_bytes, first_pad_codeword)``.
        ``data_cache`` 는 인터리브된 코드워드, ``first_pad_codeword`` 는
        인터리브 **전** 버퍼에서 첫 패딩 바이트 인덱스(없으면 -1).

    ``randpad_seed`` 가 None이면 규격대로 0xEC/0x11 교대 패딩,
    아니면 ``random.Random(randpad_seed)`` 기반 의사난수 바이트로 채운다.
    패딩 바이트는 길이 필드 밖이라 디코딩 결과는 동일하다(스펙 1.1-4).
    """
    data = _byte_qrdata(url)

    buffer = util.BitBuffer()
    buffer.put(data.mode, 4)
    buffer.put(len(data), util.length_in_bits(data.mode, version))
    data.write(buffer)

    rs_blocks = base.rs_blocks(version, ec)
    bit_limit = sum(block.data_count * 8 for block in rs_blocks)
    if len(buffer) > bit_limit:
        raise exceptions.DataOverflowError(
            f"Code length overflow. Data size ({len(buffer)}) "
            f"> size available ({bit_limit})"
        )

    # 종단자(최대 0000) + 바이트 정렬.
    for _ in range(min(bit_limit - len(buffer), 4)):
        buffer.put_bit(False)
    delimit = len(buffer) % 8
    if delimit:
        for _ in range(8 - delimit):
            buffer.put_bit(False)

    first_pad_codeword = len(buffer) // 8
    bytes_to_fill = (bit_limit - len(buffer)) // 8
    if bytes_to_fill == 0:
        first_pad_codeword = -1

    if randpad_seed is None:
        for i in range(bytes_to_fill):
            buffer.put(util.PAD0 if i % 2 == 0 else util.PAD1, 8)
    else:
        rng = random.Random(randpad_seed)
        for _ in range(bytes_to_fill):
            buffer.put(rng.getrandbits(8), 8)

    return util.create_bytes(buffer, rs_blocks), bytes_to_fill, first_pad_codeword


def create_data_spec(version: int, ec: int, url: str) -> list[int]:
    """규격 패딩 경로. ``util.create_data`` 와 바이트 단위로 동일해야 한다."""
    return build_data(version, ec, url, randpad_seed=None)[0]


def create_data_randpad(version: int, ec: int, url: str, seed: int) -> list[int]:
    """``PAD-rand`` 조건용. 패딩만 시드 기반 의사난수로 교체한다."""
    return build_data(version, ec, url, randpad_seed=seed)[0]


def _required_bits(url: str, version: int) -> int:
    """해당 버전의 문자 카운트 필드 폭 기준으로 필요한 비트 수 (모드 4비트 포함)."""
    return 4 + util.length_in_bits(util.MODE_8BIT_BYTE, version) + 8 * len(url_bytes(url))


def fits_in_qr(url: str, ec: int = ERROR_CORRECT_L) -> bool:
    """URL이 해당 EC의 최대 버전(v40)에 들어가는지."""
    return _required_bits(url, 40) <= util.BIT_LIMIT_TABLE[ec][40]


def natural_version(url: str, ec: int = ERROR_CORRECT_L) -> int | None:
    """격자 생성 없이 best_fit 버전만 구한다 (층 배정용).

    Returns:
        1~40 버전. 해당 EC의 v40 용량(비트)을 넘겨 어느 버전에도 들어가지
        않으면 예외 대신 ``None``. 호출부는 None 행을 걸러내야 한다.
        (예: EC=M의 v40 데이터 용량은 2,331바이트라 그보다 긴 URL은 None.)
    """
    if not fits_in_qr(url, ec):
        return None
    qr = qrcode.QRCode(version=None, error_correction=ec, border=0)
    qr.data_list = [_byte_qrdata(url)]
    qr.data_cache = None
    return int(qr.best_fit())


def encode(
    url: str,
    *,
    ec: int = ERROR_CORRECT_L,
    mask_pattern: int | None = 0,
    version: int | None = None,
    randpad_seed: int | None = None,
) -> QRArtifact:
    """URL 하나를 QR 모듈 격자로 인코딩한다.

    Args:
        url: 인코딩할 URL. UTF-8 바이트로 단일 8비트 바이트 청크 인코딩된다.
        ec: 오류정정 레벨. 기본 L(스펙 1.3).
        mask_pattern: 0~7 고정 또는 ``None``(라이브러리 자동 선택).
            자동일 때도 실제 사용된 패턴이 결과에 기록된다.
        version: 고정 버전. ``None`` 이면 best_fit.
        randpad_seed: 주어지면 ``PAD-rand`` 패딩을 사용한다.

    Raises:
        qrcode.exceptions.DataOverflowError: 지정 버전에 URL이 들어가지 않을 때.
        ValueError: ``version=None`` 인데 URL이 해당 EC의 v40 용량도 넘을 때.
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=ec,
        border=0,
        mask_pattern=mask_pattern,
    )
    # add_data(url, optimize=0) 과 동등하되, 모드를 8비트 바이트로 못박는다.
    # optimize=0 이어도 QRData 는 optimal_mode 로 numeric/alnum 을 고를 수 있는데
    # 그러면 스펙 1.10 의 "단일 8비트 바이트 청크" 전제가 깨진다.
    qr.add_data(url, optimize=0)
    qr.data_list = [_byte_qrdata(url)]
    qr.data_cache = None

    if version is None:
        if not fits_in_qr(url, ec):
            raise ValueError(
                f"URL이 EC={ec}의 최대 버전(v40) 용량을 넘는다 "
                f"({len(url_bytes(url))} bytes). natural_version()은 이 경우 None을 반환한다."
            )
        qr.best_fit()
    ver = int(qr.version)

    data_cache, n_pad, first_pad = build_data(ver, ec, url, randpad_seed=randpad_seed)
    qr.data_cache = data_cache
    qr.make(fit=False)

    used_mask = int(qr.mask_pattern) if qr.mask_pattern is not None else _detect_mask(qr)

    n = qr.modules_count
    modules = np.array(qr.modules, dtype=bool)
    from qrphish.mapping import function_mask  # 지연 import (순환 방지)

    n_data_bits = int((~function_mask(ver, ec)).sum())

    return QRArtifact(
        version=ver,
        ec=ec,
        mask_pattern=used_mask,
        n=n,
        modules=modules,
        data_cache=list(data_cache),
        n_data_bits=n_data_bits,
        n_pad_bytes=n_pad,
        first_pad_codeword=first_pad,
    )


def _detect_mask(qr: qrcode.QRCode) -> int:
    """자동 마스크 선택 시 실제 사용된 패턴을 재현한다.

    ``QRCode.make`` 는 ``best_mask_pattern()`` 결과를 저장하지 않으므로,
    동일한 data_cache 로 ``best_mask_pattern`` 을 한 번 더 계산한다.
    ``lost_point`` 는 결정적이라 같은 값이 나온다.
    """
    modules_backup = qr.modules
    pattern = int(qr.best_mask_pattern())
    qr.makeImpl(False, pattern)
    assert np.array_equal(np.array(qr.modules), np.array(modules_backup))
    return pattern
