"""qrgen 테스트 — 인코딩 규칙, 패딩 메타, PAD-rand (스펙 4절, 1.1)."""

from __future__ import annotations

import pytest
import qrcode
from qrcode import base, util

from qrphish.qrgen import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    build_data,
    create_data_randpad,
    create_data_spec,
    encode,
    natural_version,
    url_bytes,
)

# 실제 데이터(backups/data/URL.xlsx)는 읽지 않는다. 스펙에 등장한 예시와
# 형태를 모방한 합성 URL 만 사용한다.
EXAMPLE_URLS = [
    "www.google.com",
    "https://www.google.com/search?q=qr+code",
    "http://000webhostapp.com/secure/login.php?id=99",
    "https://sub.weebly.com/paypal-verify/account/update",
    "http://192.168.0.1/admin",
    "https://blogspot.com/a",
    "http://duckdns.org/wp-content/plugins/x/verify.html?session=abc123def456",
]

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_./?=&%~+"


def synth_urls(count: int, seed: int = 0, min_len: int = 8, max_len: int = 90):
    """길이·문자 구성이 다양한 합성 URL 목록. 시드로 완전 재현된다."""
    import random as _random

    rng = _random.Random(seed)
    schemes = ["http://", "https://", ""]
    out = []
    for _ in range(count):
        body_len = rng.randint(min_len, max_len)
        body = "".join(rng.choice(_ALPHABET) for _ in range(body_len))
        out.append(rng.choice(schemes) + body)
    return out



def _lib_create_data(version: int, ec: int, url: str) -> list[int]:
    """라이브러리 원본 경로(단일 8비트 바이트 청크)."""
    return util.create_data(
        version, ec, [util.QRData(url, mode=util.MODE_8BIT_BYTE)]
    )


class TestEncode:
    def test_grid_shape_and_border(self):
        for url in EXAMPLE_URLS:
            art = encode(url)
            assert art.n == 4 * art.version + 17
            assert art.modules.shape == (art.n, art.n)
            assert art.modules.dtype == bool

    def test_matches_library_grid(self):
        """encode 결과가 라이브러리 QRCode 격자와 완전히 같아야 한다."""
        for url in synth_urls(40, seed=7):
            for mp in (0, 5):
                art = encode(url, mask_pattern=mp)
                qr = qrcode.QRCode(
                    version=art.version,
                    error_correction=ERROR_CORRECT_L,
                    border=0,
                    mask_pattern=mp,
                )
                qr.add_data(util.QRData(url, mode=util.MODE_8BIT_BYTE))
                qr.make(fit=False)
                assert art.modules.tolist() == qr.modules

    def test_spec_padding_equals_library(self):
        for url in synth_urls(60, seed=11):
            v = natural_version(url, ERROR_CORRECT_L)
            assert create_data_spec(v, ERROR_CORRECT_L, url) == _lib_create_data(
                v, ERROR_CORRECT_L, url
            )

    def test_natural_version_matches_encode(self):
        for url in synth_urls(60, seed=13):
            assert natural_version(url, ERROR_CORRECT_L) == encode(url).version
            assert natural_version(url, ERROR_CORRECT_M) == encode(
                url, ec=ERROR_CORRECT_M
            ).version

    def test_explicit_version_overrides(self):
        art = encode("https://a.example.com/", version=6)
        assert art.version == 6 and art.n == 41

    def test_overflow_raises(self):
        with pytest.raises(qrcode.exceptions.DataOverflowError):
            encode("https://example.com/" + "a" * 100, version=1)

    def test_auto_mask_records_actual_pattern(self):
        """mask_pattern=None 이면 실제 사용된 패턴이 기록되어야 한다."""
        seen = set()
        for url in synth_urls(40, seed=17):
            art = encode(url, mask_pattern=None)
            assert 0 <= art.mask_pattern <= 7
            seen.add(art.mask_pattern)
            fixed = encode(url, mask_pattern=art.mask_pattern, version=art.version)
            assert fixed.modules.tolist() == art.modules.tolist()
        assert len(seen) > 1, "자동 선택이 한 패턴으로만 몰리면 검증 의미가 없다"

    def test_utf8_byte_length_basis(self):
        """길이 계산은 문자 수가 아니라 UTF-8 바이트 수 기준이다."""
        url = "https://예시.example.com/한글"
        art = encode(url)
        nb = len(url_bytes(url))
        assert nb > len(url)
        v = art.version
        header = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, v)
        cap = sum(b.data_count * 8 for b in base.rs_blocks(v, ERROR_CORRECT_L))
        assert header + 8 * nb <= cap


class TestPaddingMeta:
    def test_pad_fields_consistent(self):
        for url in synth_urls(80, seed=19):
            art = encode(url)
            v = art.version
            header = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, v)
            cap = sum(b.data_count * 8 for b in base.rs_blocks(v, ERROR_CORRECT_L))
            nb = len(url_bytes(url))
            used_bits = header + 8 * nb
            # 종단자(최대 4비트) + 바이트 정렬 후의 바이트 인덱스.
            term = min(cap - used_bits, 4)
            aligned = used_bits + term
            aligned += (-aligned) % 8
            expected_pads = (cap - aligned) // 8
            assert art.n_pad_bytes == expected_pads
            if expected_pads == 0:
                assert art.first_pad_codeword == -1
            else:
                assert art.first_pad_codeword == aligned // 8

    def test_first_pad_codeword_is_monotone_in_length(self):
        """고정 버전 안에서 패딩 시작 인덱스는 길이의 단조 함수다 (스펙 1.1 누출)."""
        prev = -1
        for length in range(10, 50):
            url = "https://e.com/" + "a" * (length)
            art = encode(url, version=5)
            assert art.first_pad_codeword >= prev
            prev = art.first_pad_codeword

    def test_no_padding_case(self):
        """데이터가 용량을 꽉 채우면 패딩이 없고 -1 이 기록된다."""
        v = 1
        cap = sum(b.data_count * 8 for b in base.rs_blocks(v, ERROR_CORRECT_L))
        nb = (cap - 12) // 8  # 헤더 12비트를 뺀 최대 바이트 수
        art = encode("a" * nb, version=v)
        assert art.n_pad_bytes == 0
        assert art.first_pad_codeword == -1


class TestRandPad:
    def test_randpad_deterministic_and_differs(self):
        url = "https://phish.example.org/login/verify?id=42"
        v = natural_version(url, ERROR_CORRECT_L)
        spec = create_data_spec(v, ERROR_CORRECT_L, url)
        r1 = create_data_randpad(v, ERROR_CORRECT_L, url, seed=123)
        r2 = create_data_randpad(v, ERROR_CORRECT_L, url, seed=123)
        r3 = create_data_randpad(v, ERROR_CORRECT_L, url, seed=124)
        assert r1 == r2
        assert r1 != spec
        assert r1 != r3
        assert len(r1) == len(spec)

    def test_randpad_preserves_header_and_payload(self):
        """패딩만 바뀌므로 프리인터리브 버퍼의 헤더+본문 구간은 동일해야 한다."""
        url = "https://phish.example.org/login/verify?id=42"
        v = natural_version(url, ERROR_CORRECT_L)
        _, n_pad, first_pad = build_data(v, ERROR_CORRECT_L, url)
        assert n_pad > 0 and first_pad > 0
        # 단일 블록 버전에서는 인터리브가 항등이므로 앞 first_pad 코드워드가 그대로 비교된다.
        assert len(base.rs_blocks(v, ERROR_CORRECT_L)) == 1
        spec = create_data_spec(v, ERROR_CORRECT_L, url)
        rand = create_data_randpad(v, ERROR_CORRECT_L, url, seed=5)
        assert spec[:first_pad] == rand[:first_pad]
        assert spec[first_pad : first_pad + n_pad] != rand[first_pad : first_pad + n_pad]

    def test_randpad_encode_roundtrips(self):
        from qrphish.mapping import verify_roundtrip

        url = "https://phish.example.org/login/verify?id=42"
        art = encode(url, randpad_seed=99)
        assert art.n_pad_bytes > 0
        assert verify_roundtrip(url, art)
