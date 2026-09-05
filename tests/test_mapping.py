"""mapping 테스트 — 스펙 9장 정확성 게이트 1~5 + 형상 12.

이 파일의 라운드트립 테스트(스펙 2.3)가 통과해야 역매핑 결과를 논문에 쓸 수 있다.
"""

from __future__ import annotations

import numpy as np
import pytest
import qrcode
from qrcode import base, util

from qrphish.mapping import (
    KIND_CHAR,
    KIND_EC,
    KIND_FUNCTION,
    KIND_LENGTH_HEADER,
    KIND_MODE_HEADER,
    KIND_PAD,
    KIND_REMAINDER,
    bit_placement_order,
    char_block,
    fixed_permutation,
    function_mask,
    interleave_inverse,
    provenance,
    reconstruct_modules,
    to_tensor_channels,
    unmask,
    verify_roundtrip,
)
from qrphish.qrgen import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    encode,
    url_bytes,
)

from .test_qrgen import EXAMPLE_URLS, synth_urls

EC_L = ERROR_CORRECT_L


# --------------------------------------------------------------------------
# 1. 라운드트립 (스펙 9장 1, 2.3절 검증 게이트)
# --------------------------------------------------------------------------


class TestRoundtrip:
    def test_roundtrip_exact(self):
        """무작위 URL × v1~v6 × 마스크 0~7 전수 대조. 하나라도 틀리면 파이프라인 중단."""
        n_urls = 500  # 스펙 9장 1번 그대로
        checked = 0
        for version in range(1, 7):
            cap = sum(b.data_count for b in base.rs_blocks(version, EC_L)) - 2
            urls = [
                u
                for u in synth_urls(n_urls * 2, seed=1000 + version, min_len=1, max_len=cap)
                if len(url_bytes(u)) <= cap
            ][:n_urls]
            assert len(urls) == n_urls, f"v{version} 표본 부족: {len(urls)}"
            for url in urls:
                for mask in range(8):
                    art = encode(url, version=version, mask_pattern=mask, ec=EC_L)
                    assert verify_roundtrip(url, art), (version, mask, url)
                    checked += 1
        assert checked == 500 * 6 * 8

    def test_roundtrip_example_urls_auto_version(self):
        for url in EXAMPLE_URLS:
            for mask in list(range(8)) + [None]:
                art = encode(url, mask_pattern=mask)
                assert verify_roundtrip(url, art)

    def test_roundtrip_ec_levels(self):
        for ec in (ERROR_CORRECT_L, ERROR_CORRECT_M):
            for url in synth_urls(30, seed=3, max_len=40):
                art = encode(url, ec=ec, mask_pattern=0)
                assert verify_roundtrip(url, art)

    def test_roundtrip_randpad(self):
        for url in synth_urls(30, seed=4, max_len=40):
            art = encode(url, randpad_seed=hash(url) % 10_000, mask_pattern=2)
            assert verify_roundtrip(url, art)

    def test_reconstruct_differs_when_data_perturbed(self):
        """재구성이 실제로 data_cache 를 쓰는지 (자명 통과 방지)."""
        art = encode(EXAMPLE_URLS[1], version=4, mask_pattern=0)
        broken = art.data_cache[:]
        broken[0] ^= 0xFF
        object.__setattr__(art, "data_cache", broken)
        assert not np.array_equal(reconstruct_modules(art), art.modules)


# --------------------------------------------------------------------------
# 2. 비트 개수 (스펙 9장 2)
# --------------------------------------------------------------------------


class TestBitCount:
    def test_bit_count_matches_free_modules(self):
        for version in range(1, 11):
            for ec in (ERROR_CORRECT_L, ERROR_CORRECT_M):
                order = bit_placement_order(version, ec)
                fm = function_mask(version, ec)
                assert order.shape[0] == int((~fm).sum())
                # 좌표가 중복 없이 모든 자유 모듈을 정확히 한 번씩 덮는가.
                coords = {(int(r), int(c)) for r, c in order}
                assert len(coords) == order.shape[0]
                free = {(int(r), int(c)) for r, c in zip(*np.nonzero(~fm), strict=True)}
                assert coords == free

    def test_bit_count_equals_codewords_plus_remainder(self):
        for version in range(1, 11):
            n_bits = bit_placement_order(version, EC_L).shape[0]
            total_cw = sum(b.total_count for b in base.rs_blocks(version, EC_L))
            remainder = n_bits - 8 * total_cw
            assert 0 <= remainder < 8
            assert n_bits == 8 * total_cw + remainder

    def test_artifact_n_data_bits(self):
        for url in synth_urls(20, seed=5, max_len=60):
            art = encode(url)
            assert art.n_data_bits == bit_placement_order(art.version, art.ec).shape[0]
            assert art.n_data_bits >= 8 * len(art.data_cache)

    def test_grid_size_formula(self):
        for version in range(1, 11):
            assert function_mask(version, EC_L).shape == (
                4 * version + 17,
                4 * version + 17,
            )


# --------------------------------------------------------------------------
# 3. 인터리브 역산 (스펙 9장 3)
# --------------------------------------------------------------------------


def _reference_interleave(version: int, ec: int) -> list[tuple[str, int | None, int]]:
    """``util.create_bytes`` 를 마커 바이트로 실행해 얻은 정답 인덱스 대조표.

    데이터 코드워드에는 프리인터리브 버퍼 오프셋을 값으로 넣어(0..255 범위 안에서)
    인터리브 후 위치를 직접 읽는다. EC 는 블록 개수만 검증한다.
    """
    blocks = base.rs_blocks(version, ec)
    total_dc = sum(b.data_count for b in blocks)
    assert total_dc <= 256
    buf = util.BitBuffer()
    for i in range(total_dc):
        buf.put(i, 8)
    out = util.create_bytes(buf, blocks)

    # 블록 소속표.
    owner: list[int] = []
    for bi, b in enumerate(blocks):
        owner.extend([bi] * b.data_count)

    ref: list[tuple[str, int | None, int]] = []
    for j in range(total_dc):
        v = out[j]
        ref.append(("data", v, owner[v]))
    n_ec = sum(b.total_count - b.data_count for b in blocks)
    assert len(out) == total_dc + n_ec
    return ref


class TestInterleaveInverse:
    def test_single_block_is_identity(self):
        """v4/L 은 단일 블록이라 데이터 구간이 항등이어야 한다."""
        blocks = base.rs_blocks(4, EC_L)
        assert len(blocks) == 1
        inv = interleave_inverse(4, EC_L)
        dc = blocks[0].data_count
        for j in range(dc):
            assert inv[j] == ("data", j, 0)
        for j in range(dc, len(inv)):
            assert inv[j] == ("ec", None, 0)

    def test_multi_block_matches_create_bytes(self):
        """v5/L 부터 블록이 갈린다. create_bytes 실측과 인덱스 대조."""
        multi = [
            (v, ec)
            for v in range(1, 13)
            for ec in (ERROR_CORRECT_L, ERROR_CORRECT_M)
            if len(base.rs_blocks(v, ec)) > 1
            # 마커 바이트가 0..255 라 데이터 코드워드가 256개 이하인 버전만 대조 가능.
            and sum(b.data_count for b in base.rs_blocks(v, ec)) <= 256
        ]
        assert any(v >= 5 and ec == EC_L for v, ec in multi), "다중 블록 케이스가 없다"
        for version, ec in multi:
            inv = interleave_inverse(version, ec)
            ref = _reference_interleave(version, ec)
            assert inv[: len(ref)] == ref, (version, ec)

    def test_ec_section_block_counts(self):
        for version in range(1, 13):
            blocks = base.rs_blocks(version, EC_L)
            inv = interleave_inverse(version, EC_L)
            assert len(inv) == sum(b.total_count for b in blocks)
            for bi, b in enumerate(blocks):
                n_ec = sum(1 for k, _, blk in inv if k == "ec" and blk == bi)
                n_dc = sum(1 for k, _, blk in inv if k == "data" and blk == bi)
                assert n_dc == b.data_count
                assert n_ec == b.total_count - b.data_count

    def test_data_buffer_offsets_are_a_permutation(self):
        for version in range(1, 13):
            inv = interleave_inverse(version, EC_L)
            offs = sorted(b for k, b, _ in inv if k == "data")
            assert offs == list(range(len(offs)))


# --------------------------------------------------------------------------
# 4. provenance 인과 검증 (스펙 9장 4 — 가장 강한 역매핑 검증)
# --------------------------------------------------------------------------


class TestProvenance:
    def test_kind_partition(self):
        for url in EXAMPLE_URLS:
            art = encode(url, mask_pattern=0)
            prov = provenance(url, art)
            fm = function_mask(art.version, art.ec)
            assert np.array_equal(prov.kind == KIND_FUNCTION, fm)
            # 모든 자유 모듈이 정확히 하나의 비-function kind 를 가진다.
            assert int((prov.kind[~fm] == KIND_FUNCTION).sum()) == 0
            # 헤더 비트 수: v1~9 는 12비트.
            header = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, art.version)
            assert int((prov.kind == KIND_MODE_HEADER).sum()) == 4
            assert int((prov.kind == KIND_LENGTH_HEADER).sum()) == header - 4
            # 문자 비트 = 8 × UTF-8 바이트 수.
            assert int((prov.kind == KIND_CHAR).sum()) == 8 * len(url_bytes(url))
            # 각 문자 인덱스는 정확히 8비트를 차지한다.
            for k in range(len(url_bytes(url))):
                assert int((prov.char_index == k).sum()) == 8
            # EC/remainder.
            n_ec = sum(
                b.total_count - b.data_count for b in base.rs_blocks(art.version, art.ec)
            )
            assert int((prov.kind == KIND_EC).sum()) == 8 * n_ec
            n_bits = bit_placement_order(art.version, art.ec).shape[0]
            assert int((prov.kind == KIND_REMAINDER).sum()) == n_bits - 8 * len(
                art.data_cache
            )
            assert (prov.block_index[prov.kind != KIND_EC] == -1).all()
            assert (prov.block_index[prov.kind == KIND_EC] >= 0).all()

    def test_header_bits_v10_is_20(self):
        """v10 이상은 길이 필드가 16비트라 헤더가 20비트다."""
        url = "https://long.example.com/" + "a" * 260
        art = encode(url, mask_pattern=0)
        assert art.version >= 10
        prov = provenance(url, art)
        assert int((prov.kind == KIND_MODE_HEADER).sum()) == 4
        assert int((prov.kind == KIND_LENGTH_HEADER).sum()) == 16

    @pytest.mark.parametrize("version", [3, 5])
    def test_provenance_causal(self, version):
        """k번째 문자를 바꾸면 변하는 모듈 ⊆ (char_index==k) ∪ (해당 블록의 ec).

        마스크는 0으로 고정한다(마스크가 바뀌면 격자 전체가 달라져 무의미).
        """
        base_url = "https://phish.example.org/login/verify?id=0000000000"
        cap = sum(b.data_count for b in base.rs_blocks(version, EC_L)) - 2
        base_url = base_url[:cap]
        art0 = encode(base_url, version=version, mask_pattern=0)
        prov = provenance(base_url, art0)
        nb = len(url_bytes(base_url))

        tested = 0
        for k in range(nb):
            orig = base_url[k]
            repl = "Z" if orig != "Z" else "Y"
            mutated = base_url[:k] + repl + base_url[k + 1 :]
            art1 = encode(mutated, version=version, mask_pattern=0)
            changed = art0.modules != art1.modules
            if not changed.any():
                continue  # 이론상 없지만 우연 충돌 방어
            blocks = char_block(base_url, art0, k)
            allowed = (prov.char_index == k) | (
                (prov.kind == KIND_EC)
                & np.isin(prov.block_index, sorted(blocks))
            )
            extra = changed & ~allowed
            assert not extra.any(), (
                f"v{version} k={k}: 허용 밖 모듈 {int(extra.sum())}개 변경"
            )
            # 문자 모듈 중 최소 하나는 실제로 변해야 한다(공허한 통과 방지).
            assert (changed & (prov.char_index == k)).any()
            tested += 1
        assert tested == nb

    def test_provenance_pad_includes_terminator(self):
        """종단자·정렬 0비트는 pad 로 분류된다(문서화된 규약)."""
        url = "https://e.com/short"
        art = encode(url, version=5, mask_pattern=0)
        prov = provenance(url, art)
        header = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, art.version)
        data_bits = sum(b.data_count * 8 for b in base.rs_blocks(art.version, art.ec))
        expected_pad = data_bits - header - 8 * len(url_bytes(url))
        assert int((prov.kind == KIND_PAD).sum()) == expected_pad


# --------------------------------------------------------------------------
# 5. unmask involution (스펙 9장 5)
# --------------------------------------------------------------------------


class TestUnmask:
    def test_unmask_involution(self):
        for url in EXAMPLE_URLS:
            for mask in range(8):
                art = encode(url, mask_pattern=mask)
                once = unmask(art.modules, mask)
                twice = unmask(once, mask)
                assert np.array_equal(twice, art.modules)

    def test_unmask_recovers_raw_bits(self):
        """언마스크 후 데이터 모듈 값 = 코드워드 비트 그 자체 (스펙 1.6 mask-off)."""
        url = EXAMPLE_URLS[1]
        for mask in range(8):
            art = encode(url, version=4, mask_pattern=mask)
            raw = unmask(art.modules, mask)
            order = bit_placement_order(art.version, art.ec)
            n_bits = 8 * len(art.data_cache)
            for i in range(order.shape[0]):
                r, c = int(order[i, 0]), int(order[i, 1])
                bit = (
                    ((art.data_cache[i // 8] >> (7 - (i % 8))) & 1) == 1
                    if i < n_bits
                    else False
                )
                assert bool(raw[r, c]) == bit

    def test_unmask_preserves_function_modules(self):
        art = encode(EXAMPLE_URLS[1], version=4, mask_pattern=3)
        fm = function_mask(art.version, art.ec)
        raw = unmask(art.modules, 3)
        assert np.array_equal(raw[fm], art.modules[fm])

    def test_unmask_is_a_real_change(self):
        art = encode(EXAMPLE_URLS[1], version=4, mask_pattern=1)
        assert not np.array_equal(unmask(art.modules, 1), art.modules)


# --------------------------------------------------------------------------
# 12. 형상 + shuffle-pos 유틸
# --------------------------------------------------------------------------


class TestTensorShape:
    @pytest.mark.parametrize("features", ["all", "data_only"])
    @pytest.mark.parametrize("mask_mode", ["fixed", "auto", "off"])
    def test_tensor_shape(self, features, mask_mode):
        for url in EXAMPLE_URLS:
            art = encode(url, mask_pattern=None if mask_mode == "auto" else 0)
            x = to_tensor_channels(art, features, mask_mode)
            n = art.n
            assert x.shape == (2, n, n)
            assert n == 4 * art.version + 17
            assert x.dtype == np.float32
            assert set(np.unique(x)).issubset({0.0, 1.0})

    def test_channel1_is_data_mask(self):
        art = encode(EXAMPLE_URLS[1])
        x = to_tensor_channels(art, "data_only", "fixed")
        fm = function_mask(art.version, art.ec)
        assert np.array_equal(x[1] == 1.0, ~fm)
        assert (x[0][fm] == 0.0).all()

    def test_feat_all_keeps_function_values(self):
        art = encode(EXAMPLE_URLS[1])
        x = to_tensor_channels(art, "all", "fixed")
        assert np.array_equal(x[0] == 1.0, art.modules)

    def test_mask_off_channel_matches_unmask(self):
        art = encode(EXAMPLE_URLS[1], mask_pattern=6)
        x = to_tensor_channels(art, "all", "off")
        assert np.array_equal(x[0] == 1.0, unmask(art.modules, 6))

    def test_invalid_options(self):
        art = encode(EXAMPLE_URLS[0])
        with pytest.raises(ValueError):
            to_tensor_channels(art, "nope", "fixed")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            to_tensor_channels(art, "all", "nope")  # type: ignore[arg-type]


class TestFixedPermutation:
    def test_is_permutation_and_deterministic(self):
        p = fixed_permutation(567, 0)
        assert sorted(p.tolist()) == list(range(567))
        assert np.array_equal(p, fixed_permutation(567, 0))
        assert not np.array_equal(p, fixed_permutation(567, 1))

    def test_readonly(self):
        p = fixed_permutation(64, 2)
        with pytest.raises(ValueError):
            p[0] = 0


class TestFunctionMask:
    def test_function_mask_matches_library_blank(self):
        """기능 패턴 마스크가 map_data 진입 시점 non-None 집합과 일치하는가."""
        for version in (1, 4, 7, 10):
            fm = function_mask(version, EC_L)
            qr = qrcode.QRCode(
                version=version, error_correction=EC_L, border=0, mask_pattern=0
            )
            qr.add_data(util.QRData("a", mode=util.MODE_8BIT_BYTE))
            qr.make(fit=False)
            # 파인더 3개 코너는 반드시 기능 패턴.
            n = 4 * version + 17
            assert fm[0, 0] and fm[6, 6] and fm[n - 1, 0] and fm[0, n - 1]
            # 타이밍 패턴 행/열.
            assert fm[6, 10] and fm[10, 6]
            # 다크 모듈.
            assert fm[4 * version + 9, 8]
            # v>=7 은 버전 정보 블록 포함.
            if version >= 7:
                assert fm[n - 11, 0] and fm[0, n - 11]

    def test_function_mask_independent_of_ec(self):
        for version in (1, 5, 9):
            assert np.array_equal(
                function_mask(version, ERROR_CORRECT_L),
                function_mask(version, ERROR_CORRECT_M),
            )

    def test_function_mask_readonly(self):
        with pytest.raises(ValueError):
            function_mask(3, EC_L)[0, 0] = False


class TestTensorChannelsContract:
    """``mask_mode`` 정본 적용 시점은 ``build_stratum``이다(통합 계약).

    :func:`to_tensor_channels`는 단건 시각화·테스트 전용 경로이므로, 층 빌드 →
    ``QRGridDataset`` 경로와 **같은 텐서**를 내야 한다. 두 경로가 갈라지면
    ``mask_mode="off"``가 두 번 적용되거나 아예 적용되지 않는 사고가 난다.
    """

    URLS = [
        "http://example.com/a/b",
        "http://secure-login.example.org/verify/account.php?id=1",
    ]

    @pytest.mark.parametrize("mask_mode", ["fixed", "off"])
    @pytest.mark.parametrize("features", ["data_only", "all"])
    def test_to_tensor_channels_matches_build_stratum(self, tmp_path, mask_mode, features):
        import pandas as pd

        from qrphish.dataset import QRGridDataset, build_stratum, load_stratum_arrays
        from qrphish.mapping import to_tensor_channels
        from qrphish.qrgen import encode, natural_version

        ec = EC_L
        # 층 내 격자 크기가 같아야 중앙 패딩 없이 1:1 비교가 된다.
        version = natural_version(self.URLS[0], ec)
        stratum = f"v{version}"
        urls = [u for u in self.URLS if natural_version(u, ec) == version]
        assert urls, "동일 버전 표본이 필요하다"
        df = pd.DataFrame(
            {
                "url": urls,
                "label": [0] * len(urls),
                "group": [f"g{i}" for i in range(len(urls))],
                "url_len": [len(u) for u in urls],
                "path_depth": [1] * len(urls),
                "split": ["train"] * len(urls),
            }
        )
        cond = {"features": features}
        qr = {"ec": "L", "mask_mode": mask_mode, "mask_pattern": 0, "pad_mode": "spec"}
        build_stratum(df, stratum, cond, tmp_path / "st", qr=qr)

        arr = load_stratum_arrays(tmp_path / "st")
        ds = QRGridDataset(
            arr["X_packed"],
            arr["y"],
            int(arr["n"]),
            arr["versions"],
            int(arr["ec"]),
            features=features,
            mask_mode=mask_mode,
        )
        for i, url in enumerate(urls):
            art = encode(url, ec=ec, mask_pattern=0)
            want = to_tensor_channels(art, features=features, mask_mode=mask_mode)
            got = ds[i][0].numpy()
            assert np.array_equal(got, want), f"{url} / {features} / {mask_mode}"

    def test_off_actually_differs_from_fixed(self):
        """스모크: unmask가 실제로 격자를 바꾼다(위 동치 테스트가 자명해지지 않게)."""
        from qrphish.mapping import to_tensor_channels
        from qrphish.qrgen import encode

        art = encode(self.URLS[0], ec=EC_L, mask_pattern=0)
        a = to_tensor_channels(art, mask_mode="fixed")
        b = to_tensor_channels(art, mask_mode="off")
        assert not np.array_equal(a, b)


def test_fixed_permutation_length_matches_dataset_usage():
    """``fixed_permutation(n, seed)``의 n은 순열 길이 = 평탄화 격자 크기다."""
    from qrphish.dataset import QRGridDataset

    grid_n = 29
    y = np.zeros(4, dtype=np.uint8)
    X_packed = np.zeros((4, (grid_n * grid_n + 7) // 8), dtype=np.uint8)
    ds = QRGridDataset(X_packed, y, grid_n, 3, EC_L, shuffle_positions=True, perm_seed=7)
    assert ds.perm is not None
    assert ds.perm.shape == (grid_n * grid_n,)
    assert np.array_equal(np.sort(ds.perm), np.arange(grid_n * grid_n))
    assert np.array_equal(ds.perm, fixed_permutation(grid_n * grid_n, 7))
