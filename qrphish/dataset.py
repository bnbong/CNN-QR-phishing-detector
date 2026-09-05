"""층(stratum) 빌드와 torch Dataset (스펙 5절).

층 하나 = ``artifacts/{cond_id}/{stratum}/`` 아래 ``grids.npz`` + ``meta.parquet`` +
``stratum_meta.json``.

v5plus 층은 버전이 섞여 격자 크기가 다르다. 이 경우 층 내 **최대 n**으로 중앙 zero-pad
하고, 데이터 마스크 채널(채널 1)도 같은 방식으로 중앙 정렬해 붙인다. 패딩 영역은 두 채널
모두 0이므로 "기능 패턴"과 같은 취급을 받는다(값 0 + 데이터 아님).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from qrphish.mapping import fixed_permutation, function_mask, unmask
from qrphish.qrgen import encode, natural_version

if TYPE_CHECKING:  # 워커 A 소유 모듈. 런타임 의존을 만들지 않는다.
    from qrphish.config import ConditionConfig

__all__ = [
    "StratumMeta",
    "VERSION_SPECS",
    "ec_const",
    "version_in_spec",
    "assign_stratum",
    "build_stratum",
    "load_stratum",
    "load_stratum_arrays",
    "QRGridDataset",
    "stratum_tier",
]

VERSION_SPECS = ("v2", "v3", "v4", "v5plus")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def ec_const(ec: Any) -> int:
    """"L"/"M"/"Q"/"H" 또는 qrcode 상수를 qrcode 정수 상수로 정규화한다."""
    if isinstance(ec, int):
        return ec
    import qrcode.constants as C

    table = {
        "L": C.ERROR_CORRECT_L,
        "M": C.ERROR_CORRECT_M,
        "Q": C.ERROR_CORRECT_Q,
        "H": C.ERROR_CORRECT_H,
    }
    return table[str(ec).upper()]


def version_in_spec(version: int, spec: str) -> bool:
    if spec == "v5plus":
        return version >= 5
    return version == int(spec[1:])


def assign_stratum(version: int) -> str:
    """버전 → 층 이름. v1은 보고 전용이라 별도 이름을 준다(스펙 1.2)."""
    if version <= 1:
        return "v1"
    if version >= 5:
        return "v5plus"
    return f"v{version}"


@dataclass
class StratumMeta:
    """``stratum_meta.json``의 내용."""

    condition_id: str
    stratum: str
    n: int
    ec: int
    mask_mode: str
    mask_pattern: int | None
    pad_mode: str
    features: str
    versions: list[int]
    n_total: int
    n_by_class: dict[str, int]
    n_by_split: dict[str, int]
    class_ratio: dict[str, float]
    n_groups: dict[str, int]
    tier: str
    created_at: str
    qrphish_version: str
    git_sha: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def stratum_tier(n_per_class: int, primary_min: int = 1000, secondary_min: int = 300) -> str:
    """층 채택 등급(스펙 1.2, 사전 등록 규칙). 기준은 **클래스당 최소 개수**다."""
    if n_per_class >= primary_min:
        return "primary"
    if n_per_class >= secondary_min:
        return "secondary"
    return "dropped"


def _getattr_chain(obj: Any, names: tuple[str, ...], default: Any) -> Any:
    for nm in names:
        if obj is not None and hasattr(obj, nm):
            return getattr(obj, nm)
        if isinstance(obj, dict) and nm in obj:
            return obj[nm]
    return default


def _qr_params(cond: Any, qr: Any) -> dict:
    """cond/qr 어느 쪽에 QR 설정이 있어도 읽어낸다(통합 시 계약 정리 대상)."""
    src = qr if qr is not None else cond
    return {
        "ec": ec_const(_getattr_chain(src, ("ec",), "L")),
        "mask_mode": _getattr_chain(src, ("mask_mode",), "fixed"),
        "mask_pattern": _getattr_chain(src, ("mask_pattern",), 0),
        "pad_mode": _getattr_chain(src, ("pad_mode",), "spec"),
    }


def _randpad_seed(url: str, pad_mode: str) -> int | None:
    """PAD-rand 조건(스펙 1.1-4). URL 해시 시드라 URL마다 결정적이다."""
    if pad_mode != "random":
        return None
    return int(zlib.crc32(url.encode("utf-8", "surrogatepass")) & 0x7FFFFFFF)


def _center_pad(grid: np.ndarray, n_max: int) -> np.ndarray:
    n = grid.shape[0]
    if n == n_max:
        return grid
    out = np.zeros((n_max, n_max), dtype=grid.dtype)
    off = (n_max - n) // 2
    out[off : off + n, off : off + n] = grid
    return out


def data_mask_for(version: int, ec: int, n_max: int) -> np.ndarray:
    """(n_max,n_max) bool 데이터 모듈 마스크. 중앙 정렬 zero-pad."""
    fm = np.asarray(function_mask(version, ec), dtype=bool)
    return _center_pad(~fm, n_max)


def build_stratum(
    df: pd.DataFrame,
    version_spec: str,
    cond: ConditionConfig,
    out: Path,
    *,
    qr: Any = None,
    condition_id: str = "",
    rules: Any = None,
    git_sha: str = "",
    length_match: dict | None = None,
) -> StratumMeta:
    """정규화·dedup·매칭·split이 끝난 df에서 한 층의 아티팩트를 만든다.

    df 필수 컬럼: ``url, label, group, url_len, path_depth, split``. ``split``은 문자열
    (train/val/test) 또는 0/1/2 정수 둘 다 받는다.

    ``length_match``는 러너가 넘기는 split별 길이 매칭 진단이다. 그대로
    ``stratum_meta.json``의 ``extra``에 기록한다(스펙 5절).

    ``mask_mode`` 는 **여기가 정본 적용 시점**이다. ``"off"``면 unmask 한 격자를
    ``grids.npz``에 저장하므로, 이후 :class:`QRGridDataset`은 마스크를 다시 건드리지
    않는다. :func:`qrphish.mapping.to_tensor_channels`는 단건 시각화·테스트용 경로로
    같은 결과를 낸다.
    """
    from datetime import UTC, datetime

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = _qr_params(cond, qr)
    ec = p["ec"]
    mask_mode = p["mask_mode"]
    pad_mode = p["pad_mode"]
    features = _getattr_chain(cond, ("features",), "data_only")

    work = df.copy()
    if "version" not in work.columns:
        work["version"] = [natural_version(u, ec) for u in work["url"]]
    work = work[[version_in_spec(int(v), version_spec) for v in work["version"]]].reset_index(
        drop=True
    )
    if len(work) == 0:
        raise ValueError(f"stratum {version_spec}: 표본이 없다")

    n_max = int(4 * work["version"].max() + 17)

    grids: list[np.ndarray] = []
    mask_used: list[int] = []
    n_pad_bytes: list[int] = []
    first_pad_cw: list[int] = []
    for url in work["url"].tolist():
        mp = None if mask_mode == "auto" else int(p["mask_pattern"] or 0)
        art = encode(
            url,
            ec=ec,
            mask_pattern=mp,
            version=None,
            randpad_seed=_randpad_seed(url, pad_mode),
        )
        mods = np.asarray(art.modules, dtype=bool)
        if mask_mode == "off":
            # 마스크를 XOR로 되돌려 순수 비트 배치 격자를 쓴다(스펙 1.6, 해석 전용).
            mods = np.asarray(unmask(mods, int(art.mask_pattern)), dtype=bool)
        grids.append(_center_pad(mods, n_max))
        mask_used.append(int(art.mask_pattern))
        n_pad_bytes.append(int(getattr(art, "n_pad_bytes", -1)))
        first_pad_cw.append(int(getattr(art, "first_pad_codeword", -1)))

    X = np.stack(grids).reshape(len(grids), -1)
    X_packed = np.packbits(X, axis=1)
    y = work["label"].to_numpy(dtype=np.uint8)

    split_raw = work["split"].tolist()
    split = np.asarray(
        [SPLIT_CODE[s] if isinstance(s, str) else int(s) for s in split_raw], dtype=np.uint8
    )
    groups = work["group"].astype(str).to_numpy()
    uniq_groups, group_id = np.unique(groups, return_inverse=True)

    np.savez_compressed(
        out / "grids.npz",
        X_packed=X_packed,
        y=y,
        n=np.int32(n_max),
        version=np.int32(int(work["version"].iloc[0])),
        versions=work["version"].to_numpy(dtype=np.int32),
        ec=np.int32(ec),
        mask_pattern=np.int32(-1 if mask_mode == "auto" else int(p["mask_pattern"] or 0)),
        mask_used=np.asarray(mask_used, dtype=np.int8),
        mask_mode=np.str_(mask_mode),
        split=split,
        group_id=group_id.astype(np.int32),
    )

    meta = pd.DataFrame(
        {
            "row_id": np.arange(len(work), dtype=np.int64),
            "url": work["url"].astype(str).to_numpy(),
            "label": y.astype(np.int64),
            "group": groups,
            "url_len": work.get("url_len", pd.Series([len(u) for u in work["url"]])).to_numpy(),
            "path_depth": work.get("path_depth", pd.Series([0] * len(work))).to_numpy(),
            "split": split.astype(np.int64),
            "version": work["version"].to_numpy(dtype=np.int64),
            "mask_used": np.asarray(mask_used, dtype=np.int64),
            "n_pad_bytes": np.asarray(n_pad_bytes, dtype=np.int64),
            "first_pad_codeword": np.asarray(first_pad_cw, dtype=np.int64),
        }
    )
    meta.to_parquet(out / "meta.parquet", index=False)

    n_by_class = {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}
    per_class_min = min(n_by_class.values()) if n_by_class else 0
    primary_min = _getattr_chain(rules, ("primary_min_per_class",), 1000)
    secondary_min = _getattr_chain(rules, ("secondary_min_per_class",), 300)

    n_by_split, class_ratio, n_groups = {}, {}, {}
    for name, code in SPLIT_CODE.items():
        m = split == code
        n_by_split[name] = int(m.sum())
        class_ratio[name] = float(y[m].mean()) if m.any() else 0.0
        n_groups[name] = int(len(np.unique(group_id[m]))) if m.any() else 0

    sm = StratumMeta(
        condition_id=condition_id,
        stratum=version_spec,
        n=n_max,
        ec=int(ec),
        mask_mode=str(mask_mode),
        mask_pattern=None if mask_mode == "auto" else int(p["mask_pattern"] or 0),
        pad_mode=str(pad_mode),
        features=str(features),
        versions=sorted(int(v) for v in work["version"].unique()),
        n_total=int(len(work)),
        n_by_class=n_by_class,
        n_by_split=n_by_split,
        class_ratio=class_ratio,
        n_groups=n_groups,
        tier=stratum_tier(per_class_min, int(primary_min), int(secondary_min)),
        created_at=datetime.now(UTC).isoformat(),
        qrphish_version=_pkg_version(),
        git_sha=git_sha,
        extra={
            "n_unique_groups": int(len(uniq_groups)),
            "per_class_min": int(per_class_min),
            # 스펙 5절: split별 **매칭 후** 클래스별 개수와 길이 히스토그램 동일성.
            "n_by_split_class": {
                name: {
                    "benign": int(((split == code) & (y == 0)).sum()),
                    "phishing": int(((split == code) & (y == 1)).sum()),
                }
                for name, code in SPLIT_CODE.items()
            },
            "length_match": length_match,
            # test에서 상위 5개 그룹이 차지하는 비율(스펙 1.4-4 진단 지표)
            "top5_test_group_frac": _top_k_group_frac(group_id[split == SPLIT_CODE["test"]], 5),
        },
    )
    (out / "stratum_meta.json").write_text(sm.to_json(), encoding="utf-8")
    return sm


def _top_k_group_frac(group_id: np.ndarray, k: int = 5) -> float:
    if group_id.size == 0:
        return 0.0
    counts = np.sort(np.bincount(group_id))[::-1]
    return float(counts[:k].sum() / group_id.size)


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("qrphish")
    except Exception:
        return "0.0.0+dev"


def load_stratum(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """``(X_packed, y, meta_df)``를 돌려준다."""
    path = Path(path)
    with np.load(path / "grids.npz", allow_pickle=False) as z:
        X_packed = z["X_packed"]
        y = z["y"]
    meta = pd.read_parquet(path / "meta.parquet")
    return X_packed, y, meta


def load_stratum_arrays(path: Path) -> dict:
    """npz의 모든 배열 + stratum_meta.json을 dict로."""
    path = Path(path)
    with np.load(path / "grids.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    mj = path / "stratum_meta.json"
    if mj.exists():
        d["stratum_meta"] = json.loads(mj.read_text(encoding="utf-8"))
    return d


class QRGridDataset(torch.utils.data.Dataset):
    """``__getitem__ -> ((2,n,n) float32, label float32)``.

    features="data_only"면 채널 0의 기능 패턴 위치를 0으로 만든다(스펙 1.7).
    shuffle_positions=True면 층 전체에 **동일한 고정 순열**을 적용한다(H4 검정).
    label_shuffle_seed가 주어지면 라벨을 셔플한다(영가설 바닥). 셔플 범위는
    ``label_shuffle_rows`` bool 마스크로 지정하며, 러너는 **train+val만** 넘긴다.
    test는 원 라벨로 남겨야 "라벨과 무관한 신호로 test를 맞출 수 있는가"라는
    영가설이 성립한다(마스크를 주지 않으면 전 행을 섞는다).

    ``mask_mode``는 **기록용 메타데이터**다. 마스크 적용/해제는 :func:`build_stratum`이
    이미 끝냈고 ``grids.npz``에 그 결과가 들어 있으므로, 여기서 다시 unmask 하지 않는다.
    """

    def __init__(
        self,
        X_packed: np.ndarray,
        y: np.ndarray,
        n: int,
        versions: np.ndarray | int,
        ec: int,
        *,
        features: str = "data_only",
        mask_mode: str = "fixed",
        shuffle_positions: bool = False,
        perm_seed: int = 0,
        label_shuffle_seed: int | None = None,
        label_shuffle_rows: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        self.n = int(n)
        self.ec = int(ec)
        self.features = features
        self.mask_mode = mask_mode
        self.X_packed = X_packed
        y = np.asarray(y, dtype=np.float32)
        if label_shuffle_seed is not None:
            rng = np.random.default_rng(label_shuffle_seed)
            if label_shuffle_rows is None:
                rows = np.arange(len(y), dtype=np.int64)
            else:
                rows = np.flatnonzero(np.asarray(label_shuffle_rows, dtype=bool))
            y = y.copy()
            y[rows] = y[rows][rng.permutation(len(rows))]
        self.y = y
        if np.isscalar(versions):
            versions = np.full(len(y), int(versions), dtype=np.int32)  # type: ignore[arg-type]
        self.versions = np.asarray(versions, dtype=np.int32)
        self.indices = (
            np.arange(len(y)) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        self._masks: dict[int, np.ndarray] = {}
        self.perm = (
            np.asarray(fixed_permutation(self.n * self.n, perm_seed), dtype=np.int64)
            if shuffle_positions
            else None
        )

    def data_mask(self, version: int) -> np.ndarray:
        m = self._masks.get(int(version))
        if m is None:
            m = data_mask_for(int(version), self.ec, self.n).astype(np.float32)
            self._masks[int(version)] = m
        return m

    @property
    def canonical_data_mask(self) -> np.ndarray:
        """층 대표 데이터 마스크(BitMLP/LinearProbe 인덱스 구성용).

        v5plus처럼 버전이 섞인 층에서 최빈 버전 하나만 쓰면 다른 버전에서만 데이터인
        위치가 통째로 입력에서 빠진다. 층에 등장하는 **모든 버전 마스크의 합집합**을
        쓴다(작은 버전의 중앙 패딩 영역은 두 채널 모두 0이라 상수로 들어갈 뿐이다).
        """
        m = np.zeros((self.n, self.n), dtype=bool)
        for v in np.unique(self.versions):
            m |= self.data_mask(int(v)).astype(bool)
        if self.perm is not None:
            m = m.reshape(-1)[self.perm].reshape(self.n, self.n)
        return m

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        n = self.n
        bits = np.unpackbits(self.X_packed[j])[: n * n].astype(np.float32)
        val = bits
        dm = self.data_mask(int(self.versions[j])).reshape(-1)
        if self.features == "data_only":
            val = val * dm
        if self.perm is not None:
            val = val[self.perm]
            dm = dm[self.perm]
        x = np.stack([val.reshape(n, n), dm.reshape(n, n)]).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[j], dtype=torch.float32)
