"""실험 config 스키마(스펙 6절)와 yaml 로더.

pydantic 없이 dataclass + 수동 검증으로 구성한다. 알 수 없는 키, 그리고
재논의 불가 결정(`optimize=0`, `border=0`) 위반은 모두 ValueError다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

EC_LEVELS = ("L", "M", "Q", "H")
MASK_MODES = ("fixed", "auto", "off")
PAD_MODES = ("spec", "random")
URL_MODES = ("raw", "norm")
LENGTH_MATCHES = ("exact", "quantile", "none")
FEATURES = ("data_only", "all")
ARCHES = ("small_cnn", "bit_mlp", "linear_probe")
STRATA = ("v1", "v2", "v3", "v4", "v5plus")


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "data/webphish.csv"
    category_col: str = "Category"
    url_col: str = "Data"
    positive_label: str = "spam"


@dataclass(frozen=True)
class QRConfig:
    ec: str = "L"
    optimize: int = 0
    border: int = 0
    mask_mode: str = "fixed"
    mask_pattern: int = 0
    pad_mode: str = "spec"


@dataclass(frozen=True)
class ConditionConfig:
    url_mode: str = "norm"
    length_match: str = "exact"
    length_bucket: int = 1
    features: str = "data_only"
    path_filter: bool = False
    shuffle_positions: bool = False


@dataclass(frozen=True)
class StratumRules:
    primary_min_per_class: int = 1000
    secondary_min_per_class: int = 300


@dataclass(frozen=True)
class SplitConfig:
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    max_group_frac: float = 0.05


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "small_cnn"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    class_weight: str = "balanced"
    amp: bool = True


@dataclass(frozen=True)
class EvalConfig:
    n_bootstrap: int = 2000
    primary_metric: str = "auroc"


@dataclass(frozen=True)
class Config:
    seed_list: tuple[int, ...] = (0, 1, 2, 3, 4)
    data: DataConfig = DataConfig()
    qr: QRConfig = QRConfig()
    condition: ConditionConfig = ConditionConfig()
    strata: tuple[str, ...] = ("v2", "v3", "v4", "v5plus")
    stratum_rules: StratumRules = StratumRules()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    eval: EvalConfig = EvalConfig()
    output_dir: str = "artifacts"
    reports_dir: str = "reports"


def _check_unknown(raw: dict[str, Any], cls: type, where: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"알 수 없는 config 키: {where}: {', '.join(unknown)}")


def _sub(raw: dict[str, Any], key: str, cls: type) -> Any:
    section = raw.get(key, {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(f"config 섹션 '{key}'는 매핑이어야 한다 (got {type(section).__name__})")
    _check_unknown(section, cls, key)
    return cls(**section)


def _one_of(value: Any, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where}는 {allowed} 중 하나여야 한다 (got {value!r})")


def _validate(cfg: Config) -> None:
    # 재논의 불가 결정 (스펙 1.10, 12절 #3)
    if cfg.qr.optimize != 0:
        raise ValueError("qr.optimize는 0으로 고정한다 (단일 8비트 바이트 청크). 변경 금지.")
    if cfg.qr.border != 0:
        raise ValueError("qr.border는 0으로 고정한다.")

    _one_of(cfg.qr.ec, EC_LEVELS, "qr.ec")
    _one_of(cfg.qr.mask_mode, MASK_MODES, "qr.mask_mode")
    _one_of(cfg.qr.pad_mode, PAD_MODES, "qr.pad_mode")
    if not 0 <= cfg.qr.mask_pattern <= 7:
        raise ValueError(f"qr.mask_pattern은 0~7이어야 한다 (got {cfg.qr.mask_pattern})")

    _one_of(cfg.condition.url_mode, URL_MODES, "condition.url_mode")
    _one_of(cfg.condition.length_match, LENGTH_MATCHES, "condition.length_match")
    _one_of(cfg.condition.features, FEATURES, "condition.features")
    if cfg.condition.length_match == "exact" and cfg.condition.length_bucket != 1:
        raise ValueError("length_match='exact'에서 length_bucket은 1이어야 한다")
    if cfg.condition.length_bucket < 1:
        raise ValueError("condition.length_bucket은 1 이상이어야 한다")

    _one_of(cfg.model.arch, ARCHES, "model.arch")
    if cfg.model.batch_size < 1 or cfg.model.max_epochs < 1 or cfg.model.patience < 1:
        raise ValueError("model.batch_size/max_epochs/patience는 1 이상이어야 한다")
    if cfg.model.lr <= 0:
        raise ValueError("model.lr은 양수여야 한다")

    if not cfg.seed_list:
        raise ValueError("seed_list가 비어 있다")
    for s in cfg.strata:
        _one_of(s, STRATA, "strata 원소")

    if len(cfg.split.ratios) != 3:
        raise ValueError("split.ratios는 (train, val, test) 3개여야 한다")
    if any(r <= 0 for r in cfg.split.ratios):
        raise ValueError("split.ratios 원소는 모두 양수여야 한다")
    if abs(sum(cfg.split.ratios) - 1.0) > 1e-9:
        raise ValueError(f"split.ratios 합이 1이 아니다 (got {sum(cfg.split.ratios)})")
    if not 0 < cfg.split.max_group_frac <= 1:
        raise ValueError("split.max_group_frac은 (0, 1] 범위여야 한다")

    if cfg.eval.n_bootstrap < 1:
        raise ValueError("eval.n_bootstrap은 1 이상이어야 한다")
    if cfg.stratum_rules.secondary_min_per_class > cfg.stratum_rules.primary_min_per_class:
        raise ValueError("secondary_min_per_class는 primary_min_per_class 이하여야 한다")


def from_dict(raw: dict[str, Any]) -> Config:
    """검증된 dict를 Config로 변환한다."""
    if not isinstance(raw, dict):
        raise ValueError("config 최상위는 매핑이어야 한다")
    _check_unknown(raw, Config, "<root>")

    seeds = raw.get("seed_list", (0, 1, 2, 3, 4))
    strata = raw.get("strata", ("v2", "v3", "v4", "v5plus"))
    split_raw = dict(raw.get("split") or {})
    _check_unknown(split_raw, SplitConfig, "split")
    if "ratios" in split_raw:
        split_raw["ratios"] = tuple(float(r) for r in split_raw["ratios"])
    cfg = Config(
        seed_list=tuple(int(s) for s in seeds),
        data=_sub(raw, "data", DataConfig),
        qr=_sub(raw, "qr", QRConfig),
        condition=_sub(raw, "condition", ConditionConfig),
        strata=tuple(str(s) for s in strata),
        stratum_rules=_sub(raw, "stratum_rules", StratumRules),
        split=SplitConfig(**split_raw),
        model=_sub(raw, "model", ModelConfig),
        eval=_sub(raw, "eval", EvalConfig),
        output_dir=str(raw.get("output_dir", "artifacts")),
        reports_dir=str(raw.get("reports_dir", "reports")),
    )
    _validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """yaml 파일을 읽어 검증된 Config를 반환한다."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config 파일이 없다: {p}")
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)


def to_dict(obj: Any) -> Any:
    """results.json의 config 스냅샷용 재귀 직렬화."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj
