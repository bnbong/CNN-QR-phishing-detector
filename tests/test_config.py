"""config.py 검증 규칙 테스트 (스펙 6절)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qrphish.config import from_dict, load_config, to_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs" / "base.yaml"


def _base_raw() -> dict:
    with BASE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_load_base_config():
    cfg = load_config(BASE)
    assert cfg.qr.optimize == 0
    assert cfg.qr.border == 0
    assert cfg.qr.ec == "L"
    assert cfg.condition.length_match == "exact"
    assert cfg.split.ratios == (0.70, 0.15, 0.15)
    assert cfg.seed_list == (0, 1, 2, 3, 4)
    assert cfg.stratum_rules.primary_min_per_class == 1000


def test_optimize_must_be_zero():
    raw = _base_raw()
    raw["qr"]["optimize"] = 20
    with pytest.raises(ValueError, match="optimize"):
        from_dict(raw)


def test_border_must_be_zero():
    raw = _base_raw()
    raw["qr"]["border"] = 4
    with pytest.raises(ValueError, match="border"):
        from_dict(raw)


def test_unknown_key_root_and_section():
    raw = _base_raw()
    raw["mystery"] = 1
    with pytest.raises(ValueError, match="알 수 없는 config 키"):
        from_dict(raw)

    raw = _base_raw()
    raw["qr"]["mystery"] = 1
    with pytest.raises(ValueError, match="알 수 없는 config 키"):
        from_dict(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("qr", "ec", "X"),
        ("qr", "mask_mode", "sometimes"),
        ("qr", "mask_pattern", 8),
        ("condition", "url_mode", "lower"),
        ("condition", "length_match", "loose"),
        ("condition", "features", "some"),
        ("model", "arch", "resnet"),
        ("eval", "n_bootstrap", 0),
    ],
)
def test_invalid_enum_values(section, key, value):
    raw = _base_raw()
    raw[section][key] = value
    with pytest.raises(ValueError):
        from_dict(raw)


def test_ratios_must_sum_to_one():
    raw = _base_raw()
    raw["split"]["ratios"] = [0.5, 0.3, 0.3]
    with pytest.raises(ValueError, match="ratios"):
        from_dict(raw)


def test_exact_requires_bucket_one():
    raw = _base_raw()
    raw["condition"]["length_bucket"] = 5
    with pytest.raises(ValueError, match="length_bucket"):
        from_dict(raw)
    raw["condition"]["length_match"] = "quantile"
    assert from_dict(raw).condition.length_bucket == 5


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "configs" / "nope.yaml")


def test_to_dict_roundtrip():
    cfg = load_config(BASE)
    d = to_dict(cfg)
    assert d["qr"]["optimize"] == 0
    assert d["split"]["ratios"] == [0.70, 0.15, 0.15]
    assert from_dict(d) == cfg
