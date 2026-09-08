"""고정 캠페인 test 집합 위의 쌍체 A/B 비교 (review_02 3절, qrphish.campaign).

합성 데이터는 "같은 키트 골격을 여러 도메인에 심은 캠페인"이다. 그래야
(1) T가 템플릿·eTLD+1 양쪽으로 격리되는지, (2) A의 train에만 같은 캠페인의
형제 도메인이 남는지를 직접 검사할 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrphish.campaign import (
    SOLO_PREFIX,
    build_campaign_split,
    length_match_campaign,
    sizematched_train_a,
)
from qrphish.config import from_dict
from qrphish.urls import load_webphish

# 20개 키트 패밀리 × 12 도메인. 도메인은 전부 다르지만 경로 골격은 20종뿐이라
# eTLD+1만 격리해서는 캠페인이 train/test에 걸쳐 남는다.
N_KITS = 20
N_DOMAINS_PER_KIT = 12
_TARGET_LEN = 45


def _pad(host: str, rest: str) -> str:
    """호스트를 늘려 전체 URL을 45바이트로 맞춘다(L-exact 매칭에서 살아남게)."""
    return host + "x" * (_TARGET_LEN - (len(host) + len(rest))) + rest


def _tag(i: int) -> str:
    """숫자 대신 **문자** 태그를 쓴다.

    ``url_template``은 숫자런을 ``#``으로 마스킹하므로 ``k01``/``k02`` 같은 이름은
    전부 같은 골격 ``/k#/...``으로 뭉쳐 캠페인 20개가 하나가 돼 버린다.
    """
    return chr(ord("a") + i // 26) + chr(ord("a") + i % 26)


def _campaign_csv(tmp_path: Path) -> Path:
    rows: list[tuple[str, str]] = []
    for k in range(N_KITS):
        kit = _tag(k)
        for d in range(N_DOMAINS_PER_KIT):
            dt = _tag(k * N_DOMAINS_PER_KIT + d)
            rows.append(("spam", _pad(f"c{kit}{dt}", f".com/{kit}/user/login.php?id={dt}")))
    # benign: 절반은 맨 도메인(설계 5.2의 함정), 절반은 각자 다른 경로
    for i in range(N_KITS * N_DOMAINS_PER_KIT):
        t = _tag(i // 26) + _tag(i % 26)
        rows.append(("ham", _pad(f"b{t}", ".org")))
        rows.append(("ham", _pad(f"n{t}", f".org/s{t}/{t}/read.html")))
    path = tmp_path / "campaign.csv"
    pd.DataFrame(rows, columns=["Category", "Data"]).to_csv(path, index=False)
    return path


@pytest.fixture
def campaign_frame(tmp_path: Path) -> pd.DataFrame:
    df, _ = load_webphish(_campaign_csv(tmp_path), "norm")
    return df


def _split(frame: pd.DataFrame, **kw):
    kw.setdefault("seed", 0)
    kw.setdefault("max_group_frac", 0.3)
    return build_campaign_split(frame, **kw)


# --- 분할 불변식 ------------------------------------------------------------
def test_test_set_isolated_by_template_and_domain(campaign_frame: pd.DataFrame) -> None:
    """T는 Model B의 train과 템플릿·eTLD+1 둘 다 겹치지 않는다."""
    cs = _split(campaign_frame)
    t_dom = set(cs.test["group_etld1"])
    assert t_dom and not (t_dom & set(cs.train_b["group_etld1"]))
    assert not (t_dom & set(cs.val["group_etld1"]))

    def tpl(f: pd.DataFrame) -> set[str]:
        s = f["template_id"].astype(str)
        return set(s[~s.str.startswith(SOLO_PREFIX)])

    assert tpl(cs.test) and not (tpl(cs.test) & tpl(cs.train_b))


def test_model_a_train_has_campaign_siblings_model_b_none(
    campaign_frame: pd.DataFrame,
) -> None:
    """A의 train에는 T와 같은 캠페인의 다른 도메인 행이 있고 B에는 없다."""
    cs = _split(campaign_frame)
    ov = cs.diagnostics["overlap"]["n_template_overlap_in_train"]
    assert ov["A"] > 0, "형제가 하나도 없으면 A/B 개입이 일어나지 않은 것이다"
    assert ov["B"] == 0
    assert len(cs.siblings) == ov["A"]
    # A = B ∪ 형제. 행 수와 집합 포함 관계 둘 다 확인한다.
    assert set(cs.train_b["url"]) <= set(cs.train_a["url"])
    assert len(cs.train_a) == len(cs.train_b) + len(cs.siblings)
    # 형제 도메인은 T의 도메인과 겹치지 않는다(eTLD+1 격리는 A에서도 유지된다).
    assert not (set(cs.siblings["group_etld1"]) & set(cs.test["group_etld1"]))
    # 그런데 형제의 템플릿은 T의 템플릿이다 — 이것이 재려는 누출이다.
    assert set(cs.siblings["template_id"]) & set(cs.test["template_id"])


def test_no_row_appears_in_two_roles(campaign_frame: pd.DataFrame) -> None:
    cs = _split(campaign_frame)
    t, v, a = set(cs.test["url"]), set(cs.val["url"]), set(cs.train_a["url"])
    assert not (t & v) and not (t & a) and not (v & a)


def test_deterministic_for_same_seed(campaign_frame: pd.DataFrame) -> None:
    a = _split(campaign_frame)
    b = _split(campaign_frame)
    for name in ("test", "val", "train_a", "train_b", "siblings"):
        assert list(getattr(a, name)["url"]) == list(getattr(b, name)["url"])
    assert a.diagnostics["overlap"] == b.diagnostics["overlap"]


def test_sibling_frac_controls_holdout_size(campaign_frame: pd.DataFrame) -> None:
    """형제 비율을 올리면 형제가 늘고 T가 줄어든다(성분마다 최소 1그룹씩은 남는다)."""
    lo = _split(campaign_frame, sibling_frac=0.25)
    hi = _split(campaign_frame, sibling_frac=0.75)
    assert len(hi.siblings) > len(lo.siblings)
    assert len(hi.test) < len(lo.test)
    assert len(lo.test) > 0 and len(hi.test) > 0


def test_rejects_bad_input(campaign_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_campaign_split(campaign_frame.drop(columns=["group"]))
    with pytest.raises(ValueError):
        build_campaign_split(campaign_frame, sibling_frac=1.0)
    dup = pd.concat([campaign_frame, campaign_frame.head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        build_campaign_split(dup)


# --- 길이 매칭 --------------------------------------------------------------
def test_length_match_keeps_one_shared_test_set(campaign_frame: pd.DataFrame) -> None:
    """T는 한 번만 매칭한다 — 모델과 무관하게 같은 행 집합이어야 한다."""
    cs = _split(campaign_frame)
    m1 = length_match_campaign(cs, 1, seed=0)
    m2 = length_match_campaign(cs, 1, seed=0)
    assert list(m1.test["url"]) == list(m2.test["url"])
    assert len(m1.test) > 0
    # 매칭 후에도 격리와 개입이 유지된다.
    assert not (set(m1.test["group_etld1"]) & set(m1.train_b["group_etld1"]))
    assert m1.diagnostics["overlap"]["n_template_overlap_in_train"]["B"] == 0
    # 깨끗한 기준선인 B만 정확히 매칭된다. A는 형제(거의 전부 피싱)를 전부 들고 있어
    # 클래스별 길이 주변분포가 설계상 틀어지며, 진단이 그 잔여 불균형을 보고한다.
    assert m1.diagnostics["length_match"]["B"]["all_identical"] is True
    assert "all_identical" in m1.diagnostics["length_match"]["A"]


def test_length_match_none_is_identity(campaign_frame: pd.DataFrame) -> None:
    cs = _split(campaign_frame)
    assert length_match_campaign(cs, None, seed=0) is cs


# --- 러너 스모크 ------------------------------------------------------------
def _smoke_cfg(tmp_path: Path):
    return from_dict(
        {
            "data": {"csv_path": str(_campaign_csv(tmp_path))},
            "seed_list": [0, 1],
            "strata": ["v3"],
            "split": {"max_group_frac": 0.3},
            "model": {"max_epochs": 2, "patience": 2, "batch_size": 64, "amp": False},
            "eval": {"n_bootstrap": 50},
            "reports_dir": str(tmp_path / "reports"),
            "output_dir": str(tmp_path / "artifacts"),
        }
    )


def test_run_campaign_holdout_smoke(tmp_path: Path) -> None:
    from qrphish.runner import run_campaign_holdout

    summary = run_campaign_holdout(_smoke_cfg(tmp_path), strata=["v3"])
    res = summary["strata"]["v3"]
    assert "error" not in res
    assert res["phase"] == "campaign"
    assert len(res["per_seed"]) == 2

    for row in res["per_seed"]:
        assert "error" not in row, row.get("error")
        assert row["paired_ok"] is True
        ov = row["split"]["overlap"]["n_template_overlap_in_train"]
        assert ov["A"] > 0 and ov["B"] == 0
        assert row["split"]["sizes"]["test"]["n"] == row["n_test"]
        assert 0 < row["n_test_leaky"] <= row["n_test"]
        for m in ("A", "A_sm", "B"):
            block = row["models"][m]
            assert 0.0 <= block["test"]["auroc"] <= 1.0
            assert set(block["baselines"]) == {"charngram_lr", "bytehist_lr"}
            assert (
                tmp_path / "artifacts" / "campaign" / "v3" / f"seed{row['seed']}" / m / "model.pt"
            ).exists()
        assert row["models"]["A"]["n_train"] > row["models"]["B"]["n_train"]
        # A_sm은 형제를 전부 두고 비형제만 덜어 B와 같은 train 크기가 된다.
        assert row["models"]["A_sm"]["n_train"] == row["models"]["B"]["n_train"]
        assert row["n_train_sizematch_gap"] == 0
        # A와 B가 정말 같은 T 행을 평가했는가 (쌍체 비교의 전제)
        sdir = tmp_path / "artifacts" / "campaign" / "v3" / f"seed{row['seed']}"
        with np.load(sdir / "A" / "preds_test.npz") as za, np.load(sdir / "B" / "preds_test.npz") as zb:
            assert list(za["url"]) == list(zb["url"]) and za["url"].size == row["n_test"]
            assert list(za["y"]) == list(zb["y"])

    agg = res["aggregate"]
    assert agg["n_seeds_ok"] == 2
    for tag in (
        "cnn_sm",
        "cnn_sm_leaky_vs_contrast",
        "cnn",
        "cnn_leaky_vs_contrast",
        "charngram_lr_sm",
        "bytehist_lr_sm",
        "charngram_lr",
        "bytehist_lr",
    ):
        assert agg[tag]["paired"] is True
        assert agg[tag]["delta_ci"][0] <= agg[tag]["delta_auroc"] <= agg[tag]["delta_ci"][1]
        assert 0.0 <= agg[tag]["perm_p"] <= 1.0
    # 사전 등록 판정은 주 지표(A_sm − B)에만 붙는다.
    assert agg["cnn_sm"]["verdict"]
    assert "verdict" not in agg["cnn"]
    assert agg["cnn_sm"]["model_a"] == "A_sm" and agg["cnn"]["model_a"] == "A"

    written = json.loads(
        (tmp_path / "reports" / "campaign" / "v3" / "results.json").read_text(encoding="utf-8")
    )
    assert written["stratum"] == "v3"
    assert (tmp_path / "reports" / "campaign" / "results.json").exists()


def test_run_campaign_holdout_skips_completed(tmp_path: Path) -> None:
    from qrphish.runner import run_campaign_holdout

    cfg = _smoke_cfg(tmp_path)
    rpath = tmp_path / "reports" / "campaign" / "v3" / "results.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(
        json.dumps({"stratum": "v3", "per_seed": [{"seed": 0}], "aggregate": {"sentinel": 1}}),
        encoding="utf-8",
    )
    summary = run_campaign_holdout(cfg, strata=["v3"])
    assert summary["strata"]["v3"]["aggregate"] == {"sentinel": 1}
    # 오류 기록은 반대로 다시 돌려야 한다.
    rpath.write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    summary = run_campaign_holdout(cfg, strata=["v3"])
    assert "sentinel" not in summary["strata"]["v3"]["aggregate"]


def test_length_match_campaign_keeps_train_b_subset_of_train_a(
    campaign_frame: pd.DataFrame,
) -> None:
    """길이 매칭 후에도 ``train_b ⊆ train_a``와 ``train_a = train_b ∪ siblings``가 성립한다.

    A/B를 독립으로 매칭하면 각자 다른 비형제 행을 버려 부분집합 관계가 깨지고, Δ에
    "형제 유무"가 아니라 "비형제 train 행이 다르다"는 교란이 섞인다.
    """
    cs = _split(campaign_frame, sibling_frac=0.5)
    assert len(cs.siblings) > 0
    matched = length_match_campaign(cs, bucket=1, seed=0)

    a = set(matched.train_a["url"].astype(str))
    b = set(matched.train_b["url"].astype(str))
    sib = set(matched.siblings["url"].astype(str))
    assert b <= a
    assert a == b | sib
    assert not (b & sib)
    # 매칭이 실제로 행을 버렸는지 확인한다(항등 매칭이면 이 테스트가 무의미하다).
    assert len(matched.train_b) < len(cs.train_b)


def test_sizematched_train_a_keeps_all_siblings(campaign_frame: pd.DataFrame) -> None:
    cs = _split(campaign_frame, sibling_frac=0.5)
    sm = sizematched_train_a(cs, seed=0)

    sib = set(cs.siblings["url"].astype(str))
    got = set(sm["url"].astype(str))
    assert sib <= got                                  # 형제는 하나도 버리지 않는다
    assert got <= set(cs.train_a["url"].astype(str))   # train_a 밖의 행은 없다
    assert len(sm) == len(cs.train_b)                  # B와 같은 크기
    assert sizematched_train_a(cs, seed=0)["url"].tolist() == sm["url"].tolist()
