"""고정 캠페인 test 집합 위의 쌍체 비교(G 재설계, review_02 3절).

이전 G(``run_template_split`` + matrix ``template_split`` 항목)는 eTLD+1 분할과
템플릿 합집합 분할의 AUROC를 그냥 뺐다. 그런데 진단상 **1차 test 행의 88.4%가
템플릿 분할에서는 test를 떠난다**(v3, 자카드 0.062). 학습 조건뿐 아니라 평가
대상이 통째로 바뀌므로 그 차이를 "템플릿 누출 크기"라고 부를 수 없다.

이 모듈은 리뷰가 권한 구조를 만든다: **캠페인/템플릿 격리 test 집합 T를 먼저 하나
고정**하고, 같은 T 위에서 두 모델을 비교한다.

* ``Model B`` (완전 격리) — train에 T와 같은 eTLD+1도, T와 유사한 템플릿도 없다.
* ``Model A`` (eTLD+1만 격리) — train에 T와 같은 eTLD+1은 없지만 **T와 같은
  캠페인(템플릿 클러스터)의 다른 도메인 행은 있다**. 1차 실험의 학습 조건이다.

그러면 ``ΔAUROC = A − B``가 같은 행 위의 쌍체 비교가 되고, 이것이 곧 "학습에서
템플릿 shortcut을 허용했을 때 얻는 이득"이다.

핵심 구현 포인트 (여기서 틀리면 실험이 통째로 무의미해진다)
-----------------------------------------------------------
템플릿 클러스터와 eTLD+1을 union-find로 합친 :func:`qrphish.templates.combined_group_key`
성분은 **전이적 폐포**다. 그래서 성분 단위로 test를 배정하면 "T와 템플릿이 겹치는
행"은 하나도 남김없이 T 안에 들어간다 — 즉 A의 train 후보가 B와 완전히 같아져
ΔAUROC가 항등적으로 0이 된다. (실제로 v3에서 다운샘플로 빠지는 행은 11,996 중
19개뿐이라 "빠진 행을 A에 돌려주는" 우회도 성립하지 않는다.)

따라서 test로 배정된 성분 안에서 **eTLD+1 단위로 한 번 더 쪼갠다**:

* T-도메인 → 고정 test 집합 T
* 형제(sibling) 도메인 → T와 **같은 캠페인, 다른 도메인**. A의 train에만 허용.

이것이 리뷰 문장 "train 후보 = T·val에 속하지 않는 모든 행 중 eTLD+1이 T와 겹치지
않는 행"을 만족시키는 유일한 구성이다. 형제가 없는 단일 도메인 성분은 그대로 전부
T로 간다(그 행에 대해서는 누출 기회 자체가 없으며, 진단으로 그 비율을 남긴다).

val 성분은 쪼개지 않는다. val을 쪼개면 그 형제 행이 A의 조기종료·임계값 선택에만
val 누출을 주어 A 대 B 비교에 템플릿과 무관한 교란이 끼어든다.

이 모듈은 torch를 import하지 않는다(순수 데이터 평면). 학습·평가는
``runner.run_campaign_holdout``이 맡는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qrphish.splits import LENGTH_COL, group_split, length_match_report, match_by_length
from qrphish.templates import combined_group_key, template_groups

__all__ = [
    "CampaignSplit",
    "build_campaign_split",
    "length_match_campaign",
    "sizematched_train_a",
    "SOLO_PREFIX",
    "SESOI",
    "MAJOR_THRESHOLD",
    "VERDICT_DEFINITIONS",
    "campaign_verdict",
]

# --------------------------------------------------------------------- 판정 상수
# SESOI(smallest effect size of interest) — "실질적으로 무시 가능"이라고 부를 수 있는
# ΔAUROC의 상한. review_03 5절: CI 하한이 0 이하라는 것만으로 "누출 무시 가능"이라고
# 쓰면 CI가 [-0.10, +0.30]이어도 같은 판정이 나온다. 그건 "차이를 검출하지 못했다"이지
# "차이가 없다"가 아니다. 무시 가능을 주장하려면 **사전에 정한 SESOI보다 CI 상한이
# 작다**는 등가성(equivalence) 형태의 근거가 있어야 한다.
#
# 값 0.02의 근거: 1차 실험의 층 간 AUROC 변동(v2 0.872 → v4 0.805)의 약 1/3, 시드 간
# 표준편차 수준이며, 설계 6.4의 "결론 수정" 임계 0.05보다 뚜렷이 작다. 두 임계는 서로
# 다른 질문에 답한다 — 0.02는 "무시해도 되는가", 0.05는 "결론을 고쳐야 하는가".
SESOI = 0.02
# 검출된 이득이 이 크기 이상이면 논문의 결론 자체를 수정해야 한다(설계 6.4).
MAJOR_THRESHOLD = 0.05

VERDICT_DEFINITIONS: dict[str, str] = {
    "negligible": (
        f"무시 가능 — 쌍체 95% CI 상한 < SESOI({SESOI}). 누출 이득이 실질적으로 "
        "무시할 수 있는 크기임을 등가성 형태로 보였다."
    ),
    "not_detected": (
        f"검출 실패 — CI가 0을 포함하지만 상한 ≥ SESOI({SESOI}). 누출 허용에 따른 "
        "AUROC 상승을 검출하지 못했으나, 실질적으로 의미 있는 상승을 배제하지도 못했다."
    ),
    "detected_minor": (
        f"이득 검출, 크기 작음 — CI 하한 > 0이고 |Δ| < {MAJOR_THRESHOLD}."
    ),
    "detected_major": (
        f"결론 수정 필요 — CI 하한 > 0이고 |Δ| ≥ {MAJOR_THRESHOLD}."
    ),
    "negative": (
        "역방향 — CI 상한 < 0. 누출을 허용한 모델이 오히려 더 나쁘다(검정력·교란 점검 필요)."
    ),
    "undetermined": "판정 불가 — CI가 유한하지 않다.",
}


def campaign_verdict(lo: float, hi: float, point: float) -> dict[str, Any]:
    """사전 등록 판정 (설계 5.5·6.4, review_03 5절 반영).

    기존 규칙은 ``lo <= 0``이면 곧장 "누출 무시 가능"이었다. 그 규칙은 CI가
    ``[-0.10, +0.30]``이어도 같은 판정을 낸다 — 즉 **"차이를 검출하지 못했다"와 "차이가
    없다"를 구분하지 못한다**. 새 규칙은 SESOI를 기준으로 둘을 가른다.

    * ``negligible``     : ``hi < SESOI``  (등가성 근거가 있는 "무시 가능")
    * ``not_detected``   : ``lo <= 0 <= hi`` 이고 ``hi >= SESOI``
    * ``negative``       : ``hi < 0``
    * ``detected_minor`` : ``lo > 0`` 이고 ``|point| < MAJOR_THRESHOLD``
    * ``detected_major`` : ``lo > 0`` 이고 ``|point| >= MAJOR_THRESHOLD``

    ``negligible``이 ``hi < 0`` 인 경우까지 포함하지 않도록 ``negative``를 먼저 본다
    (역방향 효과를 "무시 가능"으로 삼키면 진단을 놓친다).

    Returns:
        ``{"code", "label", "sesoi", "major_threshold"}``. 문자열이 아니라 dict인 이유는
        표·문서가 코드로 분기하고 사람이 읽는 문장은 따로 두기 위해서다.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        code = "undetermined"
    elif hi < 0.0:
        code = "negative"
    elif lo > 0.0:
        code = (
            "detected_major" if abs(point) >= MAJOR_THRESHOLD else "detected_minor"
        )
    elif hi < SESOI:
        code = "negligible"
    else:
        code = "not_detected"
    return {
        "code": code,
        "label": VERDICT_DEFINITIONS[code],
        "sesoi": SESOI,
        "major_threshold": MAJOR_THRESHOLD,
    }


# ``template_groups``가 클러스터링에서 제외한 행에 붙이는 접두사. 이 id는 행마다
# 고유하므로 "템플릿이 겹친다"의 근거로 쓰면 안 된다.
SOLO_PREFIX = "solo:"
# 길이 매칭 시드 오프셋. 네 부분집합이 서로 다른 rng 스트림을 쓰되 seed만으로
# 완전히 결정된다.
_MATCH_SLOTS = ("test", "val", "train_a", "train_b")
# A-sizematched 다운샘플 rng의 시드 오프셋. 길이 매칭 스트림과 겹치지 않게 띄운다.
_SIZEMATCH_SEED_BASE = 900_000


@dataclass(frozen=True)
class CampaignSplit:
    """고정 T 위의 A/B 학습 집합.

    ``train_b ⊆ train_a``이고 ``train_a = train_b ∪ siblings``이다(길이 매칭 전).
    네 프레임 모두 원 프레임의 컬럼을 그대로 들고 있으며 인덱스는 재설정돼 있다.
    """

    test: pd.DataFrame
    val: pd.DataFrame
    train_a: pd.DataFrame
    train_b: pd.DataFrame
    siblings: pd.DataFrame
    diagnostics: dict[str, Any]


def _template_ids(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    return template_groups(
        df["url"].tolist(),
        threshold=float(params.get("template_threshold", 0.7)),
        k=int(params.get("template_shingle", 3)),
        min_template_tokens=int(params.get("min_template_tokens", 3)),
    )


def _clusterable(ids) -> np.ndarray:
    """``solo:`` 단독 id가 아닌 행 마스크. 템플릿 겹침 판정의 유일한 근거다."""
    arr = np.asarray(list(ids), dtype=object).astype(str)
    return ~np.char.startswith(arr, SOLO_PREFIX)


def build_campaign_split(
    df: pd.DataFrame,
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    max_group_frac: float = 0.05,
    sibling_frac: float = 0.5,
    template_params: dict[str, Any] | None = None,
) -> CampaignSplit:
    """층 프레임(정규화·층 필터 완료, 길이 매칭 **이전**)에서 캠페인 홀드아웃을 만든다.

    Args:
        df: ``url, label, group``(eTLD+1), ``url_bytes_len``을 가진 층 전체 프레임.
            ``split`` 컬럼이 있으면 무시하고 새로 만든다.
        seed: 그룹 분할과 형제 도메인 선택에 함께 쓰는 시드. 같은 시드면 같은 결과다.
        ratios: ``group_split``에 넘길 (train, val, test) 비율. test 성분의 일부
            도메인이 형제로 빠지므로 **실제 T는 ratios[2]보다 작다**.
        max_group_frac: 단일 합집합 성분이 층에서 차지할 수 있는 최대 비율.
        sibling_frac: 다중 도메인 test 성분에서 형제로 돌릴 eTLD+1 그룹의 목표 비율.
            성분마다 T와 형제 각각 최소 한 그룹은 남긴다.
        template_params: ``template_threshold``/``template_shingle``/
            ``min_template_tokens``. ``_prepare_frame``의 ``split`` 설정과 같은 값을
            넘겨야 1차 실험과 클러스터 정의가 일치한다.

    Returns:
        :class:`CampaignSplit`.
    """
    for col in ("url", "label", "group", LENGTH_COL):
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {col!r} (있는 컬럼: {list(df.columns)})")
    if not 0.0 < sibling_frac < 1.0:
        raise ValueError(f"sibling_frac은 (0, 1) 범위여야 한다 (got {sibling_frac})")

    work = df.reset_index(drop=True).copy()
    if work["url"].duplicated().any():
        raise ValueError("url이 중복됐다 — load_webphish의 dedup을 거친 프레임이어야 한다")
    work = work.drop(columns=[c for c in ("split",) if c in work.columns])

    params = dict(template_params or {})
    work["template_id"] = _template_ids(work, params).astype(str)
    work["group_etld1"] = work["group"].astype(str)
    work["group"] = combined_group_key(
        work["group_etld1"].tolist(), work["template_id"].tolist()
    ).astype(str)

    split_df, sdiag = group_split(
        work, ratios=ratios, seed=seed, max_group_frac=max_group_frac
    )
    split_df = split_df.reset_index(drop=True)

    test_pool = split_df[split_df["split"] == "test"].reset_index(drop=True)
    val = split_df[split_df["split"] == "val"].reset_index(drop=True)
    train_b = split_df[split_df["split"] == "train"].reset_index(drop=True)

    is_sibling, comp_diag = _pick_siblings(test_pool, seed=seed, sibling_frac=sibling_frac)
    test = test_pool[~is_sibling].reset_index(drop=True)
    siblings = test_pool[is_sibling].reset_index(drop=True)

    _assert_isolation(test, val, train_b, siblings)
    train_a = (
        pd.concat([train_b, siblings], ignore_index=True) if len(siblings) else train_b.copy()
    )
    train_a["split"] = "train"
    siblings = siblings.assign(split="train")

    diagnostics: dict[str, Any] = {
        "seed": int(seed),
        "sibling_frac": float(sibling_frac),
        "ratios": [float(r) for r in ratios],
        "max_group_frac": float(max_group_frac),
        "template_params": {
            "template_threshold": float(params.get("template_threshold", 0.7)),
            "template_shingle": int(params.get("template_shingle", 3)),
            "min_template_tokens": int(params.get("min_template_tokens", 3)),
        },
        "group_split": {
            "n_groups": int(sdiag.get("n_groups", 0)),
            "groups_disjoint": bool(sdiag.get("groups_disjoint", False)),
            "max_group_frac_observed": float(sdiag.get("max_group_frac_observed", 0.0)),
            "n_downsampled_groups": int(len(sdiag.get("downsampled_groups", []))),
        },
        "n_stratum": int(len(work)),
        "n_after_split": int(len(split_df)),
        "n_groups_etld1": int(work["group_etld1"].nunique()),
        "n_groups_combined": int(work["group"].nunique()),
        "components": comp_diag,
        "sizes": _sizes(test, val, train_a, train_b, siblings),
        "overlap": overlap_diagnostics(test, train_a, train_b),
    }
    return CampaignSplit(
        test=test,
        val=val,
        train_a=train_a.reset_index(drop=True),
        train_b=train_b,
        siblings=siblings.reset_index(drop=True),
        diagnostics=diagnostics,
    )


def _pick_siblings(
    test_pool: pd.DataFrame, *, seed: int, sibling_frac: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """test 성분마다 eTLD+1 그룹을 T-도메인/형제 도메인으로 가른다.

    한 성분에 eTLD+1 그룹이 하나뿐이면 형제를 만들 수 없다 — 전부 T로 둔다.
    두 개 이상이면 T와 형제 양쪽에 최소 한 그룹씩 남긴다.
    """
    n = len(test_pool)
    is_sibling = np.zeros(n, dtype=bool)
    if n == 0:
        return is_sibling, {
            "n_test_components": 0,
            "n_multi_domain_components": 0,
            "n_sibling_domains": 0,
        }

    rng = np.random.default_rng(seed + 90_001)
    comp = test_pool["group"].astype(str).to_numpy()
    dom = test_pool["group_etld1"].astype(str).to_numpy()
    n_multi = 0
    n_sib_dom = 0
    for c in sorted(set(comp.tolist())):  # 정렬로 결정성 확보
        rows = np.flatnonzero(comp == c)
        domains = sorted(set(dom[rows].tolist()))
        if len(domains) < 2:
            continue
        n_multi += 1
        k = int(np.floor(sibling_frac * len(domains)))
        k = max(1, min(k, len(domains) - 1))  # T와 형제 각각 최소 1그룹
        picked = set(rng.permutation(np.asarray(domains, dtype=object))[:k].tolist())
        n_sib_dom += len(picked)
        is_sibling[rows] = np.isin(dom[rows], list(picked))
    return is_sibling, {
        "n_test_components": int(len(set(comp.tolist()))),
        "n_multi_domain_components": int(n_multi),
        "n_sibling_domains": int(n_sib_dom),
    }


def _assert_isolation(
    test: pd.DataFrame, val: pd.DataFrame, train_b: pd.DataFrame, siblings: pd.DataFrame
) -> None:
    """설계가 요구하는 격리를 하드 검사한다. 깨지면 조용히 넘어가지 않는다."""
    t_dom = set(test["group_etld1"].astype(str))
    v_dom = set(val["group_etld1"].astype(str))
    b_dom = set(train_b["group_etld1"].astype(str))
    s_dom = set(siblings["group_etld1"].astype(str))
    if b_dom & (t_dom | v_dom):
        raise ValueError("Model B train이 T/val과 eTLD+1을 공유한다")
    if s_dom & (t_dom | v_dom):
        raise ValueError("형제 도메인이 T/val과 eTLD+1을 공유한다 — 성분 내 도메인 분할 오류")

    t_tpl = set(test.loc[_clusterable(test["template_id"]), "template_id"].astype(str))
    b_tpl = set(train_b.loc[_clusterable(train_b["template_id"]), "template_id"].astype(str))
    if t_tpl & b_tpl:
        raise ValueError("Model B train이 T와 템플릿 클러스터를 공유한다")


def overlap_diagnostics(
    test: pd.DataFrame, train_a: pd.DataFrame, train_b: pd.DataFrame
) -> dict[str, Any]:
    """A/B train에 T와 템플릿이 겹치는 행이 실제로 몇 개인지 (설계 5).

    ``n_template_overlap_in_train``은 A에서만 양수여야 하고 B에서는 0이어야 한다.
    이 값이 0이면 개입 자체가 일어나지 않은 것이므로 ΔAUROC를 해석하면 안 된다.
    """
    t_tpl = set(test.loc[_clusterable(test["template_id"]), "template_id"].astype(str))

    def count(frame: pd.DataFrame) -> int:
        if not len(frame) or not t_tpl:
            return 0
        ok = _clusterable(frame["template_id"])
        return int(frame.loc[ok, "template_id"].astype(str).isin(t_tpl).sum())

    n_a, n_b = count(train_a), count(train_b)
    leaky = (
        int(test.loc[_clusterable(test["template_id"]), "template_id"]
            .astype(str)
            .isin(set(train_a.loc[_clusterable(train_a["template_id"]), "template_id"]
                      .astype(str)))
            .sum())
        if len(train_a)
        else 0
    )
    return {
        "n_template_overlap_in_train": {"A": n_a, "B": n_b},
        "n_test_templates": int(len(t_tpl)),
        # T 행 중 A의 train에 같은 캠페인 형제가 실제로 있는 비율. 낮으면 ΔAUROC가
        # 희석되므로 판정 전에 반드시 함께 본다.
        "n_test_rows_with_sibling_in_A": leaky,
        "frac_test_rows_with_sibling_in_A": (float(leaky / len(test)) if len(test) else 0.0),
    }


def _class_ratio(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    n_pos = int((frame["label"] == 1).sum()) if n else 0
    return {"n": n, "n_pos": n_pos, "pos_ratio": (float(n_pos / n) if n else 0.0)}


def _sizes(*frames: pd.DataFrame) -> dict[str, Any]:
    names = ("test", "val", "train_a", "train_b", "siblings")
    return {name: _class_ratio(f) for name, f in zip(names, frames, strict=True)}


def length_match_campaign(cs: CampaignSplit, bucket: int | None, seed: int) -> CampaignSplit:
    """T·val·A train·B train에 길이 매칭을 적용한다 (설계 4).

    **T는 한 번만 매칭한다.** A와 B가 같은 행 집합을 평가해야 쌍체 비교가 성립하므로
    T의 매칭 rng는 모델과 무관하게 seed에만 의존한다.

    **A와 B의 train은 독립으로 매칭하지 않는다.** 두 프레임을 따로 ``match_by_length``에
    넣으면 각자 다른 비형제 행을 버려 ``train_b ⊆ train_a`` 불변식이 깨지고, 그러면 A와 B의
    차이에 "형제 행의 유무"가 아니라 "비형제 train 행이 서로 다르다"는 교란이 섞인다.

    그래서 **B를 먼저 매칭하고 A를 그 위에 쌓는다**: ``train_b = match(cs.train_b)``,
    ``train_a = train_b ∪ siblings``(형제는 전부 유지). ``train_b ⊆ train_a``와
    ``train_a = train_b ∪ siblings``가 매칭 후에도 정확히 성립한다.

    **A와 B를 동시에 정확히 매칭할 수는 없다.** 형제 행은 거의 전부 피싱(단일 클래스)이라,
    A가 형제를 전부 가지는 한 A의 클래스별 길이 주변분포는 반드시 틀어진다. 어느 쪽을
    정확히 맞출지 골라야 하고, **깨끗한 기준선인 B를 정확히 맞춘다**. A의 불균형은
    설계상 의도된 것(누출 행을 일부러 넣는다)이고, 그 크기 효과는 ``A_sm``이 걷어낸다.
    진단 ``length_match["A"]``는 그 잔여 불균형을 그대로 보고한다.

    형제 프레임을 따로 ``match_by_length``에 넣는 방식은 쓰지 않는다. 단일 클래스라
    클래스 균형이 형제 행을 **전부** 지워 버려 A와 B가 같아진다.
    """
    if bucket is None:
        return cs
    base = seed * len(_MATCH_SLOTS)
    off = {name: base + i for i, name in enumerate(_MATCH_SLOTS)}
    test = match_by_length(cs.test, bucket, off["test"])
    val = match_by_length(cs.val, bucket, off["val"])
    train_b = match_by_length(cs.train_b, bucket, off["train_b"]).reset_index(drop=True)
    siblings = cs.siblings.assign(split="train").reset_index(drop=True)
    train_a = (
        pd.concat([train_b, siblings], ignore_index=True) if len(siblings) else train_b.copy()
    )
    train_a["split"] = "train"

    diag = dict(cs.diagnostics)
    diag["length_match"] = {
        "bucket": int(bucket),
        "A": length_match_report(
            pd.concat([train_a, val, test], ignore_index=True), bucket
        ),
        "B": length_match_report(
            pd.concat([train_b, val, test], ignore_index=True), bucket
        ),
    }
    diag["sizes"] = _sizes(test, val, train_a, train_b, siblings)
    diag["overlap"] = overlap_diagnostics(test, train_a, train_b)
    return CampaignSplit(
        test=test.reset_index(drop=True),
        val=val.reset_index(drop=True),
        train_a=train_a.reset_index(drop=True),
        train_b=train_b.reset_index(drop=True),
        siblings=siblings,
        diagnostics=diag,
    )


def sizematched_train_a(cs: CampaignSplit, seed: int) -> pd.DataFrame:
    """A-sizematched(``A_sm``)의 train — 형제는 전부 두고 비형제를 무작위로 덜어낸다.

    설계 5.5가 사전 등록한 조건이다. ``train_a = train_b ∪ siblings``이므로 A는 B보다
    정확히 ``|siblings|``만큼 크고, 그 차이만으로도 AUROC가 오를 수 있다. ``A_sm``은
    **형제 행을 전부 유지한 채** 비형제 행에서 ``|siblings|``개를 무작위 제거해
    ``|train_A_sm| = |train_B|``로 맞춘다. 그러면 A_sm − B의 Δ에서 학습 표본 수 효과가
    빠지고 남는 것은 "형제 행이 비형제 행을 대체했을 때의 이득"뿐이다.

    형제가 train_b보다 많아 비형제를 다 덜어내도 크기를 맞출 수 없으면 비형제를 전부
    버린 프레임을 돌려준다(그 경우 크기가 |train_B|보다 작을 수 있다).
    """
    sib_urls = set(cs.siblings["url"].astype(str))
    is_sib = cs.train_a["url"].astype(str).isin(sib_urls).to_numpy()
    sib_rows = cs.train_a[is_sib]
    non_sib = cs.train_a[~is_sib]
    n_keep = max(0, len(cs.train_b) - len(sib_rows))
    if n_keep >= len(non_sib):
        return cs.train_a.reset_index(drop=True)
    rng = np.random.default_rng(_SIZEMATCH_SEED_BASE + int(seed))
    keep = rng.choice(len(non_sib), size=n_keep, replace=False)
    out = pd.concat(
        [non_sib.iloc[np.sort(keep)], sib_rows], ignore_index=True
    )
    out["split"] = "train"
    return out.reset_index(drop=True)
