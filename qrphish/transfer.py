"""외부 검증(F) 전이 평가 — 설계 스펙 `EXTERNAL_VALIDATION_DESIGN.md` 3·4·6·7절.

여기서 답하는 질문 하나: **WebPhish에서 학습한 모델이, 출처 결합 구조가 다른 데이터에서도
작동하는가.** 1차 실험은 benign=Alexa 상위 도메인 / phishing=별도 수집이라 라벨과 출처가
결합돼 있어, 길이·버전·패딩을 통제하고 남은 신호가 "피싱성"인지 "수집 파이프라인의 지문"인지
원리적으로 구분할 수 없었다.

네 가지 실행 (설계 3절):

===  =====================  ================  ==================================
id   학습                    평가               답하는 질문
===  =====================  ================  ==================================
F-a  WebPhish (기존 ckpt)    외부               신호가 그대로 전이되는가 (zero-shot)
F-b  외부                    외부 (그룹 분할)    외부 데이터 **자체에** 신호가 있는가
F-c  외부                    WebPhish test     역방향 전이 (비대칭 진단)
F-d  WebPhish + 외부 혼합     양쪽 각각          공통 신호 (이번 범위 밖)
===  =====================  ================  ==================================

이 모듈이 지키는 규약 세 가지. 하나라도 깨지면 결과가 조용히 무의미해진다.

1. **외부 프레임도 WebPhish와 같은 절차**로 만든다. 층 필터 → ``group_split`` →
   split별 길이 매칭 → ``assert_length_matched``. 중복 구현을 만들지 않고
   :func:`qrphish.runner._prepare_frame`에 프레임을 주입한다. 길이 매칭을 빼면 외부에서도
   "benign은 짧다"가 성립할 때 AUROC가 길이만으로 올라가, 전이 성공/실패 판정이
   신호 전이가 아니라 길이 분포의 우연한 일치를 재게 된다(설계 3.3).
2. **임계값은 WebPhish val에서 고른 값을 그대로** 쓴다. 외부에서 다시 고르면 zero-shot이
   아니다. 주 지표는 임계값 무관한 AUROC다(설계 3.1 #4).
3. **베이스라인도 WebPhish train에서 fit한 것을 외부에 apply**한다. 외부에서 다시 fit하면
   그것은 F-a가 아니라 F-b다(설계 3.7). ``baselines.fit_*``/``apply_*`` 경로를 쓴다.

바닥선은 라벨 셔플 재학습이 아니라 **그룹 단위 라벨 순열 검정**이다. 모델이 고정이므로
재학습 없이 라벨만 섞을 수 있다. 행 단위 순열은 그룹 안의 라벨 상관을 깨서 바닥선을 너무
낮게 잡으므로 쓰지 않는다(설계 3.1 #6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "EXTERNAL_COLUMNS",
    "load_external_frame",
    "ensure_version_column",
    "prepare_external_frame",
    "build_external_stratum",
    "permutation_null",
    "row_permutation_null",
    "path_frac_by_class",
    "bias_direction_ok",
    "match_by_length_and_path",
    "path_shortcut_flag",
    "paired_delta_auroc",
    "fit_path_lr",
    "apply_path_lr",
    "transfer_baselines",
    "motif_replication",
    "verdict",
    "gates_from",
    "collection_bias_gate",
    "REQUIRED_TRANSFER_KEYS",
]

# ``load_webphish``와 동일한 계약 컬럼. 외부 로더도 이걸 그대로 내야 한다.
EXTERNAL_COLUMNS = ("url", "label", "group", "url_len", "url_bytes_len", "path_depth")

# 결과 JSON이 반드시 들고 있어야 하는 최상위 키(설계 7.5). 테스트가 이 목록을 검사한다.
REQUIRED_TRANSFER_KEYS = (
    "phase",
    "mode",
    "source_tag",
    "condition_id",
    "stratum",
    "data",
    "model",
    "null_permutation",
    "baselines",
    "comparison_to_in_domain",
    "verdict",
)

# 판정 임계값(설계 6.1). 데이터를 보기 전에 확정했고, 결과를 본 뒤 고치지 않는다.
DROP_STRONG = 0.10
DROP_WEAK = 0.20
# 게이트: 층 안에서 길이·버전 단독 LR은 우연 수준이어야 한다(설계 6.1 추가 필수 조건).
GATE_NEUTRAL = (0.48, 0.52)
GATE_TOP5_GROUP_FRAC = 0.40
# 경로 지름길 게이트: 외부 cohort에서 경로 모양만 쓰는 LR(``path_lr``)이 이 값 이상이면
# 그 세트는 경로 유무만으로 거의 갈린다(EXT-B2 같은 경우). 전이 검증으로 무의미하므로
# 판정을 descriptive_only로 내린다.
GATE_PATH_LR_MAX = 0.80


# --------------------------------------------------------------------- 외부 프레임
def load_external_frame(
    csv_path: str | Path,
    *,
    mode: str = "norm",
    dedup: str = "etld1",
    webphish_csv: str | Path | None = None,
    benign_cleaning: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """외부 평가 세트를 :func:`qrphish.urls.load_webphish`와 **같은 컬럼**으로 읽는다.

    두 형태를 모두 받는다.

    - **수집 원본**(``Category``/``Data`` 컬럼): ``qrphish.external.load_external``에 넘긴다.
      정규화·스킴 정책·WebPhish 대조 중복제거·편향 진단이 거기서 일어난다.
      ``benign_cleaning``(``clean``/``keep_phish_domains``/``no_hosting_blocklist``)도
      이 경로에서만 의미가 있다. 이미 조립된 세트는 파일 자체가 조건을 담고 있다.
    - **이미 병합·정규화가 끝난 세트**(``url``/``label`` 컬럼): 그대로 읽는다. 이 경로는
      중복 제거를 다시 하지 않으므로 ``stats["loader"]``에 그 사실을 남긴다. 수집 파이프라인이
      이미 제거를 마친 ``external_{date}.csv``와 테스트·사전 점검용 합성 CSV가 여기로 온다.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"외부 데이터 파일이 없다: {path}")
    header = set(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    if {"url", "label"} <= header:
        df, stats = _load_external_fallback(path)
        stats["loader"] = "transfer._load_external_fallback(already_normalized)"
        return df, stats
    try:
        from qrphish.external import load_external  # type: ignore[attr-defined]
    except Exception as exc:  # 수집 모듈 없이 원본 CSV를 받으면 여기서 명확히 실패한다
        raise RuntimeError(
            f"{path}는 수집 원본 형식인데 qrphish.external을 불러올 수 없다: {exc!r}"
        ) from exc
    df, stats = load_external(
        path,  # type: ignore[arg-type]
        mode,  # type: ignore[arg-type]
        dedup=dedup,  # type: ignore[arg-type]
        webphish_csv=webphish_csv,
        benign_cleaning=benign_cleaning,
    )
    stats = dict(stats or {})
    stats["loader"] = "qrphish.external.load_external"
    _check_columns(df)
    return df.reset_index(drop=True), stats


def _load_external_fallback(path: Path) -> tuple[pd.DataFrame, dict]:
    """이미 정규화·중복제거가 끝난 CSV를 읽는 최소 경로(테스트·사전 점검용)."""
    from qrphish.urls import etld1, path_depth

    df = pd.read_csv(path, encoding="utf-8")
    missing = [c for c in ("url", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"외부 CSV에 필수 컬럼이 없다: {missing} ({path})")
    df = df[df["url"].notna() & (df["url"].astype(str).str.len() > 0)].copy()
    df["url"] = df["url"].astype(str)
    df["label"] = df["label"].astype(int)
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "group" not in df.columns:
        df["group"] = [etld1(u) for u in df["url"]]
    if "url_len" not in df.columns:
        df["url_len"] = df["url"].str.len().astype(int)
    if "url_bytes_len" not in df.columns:
        df["url_bytes_len"] = [len(u.encode("utf-8")) for u in df["url"]]
    if "path_depth" not in df.columns:
        df["path_depth"] = [path_depth(u) for u in df["url"]]
    df = df.reset_index(drop=True)
    _check_columns(df)
    stats = {
        "n_final": int(len(df)),
        "n_final_benign": int((df["label"] == 0).sum()),
        "n_final_phishing": int((df["label"] == 1).sum()),
        "n_groups": int(df["group"].nunique()),
        "dedup_against_webphish": False,
    }
    return df, stats


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXTERNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"외부 프레임의 컬럼이 load_webphish 계약과 다르다. 없는 컬럼: {missing}"
        )


def filter_sources(df: pd.DataFrame, sources: list[str] | None) -> pd.DataFrame:
    """``source`` 컬럼(세미콜론 구분)에 주어진 소스가 하나라도 있는 행만 남긴다."""
    if not sources or "source" not in df.columns:
        return df
    want = {s.strip() for s in sources if s.strip()}
    keep = [
        bool(want & {t.strip() for t in str(v).split(";")}) for v in df["source"].tolist()
    ]
    return df[np.asarray(keep)].reset_index(drop=True)


def path_frac_by_class(df: pd.DataFrame) -> dict[str, float]:
    """클래스별 ``path_depth>=1`` 보유율. 한쪽이 비면 NaN."""
    out: dict[str, float] = {}
    for lab, name in ((0, "benign"), (1, "phishing")):
        sub = df[df["label"] == lab]
        out[name] = float((sub["path_depth"] >= 1).mean()) if len(sub) else float("nan")
    return out


#: 부호를 "0"으로 볼 경로 보유율 차이의 크기. 이보다 작으면 편향 방향이 없다고 본다.
BIAS_DIFF_EPS = 0.02


def bias_direction_ok(
    ext_df: pd.DataFrame, wp_df: pd.DataFrame | None, *, eps: float = BIAS_DIFF_EPS
) -> dict:
    """**공통 경로 편향 플래그** (리뷰 03 항목 4). 차단 게이트가 아니다.

    각 데이터셋에서 ``d = P(path_depth>=1 | benign) - P(path_depth>=1 | phishing)``를 재고,
    두 데이터셋의 ``d`` **부호가 같으면** ``common_direction_bias``를 켠다. 부호가 같다는 것은
    외부 세트가 WebPhish와 같은 축의 구조 차이를 (방향까지) 되풀이한다는 뜻이고, 그러면
    길이만 맞춘 cohort의 전이 AUROC는 "피싱성"의 전이인지 "경로 유무"라는 공통 편향의
    전이인지 가를 수 없다.

    옛 규칙은 ``phishing > benign``인 경우만 실패로 봤다. 실제 두 데이터셋은 모두
    ``benign > phishing``(WebPhish 0.998 대 0.443, 외부 0.868 대 0.337)이라 옛 규칙에서는
    항상 통과했다 — 잡으려던 상황을 정확히 놓치고 있었다.

    **플래그이지 차단이 아니다.** 플래그가 켜진 세트에서는 층을 버리는 대신 **매칭을 더 한다**:
    길이 + 경로 유무를 동시에 맞춘 cohort(``run_transfer(match="length_path")``)의 결과가
    그 세트의 주 판정이 되고, 길이만 맞춘 cohort의 판정은 ``reported_unmatched_path``로
    표시된다(:func:`verdict`). 플래그가 꺼져 있으면 길이 매칭 cohort가 그대로 주 판정이다.

    ``|d| <= eps``면 그 데이터셋에는 방향이 없다고 보고(부호 0), 공통 편향으로 세지 않는다.

    **방향만 본다.** 크기 축 — 경로 유무만으로 거의 완전히 갈리는 세트(EXT-B2 등) — 은
    ``path_lr`` 기준선이 맡는다(:data:`GATE_PATH_LR_MAX`, :func:`path_shortcut_flag`).

    Returns:
        ``common_direction_bias``가 플래그다. ``ok``는 그 부정으로, 표시용으로만 남긴다
        (판정을 차단하지 않는다). ``checked``가 ``False``면 참조 프레임이 없어 판단하지
        않았다는 뜻이고, 그때 플래그는 꺼진 상태다.
    """
    ext = path_frac_by_class(ext_df)
    ref = path_frac_by_class(wp_df) if wp_df is not None and len(wp_df) else None

    def diff(d: dict[str, float]) -> float:
        return float(d["benign"] - d["phishing"])

    def sign(v: float) -> int:
        if v != v:  # NaN
            return 0
        return 0 if abs(v) <= eps else (1 if v > 0 else -1)

    d_ext = diff(ext)
    res: dict = {
        "external": ext,
        "webphish": ref,
        "diff_external": d_ext,
        "sign_external": sign(d_ext),
        "eps": float(eps),
        "rule": (
            "두 데이터셋의 (benign - phishing) 경로 보유율 차이의 부호가 같으면 "
            "공통 편향으로 보고 게이트 실패"
        ),
        "rule_version": "common_sign_v2",
    }
    if ref is None:
        # WebPhish 프레임을 못 받았으면 외부 방향만 기록하고 게이트는 통과시킨다
        # (판단 근거가 없는 상태에서 층을 강등하지 않는다).
        res.update(
            {
                "diff_webphish": None,
                "sign_webphish": 0,
                "common_direction_bias": False,
                "ok": True,
                "checked": False,
            }
        )
        return res
    d_wp = diff(ref)
    same = sign(d_ext) != 0 and sign(d_ext) == sign(d_wp)
    res.update(
        {
            "diff_webphish": d_wp,
            "sign_webphish": sign(d_wp),
            "same_direction": bool(same),
            "common_direction_bias": bool(same),
            "ok": not same,
            "checked": True,
            "note": (
                f"두 데이터셋 모두 benign-phishing 경로 보유율 차이가 "
                f"{'양수' if sign(d_ext) > 0 else '음수'}다 "
                f"(외부 {d_ext:+.3f}, WebPhish {d_wp:+.3f}) — 공통 편향. "
                "길이+경로 동시 매칭 cohort의 결과를 주 판정으로 쓴다"
                if same
                else f"부호가 다르거나 한쪽이 중립이다 (외부 {d_ext:+.3f}, WebPhish {d_wp:+.3f})"
            ),
        }
    )
    return res


def match_by_length_and_path(
    df: pd.DataFrame, bucket: int | None = 1, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """이미 split별 길이 매칭이 끝난 프레임에서 **경로 유무까지** 1:1로 맞춘다.

    민감도 분석용이다(리뷰 03 항목 4). ``(split, 길이 버킷, path_depth>=1)`` 셀마다 두
    클래스를 같은 수로 잘라 내므로, 길이 주변분포 동일성은 그대로 유지된 채 경로 보유율의
    주변분포까지 두 클래스에서 같아진다. 남은 AUROC는 "길이도 경로 유무도 아닌 무엇"이다.

    행 제거만 하므로 split 간 그룹 교집합 0은 유지된다. ``bucket``은 길이 버킷 폭이고
    ``exact`` 매칭에서는 1이다. ``None``이면 길이를 셀에 넣지 않고 경로 유무만 맞춘다
    (``length_match: none`` 조건).
    """
    if not len(df):
        return df, {"n_before": 0, "n_after": 0, "n_dropped": 0, "bucket": bucket}
    work = df.reset_index(drop=True)
    b = None if bucket is None else max(int(bucket), 1)
    len_key = (
        pd.Series(["*"] * len(work))
        if b is None
        else (work["url_bytes_len"].astype(int) // b).astype(str)
    )
    cell = (
        work["split"].astype(str).to_numpy()
        + "|"
        + len_key.to_numpy().astype(str)
        + "|"
        + (work["path_depth"].astype(int) >= 1).astype(int).astype(str).to_numpy()
    )
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for _, idx in work.groupby(cell, sort=True).indices.items():
        idx = np.asarray(idx)
        lab = work["label"].to_numpy()[idx]
        pos, neg = idx[lab == 1], idx[lab == 0]
        n = min(pos.size, neg.size)
        if n == 0:
            continue
        keep.append(rng.permutation(pos)[:n])
        keep.append(rng.permutation(neg)[:n])
    if not keep:
        out = work.iloc[[]].reset_index(drop=True)
    else:
        sel = np.sort(np.concatenate(keep))
        out = work.iloc[sel].reset_index(drop=True)
    report = {
        "n_before": int(len(work)),
        "n_after": int(len(out)),
        "n_dropped": int(len(work) - len(out)),
        "bucket": b,
        "cells": int(len(set(cell.tolist()))),
        "mode": "length_path",
    }
    return out, report


def ensure_version_column(cfg: Any, ext_df: pd.DataFrame) -> pd.DataFrame:
    """외부 프레임에 ``version`` 컬럼을 **한 번만** 계산해 붙인다(캐시).

    ``_prepare_frame``은 ``version``이 없을 때만 :func:`qrphish.qrgen.natural_version`을
    행마다 돌린다. 외부 세트는 60만 행 규모이고 ``prepare_external_frame``이 층 × 시드마다
    호출되므로(주 실험은 4층 × 5시드 = 20회) 캐시가 없으면 같은 계산을 20번 반복하며
    수 분을 태운다. 층 필터 **이전에** 전체 프레임에 대해 한 번 계산해 두면 이후 호출은
    컬럼을 그대로 재사용한다.

    호출자가 넘긴 프레임을 그대로 수정한다(그게 캐시의 목적이다). ``ec``는 ``cfg.qr.ec``에서
    읽으므로, EC가 다른 조건을 섞어 돌릴 때는 프레임을 나눠 써야 한다.
    """
    from qrphish.qrgen import natural_version
    from qrphish.runner import _get, ec_const

    if "version" not in ext_df.columns:
        ec = ec_const(_get(_get(cfg, "qr"), "ec", "L"))
        ext_df["version"] = [natural_version(u, ec) for u in ext_df["url"]]
    return ext_df


def prepare_external_frame(
    cfg: Any, ext_df: pd.DataFrame, stratum: str, seed: int, *, match: str = "length"
) -> tuple[pd.DataFrame, dict]:
    """``runner._prepare_frame``의 외부판. **같은 함수**에 프레임만 주입한다.

    별도 구현을 두면 두 경로가 갈라져 F-a 대 in-domain 비교가 조용히 무의미해진다.
    ``version`` 컬럼은 :func:`ensure_version_column`으로 한 번만 계산해 캐시한다.

    ``match="length_path"``면 표준 절차가 끝난 뒤 :func:`match_by_length_and_path`를 한 번 더
    적용한다(민감도 분석). ``"length"``(기본)는 1차 실험과 완전히 같은 경로다.
    """
    from qrphish.runner import _prepare_frame

    if match not in ("length", "length_path"):
        raise ValueError(f"match는 'length'|'length_path'여야 한다 (got {match!r})")
    _check_columns(ext_df)
    ensure_version_column(cfg, ext_df)
    df, diag = _prepare_frame(cfg, stratum, seed, frame=ext_df)
    diag["match"] = match
    if match == "length_path" and len(df):
        from qrphish.runner import _get, _length_bucket
        from qrphish.splits import assert_length_matched

        bucket = _length_bucket(_get(cfg, "condition"))
        df, rep = match_by_length_and_path(df, bucket, seed)
        diag["path_match"] = rep
        diag["n_after_match"] = int(len(df))
        if len(df) and bucket is not None:
            assert_length_matched(df, bucket)
    return df, diag


def build_external_stratum(cfg: Any, df: pd.DataFrame, stratum: str, seed: int, out_dir: Path):
    """``dataset.build_stratum`` 래퍼. cond/qr는 체크포인트 스냅샷과 같아야 한다.

    아티팩트는 반드시 ``artifacts/external/...`` 아래로 간다. 1차 체크포인트 경로
    (``artifacts/{condition_id}/``)를 침범하면 되돌릴 수 없다(설계 9절 #19).
    """
    from qrphish.dataset import build_stratum
    from qrphish.runner import _get, git_sha

    out_dir = Path(out_dir)
    return build_stratum(
        df,
        stratum,
        _get(cfg, "condition"),
        out_dir,
        qr=_get(cfg, "qr"),
        condition_id=out_dir.parent.parent.name,
        rules=_get(cfg, "stratum_rules"),
        git_sha=git_sha(),
        length_match=None,
    )


# ----------------------------------------------------------------- 순열 바닥선
def _group_blocks(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(np.asarray(groups).astype(str), kind="stable")
    g = np.asarray(groups).astype(str)[order]
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return [order[a:b] for a, b in zip(bounds, np.r_[bounds[1:], g.size], strict=True)]


def _permute_labels_by_group(y: np.ndarray, blocks: list[np.ndarray], rng) -> np.ndarray:
    """그룹 → 그룹 라벨 블록 재배정.

    그룹 ``i``의 행들은 무작위로 고른 다른 그룹 ``perm(i)``의 라벨 **구성**을 (크기가 다르면
    순환 반복해서) 받고, 그 안에서 다시 섞인다. 그룹 안의 라벨 구성이 통째로 옮겨가므로
    "같은 도메인은 라벨이 비슷하다"는 구조가 귀무 분포에도 남는다 — 라벨이 한 종류뿐인
    순수 그룹에서는 그룹 단위 교환 그대로다. 행 단위 순열은 이 구조를 깨서 바닥선을 너무
    낮게 잡는다(설계 3.1 #6).

    그룹 안 재섞기가 없으면, 모든 그룹의 라벨 구성이 같을 때(예: 도메인마다 benign·phishing이
    반반) 재배정이 항등이 되어 귀무 분포가 원 AUROC에 붙어버린다. 그러면 어떤 결과도
    ``collapse``로 찍힌다.
    """
    y = np.asarray(y)
    out = np.empty_like(y)
    perm = rng.permutation(len(blocks))
    for i, rows in enumerate(blocks):
        src = y[blocks[perm[i]]]
        taken = src[np.arange(rows.size) % src.size]
        out[rows] = taken[rng.permutation(taken.size)]
    return out


def permutation_null(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    seeds: np.ndarray | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """그룹 단위 라벨 순열 → AUROC 귀무 분포. 같은 ``seed``에서 결정적이다.

    ``seeds``가 주어지면 시드별로 AUROC를 낸 뒤 평균낸다(점추정 정의를
    ``cluster_bootstrap_by_seed``와 맞춘다). 보고값 ``ci_upper``(97.5 백분위)가
    설계 6.1의 바닥선이다.
    """
    from qrphish.evaluate import auroc as auroc_fn

    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if seeds is None:
        seeds = np.zeros(y.size, dtype=np.int64)
    seeds = np.asarray(seeds).astype(np.int64)

    codes = np.unique(seeds)
    per_seed_rows = [np.flatnonzero(seeds == c) for c in codes]
    # 순열은 시드와 무관한 그룹 축에서 한 번만 하고, 지표는 시드별로 낸다.
    blocks = _group_blocks(groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_perm), dtype=np.float64)
    n_pos = np.empty(int(n_perm), dtype=np.int64)
    for b in range(int(n_perm)):
        yp = _permute_labels_by_group(y, blocks, rng)
        n_pos[b] = int(yp.sum())
        per = [float(auroc_fn(yp[r], p[r])) for r in per_seed_rows]
        per = [v for v in per if v == v]
        vals[b] = float(np.mean(per)) if per else float("nan")
    ok = vals[np.isfinite(vals)]
    lo = float(np.percentile(ok, 2.5)) if ok.size else float("nan")
    hi = float(np.percentile(ok, 97.5)) if ok.size else float("nan")
    n_pos_obs = int(y.sum())
    return {
        "auroc_mean": float(np.mean(ok)) if ok.size else float("nan"),
        "ci": [lo, hi],
        "ci_upper": hi,
        "n_perm": int(n_perm),
        "n_valid": int(ok.size),
        "unit": "etld1_group",
        "method": "group_block_exchange_approx",
        "note": (
            "정식 순열 검정이 아니라 **그룹 블록 교환 근사**다. 교환 가능성 가정은 "
            "'eTLD+1 그룹은 서로 바꿔 놓아도 무방하다'이며, 크기가 다른 그룹 사이에서는 "
            "라벨 구성을 순환 복사하므로 양성 총수가 보존되지 않는다. 행 단위 라벨 순열 "
            "바닥선은 별도로 `null_row_permutation`에 낸다."
        ),
        "positives_observed": n_pos_obs,
        "positives_preserved": bool(np.all(n_pos == n_pos_obs)) if n_perm else True,
        "positives_range": [int(n_pos.min()), int(n_pos.max())] if n_perm else [n_pos_obs, n_pos_obs],
    }


def row_permutation_null(
    y: np.ndarray,
    p: np.ndarray,
    seeds: np.ndarray | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """참조용 **행 단위** 라벨 순열 바닥선. 양성 총수를 정확히 보존한다.

    행을 교환 가능하다고 가정하므로 그룹 안의 라벨 상관 구조를 깬다 — 그만큼 바닥선이
    낮게(=관대하게) 잡힌다. 판정에는 :func:`permutation_null`(그룹 블록 교환)을 쓰고,
    이 값은 "두 바닥선 중 어느 쪽을 써도 결론이 같은가"를 보이기 위해 함께 보고한다.
    """
    from qrphish.evaluate import auroc as auroc_fn

    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if seeds is None:
        seeds = np.zeros(y.size, dtype=np.int64)
    seeds = np.asarray(seeds).astype(np.int64)
    per_seed_rows = [np.flatnonzero(seeds == c) for c in np.unique(seeds)]

    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_perm), dtype=np.float64)
    for b in range(int(n_perm)):
        yp = y[rng.permutation(y.size)]
        per = [float(auroc_fn(yp[r], p[r])) for r in per_seed_rows]
        per = [v for v in per if v == v]
        vals[b] = float(np.mean(per)) if per else float("nan")
    ok = vals[np.isfinite(vals)]
    lo = float(np.percentile(ok, 2.5)) if ok.size else float("nan")
    hi = float(np.percentile(ok, 97.5)) if ok.size else float("nan")
    return {
        "auroc_mean": float(np.mean(ok)) if ok.size else float("nan"),
        "ci": [lo, hi],
        "ci_upper": hi,
        "n_perm": int(n_perm),
        "n_valid": int(ok.size),
        "unit": "row",
        "method": "row_label_permutation",
        "note": (
            "행 교환 가능성을 가정한 참조 바닥선. 양성 총수는 정확히 보존되지만 "
            "그룹 안 라벨 상관을 깨므로 판정 기준으로 쓰지 않는다."
        ),
        "positives_preserved": True,
    }


# -------------------------------------------------------------------- 경로 기준선
def fit_path_lr(urls_fit, y_fit, seed: int = 0):
    """경로 모양만 쓰는 LR (리뷰 03 항목 4). 입력은 :func:`qrphish.urls.shape_features`.

    길이·문자 조성·n-gram은 넣지 않는다. 이 기준선이 CNN에 가깝게 나오면, 외부 전이의
    상당 부분이 "경로/구분자 구조"라는 두 데이터셋 공통의 표면 특징으로 설명된다는 뜻이다.
    """
    from qrphish import baselines as bl
    from qrphish.urls import shape_features

    return bl._fit_core("path_lr", shape_features(urls_fit), y_fit, seed)


def apply_path_lr(model, urls) -> np.ndarray:
    from qrphish.urls import shape_features

    return model.predict(shape_features(urls))


def path_shortcut_flag(
    baselines: dict, *, threshold: float = GATE_PATH_LR_MAX
) -> dict:
    """경로 지름길 플래그 — 평가 cohort가 경로 모양만으로 갈리는가.

    ``path_lr`` AUROC가 ``threshold`` 이상이면 그 세트는 "경로 있음/없음"만으로 라벨이
    거의 결정된다(EXT-B2: Tranco 맨 도메인 benign 대 경로 있는 phishing). 그런 세트에서
    CNN이 잘 맞히는 것은 전이의 증거가 아니므로 판정을 ``descriptive_only``로 내린다.

    ``path_lr``을 못 냈으면 fail-closed로 **플래그를 켠다** — 못 잰 축을 통과로 두면
    지름길 세트가 조용히 결론에 쓰인다.
    """
    v = (baselines.get("path_lr") or {}).get("auroc")
    if v is None or v != v:
        return {
            "path_lr_auroc": None,
            "threshold": float(threshold),
            "path_shortcut_dominant": True,
            "note": "path_lr AUROC를 내지 못해 fail-closed로 플래그를 켠다.",
        }
    dominant = bool(float(v) >= float(threshold))
    return {
        "path_lr_auroc": float(v),
        "threshold": float(threshold),
        "path_shortcut_dominant": dominant,
        "note": (
            f"path_lr AUROC={float(v):.4f} >= {threshold} — 경로 모양만으로 거의 갈리는 "
            "세트다. 전이 검증으로 쓸 수 없다."
            if dominant
            else f"path_lr AUROC={float(v):.4f} < {threshold} — 경로 지름길이 지배적이지 않다."
        ),
    }


# ------------------------------------------------------------------ 쌍체 ΔAUROC
def paired_delta_auroc(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    seeds: np.ndarray | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """같은 행에서 두 점수의 AUROC 차이(``a - b``)와 그 CI. 시드 층화 그룹 부트스트랩.

    각 리샘플에서 **같은 그룹 집합**으로 두 AUROC를 모두 다시 계산하므로, 두 CI를 따로
    내서 겹침을 눈으로 보는 것과 달리 상관을 반영한 쌍체 비교가 된다(리뷰 03 항목 8).
    """
    from qrphish.evaluate import (
        _block_indices,
        _block_metric_fns,
        _mean_over_seeds,
        _seed_blocks,
        auroc,
        percentile_ci,
    )

    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if seeds is None:
        seeds = np.zeros(y.size, dtype=np.int64)
    seeds = np.asarray(seeds).astype(np.int64)

    n_codes, blocks = _seed_blocks(groups, seeds)
    fa = _block_metric_fns(y, p_a, blocks, auroc)
    fb = _block_metric_fns(y, p_b, blocks, auroc)
    point_a = _mean_over_seeds(float(auroc(y[b["rows"]], p_a[b["rows"]])) for b in blocks)
    point_b = _mean_over_seeds(float(auroc(y[b["rows"]], p_b[b["rows"]])) for b in blocks)

    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        if n_codes == 0:
            vals[i] = float("nan")
            continue
        pick = rng.integers(0, n_codes, size=n_codes)
        idx = [_block_indices(blk, pick) for blk in blocks]
        va = _mean_over_seeds(f(ix) for f, ix in zip(fa, idx, strict=True))
        vb = _mean_over_seeds(f(ix) for f, ix in zip(fb, idx, strict=True))
        vals[i] = va - vb
    lo, hi, n_valid = percentile_ci(vals, alpha)
    return {
        "auroc_a": point_a,
        "auroc_b": point_b,
        "delta": float(point_a - point_b),
        "ci": [lo, hi],
        "n_boot": int(n_boot),
        "n_valid": int(n_valid),
        "paired": True,
        "pooling": "seed_stratified_group_cluster_bootstrap",
    }


# -------------------------------------------------------------------- 베이스라인
def transfer_baselines(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    seed: int = 0,
    *,
    fit_on: str = "webphish_train",
    motif: dict | None = None,
    preds_out: dict | None = None,
) -> dict:
    """``fit_frame``(train)에서 fit → ``eval_frame``에 apply. 재fit하지 않는다.

    Args:
        fit_frame: ``url``/``label``(+선택 ``version``) 컬럼을 가진 학습용 프레임.
        eval_frame: 같은 컬럼의 평가 프레임.
        motif: ``{"H_fit": (N,D), "H_eval": (M,D)}`` motif 히스토그램 쌍. 없으면 생략.
        preds_out: 주면 기준선별 ``eval_frame`` 예측 점수 배열을 여기에 담는다. CNN과의
            쌍체 ΔAUROC(:func:`paired_delta_auroc`)를 내려면 필요하다.

    Returns:
        ``{name: {"auroc", "ci", "p", "fit_on"}}``. ``length_lr``/``version_lr``은 게이트다.
    """
    from qrphish import baselines as bl
    from qrphish.evaluate import auroc as auroc_fn
    from qrphish.evaluate import cluster_bootstrap

    y_fit = np.asarray(fit_frame["label"], dtype=np.int64)
    y_ev = np.asarray(eval_frame["label"], dtype=np.int64)
    g_ev = np.asarray(eval_frame["group"], dtype=object).astype(str)
    u_fit = np.asarray(fit_frame["url"], dtype=object)
    u_ev = np.asarray(eval_frame["url"], dtype=object)

    out: dict[str, Any] = {}

    def record(name: str, p: np.ndarray) -> None:
        p = np.asarray(p, dtype=np.float64)
        if preds_out is not None:
            preds_out[name] = p
        cb = cluster_bootstrap(y_ev, p, g_ev, auroc_fn, n_boot=200, seed=seed)
        out[name] = {
            "auroc": float(auroc_fn(y_ev, p)),
            "ci": [float(cb["lo"]), float(cb["hi"])],
            "fit_on": fit_on,
            "n": int(y_ev.size),
        }

    record("charngram_lr", bl.apply_charngram_lr(bl.fit_charngram_lr(u_fit, y_fit, seed), u_ev))
    record("bytehist_lr", bl.apply_bytehist_lr(bl.fit_bytehist_lr(u_fit, y_fit, seed), u_ev))
    record("length_lr", bl.apply_length_lr(bl.fit_length_lr(u_fit, y_fit, seed), u_ev))
    # 경로 모양 단독 기준선(리뷰 03 항목 4). 게이트가 아니라 해석용 대조다.
    record("path_lr", apply_path_lr(fit_path_lr(u_fit, y_fit, seed), u_ev))
    if "version" in fit_frame.columns and "version" in eval_frame.columns:
        m = bl.fit_version_lr(np.asarray(fit_frame["version"], dtype=float), y_fit, seed)
        record("version_lr", bl.apply_version_lr(m, np.asarray(eval_frame["version"], dtype=float)))
    if motif:
        for key in sorted(motif):
            pair = motif[key]
            if pair is None:
                continue
            m = bl.fit_motifhist_lr(pair["H_fit"], y_fit, seed)
            record(f"motif_{key}_lr", bl.apply_motifhist_lr(m, pair["H_eval"]))
    return out


# ------------------------------------------------------------------ motif 재현성
def _select_topk(enr: dict, k: int) -> list[int]:
    """WebPhish enrichment에서 상위 K motif를 고른다. **선택은 WebPhish에서만** 한다.

    사전 등록된 ``top_phishing``/``top_benign`` 목록을 먼저 쓰고(안정 motif 필터를 이미
    통과한 것들이다), 모자라면 |log OR| 순으로 채운다. 포화 motif(``ubiquitous``)는
    어느 경로에서도 넣지 않는다.
    """
    picked: list[int] = []
    for key in ("top_phishing", "top_benign"):
        for i in enr.get(key, []) or []:
            if int(i) not in picked:
                picked.append(int(i))
    if len(picked) < k:
        rest = sorted(
            (m for m in enr.get("motifs", []) if not m.get("ubiquitous")),
            key=lambda m: -abs(float(m.get("log_odds", 0.0))),
        )
        for m in rest:
            if int(m["id"]) not in picked:
                picked.append(int(m["id"]))
            if len(picked) >= k:
                break
    return picked[:k]


def _log_odds_vector(enr: dict, size: int) -> np.ndarray:
    """enrichment.json의 motif 목록 → 길이 ``2**(size*size)`` log OR 벡터(빠진 곳은 0)."""
    from qrphish.motifs import n_motifs

    v = np.zeros(n_motifs(size), dtype=np.float64)
    for m in enr.get("motifs", []):
        v[int(m["id"])] = float(m.get("log_odds", 0.0))
    return v


def _usable_mask(enr_wp: dict, ext: dict, size: int) -> np.ndarray:
    """포화(양 클래스 존재율 > 0.99)·희소(양 클래스 모두 출현율 < 1%) motif 제외.

    1차(``combine_seed_enrichments``)와 같은 기준이다.
    """
    from qrphish.motifs import n_motifs

    m = n_motifs(size)
    keep = np.zeros(m, dtype=bool)
    pres = {int(d["id"]): (float(d.get("present_phishing", 0.0)),
                           float(d.get("present_benign", 0.0)))
            for d in enr_wp.get("motifs", [])}
    for i in range(m):
        pp, pb = pres.get(i, (0.0, 0.0))
        ep, eb = float(ext["present_phishing"][i]), float(ext["present_benign"][i])
        saturated = min(pp, pb) > 0.99 or min(ep, eb) > 0.99
        sparse = max(pp, pb) < 0.01 and max(ep, eb) < 0.01
        keep[i] = not (saturated or sparse)
    return keep


def sign_match_rate(wp_log_odds: np.ndarray, ext_log_odds: np.ndarray, ids: list[int]) -> float:
    """상위 K motif의 log OR 부호 일치율. 귀무 기대값 0.5."""
    if not ids:
        return float("nan")
    a = np.sign(np.asarray(wp_log_odds)[ids])
    b = np.sign(np.asarray(ext_log_odds)[ids])
    return float(np.mean(a == b))


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / len(sa | sb))


def motif_replication(
    enr_wp: dict,
    H_ext: np.ndarray,
    y_ext: np.ndarray,
    groups_ext: np.ndarray,
    *,
    size: int = 3,
    k: int = 20,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """WebPhish에서 찾은 motif가 외부 데이터에서 다시 보이는지 (설계 4절).

    주 지표는 **전체 512종 log OR의 Spearman ρ**다. 상위 K 선택에 의존하지 않고 정보량이
    많다. 보조로 상위 K 부호 일치율(귀무 0.5)과 독립 발견 상위 K와의 Jaccard를 낸다.
    CI는 외부 eTLD+1 그룹 클러스터 부트스트랩이고, motif 축 전체를 흔들어야 해서
    지표 CI보다 낮은 ``n_boot``를 쓴다.
    """
    from scipy.stats import spearmanr

    from qrphish.motifs import _group_counts, _log_odds, group_bootstrap_weights, motif_enrichment

    ext = motif_enrichment(H_ext, y_ext, groups_ext, size=size, n_boot=0, seed=seed)
    wp_vec = _log_odds_vector(enr_wp, size)
    ext_vec = np.asarray(ext["log_odds"], dtype=np.float64)
    keep = _usable_mask(enr_wp, ext, size)

    ids = _select_topk(enr_wp, k)
    smr = sign_match_rate(wp_vec, ext_vec, ids)
    # 외부에서 **독립적으로** 발견한 상위 K (선택 규칙은 WebPhish 쪽과 같다).
    order = np.argsort(-np.abs(np.where(keep, ext_vec, 0.0)))
    ext_top = [int(i) for i in order[:k]]

    rho = float("nan")
    if keep.sum() >= 3:
        r = spearmanr(wp_vec[keep], ext_vec[keep]).statistic
        rho = float(r) if r == r else float("nan")

    # 그룹 클러스터 부트스트랩: 그룹 리샘플 가중치로 외부 log OR을 다시 계산한다.
    rho_ci = [float("nan"), float("nan")]
    smr_ci = [float("nan"), float("nan")]
    if n_boot > 0:
        presence = (np.asarray(H_ext, dtype=np.float32) > 0).astype(np.float64)
        gidx, W = group_bootstrap_weights(groups_ext, n_boot=n_boot, seed=seed)
        n_g = W.shape[1]
        if n_g:
            a, c, n1, n0 = _group_counts(presence, np.asarray(y_ext).astype(np.int64), gidx, n_g)
            boot = _log_odds(W @ a, W @ c, W @ n1, W @ n0)  # (n_boot, m)
            rhos, smrs = [], []
            for b in range(boot.shape[0]):
                if keep.sum() >= 3:
                    r = spearmanr(wp_vec[keep], boot[b][keep]).statistic
                    if r == r:
                        rhos.append(float(r))
                smrs.append(sign_match_rate(wp_vec, boot[b], ids))
            if rhos:
                rho_ci = [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))]
            smrs = [v for v in smrs if v == v]
            if smrs:
                smr_ci = [float(np.percentile(smrs, 2.5)), float(np.percentile(smrs, 97.5))]

    # 귀무 Jaccard: 512종에서 K개를 두 번 무작위로 뽑았을 때. 해석적 기대값과 시뮬레이션 CI.
    rng = np.random.default_rng(seed)
    m_all = int(keep.size)
    null_j = [
        jaccard(list(rng.choice(m_all, size=k, replace=False)),
                list(rng.choice(m_all, size=k, replace=False)))
        for _ in range(200)
    ]

    return {
        "K": int(k),
        "size": int(size),
        "n_motifs_used": int(keep.sum()),
        "topk_motifs_webphish": ids,
        "topk_motifs_external": ext_top,
        "spearman_logodds": {"rho": rho, "ci": rho_ci, "n_motifs_used": int(keep.sum())},
        "sign_match_rate": {"value": smr, "ci": smr_ci, "null": 0.5},
        "jaccard_topk": {
            "value": jaccard(ids, ext_top),
            "null_mean": float(np.mean(null_j)),
            "null_ci": [float(np.percentile(null_j, 2.5)), float(np.percentile(null_j, 97.5))],
        },
        # 외부 인과 절제(설계 4절 #4)는 이번 회차 범위 밖이다. 필드는 자리를 지킨다.
        "ablation": None,
        "verdict": _motif_verdict(rho, rho_ci, smr, smr_ci, jaccard(ids, ext_top), null_j),
    }


def _motif_verdict(rho, rho_ci, smr, smr_ci, jac, null_j) -> dict:
    """설계 6.3 — 셋 중 둘 이상 충족이면 "motif 재현"."""
    crit = {
        "spearman": bool(rho_ci[0] == rho_ci[0] and rho_ci[0] > 0.2),
        "sign_match": bool(smr_ci[0] == smr_ci[0] and smr_ci[0] > 0.5),
        "jaccard": bool(jac == jac and jac > float(np.percentile(null_j, 97.5))),
    }
    n = sum(crit.values())
    label = "reproduced" if n >= 2 else ("partial" if n == 1 else "failed")
    return {"label": label, "criteria": crit, "n_passed": n, "criteria_version": "prereg_v1"}


# ------------------------------------------------------------------------ 판정
def verdict(
    transfer_auroc: float,
    transfer_ci: list[float],
    null_ci_upper: float,
    in_domain_auroc: float | None,
    gates: dict,
    *,
    common_direction_bias: bool = False,
    path_shortcut_dominant: bool = False,
    match: str = "length",
) -> dict:
    """설계 6.1의 사전 등록 판정 + 리뷰 03 항목 4의 두 플래그.

    순서가 규칙이다.

    1. ``path_shortcut_dominant``(``path_lr`` AUROC >= :data:`GATE_PATH_LR_MAX`)이면
       ``descriptive_only``. 경로 유무만으로 갈리는 세트는 전이 검증으로 무의미하다.
    2. 게이트(길이·버전 중립, 그룹 집중도, 표본 등급)가 깨졌거나 CI/바닥선을 못 냈으면
       ``descriptive_only``.
    3. ``common_direction_bias``가 켜졌는데 cohort가 길이만 맞춘 것(``match="length"``)이면
       ``reported_unmatched_path`` — 숫자는 내되 주 판정으로 쓰지 않는다는 표시다. 그 세트의
       주 판정은 같은 조건을 ``match="length_path"``로 다시 돌린 결과가 갖는다.
    4. 그 외에는 기존 규칙(collapse / reproduced_strong / weak / partial_collapse).

    ``match="length_path"``인 cohort는 경로 유무 주변분포까지 맞췄으므로 3을 건너뛴다 —
    공통 편향 플래그가 켜져 있어도 그 결과가 주 판정이다.
    """
    gates_passed = {k: bool(v) for k, v in gates.items()}
    have_numbers = bool(transfer_ci) and transfer_ci[0] == transfer_ci[0] and (
        null_ci_upper == null_ci_upper
    )
    flags = {
        "common_direction_bias": bool(common_direction_bias),
        "path_shortcut_dominant": bool(path_shortcut_dominant),
        "match": str(match),
    }
    if path_shortcut_dominant:
        label = "descriptive_only"
        reason = "path_shortcut_dominant"
    elif not all(gates_passed.values()) or not have_numbers:
        # 게이트가 깨졌거나 CI/바닥선을 못 낸 층은 결론에 쓰지 않는다(설계 6.1).
        label = "descriptive_only"
        reason = "gate_failed" if not all(gates_passed.values()) else "missing_numbers"
    elif common_direction_bias and match != "length_path":
        label = "reported_unmatched_path"
        reason = "common_direction_bias_without_path_matching"
    elif transfer_ci[0] <= null_ci_upper:
        # 우연과 구분 불가.
        label = "collapse"
        reason = "ci_lower_below_null"
    elif in_domain_auroc is None or in_domain_auroc != in_domain_auroc:
        label = "reproduced_unknown_reference"
        reason = "no_in_domain_reference"
    else:
        drop = float(in_domain_auroc) - float(transfer_auroc)
        if drop <= DROP_STRONG:
            label = "reproduced_strong"
        elif drop <= DROP_WEAK:
            label = "reproduced_weak"
        else:
            label = "partial_collapse"
        reason = f"drop={drop:.4f}"
    return {
        "label": label,
        "gates_passed": gates_passed,
        "flags": flags,
        "reason": reason,
        "is_primary_verdict": bool(label != "reported_unmatched_path"),
        "criteria_version": "prereg_v2",
    }


def gates_from(
    baselines: dict,
    top5_group_frac: float | None,
    *,
    notes: dict | None = None,
) -> dict:
    """설계 6.1 게이트를 결과 블록에서 뽑아낸다.

    편향 방향은 더 이상 게이트가 아니다 — :func:`bias_direction_ok`의 플래그로 빠졌고,
    판정 규칙은 :func:`verdict`가 들고 있다. 경로 지름길(``path_shortcut``)도 게이트가
    아니라 플래그로, 역시 :func:`verdict`에서 처리한다.

    값이 없는 게이트는 **실패**로 둔다. 못 낸 게이트를 통과로 두면(fail-open) 베이스라인이
    터진 층이 조용히 결론에 쓰인다. 이유는 ``notes``(호출자가 넘긴 dict)에 남는다.
    """

    def neutral(name: str) -> bool:
        v = (baselines.get(name) or {}).get("auroc")
        if v is None or v != v:
            if notes is not None:
                notes[f"{name}_neutral"] = (
                    f"{name} AUROC를 내지 못해 게이트를 실패로 둔다(fail-closed)."
                )
            return False
        ok = GATE_NEUTRAL[0] <= float(v) <= GATE_NEUTRAL[1]
        if not ok and notes is not None:
            notes[f"{name}_neutral"] = (
                f"{name} AUROC={float(v):.4f}가 중립 구간 {GATE_NEUTRAL} 밖이다."
            )
        return ok

    if top5_group_frac is None and notes is not None:
        notes["group_concentration_ok"] = "top5_test_group_frac이 없어 0으로 본다."
    frac = float(top5_group_frac) if top5_group_frac is not None else 0.0
    return {
        "length_lr_neutral": neutral("length_lr"),
        "version_lr_neutral": neutral("version_lr"),
        "group_concentration_ok": bool(frac <= GATE_TOP5_GROUP_FRAC),
    }


def collection_bias_gate(
    external_csv: str | Path, reports_dir: str | Path | None = None
) -> dict | None:
    """수집 단계가 기록한 편향 플래그(``bias_diagnostics.json``)를 찾는다.

    전이 단계에서 길이 매칭된 프레임으로 편향을 **재계산**하면 수집 단계의 값이 조용히
    뒤집힌다(길이 매칭이 경로 보유율 분포를 바꾸므로). 수집 단계 기록이 있으면 그것을
    승계하고, 없을 때만 재계산한다. 이 값은 차단 게이트가 아니라
    :func:`verdict`에 넘기는 ``common_direction_bias`` 플래그다.

    찾는 곳: 외부 CSV와 같은 폴더 → ``reports_dir`` → ``reports/external/``.
    어느 블록인지는 진단에 기록된 ``output_csv``로 맞춘다(``sets`` 아래 세트별 블록 포함).
    못 맞추면 파일명에 ``_b2_``가 있는지로 ``main``/``ext_b2``를 고른다.
    """
    csv_path = Path(external_csv)
    cands = [csv_path.parent / "bias_diagnostics.json"]
    if reports_dir is not None:
        cands.append(Path(reports_dir) / "external" / "bias_diagnostics.json")
    cands.append(Path("reports/external/bias_diagnostics.json"))
    for c in cands:
        diag = read_json(c) if c.exists() else None
        if not diag:
            continue
        # 수집 CLI는 세트별 진단을 ``sets``에 넣고, 하위 호환을 위해 primary/EXT-B2를
        # ``main``/``ext_b2``에도 복제한다. 세트 이름이 늘어도 output_csv로 짝을 찾는다.
        blocks: dict[str, Any] = {
            k: v for k, v in diag.items() if isinstance(v, dict) and k != "sets"
        }
        for k, v in (diag.get("sets") or {}).items():
            if isinstance(v, dict):
                blocks.setdefault(f"sets.{k}", v)
        key = None
        for name, blk in blocks.items():
            if blk.get("output_csv") and Path(str(blk["output_csv"])).name == csv_path.name:
                key = name
                break
        if key is None:
            key = "ext_b2" if "_b2_" in csv_path.name else "main"
        blk = blocks.get(key)
        if not isinstance(blk, dict) or "gate_passed" not in blk:
            continue
        direction = blk.get("direction") or {}
        return {
            "gate_passed": bool(blk["gate_passed"]),
            "common_direction_bias": bool(
                direction.get("common_direction_bias", not blk["gate_passed"])
            ),
            "path_shortcut": blk.get("path_shortcut"),
            "gate_note": blk.get("gate_note"),
            "source": str(c),
            "block": key,
            "matched_output_csv": bool(blk.get("output_csv")),
        }
    return None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
