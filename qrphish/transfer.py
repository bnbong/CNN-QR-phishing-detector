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


# --------------------------------------------------------------------- 외부 프레임
def load_external_frame(
    csv_path: str | Path,
    *,
    mode: str = "norm",
    dedup: str = "etld1",
    webphish_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """외부 평가 세트를 :func:`qrphish.urls.load_webphish`와 **같은 컬럼**으로 읽는다.

    두 형태를 모두 받는다.

    - **수집 원본**(``Category``/``Data`` 컬럼): ``qrphish.external.load_external``에 넘긴다.
      정규화·스킴 정책·WebPhish 대조 중복제거·편향 진단이 거기서 일어난다.
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
    df, stats = load_external(path, mode, dedup=dedup, webphish_csv=webphish_csv)  # type: ignore[arg-type]
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


def bias_direction_ok(ext_df: pd.DataFrame, wp_df: pd.DataFrame | None) -> dict:
    """설계 2.3·6.1 게이트 — 외부의 경로 편향이 WebPhish와 **같은 방향**이면 안 된다.

    WebPhish의 편향축은 "benign은 루트 도메인, phishing은 긴 경로"다. 외부에서도
    ``path_depth>=1`` 보유율이 benign < phishing이면 편향을 복제한 것이라 전이 실험이
    검증이 아니게 된다. 두 데이터셋 모두에서 phishing 쪽이 더 높을 때만 게이트가 깨진다.
    """

    def frac(df: pd.DataFrame) -> dict[str, float]:
        out = {}
        for lab, name in ((0, "benign"), (1, "phishing")):
            sub = df[df["label"] == lab]
            out[name] = float((sub["path_depth"] >= 1).mean()) if len(sub) else float("nan")
        return out

    ext = frac(ext_df)
    ref = frac(wp_df) if wp_df is not None and len(wp_df) else None
    ext_same = bool(ext["phishing"] > ext["benign"])
    if ref is None:
        # WebPhish 프레임을 못 받았으면 외부 방향만 기록하고 게이트는 통과시킨다
        # (판단 근거가 없는 상태에서 층을 강등하지 않는다).
        return {"external": ext, "webphish": None, "ok": True, "checked": False}
    wp_same = bool(ref["phishing"] > ref["benign"])
    return {
        "external": ext,
        "webphish": ref,
        "ok": not (ext_same and wp_same),
        "checked": True,
    }


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
    cfg: Any, ext_df: pd.DataFrame, stratum: str, seed: int
) -> tuple[pd.DataFrame, dict]:
    """``runner._prepare_frame``의 외부판. **같은 함수**에 프레임만 주입한다.

    별도 구현을 두면 두 경로가 갈라져 F-a 대 in-domain 비교가 조용히 무의미해진다.
    ``version`` 컬럼은 :func:`ensure_version_column`으로 한 번만 계산해 캐시한다.
    """
    from qrphish.runner import _prepare_frame

    _check_columns(ext_df)
    ensure_version_column(cfg, ext_df)
    return _prepare_frame(cfg, stratum, seed, frame=ext_df)


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
    for b in range(int(n_perm)):
        yp = _permute_labels_by_group(y, blocks, rng)
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
        "unit": "etld1_group",
    }


# -------------------------------------------------------------------- 베이스라인
def transfer_baselines(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    seed: int = 0,
    *,
    fit_on: str = "webphish_train",
    motif: dict | None = None,
) -> dict:
    """``fit_frame``(train)에서 fit → ``eval_frame``에 apply. 재fit하지 않는다.

    Args:
        fit_frame: ``url``/``label``(+선택 ``version``) 컬럼을 가진 학습용 프레임.
        eval_frame: 같은 컬럼의 평가 프레임.
        motif: ``{"H_fit": (N,D), "H_eval": (M,D)}`` motif 히스토그램 쌍. 없으면 생략.

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
) -> dict:
    """설계 6.1의 사전 등록 판정. 게이트가 하나라도 깨지면 ``descriptive_only``."""
    gates_passed = {k: bool(v) for k, v in gates.items()}
    have_numbers = bool(transfer_ci) and transfer_ci[0] == transfer_ci[0] and (
        null_ci_upper == null_ci_upper
    )
    if not all(gates_passed.values()) or not have_numbers:
        # 게이트가 깨졌거나 CI/바닥선을 못 낸 층은 결론에 쓰지 않는다(설계 6.1).
        label = "descriptive_only"
    elif transfer_ci[0] <= null_ci_upper:
        # 우연과 구분 불가.
        label = "collapse"
    elif in_domain_auroc is None or in_domain_auroc != in_domain_auroc:
        label = "reproduced_unknown_reference"
    else:
        drop = float(in_domain_auroc) - float(transfer_auroc)
        if drop <= DROP_STRONG:
            label = "reproduced_strong"
        elif drop <= DROP_WEAK:
            label = "reproduced_weak"
        else:
            label = "partial_collapse"
    return {"label": label, "gates_passed": gates_passed, "criteria_version": "prereg_v1"}


def gates_from(
    baselines: dict,
    top5_group_frac: float | None,
    bias_ok: bool,
    *,
    notes: dict | None = None,
) -> dict:
    """설계 6.1 게이트 네 개를 결과 블록에서 뽑아낸다.

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
        "bias_direction_ok": bool(bias_ok),
        "group_concentration_ok": bool(frac <= GATE_TOP5_GROUP_FRAC),
    }


def collection_bias_gate(
    external_csv: str | Path, reports_dir: str | Path | None = None
) -> dict | None:
    """수집 단계가 내린 편향 게이트(``bias_diagnostics.json``의 ``gate_passed``)를 찾는다.

    전이 단계에서 길이 매칭된 프레임으로 편향을 **재계산**하면 수집 단계의 판정이 조용히
    뒤집힌다(길이 매칭이 경로 보유율 분포를 바꾸므로). 수집 단계 게이트가 있으면 그것을
    승계하고, 없을 때만 재계산한다.

    찾는 곳: 외부 CSV와 같은 폴더 → ``reports_dir`` → ``reports/external/``.
    ``main``/``ext_b2`` 중 어느 블록인지는 진단에 기록된 ``output_csv``로 맞추고,
    없으면 파일명에 ``_b2_``가 있는지로 판단한다.
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
        key = None
        for name in ("main", "ext_b2"):
            blk = diag.get(name)
            if (
                isinstance(blk, dict)
                and blk.get("output_csv")
                and Path(str(blk["output_csv"])).name == csv_path.name
            ):
                key = name
                break
        if key is None:
            key = "ext_b2" if "_b2_" in csv_path.name else "main"
        blk = diag.get(key)
        if not isinstance(blk, dict) or "gate_passed" not in blk:
            continue
        return {
            "gate_passed": bool(blk["gate_passed"]),
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
