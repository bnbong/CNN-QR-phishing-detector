"""어휘 프로브 — CNN 임베딩에서 URL 어휘 속성을 선형으로 읽어낼 수 있는가.

review_01 "두 번째 질문도 표현을 낮추는 것이 좋습니다" 절의 D안이다.

``QR 모듈 격자 → SmallCNN(freeze) → GAP 128차원 → linear probe → URL 어휘 목표``

목표는 URL에서 직접 계산한다(3-gram 존재, 키워드 존재, 숫자 비율 등). 프로브가
셔플 기준선과 무작위 초기화 CNN 기준선을 **둘 다** 넘을 때만 유의하다고 본다.
프로브 성능은 "CNN이 문자를 복원했다"의 증거가 아니라 "그 정보가 임베딩에서
선형으로 접근 가능하다"의 증거임을 유의할 것.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

import numpy as np
import torch

from qrphish.evaluate import auroc as auroc_fn
from qrphish.evaluate import (
    bootstrap_with_resamples,
    cluster_bootstrap,
    group_bootstrap,
    precompute_cluster_resamples,
)
from qrphish.urls import strip_leading_scheme

__all__ = [
    "KEYWORDS",
    "ProbeTarget",
    "embed",
    "top_char_ngrams",
    "build_targets",
    "group_shuffle",
    "r2_score_fn",
    "fit_binary_probe",
    "fit_continuous_probe",
    "run_probe_suite",
    "PROBE_N_BOOT",
    # group_bootstrap은 evaluate로 옮겼다. 기존 import 경로 호환을 위해 재수출한다.
    "group_bootstrap",
    "metric_name",
]

# review_01의 예시(login/verify/.com)를 포함한 피싱 문헌의 상용 토큰.
KEYWORDS = (
    "login", "verify", "account", "secure", "update", "signin", "bank",
    "paypal", "confirm", "www", ".com", ".php", ".html", ".net", ".org", "http",
)

# 부트스트랩 반복 수 기본값. 500 -> 200으로 낮췄다. 목표당 3회(실제/셔플/무작위 초기화)를
# 120개 목표 x 5시드 x 3층 돌리므로 반복 수가 그대로 벽시계 시간이 된다. 95% 백분위 CI의
# 꼬리 분위수는 200 리샘플에서도 판정(하한 vs 상한 비교)을 뒤집을 만큼 흔들리지 않는다.
PROBE_N_BOOT = 200

_SPECIAL_RE = re.compile(r"[^A-Za-z0-9]")
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# --------------------------------------------------------------------------- 임베딩
def embed(
    model,
    dataset,
    indices: np.ndarray | None = None,
    batch_size: int = 256,
    device: Any = None,
) -> np.ndarray:
    """``(N, 128)`` GAP 출력(분류 헤드 직전). 모델은 freeze + eval.

    ``SmallCNN.feature_maps``의 공간 평균이 곧 ``forward``가 ``head``에 넣는 벡터다
    (dropout은 eval에서 항등). 따라서 이 임베딩은 로짓의 선형 입력과 정확히 같다.
    """
    if not hasattr(model, "feature_maps"):
        raise TypeError("embed는 feature_maps를 가진 conv 모델(small_cnn)에만 쓸 수 있다")
    device = device or next(model.parameters()).device
    model.eval()
    idx = np.arange(len(dataset)) if indices is None else np.asarray(indices, dtype=np.int64)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, idx.size, batch_size):
            chunk = idx[s : s + batch_size]
            xs = torch.stack([dataset[int(i)][0] for i in chunk]).to(device)
            z = model.feature_maps(xs).mean(dim=(2, 3))
            out.append(z.cpu().numpy().astype(np.float64))
    if not out:
        return np.zeros((0, 128), dtype=np.float64)
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------- 목표 정의
def top_char_ngrams(urls, size: int = 3, top: int = 100) -> list[str]:
    """train URL에서 빈도 상위 char n-gram. **클래스 라벨을 보지 않는다.**

    라벨을 보고 고르면 목표 선택 자체가 누출이 되어 프로브 AUROC를 부풀린다.
    """
    c: Counter[str] = Counter()
    for u in urls:
        s = str(u)
        c.update(s[i : i + size] for i in range(len(s) - size + 1))
    # 빈도 동률은 사전순으로 깨서 시드·플랫폼과 무관하게 같은 목록이 나오게 한다.
    return [g for g, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


def _host(url: str) -> str:
    # 선행 스킴만 벗긴다. ``split("://")``는 쿼리 안의 URL까지 스킴으로 오인한다.
    s = strip_leading_scheme(url)
    return s.split("/", 1)[0].split("?", 1)[0].split("@")[-1].split(":", 1)[0]


def _path_depth(url: str) -> int:
    s = strip_leading_scheme(url)
    rest = s.split("/", 1)[1] if "/" in s else ""
    rest = rest.split("?", 1)[0].split("#", 1)[0]
    return len([p for p in rest.split("/") if p])


class ProbeTarget:
    """프로브 목표 하나. ``kind``는 ``"binary"``(AUROC) 또는 ``"continuous"``(R²)."""

    __slots__ = ("name", "kind", "values", "family")

    def __init__(self, name: str, kind: str, values: np.ndarray, family: str) -> None:
        self.name = name
        self.kind = kind
        self.values = values
        self.family = family


def build_targets(
    urls, labels: np.ndarray, ngrams: list[str], keywords=KEYWORDS
) -> list[ProbeTarget]:
    """URL 배열 → 프로브 목표 목록.

    이진: n-gram 존재, 키워드 존재, path 존재, query 존재, 숫자/하이픈 포함, IP 호스트,
    그리고 참고용으로 라벨(phishing) 자체.
    연속: 숫자 비율, 특수문자 비율, 점 개수, 경로 깊이, 서브도메인 수.
    """
    us = [str(u) for u in urls]
    out: list[ProbeTarget] = []
    for g in ngrams:
        out.append(
            ProbeTarget(f"ngram:{g}", "binary", np.array([g in u for u in us], np.int64), "ngram")
        )
    for kw in keywords:
        out.append(
            ProbeTarget(
                f"keyword:{kw}", "binary",
                np.array([kw in u.lower() for u in us], np.int64), "keyword",
            )
        )
    depth = np.array([_path_depth(u) for u in us], dtype=np.float64)
    hosts = [_host(u) for u in us]
    out += [
        ProbeTarget("has_path", "binary", (depth >= 1).astype(np.int64), "structure"),
        ProbeTarget("has_query", "binary", np.array(["?" in u for u in us], np.int64), "structure"),
        ProbeTarget(
            "has_digit", "binary",
            np.array([any(ch.isdigit() for ch in u) for u in us], np.int64), "structure",
        ),
        ProbeTarget("has_hyphen", "binary", np.array(["-" in u for u in us], np.int64), "structure"),
        ProbeTarget(
            "ip_host", "binary",
            np.array([bool(_IP_RE.match(h)) for h in hosts], np.int64), "structure",
        ),
        # 라벨 자체 프로브. 임베딩이 애초에 라벨을 선형 분리하도록 학습됐으므로
        # 이 행은 "유의한 어휘 프로브"의 상한 참고선일 뿐, 어휘 증거가 아니다.
        ProbeTarget("label:phishing", "binary", np.asarray(labels, np.int64), "label"),
        ProbeTarget(
            "digit_ratio", "continuous",
            np.array([sum(c.isdigit() for c in u) / max(len(u), 1) for u in us]), "lexical",
        ),
        ProbeTarget(
            "special_ratio", "continuous",
            np.array([len(_SPECIAL_RE.findall(u)) / max(len(u), 1) for u in us]), "lexical",
        ),
        ProbeTarget("dot_count", "continuous", np.array([u.count(".") for u in us], float),
                    "lexical"),
        ProbeTarget("path_depth", "continuous", depth, "lexical"),
        ProbeTarget(
            "subdomain_count", "continuous",
            np.array([max(len(h.split(".")) - 2, 0) for h in hosts], float), "lexical",
        ),
    ]
    return out


# ------------------------------------------------------------------------- 기준선
def group_shuffle(values: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """그룹 단위 셔플 기준선.

    각 그룹에 다른 그룹(도너)을 배정하고, 그 그룹의 행에서 목표값을 복원추출로 가져온다.
    행 단위로 섞으면 그룹 안에서 목표가 거의 상수인 경우(예: 같은 eTLD+1의 ``.com``)
    셔플해도 정보가 남아 기준선이 부풀려진다. 그룹째 갈아끼워야 "임베딩↔목표"의 연결만
    끊고 목표의 주변분포와 그룹 내 상관 구조는 보존된다.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    donor = rng.permutation(len(uniq))
    per_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = np.empty_like(values)
    for g in range(len(uniq)):
        rows = per_group[g]
        src = per_group[int(donor[g])]
        out[rows] = values[rng.choice(src, size=rows.size, replace=True)]
    return out


def r2_score_fn(y: np.ndarray, p: np.ndarray) -> float:
    """결정계수. 분산이 0이면 정의되지 않으므로 NaN."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ((y - p) ** 2).sum() / ss_tot)


# --------------------------------------------------------------------------- 프로브
def standardize(z_tr: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """``z_tr``에 맞춘 StandardScaler로 train과 나머지 배열을 한 번에 표준화한다.

    임베딩은 시드·층마다 하나뿐이고 목표만 바뀌므로, 목표마다 스케일러를 다시 적합하는
    것은 그대로 낭비다(120개 목표 x 3변형 = 360회). 여기서 한 번 적합해 재사용한다.
    """
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(np.asarray(z_tr, dtype=np.float64))
    return tuple(
        np.asarray(sc.transform(np.asarray(a, dtype=np.float64)), dtype=np.float64)
        for a in (z_tr, *others)
    )


def fit_binary_probe(
    z_tr: np.ndarray, t_tr: np.ndarray, z_te: np.ndarray, seed: int = 0, scale: bool = True
) -> np.ndarray:
    """(표준화 +) L2 로지스틱 회귀 → test 결정값. 목표가 단일 클래스면 상수를 돌려준다.

    ``scale=False``는 호출부가 :func:`standardize`로 이미 표준화한 임베딩을 넘길 때 쓴다.
    solver는 lbfgs, ``tol=1e-3``, ``max_iter=300``이다. 128차원 · 수만 행의 볼록 문제라
    lbfgs는 수십 회 안에 수렴하며, 더 조인 tol은 결정값의 순위(AUROC)를 바꾸지 않는다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(t_tr)) < 2:
        return np.zeros(z_te.shape[0], dtype=np.float64)
    lr = LogisticRegression(
        solver="lbfgs", tol=1e-3, max_iter=300, C=1.0,
        class_weight="balanced", random_state=seed,
    )
    clf = make_pipeline(StandardScaler(), lr) if scale else lr
    clf.fit(z_tr, np.asarray(t_tr).astype(int))
    return np.asarray(clf.decision_function(z_te), dtype=np.float64)


def fit_continuous_probe(
    z_tr: np.ndarray, t_tr: np.ndarray, z_te: np.ndarray, alpha: float = 1.0,
    scale: bool = True,
) -> np.ndarray:
    """(표준화 +) 릿지 회귀 → test 예측값."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ridge = Ridge(alpha=alpha)
    reg = make_pipeline(StandardScaler(), ridge) if scale else ridge
    reg.fit(z_tr, np.asarray(t_tr, dtype=np.float64))
    return np.asarray(reg.predict(z_te), dtype=np.float64)


def _score_one(
    kind: str,
    z_tr: np.ndarray,
    t_tr: np.ndarray,
    z_te: np.ndarray,
    t_te: np.ndarray,
    groups_te: np.ndarray,
    seed: int,
    n_boot: int,
    with_ci: bool,
    resamples=None,
) -> dict:
    """목표 하나·변형 하나의 점추정과 CI.

    ``resamples``가 주어지면 그 리샘플 인덱스를 그대로 쓴다(그룹 리샘플을 목표마다 다시
    뽑지 않는다). 없으면 예전처럼 ``cluster_bootstrap``으로 즉석에서 뽑는다.
    임베딩은 호출부에서 이미 표준화됐다고 보고 ``scale=False``로 적합한다.
    """
    if kind == "binary":
        pred = fit_binary_probe(z_tr, t_tr, z_te, seed=seed, scale=False)
        metric_fn = auroc_fn
    else:
        pred = fit_continuous_probe(z_tr, t_tr, z_te, scale=False)
        metric_fn = r2_score_fn
    point = float(metric_fn(np.asarray(t_te), pred))
    out = {"score": point}
    if with_ci:
        if resamples is None:
            cb = cluster_bootstrap(
                np.asarray(t_te), pred, groups_te, metric_fn, n_boot=n_boot, seed=seed
            )
        else:
            cb = bootstrap_with_resamples(np.asarray(t_te), pred, resamples, metric_fn)
        out["lo"], out["hi"] = float(cb["lo"]), float(cb["hi"])
    return out


def _pack_resamples(resamples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """리샘플 인덱스 목록 → ``(flat, offsets)``.

    joblib은 큰 **배열** 인자만 메모리맵으로 워커에 넘긴다. 배열 200개짜리 리스트는
    태스크마다 통째로 피클되므로, 하나로 이어 붙여 메모리맵 대상이 되게 한다.
    """
    if not resamples:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    flat = np.concatenate(resamples).astype(np.int64, copy=False)
    offsets = np.concatenate(([0], np.cumsum([r.size for r in resamples]))).astype(np.int64)
    return flat, offsets


def _probe_one_target(
    t: ProbeTarget,
    idx: int,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    z_tr_rand: np.ndarray,
    z_te_rand: np.ndarray,
    v_tr: np.ndarray,
    v_te: np.ndarray,
    g_tr: np.ndarray,
    g_te: np.ndarray,
    seed: int,
    n_boot: int,
    flat: np.ndarray,
    offsets: np.ndarray,
) -> tuple[str, dict]:
    """목표 하나의 전체 행(실제/셔플/무작위 초기화 + 유의 판정). 병렬 태스크 단위.

    큰 배열은 모두 **인자로** 받는다. joblib이 1MB 넘는 ndarray 인자를 메모리맵으로
    바꿔 워커와 공유하므로, 클로저로 잡아 두면 태스크마다 통째로 피클된다.
    """
    packed = (flat, offsets)
    row: dict[str, Any] = {"kind": t.kind, "family": t.family}
    real = _score_one(t.kind, z_tr, v_tr, z_te, v_te, g_te, seed, n_boot, True, packed)
    # 셔플 기준선: train/test 각각 그룹 단위로 목표를 갈아끼운다.
    # 시드는 목표 인덱스에서 파생한다 — 병렬 실행 순서와 무관하게 결정적이고,
    # 모든 목표가 같은 도너 배치를 공유해 기준선이 함께 치우치는 일도 없다.
    s_tr = group_shuffle(v_tr, g_tr, seed=10_000 + seed + 7919 * idx)
    s_te = group_shuffle(v_te, g_te, seed=20_000 + seed + 7919 * idx)
    shuf = _score_one(t.kind, z_tr, s_tr, z_te, s_te, g_te, seed, n_boot, True, packed)
    rnd = _score_one(t.kind, z_tr_rand, v_tr, z_te_rand, v_te, g_te, seed, n_boot, True, packed)

    # 학습된 임베딩이 두 기준선을 **둘 다** CI 수준에서 넘어야 유의로 센다.
    # 무작위 초기화 점추정만 넘는 것으로는 구조/입력만으로 얻어지는 몫을 배제하지 못한다.
    bar = max(
        v for v in (shuf["hi"], rnd["hi"]) if np.isfinite(v)
    ) if np.isfinite(shuf["hi"]) or np.isfinite(rnd["hi"]) else float("nan")
    sig = bool(np.isfinite(real["lo"]) and np.isfinite(bar) and real["lo"] > bar)
    return t.name, {
        **row,
        "score": real["score"],
        "ci": [real["lo"], real["hi"]],
        "shuffle": shuf["score"],
        "shuffle_ci": [shuf["lo"], shuf["hi"]],
        "random_init": rnd["score"],
        "random_init_ci": [rnd["lo"], rnd["hi"]],
        "above_random_init": bool(np.isfinite(rnd["hi"]) and real["lo"] > rnd["hi"]),
        "significant": sig,
    }


def run_probe_suite(
    z_tr: np.ndarray,
    z_te: np.ndarray,
    z_tr_rand: np.ndarray,
    z_te_rand: np.ndarray,
    targets: list[ProbeTarget],
    tr_rows: np.ndarray,
    te_rows: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    n_boot: int = PROBE_N_BOOT,
    min_positive: int = 20,
    n_jobs: int | None = None,
    progress: bool = False,
) -> dict[str, dict]:
    """한 시드의 전체 프로브 결과.

    Args:
        z_tr/z_te: 학습된 CNN 임베딩. ``z_*_rand``는 무작위 초기화 CNN 임베딩.
        targets: 전체 행(층 전체) 기준 목표값.
        tr_rows/te_rows: 층 전체 인덱스 기준의 train/test 행.
        groups: 층 전체 그룹 배열(eTLD+1).
        n_jobs: joblib 프로세스 수. ``None``이면 목표가 8개 이상일 때만 ``-1``.
        progress: 목표 진행률을 10% 단위로 출력한다.

    Returns:
        ``{target_name: {kind, family, score, ci, shuffle, shuffle_ci, random_init,
        random_init_ci, above_random_init, significant, skipped?}}``

    유의 판정은 ``CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한)``이다.

    비용 구조상 두 가지를 시드·층당 한 번만 한다: (1) 그룹 부트스트랩 리샘플 인덱스,
    (2) 임베딩 표준화. 목표마다 바뀌는 것은 목표값뿐이므로 결과의 의미는 그대로다.
    """
    g_te = np.asarray(groups)[te_rows]
    g_tr = np.asarray(groups)[tr_rows]

    # (1) 리샘플 인덱스를 한 번만 만든다. 모든 목표·변형이 같은 리샘플을 공유한다.
    packed = _pack_resamples(precompute_cluster_resamples(g_te, n_boot=n_boot, seed=seed))
    # (2) 표준화도 시드당 한 번. train에 적합한 스케일러를 test에 적용한다(원래 파이프라인과 동일).
    z_tr_s, z_te_s = standardize(z_tr, z_te)
    zr_tr_s, zr_te_s = standardize(z_tr_rand, z_te_rand)

    res: dict[str, dict] = {}
    jobs: list[tuple] = []
    for idx, t in enumerate(targets):
        v_tr, v_te = t.values[tr_rows], t.values[te_rows]
        row: dict[str, Any] = {"kind": t.kind, "family": t.family}
        if t.kind == "binary":
            pos_tr, pos_te = int(np.sum(v_tr)), int(np.sum(v_te))
            neg_tr, neg_te = int(v_tr.size - pos_tr), int(v_te.size - pos_te)
            if min(pos_tr, pos_te, neg_tr, neg_te) < min_positive:
                res[t.name] = {**row, "skipped": "양·음성 표본 부족", "n_pos_test": pos_te}
                continue
        elif float(np.var(v_te)) <= 0:
            res[t.name] = {**row, "skipped": "test에서 목표 분산 0"}
            continue
        jobs.append((t, idx, v_tr, v_te))

    if not jobs:
        return res

    if n_jobs is None:
        # 태스크가 몇 개뿐이면 프로세스 기동 비용이 계산보다 크다.
        n_jobs = -1 if len(jobs) >= 8 else 1

    flat, offsets = packed

    if n_jobs == 1:
        step = max(len(jobs) // 10, 1)
        for i, job in enumerate(jobs):
            t, idx, v_tr, v_te = job
            name, row = _probe_one_target(
                t, idx, z_tr_s, z_te_s, zr_tr_s, zr_te_s, v_tr, v_te,
                g_tr, g_te, seed, n_boot, flat, offsets,
            )
            res[name] = row
            if progress and (i + 1) % step == 0:
                print(f"      목표 {i + 1}/{len(jobs)} ({100 * (i + 1) // len(jobs)}%)", flush=True)
    else:
        from joblib import Parallel, delayed

        # 10% 단위 청크로 끊어 던져 병렬 실행 중에도 진행률이 보이게 한다.
        chunk = max(len(jobs) // 10, 1)
        done = 0
        with Parallel(n_jobs=n_jobs, prefer="processes") as par:
            for s in range(0, len(jobs), chunk):
                part = jobs[s : s + chunk]
                tasks = (
                    delayed(_probe_one_target)(
                        t, idx, z_tr_s, z_te_s, zr_tr_s, zr_te_s, v_tr, v_te,
                        g_tr, g_te, seed, n_boot, flat, offsets,
                    )
                    for t, idx, v_tr, v_te in part
                )
                for name, row in par(tasks):
                    res[name] = row
                done += len(part)
                if progress:
                    print(
                        f"      목표 {done}/{len(jobs)} ({100 * done // len(jobs)}%)", flush=True
                    )
    return res


def metric_name(kind: Literal["binary", "continuous"]) -> str:
    return "auroc" if kind == "binary" else "r2"
