"""인과 motif 절제 — 상위 motif를 뒤집으면 CNN 예측이 실제로 무너지는가.

review_01 "2차 실험 설계" C안이다. :mod:`qrphish.motifs`가 만든
``reports/motifs/{stratum}/enrichment.json``의 상위 phishing motif를 test QR에서
찾아 **창 중심 모듈을 뒤집고**, 같은 개수의 무작위 데이터 모듈 뒤집기 및 상위 benign
motif 뒤집기와 비교한다.

.. warning::
   뒤집은 격자는 **더 이상 유효한 QR이 아니다.** RS 오류정정 덕분에 실제 스캐너는
   여전히 디코딩할 수도 있지만, 이 절제는 "실제 세계의 QR 변형"이 아니라 모델
   입력에 대한 개입(intervention)이다. 결과는 "이 국소 패턴이 CNN 결정에 인과적으로
   기여하는가"로만 읽어야 하고 "공격자가 이렇게 QR을 바꿀 수 있다"로 읽으면 안 된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qrphish.evaluate import auroc as auroc_fn
from qrphish.evaluate import group_bootstrap

__all__ = [
    "load_enrichment",
    "window_centers",
    "window_ids",
    "match_centers",
    "random_centers",
    "flip_centers",
    "batch_logits",
    "occlude_stratum",
]


def load_enrichment(path: Path | str) -> dict:
    """``enrichment.json``을 읽는다. 없으면 명확한 에러(자동 생성하지 않는다)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"motif enrichment 파일이 없다: {p}\n"
            "먼저 run_motifs(cfg)로 3x3 motif enrichment를 만든 뒤 절제를 돌려라."
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    for key in ("top_phishing", "top_benign"):
        if key not in d:
            raise ValueError(f"{p}에 '{key}'가 없다 (motif 스키마 불일치)")
    return d


# ------------------------------------------------------------------ 창 열거·매칭
def window_centers(data_mask: np.ndarray, size: int = 3) -> np.ndarray:
    """``(M, 2)`` — 창 전체가 데이터 모듈인 창의 **중심** 좌표.

    기능 패턴(파인더·타이밍·포맷)에 걸친 창은 제외한다. 그 모듈은 URL과 무관하고
    뒤집으면 QR 골격 자체가 깨져 비교가 성립하지 않는다.
    """
    m = np.asarray(data_mask).astype(bool)
    n = m.shape[0]
    r = size // 2
    if n < size:
        return np.zeros((0, 2), dtype=np.int64)
    # 슬라이딩 창 전부 True인지: 적분영상으로 한 번에.
    cs = np.zeros((n + 1, n + 1), dtype=np.int64)
    cs[1:, 1:] = np.cumsum(np.cumsum(m.astype(np.int64), axis=0), axis=1)
    tot = (
        cs[size:, size:] - cs[:-size, size:] - cs[size:, :-size] + cs[:-size, :-size]
    )  # (n-size+1, n-size+1)
    rows, cols = np.nonzero(tot == size * size)
    return np.stack([rows + r, cols + r], axis=1).astype(np.int64)


def window_ids(values: np.ndarray, centers: np.ndarray, size: int = 3) -> np.ndarray:
    """각 창의 motif id(창 비트의 row-major MSB-first 정수)."""
    v = (np.asarray(values) > 0.5).astype(np.int64)
    r = size // 2
    out = np.zeros(len(centers), dtype=np.int64)
    weights = (1 << np.arange(size * size - 1, -1, -1)).astype(np.int64)
    for i, (cr, cc) in enumerate(np.asarray(centers, dtype=np.int64)):
        win = v[cr - r : cr + r + 1, cc - r : cc + r + 1].reshape(-1)
        out[i] = int((win * weights).sum())
    return out


def match_centers(
    values: np.ndarray, data_mask: np.ndarray, motif_ids, size: int = 3
) -> np.ndarray:
    """``motif_ids`` 집합과 일치하는 창들의 중심 좌표 ``(K, 2)``.

    매칭은 **원본 격자 한 벌에서 모두 계산한 뒤** 한꺼번에 뒤집는다. 하나 뒤집을
    때마다 이웃 창의 id가 바뀌므로 순차 처리하면 결과가 순서에 의존한다.
    """
    from qrphish.motifs import extract_patch_ids

    want = np.asarray(list(motif_ids), dtype=np.int64)
    # motif id 규약(row-major MSB-first, 창 전체가 데이터 모듈)을 enrichment를 만든
    # 쪽과 정확히 공유하려고 motifs의 추출기를 그대로 쓴다.
    ids, rows, cols = extract_patch_ids(
        np.asarray(values) > 0.5, np.asarray(data_mask) > 0.5, size, return_positions=True
    )
    if ids.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    sel = np.isin(ids.astype(np.int64), want)
    r = size // 2
    return np.stack([rows[sel] + r, cols[sel] + r], axis=1).astype(np.int64)


def random_centers(data_mask: np.ndarray, k: int, rng, size: int = 3) -> np.ndarray:
    """motif 매칭과 **같은 후보 풀**에서 뽑은 무작위 창 중심 ``k``개(비복원)."""
    centers = window_centers(data_mask, size)
    if centers.size == 0 or k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(int(k), len(centers))
    return centers[rng.choice(len(centers), size=k, replace=False)]


def flip_centers(x: torch.Tensor, centers: np.ndarray) -> torch.Tensor:
    """``(2,n,n)`` 입력의 **값 채널만** 지정 좌표에서 0↔1 뒤집는다. 마스크 채널 불변."""
    out = x.clone()
    c = np.asarray(centers, dtype=np.int64)
    if c.size == 0:
        return out
    rr = torch.as_tensor(c[:, 0], dtype=torch.long)
    cc = torch.as_tensor(c[:, 1], dtype=torch.long)
    out[0, rr, cc] = 1.0 - out[0, rr, cc]
    return out


def batch_logits(model, xs: list[torch.Tensor], batch_size: int = 256, device=None) -> np.ndarray:
    """``(N,)`` 로짓."""
    device = device or next(model.parameters()).device
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(xs), batch_size):
            b = torch.stack(xs[s : s + batch_size]).to(device)
            out.append(model(b).reshape(-1).cpu().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


# ----------------------------------------------------------------------- 본 실험
def _count_stats(counts: np.ndarray) -> dict:
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0:
        return {"mean": 0.0, "sd": 0.0, "median": 0.0, "max": 0, "frac_zero": 1.0}
    return {
        "mean": float(c.mean()),
        "sd": float(c.std(ddof=0)),
        "median": float(np.median(c)),
        "max": int(c.max()),
        "frac_zero": float((c == 0).mean()),
    }


def _paired_block(
    y: np.ndarray,
    groups: np.ndarray,
    base: np.ndarray,
    new: np.ndarray,
    counts: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """원본 대비 쌍체 비교 블록. AUROC 차와 평균 로짓 변화 모두 같은 그룹 리샘플로."""
    delta = new - base

    def d_auroc(idx: np.ndarray) -> float:
        return auroc_fn(y[idx], new[idx]) - auroc_fn(y[idx], base[idx])

    def d_logit(idx: np.ndarray) -> float:
        return float(delta[idx].mean())

    ci_a = group_bootstrap(groups, d_auroc, n_boot=n_boot, seed=seed)
    ci_l = group_bootstrap(groups, d_logit, n_boot=n_boot, seed=seed + 1)
    y = np.asarray(y)
    return {
        "auroc": float(auroc_fn(y, new)),
        "d_auroc": float(auroc_fn(y, new) - auroc_fn(y, base)),
        "d_auroc_ci": [ci_a["lo"], ci_a["hi"]],
        "mean_logit_delta": float(delta.mean()),
        "mean_logit_delta_ci": [ci_l["lo"], ci_l["hi"]],
        "mean_logit_delta_phishing": float(delta[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_logit_delta_benign": float(delta[y == 0].mean()) if (y == 0).any() else float("nan"),
        "n_flipped": _count_stats(counts),
    }


def occlude_stratum(
    model,
    dataset,
    test_rows: np.ndarray,
    groups: np.ndarray,
    enrichment: dict,
    *,
    top_k: int = 10,
    size: int = 3,
    n_random_rep: int = 5,
    seed: int = 0,
    n_boot: int = 500,
    device=None,
) -> dict:
    """한 (층, 시드)의 절제 결과.

    조건: ``phishing_motif`` / ``benign_motif`` / ``random`` / ``random_benign_matched``
    (무작위 조건은 각각 ``n_random_rep``회 평균).

    무작위 대조는 각 샘플에서 motif가 맞은 **개수와 정확히 같은 수**를 뒤집어 "뒤집은 모듈
    수" 자체의 효과를 상쇄한다. phishing motif와 benign motif는 매칭 개수가 서로 다르므로
    무작위 대조도 둘로 나눈다. ``random``은 phishing 매칭 수에, ``random_benign_matched``는
    benign 매칭 수에 맞춘다. 짝이 맞는 대조와만 비교해야 개수 효과가 실제로 상쇄된다.
    """
    rows = np.asarray(test_rows, dtype=np.int64)
    top_p = [int(v) for v in enrichment.get("top_phishing", [])[:top_k]]
    top_b = [int(v) for v in enrichment.get("top_benign", [])[:top_k]]

    xs = [dataset[int(i)][0] for i in rows]
    y = np.array([float(dataset[int(i)][1]) for i in rows], dtype=np.int64)

    x_phi: list[torch.Tensor] = []
    x_ben: list[torch.Tensor] = []
    x_rnd: list[list[torch.Tensor]] = [[] for _ in range(n_random_rep)]
    x_rnd_b: list[list[torch.Tensor]] = [[] for _ in range(n_random_rep)]
    n_phi, n_ben = [], []
    rngs = [np.random.default_rng(seed * 1000 + 7 * r) for r in range(n_random_rep)]
    rngs_b = [np.random.default_rng(seed * 1000 + 7 * r + 3) for r in range(n_random_rep)]
    for x in xs:
        val = x[0].numpy()
        dm = x[1].numpy() > 0.5
        cp = match_centers(val, dm, top_p, size) if top_p else np.zeros((0, 2), np.int64)
        cb = match_centers(val, dm, top_b, size) if top_b else np.zeros((0, 2), np.int64)
        x_phi.append(flip_centers(x, cp))
        x_ben.append(flip_centers(x, cb))
        n_phi.append(len(cp))
        n_ben.append(len(cb))
        for r in range(n_random_rep):
            x_rnd[r].append(flip_centers(x, random_centers(dm, len(cp), rngs[r], size)))
            x_rnd_b[r].append(flip_centers(x, random_centers(dm, len(cb), rngs_b[r], size)))

    base = batch_logits(model, xs, device=device)
    out: dict[str, Any] = {
        "n_test": int(rows.size),
        "auroc_original": float(auroc_fn(y, base)),
        "top_k": int(top_k),
        "size": int(size),
        "n_motifs_used": {"phishing": len(top_p), "benign": len(top_b)},
        "conditions": {},
    }
    npi = np.asarray(n_phi)
    out["conditions"]["phishing_motif"] = _paired_block(
        y, groups, base, batch_logits(model, x_phi, device=device), npi, n_boot, seed
    )
    out["conditions"]["benign_motif"] = _paired_block(
        y, groups, base, batch_logits(model, x_ben, device=device),
        np.asarray(n_ben), n_boot, seed + 100,
    )
    # 무작위 대조는 반복별 로짓을 평균해 한 벌로 만든 뒤 같은 쌍체 절차를 쓴다.
    rnd_logits = np.mean(
        [batch_logits(model, x_rnd[r], device=device) for r in range(n_random_rep)], axis=0
    )
    out["conditions"]["random"] = _paired_block(
        y, groups, base, rnd_logits, npi, n_boot, seed + 200
    )
    out["conditions"]["random"]["n_repeat"] = int(n_random_rep)
    rnd_b_logits = np.mean(
        [batch_logits(model, x_rnd_b[r], device=device) for r in range(n_random_rep)], axis=0
    )
    out["conditions"]["random_benign_matched"] = _paired_block(
        y, groups, base, rnd_b_logits, np.asarray(n_ben), n_boot, seed + 300
    )
    out["conditions"]["random_benign_matched"]["n_repeat"] = int(n_random_rep)
    return out
