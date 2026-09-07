"""참조 기준/하한선 베이스라인 (스펙 0절 3층 구조).

전부 동일 인터페이스:
    ``fit_predict(urls, y, split, meta=None, seed=0) -> (p_val, p_test)``
동일 split 인덱스를 받아 sklearn으로 학습하고 val/test 확률을 돌려준다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

__all__ = [
    "BASELINES",
    "get_baseline",
    "charngram_lr",
    "version_lr",
    "length_lr",
    "bytehist_lr",
    "maskindex_lr",
    "charcnn",
]

TRAIN, VAL, TEST = 0, 1, 2


def _masks(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = np.asarray(split)
    return split == TRAIN, split == VAL, split == TEST


def _fit_lr(Xtr, ytr, Xva, Xte, seed: int, C: float = 1.0):
    clf = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed
    )
    if len(np.unique(ytr)) < 2:
        # 한 클래스뿐이면 학습 불가 → 상수 확률
        const = float(np.mean(ytr)) if len(ytr) else 0.5
        return np.full(Xva.shape[0], const), np.full(Xte.shape[0], const)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]


def charngram_lr(urls, y, split, meta=None, seed: int = 0):
    """디코딩 텍스트 참조 기준: char_wb 1~5-gram TF-IDF + LR.

    디코딩된 URL을 직접 쓰는 **강한 텍스트 베이스라인**이지 Bayes 최적 분류기가 아니다.
    따라서 천장이 아니며, CNN이 이를 넘더라도 누출의 증거가 되지 않는다.
    """
    urls = np.asarray(urls, dtype=object)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(urls[tr].tolist())
    return _fit_lr(Xtr, y[tr], vec.transform(urls[va].tolist()), vec.transform(urls[te].tolist()), seed)


def _from_meta(meta, col: str, urls) -> np.ndarray:
    if meta is not None and col in getattr(meta, "columns", []):
        return np.asarray(meta[col])
    raise KeyError(f"baseline requires meta column '{col}'")


def version_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: QR 버전 단독."""
    v = _from_meta(meta, "version", urls).astype(np.float64).reshape(-1, 1)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    return _fit_lr(v[tr], y[tr], v[va], v[te], seed)


def length_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: URL 길이 단독(길이 + 로그 길이)."""
    urls = np.asarray(urls, dtype=object)
    ln = np.asarray([len(u.encode("utf-8", "surrogatepass")) for u in urls], dtype=np.float64)
    X = np.stack([ln, np.log1p(ln)], axis=1)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    return _fit_lr(X[tr], y[tr], X[va], X[te], seed)


def bytehist_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: 바이트 히스토그램(256차원, 길이 정규화). 순서 정보 없음."""
    urls = np.asarray(urls, dtype=object)
    X = np.zeros((len(urls), 256), dtype=np.float64)
    for i, u in enumerate(urls):
        b = np.frombuffer(u.encode("utf-8", "surrogatepass"), dtype=np.uint8)
        if b.size:
            X[i] = np.bincount(b, minlength=256) / b.size
    y = np.asarray(y)
    tr, va, te = _masks(split)
    return _fit_lr(X[tr], y[tr], X[va], X[te], seed)


def maskindex_lr(urls, y, split, meta=None, seed: int = 0):
    """``mask-auto`` 절제 전용: 선택된 마스크 인덱스(0~7) 원핫 단독 LR (스펙 1.6)."""
    mi = _from_meta(meta, "mask_used", urls).astype(int)
    X = np.zeros((len(mi), 8), dtype=np.float64)
    ok = (mi >= 0) & (mi < 8)
    X[np.nonzero(ok)[0], mi[ok]] = 1.0
    y = np.asarray(y)
    tr, va, te = _masks(split)
    return _fit_lr(X[tr], y[tr], X[va], X[te], seed)


def charcnn(urls, y, split, meta=None, seed: int = 0):
    """디코딩 텍스트 참조 기준(선택 사항): 문자 임베딩 CNN.

    스펙 13절 열린질문 4에 따라 1단계에서는 구현하지 않는다. char n-gram LR만으로
    참조 기준 논증이 성립하며, 필요해지면 이 함수만 채우면 된다.
    """
    raise NotImplementedError(
        "char-CNN 참조 기준은 선택 사항이라 1단계에서 구현하지 않는다 (스펙 13절 Q4)."
    )


BASELINES: dict[str, Callable] = {
    "charngram_lr": charngram_lr,
    "charcnn": charcnn,
    "version_lr": version_lr,
    "length_lr": length_lr,
    "bytehist_lr": bytehist_lr,
    "maskindex_lr": maskindex_lr,
}


def get_baseline(name: str) -> Callable:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name}")
    return BASELINES[name]
