"""참조 기준/하한선 베이스라인 (스펙 0절 3층 구조).

전부 동일 인터페이스:
    ``fit_predict(urls, y, split, meta=None, seed=0) -> (p_val, p_test)``
동일 split 인덱스를 받아 sklearn으로 학습하고 val/test 확률을 돌려준다.

**fit/apply 분리 경로** (외부 검증 설계 7.2·9절 #12): 위 인터페이스는 한 프레임 안에서
fit과 predict를 모두 한다. F-a(zero-shot transfer)는 WebPhish train에서 fit한 벡터라이저·
계수를 **외부 데이터에 그대로 적용**해야 하므로 (다시 fit하면 그것은 F-a가 아니라 F-b다)
``fit_*(...) -> FittedBaseline`` / ``apply_*(model, ...) -> p`` 쌍을 따로 둔다.
기존 함수는 시그니처·수치 모두 그대로 두되, **내부에서 같은 fit/apply 경로를 호출**하도록
고쳐 두 경로가 조용히 갈라지는 일을 막았다(``tests/test_transfer.py``가 동치를 검증한다).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = [
    "BASELINES",
    "get_baseline",
    "charngram_lr",
    "version_lr",
    "length_lr",
    "bytehist_lr",
    "maskindex_lr",
    "charcnn",
    # fit/apply 분리 경로
    "FittedBaseline",
    "fit_charngram_lr",
    "apply_charngram_lr",
    "fit_bytehist_lr",
    "apply_bytehist_lr",
    "fit_length_lr",
    "apply_length_lr",
    "fit_version_lr",
    "apply_version_lr",
    "fit_motifhist_lr",
    "apply_motifhist_lr",
    "bytehist_features",
    "length_features",
]

TRAIN, VAL, TEST = 0, 1, 2


def _masks(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = np.asarray(split)
    return split == TRAIN, split == VAL, split == TEST


@dataclass
class FittedBaseline:
    """train에서 한 번 fit한 베이스라인. ``apply_*``가 이 객체만 보고 확률을 낸다.

    ``const``가 채워져 있으면 train에 한 클래스만 있어 학습이 불가능했던 경우이고,
    그때는 모든 행에 같은 상수 확률을 낸다(기존 ``_fit_lr``의 규약과 동일).
    """

    kind: str
    clf: Any = None
    const: float | None = None
    vectorizer: Any = None
    scaler: Any = None
    meta: dict = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        if self.const is not None:
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            return np.full(int(n), float(self.const), dtype=np.float64)
        assert self.clf is not None
        return np.asarray(self.clf.predict_proba(X)[:, 1], dtype=np.float64)


def _fit_core(kind: str, Xtr, ytr, seed: int, C: float = 1.0, **extra) -> FittedBaseline:
    """행렬 하나에서 LR을 fit한다. ``_fit_lr``과 **정확히 같은** 추정기 설정을 쓴다."""
    ytr = np.asarray(ytr)
    if len(np.unique(ytr)) < 2:
        const = float(np.mean(ytr)) if len(ytr) else 0.5
        return FittedBaseline(kind=kind, const=const, **extra)
    clf = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed
    )
    clf.fit(Xtr, ytr)
    return FittedBaseline(kind=kind, clf=clf, **extra)


def _fit_lr(Xtr, ytr, Xva, Xte, seed: int, C: float = 1.0):
    m = _fit_core("generic", Xtr, ytr, seed, C=C)
    return m.predict(Xva), m.predict(Xte)


def charngram_lr(urls, y, split, meta=None, seed: int = 0):
    """디코딩 텍스트 참조 기준: char_wb 1~5-gram TF-IDF + LR.

    디코딩된 URL을 직접 쓰는 **강한 텍스트 베이스라인**이지 Bayes 최적 분류기가 아니다.
    따라서 천장이 아니며, CNN이 이를 넘더라도 누출의 증거가 되지 않는다.
    """
    urls = np.asarray(urls, dtype=object)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    m = fit_charngram_lr(urls[tr], y[tr], seed)
    return apply_charngram_lr(m, urls[va]), apply_charngram_lr(m, urls[te])


def fit_charngram_lr(urls_tr, y_tr, seed: int = 0) -> FittedBaseline:
    """char_wb 1~5-gram TF-IDF 벡터라이저 + LR을 **train에서만** fit한다."""
    urls_tr = np.asarray(urls_tr, dtype=object)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(urls_tr.tolist())
    return _fit_core("charngram_lr", Xtr, y_tr, seed, vectorizer=vec)


def apply_charngram_lr(model: FittedBaseline, urls) -> np.ndarray:
    """fit된 벡터라이저로 변환만 하고 계수를 그대로 적용한다(재fit 금지)."""
    urls = np.asarray(urls, dtype=object)
    if model.const is not None:
        return np.full(urls.size, float(model.const), dtype=np.float64)
    return model.predict(model.vectorizer.transform(urls.tolist()))


def _from_meta(meta, col: str, urls) -> np.ndarray:
    if meta is not None and col in getattr(meta, "columns", []):
        return np.asarray(meta[col])
    raise KeyError(f"baseline requires meta column '{col}'")


def version_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: QR 버전 단독."""
    v = _from_meta(meta, "version", urls).astype(np.float64).reshape(-1, 1)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    m = fit_version_lr(v[tr], y[tr], seed)
    return apply_version_lr(m, v[va]), apply_version_lr(m, v[te])


def fit_version_lr(versions_tr, y_tr, seed: int = 0) -> FittedBaseline:
    """게이트용: QR 버전 단독. 층 안에서는 AUROC 0.5여야 한다."""
    X = np.asarray(versions_tr, dtype=np.float64).reshape(-1, 1)
    return _fit_core("version_lr", X, y_tr, seed)


def apply_version_lr(model: FittedBaseline, versions) -> np.ndarray:
    return model.predict(np.asarray(versions, dtype=np.float64).reshape(-1, 1))


def length_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: URL 길이 단독(길이 + 로그 길이)."""
    X = length_features(urls)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    m = _fit_core("length_lr", X[tr], y[tr], seed)
    return m.predict(X[va]), m.predict(X[te])


def length_features(urls) -> np.ndarray:
    """``(N, 2)`` — 바이트 길이와 log1p(바이트 길이)."""
    urls = np.asarray(urls, dtype=object)
    ln = np.asarray([len(u.encode("utf-8", "surrogatepass")) for u in urls], dtype=np.float64)
    return np.stack([ln, np.log1p(ln)], axis=1)


def fit_length_lr(urls_tr, y_tr, seed: int = 0) -> FittedBaseline:
    """게이트용: URL 길이 단독. ``L-exact``에서는 AUROC 0.5여야 한다."""
    return _fit_core("length_lr", length_features(urls_tr), y_tr, seed)


def apply_length_lr(model: FittedBaseline, urls) -> np.ndarray:
    return model.predict(length_features(urls))


def bytehist_lr(urls, y, split, meta=None, seed: int = 0):
    """하한선: 바이트 히스토그램(256차원, 길이 정규화). 순서 정보 없음."""
    X = bytehist_features(urls)
    y = np.asarray(y)
    tr, va, te = _masks(split)
    m = _fit_core("bytehist_lr", X[tr], y[tr], seed)
    return m.predict(X[va]), m.predict(X[te])


def bytehist_features(urls) -> np.ndarray:
    """``(N, 256)`` — 길이 정규화 바이트 히스토그램. 순서 정보 없음."""
    urls = np.asarray(urls, dtype=object)
    X = np.zeros((len(urls), 256), dtype=np.float64)
    for i, u in enumerate(urls):
        b = np.frombuffer(u.encode("utf-8", "surrogatepass"), dtype=np.uint8)
        if b.size:
            X[i] = np.bincount(b, minlength=256) / b.size
    return X


def fit_bytehist_lr(urls_tr, y_tr, seed: int = 0) -> FittedBaseline:
    return _fit_core("bytehist_lr", bytehist_features(urls_tr), y_tr, seed)


def apply_bytehist_lr(model: FittedBaseline, urls) -> np.ndarray:
    return model.predict(bytehist_features(urls))


def fit_motifhist_lr(H_tr, y_tr, seed: int = 0, C: float = 1.0) -> FittedBaseline:
    """motif 히스토그램 LR(RQ3 직접 검정의 전이판)을 train 히스토그램에서 fit한다.

    ``qrphish.motifs.bag_of_patches_lr``와 **같은** 표준화 + lbfgs LR을 쓴다. 다만 C는
    val로 고르지 않고 넘겨받는다(외부에는 fit용 val이 없다). ``C=1.0``이면
    ``bag_of_patches_lr(..., X_hist_val=None)``과 수치가 일치한다.
    """
    Xtr = np.asarray(H_tr, dtype=np.float64)
    ytr = np.asarray(y_tr).astype(np.int64).reshape(-1)
    if len(np.unique(ytr)) < 2:
        const = float(ytr.mean()) if ytr.size else 0.5
        return FittedBaseline(kind="motifhist_lr", const=const)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(
        C=C, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf.fit(scaler.transform(Xtr), ytr)
    return FittedBaseline(kind="motifhist_lr", clf=clf, scaler=scaler, meta={"C": float(C)})


def apply_motifhist_lr(model: FittedBaseline, H) -> np.ndarray:
    X = np.asarray(H, dtype=np.float64)
    if model.const is not None:
        return np.full(X.shape[0], float(model.const), dtype=np.float64)
    return model.predict(model.scaler.transform(X))


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
