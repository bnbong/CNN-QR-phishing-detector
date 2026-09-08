"""URL 경로 템플릿(피싱 키트 골격) 클러스터링 — 설계 5절(G, 템플릿 단위 분할).

eTLD+1 그룹 분할은 "같은 도메인이 train/test에 갈리는 것"만 막는다. 피싱 키트는
여러 도메인에 **같은 경로 구조**를 복제하므로 도메인이 달라도 test에 train과 사실상
같은 URL이 남는다. 이 모듈은 그 골격(template)을 뽑아 근사 중복끼리 묶고,
eTLD+1 그룹과 union-find로 합쳐 최종 그룹 키를 만든다.

핵심 주의 (설계 5.2의 "가장 흔한 실패 모드"):
``path_depth == 0``인 URL(맨 도메인)은 골격이 빈 문자열이 되어 전부 한 덩어리로
뭉친다. WebPhish benign의 절반이 여기 해당하므로 **반드시 클러스터링에서 제외**하고
각자 단독 클러스터로 둔다. 그러지 않으면 ``group_split``의 5% 상한 다운샘플이 폭발한다.

모든 함수는 시드가 같으면 바이트 단위로 동일한 결과를 낸다(외부 의존성 없음,
해시는 hashlib.blake2b로 고정). ``splits.py``는 건드리지 않는다 — 이 모듈은
``group`` 컬럼에 넣을 문자열만 만든다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from qrphish.urls import ensure_scheme

__all__ = [
    "url_template",
    "is_clusterable_template",
    "template_shingles",
    "minhash_signature",
    "minhash_signatures",
    "lsh_candidates",
    "template_groups",
    "combined_group_key",
    "template_diagnostics",
    "UnionFind",
]

_DIGIT_RUN = re.compile(r"[0-9]+")
_HEX_RUN = re.compile(r"[0-9a-f]{8,}")
_ALL_DIGITS = re.compile(r"^[0-9]+$")
_ALL_HEX = re.compile(r"^[0-9a-f]{8,}$")
_ALNUM = re.compile(r"^[A-Za-z0-9_-]+$")
_TOKEN_SEP = re.compile(r"[/?&.]")
# 확장자로 인정하는 꼬리(키트 식별에 유용하므로 보존한다).
_EXT_RE = re.compile(r"^[a-z0-9]{1,6}$")
# 치환 자리표시자. ``_mask_segment``는 base를 소문자로 낮춘 뒤에만 이 문자들을 넣으므로
# 골격 문자열에 이들이 보이면 반드시 치환의 결과다(원문 유래가 아니다).
_PLACEHOLDER_RE = re.compile(r"[#HR]")
# MinHash 순열용 소수 (2^31 - 1). 32bit 해시를 이 위에서 아핀 변환하면
# a*h + b가 2^62 미만이라 int64 안에서 완전히 벡터화된다(오버플로 없음).
_PRIME = (1 << 31) - 1
# 한 LSH 밴드 버킷이 이보다 크면 전체 쌍 대신 대표 원소와의 체인만 만든다.
_MAX_BUCKET_PAIRS = 512


# --------------------------------------------------------------------- union-find
class UnionFind:
    """경로 압축 + 랭크 병합. 병합 순서와 무관하게 같은 분할을 낸다."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 경로 압축
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> np.ndarray:
        """행 i의 루트 인덱스 배열."""
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


# ------------------------------------------------------------------ 템플릿 생성
def _mask_segment(seg: str) -> str:
    """경로 세그먼트 하나를 골격 토큰으로 치환한다 (설계 5.2)."""
    low = seg.lower()
    base, ext = low, ""
    if "." in low:
        head, _, tail = low.rpartition(".")
        if head and _EXT_RE.match(tail):
            base, ext = head, "." + tail

    if _ALL_DIGITS.match(base):
        return "#" + ext
    if _ALL_HEX.match(base):
        return "H" + ext
    # base64/난수스러운 문자열: 충분히 길고 (숫자 비율이 높거나 대소문자 혼합)
    if len(base) >= 12 and _ALNUM.match(base):
        digits = sum(ch.isdigit() for ch in base)
        mixed_case = any(c.islower() for c in seg) and any(c.isupper() for c in seg)
        if digits / len(base) >= 0.3 or mixed_case:
            return "R" + ext
    # 그 외에는 세그먼트 내부의 16진런·숫자런만 치환한다.
    masked = _HEX_RUN.sub("H", base)
    masked = _DIGIT_RUN.sub("#", masked)
    return masked + ext


def url_template(url: str) -> str:
    """URL의 **경로+쿼리 골격**. 호스트는 넣지 않는다.

    ``http://x.com/wp-admin/user/verify2.php?id=8a3f&ref=44``
    → ``/wp-admin/user/verify#.php?id&ref``

    경로 세그먼트가 하나도 없으면(``path_depth == 0``) 빈 문자열을 돌려준다.
    호출부는 이 경우를 **클러스터링에서 제외**해야 한다(모듈 docstring 참조).
    호스트는 이미 eTLD+1 그룹이 담당하므로 골격에 넣지 않는다.
    """
    # 선행 스킴만 인식한다. ``"://" in s``로 판정하면 쿼리 안에 URL을 품은 주소
    # (``good.com/r?url=https://evil.com``)의 호스트가 통째로 사라진다.
    s = ensure_scheme(url)
    try:
        parts = urlsplit(s)
    except ValueError:
        return ""
    segments = [seg for seg in parts.path.split("/") if seg]
    if not segments:
        return ""
    skeleton = "/" + "/".join(_mask_segment(seg) for seg in segments)
    if parts.query:
        keys = sorted({kv.split("=", 1)[0].lower() for kv in parts.query.split("&") if kv})
        if keys:
            skeleton += "?" + "&".join(keys)
    return skeleton


def is_clusterable_template(tpl: str, min_tokens: int = 3) -> bool:
    """이 골격을 클러스터링 대상으로 볼지 여부 (설계 5.2의 과병합 게이트).

    ``/index.html``, ``/index.htm``처럼 **범용 웹 골격**은 캠페인 신호가 아니라 웹 전반의
    기본 파일명이다. 실측(v2)에서 ``/index.html`` 하나가 무관한 도메인 276개를 한 클러스터로
    묶었다. 반면 ``tools.ietf.org/html/rfc###`` → ``/html/rfc#``은 토큰이 둘뿐이어도
    치환 자리표시자를 갖는 **구조적** 골격이라 정당한 클러스터다.

    그래서 다음 중 하나라도 만족할 때만 클러스터링 대상이다.

    1. 토큰이 ``min_tokens``개 이상 (구조가 충분히 구체적이다)
    2. 치환 자리표시자(``#``/``H``/``R``)를 포함 (숫자·해시·난수 자리 = 키트 골격의 지문)
    3. 쿼리 키를 포함 (``?id&ref`` 같은 파라미터 구조)

    나머지는 호출부에서 **각자 단독 클러스터**(``solo:``)로 돌린다. 빈 골격
    (``path_depth == 0``)도 여기서 False다.
    """
    if min_tokens < 1:
        raise ValueError(f"min_tokens는 1 이상이어야 한다 (got {min_tokens})")
    if not tpl:
        return False
    if "?" in tpl:  # 쿼리 키 보유
        return True
    if _PLACEHOLDER_RE.search(tpl):
        return True
    tokens = [t for t in _TOKEN_SEP.split(tpl) if t]
    return len(tokens) >= min_tokens


def template_shingles(tpl: str, k: int = 3) -> set[str]:
    """골격 문자열의 토큰 k-shingle 집합. 토큰 구분자는 ``/ ? & .``.

    토큰 수가 k보다 적으면 전체를 단일 shingle로 쓴다(짧은 골격이 사라지지 않게).
    """
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 한다 (got {k})")
    tokens = [t for t in _TOKEN_SEP.split(tpl) if t]
    if not tokens:
        return set()
    if len(tokens) < k:
        return {"\x00".join(tokens)}
    return {"\x00".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _hash32(text: str) -> int:
    """32bit 해시를 소수로 접은 값. shingle 하나당 한 번만 계산한다."""
    return (
        int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "big")
        % _PRIME
    )


def _perm_params(n_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=n_perm).astype(np.int64)
    b = rng.integers(0, _PRIME, size=n_perm).astype(np.int64)
    return a, b


def _signature_from_hashes(h: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return np.full(a.size, _PRIME, dtype=np.int64)
    return ((h[:, None] * a[None, :] + b[None, :]) % _PRIME).min(axis=0)


def minhash_signature(shingles: Iterable[str], n_perm: int = 64, seed: int = 0) -> np.ndarray:
    """순수 numpy MinHash 서명 (외부 datasketch 의존 없음).

    빈 shingle 집합은 소수 값으로 채워 서로 같은 서명을 갖는다(전부 빈 골격이므로
    클러스터링에서 제외되는 경로와 일관된다).
    """
    if n_perm < 1:
        raise ValueError(f"n_perm은 1 이상이어야 한다 (got {n_perm})")
    a, b = _perm_params(n_perm, seed)
    h = np.array([_hash32(s) for s in shingles], dtype=np.int64)
    return _signature_from_hashes(h, a, b)


def minhash_signatures(
    shingle_sets: Sequence[Iterable[str]], n_perm: int = 64, seed: int = 0
) -> np.ndarray:
    """여러 집합의 서명을 한 번에. 반환 shape ``(len(shingle_sets), n_perm)``.

    shingle 문자열 해시를 캐시해 중복 계산을 없앤다.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm은 1 이상이어야 한다 (got {n_perm})")
    a, b = _perm_params(n_perm, seed)
    out = np.empty((len(shingle_sets), n_perm), dtype=np.int64)
    cache: dict[str, int] = {}
    for i, sh in enumerate(shingle_sets):
        hs = []
        for s in sh:
            v = cache.get(s)
            if v is None:
                v = cache[s] = _hash32(s)
            hs.append(v)
        out[i] = _signature_from_hashes(np.array(hs, dtype=np.int64), a, b)
    return out


def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """밴딩 LSH 후보쌍. ``bands * rows``는 서명 길이 이하여야 한다."""
    sig = np.asarray(signatures)
    if sig.ndim != 2:
        raise ValueError(f"signatures는 2차원이어야 한다 (got shape {sig.shape})")
    if bands < 1 or rows < 1:
        raise ValueError(f"bands/rows는 1 이상이어야 한다 (got {bands}, {rows})")
    if bands * rows > sig.shape[1]:
        raise ValueError(f"bands*rows({bands * rows})가 서명 길이({sig.shape[1]})를 넘는다")

    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        chunk = sig[:, band * rows : (band + 1) * rows]
        buckets: dict[bytes, list[int]] = {}
        for i, row in enumerate(chunk):
            buckets.setdefault(row.tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > _MAX_BUCKET_PAIRS:
                # 거대 버킷은 전체 쌍(O(n^2)) 대신 대표 원소와의 체인만 만든다.
                # 연결 요소 관점에서는 동일한 성분을 낳는다.
                head = members[0]
                pairs.update((head, j) for j in members[1:])
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


def _estimated_jaccard(sig: np.ndarray, i: int, j: int) -> float:
    return float(np.mean(sig[i] == sig[j]))


def _cluster_id(rep: str) -> str:
    return "t:" + hashlib.blake2b(rep.encode("utf-8"), digest_size=6).hexdigest()


def template_groups(
    urls: Sequence[str],
    threshold: float = 0.7,
    k: int = 3,
    *,
    n_perm: int = 64,
    rows: int = 4,
    seed: int = 0,
    min_template_tokens: int = 3,
) -> np.ndarray:
    """각 URL의 템플릿 클러스터 id(문자열) 배열.

    - 경로가 없는 URL(``url_template`` 이 ``""``)은 **자기 자신만의 고유 id**를 받는다.
    - :func:`is_clusterable_template` 게이트를 통과하지 못한 범용 골격(``/index.html`` 등)도
      마찬가지로 단독 id를 받는다. ``min_template_tokens``로 조절한다.
    - 골격이 정확히 같은 URL은 MinHash 이전에 이미 같은 노드다(정확 일치 지름길).
    - 근사 중복은 LSH 후보쌍 중 추정 Jaccard ≥ ``threshold``인 쌍만 union한다.

    클러스터 id는 성분 내 사전순 최소 골격에서 유도하므로 입력 순서와 무관하게
    결정적이다.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold는 (0, 1] 범위여야 한다 (got {threshold})")
    n = len(urls)
    out = np.empty(n, dtype=object)
    if n == 0:
        return out.astype(str)

    templates = [url_template(u) for u in urls]
    # 과병합 게이트: 범용 골격은 아예 노드를 만들지 않고 단독으로 남긴다.
    eligible = {
        tpl: is_clusterable_template(tpl, min_template_tokens) for tpl in set(templates)
    }
    # 정확 일치 지름길: 골격 문자열 단위로 노드를 만든다.
    uniq: dict[str, int] = {}
    for tpl in templates:
        if eligible[tpl] and tpl not in uniq:
            uniq[tpl] = len(uniq)
    keys = sorted(uniq, key=lambda t: uniq[t])  # 첫 등장 순서 = 안정적
    m = len(keys)

    if m:
        sig = minhash_signatures([template_shingles(t, k) for t in keys], n_perm=n_perm, seed=seed)
        bands = max(1, n_perm // max(1, rows))
        uf = UnionFind(m)
        for i, j in sorted(lsh_candidates(sig, bands, rows)):
            if _estimated_jaccard(sig, i, j) >= threshold:
                uf.union(i, j)
        # 성분 대표 = 사전순 최소 골격 -> 결정적 id
        reps: dict[int, str] = {}
        for idx, tpl in enumerate(keys):
            root = uf.find(idx)
            if root not in reps or tpl < reps[root]:
                reps[root] = tpl
        node_id = [_cluster_id(reps[uf.find(idx)]) for idx in range(m)]
    else:
        node_id = []

    for i, tpl in enumerate(templates):
        out[i] = node_id[uniq[tpl]] if eligible[tpl] else f"solo:{i}"
    return out.astype(str)


def combined_group_key(
    etld1_groups: Sequence[str], template_groups_arr: Sequence[str]
) -> np.ndarray:
    """eTLD+1 그룹과 템플릿 클러스터를 union-find로 합친 최종 그룹 키 (설계 5.4).

    같은 eTLD+1이거나 같은 템플릿 클러스터면 같은 그룹이다. 성분 id는 성분에 속한
    eTLD+1 값 중 사전순 최소값으로 정하므로 입력 순서와 무관하게 결정적이다.
    """
    e = list(etld1_groups)
    t = list(template_groups_arr)
    if len(e) != len(t):
        raise ValueError(f"길이가 다르다: etld1={len(e)}, template={len(t)}")
    n = len(e)
    uf = UnionFind(n)
    first: dict[str, int] = {}
    for i, key in enumerate(e):
        anchor = first.setdefault("e:" + str(key), i)
        uf.union(anchor, i)
    for i, key in enumerate(t):
        anchor = first.setdefault("t:" + str(key), i)
        uf.union(anchor, i)

    reps: dict[int, str] = {}
    for i in range(n):
        root = uf.find(i)
        val = str(e[i])
        if root not in reps or val < reps[root]:
            reps[root] = val
    return np.array([f"g:{reps[uf.find(i)]}" for i in range(n)], dtype=object).astype(str)


# ------------------------------------------------------------------------ 진단
def template_diagnostics(
    groups: Sequence[str], labels: Sequence[int] | None = None
) -> dict[str, Any]:
    """클러스터 크기 분포, 최대 클러스터 점유율, 클래스별 클러스터 수."""
    arr = np.asarray(list(groups), dtype=object)
    n = int(arr.size)
    if n == 0:
        return {"n_rows": 0, "n_clusters": 0, "largest_cluster_frac": 0.0}
    uniq, counts = np.unique(arr.astype(str), return_counts=True)
    quant = {
        "p50": int(np.percentile(counts, 50)),
        "p90": int(np.percentile(counts, 90)),
        "p99": int(np.percentile(counts, 99)),
        "max": int(counts.max()),
    }
    out: dict[str, Any] = {
        "n_rows": n,
        "n_clusters": int(uniq.size),
        "n_singletons": int((counts == 1).sum()),
        "largest_cluster_frac": float(counts.max() / n),
        "cluster_size_quantiles": quant,
        "top5_clusters": {
            str(uniq[i]): int(counts[i]) for i in np.argsort(-counts, kind="stable")[:5]
        },
    }
    if labels is not None:
        lab = np.asarray(list(labels), dtype=np.int64)
        if lab.size != n:
            raise ValueError(f"labels 길이가 다르다: {lab.size} != {n}")
        out["n_clusters_by_class"] = {
            "benign": int(np.unique(arr[lab == 0].astype(str)).size),
            "phishing": int(np.unique(arr[lab == 1].astype(str)).size),
        }
        _, inv = np.unique(arr.astype(str), return_inverse=True)
        pos = np.bincount(inv, weights=(lab == 1), minlength=uniq.size)
        out["n_mixed_class_clusters"] = int(((pos > 0) & (pos < counts)).sum())
    return out
