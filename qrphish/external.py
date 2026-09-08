"""외부 검증(F)용 URL 수집·정규화·로딩 (설계 스펙 1~3절, 7.1절).

수집 소스는 2026-09-08 실측 결과에 맞춰 확정했다.

- phishing **주** 소스(primary): OpenPhish public_feed의 최근 90일 커밋 이력을 누적한 것
  (고유 URL 약 4.4만). 수집 시점과 시간 축이 맞아 진짜 temporal/domain shift 검증이 된다.
  외부 test는 크기보다 독립성·신선도·출처 명확성이 중요하다(리뷰 02).
- phishing 보조 소스(secondary robustness): Phishing.Database ``phishing-links-ACTIVE.txt``
  (MIT, 약 79만 줄). ACTIVE 파일의 최신 커밋이 2025-12-22이므로 "살아있는 피싱"이 아니라
  **아카이브**로 취급한다.
- benign 주 소스: Common Crawl ``CC-MAIN-2026-34`` columnar index parquet의 row group 부분 읽기 결과를
  Tranco(list ``GQJ9K``) 상위 도메인과 eTLD+1 조인한 것. primary와 secondary는 **같은 benign 행**을
  쓴다(정제를 합집합에서 한 번만 한다 — :func:`split_by_source`).
- benign 민감도 세트: 같은 CC row group에서 Tranco 조인을 **빼고** 뽑은 표본. "benign = 인기
  웹사이트"라는 축(popularity confound)이 결과를 만드는지 본다.
- benign 대조군(EXT-B2): Tranco 맨 도메인(경로 없음).

PhishTank(Cloudflare 403)와 URLhaus(malware 전용)는 제외했다.

네트워크를 쓰는 함수는 ``fetch_*`` 접두사로 분리해 두었다. 파싱·정규화·중복 제거·진단은
모두 순수 함수이므로 네트워크 없이 테스트할 수 있다.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from qrphish.urls import (
    default_extractor,
    etld1,
    has_leading_scheme,
    normalize_url,
    path_depth,
    strip_leading_scheme,
)

__all__ = [
    "COLLECT_DATE",
    "USER_AGENT",
    "HOSTING_BLOCKLIST",
    "BENIGN_CLEANING_MODES",
    "SourceSpec",
    "SOURCES",
    "strip_scheme",
    "normalize_external_url",
    "parse_source",
    "fetch_phishdb",
    "fetch_openphish_history",
    "fetch_tranco",
    "fetch_commoncrawl",
    "build_external_dataset",
    "load_external",
    "split_by_source",
    "dedup_against",
    "bias_diagnostics",
    "sha256_file",
]

COLLECT_DATE = "2026-09-08"
USER_AGENT = "qrphish-research/1.0 (bbbong9@gmail.com)"

CC_CRAWL_ID = "CC-MAIN-2026-34"
TRANCO_LIST_ID = "GQJ9K"

#: 무료 동적 호스팅 / 단축 도메인. benign 쪽에서 제거한다 (설계 1.4절 2).
HOSTING_BLOCKLIST: frozenset[str] = frozenset(
    {
        "000webhostapp.com",
        "weebly.com",
        "blogspot.com",
        "duckdns.org",
        "bit.ly",
        "t.co",
        "github.io",
        "firebaseapp.com",
        "web.app",
        "netlify.app",
        "pages.dev",
        "r2.dev",
        "glitch.me",
    }
)

# --------------------------------------------------------------------------------------
# 소스 레지스트리
# --------------------------------------------------------------------------------------


#: benign 정제 절제 3조건 (리뷰 02 — "benign cleaning sensitivity"는 권장이 아니라 필수).
#:
#: - ``clean``: 기본. phishing eTLD+1 위의 benign 제거 + 공유 호스팅/단축기 블록리스트 제거.
#: - ``keep_phish_domains``: phishing 도메인 위의 benign을 남긴다.
#: - ``no_hosting_blocklist``: 공유 호스팅/단축기 블록리스트를 적용하지 않는다.
BENIGN_CLEANING_MODES: tuple[str, ...] = (
    "clean",
    "keep_phish_domains",
    "no_hosting_blocklist",
)


@dataclass(frozen=True)
class SourceSpec:
    """외부 소스 하나의 접근 규약."""

    name: str
    label: int  # 1=phishing, 0=benign
    kind: Literal["text_lines", "text_lines_dated", "parquet_join", "csv_rank"]
    url_template: str
    env_key: str | None
    license_note: str


SOURCES: dict[str, SourceSpec] = {
    "phishdb": SourceSpec(
        name="phishdb",
        label=1,
        kind="text_lines",
        url_template=(
            "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
            "master/phishing-links-ACTIVE.txt"
        ),
        env_key=None,
        license_note="MIT (Phishing.Database). ACTIVE 파일 최신 커밋 2025-12-22 — 아카이브 성격",
    ),
    "openphish": SourceSpec(
        name="openphish",
        label=1,
        kind="text_lines_dated",
        url_template="https://raw.githubusercontent.com/openphish/public_feed/{ref}/feed.txt",
        env_key=None,
        license_note="OpenPhish community feed, openphish.com/terms.html (비상업 연구, 재배포 금지)",
    ),
    "commoncrawl": SourceSpec(
        name="commoncrawl",
        label=0,
        kind="parquet_join",
        url_template=(
            "https://data.commoncrawl.org/crawl-data/{crawl}/cc-index-table.paths.gz"
        ),
        env_key=None,
        license_note=f"Common Crawl terms-of-use, crawl {CC_CRAWL_ID}",
    ),
    "commoncrawl_unranked": SourceSpec(
        name="commoncrawl_unranked",
        label=0,
        kind="parquet_join",
        url_template=(
            "https://data.commoncrawl.org/crawl-data/{crawl}/cc-index-table.paths.gz"
        ),
        env_key=None,
        license_note=(
            f"Common Crawl terms-of-use, crawl {CC_CRAWL_ID} (Tranco 조인 없음 — 민감도 세트)"
        ),
    ),
    "tranco": SourceSpec(
        name="tranco",
        label=0,
        kind="csv_rank",
        url_template="https://tranco-list.eu/download/{list_id}/1000000",
        env_key=None,
        license_note=f"Tranco list {TRANCO_LIST_ID} (연구용 공개, 구성 소스 라이선스 상속)",
    ),
}


# --------------------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------------------


def strip_scheme(url: str) -> str:
    """`http://`, `https://`, `ftp://` 등 선행 스킴을 1회 제거한다.

    WebPhish의 `Data` 컬럼에는 스킴이 없으므로(`head -3 data/webphish.csv` 확인) 외부 URL도
    스킴을 제거해 같은 규약으로 맞춘다. 스킴 유무는 URL 바이트 길이를 7~8바이트 바꾸므로
    QR 버전 배정과 길이 매칭을 직접 흔든다.

    **선행** 스킴만 본다. 쿼리 안에 다른 URL을 품은 주소
    (``good.com/redirect?url=https://evil.com``)는 스킴이 없는 것으로 취급한다.
    """
    return strip_leading_scheme(url)


def normalize_external_url(url: str, mode: Literal["raw", "norm"] = "norm") -> str:
    """스킴 제거 후 WebPhish와 **동일한** `normalize_url`을 적용한다."""
    return normalize_url(strip_scheme(url), mode)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# 파싱 (네트워크 없음 — 테스트 대상)
# --------------------------------------------------------------------------------------


def _iter_lines(raw_path: Path) -> Iterable[str]:
    opener = gzip.open if str(raw_path).endswith(".gz") else open
    with opener(raw_path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[operator]
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s


def parse_source(spec: SourceSpec, raw_path: str | Path) -> pd.DataFrame:
    """원본 파일 -> DataFrame(url_src, label, source, first_seen). 네트워크 접근 없음."""
    path = Path(raw_path)
    if spec.kind == "text_lines":
        urls = list(_iter_lines(path))
        first_seen: list[str] = [""] * len(urls)
    elif spec.kind == "text_lines_dated":
        # "<ISO date>\t<url>" 형식 (fetch_openphish_history가 쓴다).
        urls, first_seen = [], []
        for line in _iter_lines(path):
            if "\t" in line:
                d, u = line.split("\t", 1)
            else:
                d, u = "", line
            urls.append(u)
            first_seen.append(d)
    elif spec.kind in ("parquet_join", "csv_rank"):
        # fetch_* 단계에서 이미 "url" 컬럼을 가진 CSV로 떨궈 둔다.
        df_raw = pd.read_csv(path, dtype=str)
        col = "url" if "url" in df_raw.columns else df_raw.columns[0]
        urls = [str(u) for u in df_raw[col].tolist()]
        first_seen = [""] * len(urls)
    else:  # pragma: no cover - 열거값 방어
        raise ValueError(f"알 수 없는 kind: {spec.kind}")

    return pd.DataFrame(
        {
            "url_src": pd.Series(urls, dtype="string"),
            "label": pd.Series([spec.label] * len(urls), dtype="int64"),
            "source": pd.Series([spec.name] * len(urls), dtype="string"),
            "first_seen": pd.Series(first_seen, dtype="string"),
        }
    )


# --------------------------------------------------------------------------------------
# 네트워크 (fetch_*) — 테스트에서 monkeypatch 대상
# --------------------------------------------------------------------------------------


def _session() -> Any:
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(session: Any, url: str, *, retries: int = 4, timeout: int = 120, **kw: Any) -> Any:
    """재시도 포함 GET. 503/429는 지수 백오프."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, **kw)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code} for {url}")
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - 재시도 후 재던진다
            last = exc
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"GET 실패: {url}") from last


def fetch_phishdb(raw_dir: str | Path, *, session: Any = None, date: str = COLLECT_DATE) -> dict:
    """Phishing.Database ACTIVE 링크 목록을 원본 그대로 저장한다."""
    sess = session if session is not None else _session()
    spec = SOURCES["phishdb"]
    out = Path(raw_dir) / f"phishdb_active_{date}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    resp = _get(sess, spec.url_template)
    out.write_bytes(resp.content)

    pinned = ""
    committed_at = ""
    try:
        api = _get(
            sess,
            "https://api.github.com/repos/mitchellkrogza/Phishing.Database/commits"
            "?path=phishing-links-ACTIVE.txt&per_page=1",
            retries=2,
            timeout=30,
        )
        commits = api.json()
        if commits:
            pinned = str(commits[0]["sha"])
            committed_at = str(commits[0]["commit"]["committer"]["date"])
    except Exception:  # noqa: BLE001 - 핀 정보는 있으면 좋고 없으면 경고
        pinned = ""

    n_raw = sum(1 for _ in _iter_lines(out))
    return {
        "name": spec.name,
        "label": spec.label,
        "url": spec.url_template,
        "raw_path": str(out),
        "pinned_id": pinned,
        "pinned_committed_at": committed_at,
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(out),
        "n_raw": n_raw,
        "license_note": spec.license_note,
        "freshness_note": (
            "ACTIVE 파일의 마지막 갱신 커밋은 2025-12-22이다. 현재 살아있는 피싱이 아니라 "
            "역사적 피싱 URL 아카이브로 해석해야 한다."
        ),
    }


def fetch_openphish_history(
    raw_dir: str | Path,
    *,
    days: int = 90,
    max_commits: int = 200,
    session: Any = None,
    date: str = COLLECT_DATE,
    sleep: float = 0.15,
) -> dict:
    """openphish/public_feed의 최근 커밋 이력에서 feed.txt를 누적한다.

    `https://openphish.com/feed.txt`는 이 리포의 raw로 302 리다이렉트되고, 스냅샷 1회가 300건뿐이라
    누적하지 않으면 규모가 나오지 않는다. 리포를 clone하지 않고 GitHub API 커밋 목록 + raw 파일
    조회만 쓴다(커밋 목록만 API 쿼터를 소모하고, raw는 CDN이라 제한이 느슨하다).

    출력 형식: `<커밋 ISO 날짜>\\t<url>` 줄. 같은 URL은 **가장 이른** 커밋 날짜를 first_seen으로 남긴다.
    """
    sess = session if session is not None else _session()
    spec = SOURCES["openphish"]
    out = Path(raw_dir) / f"openphish_history_{date}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    commits: list[tuple[str, str]] = []
    page = 1
    while len(commits) < max_commits:
        api = _get(
            sess,
            "https://api.github.com/repos/openphish/public_feed/commits"
            f"?path=feed.txt&since={since}&per_page=100&page={page}",
            retries=3,
            timeout=60,
        )
        batch = api.json()
        if not isinstance(batch, list) or not batch:
            break
        for c in batch:
            commits.append((str(c["sha"]), str(c["commit"]["committer"]["date"])))
        if len(batch) < 100:
            break
        page += 1
    commits = commits[:max_commits]
    # 오래된 커밋부터 훑어야 first_seen이 "가장 이른 등장"이 된다.
    commits.reverse()

    first_seen: dict[str, str] = {}
    n_fetched = 0
    for sha, when in commits:
        try:
            resp = _get(
                sess, spec.url_template.format(ref=sha), retries=2, timeout=60
            )
        except Exception:  # noqa: BLE001 - 개별 커밋 실패는 건너뛴다
            continue
        n_fetched += 1
        for line in resp.text.splitlines():
            u = line.strip()
            if u and u not in first_seen:
                first_seen[u] = when
        time.sleep(sleep)

    with open(out, "w", encoding="utf-8") as fh:
        for u, when in first_seen.items():
            fh.write(f"{when}\t{u}\n")

    return {
        "name": spec.name,
        "label": spec.label,
        "url": spec.url_template.format(ref="<commit>"),
        "raw_path": str(out),
        "pinned_id": ";".join(sha for sha, _ in commits[-5:]),
        "n_commits_listed": len(commits),
        "n_commits_fetched": n_fetched,
        "window_days": days,
        "since": since,
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(out),
        "n_raw": len(first_seen),
        "license_note": spec.license_note,
        "method_note": (
            "shallow clone 대신 GitHub API 커밋 목록 + 커밋별 raw feed.txt 누적. "
            "리포 전체를 내려받지 않는다."
        ),
    }


def fetch_tranco(
    raw_dir: str | Path,
    *,
    list_id: str = TRANCO_LIST_ID,
    session: Any = None,
    date: str = COLLECT_DATE,
    n_domains: int = 50_000,
) -> dict:
    """Tranco 영구 list id로 top-1m을 받아 저장하고, 상위 `n_domains`개를 EXT-B2로 떨군다."""
    sess = session if session is not None else _session()
    spec = SOURCES["tranco"]
    raw = Path(raw_dir) / f"tranco_{date}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)

    resp = _get(sess, spec.url_template.format(list_id=list_id))
    body = resp.content
    if body[:2] == b"PK":  # zip
        import zipfile

        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            body = zf.read(zf.namelist()[0])
    raw.write_bytes(body)

    df = pd.read_csv(raw, names=["rank", "domain"], dtype=str)
    top = df.head(n_domains)["domain"].tolist()
    # 주의: build_external_dataset의 출력(`tranco_{date}.csv`)과 이름이 겹치면
    # 두 번째 실행에서 조립된 CSV를 원본으로 잘못 다시 파싱한다. 접미사로 분리한다.
    ext_b2 = Path(raw_dir).parent / f"tranco_urls_{date}.csv"
    pd.DataFrame({"url": top}).to_csv(ext_b2, index=False)

    return {
        "name": spec.name,
        "label": spec.label,
        "url": spec.url_template.format(list_id=list_id),
        "raw_path": str(raw),
        "parsed_path": str(ext_b2),
        "pinned_id": list_id,
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(raw),
        "n_raw": int(len(df)),
        "n_top": len(top),
        "license_note": spec.license_note,
    }


class _HttpRangeFile(io.RawIOBase):
    """HTTP Range 요청으로 원격 파일을 seek 가능한 바이너리 스트림처럼 노출한다.

    fsspec의 http 백엔드는 aiohttp를 요구하는데 이 환경에는 없다. parquet row group 부분 읽기에
    필요한 것은 `read`/`seek`/`tell`뿐이라 requests로 직접 구현한다. UA 헤더가 없으면
    data.commoncrawl.org가 503을 준다.
    """

    def __init__(self, session: Any, url: str) -> None:
        self._sess = session
        self._url = url
        head = _get(session, url, retries=3, timeout=60, stream=True)
        self._size = int(head.headers["Content-Length"])
        head.close()
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    @property
    def size(self) -> int:
        return self._size

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        resp = _get(
            self._sess,
            self._url,
            retries=4,
            timeout=180,
            headers={"Range": f"bytes={self._pos}-{end}"},
        )
        data = resp.content
        self._pos += len(data)
        return data

    def readall(self) -> bytes:
        return self.read(-1)


def fetch_commoncrawl(
    raw_dir: str | Path,
    *,
    n_row_groups: int = 2,
    session: Any = None,
    date: str = COLLECT_DATE,
    crawl: str = CC_CRAWL_ID,
    tranco_csv: str | Path | None = None,
    per_domain_cap: int = 5,
    max_row_groups_per_part: int = 1,
    seed: int = 0,
    unranked_sample: int | None = None,
) -> dict:
    """CC columnar index parquet의 row group을 부분 읽기해 benign 후보를 만든다.

    S3 버킷 리스팅은 403이므로 `cc-index-table.paths.gz`로 part 경로를 얻는다.
    `fetch_status==200 & mime==text/html` 필터 후 `url_host_registered_domain`을 Tranco 상위
    도메인과 조인해 남긴다. 도메인당 `per_domain_cap`개로 상한을 건다.

    ``unranked_sample``을 주면 **Tranco 조인을 거치지 않은** benign 표본을 같은 row group에서
    함께 떨군다(리뷰 02 — popularity confound 민감도 세트). 도메인당 상한은 조인본과 같다.
    조인을 빼면 WebPhish·피싱 피드와 겹치는 도메인이 benign에 섞일 위험이 커지므로 이 세트는
    주 결과가 아니라 민감도 세트로만 쓴다.

    parquet 파티션은 호스트(SURT) 순으로 정렬되어 있어 **한 part의 row group 하나는 도메인 몇백 개만
    담는다.** 도메인 다양성을 확보하려면 row group을 한 part에 몰지 말고 여러 part에 흩어야 한다
    (`max_row_groups_per_part`). 이 값을 키우면 전송량은 같지만 도메인 수가 급감한다.
    """
    import pyarrow.parquet as pq

    sess = session if session is not None else _session()
    spec = SOURCES["commoncrawl"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths_gz = raw_dir / f"cc_index_paths_{date}.gz"
    resp = _get(sess, spec.url_template.format(crawl=crawl))
    paths_gz.write_bytes(resp.content)
    all_paths = [
        p
        for p in gzip.decompress(paths_gz.read_bytes()).decode().splitlines()
        if "subset=warc" in p
    ]
    if not all_paths:
        raise RuntimeError("cc-index paths.gz에서 subset=warc part를 찾지 못했다")

    tranco_domains: set[str] | None = None
    if tranco_csv is not None:
        tdf = pd.read_csv(tranco_csv, names=["rank", "domain"], dtype=str)
        tranco_domains = set(tdf["domain"].dropna().str.strip().str.lower())

    cols = [
        "url",
        "url_host_registered_domain",
        "fetch_status",
        "content_mime_detected",
    ]
    frames: list[pd.DataFrame] = []
    used: list[dict] = []
    rng = random.Random(seed)
    part_order = list(all_paths)
    rng.shuffle(part_order)

    taken = 0
    for part in part_order:
        if taken >= n_row_groups:
            break
        url = f"https://data.commoncrawl.org/{part}"
        try:
            fh = _HttpRangeFile(sess, url)
            pf = pq.ParquetFile(fh)
            n_here = min(pf.num_row_groups, max_row_groups_per_part, n_row_groups - taken)
            for rg in range(n_here):
                tbl = pf.read_row_group(rg, columns=cols)
                df = tbl.to_pandas()
                df = df[
                    (df["fetch_status"] == 200)
                    & (df["content_mime_detected"] == "text/html")
                ]
                frames.append(df[["url", "url_host_registered_domain"]])
                used.append({"part": part, "row_group": rg, "n_rows": int(len(df))})
                taken += 1
                if taken >= n_row_groups:
                    break
            fh.close()
        except Exception as exc:  # noqa: BLE001 - part 하나 실패는 다음 part로 넘어간다
            used.append({"part": part, "error": str(exc)})
            continue

    if not frames:
        raise RuntimeError("Common Crawl row group을 하나도 읽지 못했다")

    cc = pd.concat(frames, ignore_index=True)
    n_cc_raw = int(len(cc))
    cc["domain"] = cc["url_host_registered_domain"].astype("string").str.lower()
    cc_all = cc
    if tranco_domains is not None:
        cc = cc[cc["domain"].isin(tranco_domains)]
    n_after_join = int(len(cc))

    unranked_path: str | None = None
    unranked_raw: str | None = None
    n_unranked = 0
    if unranked_sample:
        # 조인 **이전** 프레임에서 뽑는다. 도메인당 상한은 조인본과 같게 걸어 도메인 편중을 막는다.
        unr = cc_all.sample(frac=1.0, random_state=seed + 1)
        unr = unr.groupby("domain", sort=False).head(per_domain_cap)
        unr = unr.head(int(unranked_sample))
        n_unranked = int(len(unr))
        up = raw_dir.parent / f"commoncrawl_unranked_urls_{date}.csv"
        pd.DataFrame({"url": unr["url"].tolist()}).to_csv(up, index=False)
        ur = raw_dir / f"cc_unranked_urls_{date}.csv.gz"
        pd.DataFrame({"url": unr["url"].tolist()}).to_csv(
            ur, index=False, compression="gzip"
        )
        unranked_path, unranked_raw = str(up), str(ur)

    cc = cc.sample(frac=1.0, random_state=seed)
    cc = cc.groupby("domain", sort=False).head(per_domain_cap)

    parsed = raw_dir.parent / f"commoncrawl_urls_{date}.csv"
    pd.DataFrame({"url": cc["url"].tolist()}).to_csv(parsed, index=False)

    raw_copy = raw_dir / f"cc_urls_{date}.csv.gz"
    pd.DataFrame({"url": cc["url"].tolist()}).to_csv(raw_copy, index=False, compression="gzip")

    return {
        "name": spec.name,
        "label": spec.label,
        "url": spec.url_template.format(crawl=crawl),
        "raw_path": str(raw_copy),
        "parsed_path": str(parsed),
        "pinned_id": crawl,
        "row_groups_used": used,
        "n_parts_available": len(all_paths),
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(raw_copy),
        "n_raw": n_cc_raw,
        "n_after_tranco_join": n_after_join,
        "n_after_domain_cap": int(len(cc)),
        "per_domain_cap": per_domain_cap,
        "tranco_list_id": TRANCO_LIST_ID if tranco_domains is not None else None,
        "unranked_parsed_path": unranked_path,
        "unranked_raw_path": unranked_raw,
        "n_unranked": n_unranked,
        "unranked_note": (
            "Tranco 조인을 뺀 benign 표본(민감도 세트). 인기도 confound는 줄지만 피싱·"
            "WebPhish 도메인 오염 위험이 커진다."
        )
        if unranked_path
        else None,
        "license_note": spec.license_note,
    }


# --------------------------------------------------------------------------------------
# 데이터셋 조립
# --------------------------------------------------------------------------------------


def build_external_dataset(
    out_dir: str | Path,
    parts: dict[str, pd.DataFrame],
    *,
    date: str = COLLECT_DATE,
) -> dict[str, str]:
    """소스별 파싱 결과를 `data/external/{source}_{date}.csv`로 저장한다.

    컬럼은 WebPhish CSV와 같은 `Category`(spam/ham) / `Data`(url) 규약에 `source`, `first_seen`을
    더한 형태다. 이렇게 두면 `load_webphish`와 같은 리더 규약을 그대로 재사용할 수 있다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, df in parts.items():
        label = int(df["label"].iloc[0]) if len(df) else SOURCES[name].label
        frame = pd.DataFrame(
            {
                "Category": ["spam" if label == 1 else "ham"] * len(df),
                "Data": df["url_src"].astype("string"),
                "source": df["source"].astype("string"),
                "first_seen": df["first_seen"].astype("string")
                if "first_seen" in df.columns
                else "",
            }
        )
        path = out / f"{name}_{date}.csv"
        frame.to_csv(path, index=False)
        written[name] = str(path)
    return written


def dedup_against(
    df: pd.DataFrame,
    webphish_df: pd.DataFrame,
    *,
    dedup: Literal["url", "etld1"] = "etld1",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """WebPhish와 겹치는 외부 행을 제거한다. 삭제는 **외부 쪽에서만** 한다.

    `dedup="url"`은 정규화 URL 정확 일치, `dedup="etld1"`은 eTLD+1까지 제거한다.
    양방향 겹침 규모는 보고용으로 둘 다 계산한다.
    """
    wp_urls = set(webphish_df["url"].astype(str))
    wp_groups = set(webphish_df["group"].astype(str))

    hit_url = df["url"].astype(str).isin(wp_urls)
    hit_grp = df["group"].astype(str).isin(wp_groups)
    stats = {
        "n_overlap_exact_url_vs_webphish": int(hit_url.sum()),
        "n_overlap_etld1_vs_webphish": int(hit_grp.sum()),
        "n_webphish_groups_seen_in_external": int(
            len(wp_groups & set(df["group"].astype(str)))
        ),
    }
    drop = hit_grp if dedup == "etld1" else hit_url
    stats["n_dropped_vs_webphish"] = int(drop.sum())
    return df[~drop].copy(), stats


def bias_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """설계 3절 하드 게이트용 편향 진단.

    benign의 `path_depth>=1` 비율이 phishing보다 낮으면 WebPhish와 **같은 방향**의 편향이므로
    수집을 다시 설계해야 한다. 그 판정을 `gate_passed`로 낸다.
    """
    out: dict[str, Any] = {}
    for key, lab in (("benign", 0), ("phishing", 1)):
        sub = df[df["label"] == lab]
        if len(sub) == 0:
            out[key] = {"n": 0}
            continue
        b = sub["url_bytes_len"]
        out[key] = {
            "n": int(len(sub)),
            "path_depth_ge1_frac": float((sub["path_depth"] >= 1).mean()),
            "path_depth_mean": float(sub["path_depth"].mean()),
            "url_bytes_len": {
                "mean": float(b.mean()),
                "p10": float(b.quantile(0.10)),
                "median": float(b.median()),
                "p90": float(b.quantile(0.90)),
                "max": int(b.max()),
            },
            "scheme_frac": float(
                sum(has_leading_scheme(u) for u in sub["url"].astype(str)) / len(sub)
            ),
            "n_groups": int(sub["group"].nunique()),
            "top5_group_frac": float(
                sub["group"].value_counts().head(5).sum() / len(sub)
            ),
            "top_groups": {
                str(k): int(v) for k, v in sub["group"].value_counts().head(10).items()
            },
        }

    ben = out.get("benign", {}).get("path_depth_ge1_frac")
    phi = out.get("phishing", {}).get("path_depth_ge1_frac")
    if ben is None or phi is None:
        out["gate_passed"] = False
        out["gate_note"] = "한쪽 클래스가 비어 진단 불가"
    else:
        out["gate_passed"] = bool(ben >= phi)
        out["gate_note"] = (
            "benign 경로 보유율이 phishing 이상 — WebPhish와 같은 방향의 편향이 아니다"
            if ben >= phi
            else (
                f"경고: benign 경로 보유율({ben:.3f})이 phishing({phi:.3f})보다 낮다. "
                "WebPhish와 같은 방향의 편향이므로 전이 실험은 검증이 아니다."
            )
        )
    return out


def load_external(
    csv_path: str | Path | Sequence[str | Path],
    mode: Literal["raw", "norm"] = "norm",
    *,
    dedup: Literal["url", "etld1"] | None = None,
    webphish_csv: str | Path | None = None,
    hosting_blocklist: frozenset[str] = HOSTING_BLOCKLIST,
    drop_hosting_from_benign: bool = True,
    drop_benign_on_phish_domains: bool = True,
    benign_cleaning: str | None = None,
    extractor: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """외부 수집 CSV(들)를 읽어 `load_webphish`와 **동일한 컬럼 규약**으로 돌려준다.

    반환 컬럼: url, label, group, url_len, url_bytes_len, path_depth (+ source).
    두 번째 값은 제거 통계 + 편향 진단 dict.

    ``benign_cleaning``은 benign 정제 절제 3조건(:data:`BENIGN_CLEANING_MODES`)을 한 인자로
    고른다. 주면 ``drop_*`` 불리언보다 우선한다.

    - ``"clean"``: 둘 다 적용(기본과 동일).
    - ``"keep_phish_domains"``: phishing eTLD+1 위의 benign을 남긴다.
    - ``"no_hosting_blocklist"``: 공유 호스팅/단축기 블록리스트를 적용하지 않는다.

    ``drop_benign_on_phish_domains``(설계 1.4절 1)는 phishing으로 등장한 eTLD+1을 benign에서
    지운다. 라벨 잡음은 줄지만 benign이 실제보다 깨끗해진다. benign 쪽 라벨 오염은 일반적으로
    성능을 감쇠시키는 방향으로 예상되지만 단조성이 보장되지는 않으므로(오염이 체계적일 때
    AUROC가 오히려 오를 수도 있다), 방향을 가정하지 말고 세 조건을 모두 돌려 비교한다.
    제거 건수는 조건과 무관하게 항상 ``stats["n_benign_on_phish_domains"]``에 남는다.
    """
    if benign_cleaning is not None:
        if benign_cleaning not in BENIGN_CLEANING_MODES:
            raise ValueError(
                f"benign_cleaning은 {BENIGN_CLEANING_MODES} 중 하나여야 한다 "
                f"(got {benign_cleaning!r})"
            )
        drop_hosting_from_benign = benign_cleaning != "no_hosting_blocklist"
        drop_benign_on_phish_domains = benign_cleaning != "keep_phish_domains"
    paths = (
        [Path(csv_path)]
        if isinstance(csv_path, (str, Path))
        else [Path(p) for p in csv_path]
    )
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"외부 데이터 파일이 없다: {p}")
        frames.append(pd.read_csv(p, dtype=str))
    raw = pd.concat(frames, ignore_index=True)
    for col in ("Category", "Data"):
        if col not in raw.columns:
            raise ValueError(f"필수 컬럼 없음: {col} (있는 컬럼: {list(raw.columns)})")

    stats: dict[str, Any] = {"n_raw": int(len(raw))}

    df = pd.DataFrame(
        {
            "url_src": raw["Data"].astype("string"),
            "category": raw["Category"].astype("string").str.strip().str.lower(),
            "source": raw["source"].astype("string")
            if "source" in raw.columns
            else pd.Series(["external"] * len(raw), dtype="string"),
        }
    )

    before = len(df)
    df = df[df["url_src"].notna() & (df["url_src"].str.strip().str.len() > 0)]
    stats["n_dropped_empty_url"] = before - len(df)

    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label[df["category"] == "spam"] = 1
    label[df["category"].isin({"ham", "benign", "legitimate", "good", "0"})] = 0
    keep = label.notna()
    stats["n_dropped_unknown_label"] = int((~keep).sum())
    df = df[keep].copy()
    df["label"] = label[keep].astype(int).to_numpy()

    df["url"] = [normalize_external_url(u, mode) for u in df["url_src"]]
    before = len(df)
    df = df[df["url"].str.len() > 0]
    stats["n_dropped_empty_after_norm"] = before - len(df)

    # 라벨 충돌: 같은 URL이 양쪽 클래스 -> 양쪽 모두 제거 (load_webphish와 같은 규약)
    nunique = df.groupby("url")["label"].transform("nunique")
    conflict = nunique > 1
    stats["n_conflicting_urls"] = int(df.loc[conflict, "url"].nunique())
    stats["n_dropped_label_conflict"] = int(conflict.sum())
    df = df[~conflict].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").copy()
    stats["n_dropped_duplicate"] = before - len(df)

    ex = extractor if extractor is not None else default_extractor()
    df["group"] = [etld1(u, ex) for u in df["url"]]
    df["url_len"] = df["url"].str.len().astype(int)
    df["url_bytes_len"] = [len(u.encode("utf-8")) for u in df["url"]]
    df["path_depth"] = [path_depth(u) for u in df["url"]]

    if drop_hosting_from_benign:
        hit = (df["label"] == 0) & df["group"].isin(hosting_blocklist)
        stats["n_dropped_hosting_blocklist"] = int(hit.sum())
        df = df[~hit].copy()
    else:
        stats["n_dropped_hosting_blocklist"] = 0

    # benign에서 피싱 도메인 제거 (설계 1.4절 1). 옵션을 끄면 건수만 세고 남긴다.
    phish_groups = set(df.loc[df["label"] == 1, "group"].astype(str))
    hit = (df["label"] == 0) & df["group"].astype(str).isin(phish_groups)
    stats["n_benign_on_phish_domains"] = int(hit.sum())
    stats["drop_benign_on_phish_domains"] = bool(drop_benign_on_phish_domains)
    if drop_benign_on_phish_domains:
        stats["n_dropped_phish_domain_from_benign"] = int(hit.sum())
        df = df[~hit].copy()
    else:
        stats["n_dropped_phish_domain_from_benign"] = 0

    if dedup is not None and webphish_csv is not None:
        from qrphish.urls import load_webphish

        wp, _ = load_webphish(webphish_csv, mode, extractor=ex)
        df, wp_stats = dedup_against(df, wp, dedup=dedup)
        stats.update(wp_stats)
        stats["dedup_mode"] = dedup

    df = df.drop(columns=["url_src", "category"]).reset_index(drop=True)

    stats["n_final"] = int(len(df))
    stats["n_final_benign"] = int((df["label"] == 0).sum())
    stats["n_final_phishing"] = int((df["label"] == 1).sum())
    stats["n_groups"] = int(df["group"].nunique())
    stats["by_source"] = {
        str(k): int(v) for k, v in df["source"].value_counts().items()
    }
    stats["benign_cleaning"] = str(
        benign_cleaning
        if benign_cleaning is not None
        else (
            "clean"
            if (drop_hosting_from_benign and drop_benign_on_phish_domains)
            else "custom"
        )
    )
    stats["drop_hosting_from_benign"] = bool(drop_hosting_from_benign)
    stats["bias_diagnostics"] = bias_diagnostics(df)
    return df, stats


def split_by_source(
    df: pd.DataFrame, keep: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``load_external`` 결과에서 주어진 소스의 행만 남기고 통계를 다시 낸다.

    primary(OpenPhish)와 secondary(Phishing.Database)는 **같은 benign 행 집합**을 써야
    비교가 성립한다. 그러려면 정제(라벨 충돌·중복·phishing 도메인 위 benign 제거)를 두 세트에
    따로 돌리면 안 된다 — phishing 소스가 다르면 제거되는 benign도 달라지기 때문이다.
    그래서 phishing 소스를 합집합으로 한 번 정제한 뒤, 여기서 소스로만 잘라 낸다.
    """
    want = {str(s).strip() for s in keep if str(s).strip()}
    sub = df[df["source"].astype(str).isin(want)].reset_index(drop=True).copy()
    stats: dict[str, Any] = {
        "kept_sources": sorted(want),
        "n_final": int(len(sub)),
        "n_final_benign": int((sub["label"] == 0).sum()),
        "n_final_phishing": int((sub["label"] == 1).sum()),
        "n_groups": int(sub["group"].nunique()),
        "by_source": {str(k): int(v) for k, v in sub["source"].value_counts().items()},
        "bias_diagnostics": bias_diagnostics(sub),
    }
    return sub, stats


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")


def env_key_present(spec: SourceSpec) -> bool:
    return spec.env_key is None or bool(os.environ.get(spec.env_key))
