"""외부 검증(F)용 URL 수집 CLI.

사용 예::

    uv run python scripts/collect_external.py --sources phishdb,openphish,commoncrawl,tranco \
        --out data/external --cc-row-groups 2 --openphish-days 90

원본은 `data/external/raw/`(gitignore)에, 정규화 CSV는 `data/external/`에, 재현용 메타와
편향 진단은 `reports/external/`(커밋 대상)에 쓴다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrphish import external as ext  # noqa: E402

DEFAULT_SOURCES = "phishdb,openphish,commoncrawl,tranco"


def _source_meta_path(rep_dir: Path, name: str, date: str) -> Path:
    return rep_dir / "sources" / f"{name}_{date}.json"


def _remember(rep_dir: Path, name: str, date: str, meta: dict, manifest: dict) -> None:
    """fetch 메타를 소스별 파일로 남긴다. --skip-fetch 실행에서도 manifest를 온전히 재조립한다."""
    ext.write_json(_source_meta_path(rep_dir, name, date), meta)
    manifest["sources"].append(meta)


def _recall(rep_dir: Path, name: str, date: str, manifest: dict) -> None:
    p = _source_meta_path(rep_dir, name, date)
    if p.exists():
        manifest["sources"].append(json.loads(p.read_text(encoding="utf-8")))
    else:
        print(f"[{name}] 경고: 캐시된 fetch 메타가 없다 ({p}). manifest에서 누락된다.")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="외부 검증용 URL 수집")
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--out", default="data/external")
    ap.add_argument("--reports", default="reports/external")
    ap.add_argument("--date", default=ext.COLLECT_DATE)
    ap.add_argument("--cc-row-groups", type=int, default=24)
    ap.add_argument("--cc-per-domain-cap", type=int, default=8)
    ap.add_argument("--cc-row-groups-per-part", type=int, default=1)
    ap.add_argument("--openphish-days", type=int, default=90)
    ap.add_argument("--openphish-max-commits", type=int, default=200)
    ap.add_argument("--tranco-top", type=int, default=50_000)
    ap.add_argument("--webphish", default="data/webphish.csv")
    ap.add_argument("--dedup", default="etld1", choices=["url", "etld1"])
    ap.add_argument(
        "--keep-benign-on-phish-domains",
        action="store_true",
        help="benign에서 피싱 도메인을 제거하지 않는다(민감도 분석용). 기본은 제거.",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="이미 받은 원본으로 재조립만 한다")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    rep_dir = Path(args.reports)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    names = [s.strip() for s in args.sources.split(",") if s.strip()]

    session = ext._session()
    manifest: dict = {
        "collect_date": args.date,
        "git_sha": _git_sha(),
        "scheme_policy_resolved": "strip",
        "dedup_mode": args.dedup,
        "drop_benign_on_phish_domains": not args.keep_benign_on_phish_domains,
        "sources": [],
        "excluded_sources": {
            "phishtank": "HTTP 403 (Cloudflare). 앱 키 발급 사실상 중단",
            "urlhaus": "threat 컬럼 100% malware_download — 피싱 라벨 아님",
        },
    }
    t0 = time.time()

    # Tranco를 먼저 받아야 Common Crawl 조인에 쓸 수 있다.
    tranco_raw: Path | None = None
    if "tranco" in names or "commoncrawl" in names:
        tranco_raw = raw_dir / f"tranco_{args.date}.csv"
        if not args.skip_fetch or not (tranco_raw.exists() and (out_dir / f"tranco_urls_{args.date}.csv").exists()):
            print("[tranco] 다운로드 중...", flush=True)
            meta = ext.fetch_tranco(
                raw_dir, session=session, date=args.date, n_domains=args.tranco_top
            )
            print(f"[tranco] {meta['n_raw']:,}개 도메인, 상위 {meta['n_top']:,}개 저장")
            ext.write_json(_source_meta_path(rep_dir, "tranco", args.date), meta)
            if "tranco" in names:
                manifest["sources"].append(meta)
        elif "tranco" in names:
            _recall(rep_dir, "tranco", args.date, manifest)

    fetch_meta: dict[str, dict] = {}
    if "phishdb" in names:
        p = raw_dir / f"phishdb_active_{args.date}.txt"
        if not args.skip_fetch or not p.exists():
            print("[phishdb] 다운로드 중 (~66MB)...", flush=True)
            fetch_meta["phishdb"] = ext.fetch_phishdb(raw_dir, session=session, date=args.date)
            print(f"[phishdb] {fetch_meta['phishdb']['n_raw']:,}줄")
            _remember(rep_dir, "phishdb", args.date, fetch_meta["phishdb"], manifest)
        else:
            _recall(rep_dir, "phishdb", args.date, manifest)

    if "openphish" in names:
        p = raw_dir / f"openphish_history_{args.date}.txt"
        if not args.skip_fetch or not p.exists():
            print(f"[openphish] 최근 {args.openphish_days}일 커밋 이력 누적 중...", flush=True)
            m = ext.fetch_openphish_history(
                raw_dir,
                days=args.openphish_days,
                max_commits=args.openphish_max_commits,
                session=session,
                date=args.date,
            )
            fetch_meta["openphish"] = m
            print(
                f"[openphish] 커밋 {m['n_commits_fetched']}/{m['n_commits_listed']}개에서 "
                f"고유 URL {m['n_raw']:,}건"
            )
            _remember(rep_dir, "openphish", args.date, m, manifest)
        else:
            _recall(rep_dir, "openphish", args.date, manifest)

    if "commoncrawl" in names:
        p = out_dir / f"commoncrawl_urls_{args.date}.csv"
        if not args.skip_fetch or not p.exists():
            print(f"[commoncrawl] {ext.CC_CRAWL_ID} row group {args.cc_row_groups}개 읽는 중...", flush=True)
            m = ext.fetch_commoncrawl(
                raw_dir,
                n_row_groups=args.cc_row_groups,
                session=session,
                date=args.date,
                tranco_csv=tranco_raw,
                per_domain_cap=args.cc_per_domain_cap,
                max_row_groups_per_part=args.cc_row_groups_per_part,
            )
            fetch_meta["commoncrawl"] = m
            print(
                f"[commoncrawl] 원본 {m['n_raw']:,} → Tranco 조인 {m['n_after_tranco_join']:,} "
                f"→ 도메인 상한 후 {m['n_after_domain_cap']:,}"
            )
            _remember(rep_dir, "commoncrawl", args.date, m, manifest)
        else:
            _recall(rep_dir, "commoncrawl", args.date, manifest)

    # 파싱 → 소스별 CSV
    parts = {}
    raw_paths = {
        "phishdb": raw_dir / f"phishdb_active_{args.date}.txt",
        "openphish": raw_dir / f"openphish_history_{args.date}.txt",
        "commoncrawl": out_dir / f"commoncrawl_urls_{args.date}.csv",
        "tranco": out_dir / f"tranco_urls_{args.date}.csv",
    }
    for name in names:
        path = raw_paths[name]
        if not path.exists():
            print(f"[{name}] 원본 없음, 건너뜀: {path}")
            continue
        parts[name] = ext.parse_source(ext.SOURCES[name], path)
        print(f"[{name}] 파싱 {len(parts[name]):,}행")

    written = ext.build_external_dataset(out_dir, parts, date=args.date)
    print("소스별 CSV:", written)

    # 주 평가 세트 = phishing 소스 + Common Crawl benign (Tranco는 EXT-B2 대조군이라 분리)
    main_names = [n for n in written if n != "tranco"]
    main_paths = [written[n] for n in main_names]
    print(f"\n[load] 주 세트 조립: {main_names}", flush=True)
    drop_benign = not args.keep_benign_on_phish_domains
    df, stats = ext.load_external(
        main_paths, "norm", dedup=args.dedup, webphish_csv=args.webphish,
        drop_benign_on_phish_domains=drop_benign,
    )
    merged = out_dir / f"external_{args.date}.csv"
    df.to_csv(merged, index=False)
    stats["output_csv"] = str(merged)
    print(
        f"[load] 최종 {stats['n_final']:,}행 "
        f"(benign {stats['n_final_benign']:,} / phishing {stats['n_final_phishing']:,}), "
        f"그룹 {stats['n_groups']:,}"
    )

    # EXT-B2 (Tranco 대조군)는 phishing과 짝지어 별도로 낸다.
    b2_stats = None
    if "tranco" in written:
        b2_paths = [written[n] for n in written if n in ("tranco", "phishdb", "openphish")]
        df_b2, b2_stats = ext.load_external(
            b2_paths, "norm", dedup=args.dedup, webphish_csv=args.webphish,
            drop_benign_on_phish_domains=drop_benign,
        )
        b2_path = out_dir / f"external_b2_{args.date}.csv"
        df_b2.to_csv(b2_path, index=False)
        b2_stats["output_csv"] = str(b2_path)
        print(f"[load] EXT-B2 {b2_stats['n_final']:,}행")

    manifest["counts"] = {
        k: v for k, v in stats.items() if k.startswith("n_") or k == "by_source"
    }
    manifest["ext_b2_counts"] = (
        {k: v for k, v in b2_stats.items() if k.startswith("n_") or k == "by_source"}
        if b2_stats
        else None
    )
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    ext.write_json(rep_dir / "manifest.json", manifest)

    # 어느 CSV의 진단인지 남긴다 — 전이 단계가 게이트를 승계할 때 이 값으로 짝을 찾는다.
    main_diag = dict(stats["bias_diagnostics"])
    main_diag["output_csv"] = str(merged)
    b2_diag = None
    if b2_stats:
        b2_diag = dict(b2_stats["bias_diagnostics"])
        b2_diag["output_csv"] = b2_stats["output_csv"]
    diag = {
        "collect_date": args.date,
        "dedup_mode": args.dedup,
        "drop_benign_on_phish_domains": not args.keep_benign_on_phish_domains,
        "main": main_diag,
        "ext_b2": b2_diag,
        "dedup_stats": {k: v for k, v in stats.items() if k.startswith("n_")},
    }
    ext.write_json(rep_dir / "bias_diagnostics.json", diag)

    m = stats["bias_diagnostics"]
    print("\n=== 편향 진단 (하드 게이트) ===")
    for key in ("benign", "phishing"):
        d = m.get(key, {})
        if d.get("n"):
            print(
                f"  {key:9s} n={d['n']:>8,}  path>=1 {d['path_depth_ge1_frac']:.3f}  "
                f"bytes med {d['url_bytes_len']['median']:.0f} "
                f"(p10 {d['url_bytes_len']['p10']:.0f} / p90 {d['url_bytes_len']['p90']:.0f})  "
                f"top5 group {d['top5_group_frac']:.3f}"
            )
    print(f"  gate_passed = {m['gate_passed']}  — {m['gate_note']}")
    print(f"\n총 소요 {manifest['elapsed_sec']}초. 메타: {rep_dir}/manifest.json")
    return 0 if m["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
