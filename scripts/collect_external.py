"""외부 검증(F)용 URL 수집 CLI.

사용 예::

    # 전체 수집 (네트워크). unranked CC benign 민감도 세트까지 함께 만든다
    uv run python scripts/collect_external.py --cc-row-groups 24 --cc-unranked-sample 65000

    # 이미 받은 원본으로 재조립만 (네트워크 없음)
    uv run python scripts/collect_external.py --skip-fetch --cc-unranked-sample 65000

만드는 세트 (모두 `data/external/external_{set}_{date}.csv`):

===========================  ==========================================================
set                          내용
===========================  ==========================================================
primary                      OpenPhish 최근 90일 + CC×Tranco benign. **주 결과**
secondary                    Phishing.Database 아카이브 + **같은 benign 행**. robustness
primary_ccunranked           Tranco 조인을 뺀 CC benign. popularity confound 민감도
primary_keep_phish_domains   benign 정제 절제 — 피싱 도메인 위 benign 유지
primary_no_hosting_blocklist benign 정제 절제 — 호스팅/단축기 블록리스트 미적용
ext_b2                       Tranco 맨 도메인 대조군(설계상 편향 게이트를 통과하지 못한다)
===========================  ==========================================================

primary와 secondary가 같은 benign 행을 갖도록, 정제는 phishing 소스 합집합에서 **한 번만**
하고 `qrphish.external.split_by_source`로 잘라 낸다.

원본은 `data/external/raw/`(gitignore)에, 정규화 CSV는 `data/external/`에, 재현용 메타와
편향 진단은 `reports/external/`(커밋 대상)에 쓴다. 종료 코드는 ext_b2를 뺀 모든 세트가
편향 게이트를 통과할 때만 0이다.
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
    ap.add_argument(
        "--cc-unranked-sample",
        type=int,
        default=0,
        help=(
            "Tranco 조인 없이 뽑는 benign 표본 수(민감도 세트). >0이면 소스 목록에 "
            "commoncrawl_unranked를 자동으로 넣는다. 0이면 만들지 않는다"
        ),
    )
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
    if args.cc_unranked_sample and "commoncrawl_unranked" not in names:
        names.append("commoncrawl_unranked")

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

    if "commoncrawl" in names or "commoncrawl_unranked" in names:
        p = out_dir / f"commoncrawl_urls_{args.date}.csv"
        pu = out_dir / f"commoncrawl_unranked_urls_{args.date}.csv"
        need_unranked = bool(args.cc_unranked_sample) and not pu.exists()
        if not args.skip_fetch or not p.exists() or need_unranked:
            print(f"[commoncrawl] {ext.CC_CRAWL_ID} row group {args.cc_row_groups}개 읽는 중...", flush=True)
            m = ext.fetch_commoncrawl(
                raw_dir,
                n_row_groups=args.cc_row_groups,
                session=session,
                date=args.date,
                tranco_csv=tranco_raw,
                per_domain_cap=args.cc_per_domain_cap,
                max_row_groups_per_part=args.cc_row_groups_per_part,
                unranked_sample=args.cc_unranked_sample or None,
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
        "commoncrawl_unranked": out_dir / f"commoncrawl_unranked_urls_{args.date}.csv",
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

    drop_benign = not args.keep_benign_on_phish_domains
    phish_names = [n for n in ("openphish", "phishdb") if n in written]

    def _union(benign: str, cleaning: str | None):
        """phishing 소스 합집합 + benign 소스 하나를 **한 번에** 정제한다.

        primary(OpenPhish)와 secondary(Phishing.Database)가 같은 benign 행 집합을 갖게 하려면
        정제를 세트별로 따로 돌리면 안 된다. 합집합에서 한 번 정제한 뒤 소스로 잘라 낸다.
        """
        paths = [written[n] for n in phish_names + [benign] if n in written]
        return ext.load_external(
            paths,
            "norm",
            dedup=args.dedup,
            webphish_csv=args.webphish,
            drop_benign_on_phish_domains=drop_benign,
            benign_cleaning=cleaning,
        )

    sets: dict[str, dict] = {}

    def _emit(name: str, df, stats: dict, filename: str, note: str | None = None) -> None:
        path = out_dir / filename
        df.to_csv(path, index=False)
        stats = dict(stats)
        stats["output_csv"] = str(path)
        if note:
            stats["note"] = note
        sets[name] = stats
        d = stats["bias_diagnostics"]
        print(
            f"[set] {name:28s} n={stats['n_final']:>8,} "
            f"(benign {stats['n_final_benign']:,} / phishing {stats['n_final_phishing']:,}) "
            f"gate={d['gate_passed']}  -> {path.name}"
        )

    # --- clean 조건: primary / secondary는 같은 benign 행 집합을 공유한다 ---------------
    if "commoncrawl" in written:
        df_cc, st_cc = _union("commoncrawl", "clean")
        for name, phish in (("primary", "openphish"), ("secondary", "phishdb")):
            if phish not in written:
                continue
            sub, sub_st = ext.split_by_source(df_cc, [phish, "commoncrawl"])
            sub_st["benign_cleaning"] = "clean"
            sub_st["dedup_stats"] = {k: v for k, v in st_cc.items() if k.startswith("n_")}
            _emit(
                name,
                sub,
                sub_st,
                f"external_{name}_{args.date}.csv",
                note=(
                    "primary: OpenPhish 최근 90일 phishing + CC×Tranco benign"
                    if name == "primary"
                    else "secondary(robustness): Phishing.Database 아카이브 + 동일 benign"
                ),
            )

        # --- benign 정제 절제 (리뷰 02: 필수). primary 기준 3조건 -----------------------
        for cleaning in ext.BENIGN_CLEANING_MODES:
            if cleaning == "clean" or "openphish" not in written:
                continue
            df_c, st_c = _union("commoncrawl", cleaning)
            sub, sub_st = ext.split_by_source(df_c, ["openphish", "commoncrawl"])
            sub_st["benign_cleaning"] = cleaning
            sub_st["dedup_stats"] = {k: v for k, v in st_c.items() if k.startswith("n_")}
            _emit(
                f"primary_{cleaning}",
                sub,
                sub_st,
                f"external_primary_{cleaning}_{args.date}.csv",
                note=(
                    "benign 정제 절제 조건 — primary와 같은 phishing 행, benign만 다르다. "
                    "benign 쪽 라벨 오염은 일반적으로 성능을 감쇠시키는 방향으로 예상되지만 "
                    "단조성이 보장되지는 않는다. 세 조건의 AUROC 차이를 직접 본다."
                ),
            )

    # --- unranked CC benign 민감도 세트 -----------------------------------------------
    if "commoncrawl_unranked" in written and "openphish" in written:
        df_u, st_u = _union("commoncrawl_unranked", "clean")
        sub, sub_st = ext.split_by_source(df_u, ["openphish", "commoncrawl_unranked"])
        sub_st["benign_cleaning"] = "clean"
        # 오염 위험의 크기: 조인을 뺀 benign 중 피싱 도메인/WebPhish와 겹쳐 제거된 행 수.
        sub_st["dedup_stats"] = {k: v for k, v in st_u.items() if k.startswith("n_")}
        _emit(
            "primary_ccunranked",
            sub,
            sub_st,
            f"external_primary_ccunranked_{args.date}.csv",
            note=(
                "Tranco 조인을 뺀 CC benign. 인기도(popularity) confound를 줄이는 대신, "
                "benign에 실제 피싱/저품질 도메인이 섞일 오염 위험이 커진다. "
                "주 결과가 아니라 민감도 세트다."
            ),
        )

    # --- EXT-B2 (Tranco 맨 도메인 대조군) ---------------------------------------------
    if "tranco" in written:
        df_b2, st_b2 = _union("tranco", "clean")
        _emit("ext_b2", df_b2, st_b2, f"external_b2_{args.date}.csv",
              note="EXT-B2 대조군: Tranco 맨 도메인 benign")

    primary = sets.get("primary") or next(iter(sets.values()), None)
    if primary is None:
        print("경고: 조립된 세트가 하나도 없다.")
        return 2

    manifest["sets"] = {
        k: {
            **{kk: vv for kk, vv in v.items() if kk.startswith("n_") or kk == "by_source"},
            "output_csv": v["output_csv"],
            "benign_cleaning": v.get("benign_cleaning"),
            "note": v.get("note"),
            "union_dedup_stats": v.get("dedup_stats"),
            "gate_passed": v["bias_diagnostics"]["gate_passed"],
            "gate_note": v["bias_diagnostics"]["gate_note"],
        }
        for k, v in sets.items()
    }
    manifest["primary_set"] = "primary"
    manifest["counts"] = {
        k: v for k, v in primary.items() if k.startswith("n_") or k == "by_source"
    }
    manifest["ext_b2_counts"] = (
        {
            k: v
            for k, v in sets["ext_b2"].items()
            if k.startswith("n_") or k == "by_source"
        }
        if "ext_b2" in sets
        else None
    )
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    ext.write_json(rep_dir / "manifest.json", manifest)

    # 어느 CSV의 진단인지 남긴다 — 전이 단계가 게이트를 승계할 때 이 값으로 짝을 찾는다.
    def _diag(v: dict) -> dict:
        d = dict(v["bias_diagnostics"])
        d["output_csv"] = v["output_csv"]
        d["benign_cleaning"] = v.get("benign_cleaning")
        return d

    diag = {
        "collect_date": args.date,
        "dedup_mode": args.dedup,
        "drop_benign_on_phish_domains": drop_benign,
        "main": _diag(primary),
        "ext_b2": _diag(sets["ext_b2"]) if "ext_b2" in sets else None,
        "sets": {k: _diag(v) for k, v in sets.items()},
        "dedup_stats": primary.get("dedup_stats", {}),
    }
    ext.write_json(rep_dir / "bias_diagnostics.json", diag)

    print("\n=== 편향 진단 (하드 게이트) ===")
    all_ok = True
    for name, v in sets.items():
        m = v["bias_diagnostics"]
        # EXT-B2(Tranco 맨 도메인)는 설계상 benign이 전부 루트 URL이라 게이트를 통과할 수
        # 없는 **대조군**이다. 종료 코드에는 넣지 않는다.
        if name != "ext_b2":
            all_ok = all_ok and bool(m["gate_passed"])
        print(f"\n[{name}] {Path(v['output_csv']).name}")
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
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
