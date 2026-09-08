"""reports/의 집계 CSV·JSON에서 docs/RESULTS.md용 마크다운 표를 생성한다.

수치를 손으로 옮기다 생기는 오타를 막기 위한 스크립트다. 표준 출력으로 나온
마크다운을 그대로 문서에 붙인다.

    uv run python scripts/make_results_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

STRATA = ["v2", "v3", "v4"]
BASE = "norm-exact-data_only-fixed-small_cnn"
# motif 러너는 모델을 학습하지 않으므로 arch가 붙지 않은 조건 id를 쓴다.
MOTIF_COND = "norm-exact-data_only-fixed"


def load_main() -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with (REPORTS / "table_main.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[(row["condition_id"], row["stratum"])] = row
    return rows


def load_baselines() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with (REPORTS / "table_baselines.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[(row["condition_id"], row["stratum"], row["baseline"])] = row
    return rows


MAIN = load_main()
BASELINES = load_baselines()


def col(row: dict[str, str] | None, *names: str) -> str | None:
    """옛/새 컬럼명을 모두 받아준다.

    ``text_upper_bound_auroc`` → ``decoded_text_reference_auroc`` 개명 이후에도
    재집계 전의 기존 ``reports/*.csv``를 그대로 읽을 수 있어야 한다.
    """
    if row is None:
        return None
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def f3(value: str | float | None) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.3f}"


def cell(condition: str, stratum: str) -> str:
    """조건×층 AUROC와 클러스터 부트스트랩 95% CI."""
    row = MAIN.get((condition, stratum))
    if row is None:
        return "—"
    point = col(row, "auroc_mean", "auroc_pooled")
    return f"{f3(point)} [{f3(col(row, 'auroc_ci_lo'))}, {f3(col(row, 'auroc_ci_hi'))}]"


def baseline_cell(condition: str, stratum: str, baseline: str) -> str:
    row = BASELINES.get((condition, stratum, baseline))
    if row is None:
        return "—"
    return f3(row["auroc"])


def auroc(condition: str, stratum: str) -> float | None:
    v = col(MAIN.get((condition, stratum)), "auroc_mean", "auroc_pooled")
    return float(v) if v is not None else None


def diff(cond_a: str, cond_b: str, stratum: str) -> str:
    """cond_a - cond_b (부호 있는 3자리)."""
    a, b = auroc(cond_a, stratum), auroc(cond_b, stratum)
    if a is None or b is None:
        return "—"
    return f"{a - b:+.3f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def strata_table(header0: str, rows: list[tuple[str, list[str]]]) -> str:
    return table([header0, *STRATA], [[name, *cells] for name, cells in rows])


def table_counts() -> str:
    counts = json.loads((REPORTS / "stratum_counts.json").read_text(encoding="utf-8"))
    rows = []
    for c in counts["cells"]:
        per_class = c["n_by_class"]
        rows.append(
            [
                c["length_match"],
                c["stratum"],
                str(c["n_in_stratum_before_match"]),
                str(c["n_total"]),
                str(per_class.get("0", 0)),
                str(per_class.get("1", 0)),
                c["tier"],
            ]
        )
    return table(
        ["길이 통제", "층", "매칭 전 n", "매칭 후 n", "benign", "phishing", "등급"], rows
    )


def table_main() -> str:
    rows: list[tuple[str, list[str]]] = [
        ("SmallCNN (주 조건)", [cell(BASE, s) for s in STRATA]),
        ("BitMLP", [cell("norm-exact-data_only-fixed-bit_mlp", s) for s in STRATA]),
        ("라벨 셔플 (바닥)", [cell(f"{BASE}-labelshuffle", s) for s in STRATA]),
        (
            "char n-gram LR (디코딩 텍스트 참조 기준)",
            [baseline_cell(BASE, s, "charngram_lr") for s in STRATA],
        ),
        ("byte-hist LR", [baseline_cell(BASE, s, "bytehist_lr") for s in STRATA]),
        ("length LR", [baseline_cell(BASE, s, "length_lr") for s in STRATA]),
        ("version LR", [baseline_cell(BASE, s, "version_lr") for s in STRATA]),
    ]
    gaps = []
    for s in STRATA:
        ref = col(MAIN.get((BASE, s)), "decoded_text_reference_auroc", "text_upper_bound_auroc")
        if ref is None:
            brow = BASELINES.get((BASE, s, "charngram_lr"))
            ref = brow["auroc"] if brow else None
        cnn = auroc(BASE, s)
        gaps.append("—" if ref is None or cnn is None else f"{float(ref) - cnn:+.3f}")
    rows.append(("참조 기준 − CNN 갭", gaps))
    return strata_table("모델 / 기준선", rows)


def table_h4() -> str:
    rows: list[tuple[str, list[str]]] = [
        ("CNN · 정상 배치", [cell(BASE, s) for s in STRATA]),
        ("CNN · shuffle-pos", [cell(f"{BASE}-shufflepos", s) for s in STRATA]),
        ("CNN 차이", [diff(f"{BASE}-shufflepos", BASE, s) for s in STRATA]),
        (
            "MLP · 정상 배치",
            [cell("norm-exact-data_only-fixed-bit_mlp", s) for s in STRATA],
        ),
        (
            "MLP · shuffle-pos",
            [cell("norm-exact-data_only-fixed-bit_mlp-shufflepos", s) for s in STRATA],
        ),
        (
            "MLP 차이",
            [
                diff(
                    "norm-exact-data_only-fixed-bit_mlp-shufflepos",
                    "norm-exact-data_only-fixed-bit_mlp",
                    s,
                )
                for s in STRATA
            ],
        ),
    ]
    return strata_table("조건", rows)


def table_length() -> str:
    none_c = "norm-none-data_only-fixed-small_cnn"
    rows: list[tuple[str, list[str]]] = [
        ("L-exact (주 조건)", [cell(BASE, s) for s in STRATA]),
        (
            "L-quantile",
            [cell("norm-quantile-data_only-fixed-small_cnn", s) for s in STRATA],
        ),
        ("L-none", [cell(none_c, s) for s in STRATA]),
        ("L-none − L-exact", [diff(none_c, BASE, s) for s in STRATA]),
        ("PAD-rand (L-none 위)", [cell(f"{none_c}-padrand", s) for s in STRATA]),
        ("PAD-rand − L-none", [diff(f"{none_c}-padrand", none_c, s) for s in STRATA]),
        ("length LR @ L-exact", [baseline_cell(BASE, s, "length_lr") for s in STRATA]),
        ("length LR @ L-none", [baseline_cell(none_c, s, "length_lr") for s in STRATA]),
        ("version LR @ L-none", [baseline_cell(none_c, s, "version_lr") for s in STRATA]),
    ]
    return strata_table("조건", rows)


def table_ablation() -> str:
    rows: list[tuple[str, list[str]]] = [
        ("norm · mask-fixed · data_only (주 조건)", [cell(BASE, s) for s in STRATA]),
        ("raw (정규화 해제)", [cell("raw-exact-data_only-fixed-small_cnn", s) for s in STRATA]),
        ("raw − norm", [diff("raw-exact-data_only-fixed-small_cnn", BASE, s) for s in STRATA]),
        ("feat-all (기능 패턴 포함)", [cell("norm-exact-all-fixed-small_cnn", s) for s in STRATA]),
        ("feat-all − data_only", [diff("norm-exact-all-fixed-small_cnn", BASE, s) for s in STRATA]),
        ("mask-auto", [cell("norm-exact-data_only-auto-small_cnn", s) for s in STRATA]),
        ("mask-auto − mask-fixed", [diff("norm-exact-data_only-auto-small_cnn", BASE, s) for s in STRATA]),
        (
            "maskindex LR @ mask-auto",
            [
                baseline_cell("norm-exact-data_only-auto-small_cnn", s, "maskindex_lr")
                for s in STRATA
            ],
        ),
        ("mask-off (언마스킹)", [cell("norm-exact-data_only-off-small_cnn", s) for s in STRATA]),
        ("mask-off − mask-fixed", [diff("norm-exact-data_only-off-small_cnn", BASE, s) for s in STRATA]),
        ("path+ 부분집합", [cell(f"{BASE}-pathplus", s) for s in STRATA]),
        (
            "path+ 참조 기준 (char n-gram LR)",
            [baseline_cell(f"{BASE}-pathplus", s, "charngram_lr") for s in STRATA],
        ),
        ("EC-M (v3 한정)", [cell(f"{BASE}-ecm", s) for s in STRATA]),
        (
            "EC-M 참조 기준 (char n-gram LR)",
            [baseline_cell(f"{BASE}-ecm", s, "charngram_lr") for s in STRATA],
        ),
    ]
    return strata_table("조건", rows)


CAM_KINDS = [
    ("char", "char (URL 문자)"),
    ("pad", "pad (패딩)"),
    ("ec", "ec (오류정정)"),
    ("function", "function (기능 패턴)"),
    ("length_header", "length_header"),
    ("mode_header", "mode_header"),
    ("remainder", "remainder"),
]


def _explain() -> dict:
    """``reports/explain_summary.json``. 아직 안 돌렸으면 빈 dict."""
    path = REPORTS / "explain_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def table_cam() -> str:
    explain = _explain()
    rows: list[tuple[str, list[str]]] = []
    for key, label in CAM_KINDS:
        rows.append(
            (
                label,
                [
                    f"{explain[s]['cam_mass_by_kind'][key] * 100:.1f}%" if s in explain else "—"
                    for s in STRATA
                ],
            )
        )
    data_only = []
    for s in STRATA:
        if s not in explain:
            data_only.append("—")
            continue
        mass = explain[s]["cam_mass_by_kind"]
        total = mass["char"] + mass["pad"] + mass["ec"] + mass["length_header"] + mass["mode_header"]
        data_only.append(f"{mass['char'] / total * 100:.1f}%")
    rows.append(("char / 데이터 모듈 내 비율", data_only))
    rows.append(("n_samples", [str(explain[s]["n_samples"]) if s in explain else "—" for s in STRATA]))
    return strata_table("CAM 질량 (kind)", rows)


def _enr_cell(explain: dict, stratum: str, block: str, kind: str) -> str:
    d = (explain.get(stratum) or {}).get(block) or {}
    v = d.get(kind)
    return "—" if v is None else f"{float(v):.2f}"


def table_cam_enrichment() -> str:
    """CAM enrichment(질량 분수 / 면적 분수)와 균일 난수 기준선, 무작위화 sanity 값.

    enrichment는 1이면 "그 영역을 면적만큼만 봤다"는 뜻이다. 난수 attribution 기준선이
    1 근처인지 확인해야 enrichment 값을 그대로 읽을 수 있다.
    """
    explain = _explain()
    rows: list[tuple[str, list[str]]] = []
    for key, label in CAM_KINDS:
        rows.append(
            (f"{label} · enrichment", [_enr_cell(explain, s, "cam_enrichment", key) for s in STRATA])
        )
        rows.append(
            (
                f"{label} · 난수 기준선",
                [_enr_cell(explain, s, "random_attribution_enrichment", key) for s in STRATA],
            )
        )
    rows.append(
        (
            "무작위화 모델 CAM 상관 (sanity)",
            [
                f"{float(explain[s]['sanity_randomized_cam_corr']):.3f}"
                if s in explain and explain[s].get("sanity_randomized_cam_corr") is not None
                else "—"
                for s in STRATA
            ],
        )
    )
    return strata_table("Grad-CAM enrichment", rows)


MOTIF_ROWS = [
    ("patch 2×2 히스토그램 LR", "patch2"),
    ("patch 3×3 히스토그램 LR", "patch3"),
    ("patch 3×3 spatial pyramid LR", "pyramid3"),
    ("patch 3×3 · 라벨 셔플 (바닥)", "patch3_labelshuffle"),
]


def load_motifs() -> dict[str, dict]:
    """``reports/motifs/{condition_id}/{stratum}/results.json``. 안 돌렸으면 빈 dict."""
    return _load_phase("motifs", MOTIF_COND)


def motif_cell(motifs: dict[str, dict], stratum: str, key: str) -> str:
    rep = (motifs.get(stratum, {}).get("representations") or {}).get(key)
    if not rep:
        return "—"
    # motif 러너의 results.json은 표현별 dict 안에만 집계값을 담는다(최상위 pooling 필드 없음).
    # auroc_pooled/auroc_pooled_ci가 짝을 이루므로 점추정도 같은 풀링에서 읽어야 CI와 맞는다.
    lo, hi = (rep.get("auroc_pooled_ci") or [None, None])[:2]
    point = rep.get("auroc_pooled")
    if point is None:
        point = rep.get("auroc_mean")
    return f"{f3(point)} [{f3(lo)}, {f3(hi)}]"


def motif_table() -> str:
    """Bag-of-QR-patches 결과를 CNN·byte-hist LR·BitMLP와 나란히 놓는다 (Q1 직접 측정).

    ``reports/motifs/*/results.json``(motif 러너)과 ``reports/table_main.csv``
    (기존 집계)를 합쳐 만든다.
    """
    motifs = load_motifs()
    rows: list[tuple[str, list[str]]] = [
        (label, [motif_cell(motifs, s, key) for s in STRATA]) for label, key in MOTIF_ROWS
    ]
    rows += [
        ("SmallCNN (주 조건)", [cell(BASE, s) for s in STRATA]),
        ("byte-hist LR", [baseline_cell(BASE, s, "bytehist_lr") for s in STRATA]),
        ("BitMLP", [cell("norm-exact-data_only-fixed-bit_mlp", s) for s in STRATA]),
    ]
    # 창 개수는 층마다 다르므로(격자 크기 차이) 함께 싣는다.
    rows.append(
        (
            "3×3 창 개수 (평균)",
            [
                f"{motifs[s]['data']['n_windows_mean']:.0f}" if s in motifs else "—"
                for s in STRATA
            ],
        )
    )
    return strata_table("표현 / 모델", rows)


def _load_phase(name: str, condition: str = BASE) -> dict[str, dict]:
    """``reports/{name}/{condition_id}/{stratum}/results.json``. 안 돌렸으면 빈 dict.

    조건 id가 경로에 들어가므로 다른 조건의 산출물과 섞이지 않는다.
    """
    out: dict[str, dict] = {}
    root = REPORTS / name / condition
    if not root.exists():
        return out
    for path in sorted(root.glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[str(data.get("stratum", path.parent.name))] = data
    return out


def table_probes() -> str:
    """어휘 프로브 요약 — 유의 목표 수/전체와 상위 10개."""
    probes = _load_phase("probes")
    if not probes:
        return ""
    rows: list[tuple[str, list[str]]] = []
    for label, key in (("유의 목표 수 / 전체", None), ("유의 비율", "frac_significant")):
        cells = []
        for s in STRATA:
            summ = (probes.get(s) or {}).get("summary")
            if not summ:
                cells.append("—")
            elif key is None:
                cells.append(f"{summ['n_significant']} / {summ['n_targets']}")
            else:
                cells.append(f"{float(summ[key]) * 100:.1f}%")
        rows.append((label, cells))
    lab = []
    for s in STRATA:
        d = ((probes.get(s) or {}).get("summary") or {}).get("label_probe")
        lab.append(f3(d["score"]) if d else "—")
    rows.append(("라벨 프로브 (참고 상한선)", lab))
    out = [strata_table("어휘 프로브", rows)]

    for s in STRATA:
        summ = (probes.get(s) or {}).get("summary")
        if not summ or not summ.get("top10"):
            continue
        out.append(f"\n**{s} 상위 10 목표**\n")
        out.append(
            table(
                ["목표", "지표", "점수", "셔플", "무작위 초기화"],
                [
                    [t["target"], t["metric"], f3(t["score"]), f3(t["shuffle"]),
                     f3(t["random_init"])]
                    for t in summ["top10"]
                ],
            )
        )
    return "\n".join(out)


OCC_CONDS = [
    ("phishing motif", "phishing_motif"),
    ("random (phishing 개수 맞춤)", "random"),
    ("benign motif", "benign_motif"),
    ("random (benign 개수 맞춤)", "random_benign_matched"),
]


def table_occlusion() -> str:
    """인과 절제 — 조건별 ΔAUROC, 평균 로짓 변화, 뒤집은 모듈 수."""
    occ = _load_phase("occlusion")
    if not occ:
        return ""
    out = []
    for s in STRATA:
        agg = (occ.get(s) or {}).get("aggregate")
        if not agg:
            continue
        rows = []
        for label, key in OCC_CONDS:
            b = agg.get(key)
            if not b:
                continue
            per_seed = (occ[s].get("per_seed") or [])
            cis = [
                d["conditions"][key]["d_auroc_ci"]
                for d in per_seed
                if key in d.get("conditions", {})
            ]
            ci = (
                f"[{f3(sum(c[0] for c in cis) / len(cis))}, "
                f"{f3(sum(c[1] for c in cis) / len(cis))}]"
                if cis
                else "—"
            )
            rows.append(
                [
                    label,
                    f"{float(b['d_auroc']):+.3f}",
                    ci,
                    f"{float(b['mean_logit_delta']):+.3f}",
                    f"{float(b['mean_logit_delta_phishing']):+.3f}",
                    f"{float(b['mean_logit_delta_benign']):+.3f}",
                    f"{float(b['n_flipped_mean']):.1f}",
                    f"{int(b['n_seeds_d_auroc_ci_excludes_0'])}",
                ]
            )
        out.append(
            f"\n**{s}** (원본 AUROC {f3(agg.get('auroc_original'))})\n\n"
            + table(
                ["조건", "ΔAUROC", "시드별 CI 평균", "로짓 Δ", "로짓 Δ(phishing)",
                 "로짓 Δ(benign)", "뒤집은 모듈 수", "CI가 0을 제외한 시드"],
                rows,
            )
        )
    return "\n".join(out)


H_LABEL = {
    "H1": "CNN > 라벨 셔플 바닥",
    "H2": "디코딩 텍스트 참조 기준 > CNN",
    "H3": "L-none > L-exact",
    "H4": "정상 배치 > shuffle-pos",
}


def _p(value: float) -> str:
    """Holm 보정 p값. 2000회 부트스트랩의 해상도(1/2001)를 넘어가면 부등호로 쓴다."""
    v = float(value)
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def hypotheses_table() -> str:
    """H1~H4 판정 표 — 층별 추정치·95% CI·Holm 보정 p·판정.

    ``reports/hypotheses.json``은 시드 층화 클러스터 부트스트랩(``pooling:
    seed_stratified``)으로 만든 값이다. 시드별로 AUROC를 계산해 평균하므로
    시드 간 점수 척도 차이가 통합 추정치를 끌고 가지 않는다.
    """
    path = REPORTS / "hypotheses.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    tests = {(t["hypothesis"], t["stratum"]): t for t in data.get("tests", [])}
    if not tests:
        return ""
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        for s in STRATA:
            t = tests.get((h, s))
            if t is None:
                continue
            lo, hi = t["ci"][:2]
            rows.append(
                [
                    f"{h}. {H_LABEL[h]}",
                    s,
                    f"{float(t['estimate']):+.3f}",
                    f"[{float(lo):+.3f}, {float(hi):+.3f}]",
                    _p(t["p_holm"]),
                    "기각" if t.get("reject") else "비기각",
                ]
            )
    return table(
        ["가설", "층", "ΔAUROC 추정치", "95% CI", "p (Holm)", "판정"],
        rows,
    )


# --------------------------------------------------------------------------- 전이(F)
TRANSFER_MODE_LABEL = {
    "a": "F-a zero-shot (WebPhish→외부)",
    "b": "F-b in-domain (외부→외부)",
    "c": "F-c 역방향 (외부→WebPhish)",
}
TRANSFER_BASELINES = [
    ("charngram_lr", "char n-gram LR"),
    ("bytehist_lr", "byte-hist LR"),
    ("motif_patch3_lr", "motif-hist LR (3x3)"),
    ("length_lr", "length LR (게이트)"),
    ("version_lr", "version LR (게이트)"),
]


def load_transfer() -> dict[tuple[str, str, str], dict]:
    """``reports/transfer/{source_tag}/{mode}/{condition_id}/{stratum}/results.json``.

    안 돌렸으면 빈 dict. 키는 ``(source_tag, mode, stratum)``이다.
    """
    out: dict[tuple[str, str, str], dict] = {}
    root = REPORTS / "transfer"
    if not root.exists():
        return out
    for path in sorted(root.glob("*/*/*/*/results.json")):
        stratum, _cid, mode, tag = (
            path.parent.name,
            path.parent.parent.name,
            path.parent.parent.parent.name,
            path.parent.parent.parent.parent.name,
        )
        try:
            out[(tag, mode, stratum)] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return out


def load_motif_replication() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    root = REPORTS / "transfer"
    if not root.exists():
        return out
    for path in sorted(root.glob("*/motif_replication/*/replication.json")):
        tag = path.parent.parent.parent.name
        try:
            out[(tag, path.parent.name)] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return out


def transfer_table() -> str:
    """F-a/F-b/F-c × 층 × {CNN, 텍스트 기준선, 순열 바닥선} + motif 재현성 (설계 6절).

    바닥선은 그룹 단위 라벨 순열의 97.5 백분위다. CNN AUROC의 CI 하한이 이 값 이하면
    "우연과 구분 불가"(collapse)다.
    """
    data = load_transfer()
    if not data:
        return ""
    tags = sorted({k[0] for k in data})
    out: list[str] = []
    for tag in tags:
        strata = sorted({k[2] for k in data if k[0] == tag})
        rows: list[list[str]] = []
        for mode in ("a", "b", "c"):
            for st in strata:
                r = data.get((tag, mode, st))
                if not r or "error" in r:
                    continue
                m = r.get("model", {})
                ci = m.get("auroc_pooled_ci") or [None, None]
                nb = r.get("null_permutation", {})
                base = r.get("baselines", {})
                rows.append(
                    [
                        TRANSFER_MODE_LABEL.get(mode, mode),
                        st,
                        f3(m.get("auroc_mean")),
                        f"[{f3(ci[0])}, {f3(ci[1])}]",
                        f3(nb.get("ci_upper")),
                        *[f3((base.get(k) or {}).get("auroc")) for k, _ in TRANSFER_BASELINES],
                        str(r.get("verdict", {}).get("label", "—")),
                    ]
                )
        if not rows:
            continue
        out.append(f"**외부 소스 `{tag}`**\n")
        out.append(
            table(
                ["실행", "층", "CNN AUROC", "95% CI", "순열 바닥선(97.5%)",
                 *[label for _, label in TRANSFER_BASELINES], "판정"],
                rows,
            )
        )
    rep = load_motif_replication()
    if rep:
        rows = []
        for (tag, st), d in sorted(rep.items()):
            if "error" in d or "skipped" in d:
                continue
            sp, sm, jc = d["spearman_logodds"], d["sign_match_rate"], d["jaccard_topk"]
            rows.append(
                [
                    tag,
                    st,
                    f"{f3(sp['rho'])} [{f3(sp['ci'][0])}, {f3(sp['ci'][1])}]",
                    f"{f3(sm['value'])} [{f3(sm['ci'][0])}, {f3(sm['ci'][1])}]",
                    f"{f3(jc['value'])} (귀무 상한 {f3(jc['null_ci'][1])})",
                    str(d.get("verdict", {}).get("label", "—")),
                ]
            )
        if rows:
            out.append("**motif 재현성 (설계 4절)**\n")
            out.append(
                table(
                    ["소스", "층", "Spearman ρ (512종)", "부호 일치율 (상위 20)",
                     "Jaccard (상위 20)", "판정"],
                    rows,
                )
            )
    return "\n".join(out)



SECTIONS = [
    ("T2 층별 표본 수와 채택 등급", table_counts),
    ("T3 주 결과", table_main),
    ("H4 공간 구조 검정 (CNN × MLP, 정상 배치 × shuffle-pos)", table_h4),
    ("T4 길이 통제 절제 (H3)", table_length),
    ("T5 표현·마스크·부분집합 절제", table_ablation),
    ("Grad-CAM kind별 CAM 질량 비율", table_cam),
    ("Grad-CAM enrichment와 난수 기준선", table_cam_enrichment),
    ("Bag-of-QR-patches (Q1 직접 측정)", motif_table),
    ("어휘 프로브 (RQ2 · D안)", table_probes),
    ("인과 motif 절제 (RQ3 · C안)", table_occlusion),
    ("가설 검정 판정 (H1~H4)", hypotheses_table),
    ("외부 검증 전이 (F) · motif 재현성", transfer_table),
]


def main() -> None:
    for title, fn in SECTIONS:
        body = fn()
        # 아직 안 돌린 phase는 산출물이 없다. 빈 섹션 제목만 남기지 않고 통째로 건너뛴다.
        if not body:
            continue
        print(f"### {title}\n")
        print(body)
        print()


if __name__ == "__main__":
    main()
