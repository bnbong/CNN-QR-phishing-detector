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
    # within-QR null (review_02 2절): 조성은 보존하고 공간 배치만 깬 대조.
    ("patch 3×3 · within-QR 모듈 셔플 null", "patch3_shuffle_module"),
    ("patch 3×3 · 코드워드 순서 셔플 null", "patch3_shuffle_codeword"),
    ("patch 3×3 · 라벨 셔플 (바닥)", "patch3_labelshuffle"),
]


def load_motifs() -> dict[str, dict]:
    """``reports/motifs/{condition_id}/{stratum}/results.json``. 안 돌렸으면 빈 dict."""
    return _load_phase("motifs", MOTIF_COND)


def motif_cell(motifs: dict[str, dict], stratum: str, key: str) -> str:
    rep = (motifs.get(stratum, {}).get("representations") or {}).get(key)
    if not rep:
        return "—"
    # 표현별 dict 안의 auroc_pooled/auroc_pooled_ci는 시드 층화 클러스터 부트스트랩
    # (results.json 최상위 pooling = seed_stratified)이라 CNN 표와 같은 estimator다.
    # 짝을 이루므로 점추정도 같은 풀링에서 읽어야 CI와 맞는다.
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


def load_campaign() -> dict[str, dict]:
    """``reports/campaign/{stratum}/results.json``. 안 돌렸으면 빈 dict.

    G 재설계(review_02 3절)는 조건 id 하위 디렉터리를 쓰지 않는다 — 조건은 주 조건으로
    고정돼 있고 바뀌는 것은 그룹 키가 아니라 A/B 학습 집합뿐이다.
    """
    out: dict[str, dict] = {}
    root = REPORTS / "campaign"
    if not root.exists():
        return out
    for path in sorted(root.glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "error" in data:
            continue
        out[str(data.get("stratum", path.parent.name))] = data
    return out


def _campaign_delta(block: dict | None) -> str:
    if not block:
        return "—"
    lo, hi = (block.get("delta_ci") or [None, None])[:2]
    return f"{float(block['delta_auroc']):+.3f} [{f3(lo)}, {f3(hi)}]"


def _campaign_mean(camp: dict, stratum: str, path: tuple[str, ...]) -> str:
    """시드별 per_seed 값의 평균. 없으면 '—'."""
    rows = [r for r in camp.get(stratum, {}).get("per_seed", []) if "error" not in r]
    vals: list[float] = []
    for row in rows:
        node: dict | float | None = row
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None and not isinstance(node, dict):
            vals.append(float(node))
    return f"{sum(vals) / len(vals):.0f}" if vals else "—"


def campaign_table() -> str:
    """고정 캠페인 test 집합 위의 쌍체 A/B 비교 (G 재설계).

    주 지표는 ``ΔAUROC = A_sm − B``다. ``A_sm``은 형제 행을 전부 유지한 채 비형제를
    덜어 ``|train| = |train_B|``로 맞춘 A-sizematched이므로, Δ에서 학습 표본 수 효과가
    빠진다. 크기를 맞추지 않은 ``A − B``는 보조 행으로 함께 싣는다. 세 모델이 **같은 T**를
    평가하므로 전부 쌍체다. 옛 template_split의 비쌍체 비교(평가 대상 88% 교체)를 대체한다.

    ``*_leaky_vs_contrast``는 "A의 train에 형제가 있는 누출 행 + 누출 행이 하나도 없는
    클래스의 행 전부(대조)"만 남긴 Δ다. 누출 행만 남기면 사실상 단일 클래스가 되어
    AUROC가 정의되지 않기 때문이며, 표본 구성이 달라 전체 T Δ와 크기를 직접 비교하면 안 된다.
    """
    camp = load_campaign()
    if not camp:
        return ""
    agg = {s: (camp.get(s, {}).get("aggregate") or {}) for s in STRATA}
    rows: list[tuple[str, list[str]]] = [
        ("AUROC — Model A_sm (누출 허용, 크기 매칭)", [f3((agg[s].get("cnn_sm") or {}).get("auroc_A")) for s in STRATA]),
        ("AUROC — Model B (완전 격리)", [f3((agg[s].get("cnn_sm") or {}).get("auroc_B")) for s in STRATA]),
        ("**ΔAUROC (A_sm−B), 쌍체 95% CI** — 주 지표", [_campaign_delta(agg[s].get("cnn_sm")) for s in STRATA]),
        ("순열 p (클러스터, 양측)", [f3((agg[s].get("cnn_sm") or {}).get("perm_p")) for s in STRATA]),
        ("ΔAUROC — 누출 행 대 무누출 클래스 대조 (A_sm−B)",
         [_campaign_delta(agg[s].get("cnn_sm_leaky_vs_contrast")) for s in STRATA]),
        ("AUROC — Model A (크기 미매칭, 보조)", [f3((agg[s].get("cnn") or {}).get("auroc_A")) for s in STRATA]),
        ("ΔAUROC (A−B), 쌍체 95% CI — 보조", [_campaign_delta(agg[s].get("cnn")) for s in STRATA]),
        ("ΔAUROC — 누출 행 대 무누출 클래스 대조 (A−B)",
         [_campaign_delta(agg[s].get("cnn_leaky_vs_contrast")) for s in STRATA]),
        ("Δ char n-gram LR (A_sm−B)", [_campaign_delta(agg[s].get("charngram_lr_sm")) for s in STRATA]),
        ("Δ byte-hist LR (A_sm−B)", [_campaign_delta(agg[s].get("bytehist_lr_sm")) for s in STRATA]),
        ("T 표본 수", [_campaign_mean(camp, s, ("n_test",)) for s in STRATA]),
        ("그중 A train에 형제가 있는 행", [_campaign_mean(camp, s, ("n_test_leaky",)) for s in STRATA]),
        (
            "A train의 T 템플릿 중복 행 수",
            [_campaign_mean(camp, s, ("split", "overlap", "n_template_overlap_in_train", "A"))
             for s in STRATA],
        ),
        (
            "B train의 T 템플릿 중복 행 수",
            [_campaign_mean(camp, s, ("split", "overlap", "n_template_overlap_in_train", "B"))
             for s in STRATA],
        ),
        ("판정 — 주 지표 (A_sm−B)", [_verdict_code(agg[s].get("cnn_sm")) for s in STRATA]),
        ("판정 — 보조 (A−B)", [_verdict_code(agg[s].get("cnn")) for s in STRATA]),
        ("판정 — 누출 대조 (A_sm−B)",
         [_verdict_code(agg[s].get("cnn_sm_leaky_vs_contrast")) for s in STRATA]),
        ("판정 — 누출 대조 (A−B)",
         [_verdict_code(agg[s].get("cnn_leaky_vs_contrast")) for s in STRATA]),
    ]
    return strata_table("지표", rows) + _campaign_verdict_footnote(camp)


def _verdict_code(block: dict | None) -> str:
    """판정 코드. 옛 결과(문자열 verdict)는 그대로 보여준다."""
    v = (block or {}).get("verdict")
    if isinstance(v, dict):
        return f"`{v.get('code', '—')}`"
    return str(v) if v else "—"


def _campaign_verdict_footnote(camp: dict) -> str:
    """판정 정의 각주 — 코드만 봐서는 무엇을 주장하는지 알 수 없다.

    특히 ``negligible``과 ``not_detected``의 차이(등가성 근거 유무)가 review_03 5절의
    핵심이므로, 표 아래에 정의와 SESOI를 항상 함께 싣는다.
    """
    rule: dict = {}
    for s in STRATA:
        r = (camp.get(s, {}).get("aggregate") or {}).get("verdict_rule")
        if isinstance(r, dict):
            rule = r
            break
    if not rule:
        return ""
    lines = [
        "",
        f"판정 정의 (사전 등록). SESOI = **{rule['sesoi']}** AUROC — "
        f"\"실질적으로 무시 가능\"이라고 부를 수 있는 최대 차이. "
        f"결론 수정 임계 **{rule['major_threshold']}**과는 다른 질문의 임계다.",
        "",
    ]
    lines += [f"- `{k}` — {v}" for k, v in (rule.get("definitions") or {}).items()]
    warn = None
    for s in STRATA:
        b = ((camp.get(s, {}).get("aggregate") or {}).get("cnn_sm_leaky_vs_contrast") or {})
        w = (b.get("sample_size_warning") or {}).get("warning")
        if w:
            warn = True
            n = b["sample_size_warning"].get("n_leaky_rows_mean")
            lines.append(f"- 누출 대조 표본 수 ({s}): 시드당 평균 {n:.0f}행")
    if warn:
        lines.append(
            "- 누출 대조 부집합은 표본이 수십 행뿐이라 CI가 넓다. 구조적으로 "
            "`negligible`이 나올 수 없으며, \"phishing kit 암기 가능성을 배제했다\"는 "
            "결론의 근거로 쓸 수 없다."
        )
    return "\n".join(lines) + "\n"

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


# 새 필드가 없을 때 쓰는 표시. 옛 값을 대신 채우지 않는다.
UNAGGREGATED = "미집계"

# 어휘 프로브가 답하는(또는 답하지 못하는) 세 질문. review_03 6절.
PROBE_QUESTION_ROWS = (
    ("① 선형으로 접근 가능한가 (`accessible`)", "accessible"),
    ("② 학습이 접근성을 높였는가 (`learned_gain`)", "learned_gain"),
    ("③ 둘 다 (`accessible_and_gained`, 옛 significant)", "accessible_and_gained"),
)


def table_probes() -> str:
    """어휘 프로브 요약 — 세 질문별 목표 수/전체와 상위 목표.

    옛 표는 단일 "유의" 한 줄이었다. 그 기준은 셔플과 무작위 초기화를 **둘 다** 넘을
    것을 요구하므로 "정보가 있는가"가 아니라 "학습이 접근성을 높였는가"에 가깝다
    (review_03 6절). 이제 두 질문을 나누어 싣고, 답할 수 없는 세 번째 질문은 답할 수
    없다고 적는다.
    """
    probes = _load_phase("probes")
    if not probes:
        return ""
    rows: list[tuple[str, list[str]]] = []
    missing = False
    for label, key in PROBE_QUESTION_ROWS:
        cells = []
        for s in STRATA:
            summ = (probes.get(s) or {}).get("summary")
            blk = (summ or {}).get(key)
            if not summ:
                cells.append("—")
            elif isinstance(blk, dict):
                cells.append(f"{blk['n']} / {summ['n_targets']} ({blk['frac'] * 100:.1f}%)")
            else:
                # 옛 폴백은 여기서 `n_significant`(= 질문 ③의 답)를 세 질문 칸에 모두
                # 넣었다. 그래서 "접근성 2/91"처럼 서로 다른 질문의 답이 같은 수로
                # 보고됐다(review_04). 이제 대신 넣지 않고 미집계로 남긴다.
                cells.append(UNAGGREGATED)
                missing = True
        rows.append((label, cells))
    only = []
    for s in STRATA:
        summ = (probes.get(s) or {}).get("summary")
        blk = (summ or {}).get("accessible_only")
        if isinstance(blk, dict):
            only.append(str(blk["n"]))
        else:
            only.append(UNAGGREGATED if summ else "—")
            missing = missing or bool(summ)
    rows.append(("①만 만족 (②는 아님)", only))
    rows.append(
        ("④ 실제 분류 결정에 쓰이는가 (`used_in_decision`)",
         ["측정 불가" if probes.get(s) else "—" for s in STRATA])
    )
    lab = []
    for s in STRATA:
        d = ((probes.get(s) or {}).get("summary") or {}).get("label_probe")
        lab.append(f3(d["score"]) if d else "—")
    rows.append(("라벨 프로브 (참고 상한선)", lab))
    out = [
        strata_table("어휘 프로브", rows),
        "",
        "① `accessible` = 전 시드에서 학습 표현 프로브 CI 하한 > 셔플 기준선 CI 상한. "
        "② `learned_gain` = 전 시드에서 학습 표현 프로브 CI 하한 > 무작위 초기화 CNN "
        "기준선 CI 상한. ④는 동결 표현 위의 선형 프로브로는 답할 수 없다 — 프로브는 "
        "표현을 읽을 뿐 결정 경로에 개입하지 않는다.",
        "",
    ]
    if missing:
        out.insert(2, "")
        out.insert(
            2,
            f"각주: **{UNAGGREGATED}**는 저장된 프로브 결과에 목표별·시드별 CI가 없어 "
            "'전 시드에서 성립'을 판정할 수 없다는 뜻이다(값이 0이라는 뜻이 아니다). "
            "`reports/probes/{cid}/{stratum}/seed{k}.json` 캐시가 있으면 "
            "`recompute_probe_summaries`로 정확히 재집계되고, 없으면 저장된 CNN "
            "체크포인트에서 프로브만 다시 돌려야 한다(노트북 [11]). "
            "이전 문서의 \"접근성 2/91\"은 이 칸에 질문 ③의 답(옛 `n_significant`)을 "
            "대신 채운 폴백의 산물이며, 접근성의 집계 결과가 아니다.",
        )

    for s in STRATA:
        summ = (probes.get(s) or {}).get("summary")
        if not summ:
            continue
        for title, block in (
            (f"{s} 상위 목표 — ①만 만족(정보는 읽히지만 학습 이득은 미확인)",
             summ.get("accessible_only")),
            (f"{s} 상위 목표 — ③ 둘 다 만족", summ.get("accessible_and_gained")),
        ):
            top = block.get("top10") if isinstance(block, dict) else None
            if not top:
                continue
            out.append(f"\n**{title}**\n")
            out.append(
                table(
                    ["목표", "지표", "점수", "셔플", "무작위 초기화"],
                    [
                        [t["target"], t["metric"], f3(t["score"]), f3(t["shuffle"]),
                         f3(t["random_init"])]
                        for t in top
                    ],
                )
            )
    return "\n".join(out)


OCC_CONDS = [
    ("phishing motif", "phishing_motif"),
    ("random (phishing 개수 맞춤)", "random"),
    ("benign motif", "benign_motif"),
    ("random (benign 개수 맞춤)", "random_benign_matched"),
    # topology-matched 대조(review_02 4절): 개수 + 사분면 + 국소 흑색 밀도까지 맞춘다.
    ("random (phishing topology 맞춤)", "random_matched"),
    ("random (benign topology 맞춤)", "random_matched_benign"),
]

OCC_CONTRASTS = [
    ("phishing motif − random(개수)", "phishing_motif_vs_random"),
    ("phishing motif − random(topology)", "phishing_motif_vs_random_matched"),
    ("benign motif − random(개수)", "benign_motif_vs_random"),
    ("benign motif − random(topology)", "benign_motif_vs_random_matched"),
]


def _ci(pair) -> str:
    """``[lo, hi]`` -> ``"[-0.120, -0.031]"``. 값이 없으면 em dash."""
    if not pair or any(v is None for v in pair):
        return "—"
    try:
        lo, hi = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return "—"
    if lo != lo or hi != hi:
        return "—"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def table_occlusion() -> str:
    """인과 절제 — 절대 ΔAUROC와 시드 층화 CI, 그리고 ΔΔ(표적 − 무작위).

    "무작위 대비 몇 배" 같은 비율은 싣지 않는다. 무작위 대조의 하락폭이 0 근처라 비율이
    불안정하다(review_03 3절). 시드별 CI 경계 평균도 통합 효과의 95% CI로 읽히므로 뺐고,
    시드 층화 클러스터 부트스트랩으로 만든 통합 CI 하나만 싣는다.
    """
    occ = _load_phase("occlusion")
    if not occ:
        return ""
    out = []
    for s in STRATA:
        agg = (occ.get(s) or {}).get("aggregate")
        if not agg:
            continue
        pooled = agg.get("d_auroc_pooled") or {}
        rows = []
        for label, key in OCC_CONDS:
            b = agg.get(key)
            if not b:
                continue
            pb = pooled.get(key) or {}
            rows.append(
                [
                    label,
                    f"{float(b['d_auroc']):+.3f}",
                    _ci(pb.get("ci")),
                    f3(b.get("d_auroc_sd_across_seeds")),
                    (f3(b["d_auroc_sd_across_reps"]) if "d_auroc_sd_across_reps" in b
                     else "—"),
                    f"{float(b['mean_logit_delta']):+.3f}",
                    f"{float(b['mean_logit_delta_phishing']):+.3f}",
                    f"{float(b['mean_logit_delta_benign']):+.3f}",
                    f"{float(b['n_flipped_mean']):.1f}",
                ]
            )
        head = (
            f"\n**{s}** (원본 AUROC {f3(agg.get('auroc_original'))})\n\n"
            + table(
                ["조건", "ΔAUROC", "95% CI(시드 층화)", "시드 간 SD", "반복 간 SD",
                 "로짓 Δ", "로짓 Δ(phishing)", "로짓 Δ(benign)", "개입한 모듈 수"],
                rows,
            )
        )

        dd = agg.get("delta_vs_random") or {}
        dd_seed = agg.get("contrasts") or {}
        dd_rows = []
        for label, key in OCC_CONTRASTS:
            blk = dd.get(key)
            if not blk:
                continue
            per_seed_p = (dd_seed.get(key) or {}).get("perm_p_per_seed") or []
            dd_rows.append(
                [
                    label,
                    f"{float(blk['estimate']):+.3f}",
                    _ci(blk.get("ci")),
                    f3((dd_seed.get(key) or {}).get("sd_across_seeds")),
                    (
                        "/".join(f"{float(v):.3f}" for v in per_seed_p)
                        if per_seed_p else "—"
                    ),
                ]
            )
        if dd_rows:
            head += (
                "\n\nΔΔAUROC = 표적 개입 AUROC − 무작위 개입 AUROC(같은 test, 반복 평균). "
                "음수면 표적 개입이 더 크게 무너뜨렸다는 뜻이다.\n\n"
                + table(
                    ["대비", "ΔΔAUROC", "95% CI(시드 층화)", "시드 간 SD", "순열 p(시드별)"],
                    dd_rows,
                )
            )
        ov = agg.get("motif_top_overlap")
        if ov:
            head += (
                f"\n\nmotif 목록은 시드별 train에서만 골랐다 — 시드 간 top 목록 Jaccard "
                f"평균 {f3(ov.get('jaccard_mean'))} (최소 {f3(ov.get('jaccard_min'))}), "
                f"모든 시드에 공통 {ov.get('n_in_all_seeds')}개 / 합집합 "
                f"{ov.get('n_in_any_seed')}개.\n"
            )
        out.append(head)
    return "\n".join(out)


H_LABEL = {
    "H1": "CNN > 라벨 셔플 바닥",
    "H2": "디코딩 텍스트 참조 기준 > CNN",
    "H3": "L-none > L-exact",
    "H4": "정상 배치 > shuffle-pos",
}

# 각 가설의 귀무가설. "기각"만 적으면 무엇을 기각했는지 알 수 없어서 판정 문구에
# 그대로 넣는다(review_04 "통계 표 정리").
H_NULL = {
    "H1": "H0: ΔAUROC ≤ 0 (CNN이 라벨 셔플 바닥보다 높지 않다)",
    "H2": "H0: ΔAUROC ≤ 0 (참조 기준이 CNN보다 높지 않다)",
    "H3": "H0: ΔAUROC ≤ 0 (L-none이 L-exact보다 높지 않다)",
    "H4": "H0: ΔAUROC ≤ 0 (정상 배치가 shuffle-pos보다 유리하지 않다)",
}

# 정식 검정이 있는 가설과 기술 추정만 있는 가설. 정식 = 영가설 분포를 실제로 만든
# 클러스터 순열 검정(H1·H4)뿐이다. H2는 쌍체 효과 추정치와 CI만, H3는 평가 표본이
# 달라 비쌍체 추정치와 CI만 싣는다.
H_FORMAL = ("H1", "H4")


def _p(value: float) -> str:
    """순열 p / pseudo-p 표기. 2000회 재표집 해상도(1/2001) 아래는 부등호로 쓴다."""
    v = float(value)
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def _est_ci(t: dict) -> tuple[str, str]:
    lo, hi = t["ci"][:2]
    return f"{float(t['estimate']):+.3f}", f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def _verdict(h: str, t: dict) -> str:
    """귀무가설을 명시한 판정 문구. '기각' 한 단어로 뭉뚱그리지 않는다."""
    ci = t.get("ci")
    excludes = bool(ci and (float(ci[0]) > 0 or float(ci[1]) < 0))
    if h in H_FORMAL:
        pp = t.get("perm_p")
        if pp is None:
            return f"{H_NULL[h]} — 순열 검정 미산출"
        verdict = "우세 근거 있음" if float(pp) < 0.05 else "우세 근거 부족"
        return f"{H_NULL[h]} — 순열 p {_p(pp)}, {verdict}"
    if h == "H2":
        base = "참조 기준이 더 높다는 근거" if excludes else "참조 기준 우세 근거 부족"
        return f"검정 없음 — 쌍체 ΔAUROC의 95% CI가 0을 {'배제' if excludes else '포함'}, {base}"
    tail = "0을 배제" if excludes else "0을 포함"
    return f"검정 없음 — 비쌍체 ΔAUROC의 95% CI가 {tail} (쌍체 해석 불가)"


def _hyp_tests() -> dict[tuple[str, str], dict]:
    path = REPORTS / "hypotheses.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(t["hypothesis"], t["stratum"]): t for t in data.get("tests", [])}


def hypotheses_table() -> str:
    """H1 ~ H4 정식 판정 표.

    이전 표는 네 가설을 한 줄 규격으로 찍어내면서 `pseudo-p (Holm)`을 함께 싣고 판정을
    "기각"으로만 적었다. pseudo-p는 부트스트랩 백분위 CI를 역전시켜 정의한 값이지 영가설
    분포에서 나온 정식 p값이 아니므로(``hypotheses.json``의 ``p_definition``), 정식 검정
    표에 같이 실으면 "정식 p값이 아니다"라고 적어 둔 한계와 "Holm 보정 후에도 기각"이라는
    결론이 한 표 안에서 충돌한다(review_04). 이제 이 표에는 근거의 종류를 열로 밝히고,
    pseudo-p는 :func:`pseudo_p_table` 부록으로 뺀다.

    * H1·H4 — 클러스터 순열 검정의 정식 p값이 중심. 쌍체 ΔAUROC와 CI를 함께 싣는다.
    * H2 — 정식 검정이 없다. 쌍체 효과 추정치와 CI만으로 "참조 기준이 더 높다는 근거"를
      서술한다.
    * H3 — L-none과 L-exact는 평가 표본이 다르다. 비쌍체 추정치와 CI만 싣는다.

    ``reports/hypotheses.json``은 시드 층화 클러스터 부트스트랩(``pooling:
    seed_stratified``)으로 만든 값이다. 시드별로 AUROC를 계산해 평균하므로 시드 간 점수
    척도 차이가 통합 추정치를 끌고 가지 않는다.
    """
    tests = _hyp_tests()
    if not tests:
        return ""
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        for st in STRATA:
            t = tests.get((h, st))
            if t is None or "ci" not in t:
                continue
            est, ci = _est_ci(t)
            if h in H_FORMAL:
                basis = "정식 (클러스터 순열)"
                pcell = _p(t["perm_p"]) if t.get("perm_p") is not None else "—"
            elif h == "H2":
                basis = "기술 추정 (쌍체 부트스트랩)"
                pcell = "—"
            else:
                basis = "기술 추정 (비쌍체 부트스트랩)"
                pcell = "—"
            rows.append([f"{h}. {H_LABEL[h]}", st, basis, est, ci, pcell, _verdict(h, t)])
    if not rows:
        return ""
    out = [
        table(
            ["가설", "층", "근거 종류", "ΔAUROC 추정치", "95% CI", "순열 p", "판정"],
            rows,
        ),
        "",
        "정식 가설 검정은 쌍체 예측이 있는 H1·H4의 클러스터 순열 검정뿐이다. 순열 null은 "
        "같은 group(도메인 클러스터) 안의 행을 한 블록으로 묶어 두 조건 라벨을 교환해 만든 "
        "**정의한 그룹 블록 null 기준**이며, 블록 단위 교환 가능성을 가정한다 — 반복 수를 "
        "늘리면 몬테카를로 오차만 줄고 그 가정이 검증되지는 않는다.",
        "",
        'H2에는 정식 검정을 붙이지 않았다. 쌍체 ΔAUROC와 95% CI를 "참조 기준이 더 높다는 '
        '근거"로만 읽는다.',
        "",
        "H3의 L-none과 L-exact는 길이 매칭 여부가 달라 **평가 표본 자체가 다르다**. "
        "쌍체 ΔAUROC로 해석할 수 없으므로 비쌍체 추정치와 CI만 싣는다.",
        "",
        "부트스트랩 CI는 저장된 모델들의 예측에 조건부인 평가 불확실성이다. 학습 데이터와 "
        "학습 과정 전체의 불확실성까지 포함하지 않는다.",
    ]
    return "\n".join(out)


def pseudo_p_table() -> str:
    """부록 — pseudo-p와 Holm 보정값.

    정식 표에서 뺀 값을 투명하게 남기기 위한 부록이다. pseudo-p는 부트스트랩 백분위 CI를
    역전시켜 정의한 수치(가장 작은 alpha에서 CI가 0을 배제)이지 영가설 분포에서 나온 p값이
    아니다. 가설 채택·기각의 근거로 쓰지 않는다.
    """
    tests = _hyp_tests()
    if not tests:
        return ""
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        for st in STRATA:
            t = tests.get((h, st))
            if t is None or t.get("pseudo_p") is None:
                continue
            pp = t["pseudo_p"]
            if isinstance(pp, float) and pp != pp:  # NaN — Holm family에서 뺀 항목
                continue
            holm = t.get("pseudo_p_holm")
            rows.append(
                [
                    f"{h}. {H_LABEL[h]}",
                    st,
                    _p(pp),
                    _p(holm) if holm is not None else "—",
                    "예" if t.get("ci_excludes_null") else "아니오",
                ]
            )
    if not rows:
        return ""
    return "\n".join(
        [
            table(["가설", "층", "pseudo-p", "pseudo-p (Holm)", "CI가 0을 배제"], rows),
            "",
            "pseudo-p는 부트스트랩 백분위 CI를 역전시켜 정의한 값이지 영가설 분포에서 나온 "
            '정식 p값이 아니다. Holm 보정도 그 값에 건 것이므로 "보정 후에도 유의"라는 '
            "표현은 쓰지 않는다. 정식 판정은 위 가설 검정 표를 본다.",
        ]
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
    ("path_lr", "path-shape LR"),
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


def load_motif_replication() -> dict[tuple[str, str, str], dict]:
    """``reports/transfer/{tag}/motif_replication*/{stratum}/replication.json``.

    ``motif_replication``(시드별 cohort)과 ``motif_replication_fixed``(고정 cohort)는
    같은 (tag, stratum)이라도 평가 대상이 다르므로 cohort를 키에 넣어 따로 저장한다
    (안 그러면 나중에 훑은 쪽이 먼저 것을 덮어쓴다).
    """
    out: dict[tuple[str, str, str], dict] = {}
    root = REPORTS / "transfer"
    if not root.exists():
        return out
    for path in sorted(root.glob("*/motif_replication*/*/replication.json")):
        tag = path.parent.parent.parent.name
        run_dir = path.parent.parent.name
        cohort = "fixed" if run_dir == "motif_replication_fixed" else "per_seed"
        try:
            out[(tag, cohort, path.parent.name)] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            continue
    return out


SET_ORDER = (
    ("primary_ccunranked", "primary · CC unranked benign(민감도)"),
    ("ccunranked", "primary · CC unranked benign(민감도)"),
    ("keep_phish_domains", "primary · benign 정제 절제: 피싱 도메인 유지"),
    ("no_hosting_blocklist", "primary · benign 정제 절제: 호스팅 블록리스트 미적용"),
    ("secondary", "secondary · Phishing.Database(robustness)"),
    ("b2", "EXT-B2 · Tranco 맨 도메인 대조군"),
    ("primary", "primary · OpenPhish 90일 × CC×Tranco benign"),
)


def set_label(tag: str, res: dict) -> str:
    """외부 세트 이름을 태그(와 결과의 ``benign_cleaning``)에서 읽는다.

    수집 CLI가 내는 파일명 규약(``external_{set}_{date}.csv``)이 그대로 source_tag로
    넘어온다는 전제다. 못 맞추면 태그를 그대로 쓴다.
    """
    t = str(tag).lower()
    cleaning = (res.get("data") or {}).get("benign_cleaning")
    for key, label in SET_ORDER:
        if key in t:
            return label
    if cleaning and cleaning != "clean":
        return f"primary · benign 정제 절제: {cleaning}"
    return str(tag)


def _set_rank(label: str) -> int:
    for i, (_, lab) in enumerate(SET_ORDER):
        if lab == label:
            return i
    return len(SET_ORDER)


def _flag_cell(r: dict) -> str:
    """편향 플래그 요약. `주 판정 아님`은 길이만 맞춘 cohort라 판정을 보류했다는 뜻이다."""
    f = (r.get("verdict") or {}).get("flags") or {}
    bits = []
    if f.get("common_direction_bias"):
        bits.append("공통방향편향")
    if f.get("path_shortcut_dominant"):
        bits.append("경로지름길")
    if (r.get("verdict") or {}).get("is_primary_verdict") is False:
        bits.append("주 판정 아님")
    return " · ".join(bits) if bits else "—"


def _delta_cell(d: dict | None) -> str:
    """쌍체 ΔAUROC 셀. 계산하지 못한 기준선은 대시로 남긴다."""
    if not d or d.get("delta") is None:
        return "—"
    ci = d.get("ci") or [None, None]
    return f"{f3(d['delta'])} [{f3(ci[0])}, {f3(ci[1])}]"


def transfer_table() -> str:
    """F-a/F-b/F-c × 외부 세트 × 층 × {CNN, 텍스트 기준선, 순열 바닥선} (설계 6절).

    세트는 primary(OpenPhish) / secondary(Phishing.Database) / CC unranked benign /
    benign 정제 절제 3조건으로 나눠 적는다. F-a는 평가 cohort(fixed·per_seed)도 함께
    적는다 — 고정 cohort가 주 결과, 시드별 cohort는 F-b와의 쌍체 비교용이다.

    바닥선은 그룹 블록 교환 순열의 97.5 백분위다. CNN AUROC의 CI 하한이 이 값 이하면
    "우연과 구분 불가"(collapse)다. 참조용 행 단위 라벨 순열 바닥선도 함께 적는다.

    표본 수 열은 리뷰 03 항목 8 때문에 있다. CNN과 텍스트 기준선은 평가 cohort 전체를 보지만
    motif-hist 기준선은 캡이 걸리면 부분집합만 본다 — 두 번째 표에 기준선별 n을 적는다.
    """
    data = load_transfer()
    if not data:
        return ""
    out: list[str] = []
    rows: list[list[str]] = []
    n_rows: list[list[str]] = []
    for (tag, mode, st), r in data.items():
        if not r or "error" in r:
            continue
        m = r.get("model", {})
        ci = m.get("auroc_pooled_ci") or [None, None]
        nb = r.get("null_permutation", {})
        base = r.get("baselines", {})
        base_mode = mode.split("_")[0]
        run = TRANSFER_MODE_LABEL.get(base_mode, mode)
        if base_mode == "a":
            run += " (고정 cohort)" if mode.startswith("a_fixed") else " (시드별 cohort)"
            if mode.endswith("_lenpath"):
                run += " · 길이+경로 매칭"
        rows.append(
            [
                set_label(tag, r),
                run,
                st,
                f3(m.get("auroc_mean")),
                f"[{f3(ci[0])}, {f3(ci[1])}]",
                f3(nb.get("ci_upper")),
                f3((r.get("null_row_permutation") or {}).get("ci_upper")),
                str(nb.get("n_perm", "—")),
                f"{m.get('n_boot', '—')}",
                str((r.get("data") or {}).get("n_total", "—")),
                _flag_cell(r),
                *[f3((base.get(k) or {}).get("auroc")) for k, _ in TRANSFER_BASELINES],
                str(r.get("verdict", {}).get("label", "—")),
            ]
        )
        for k, label in TRANSFER_BASELINES:
            b = base.get(k) or {}
            if not b:
                continue
            n_rows.append(
                [
                    set_label(tag, r),
                    run,
                    st,
                    label,
                    str(b.get("n", "—")),
                    str(b.get("n_fit", "—")),
                    str(b.get("n_seeds", "—")),
                    "예" if b.get("eval_rows_same_as_cnn", True) else "아니오",
                    _delta_cell((r.get("comparison_to_baselines") or {}).get(k)),
                ]
            )
    if rows:
        rows.sort(key=lambda r: (_set_rank(r[0]), r[1], r[2]))
        out.append(
            table(
                ["외부 세트", "실행", "층", "CNN AUROC", "95% CI", "순열 바닥선(97.5%)",
                 "행순열 바닥선(97.5%)", "n_perm", "n_boot", "평가 n", "플래그",
                 *[label for _, label in TRANSFER_BASELINES], "판정"],
                rows,
            )
        )
    if n_rows:
        n_rows.sort(key=lambda r: (_set_rank(r[0]), r[1], r[2], r[3]))
        out.append(
            "\n**기준선 표본 수와 CNN 대비 쌍체 ΔAUROC** "
            "(Δ = CNN − 기준선, 시드 층화 그룹 클러스터 부트스트랩)\n"
        )
        out.append(
            table(
                ["외부 세트", "실행", "층", "기준선", "평가 n", "fit n", "시드 수",
                 "CNN과 같은 행", "ΔAUROC [95% CI]"],
                n_rows,
            )
        )
    rep = load_motif_replication()
    if rep:
        COHORT_LABEL = {"fixed": "고정", "per_seed": "시드별"}

        def _rep_row_values(d: dict) -> tuple:
            sp, sm, jc = d["spearman_logodds"], d["sign_match_rate"], d["jaccard_topk"]
            return (
                f"{f3(sp['rho'])} [{f3(sp['ci'][0])}, {f3(sp['ci'][1])}]",
                f"{f3(sm['value'])} [{f3(sm['ci'][0])}, {f3(sm['ci'][1])}]",
                f"{f3(jc['value'])} (귀무 상한 {f3(jc['null_ci'][1])})",
                str(d.get("verdict", {}).get("label", "—")),
            )

        by_tag_st: dict[tuple[str, str], dict[str, dict]] = {}
        for (tag, cohort, st), d in rep.items():
            if "error" in d or "skipped" in d:
                continue
            by_tag_st.setdefault((tag, st), {})[cohort] = d

        rows = []
        for (tag, st), by_cohort in sorted(by_tag_st.items()):
            values = {c: _rep_row_values(d) for c, d in by_cohort.items()}
            if len(values) == 2 and values.get("fixed") == values.get("per_seed"):
                cohort_label = "고정(=시드별 동일)"
                rows.append([tag, cohort_label, st, *values["fixed"]])
            else:
                for cohort in ("fixed", "per_seed"):
                    if cohort in values:
                        rows.append(
                            [tag, COHORT_LABEL[cohort], st, *values[cohort]]
                        )
        if rows:
            out.append("**motif 재현성 (설계 4절)**\n")
            out.append(
                table(
                    ["소스", "cohort", "층", "Spearman ρ (512종)", "부호 일치율 (상위 20)",
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
    ("가설 검정 판정 (H1 ~ H4)", hypotheses_table),
    ("부록 · pseudo-p와 Holm 보정 (정식 검정 아님)", pseudo_p_table),
    ("외부 검증 전이 (F) · motif 재현성", transfer_table),
    ("템플릿 누출 쌍체 비교 (G 재설계 · 고정 캠페인 T)", campaign_table),
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
