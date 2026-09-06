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


def f3(value: str | float | None) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.3f}"


def cell(condition: str, stratum: str) -> str:
    """조건×층 AUROC와 클러스터 부트스트랩 95% CI."""
    row = MAIN.get((condition, stratum))
    if row is None:
        return "—"
    return f"{f3(row['auroc_mean'])} [{f3(row['auroc_ci_lo'])}, {f3(row['auroc_ci_hi'])}]"


def baseline_cell(condition: str, stratum: str, baseline: str) -> str:
    row = BASELINES.get((condition, stratum, baseline))
    if row is None:
        return "—"
    return f3(row["auroc"])


def auroc(condition: str, stratum: str) -> float | None:
    row = MAIN.get((condition, stratum))
    return float(row["auroc_mean"]) if row else None


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
            "char n-gram LR (상한선)",
            [baseline_cell(BASE, s, "charngram_lr") for s in STRATA],
        ),
        ("byte-hist LR", [baseline_cell(BASE, s, "bytehist_lr") for s in STRATA]),
        ("length LR", [baseline_cell(BASE, s, "length_lr") for s in STRATA]),
        ("version LR", [baseline_cell(BASE, s, "version_lr") for s in STRATA]),
    ]
    gaps = []
    for s in STRATA:
        upper = BASELINES.get((BASE, s, "charngram_lr"))
        cnn = auroc(BASE, s)
        gaps.append(
            "—" if upper is None or cnn is None else f"{float(upper['auroc']) - cnn:+.3f}"
        )
    rows.append(("상한선 − CNN 갭", gaps))
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
            "path+ 상한선 (char n-gram LR)",
            [baseline_cell(f"{BASE}-pathplus", s, "charngram_lr") for s in STRATA],
        ),
        ("EC-M (v3 한정)", [cell(f"{BASE}-ecm", s) for s in STRATA]),
        (
            "EC-M 상한선 (char n-gram LR)",
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


def table_cam() -> str:
    explain = json.loads((REPORTS / "explain_summary.json").read_text(encoding="utf-8"))
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


SECTIONS = [
    ("T2 층별 표본 수와 채택 등급", table_counts),
    ("T3 주 결과", table_main),
    ("H4 공간 구조 검정 (CNN × MLP, 정상 배치 × shuffle-pos)", table_h4),
    ("T4 길이 통제 절제 (H3)", table_length),
    ("T5 표현·마스크·부분집합 절제", table_ablation),
    ("Grad-CAM kind별 CAM 질량 비율", table_cam),
]


def main() -> None:
    for title, fn in SECTIONS:
        print(f"### {title}\n")
        print(fn())
        print()


if __name__ == "__main__":
    main()
