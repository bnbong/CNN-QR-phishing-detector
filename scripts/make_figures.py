"""reports/의 집계 CSV·JSON에서 논문·보고서용 그림(F1~F7)을 생성한다.

수치는 전부 ``reports/`` 아래에서 읽는다. 이 스크립트에 실험 결과 수치를
직접 적어 두지 않는다. 그림은 ``reports/figures/``에 PNG·SVG로 함께 쓰고,
캡션 문구는 같은 디렉터리의 ``captions.md``로 낸다.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
FIGURES = REPORTS / "figures"

STRATA = ["v2", "v3", "v4"]
BASE = "norm-exact-data_only-fixed-small_cnn"
# motif 러너는 모델을 학습하지 않으므로 arch가 붙지 않은 조건 id를 쓴다.
MOTIF_COND = "norm-exact-data_only-fixed"
TRANSFER_SOURCE = "external_primary_2026-09-08"

# 검증된 기본 카테고리 팔레트(light). 인접 쌍 CVD·정상시야 게이트를 통과한
# 순서를 그대로 쓴다. 슬롯을 재정렬하거나 색을 갈아끼우지 않는다.
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_MAGENTA = "#e87ba4"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#87857c"
GRID = "#e5e4df"
# 바닥선·참조선은 계열 색이 아니라 중성 회색으로 둔다.
NEUTRAL = "#9c9a90"

CI_KW = {"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 3.0}

# ---------------------------------------------------------------------------
# 한글 폰트: 시스템에서 찾고, 없으면 영문 라벨로 폴백한다.
# ---------------------------------------------------------------------------

KOREAN_FONT_CANDIDATES = [
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Malgun Gothic",
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Source Han Sans KR",
    "UnDotum",
]

KOREAN_FONT_FILES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def setup_font() -> bool:
    """한글 폰트를 찾아 matplotlib에 등록한다. 성공하면 True."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    for path in KOREAN_FONT_FILES:
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
            except (RuntimeError, OSError):
                continue
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_KOREAN = False


def t(ko: str, en: str) -> str:
    """한글 폰트가 있으면 한글, 없으면 영문 라벨."""
    return ko if HAS_KOREAN else en


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def index_main() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["condition_id"], row["stratum"]): row for row in load_csv(REPORTS / "table_main.csv")
    }


def index_baselines() -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["condition_id"], row["stratum"], row["baseline"]): row
        for row in load_csv(REPORTS / "table_baselines.csv")
    }


MAIN = index_main()
BASELINES = index_baselines()


def num(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def main_point(condition: str, stratum: str) -> tuple[float, float, float] | None:
    """조건×층의 (AUROC, CI 하한, CI 상한)."""
    row = MAIN.get((condition, stratum))
    if row is None:
        return None
    point = num(row.get("auroc_mean")) or num(row.get("auroc_pooled"))
    lo = num(row.get("auroc_ci_lo"))
    hi = num(row.get("auroc_ci_hi"))
    if point is None or lo is None or hi is None:
        return None
    return point, lo, hi


def baseline_point(condition: str, stratum: str, baseline: str) -> float | None:
    row = BASELINES.get((condition, stratum, baseline))
    return None if row is None else num(row["auroc"])


# ---------------------------------------------------------------------------
# 공통 축 스타일
# ---------------------------------------------------------------------------


def style_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3, width=0.8)
    ax.grid(
        axis="x" if xgrid else "y",
        color=GRID,
        linewidth=0.8,
        zorder=0,
    )
    ax.set_axisbelow(True)


def new_figure(*args: Any, **kwargs: Any) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def legend(
    fig: plt.Figure,
    handles: list[Any],
    labels: list[str],
    ncol: int,
    *,
    y: float = -0.10,
) -> None:
    """축 라벨 아래에 범례를 둔다. 겹치지 않도록 y를 직접 내려 잡는다."""
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, y),
    )
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)


def save(fig: plt.Figure, name: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "svg"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
            facecolor=SURFACE,
        )
        written.append(path)
    plt.close(fig)
    return written


def title(ax: plt.Axes, text: str, *, size: int = 10) -> None:
    ax.set_title(text, color=INK, fontsize=size, pad=8, loc="left")


# ---------------------------------------------------------------------------
# F1 · 주 결과
# ---------------------------------------------------------------------------


def figure_f1() -> str:
    series = [
        (t("SmallCNN", "SmallCNN"), C_BLUE, "cnn"),
        (t("BitMLP", "BitMLP"), C_ORANGE, "mlp"),
        (t("char n-gram LR (참조)", "char n-gram LR (ref.)"), C_AQUA, "charngram_lr"),
        (t("byte-hist LR", "byte-hist LR"), C_YELLOW, "bytehist_lr"),
        (t("라벨 셔플 (바닥)", "label shuffle (floor)"), NEUTRAL, "shuffle"),
    ]

    def value(kind: str, stratum: str) -> tuple[float, float, float] | None:
        if kind == "cnn":
            return main_point(BASE, stratum)
        if kind == "mlp":
            return main_point("norm-exact-data_only-fixed-bit_mlp", stratum)
        if kind == "shuffle":
            return main_point(f"{BASE}-labelshuffle", stratum)
        point = baseline_point(BASE, stratum, kind)
        return None if point is None else (point, point, point)

    fig, ax = new_figure(figsize=(8.6, 4.0))
    style_axes(ax)
    width = 0.16
    handles = []
    for i, (label, color, kind) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        for j, stratum in enumerate(STRATA):
            got = value(kind, stratum)
            if got is None:
                continue
            point, lo, hi = got
            err = [[point - lo], [hi - point]]
            bar = ax.bar(
                j + offset,
                point,
                width * 0.88,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
                label=label if j == 0 else None,
            )
            if hi > lo:
                ax.errorbar(j + offset, point, yerr=err, fmt="none", zorder=4, **CI_KW)
            ax.annotate(
                f"{point:.3f}",
                (j + offset, hi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=INK_SECONDARY,
                rotation=90,
                zorder=5,
            )
            if j == 0:
                handles.append(bar[0])
    ax.axhline(0.5, color=NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_ylim(0.0, 1.16)
    ax.set_ylabel("AUROC", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(t("QR 버전 층", "QR version stratum"), color=INK_SECONDARY, fontsize=9)
    title(
        ax,
        t(
            "길이·버전 통제 뒤의 층별 AUROC",
            "AUROC by stratum under length/version control",
        ),
    )
    legend(fig, handles, [s[0] for s in series], 5)
    save(fig, "F1")
    return t(
        "F1. 길이·버전을 통제한 주 조건에서 층별(v2/v3/v4) AUROC. 막대는 시드 층화 "
        "풀링 추정값, 오차막대는 95% CI다. char n-gram LR과 byte-hist LR은 단일 "
        "적합값이라 CI를 싣지 않는다. 파선은 라벨 셔플 바닥이 놓이는 0.50이다.",
        "F1. AUROC by stratum (v2/v3/v4) in the length/version-controlled main "
        "condition. Bars are seed-stratified pooled estimates with 95% CI error "
        "bars; the two LR references are single fits without CI. The dashed line "
        "marks 0.50.",
    )


# ---------------------------------------------------------------------------
# F2 · H4 (정상 배치 vs 위치 셔플)
# ---------------------------------------------------------------------------


def figure_f2() -> str:
    models = [
        (t("SmallCNN", "SmallCNN"), BASE, f"{BASE}-shufflepos"),
        (
            t("BitMLP", "BitMLP"),
            "norm-exact-data_only-fixed-bit_mlp",
            "norm-exact-data_only-fixed-bit_mlp-shufflepos",
        ),
    ]
    arms = [
        (t("정상 배치", "original layout"), C_BLUE),
        (t("위치 셔플", "position shuffle"), C_ORANGE),
    ]

    fig, axes = new_figure(1, len(STRATA), figsize=(8.6, 3.6), sharey=True)
    handles = []
    for ax, stratum in zip(axes, STRATA, strict=True):
        style_axes(ax)
        for mi, (_model_label, cond, cond_shuf) in enumerate(models):
            for ai, (_arm_label, color) in enumerate(arms):
                got = main_point(cond if ai == 0 else cond_shuf, stratum)
                if got is None:
                    continue
                point, lo, hi = got
                x = mi + (ai - 0.5) * 0.34
                bar = ax.bar(
                    x,
                    point,
                    0.3,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=1.0,
                    zorder=3,
                )
                ax.errorbar(
                    x,
                    point,
                    yerr=[[point - lo], [hi - point]],
                    fmt="none",
                    zorder=4,
                    **CI_KW,
                )
                ax.annotate(
                    f"{point:.3f}",
                    (x, hi),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=INK_SECONDARY,
                    zorder=5,
                )
                if stratum == STRATA[0] and mi == 0:
                    handles.append(bar[0])
        ax.axhline(0.5, color=NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[0] for m in models])
        ax.set_ylim(0.0, 1.06)
        title(ax, stratum, size=9)
    axes[0].set_ylabel("AUROC", color=INK_SECONDARY, fontsize=9)
    fig.suptitle(
        t(
            "정상 모듈 배치와 위치 셔플의 대비 (H4)",
            "Original module layout vs. position shuffle (H4)",
        ),
        color=INK,
        fontsize=10,
        x=0.02,
        y=1.04,
        ha="left",
    )
    legend(fig, handles, [a[0] for a in arms], 2, y=-0.02)
    save(fig, "F2")
    return t(
        "F2. 모델(SmallCNN·BitMLP) × 배치(정상·위치 셔플)의 2×2 대비를 층별로 나눈 "
        "그림. 막대는 시드 층화 풀링 AUROC, 오차막대는 95% CI다. CNN에서만 셔플 시 "
        "성능이 크게 떨어진다.",
        "F2. The 2x2 contrast of model (SmallCNN / BitMLP) by layout (original / "
        "position shuffle), panelled by stratum. Bars are seed-stratified pooled "
        "AUROC with 95% CI.",
    )


# ---------------------------------------------------------------------------
# F3 · within-QR null
# ---------------------------------------------------------------------------


def figure_f3() -> str:
    reps = [
        ("patch3", t("patch3 (실제)", "patch3 (actual)"), C_BLUE),
        ("patch3_shuffle_module", t("모듈 셔플", "module shuffle"), C_ORANGE),
        ("patch3_shuffle_codeword", t("코드워드 셔플", "codeword shuffle"), C_AQUA),
        ("patch3_labelshuffle", t("라벨 셔플", "label shuffle"), NEUTRAL),
        ("pyramid3", "pyramid3", C_MAGENTA),
    ]
    data = {
        stratum: load_json(REPORTS / "motifs" / MOTIF_COND / stratum / "results.json")
        for stratum in STRATA
    }

    fig, ax = new_figure(figsize=(8.6, 4.0))
    style_axes(ax)
    width = 0.16
    handles = []
    for i, (key, _label, color) in enumerate(reps):
        offset = (i - (len(reps) - 1) / 2) * width
        for j, stratum in enumerate(STRATA):
            rep = data[stratum]["representations"].get(key)
            if rep is None:
                continue
            point = rep["auroc_pooled"]
            lo, hi = rep["auroc_pooled_ci"]
            bar = ax.bar(
                j + offset,
                point,
                width * 0.88,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
            )
            ax.errorbar(
                j + offset,
                point,
                yerr=[[point - lo], [hi - point]],
                fmt="none",
                zorder=4,
                **CI_KW,
            )
            ax.annotate(
                f"{point:.3f}",
                (j + offset, hi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=INK_SECONDARY,
                rotation=90,
                zorder=5,
            )
            if j == 0:
                handles.append(bar[0])
    ax.axhline(0.5, color=NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("AUROC", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(t("QR 버전 층", "QR version stratum"), color=INK_SECONDARY, fontsize=9)
    title(
        ax,
        t(
            "Bag-of-QR-patches와 within-QR null",
            "Bag-of-QR-patches and within-QR nulls",
        ),
    )
    legend(fig, handles, [r[1] for r in reps], 5)
    save(fig, "F3")
    return t(
        "F3. 패치 가방 표현의 층별 AUROC와 within-QR null. 모듈 셔플과 코드워드 "
        "셔플은 격자 안에서 패턴을 흩뜨린 대조, 라벨 셔플은 바닥선이다. 막대는 시드 "
        "층화 풀링 추정값, 오차막대는 95% CI, 파선은 0.50이다.",
        "F3. AUROC of bag-of-patches representations by stratum with within-QR "
        "nulls. Module and codeword shuffles scramble patterns inside the grid; "
        "label shuffle is the floor. Bars are seed-stratified pooled estimates "
        "with 95% CI; the dashed line marks 0.50.",
    )


# ---------------------------------------------------------------------------
# F4 · motif 개입 ΔΔAUROC
# ---------------------------------------------------------------------------


def figure_f4() -> str:
    contrasts = [
        ("phishing_motif_vs_random", t("피싱 motif - 무작위", "phishing motif - random")),
        (
            "phishing_motif_vs_random_matched",
            t("피싱 motif - 무작위(매칭)", "phishing motif - random (matched)"),
        ),
        ("benign_motif_vs_random", t("정상 motif - 무작위", "benign motif - random")),
        (
            "benign_motif_vs_random_matched",
            t("정상 motif - 무작위(매칭)", "benign motif - random (matched)"),
        ),
    ]
    data = {
        stratum: load_json(REPORTS / "occlusion" / BASE / stratum / "results.json")["aggregate"][
            "delta_vs_random"
        ]
        for stratum in STRATA
    }

    fig, axes = new_figure(1, len(STRATA), figsize=(9.4, 3.2), sharex=True, sharey=True)
    for ax, stratum in zip(axes, STRATA, strict=True):
        style_axes(ax, xgrid=True)
        ax.axvline(0.0, color=INK_SECONDARY, linewidth=1.0, zorder=2)
        for i, (key, _label) in enumerate(contrasts):
            block = data[stratum].get(key)
            if block is None:
                continue
            point = block["estimate"]
            lo, hi = block["ci"]
            y = len(contrasts) - 1 - i
            ax.errorbar(
                point,
                y,
                xerr=[[point - lo], [hi - point]],
                fmt="o",
                markersize=5,
                color=C_BLUE,
                markerfacecolor=C_BLUE,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                ecolor=C_BLUE,
                elinewidth=1.0,
                capsize=3.0,
                zorder=4,
            )
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels([c[1] for c in reversed(contrasts)], fontsize=8)
        ax.set_ylim(-0.6, len(contrasts) - 0.4)
        if stratum != STRATA[0]:
            # 축을 공유하므로 두 번째 패널부터는 y 눈금 표시를 지운다.
            ax.tick_params(axis="y", length=0)
        title(ax, stratum, size=9)
        ax.set_xlabel("ΔΔAUROC", color=INK_SECONDARY, fontsize=9)
    fig.suptitle(
        t(
            "motif 개입의 표적 - 무작위 대비 (ΔΔAUROC)",
            "Motif intervention: target minus random (ΔΔAUROC)",
        ),
        color=INK,
        fontsize=10,
        x=0.02,
        y=1.04,
        ha="left",
    )
    save(fig, "F4")
    return t(
        "F4. motif 폐색 개입에서 표적 motif와 무작위 대조의 ΔAUROC 차이(ΔΔAUROC)를 "
        "네 대비로 나눈 그림. 점은 시드 층화 풀링 추정값, 가로선은 95% CI, 세로 "
        "기준선은 0이다. 대부분의 대비에서 CI가 0을 포함하고, 0을 배제하는 경우도 "
        "추정값이 음수라 표적 motif 폐색이 무작위 대조보다 성능을 더 떨어뜨린다는 "
        "방향의 근거는 없다.",
        "F4. Difference in ΔAUROC between targeted motif occlusion and its random "
        "control (ΔΔAUROC) across four contrasts. Points are seed-stratified "
        "pooled estimates with 95% CI; the vertical reference line is zero.",
    )


# ---------------------------------------------------------------------------
# F5 · 외부 전이
# ---------------------------------------------------------------------------


def figure_f5() -> str:
    modes = [
        ("a_fixed", t("CNN · 길이 매칭", "CNN - length match"), C_BLUE),
        (
            "a_fixed_lenpath",
            t("CNN · 길이×경로 매칭", "CNN - length x path match"),
            C_ORANGE,
        ),
    ]
    refs = [
        ("charngram_lr", t("char n-gram LR", "char n-gram LR"), C_AQUA, "o"),
        ("bytehist_lr", t("byte-hist LR", "byte-hist LR"), C_YELLOW, "s"),
        ("path_lr", t("path-shape LR", "path-shape LR"), C_MAGENTA, "^"),
    ]

    def read(mode: str, stratum: str) -> dict[str, Any] | None:
        path = REPORTS / "transfer" / TRANSFER_SOURCE / mode / BASE / stratum / "results.json"
        if not path.exists():
            return None
        payload = load_json(path)
        if not payload.get("model"):
            return None
        return payload

    fig, ax = new_figure(figsize=(9.0, 4.2))
    style_axes(ax)
    width = 0.3
    handles: list[Any] = []
    labels: list[str] = []
    null_handle = None
    for i, (mode, label, color) in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        for j, stratum in enumerate(STRATA):
            payload = read(mode, stratum)
            if payload is None:
                continue
            model = payload["model"]
            point = model["auroc_pooled"]
            lo, hi = model["auroc_pooled_ci"]
            bar = ax.bar(
                j + offset,
                point,
                width * 0.82,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
            )
            ax.errorbar(
                j + offset,
                point,
                yerr=[[point - lo], [hi - point]],
                fmt="none",
                zorder=4,
                **CI_KW,
            )
            half = width * 0.44
            null_upper = payload["null_permutation"]["ci_upper"]
            # 값 라벨은 이 슬롯에서 가장 높은 요소 위에 둔다 (마커와 겹치지 않게).
            slot_top = max(
                [hi]
                + [
                    payload["baselines"][key]["auroc"]
                    for key, *_ in refs
                    if key in payload["baselines"]
                ]
            )
            ax.annotate(
                f"{point:.3f}",
                (j + offset, slot_top),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK_SECONDARY,
                zorder=6,
            )
            (null_handle,) = ax.plot(
                [j + offset - half, j + offset + half],
                [null_upper, null_upper],
                color=INK_SECONDARY,
                linewidth=1.6,
                linestyle=(0, (3, 2)),
                zorder=5,
            )
            spread = width * 0.5
            for ri, (key, _ref_label, ref_color, marker) in enumerate(refs):
                base = payload["baselines"].get(key)
                if base is None:
                    continue
                ax.plot(
                    j + offset + (ri - (len(refs) - 1) / 2) * spread / len(refs),
                    base["auroc"],
                    marker=marker,
                    markersize=7,
                    color=ref_color,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    linestyle="none",
                    zorder=6,
                )
            if j == 0:
                handles.append(bar[0])
                labels.append(label)
    for _key, ref_label, ref_color, marker in refs:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=marker,
                markersize=7,
                color=ref_color,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                linestyle="none",
            )
        )
        labels.append(ref_label)
    if null_handle is not None:
        handles.append(null_handle)
        labels.append(t("순열 바닥선 97.5%", "permutation floor 97.5%"))
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("AUROC", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(t("QR 버전 층", "QR version stratum"), color=INK_SECONDARY, fontsize=9)
    title(
        ax,
        t(
            "외부 세트 전이 (primary)",
            "Transfer to the external set (primary)",
        ),
    )
    legend(fig, handles, labels, 3, y=-0.09)
    save(fig, "F5")
    return t(
        "F5. primary 외부 세트로의 전이 성능. 막대는 길이만 매칭한 cohort와 길이·경로를 "
        "함께 매칭한 cohort의 CNN AUROC(95% CI)이고, 점 표시는 같은 cohort에서 적합한 "
        "텍스트·바이트·경로 기준선이다. 짧은 파선은 그룹 블록 교환 순열 바닥선의 "
        "97.5% 분위다.",
        "F5. Transfer to the primary external set. Bars give CNN AUROC (95% CI) "
        "for the length-matched and the length-plus-path-matched cohorts; markers "
        "give the text, byte and path-shape baselines fit on the same cohort. The "
        "short dashes mark the 97.5th percentile of the group-block permutation "
        "floor.",
    )


# ---------------------------------------------------------------------------
# F6 · 캠페인 누출
# ---------------------------------------------------------------------------

SESOI = 0.02


def figure_f6() -> str:
    fig, ax = new_figure(figsize=(7.0, 3.6))
    style_axes(ax)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.0, zorder=2)
    for sign in (1, -1):
        ax.axhline(
            sign * SESOI,
            color=NEUTRAL,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
    for j, stratum in enumerate(STRATA):
        path = REPORTS / "campaign" / stratum / "results.json"
        if not path.exists():
            continue
        block = load_json(path)["aggregate"]["cnn_sm"]
        point = block["delta_auroc"]
        lo, hi = block["delta_ci"]
        ax.errorbar(
            j,
            point,
            yerr=[[point - lo], [hi - point]],
            fmt="o",
            markersize=6,
            color=C_BLUE,
            markerfacecolor=C_BLUE,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            ecolor=C_BLUE,
            elinewidth=1.0,
            capsize=3.0,
            zorder=4,
        )
        ax.annotate(
            f"{point:+.3f}",
            (j, hi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
            zorder=5,
        )
    ax.annotate(
        t("SESOI ±0.02", "SESOI +/-0.02"),
        (len(STRATA) - 0.5, SESOI),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=INK_MUTED,
    )
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_xlim(-0.5, len(STRATA) - 0.5)
    ax.set_ylabel(t("ΔAUROC (A_sm - B)", "ΔAUROC (A_sm - B)"), color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(t("QR 버전 층", "QR version stratum"), color=INK_SECONDARY, fontsize=9)
    title(
        ax,
        t(
            "캠페인·템플릿 누출: A_sm - B",
            "Campaign/template leakage: A_sm - B",
        ),
    )
    save(fig, "F6")
    return t(
        "F6. 캠페인·템플릿 누출 대비의 층별 ΔAUROC(A_sm − B)와 95% CI. 파선은 사전 "
        "지정한 최소 관심 효과크기(SESOI) ±0.02다. 세 층 모두 CI가 0을 포함한다.",
        "F6. ΔAUROC (A_sm - B) with 95% CI for the campaign/template leakage "
        "contrast, by stratum. Dashed lines mark the pre-specified smallest effect "
        "size of interest (+/-0.02).",
    )


# ---------------------------------------------------------------------------
# F7 · 어휘 프로브
# ---------------------------------------------------------------------------


def figure_f7() -> str:
    questions = [
        ("accessible", t("접근 가능", "accessible"), C_BLUE),
        ("learned_gain", t("학습 이득", "learned gain"), C_ORANGE),
        ("accessible_and_gained", t("둘 다", "both"), C_AQUA),
    ]

    fig, ax = new_figure(figsize=(7.4, 3.8))
    style_axes(ax)
    width = 0.24
    handles = []
    for i, (key, _label, color) in enumerate(questions):
        offset = (i - (len(questions) - 1) / 2) * width
        for j, stratum in enumerate(STRATA):
            path = REPORTS / "probes" / BASE / stratum / "results.json"
            if not path.exists():
                continue
            summary = load_json(path)["summary"]
            frac = summary[key]["frac"]
            bar = ax.bar(
                j + offset,
                frac,
                width * 0.88,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
            )
            ax.annotate(
                f"{frac:.3f}",
                (j + offset, frac),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK_SECONDARY,
                zorder=5,
            )
            if j == 0:
                handles.append(bar[0])
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_ylim(0.0, 0.32)
    ax.set_ylabel(
        t("통과한 목표 비율", "fraction of targets passing"),
        color=INK_SECONDARY,
        fontsize=9,
        labelpad=8,
    )
    ax.set_xlabel(t("QR 버전 층", "QR version stratum"), color=INK_SECONDARY, fontsize=9)
    title(
        ax,
        t(
            "어휘 프로브 세 질문의 통과 비율",
            "Pass rate of the three lexical-probe questions",
        ),
    )
    legend(fig, handles, [q[1] for q in questions], 3, y=-0.04)
    save(fig, "F7")
    return t(
        "F7. 어휘 프로브가 던지는 세 질문(표현에서 선형 접근 가능한가, 피싱 학습이 그 "
        "접근성을 높였는가, 둘 다인가)을 통과한 목표의 비율. 목표 수는 층마다 다르므로 "
        "절대 개수가 아니라 비율로 그린다. 세 번째 질문인 used_in_decision은 이 "
        "실험으로 답할 수 없어 그림에 넣지 않았다.",
        "F7. Fraction of probe targets passing each of the three questions "
        "(linearly accessible in the representation; accessibility gained through "
        "phishing training; both). Target counts differ by stratum, so the plot "
        "shows fractions rather than counts.",
    )


# ---------------------------------------------------------------------------


def main() -> None:
    global HAS_KOREAN
    HAS_KOREAN = setup_font()
    if not HAS_KOREAN:
        print("한글 폰트를 찾지 못했다. 영문 라벨로 폴백한다.")

    builders = [
        ("F1", figure_f1),
        ("F2", figure_f2),
        ("F3", figure_f3),
        ("F4", figure_f4),
        ("F5", figure_f5),
        ("F6", figure_f6),
        ("F7", figure_f7),
    ]
    captions = []
    for name, build in builders:
        caption = build()
        captions.append((name, caption))
        print(f"wrote reports/figures/{name}.png, reports/figures/{name}.svg")

    lines = [
        "# 그림 캡션",
        "",
        "`scripts/make_figures.py`가 생성한다. 직접 고치지 말고 스크립트를 고쳐라.",
        "",
    ]
    for name, caption in captions:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(caption)
        lines.append("")
    (FIGURES / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/figures/captions.md")


if __name__ == "__main__":
    main()
