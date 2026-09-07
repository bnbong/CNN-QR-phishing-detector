"""``qrphish`` 커맨드라인 엔트리포인트."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="QR 모듈 격자 피싱 분류 실험 러너")

_CONFIG = typer.Option("configs/base.yaml", "--config", "-c", help="config yaml 경로")


def _load(config: str):
    from qrphish.config import load_config

    return load_config(Path(config))


@app.command()
def p0(config: str = _CONFIG) -> None:
    """P0: 층 카운트 표, 패딩 경계 그림, 분할 진단, 라운드트립 게이트."""
    from qrphish.runner import run_p0

    res = run_p0(_load(config))
    gate = res["roundtrip_gate"]
    typer.echo(json.dumps(res["stratum_counts"], ensure_ascii=False, indent=2, default=str))
    typer.echo(f"roundtrip gate: passed={gate['passed']} checked={gate['n_checked']}")
    if not gate["passed"]:
        raise typer.Exit(code=1)


@app.command()
def run(
    phase: str = typer.Argument("P1", help="P1 | P2"),
    config: str = _CONFIG,
    matrix: str = typer.Option("configs/matrix.yaml", "--matrix", "-m"),
) -> None:
    """매트릭스 실행. results.json이 있는 조합은 건너뛴다."""
    from qrphish.runner import run_matrix

    res = run_matrix(_load(config), phase, matrix)
    typer.echo(f"{len(res)} condition x stratum 완료")


@app.command()
def explain(
    condition_id: str = typer.Argument(..., help="설명할 조건 id"),
    config: str = _CONFIG,
    stratum: str = typer.Option(None, "--stratum", "-s"),
) -> None:
    """P3: Grad-CAM → kind별 CAM 질량 + 문자 위치 기여도."""
    from qrphish.runner import run_explain

    res = run_explain(_load(config), condition_id, stratum)
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:4000])


@app.command()
def report(config: str = _CONFIG) -> None:
    """모든 results.json → reports/table_*.csv."""
    from qrphish.runner import aggregate_reports

    for k, p in aggregate_reports(_load(config)).items():
        typer.echo(f"{k}: {p}")


@app.command()
def motifs(
    config: str = _CONFIG,
    stratum: str = typer.Option(None, "--stratum", "-s", help="지정하면 그 층만"),
) -> None:
    """Bag-of-QR-patches (review_01 B). 학습 없이 CPU에서 돈다."""
    from qrphish.runner import run_motifs

    res = run_motifs(_load(config), [stratum] if stratum else None)
    for st, r in res.items():
        reps = r.get("representations", {})
        typer.echo(f"{st}: " + ", ".join(f"{k}={v['auroc_pooled']:.3f}" for k, v in reps.items()))


@app.command()
def occlusion(
    config: str = _CONFIG,
    stratum: str = typer.Option(None, "--stratum", "-s"),
    top_k: int = typer.Option(10, "--top-k"),
) -> None:
    """인과 motif 절제 (review_01 C). run_motifs 결과가 먼저 있어야 한다."""
    from qrphish.runner import run_occlusion

    res = run_occlusion(_load(config), [stratum] if stratum else None, top_k=top_k)
    for st, r in res.items():
        agg = r.get("aggregate", {})
        typer.echo(
            f"{st}: 원본 {agg.get('auroc_original'):.3f} / "
            + ", ".join(
                f"{c}={agg[c]['d_auroc']:+.3f}"
                for c in ("phishing_motif", "benign_motif", "random")
                if c in agg
            )
        )


@app.command()
def probes(
    config: str = _CONFIG,
    stratum: str = typer.Option(None, "--stratum", "-s"),
) -> None:
    """어휘 프로브 (review_01 D). 저장된 체크포인트만 읽는다."""
    from qrphish.runner import run_probes

    res = run_probes(_load(config), [stratum] if stratum else None)
    for st, r in res.items():
        summ = r.get("summary", {})
        typer.echo(f"{st}: 유의 {summ.get('n_significant')} / {summ.get('n_targets')}")


@app.command()
def hypotheses(
    config: str = _CONFIG,
    matrix: str = typer.Option("configs/matrix.yaml", "--matrix", "-m"),
) -> None:
    """H1~H4 부트스트랩 검정 + Holm 보정 → reports/hypotheses.json."""
    from qrphish.runner import run_hypothesis_tests

    res = run_hypothesis_tests(_load(config), matrix_path=matrix)
    for t in res["tests"]:
        if "error" in t:
            typer.echo(f"{t['hypothesis']} {t['stratum']}: {t['error']}")
            continue
        typer.echo(
            f"{t['hypothesis']} {t['stratum']}: est={t['estimate']:+.3f} "
            f"p_holm={t.get('p_holm')} reject={t.get('reject')}"
        )
    typer.echo(f"-> {res['path']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
