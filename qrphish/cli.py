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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
