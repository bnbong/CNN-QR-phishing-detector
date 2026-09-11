# RUNBOOK

이 문서는 README에 있던 운영 절차(로컬 셋업, 데이터 준비, Colab 실행 순서, 재실행 규칙, 결과 반영)를 옮겨놓은 것이다. 실험 설계, 연구 질문, 결과 해석은 다루지 않는다. 해당 내용은 `docs/EXPERIMENT_DESIGN.md`, `docs/REPORT.md`, `docs/RESULTS.md`를 본다.

## 1. 로컬 셋업, 테스트

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check qrphish tests scripts
```

데이터는 `uv run python scripts/convert_webphish.py`로 `backups/data/URL.xlsx` → `data/webphish.csv`를 만들어 쓴다.

## 2. 데이터 준비 (Drive 업로드 파일 목록)

Colab에서 쓰는 데이터는 로컬에서 만든 CSV를 Google Drive에 미리 올려둔 것이다. 노트북은 수집, 변환을 하지 않고 경로만 지정한다.

Drive 경로 규약은 `/content/drive/MyDrive/qrphish/<파일명>`이다. 올려야 할 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `qrphish/webphish.csv` | 주 학습 데이터(WebPhish) |
| `qrphish/external_primary_2026-09-08.csv` | 외부 검증 primary 세트(OpenPhish 최근 90일 + CC×Tranco benign) |
| `qrphish/external_secondary_…csv` | 외부 검증 secondary 세트(Phishing.Database 아카이브 + 같은 benign 행) |
| `qrphish/external_primary_ccunranked_…csv` | Tranco 조인을 뺀 CC benign - popularity confound 민감도 |
| `qrphish/external_primary_keep_phish_domains_…csv` | benign 정제 절제 세트 |
| `qrphish/external_primary_no_hosting_blocklist_…csv` | benign 정제 절제 세트 |

외부 검증용 CSV는 로컬에서 `scripts/collect_external.py`로 만든다(네트워크, 약관, 시간 문제로 Colab에서 직접 수집하지 않는다).

## 3. Colab 셀 순서 표

`notebooks/colab_run.ipynb`는 아래 순서를 셀로 옮긴 것이고 로직을 담지 않는다. 번호 순서대로 돌리면 되고 각 러너는 `results.json`이 있는 조합을 건너뛰므로 세션이 끊겨도 이어서 돌릴 수 있다.

| 셀 | 내용 | GPU | 비고 |
|---|---|---|---|
| `[1]` ~ `[4]` | 환경, Drive 마운트, 설정 | - | `QRPHISH_REF` 커밋 고정 포함 |
| `[5]` | P0 - 층 카운트, 패딩 진단, 라운드트립 게이트 | - | 여기서 primary 층 확정 |
| `[6]` `[7]` | P1 주 실험, P2 절제 | 필요 | 가장 오래 걸린다 |
| `[8]` | 재집계 + H1 ~ H4 검정 | - | |
| `[9]` `[10]` | motif(Bag-of-QR-patches), 인과 절제 | 일부 | motif 집계는 CPU, 인과 절제의 CNN 추론은 GPU 사용 가능 |
| `[11]` `[12]` | 어휘 프로브, Grad-CAM | 필요 | 저장된 체크포인트만 읽는다 |
| `[13]` | 외부 검증 세트 지정 + 편향 진단(플래그 방식, 차단 아님) | - | primary는 항상 실행 목록에 포함 |
| `[14]` | F - 외부 검증 전이 평가 | 일부 | F-a 고정 평가 집합은 추론만이라 GPU 없이도 된다 |
| `[15]` | G - 캠페인 단위 홀드아웃 + 표 생성 | 필요 | 층당 5시드 × 2모델 학습 |

## 4. 재실행 규칙

집계 규약과 null 조건이 바뀌었으므로 이전 산출물이 Drive에 남아 있으면 아래를 다시 실행한다.

| 셀 | 왜 | 어떻게 |
|---|---|---|
| `[8]` | 시드 층화 집계, `pseudo_p` 키 정리 | 그냥 다시 실행한다(`reaggregate`가 덮어쓴다) |
| `[9]` | within-QR 셔플 null(모듈 및 코드워드) 조건 추가 | Drive의 `qrphish_out_v2/reports/motifs` 삭제 후 실행 |
| `[10]` | `[9]`의 `enrichment.json`이 바뀌므로 함께 갱신 | Drive의 `qrphish_out_v2/reports/occlusion` 삭제 후 실행 |
| `[14]` | 게이트, 매칭 조건이 바뀌었으므로 이전 전이 결과가 무효 | Drive의 `qrphish_out_v2/reports/transfer` 삭제 후 실행 |

`[9]` `[10]` `[14]`는 성공한 `results.json`을 건너뛰기 때문에 **Drive의 해당 산출물을 먼저 지워야** 새로 계산된다. `[8]`은 저장된 예측에서 다시 계산하는 구조라 그냥 재실행하면 된다.

## 5. 커밋 고정 (`QRPHISH_REF`)

논문 재현용 실행은 커밋을 고정한다. `main`은 계속 움직이므로 같은 노트북을 다시 돌려도 다른 코드가 설치될 수 있다.

셀 `[1]`에서 `QRPHISH_REF`에 이 실행에 쓸 커밋 해시를 채운다. 저장소에서 다음으로 확인한다.

```bash
git rev-parse HEAD
```

`"main"`으로 두면 예전처럼 최신 main을 쓴다(재현 보장 없음 - 노트북이 경고를 찍는다). 셀 `[4]`(config 로드)에서도 **셀 `[1]`에서 설치한 것과 같은 ref**를 체크아웃해야 한다. 설치 패키지와 `configs/`가 어긋나면 조건 id는 같은데 내용이 다른 실행이 된다.

## 6. 결과 반영 절차

1. Colab 실행이 끝나면 Drive의 `qrphish_out_v2/reports/`를 로컬 저장소의 `reports/`로 복사한다.
2. 표를 만든다.

```bash
uv run python scripts/make_results_tables.py > reports/tables.md
```

3. 생성된 마크다운 표를 `docs/RESULTS.md`(전체 표)와 `docs/REPORT.md`(독립 결과 보고서)에 반영한다.

Colab 노트북 안에서 바로 만들려면 셀 `[15]`의 마지막 코드 셀에서 다음을 쓴다.

```bash
!python scripts/make_results_tables.py > {OUT_DIR}/reports/tables.md
!tail -40 {OUT_DIR}/reports/tables.md
```

## 7. 사전 점검 스크립트 사용법

`scripts/preflight_colab.py`는 Colab에 올리기 전에 로컬에서 파이프라인 전체(서브샘플)를 돌려보는 스크립트다. `OUT_DIR`을 Drive 폴더 흉내로 받아 실제 실행과 같은 구조의 산출물을 만든다.

```bash
uv run python scripts/preflight_colab.py --out-dir <출력 폴더> [옵션들]
```

주요 옵션은 다음과 같다.

| 옵션 | 의미 |
|---|---|
| `--out-dir` | 출력 폴더(Drive 폴더 흉내). 필수 |
| `--data-csv` | 주 학습 CSV. 미지정 시 서브샘플을 만든다 |
| `--source-csv` | 서브샘플의 원본 CSV |
| `--per-class` | 클래스별 서브샘플 크기 |
| `--sample-seed` | 서브샘플링 시드 |
| `--max-epochs` | 최대 epoch 수 |
| `--n-bootstrap` | 부트스트랩 반복 수 |
| `--seeds` | 사용할 시드 목록 |
| `--external-csv` | 외부 검증(F) 셀에 주입할 CSV. 미지정이면 F 셀이 스스로 건너뛴다 |
| `--external-tag` | 외부 소스 태그 |
| `--external-per-class` | 외부 CSV도 클래스(label)별 이 수만큼으로 줄여 주입한다. 실제 세트는 60만 행이라 축소하지 않으면 사전 점검이 몇 시간짜리가 된다. 미지정이면 원본 그대로 쓴다 |
| `--external-extra` | 보조 외부 세트. `이름=경로`를 쉼표로 잇는다(예: `secondary=/x/a.csv,ccunranked=/x/b.csv`). `--external-per-class`가 있으면 각각 같은 방식으로 줄여 주입한다 |
| `--report` | `preflight_report.md` 경로 |
| `--notebook` | 대조할 노트북 경로 |

전체 옵션 목록은 다음으로 확인한다.

```bash
uv run python scripts/preflight_colab.py --help
```

## 8. 논문 초안 PDF 빌드

`docs/PAPER_DRAFT.md`를 PDF로 만든다. 툴체인은 `brew install pandoc tectonic`으로 한 번만 깔면 된다(tectonic은 필요한 LaTeX 패키지를 첫 실행 때 알아서 내려받으므로 MacTeX 전체 설치는 필요 없다).

```bash
./scripts/build_paper_pdf.sh   # -> build/paper_draft.pdf
```

한글 본문은 Apple SD Gothic Neo, 라틴 문자는 Times New Roman이 맡는다. 폰트를 바꾸려면 `PAPER_CJKFONT`와 `PAPER_MAINFONT` 환경 변수를 넘긴다. 원본 마크다운은 건드리지 않고 `build/` 안의 복사본으로만 빌드한다.
