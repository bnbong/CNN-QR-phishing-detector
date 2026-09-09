# CNN-QR-phishing-detector

QR 코드 무늬만 보고 피싱 사이트를 알아챌 수 있을까?

## 무엇을 왜 했나

피싱 사이트의 주소(URL)를 QR 코드로 만들면, 그 흑백 무늬만 보고 피싱인지 아닌지 알아챌 수 있을까요. 이 저장소는 그 질문에서 출발했습니다. QR 코드는 URL을 손실 없이 그대로 담고 있으니 "맞힐 수 있는가"는 사실 당연히 참입니다. 코드를 읽어서 주소 텍스트를 분류하면 되니까요. 진짜 궁금한 것은 그 다음입니다. 무늬 자체에 피싱만의 공통된 특징이 있을까요. 처음 돌린 실험은 정확도가 아주 높게 나왔지만 믿을 수 없었습니다. 피싱 주소가 정상 주소보다 대체로 길고, 주소가 길어지면 QR 코드의 크기와 여백 채우기 위치가 달라집니다. 모델이 무늬가 아니라 주소 길이만 보고 맞혔을 수도 있습니다. 그래서 길이 같은 우회 단서를 하나씩 막아 놓고 처음부터 다시 검증했습니다.

## 쉽게 이해하는 핵심 결론

**길이만 보고 맞힌 것은 아닙니다.** QR 코드 크기별로 집단을 나누고, 각 집단에서 피싱과 정상의 주소 길이 분포를 1바이트 단위까지 똑같이 맞춘 뒤에도 성능이 남습니다. 길이 통제를 완전히 풀었을 때와의 차이는 0.005 ~ 0.024에 그칩니다.

**모델은 무늬의 배치를 실제로 이용합니다.** 칸의 값은 그대로 두고 위치만 뒤섞으면 CNN 성능이 0.11 ~ 0.15 떨어지는데, 위치를 보지 않는 대조 모델은 거의 변하지 않습니다. "어떤 칸이 검은가"가 아니라 "어디에 있는가"를 실제로 쓴다는 말입니다.

**그러나 피싱만의 보편적인 무늬가 있다고는 말할 수 없습니다.** 학습에 쓴 데이터와 완전히 다른 출처의 2026년 데이터에 그대로 걸면 성능이 0.649 / 0.639 / 0.677로 떨어집니다. 우연 수준(0.53 ~ 0.56)보다는 높으니 신호가 사라진 것은 아니지만, 같은 조건에서 주소 텍스트를 그냥 읽는 방식이 더 잘 옮겨 갑니다.

여기서 쓰는 성능 지표는 AUROC입니다. 1.0이면 완벽, 0.5면 동전 던지기와 같습니다. 숫자 세 개가 나란히 있으면 QR 코드 크기 세 집단(v2 / v3 / v4)의 값입니다.

## 무엇을 했나

1. 공개된 피싱·정상 URL 데이터를 모아 주소 표기를 통일하고 중복을 지웁니다.
2. 각 주소를 QR 코드로 만들되 그림 파일이 아니라 흑백 칸 격자 그대로 모델에 넣습니다.
3. 우회 단서를 막습니다. QR 코드 크기별로 집단을 나누고 그 안에서 두 클래스의 주소 길이를 같게 맞추며, 같은 도메인이 학습과 평가에 동시에 들어가지 않게 분리합니다.
4. 대조 실험을 돌립니다. 칸 위치를 뒤섞거나 정답 라벨을 뒤섞었을 때 성능이 어떻게 되는지 봅니다.
5. 작은 무늬 조각(3×3)의 빈도만으로도 구분이 되는지 확인하고, 그 조각을 일부러 망가뜨렸을 때 모델 판단이 바뀌는지 개입 실험을 합니다.
6. 마지막으로 전혀 다른 출처의 최신 데이터에 학습된 모델을 그대로 걸어 봅니다.

## 결과 표

| 질문 | 답 |
|---|---|
| 우회 단서를 다 막아도 QR 무늬에 구분 신호가 남는가 | 남습니다. 주 조건 AUROC 0.872 / 0.814 / 0.805 |
| CNN이 무늬의 배치를 이용하는가 | 이용합니다. 위치를 섞으면 0.765 / 0.671 / 0.656으로 떨어집니다 |
| 그 신호가 다른 데이터에서도 재현되는가 | 약합니다. 외부 데이터 전이 0.649 / 0.639 / 0.677 |

## 이 연구가 말하지 않는 것

- 인쇄하거나 카메라로 찍은 실제 QR 사진 이야기가 아닙니다. 입력은 완벽하게 정렬된 합성 격자입니다.
- 주소를 소문자로 바꾸고 끝의 `/`를 떼는 등 표기를 통일한 상태를 다룹니다. 이 정리 자체가 결과에 영향을 줍니다.
- 학습은 사실상 한 데이터셋에 기댑니다. 여러 출처를 아우른 결론이 아닙니다.
- 특정 작은 무늬가 모델 판단의 원인이라는 이전 결론은 절차를 바로잡은 재실험에서 지지되지 않아 철회했습니다.

## 더 자세히

- [`docs/REPORT.md`](docs/REPORT.md) — 결과 보고서(논문 초안의 뼈대)
- [`docs/RESULTS.md`](docs/RESULTS.md) — 전체 표와 해석
- [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md), [`docs/EXTERNAL_VALIDATION_DESIGN.md`](docs/EXTERNAL_VALIDATION_DESIGN.md) — 설계 스펙
- [`docs/review_01.md`](docs/review_01.md), [`docs/review_02.md`](docs/review_02.md), [`docs/review_03.md`](docs/review_03.md) — 외부 리뷰와 그에 따른 수정 이력

## 저장소 사용법

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check qrphish tests scripts
```

데이터는 `uv run python scripts/convert_webphish.py`로 `backups/data/URL.xlsx` → `data/webphish.csv`를 만들어 씁니다.

Colab 실행 순서는 아래와 같고, `notebooks/colab_run.ipynb`가 이 순서를 셀로 옮겨 놓은 것입니다. P0이 첫 산출물이며 여기서 어떤 집단이 주 분석 대상으로 살아남는지 확정합니다. 그 표가 없으면 `run`은 실행을 거부합니다.

```bash
uv run qrphish p0 --config configs/base.yaml       # 집단 카운트 + 누출 진단 + 역매핑 게이트
uv run qrphish run P1 --config configs/base.yaml   # 주 실험
uv run qrphish run P2 --config configs/base.yaml   # 절제 실험
uv run qrphish explain <condition_id> --config configs/base.yaml
uv run qrphish report --config configs/base.yaml
uv run qrphish hypotheses --config configs/base.yaml
uv run qrphish motifs --config configs/base.yaml     # 작은 무늬 조각 분석
uv run qrphish occlusion --config configs/base.yaml  # 개입 실험
uv run qrphish probes --config configs/base.yaml
uv run qrphish transfer --config configs/base.yaml   # 외부 데이터 전이 평가
```

캠페인 단위 홀드아웃에는 아직 CLI 진입점이 없어 `qrphish.runner.run_campaign_holdout(cfg)`를 직접 호출합니다(노트북 `[15]`).

저장소 구조는 `qrphish/`(패키지 본체), `configs/`, `scripts/`, `tests/`, `docs/`, `notebooks/`, `reports/`(커밋 대상 집계), `artifacts/`(gitignore), `backups/`(이전 실험 보존본)입니다.

## 데이터와 라이선스

학습·평가에는 WebPhish 데이터셋(Opara et al., 2024)을 씁니다. Kaggle 배포본 `guchiopara/look-before-you-leap`의 `Category`/`Data` 컬럼이 원본이며 `spam`을 피싱, `ham`을 정상으로 매핑합니다. 외부 검증에는 OpenPhish 공개 피드(피싱)와 Common Crawl × Tranco 조인(정상)으로 만든 2026년 세트를 씁니다. 코드는 MIT 라이선스이며 각 데이터셋은 원 배포처의 이용 조건을 따릅니다.
