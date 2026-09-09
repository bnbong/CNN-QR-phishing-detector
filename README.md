# CNN-QR-phishing-detector

> [!IMPORTANT]
> 실험: QR 코드 무늬만 보고 피싱 사이트를 알아챌 수 있을까?

## 실험 개요

피싱 사이트의 주소(URL)를 QR 코드로 만들면, 그 흑백 무늬만 보고 피싱인지 아닌지 알아챌 수 있을까? 

이 실험은 그 질문에서 출발했다. QR 코드는 URL을 손실 없이 그대로 담고 있으니 "맞힐 수 있는가"는 사실 당연히 참인데, 코드를 읽어서 주소 텍스트를 분류하면 되기 때문이다. 

진짜 궁금했던 것은 그 다음이다: *무늬 자체에 피싱만의 공통된 특징이 있을까?*

이 실험은 2025년 처음 구상했다. 당시 첫 실험은 정확도가 아주 높게 나왔지만 다음 이유들로 오류라고 판정했다: 
- 피싱 주소가 정상 주소보다 대체로 길었고
- 주소가 길어지면 QR 코드의 크기와 여백 채우기 위치가 달라졌다. 
- 모델이 무늬가 아니라 주소 길이만 보고 맞혔을 가능성도 높았다.
  
그래서 2026년 현재, 길이 같은 우회 단서를 하나씩 막아 놓고 처음부터 다시 검증했다.

## 모델에는 무엇을 넣었나

QR 코드는 흑백 칸(모듈)이 바둑판처럼 놓인 격자이며, 사진이나 PNG의 픽셀은 그 격자를 화면에 그린 것이다. 

예를 들어 버전 2 QR은 25 × 25 격자이다. 이것이 QR "무늬"의 원본이고, 이 실험이 보려는 것은 이 무늬에 집중했다. QR 코드에 대한 그 외 설명이 필요하면 [여기 참고](https://en.wikipedia.org/wiki/QR_code)

실험 최적화를 위해 모델에는 픽셀 이미지가 아니라 칸 하나가 값 하나(검정 1, 흰색 0)인 격자를 넣어 학습시켰다. 

이전 실험은 QR을 그림 파일로 만들어 크기를 맞춰 넣었는데, 그 과정에서 QR 크기(버전) 정보가 새어 들어가 결과가 탁해졌다. 그래서 격자를 그대로 써서 이 문제를 해결하고자 했고, 어떤 칸이 주소의 몇 번째 글자에서 왔는지도 되짚어볼 수 있었다.

카메라로 찍은 QR은 여기에 조명, 기울기, 해상도 같은 촬영 잡음이 더해진 별개의 문제라 제외하여, 이 연구는 "무늬 자체에 신호가 있는가"만 다룬다.

## 핵심 결론

**길이만 보고 피싱 사이트임을 맞추지 않았다.** QR 코드 크기별로 집단을 나누고, 각 집단에서 피싱과 정상의 주소 길이 분포를 1바이트 단위까지 똑같이 맞춘 뒤에도 성능이 남는 것을 확인했다. 길이 통제를 완전히 풀었을 때와의 차이는 0.005 ~ 0.024 AUROC

**모델은 무늬의 배치를 실제로 이용한다.** 칸의 값은 그대로 두고 위치만 뒤섞으면 CNN 성능이 0.11 ~ 0.15 떨어지는데, 위치를 보지 않는 대조 모델은 거의 변하지 않는다. "어떤 칸이 검은가"가 아니라 "그 특정 무늬가 어디에 있는가"를 실제로 쓴다는 의미이다.

**그러나, 피싱만의 보편적인 무늬가 있다고는 말할 수 없었다.** 학습에 쓴 데이터와 완전히 다른 출처의 2026년 데이터에 그대로 걸면 성능이 0.649 / 0.639 / 0.677로 떨어진다. 우연 수준(0.53 ~ 0.56)보다는 높으니 신호가 사라진 것은 아니지만, 같은 조건에서 주소 텍스트를 그냥 읽는 방식이 여전히 피싱 사이트 판별력이 훨씬 좋았다.

성능 지표는 AUROC 기준(1.0이면 완벽, 0.5면 동전 던지기 확률). 숫자 세 개가 나란히 있으면 QR 코드 크기 세 집단(v2 / v3 / v4)의 값.

## 무엇을 했나

1. 공개된 피싱, 정상 URL 데이터를 모아 주소 표기를 통일하고 중복 제거.
2. 각 주소를 QR 코드로 만들되 칸 격자 그대로 모델에 넣음.
3. 우회 단서를 막음. QR 코드 크기별로 집단을 나누고 그 안에서 두 클래스의 주소 길이를 같게 맞추며, 같은 도메인이 학습과 평가에 동시에 들어가지 않게 분리.
4. 대조 실험을 돌림. 칸 위치를 뒤섞거나 정답 라벨을 뒤섞었을 때 성능이 어떻게 되는지 확인.
5. 작은 무늬 조각(3×3)의 빈도만으로도 구분이 되는지 확인하고, 그 조각을 일부러 망가뜨렸을 때 모델 판단이 바뀌는지 개입 실험 수행.
6. 마지막으로 전혀 다른 출처의 최신 데이터에 학습된 모델을 돌려봄.

## 결과 표

| 질문 | 답 |
|---|---|
| 우회 단서를 다 막아도 QR 무늬에 구분 신호가 남는가 | 남는다. 주 조건 AUROC 0.872 / 0.814 / 0.805 |
| CNN이 무늬의 배치를 이용하는가 | 이용한다. 위치를 섞으면 0.765 / 0.671 / 0.656으로 하락 |
| 그 신호가 다른 데이터에서도 재현되는가 | 약하다. 외부 데이터 전이 0.649 / 0.639 / 0.677 |

## 이 연구가 말하지 않는 것

- 입력은 QR 촬영본이 아닌 위에서 설명한 합성 격자이며, 촬영 잡음 등을 배재하여 오로지 QR에만 집중한 실험이다.
- 주소를 소문자로 바꾸고 끝의 `/`를 떼는 등 표기를 통일한 상태로 실험됐다. URL을 QR로 변환할 때 '/' 등의 작은 문자 요소 차이가 QR 무늬에 변화를 주기 때문.
- 학습은 사실상 한 데이터셋에 기대며, 여러 출처를 아우른 결론이 아니다.

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

데이터는 `uv run python scripts/convert_webphish.py`로 `backups/data/URL.xlsx` → `data/webphish.csv`를 만들어서 사용함.

Colab 실행 순서는 아래와 같다:

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

캠페인 단위 홀드아웃에는 아직 CLI 진입점이 없어 `qrphish.runner.run_campaign_holdout(cfg)`를 직접 호출(노트북 `[15]`).

## 데이터

- 학습·평가: WebPhish 데이터셋(Opara et al., 2024) 
  - Kaggle 배포본 `guchiopara/look-before-you-leap`의 `Category`/`Data` 컬럼이 원본이며 `spam`을 피싱, `ham`을 정상으로 매핑. 
- 외부 검증: OpenPhish 공개 피드(피싱), Common Crawl × Tranco 조인(정상)으로 만든 2026년 세트 
