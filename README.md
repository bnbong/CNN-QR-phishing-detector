# CNN-QR-phishing-detector

> [!IMPORTANT]
> 실험: QR 코드 무늬만 보고 피싱 사이트를 알아챌 수 있을까?

QR 모듈 격자만으로 피싱 URL을 분류할 수 있는지, 그리고 그 성능을 피싱에 고유한 패턴으로 해석할 수 있는지 평가하는 실험 프로젝트다.

## 연구 목적

QR 코드는 입력 문자열을 보존하지만, 작은 CNN이 URL을 명시적으로 디코딩하지 않고 그 정보를 학습할 수 있는지는 별도로 확인해야 한다. 이 연구는 분류 성능의 존재, 공간 배치의 학습상 이점, 국소 패턴의 중요성, 외부 데이터에서의 재현성을 구분해 평가한다.

초기 실험(2025년 버전)에서는 길이와 생성 조건, 출처 편향을 충분히 통제하지 않아 높은 정확도의 원인을 판단하기 어려웠다. 현재 실험은 URL 정규화, QR 버전 층화, 길이 매칭, 도메인 분리 및 음성 대조 등의 통제를 적용해 같은 질문을 다시 검토한다. 초기 실험과는 지표와 평가 조건이 달라 성능을 직접 비교하지 않는다.

## 모델 입력

입력은 사진이나 크기를 조정한 PNG가 아니라 QR의 흑백 모듈 격자다. 예를 들어 v2 QR은 25×25 격자다. 첫 채널에는 모듈 값, 둘째 채널에는 기능 패턴 이외의 위치를 표시하는 마스크를 넣는다. 마스크는 URL 본문뿐 아니라 헤더, 패딩, 오류정정 및 잔여 비트 위치도 포함한다.

주 조건은 정규화 URL, EC=L, 단일 바이트 모드, 고정 마스크다. 카메라 촬영에 따른 조명, 기울기, 해상도 변화는 평가하지 않고 오로지 QR 패턴 무늬 자체에 어떠한 신호가 있는지 확인한다.

## 핵심 결과

**실험 조건 안에서는 QR 무늬만으로 피싱 URL을 어느 정도 구분할 수 있었다. 그러나 모든 피싱 QR에 공통된 고유 패턴을 찾았다거나, 실제 환경에서도 안정적으로 탐지할 수 있다고 결론 내릴 근거는 부족하다.**

아래 수치는 v2 / v3 / v4 순서이며, 지표는 AUROC다. AUROC 1은 완전한 순위 구분, 0.5는 무작위 수준을 뜻한다.

| 질문 | 답과 근거 |
|---|---|
| 길이와 버전을 통제하고 도메인을 분리한 뒤에도 판별력이 남는가 | **남는다.** 주 조건 SmallCNN의 AUROC는 0.872 / 0.814 / 0.805였다. 길이 및 버전 단독 LR은 0.500이므로, 이 두 특징만으로 CNN의 성능을 설명할 수는 없다. |
| 원래 QR 배치가 CNN 학습에 도움이 되는가 | **도움이 된다.** 동일한 고정 위치 순열을 모든 표본에 적용해 재학습하면 CNN은 0.765 / 0.671 / 0.656으로 낮아졌다. BitMLP의 변화는 작았다. 정보가 보존되어도 배치에 따라 CNN이 분류 규칙을 학습하는 데 차이가 생겼다. |
| 특정 국소 패턴이 판단에 특별히 중요한가 | **피싱 관련 패턴의 특별한 중요성은 일관되게 확인하지 못했다.** 해당 패턴을 겨냥한 개입은 위치와 밀도를 맞춘 무작위 개입보다 추가로 AUROC를 낮춘다는 근거가 부족했다. 다만 일부 다른 비교에서는 차이가 있어, 모든 국소 패턴이 중요하지 않다고 단정할 수도 없다. |
| 표현에서 URL 속성을 예측할 수 있는가 | **일부는 가능하지만, 피싱 분류 학습으로 얻은 이득은 제한적이다.** 접근성 기준을 통과한 목표는 26/91, 13/114, 1/104였다. 미학습 CNN과 비교한 학습 이득 기준까지 통과한 목표는 v2의 경로 유무와 깊이뿐이었다. 이 정보를 실제 분류에 사용하는지는 확인하지 않았다(프로브는 모델의 내부 표현에서 URL 속성을 예측할 수 있는지만 검사하며, 해당 정보를 제거했을 때 분류 판단이 달라지는지는 시험하지 않았기 때문). |
| 외부 데이터에서도 판별력이 남는가 | **일부 남지만, 일반화 성능은 제한적이다.** 길이와 경로 유무를 함께 맞춘 외부 평가에서 AUROC는 0.649 / 0.639 / 0.677로 순열 대조 기준값보다 높았다. 그러나 내부 성능보다 낮았고 문자 n-gram LR에도 못 미쳤다. |

위치 순열 비교에서 BitMLP는 위치 정보를 사용할 수 있는 대조 모델이다. 고정 순열 뒤에도 재학습으로 표현할 수 있는 함수의 범위가 유지되므로, CNN의 성능 하락을 정보 손실만으로 설명할 수 없다는 근거가 된다.

**도메인뿐 아니라 URL 템플릿까지 분리한 별도 실험에서도 판별력은 남았다.** 캠페인 홀드아웃의 Model B는 AUROC 0.851 / 0.774 / 0.828을 보였다. 다만 주 실험과 평가 집합이 다르므로, 이 수치를 모든 통제를 누적한 최종 성능으로 해석해서는 안 된다.

## 해석 범위

- QR 모듈 배치에 학습 가능한 판별 신호는 있었지만, 모든 피싱 QR에 통용되는 고유 패턴은 입증하지 못했다.
- 프로브의 목표에는 어휘와 구조가 함께 포함된다. 통과 수는 복원한 문자 수나 실제 판단에 사용된 정보량이 아니다. 층별 목표 집합과 표본 수도 달라 정보량을 직접 비교할 수 없다.
- 내부 학습 출처는 WebPhish 하나다. 외부 데이터도 같은 방향의 경로 편향을 공유하므로 외부 평가만으로 출처 편향이 제거되었다고 볼 수 없다.
- 외부 매칭은 길이와 경로 유무의 결합 분포를 맞춘다. 경로 깊이와 구분자 구성까지 통제하지 않으며 평가 표본도 달라지므로, 매칭 전후 성능 차이를 경로 편향의 인과적 기여량으로 해석하지 않는다.
- URL 전체 소문자화와 말미 `/` 제거는 경로나 쿼리의 의미를 바꿀 수 있다. 결론은 이 전처리를 거친 합성 QR에 한정된다.

## 문서 안내

- [논문 초안](docs/PAPER_DRAFT.md)
- [결과 보고서](docs/REPORT.md)와 [상세 결과](docs/RESULTS.md)
- [내부 실험 설계](docs/EXPERIMENT_DESIGN.md)와 [외부 검증 설계](docs/EXTERNAL_VALIDATION_DESIGN.md)
- [실행 안내](docs/RUNBOOK.md)
- 외부 검토 기록: [1차](docs/review_01.md), [2차](docs/review_02.md), [3차](docs/review_03.md), [4차](docs/review_04.md), [5차](docs/review_05.md)

## 저장소 사용법

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check qrphish tests scripts
```

데이터는 `uv run python scripts/convert_webphish.py`로 `backups/data/URL.xlsx` → `data/webphish.csv`를 변환해 사용한다.

Colab 실행 순서는 다음과 같다.

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

캠페인 단위 홀드아웃에는 아직 CLI 진입점이 없어 `qrphish.runner.run_campaign_holdout(cfg)`를 직접 호출한다(노트북 `[15]`).

## 데이터

- 학습 및 평가: WebPhish 데이터셋(Opara et al., 2024)
  - Kaggle 배포본 `guchiopara/look-before-you-leap`의 `Category`/`Data` 컬럼이 원본이며 `spam`을 피싱, `ham`을 정상으로 매핑.
- 외부 검증: OpenPhish 공개 피드(피싱), Common Crawl × Tranco 조인(정상)으로 만든 2026년 세트
