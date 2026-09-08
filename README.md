# CNN-QR-phishing-detector

QR 코드 모듈 격자에서 피싱 URL을 분류하는 실험

## 실험하고 싶은 것

> [!IMPORTANT]
> 피싱 사이트의 QR 코드에는 공통 시각 패턴이 있을까? 
> 합성곱 모델이 URL 유래 신호를 디코딩 없이 활용할 수 있을까?

## 연구 질문

QR 이미지로 피싱을 맞출 수 있는가? => 참. QR은 URL의 무손실 & 결정적 인코딩이므로(디코딩해서 텍스트 분류기를 돌리면 된다). 

이 실험의 연구 질문은 셋이고(RQ1 ~ RQ3), `docs/EXPERIMENT_DESIGN.md` 0절이 canonical 정의다.

1. **RQ1**: 길이·버전·패딩 같은 shortcut을 전부 통제한 뒤에도 QR 모듈 격자에 판별 신호가 남는가.
2. **RQ2**: 합성곱 모델이 디코딩 없이 모듈 배치만 보고 URL 유래 내용·순서 신호(URL-derived content/order signal)를 어디까지 활용하는가.
   - 척도: 참조 기준과의 갭 = URL 텍스트 분류기(char n-gram TF-IDF + LR). 상한선이 아니라 강한 텍스트 베이스라인이다.
3. **RQ3**: 그 신호가 여러 피싱 QR에 반복되는 국소 motif로 나타나며 새 도메인·데이터셋에서도 재현되는가.

## 이전 실험을 바탕으로 수행한 재설계

이전 버전은 QR을 PNG로 굽고 리사이즈해 학습했는데, 그 결과 모델이 배운 상당 부분은 URL 유래 내용 신호가 아니라 길이 프록시였다.

QR은 URL이 길어지면 버전이 올라가 격자가 커진다. 남는 데이터 코드워드는 `0xEC/0x11` 교대 패딩으로 채워지므로 **패딩 시작 위치가 URL 길이를 그대로 누출한다.** benign이 짧고 phishing이 긴 데이터셋에서는 이 하나만으로도 정확도가 천장을 친다.

또한 데이터셋 설계 오류로 인해 피싱 QR은 v5+부터 등장 빈도가 많아졌기에 자주 등장하는 시각 패턴이 아닌 QR 버전 자체만 들여다보고 피싱사이트라고 판별하는 모델이 만들어졌다.

**재설계 포인트:**

- **모듈 격자 직접 입력**: PNG 리사이즈 없이 `(2, n, n)` 비트 격자(값 채널 + 데이터 모듈 마스크)를 그대로 넣는다.
- **버전 층화**: v2/v3/v4/v5+ 층을 나눠 격자 크기를 층 안에서 고정한다.
- **엄격 길이 매칭**: 층 안에서 1바이트 단위 버킷마다 두 클래스를 같은 수로 맞춘다. 길이 주변분포가 클래스 간 동일해지므로 패딩 경계 분포도 같아진다. 완화(5자 버킷)&무통제 조건도 함께 돌려 "성능의 몇 %p가 길이 아티팩트였는지" 수치로 보고한다.
- **그룹 분할**: eTLD+1 단위로 train/val/test를 가른다. Data Leakage 방지.

여기에 역매핑(모듈 ↔ 비트 ↔ 코드워드 ↔ URL 바이트)을 붙여 Grad-CAM 질량이 URL 문자에 실리는지, 패딩 및 EC 영역에 실리는지 구분한다.

## 저장소 구조

```
qrphish/      패키지 본체 (urls, qrgen, mapping, splits, dataset, models, train, evaluate, explain, baselines, runner, cli)
configs/      base.yaml(단일 조건 기본값), matrix.yaml(P1/P2 조건 매트릭스)
scripts/      데이터 변환, P0 보조 집계
tests/        정확성 게이트(역매핑 라운드트립 포함)
docs/         EXPERIMENT_DESIGN.md — 설계 스펙 원본
notebooks/    colab_run.ipynb — Colab 실행 노트북
reports/      P0 카운트 표·진단 json·집계 csv (커밋 대상)
artifacts/    층 격자·모델·results.json (gitignore)
backups/      이전 실험의 데이터·노트북·평가 결과 보존본. 자세한 내용은 `backups/README.md`.
```

## 결과 요약

P0 ~ P3 전체 실험(Colab T4, 5시드)을 마쳤고, 연구 질문을 셋으로 재구성했다.

| 연구 질문 | 현재 답 |
|---|---|
| RQ1. 길이·버전·패딩 shortcut을 통제한 뒤에도 QR 모듈 격자에 판별 신호가 있는가 | 그렇다 |
| RQ2. CNN은 명시적 디코딩 없이 공간 배치를 이용해 URL 유래 내용·순서 신호(URL-derived content/order signal)에 접근하는가 | 접근한다. 다만 선형 프로브는 개별 n-gram·키워드가 표현에서 선형으로 복원된다는 증거를 찾지 못했다 |
| RQ3. 그 신호가 여러 피싱 QR에 반복되는 국소 motif로 나타나며 새 도메인에서도 재현되는가 | motif는 존재하고 CNN 판단에 인과적으로 기여하지만 기여분이 작다. 재현성은 미검증 |

주 조건(`norm` · 엄격 길이 매칭 · 기능 패턴 제거 · 고정 마스크)의 층별 AUROC:

| 층 | SmallCNN | BitMLP | char n-gram LR(참조 기준) | 라벨 셔플 |
|---|---|---|---|---|
| v2 | 0.872 | 0.834 | 0.969 | 0.498 |
| v3 | 0.814 | 0.732 | 0.972 | 0.508 |
| v4 | 0.805 | 0.690 | 0.992 | 0.506 |

char n-gram LR과의 갭은 0.10 ~ 0.19다. 이 기준선은 강한 텍스트 베이스라인이지 Bayes 최적 분류기가 아니므로 상한선이 아니고 CNN이 넘더라도 누출의 증거가 되지 않는다. 갭의 크기는 "남은 정보량"이 아니라 이 기준선과의 거리다. 그래도 디코딩 없이 모듈 배치만으로 URL 유래 내용·순서 신호의 상당 부분에 접근한다는 것은 분명하다.

성능의 출처는 공간 구조다. 고정 순열로 위치를 흩으면(`shuffle-pos`) CNN은 0.872/0.814/0.805 → 0.765/0.671/0.656으로 떨어지는데, 고정 순열에 대해 동등한 표현력을 갖는 BitMLP는 거의 그대로다(0.834/0.732/0.690 → 0.834/0.733/0.706).

지금 지지되는 것은 클래스와 연관된 국소 패치 빈도 분포(class-associated local patch-frequency distribution)의 존재다. 3×3 패치 히스토그램만으로 돌린 LR이 0.740/0.706/0.717로 라벨 셔플 바닥(0.49 ~ 0.53)을 뚜렷이 넘지만 CNN에는 못 미치고 개별 motif의 odds ratio도 1.4 ~ 1.6배 수준이다. 순서를 완전히 버린 byte-hist LR(0.861/0.766/0.771)이 오히려 더 높으므로, 이것을 "marginal bit/byte composition으로 설명되지 않는 국소 공간 의존성"으로 올려 쓰지는 않는다. 그 주장은 within-QR 셔플 null을 통과한 뒤에 한다. CNN과의 갭은 3×3 창보다 넓은 범위의 배치·순서에서 온다.

motif는 CNN 판단의 원인이기는 하다. 매칭된 3×3 창의 중심 모듈 하나만 뒤집는 motif 표적 중심 비트 개입(motif-targeted center-bit intervention)을 걸면 AUROC가 −0.016/−0.025/−0.037 떨어지는데 같은 개수의 무작위 모듈에 같은 개입을 걸면 −0.002 ~ −0.005에 그치고 로짓도 방향에 맞게 움직인다. 다만 절제로 사라지는 몫은 CNN과 바닥 사이 격차의 수 %p이고 위치·밀도를 맞춘 무작위 대조는 아직 없다.

"CNN이 URL 어휘를 복원한다"는 주장은 철회한다. 임베딩을 얼리고 붙인 선형 프로브에서 유의한 목표가 v2 2/91(둘 다 경로 구조 특징), v3·v4는 0개다. 라벨 자체는 0.872/0.813/0.804로 잘 분리되지만 개별 n-gram·키워드·숫자 비율은 미학습 CNN 임베딩 기준선조차 넘지 못한다. CNN이 쓰는 것은 특정 문자열이 아니라 문자 조성과 국소 비트 패턴 수준의 통계로 보인다. "피싱 QR에 보편적인 시각 signature가 있다"는 문장은 현재 자료로 쓸 수 없다.

길이 아티팩트는 작았다. 길이 통제를 완전히 푼 `L-none`과 주 조건의 차이가 +0.005/+0.002/+0.024에 그친다. 이는 버전 층화와 정규화가 이전 실험의 누출을 이미 대부분 걷어냈다는 것을 뜻한다. 반대로 정규화 자체의 효과는 크다. `raw`는 `norm`보다 0.05 ~ 0.09 높고, 이것이 이전 실험 87%대 수치의 주요 출처다.

전체 표와 해석은 [`docs/RESULTS.md`](docs/RESULTS.md).

## 로컬 셋업

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check qrphish tests scripts
```

데이터는 `uv run python scripts/convert_webphish.py`로 `backups/data/URL.xlsx` → `data/webphish.csv`를 만들어 쓴다.

## Colab 실행 순서

```bash
uv run qrphish p0 --config configs/base.yaml   # 층 카운트 표 + 패딩 누출 진단 + 역매핑 게이트
uv run qrphish run P1 --config configs/base.yaml
uv run qrphish run P2 --config configs/base.yaml
uv run qrphish explain <condition_id> --config configs/base.yaml
uv run qrphish report --config configs/base.yaml
uv run qrphish hypotheses --config configs/base.yaml   # H1~H4 + Holm 보정(bootstrap-tail pseudo-p)
uv run qrphish motifs --config configs/base.yaml       # Bag-of-QR-patches (CPU)
uv run qrphish occlusion --config configs/base.yaml    # motif 표적 중심 비트 개입
uv run qrphish probes --config configs/base.yaml       # 어휘 프로브
uv run qrphish transfer --config configs/base.yaml     # F: 외부 검증 전이 평가
```

2차 실험 G(캠페인 단위 홀드아웃)에는 아직 CLI 진입점이 없다. `qrphish.runner.run_campaign_holdout(cfg)`를
직접 호출한다(노트북 `[15]`).

P0이 첫 번째 산출물이다. `reports/stratum_counts.json`으로 어떤 층이 primary(클래스당 ≥1,000)로 살아남는지 먼저 확정한다. 그 표가 없으면 `run`은 실행을 거부한다. 노트북은 이 순서를 셀로 옮겨놓은 것뿐이고 로직을 담지 않는다.

### 노트북 셀 순서

`notebooks/colab_run.ipynb`는 위 순서를 셀로 옮긴 것이다. 번호 순서대로 돌리면 되고 각 러너는
`results.json`이 있는 조합을 건너뛰므로 세션이 끊겨도 이어서 돌릴 수 있다.

| 셀 | 내용 | GPU | 비고 |
|---|---|---|---|
| `[1]`~`[4]` | 환경·Drive 마운트·설정 | — | |
| `[5]` | P0 — 층 카운트·패딩 진단·라운드트립 게이트 | — | 여기서 primary 층 확정 |
| `[6]` `[7]` | P1 주 실험 · P2 절제 | 필요 | 가장 오래 걸린다 |
| `[8]` | 재집계 + H1 ~ H4 검정 | — | |
| `[9]` `[10]` | motif (Bag-of-QR-patches) · 인과 절제 | — | CPU, 층당 수 분 |
| `[11]` `[12]` | 어휘 프로브 · Grad-CAM | 필요 | 저장된 체크포인트만 읽는다 |
| `[13]` | 외부 검증 세트 지정 + 편향 진단 **게이트** | — | primary가 걸리면 여기서 멈춘다 |
| `[14]` | F — 외부 검증 전이 평가 | 일부 | F-a 고정 cohort는 추론만이라 GPU 없이도 된다 |
| `[15]` | G — 캠페인 단위 홀드아웃 + 표 생성 | 필요 | 층당 5시드 × 2모델 학습 |

**재실행이 필요한 셀.** 집계 규약과 null 조건이 바뀌었으므로 이전 산출물이 Drive에 남아 있으면
아래를 다시 돌린다. `[8]`은 저장된 예측에서 다시 계산하므로 그냥 실행하면 되지만 `[9]` `[10]`은
성공한 `results.json`을 건너뛰기 때문에 **Drive의 산출물을 먼저 지워야 한다.**

| 셀 | 왜 | 어떻게 |
|---|---|---|
| `[8]` | 시드 층화 집계 · `pseudo_p` 키 정리 | 그냥 다시 실행 (`reaggregate`가 덮어쓴다) |
| `[9]` | within-QR 셔플 null(모듈·코드워드) 조건 추가 | `reports/motifs/` 삭제 후 실행 |
| `[10]` | `[9]`의 `enrichment.json`이 바뀌므로 함께 | `reports/occlusion/` 삭제 후 실행 |

## 데이터

WebPhish 데이터셋(Opara et al., 2024)을 쓴다. Kaggle 배포본 `guchiopara/look-before-you-leap`의 `Category`/`Data` 두 컬럼이 원본이며, `Category`의 `spam`을 피싱(label=1), `ham`을 정상(label=0)으로 매핑한다. URL 정규화(소문자화, 선행 `www.` 제거, 말미 `/` 제거) 한 뒤 완전 중복을 제거한다. 라벨이 충돌하는 URL은 양쪽 다 버린다.
