# CNN-QR-phishing-detector

QR 코드 모듈 격자에서 피싱 URL을 분류하는 실험

## 실험하고 싶은 것

> [!IMPORTANT]
> 피싱 사이트의 QR 코드에는 공통 시각 패턴이 있을까? 
> 합성곱 모델이 URL 어휘 신호를 디코딩 없이 복원할 수 있을까?

## 연구 질문

QR 이미지로 피싱을 맞출 수 있는가? => 참. QR은 URL의 무손실 & 결정적 인코딩이므로(디코딩해서 텍스트 분류기를 돌리면 된다). 

이 실험으로 확인하고 싶은 것:
1. 합성곱 모델이 **디코딩 없이** 모듈 배치만 보고 URL의 어휘 신호를 어디까지 복원하는가. 
   - 척도: 상한선과의 갭 = URL 텍스트 분류기(char n-gram TF-IDF + LR)
2. "피싱 QR에 공통 시각 패턴이 있다"는 가설이 길이·버전·패딩 경계 같은 메타 신호를 전부 통제한 뒤에도 보장되는가.

## 이전 실험을 바탕으로 수행한 재설계

이전 버전은 QR을 PNG로 굽고 리사이즈해 학습했는데, 그 결과 모델이 배운 상당 부분은 URL 어휘가 아니라 **길이 프록시**였다.

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
```

P0이 첫 번째 산출물이다. `reports/stratum_counts.json`으로 어떤 층이 primary(클래스당 ≥1,000)로 살아남는지 먼저 확정한다. 그 표가 없으면 `run`은 실행을 거부한다. 노트북은 이 순서를 셀로 옮겨놓은 것뿐이고 로직을 담지 않는다.

## 데이터

WebPhish 데이터셋(Opara et al., 2024)을 쓴다. Kaggle 배포본 `guchiopara/look-before-you-leap`의 `Category`/`Data` 두 컬럼이 원본이며, `Category`의 `spam`을 피싱(label=1), `ham`을 정상(label=0)으로 매핑한다. URL 정규화(소문자화, 선행 `www.` 제거, 말미 `/` 제거) 한 뒤 완전 중복을 제거한다. 라벨이 충돌하는 URL은 양쪽 다 버린다.
