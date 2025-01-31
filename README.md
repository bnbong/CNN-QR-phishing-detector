# CNN QR-Code Phshing detection

QR 코드 이미지를 분석하여 해당 QR 코드에 링크된 사이트의 피싱 사이트 여부를 판단하는 CNN 모델 프로젝트

## 프로젝트 목표

- 이미지 학습 모델로 CNN을 사용하여 피싱 사이트들에 자주 나타나는 QR 코드 패턴이 존재하고 이를 분석 가능하다는 것을 증명
- 상기 가설이 타당한 것으로 증명될 시, QR 코드 이미지를 바탕으로 빠르게 피싱 사이트를 판단하는 모델 구현

## 프로젝트 구조

```
.
├── README.md
├── best_model.h5
├── data
│   ├── URL.xlsx  # 예시) 피싱 / 정상 사이트 데이터셋
│   └── images  # 예시) 파싱된 QR 코드 이미지
│       ├── benign
│       └── phishing
├── main.py
├── notebooks
│   ├── evaluate.ipynb
│   ├── predict.ipynb
│   ├── single-url-to-qr.ipynb
│   └── urls-to-qr.ipynb
├── pdm.lock
├── pyproject.toml
├── requirements.txt
├── scripts
│   ├── format.sh
│   └── lint.sh
├── src
│   └── cnn_qr_phishing_detector
│       ├── __init__.py
│       ├── model
│       │   ├── __init__.py
│       │   └── cnn.py
│       ├── preprocess
│       │   ├── __init__.py
│       │   └── preprocessor.py
│       ├── py.typed
│       ├── train
│       │   ├── __init__.py
│       │   └── train.py
│       └── utils.py
├── tests
│   └── __init__.py
└── training_log.csv  # 학습 로그, 모델 학습 과정에서 auto-created
```

## Stack

- Python 3.10
- TensorFlow 2.10.0 + keras (for Windows compatibility - [tensorflow GPU available](https://www.tensorflow.org/install/source_windows?hl=ko#gpu) 버전)

## 프로젝트 셋업

프로젝트 루트에 `data` 폴더를 생성합니다.

### 0. 모델 딥러닝

[링크](https://drive.google.com/file/d/1ufRmh9VVLdZafiXUiMFFzjg8870W9gdV/view?usp=sharing)에서 파싱된 이미지를 `data` 폴더 내에 압축해제 합니다.

압축해제 후 다음 명령어를 실행하여 학습을 수행합니다.

```bash
python main.py
```

학습된 모델은 프로젝트 루트에 `*.h5` 파일로 저장됩니다.


<details>
<summary><b>주피터 노트북 모듈 사용법</b></summary>
<div markdown="1">

### 1. notebooks/urls-to-qr.ipynb

`data` 폴더 내에 피싱 / 정상 사이트 데이터셋이 담겨있는 xlsx 파일을 넣습니다.

주석으로 `change me` 표시가 되어 있는 옵션을 변경하여 사용합니다.

### 2. notebooks/predict.ipynb

단일 QR 이미지를 바탕으로 해당 QR 코드 이미지가 피싱 사이트인지 확인하는 모듈이며, 프로젝트 루트에 이미 학습된 모델의 체크포인트 파일이 존재해야 합니다.

주석으로 `change me` 표시가 되어 있는 옵션을 변경하여 사용합니다.

### 3. notebooks/evaluate.ipynb

모델 평가 지표를 확인하는 모듈이며, 프로젝트 루트에 이미 학습된 모델의 체크포인트 파일이 존재해야 합니다.

### 4. notebooks/single-url-to-qr.ipynb

주석으로 `change me` 표시가 되어 있는 옵션을 변경하여 사용합니다.

### 5. notebooks/urls-to-qr.ipynb

주석으로 `change me` 표시가 되어 있는 옵션을 변경하여 사용합니다.

</div>
</details>

## 학습된 모델 평가

학습 데이터 양 : 45,373개 이미지 (피싱 : 22,686 / 정상 : 22,687)

- 정확도 (Accuracy) : 0.8710
- 정밀도 (Precision) : 0.8552
- 재현율 (Recall) : 0.8934
- F1 점수 (F1 Score) : 0.8739

### 1. Confusion Matrix

![Confusion Matrix](./evaluate_outputs/confusion_matrix.png)

### 2. ROC Curve

![ROC Curve](./evaluate_outputs/roc_curve.png)
