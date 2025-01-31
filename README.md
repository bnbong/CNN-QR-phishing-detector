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

- Python 3.12
- TensorFlow 2.18.0 + keras

## 프로젝트 셋업

프로젝트 루트에 `data` 폴더를 생성합니다.

### 1. notebooks/urls-to-qr.ipynb

`data` 폴더 내에 피싱 / 정상 사이트 데이터셋이 담겨있는 xlsx 파일을 넣습니다.

### 2. notebooks/predict.ipynb

[링크](https://drive.google.com/file/d/1ufRmh9VVLdZafiXUiMFFzjg8870W9gdV/view?usp=sharing)에서 파싱된 이미지를 `data` 폴더 내에 압축해제 합니다.

압축해제 후 다음 명령어를 실행하여 학습을 수행합니다.

```bash
python main.py
```

학습된 모델은 프로젝트 루트에 `best_model.h5`로 저장됩니다.
