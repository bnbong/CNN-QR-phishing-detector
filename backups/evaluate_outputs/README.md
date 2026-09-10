# 성능 평가 및 시각화 자료 분석

이 문서는 초기 모델의 평가와 시각화 기록이다. 현재 통제 실험의 결과가 아니며, 아래 정확도를 현재 AUROC와 직접 비교하지 않는다. Grad-CAM 이미지만으로 특정 패턴의 인과적 중요성을 판단할 수는 없다.

## 당시 CNN 모델 성능

당시 기록의 데이터 수: 45,373개 이미지 (피싱 : 22,686 / 정상 : 22,687)

- 정확도 (Accuracy) : 0.8710
- 정밀도 (Precision) : 0.8552
- 재현율 (Recall) : 0.8934
- F1 점수 (F1 Score) : 0.8739

### 1. Confusion Matrix

![Confusion Matrix](./confusion_matrix.png)

### 2. ROC Curve

![ROC Curve](./roc_curve.png)

## 시각화 자료 분석 및 실험 결과 가설

시각화에는 당시 `notebooks/visualizer.ipynb`를 사용했다. 경로는 초기 저장소 구조를 기준으로 한다.

시각화 자료 분석 & 모듈 코드 참고 : <https://velog.io/@tobigs_xai/CAM-Grad-CAM-Grad-CAMpp>

테스트 대상 이미지 : `./images/badqr.png` (실제 피싱 사이트를 QR 코드로 변환한 이미지, 당시 기록상 학습에 사용하지 않음)

### 1. 시각화 자료

#### Feature Map - Layer 1

![Feature Map Layer 1](./fm-1.png)

#### Feature Map - Layer 2

![Feature Map Layer 2](./fm-2.png)

#### Feature Map - Layer 3

![Feature Map Layer 3](./fm-3.png)

#### Grad-CAM

![Grad-CAM](./grad-cam_real_phishing.png)

#### Guided Grad-CAM

![Guided Grad-CAM](./guided-grad-cam_real_phishing.png)

