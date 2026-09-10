### T2 층별 표본 수와 채택 등급

| 길이 통제 | 층 | 매칭 전 n | 매칭 후 n | benign | phishing | 등급 |
|---|---|---|---|---|---|---|
| exact | v2 | 18718 | 12118 | 6059 | 6059 | primary |
| exact | v3 | 11996 | 9178 | 4589 | 4589 | primary |
| exact | v4 | 3930 | 2714 | 1357 | 1357 | primary |
| exact | v5plus | 4147 | 532 | 266 | 266 | dropped |
| quantile | v2 | 18718 | 12200 | 6100 | 6100 | primary |
| quantile | v3 | 11996 | 9230 | 4615 | 4615 | primary |
| quantile | v4 | 3930 | 2802 | 1401 | 1401 | primary |
| quantile | v5plus | 4147 | 544 | 272 | 272 | dropped |
| none | v2 | 18718 | 16627 | 10489 | 6138 | primary |
| none | v3 | 11996 | 11996 | 7350 | 4646 | primary |
| none | v4 | 3930 | 3930 | 1404 | 2526 | primary |
| none | v5plus | 4147 | 4147 | 272 | 3875 | dropped |

### T3 주 결과

| 모델 / 기준선 | v2 | v3 | v4 |
|---|---|---|---|
| SmallCNN (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| BitMLP | 0.834 [0.801, 0.862] | 0.732 [0.695, 0.773] | 0.690 [0.649, 0.727] |
| 라벨 셔플 (바닥) | 0.498 [0.477, 0.519] | 0.508 [0.489, 0.525] | 0.506 [0.478, 0.540] |
| char n-gram LR (URL 텍스트 참조 기준) | 0.969 | 0.972 | 0.992 |
| byte-hist LR | 0.861 | 0.766 | 0.771 |
| length LR | 0.500 | 0.500 | 0.500 |
| version LR | 0.500 | 0.500 | 0.500 |
| 참조 기준 - CNN 갭 | +0.097 | +0.158 | +0.187 |

### H4 공간 구조 검정 (CNN × MLP, 원래 배치 × shuffle-pos)

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| CNN, 원래 배치 | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| CNN, shuffle-pos | 0.765 [0.722, 0.803] | 0.671 [0.641, 0.703] | 0.656 [0.626, 0.689] |
| CNN 차이 | -0.107 | -0.143 | -0.149 |
| MLP, 원래 배치 | 0.834 [0.801, 0.862] | 0.732 [0.695, 0.773] | 0.690 [0.649, 0.727] |
| MLP, shuffle-pos | 0.834 [0.801, 0.863] | 0.733 [0.696, 0.772] | 0.706 [0.666, 0.744] |
| MLP 차이 | +0.000 | +0.001 | +0.016 |

### T4 길이 통제 절제 (H3)

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| L-exact (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| L-quantile | 0.853 [0.828, 0.878] | 0.817 [0.779, 0.854] | 0.797 [0.761, 0.829] |
| L-none | 0.877 [0.857, 0.894] | 0.816 [0.777, 0.853] | 0.829 [0.792, 0.863] |
| L-none - L-exact | +0.005 | +0.002 | +0.024 |
| PAD-rand (L-none 위) | 0.862 [0.840, 0.881] | 0.795 [0.755, 0.832] | 0.799 [0.759, 0.837] |
| PAD-rand - L-none | -0.015 | -0.021 | -0.030 |
| length LR @ L-exact | 0.500 | 0.500 | 0.500 |
| length LR @ L-none | 0.543 | 0.544 | 0.610 |
| version LR @ L-none | 0.500 | 0.500 | 0.500 |

### T5 표현, 마스크, 부분집합 절제

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| norm, mask-fixed, data_only (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| raw (정규화 해제) | 0.921 [0.906, 0.936] | 0.873 [0.833, 0.912] | 0.892 [0.867, 0.912] |
| raw - norm | +0.049 | +0.060 | +0.087 |
| feat-all (기능 패턴 포함) | 0.873 [0.847, 0.893] | 0.802 [0.766, 0.837] | 0.813 [0.783, 0.841] |
| feat-all - data_only | +0.001 | -0.012 | +0.008 |
| mask-auto | 0.748 [0.723, 0.771] | 0.678 [0.654, 0.702] | 0.675 [0.646, 0.702] |
| mask-auto - mask-fixed | -0.124 | -0.135 | -0.130 |
| maskindex LR @ mask-auto | 0.524 | 0.486 | 0.517 |
| mask-off (언마스킹) | 0.871 [0.843, 0.893] | 0.803 [0.757, 0.845] | 0.816 [0.784, 0.843] |
| mask-off - mask-fixed | -0.001 | -0.011 | +0.011 |
| path+ 부분집합 | 0.727 [0.677, 0.779] | 0.766 [0.713, 0.816] | 0.822 [0.784, 0.853] |
| path+ 참조 기준 (char n-gram LR) | 0.914 | 0.968 | 0.988 |
| EC-M (v3 한정) | - | 0.847 [0.816, 0.871] | - |
| EC-M 참조 기준 (char n-gram LR) | - | 0.961 | - |

### Grad-CAM kind별 CAM 질량 비율

| CAM 질량 (kind) | v2 | v3 | v4 |
|---|---|---|---|
| char (URL 문자) | 50.6% | 57.3% | 64.6% |
| pad (패딩) | 9.8% | 11.7% | 12.3% |
| ec (오류정정) | 8.7% | 9.7% | 9.0% |
| function (기능 패턴) | 28.2% | 19.6% | 12.7% |
| length_header | 1.7% | 1.0% | 0.7% |
| mode_header | 0.7% | 0.4% | 0.3% |
| remainder | 0.2% | 0.3% | 0.1% |
| char / 데이터 모듈 내 비율 | 70.7% | 71.6% | 74.4% |
| n_samples | 640 | 640 | 640 |

### Grad-CAM enrichment와 난수 기준선

| Grad-CAM enrichment | v2 | v3 | v4 |
|---|---|---|---|
| char (URL 문자), enrichment | 1.57 | 1.48 | 1.43 |
| char (URL 문자), 난수 기준선 | 1.00 | 1.00 | 1.00 |
| pad (패딩), enrichment | 1.13 | 0.98 | 0.99 |
| pad (패딩), 난수 기준선 | 1.00 | 1.00 | 1.00 |
| ec (오류정정), enrichment | 0.68 | 0.68 | 0.61 |
| ec (오류정정), 난수 기준선 | 1.00 | 1.00 | 1.00 |
| function (기능 패턴), enrichment | 0.66 | 0.60 | 0.49 |
| function (기능 패턴), 난수 기준선 | 1.00 | 1.00 | 1.00 |
| length_header, enrichment | - | - | - |
| length_header, 난수 기준선 | - | - | - |
| mode_header, enrichment | - | - | - |
| mode_header, 난수 기준선 | - | - | - |
| remainder, enrichment | 0.18 | 0.34 | 0.19 |
| remainder, 난수 기준선 | 1.00 | 0.99 | 1.00 |
| 무작위화 모델 CAM 상관 (sanity) | 0.076 | 0.021 | -0.067 |

### Bag-of-QR-patches (RQ3 국소 패턴 분석)

| 표현 / 모델 | v2 | v3 | v4 |
|---|---|---|---|
| patch 2×2 히스토그램 LR | 0.526 [0.494, 0.559] | 0.567 [0.521, 0.614] | 0.573 [0.525, 0.622] |
| patch 3×3 히스토그램 LR | 0.740 [0.705, 0.771] | 0.707 [0.667, 0.743] | 0.715 [0.676, 0.750] |
| patch 3×3 spatial pyramid LR | 0.814 [0.792, 0.834] | 0.712 [0.685, 0.742] | 0.743 [0.708, 0.775] |
| patch 3×3, within-QR 모듈 셔플 null | 0.498 [0.486, 0.510] | 0.502 [0.489, 0.516] | 0.510 [0.485, 0.535] |
| patch 3×3, 코드워드 순서 셔플 null | 0.547 [0.534, 0.562] | 0.539 [0.519, 0.561] | 0.555 [0.529, 0.585] |
| patch 3×3, 라벨 셔플 (바닥) | 0.488 [0.471, 0.508] | 0.488 [0.462, 0.512] | 0.530 [0.500, 0.560] |
| SmallCNN (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| byte-hist LR | 0.861 | 0.766 | 0.771 |
| BitMLP | 0.834 [0.801, 0.862] | 0.732 [0.695, 0.773] | 0.690 [0.649, 0.727] |
| 3×3 창 개수 (평균) | 219 | 395 | 603 |

### 어휘 프로브 (RQ2, D안)

| 어휘 프로브 | v2 | v3 | v4 |
|---|---|---|---|
| ① 선형으로 접근 가능한가 (`accessible`) | 26 / 91 (28.6%) | 13 / 114 (11.4%) | 1 / 104 (1.0%) |
| ② 학습이 접근성을 높였는가 (`learned_gain`) | 2 / 91 (2.2%) | 0 / 114 (0.0%) | 0 / 104 (0.0%) |
| ③ 둘 다 (`accessible_and_gained`, 옛 significant) | 2 / 91 (2.2%) | 0 / 114 (0.0%) | 0 / 104 (0.0%) |
| ①만 만족 (②는 아님) | 24 | 13 | 1 |
| ④ 실제 분류 결정에 쓰이는가 (`used_in_decision`) | 측정 불가 | 측정 불가 | 측정 불가 |
| 라벨 프로브 (참고 상한선) | 0.872 | 0.813 | 0.804 |

① `accessible` = 전 시드에서 학습 표현 프로브 CI 하한 > 셔플 기준선 CI 상한. ② `learned_gain` = 전 시드에서 학습 표현 프로브 CI 하한 > 무작위 초기화 CNN 기준선 CI 상한. ④는 동결 표현 위의 선형 프로브로는 답할 수 없다 - 프로브는 표현을 읽을 뿐 결정 경로에 개입하지 않는다.


**v2 상위 목표 - ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| ngram:jp. | auroc | 0.910 | 0.512 | 0.850 |
| ngram:x.h | auroc | 0.839 | 0.517 | 0.811 |
| ngram:.ht | auroc | 0.820 | 0.508 | 0.743 |
| keyword:.html | auroc | 0.817 | 0.490 | 0.775 |
| ngram:ne. | auroc | 0.815 | 0.480 | 0.799 |
| ngram:htm | auroc | 0.814 | 0.496 | 0.736 |
| ngram:du/ | auroc | 0.811 | 0.509 | 0.688 |
| ngram:tml | auroc | 0.808 | 0.516 | 0.762 |
| ngram:zon | auroc | 0.802 | 0.501 | 0.706 |
| ngram:dex | auroc | 0.792 | 0.507 | 0.783 |

**v2 상위 목표 - ③ 둘 다 만족**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| has_path | auroc | 0.889 | 0.492 | 0.694 |
| path_depth | r2 | 0.245 | -0.050 | 0.078 |

**v3 상위 목표 - ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| has_path | auroc | 0.798 | 0.523 | 0.676 |
| keyword:login | auroc | 0.786 | 0.498 | 0.649 |
| ngram:/lo | auroc | 0.785 | 0.538 | 0.668 |
| ngram:ogi | auroc | 0.756 | 0.470 | 0.645 |
| ngram:htm | auroc | 0.755 | 0.520 | 0.630 |
| ngram:.ht | auroc | 0.746 | 0.482 | 0.627 |
| keyword:.php | auroc | 0.745 | 0.481 | 0.635 |
| ngram:.ph | auroc | 0.742 | 0.513 | 0.634 |
| ngram:php | auroc | 0.741 | 0.522 | 0.633 |
| ngram:tml | auroc | 0.731 | 0.522 | 0.657 |

**v4 상위 목표 - ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| digit_ratio | r2 | 0.227 | -0.123 | 0.028 |

### 인과 motif 절제 (RQ3, C안)


**v2** (원본 AUROC 0.872)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.014 | [-0.019, -0.009] | 0.004 | - | -0.222 | -0.655 | +0.211 | 3.8 |
| random (phishing 개수 맞춤) | -0.015 | [-0.017, -0.013] | 0.003 | 0.003 | +0.168 | -0.200 | +0.536 | 3.8 |
| benign motif | -0.021 | [-0.028, -0.015] | 0.008 | - | +0.731 | +0.277 | +1.185 | 3.9 |
| random (benign 개수 맞춤) | -0.016 | [-0.018, -0.014] | 0.002 | 0.004 | +0.304 | -0.103 | +0.711 | 3.9 |
| random (phishing topology 맞춤) | -0.014 | [-0.016, -0.012] | 0.002 | 0.003 | +0.123 | -0.253 | +0.498 | 3.8 |
| random (benign topology 맞춤) | -0.019 | [-0.022, -0.017] | 0.004 | 0.002 | +0.366 | -0.102 | +0.833 | 3.9 |

ΔΔAUROC = 표적 개입 AUROC - 무작위 개입 AUROC(같은 test, 반복 평균). 음수면 표적 개입이 AUROC를 더 낮췄다는 뜻이다.

| 대비 | ΔΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 순열 p(시드별) |
|---|---|---|---|---|
| phishing motif - random(개수) | +0.000 | [-0.006, +0.006] | 0.002 | 0.649/0.721/0.998/0.908/0.976 |
| phishing motif - random(topology) | -0.000 | [-0.007, +0.005] | 0.005 | 0.355/0.868/0.495/0.128/0.581 |
| benign motif - random(개수) | -0.005 | [-0.012, +0.001] | 0.007 | 0.152/0.128/0.569/0.948/0.575 |
| benign motif - random(topology) | -0.002 | [-0.009, +0.004] | 0.007 | 0.267/0.301/0.695/0.938/0.415 |

motif 목록은 시드별 train에서만 골랐다 - 시드 간 top 목록 Jaccard 평균 0.507 (최소 0.429), 모든 시드에 공통 6개 / 합집합 21개.


**v3** (원본 AUROC 0.814)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.031 | [-0.038, -0.022] | 0.020 | - | +0.068 | -0.283 | +0.419 | 7.1 |
| random (phishing 개수 맞춤) | -0.022 | [-0.025, -0.019] | 0.002 | 0.005 | +0.181 | -0.122 | +0.484 | 7.1 |
| benign motif | -0.025 | [-0.033, -0.018] | 0.008 | - | +0.801 | +0.445 | +1.156 | 7.8 |
| random (benign 개수 맞춤) | -0.023 | [-0.027, -0.019] | 0.005 | 0.005 | +0.272 | -0.043 | +0.587 | 7.8 |
| random (phishing topology 맞춤) | -0.022 | [-0.026, -0.018] | 0.004 | 0.004 | +0.118 | -0.182 | +0.419 | 7.1 |
| random (benign topology 맞춤) | -0.024 | [-0.028, -0.020] | 0.004 | 0.005 | +0.334 | -0.008 | +0.677 | 7.8 |

ΔΔAUROC = 표적 개입 AUROC - 무작위 개입 AUROC(같은 test, 반복 평균). 음수면 표적 개입이 AUROC를 더 낮췄다는 뜻이다.

| 대비 | ΔΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 순열 p(시드별) |
|---|---|---|---|---|
| phishing motif - random(개수) | -0.009 | [-0.017, -0.001] | 0.019 | 0.010/0.405/0.609/0.707/0.154 |
| phishing motif - random(topology) | -0.009 | [-0.016, +0.001] | 0.020 | 0.008/0.146/0.439/0.932/0.198 |
| benign motif - random(개수) | -0.002 | [-0.009, +0.005] | 0.011 | 0.094/0.828/0.299/0.473/0.868 |
| benign motif - random(topology) | -0.001 | [-0.009, +0.007] | 0.011 | 0.371/0.583/0.138/0.557/0.922 |

motif 목록은 시드별 train에서만 골랐다 - 시드 간 top 목록 Jaccard 평균 0.555 (최소 0.429), 모든 시드에 공통 5개 / 합집합 16개.


**v4** (원본 AUROC 0.805)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.034 | [-0.051, -0.020] | 0.024 | - | -0.578 | -0.824 | -0.332 | 12.6 |
| random (phishing 개수 맞춤) | -0.033 | [-0.039, -0.026] | 0.010 | 0.008 | -0.044 | -0.256 | +0.167 | 12.6 |
| benign motif | -0.045 | [-0.057, -0.029] | 0.010 | - | +0.523 | +0.246 | +0.800 | 13.5 |
| random (benign 개수 맞춤) | -0.027 | [-0.034, -0.020] | 0.004 | 0.007 | -0.031 | -0.208 | +0.147 | 13.5 |
| random (phishing topology 맞춤) | -0.029 | [-0.036, -0.021] | 0.006 | 0.009 | -0.150 | -0.338 | +0.037 | 12.6 |
| random (benign topology 맞춤) | -0.026 | [-0.033, -0.019] | 0.007 | 0.010 | -0.004 | -0.169 | +0.160 | 13.5 |

ΔΔAUROC = 표적 개입 AUROC - 무작위 개입 AUROC(같은 test, 반복 평균). 음수면 표적 개입이 AUROC를 더 낮췄다는 뜻이다.

| 대비 | ΔΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 순열 p(시드별) |
|---|---|---|---|---|
| phishing motif - random(개수) | -0.002 | [-0.018, +0.014] | 0.023 | 0.419/0.226/0.172/0.794/0.261 |
| phishing motif - random(topology) | -0.005 | [-0.020, +0.012] | 0.027 | 0.076/0.072/0.086/0.425/0.735 |
| benign motif - random(개수) | -0.018 | [-0.033, -0.002] | 0.012 | 0.912/0.228/0.501/0.417/0.108 |
| benign motif - random(topology) | -0.019 | [-0.033, -0.004] | 0.013 | 0.774/0.259/0.289/0.363/0.056 |

motif 목록은 시드별 train에서만 골랐다 - 시드 간 top 목록 Jaccard 평균 0.411 (최소 0.333), 모든 시드에 공통 2개 / 합집합 19개.


### 가설 검정 판정 (H1 ~ H4)

| 가설 | 층 | 근거 종류 | ΔAUROC 추정치 | 95% CI | 순열 p | 판정 |
|---|---|---|---|---|---|---|
| H1. CNN > 라벨 셔플 바닥 | v2 | 정식 (클러스터 순열) | +0.374 | [+0.344, +0.402] | <0.001 | H0: ΔAUROC ≤ 0 (CNN이 라벨 셔플 바닥보다 높지 않다) - 순열 p <0.001, 우세 근거 있음 |
| H1. CNN > 라벨 셔플 바닥 | v3 | 정식 (클러스터 순열) | +0.306 | [+0.265, +0.346] | <0.001 | H0: ΔAUROC ≤ 0 (CNN이 라벨 셔플 바닥보다 높지 않다) - 순열 p <0.001, 우세 근거 있음 |
| H1. CNN > 라벨 셔플 바닥 | v4 | 정식 (클러스터 순열) | +0.299 | [+0.253, +0.339] | <0.001 | H0: ΔAUROC ≤ 0 (CNN이 라벨 셔플 바닥보다 높지 않다) - 순열 p <0.001, 우세 근거 있음 |
| H2. URL 텍스트 참조 기준 > CNN | v2 | 기술 추정 (쌍체 부트스트랩) | +0.097 | [+0.076, +0.122] | - | 검정 없음 - 쌍체 ΔAUROC의 95% CI가 0을 배제, 참조 기준이 더 높다는 근거 |
| H2. URL 텍스트 참조 기준 > CNN | v3 | 기술 추정 (쌍체 부트스트랩) | +0.158 | [+0.124, +0.195] | - | 검정 없음 - 쌍체 ΔAUROC의 95% CI가 0을 배제, 참조 기준이 더 높다는 근거 |
| H2. URL 텍스트 참조 기준 > CNN | v4 | 기술 추정 (쌍체 부트스트랩) | +0.187 | [+0.156, +0.221] | - | 검정 없음 - 쌍체 ΔAUROC의 95% CI가 0을 배제, 참조 기준이 더 높다는 근거 |
| H3. L-none > L-exact | v2 | 기술 추정 (비쌍체 부트스트랩) | +0.005 | [-0.023, +0.036] | - | 검정 없음 - 비쌍체 ΔAUROC의 95% CI가 0을 포함 (쌍체 해석 불가) |
| H3. L-none > L-exact | v3 | 기술 추정 (비쌍체 부트스트랩) | +0.002 | [-0.053, +0.060] | - | 검정 없음 - 비쌍체 ΔAUROC의 95% CI가 0을 포함 (쌍체 해석 불가) |
| H3. L-none > L-exact | v4 | 기술 추정 (비쌍체 부트스트랩) | +0.024 | [-0.024, +0.071] | - | 검정 없음 - 비쌍체 ΔAUROC의 95% CI가 0을 포함 (쌍체 해석 불가) |
| H4. 원래 배치 > shuffle-pos | v2 | 정식 (클러스터 순열) | +0.107 | [+0.082, +0.135] | <0.001 | H0: ΔAUROC ≤ 0 (원래 배치가 shuffle-pos보다 유리하지 않다) - 순열 p <0.001, 우세 근거 있음 |
| H4. 원래 배치 > shuffle-pos | v3 | 정식 (클러스터 순열) | +0.143 | [+0.108, +0.179] | <0.001 | H0: ΔAUROC ≤ 0 (원래 배치가 shuffle-pos보다 유리하지 않다) - 순열 p <0.001, 우세 근거 있음 |
| H4. 원래 배치 > shuffle-pos | v4 | 정식 (클러스터 순열) | +0.149 | [+0.115, +0.181] | <0.001 | H0: ΔAUROC ≤ 0 (원래 배치가 shuffle-pos보다 유리하지 않다) - 순열 p <0.001, 우세 근거 있음 |

정식 가설 검정은 쌍체 예측이 있는 H1, H4의 클러스터 순열 검정뿐이다. 순열 null은 같은 group(도메인 클러스터) 안의 행을 한 블록으로 묶어 두 조건 라벨을 교환해 만든 **정의한 그룹 블록 null 기준**이며, 블록 단위 교환 가능성을 가정한다 - 반복 수를 늘리면 몬테카를로 오차만 줄고 그 가정이 검증되지는 않는다.

H2에는 정식 검정을 붙이지 않았다. 쌍체 ΔAUROC와 95% CI를 "참조 기준이 더 높다는 근거"로만 읽는다.

H3의 L-none과 L-exact는 길이 매칭 여부가 달라 **평가 표본 자체가 다르다**. 쌍체 ΔAUROC로 해석할 수 없으므로 비쌍체 추정치와 CI만 싣는다.

부트스트랩 CI는 저장된 모델들의 예측에 조건부인 평가 불확실성이다. 학습 데이터와 학습 과정 전체의 불확실성까지 포함하지 않는다.

### 부록, pseudo-p와 Holm 보정 (정식 검정 아님)

| 가설 | 층 | pseudo-p | pseudo-p (Holm) | CI가 0을 배제 |
|---|---|---|---|---|
| H1. CNN > 라벨 셔플 바닥 | v2 | <0.001 | 0.012 | 예 |
| H1. CNN > 라벨 셔플 바닥 | v3 | <0.001 | 0.012 | 예 |
| H1. CNN > 라벨 셔플 바닥 | v4 | <0.001 | 0.012 | 예 |
| H2. URL 텍스트 참조 기준 > CNN | v2 | <0.001 | 0.012 | 예 |
| H2. URL 텍스트 참조 기준 > CNN | v3 | <0.001 | 0.012 | 예 |
| H2. URL 텍스트 참조 기준 > CNN | v4 | <0.001 | 0.012 | 예 |
| H3. L-none > L-exact | v2 | 0.759 | 1.000 | 아니오 |
| H3. L-none > L-exact | v3 | 0.932 | 1.000 | 아니오 |
| H3. L-none > L-exact | v4 | 0.346 | 1.000 | 아니오 |
| H4. 원래 배치 > shuffle-pos | v2 | <0.001 | 0.012 | 예 |
| H4. 원래 배치 > shuffle-pos | v3 | <0.001 | 0.012 | 예 |
| H4. 원래 배치 > shuffle-pos | v4 | <0.001 | 0.012 | 예 |

pseudo-p는 부트스트랩 백분위 CI를 역전시켜 정의한 값이지 영가설 분포에서 나온 정식 p값이 아니다. Holm 보정도 그 값에 건 것이므로 "보정 후에도 유의"라는 표현은 쓰지 않는다. 정식 판정은 위 가설 검정 표를 본다.

### 외부 검증 전이 (F), motif 재현성

| 외부 세트 | 실행 | 층 | CNN AUROC | 95% CI | 순열 바닥선(97.5%) | 행순열 바닥선(97.5%) | n_perm | n_boot | 평가 n | 플래그 | char n-gram LR | byte-hist LR | motif-hist LR (3x3) | path-shape LR | length LR (게이트) | version LR (게이트) | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.733 | [0.712, 0.755] | 0.545 | 0.504 | 2000 | 2000 | 14556 | 공통방향편향, 주 판정 아님 | 0.850 | 0.808 | 0.634 | 0.784 | 0.500 | 0.500 | reported_unmatched_path |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.675 | [0.620, 0.712] | 0.588 | 0.504 | 2000 | 2000 | 16290 | 공통방향편향, 주 판정 아님 | 0.875 | 0.707 | 0.597 | 0.609 | 0.500 | 0.500 | reported_unmatched_path |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.685 | [0.643, 0.730] | 0.573 | 0.506 | 2000 | 2000 | 7198 | 공통방향편향, 주 판정 아님 | 0.810 | 0.682 | 0.599 | 0.535 | 0.500 | 0.500 | reported_unmatched_path |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | 0.636 | [0.617, 0.656] | 0.524 | 0.506 | 2000 | 2000 | 6074 | 공통방향편향 | 0.723 | 0.616 | 0.574 | 0.476 | 0.500 | 0.500 | partial_collapse |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | 0.604 | [0.573, 0.635] | 0.538 | 0.505 | 2000 | 2000 | 9814 | 공통방향편향 | 0.843 | 0.562 | 0.544 | 0.341 | 0.500 | 0.500 | partial_collapse |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | 0.666 | [0.626, 0.715] | 0.558 | 0.507 | 2000 | 2000 | 6184 | 공통방향편향 | 0.786 | 0.654 | 0.578 | 0.458 | 0.500 | 0.500 | reproduced_weak |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.745 | [0.722, 0.766] | 0.545 | 0.504 | 2000 | 2000 | 15516 | 공통방향편향, 주 판정 아님 | 0.839 | 0.815 | 0.630 | 0.781 | 0.500 | 0.500 | reported_unmatched_path |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.698 | [0.650, 0.732] | 0.581 | 0.504 | 2000 | 2000 | 15722 | 공통방향편향, 주 판정 아님 | 0.861 | 0.727 | 0.609 | 0.637 | 0.500 | 0.500 | reported_unmatched_path |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.692 | [0.649, 0.736] | 0.569 | 0.506 | 2000 | 2000 | 7164 | 공통방향편향, 주 판정 아님 | 0.816 | 0.719 | 0.610 | 0.607 | 0.500 | 0.500 | reported_unmatched_path |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.744 | [0.722, 0.764] | 0.545 | 0.504 | 2000 | 2000 | 15480 | 공통방향편향, 주 판정 아님 | 0.836 | 0.814 | 0.628 | 0.780 | 0.500 | 0.500 | reported_unmatched_path |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.702 | [0.651, 0.738] | 0.581 | 0.504 | 2000 | 2000 | 15698 | 공통방향편향, 주 판정 아님 | 0.864 | 0.726 | 0.615 | 0.638 | 0.500 | 0.500 | reported_unmatched_path |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.693 | [0.653, 0.736] | 0.568 | 0.506 | 2000 | 2000 | 7160 | 공통방향편향, 주 판정 아님 | 0.815 | 0.715 | 0.606 | 0.602 | 0.500 | 0.500 | reported_unmatched_path |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.714 | [0.703, 0.727] | 0.520 | 0.504 | 2000 | 2000 | 20304 | 공통방향편향, 주 판정 아님 | 0.813 | 0.721 | 0.610 | 0.650 | 0.500 | 0.500 | reported_unmatched_path |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.709 | [0.701, 0.716] | 0.513 | 0.503 | 2000 | 2000 | 40098 | 공통방향편향, 주 판정 아님 | 0.845 | 0.606 | 0.617 | 0.415 | 0.500 | 0.500 | reported_unmatched_path |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.760 | [0.749, 0.772] | 0.521 | 0.503 | 2000 | 2000 | 30288 | 공통방향편향, 주 판정 아님 | 0.912 | 0.746 | 0.667 | 0.415 | 0.500 | 0.500 | reported_unmatched_path |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | 0.652 | [0.638, 0.667] | 0.518 | 0.504 | 2000 | 2000 | 13812 | 공통방향편향 | 0.744 | 0.616 | 0.573 | 0.482 | 0.500 | 0.500 | partial_collapse |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | 0.700 | [0.693, 0.708] | 0.512 | 0.503 | 2000 | 2000 | 36526 | 공통방향편향 | 0.837 | 0.580 | 0.611 | 0.355 | 0.500 | 0.500 | reproduced_weak |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | 0.760 | [0.748, 0.773] | 0.520 | 0.503 | 2000 | 2000 | 29594 | 공통방향편향 | 0.912 | 0.744 | 0.667 | 0.404 | 0.500 | 0.500 | reproduced_strong |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.744 | [0.722, 0.764] | 0.545 | 0.504 | 2000 | 2000 | 15480 | 공통방향편향, 주 판정 아님 | 0.836 | 0.814 | 0.628 | 0.780 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.702 | [0.651, 0.738] | 0.581 | 0.504 | 2000 | 2000 | 15698 | 공통방향편향, 주 판정 아님 | 0.864 | 0.726 | 0.615 | 0.638 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.693 | [0.653, 0.736] | 0.568 | 0.506 | 2000 | 2000 | 7160 | 공통방향편향, 주 판정 아님 | 0.815 | 0.715 | 0.606 | 0.602 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | 0.649 | [0.626, 0.670] | 0.530 | 0.506 | 2000 | 2000 | 7260 | 공통방향편향 | 0.714 | 0.644 | 0.576 | 0.509 | 0.500 | 0.500 | partial_collapse |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | 0.639 | [0.609, 0.667] | 0.537 | 0.505 | 2000 | 2000 | 9496 | 공통방향편향 | 0.834 | 0.595 | 0.559 | 0.394 | 0.500 | 0.500 | reproduced_weak |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | 0.677 | [0.639, 0.724] | 0.558 | 0.506 | 2000 | 2000 | 6200 | 공통방향편향 | 0.796 | 0.691 | 0.591 | 0.542 | 0.500 | 0.500 | reproduced_weak |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | 0.744 | [0.724, 0.765] | 0.547 | 0.504 | 2000 | 2000 | 15480 | 공통방향편향, 주 판정 아님 | 0.837 | 0.814 | 0.630 | 0.780 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | 0.701 | [0.650, 0.735] | 0.587 | 0.504 | 2000 | 2000 | 15698 | 공통방향편향, 주 판정 아님 | 0.864 | 0.725 | 0.612 | 0.638 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | 0.695 | [0.654, 0.740] | 0.583 | 0.506 | 2000 | 2000 | 7160 | 공통방향편향, 주 판정 아님 | 0.815 | 0.716 | 0.605 | 0.603 | 0.500 | 0.500 | reported_unmatched_path |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | 0.892 | [0.869, 0.914] | 0.578 | 0.510 | 2000 | 2000 | 2770 | 경로지름길 | 0.982 | 0.846 | - | 0.849 | 0.500 | 0.500 | descriptive_only |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | 0.918 | [0.891, 0.937] | 0.609 | 0.512 | 2000 | 2000 | 1740 | - | 0.988 | 0.823 | - | 0.702 | 0.500 | 0.500 | reproduced_strong |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | 0.868 | [0.801, 0.919] | 0.617 | 0.515 | 2000 | 2000 | 1014 | - | 0.910 | 0.734 | - | 0.756 | 0.500 | 0.500 | reproduced_strong |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | 0.930 | [0.881, 0.955] | 0.688 | 0.520 | 2000 | 2000 | 642 | - | 0.969 | 0.918 | - | 0.596 | 0.500 | 0.500 | reproduced_unknown_reference |

**기준선 표본 수와 CNN 대비 쌍체 ΔAUROC** (Δ = CNN - 기준선, 시드 층화 그룹 클러스터 부트스트랩)

| 외부 세트 | 실행 | 층 | 기준선 | 평가 n | fit n | 시드 수 | CNN과 같은 행 | ΔAUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 14556 | 8502 | 5 | 예 | -0.075 [-0.119, -0.023] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 14556 | 8502 | 5 | 예 | -0.117 [-0.136, -0.094] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 14556 | 8502 | 5 | 예 | 0.233 [0.212, 0.255] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 14556 | 4000 | 5 | 예 | 0.100 [0.081, 0.117] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | path-shape LR | 14556 | 8502 | 5 | 예 | -0.050 [-0.101, 0.007] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 14556 | 8502 | 5 | 예 | 0.233 [0.212, 0.255] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 16290 | 6496 | 5 | 예 | -0.032 [-0.087, 0.049] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 16290 | 6496 | 5 | 예 | -0.201 [-0.248, -0.164] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 16290 | 6496 | 5 | 예 | 0.175 [0.120, 0.212] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 16290 | 4000 | 5 | 예 | 0.077 [0.055, 0.093] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | path-shape LR | 16290 | 6496 | 5 | 예 | 0.066 [-0.032, 0.201] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 16290 | 6496 | 5 | 예 | 0.175 [0.120, 0.212] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7198 | 1932 | 5 | 예 | 0.003 [-0.058, 0.058] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7198 | 1932 | 5 | 예 | -0.125 [-0.186, -0.068] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7198 | 1932 | 5 | 예 | 0.185 [0.143, 0.230] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7198 | 1932 | 5 | 예 | 0.087 [0.047, 0.108] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | path-shape LR | 7198 | 1932 | 5 | 예 | 0.151 [0.074, 0.254] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7198 | 1932 | 5 | 예 | 0.185 [0.143, 0.230] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | byte-hist LR | 6074 | 8502 | 5 | 예 | 0.020 [-0.011, 0.051] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | char n-gram LR | 6074 | 8502 | 5 | 예 | -0.087 [-0.108, -0.060] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | length LR (게이트) | 6074 | 8502 | 5 | 예 | 0.136 [0.117, 0.156] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | motif-hist LR (3x3) | 6074 | 4000 | 5 | 예 | 0.062 [0.045, 0.081] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | path-shape LR | 6074 | 8502 | 5 | 예 | 0.160 [0.129, 0.189] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | version LR (게이트) | 6074 | 8502 | 5 | 예 | 0.136 [0.117, 0.156] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | byte-hist LR | 9814 | 6496 | 5 | 예 | 0.042 [-0.013, 0.095] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | char n-gram LR | 9814 | 6496 | 5 | 예 | -0.238 [-0.283, -0.193] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | length LR (게이트) | 9814 | 6496 | 5 | 예 | 0.104 [0.073, 0.135] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | motif-hist LR (3x3) | 9814 | 4000 | 5 | 예 | 0.061 [0.037, 0.086] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | path-shape LR | 9814 | 6496 | 5 | 예 | 0.264 [0.184, 0.330] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | version LR (게이트) | 9814 | 6496 | 5 | 예 | 0.104 [0.073, 0.135] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | byte-hist LR | 6184 | 1932 | 5 | 예 | 0.012 [-0.055, 0.066] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | char n-gram LR | 6184 | 1932 | 5 | 예 | -0.120 [-0.193, -0.054] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | length LR (게이트) | 6184 | 1932 | 5 | 예 | 0.166 [0.126, 0.215] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | motif-hist LR (3x3) | 6184 | 1932 | 5 | 예 | 0.087 [0.046, 0.111] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | path-shape LR | 6184 | 1932 | 5 | 예 | 0.208 [0.130, 0.330] |
| primary, CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | version LR (게이트) | 6184 | 1932 | 5 | 예 | 0.166 [0.126, 0.215] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15516 | 8502 | 5 | 예 | -0.070 [-0.112, -0.022] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15516 | 8502 | 5 | 예 | -0.094 [-0.114, -0.071] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15516 | 8502 | 5 | 예 | 0.245 [0.222, 0.266] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 15516 | 4000 | 5 | 예 | 0.115 [0.095, 0.133] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | path-shape LR | 15516 | 8502 | 5 | 예 | -0.036 [-0.085, 0.018] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15516 | 8502 | 5 | 예 | 0.245 [0.222, 0.266] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15722 | 6496 | 5 | 예 | -0.029 [-0.079, 0.041] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15722 | 6496 | 5 | 예 | -0.164 [-0.211, -0.127] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15722 | 6496 | 5 | 예 | 0.198 [0.150, 0.232] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 15722 | 4000 | 5 | 예 | 0.088 [0.069, 0.105] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | path-shape LR | 15722 | 6496 | 5 | 예 | 0.061 [-0.023, 0.175] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15722 | 6496 | 5 | 예 | 0.198 [0.150, 0.232] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7164 | 1932 | 5 | 예 | -0.027 [-0.082, 0.022] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7164 | 1932 | 5 | 예 | -0.125 [-0.183, -0.068] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7164 | 1932 | 5 | 예 | 0.192 [0.149, 0.236] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7164 | 1932 | 5 | 예 | 0.082 [0.043, 0.104] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | path-shape LR | 7164 | 1932 | 5 | 예 | 0.085 [0.007, 0.182] |
| primary, benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7164 | 1932 | 5 | 예 | 0.192 [0.149, 0.236] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15480 | 8502 | 5 | 예 | -0.070 [-0.109, -0.022] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15480 | 8502 | 5 | 예 | -0.093 [-0.112, -0.070] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.222, 0.264] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 15480 | 4000 | 5 | 예 | 0.115 [0.098, 0.133] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | path-shape LR | 15480 | 8502 | 5 | 예 | -0.036 [-0.081, 0.016] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.222, 0.264] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15698 | 6496 | 5 | 예 | -0.025 [-0.075, 0.046] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15698 | 6496 | 5 | 예 | -0.162 [-0.209, -0.124] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15698 | 6496 | 5 | 예 | 0.202 [0.151, 0.238] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 15698 | 4000 | 5 | 예 | 0.086 [0.068, 0.103] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | path-shape LR | 15698 | 6496 | 5 | 예 | 0.063 [-0.026, 0.179] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15698 | 6496 | 5 | 예 | 0.202 [0.151, 0.238] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7160 | 1932 | 5 | 예 | -0.021 [-0.075, 0.029] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7160 | 1932 | 5 | 예 | -0.121 [-0.180, -0.063] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7160 | 1932 | 5 | 예 | 0.193 [0.153, 0.236] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7160 | 1932 | 5 | 예 | 0.087 [0.047, 0.109] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | path-shape LR | 7160 | 1932 | 5 | 예 | 0.091 [0.017, 0.193] |
| primary, benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7160 | 1932 | 5 | 예 | 0.193 [0.153, 0.236] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 20304 | 8502 | 5 | 예 | -0.007 [-0.025, 0.009] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 20304 | 8502 | 5 | 예 | -0.099 [-0.107, -0.089] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 20304 | 8502 | 5 | 예 | 0.214 [0.203, 0.227] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 20304 | 4000 | 5 | 예 | 0.105 [0.093, 0.118] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | path-shape LR | 20304 | 8502 | 5 | 예 | 0.064 [0.045, 0.083] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 20304 | 8502 | 5 | 예 | 0.214 [0.203, 0.227] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 40098 | 6496 | 5 | 예 | 0.103 [0.082, 0.122] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 40098 | 6496 | 5 | 예 | -0.137 [-0.149, -0.124] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 40098 | 6496 | 5 | 예 | 0.209 [0.201, 0.216] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 40098 | 4000 | 5 | 예 | 0.092 [0.082, 0.102] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | path-shape LR | 40098 | 6496 | 5 | 예 | 0.294 [0.264, 0.320] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 40098 | 6496 | 5 | 예 | 0.209 [0.201, 0.216] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 30288 | 1932 | 5 | 예 | 0.014 [-0.008, 0.037] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 30288 | 1932 | 5 | 예 | -0.151 [-0.163, -0.140] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 30288 | 1932 | 5 | 예 | 0.260 [0.249, 0.272] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 30288 | 1932 | 5 | 예 | 0.093 [0.080, 0.107] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | path-shape LR | 30288 | 1932 | 5 | 예 | 0.346 [0.310, 0.381] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 30288 | 1932 | 5 | 예 | 0.260 [0.249, 0.272] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | byte-hist LR | 13812 | 8502 | 5 | 예 | 0.036 [0.018, 0.053] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | char n-gram LR | 13812 | 8502 | 5 | 예 | -0.092 [-0.102, -0.082] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | length LR (게이트) | 13812 | 8502 | 5 | 예 | 0.152 [0.138, 0.167] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | motif-hist LR (3x3) | 13812 | 4000 | 5 | 예 | 0.078 [0.063, 0.095] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | path-shape LR | 13812 | 8502 | 5 | 예 | 0.170 [0.152, 0.190] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | version LR (게이트) | 13812 | 8502 | 5 | 예 | 0.152 [0.138, 0.167] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | byte-hist LR | 36526 | 6496 | 5 | 예 | 0.121 [0.101, 0.139] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | char n-gram LR | 36526 | 6496 | 5 | 예 | -0.136 [-0.149, -0.123] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | length LR (게이트) | 36526 | 6496 | 5 | 예 | 0.200 [0.193, 0.208] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | motif-hist LR (3x3) | 36526 | 4000 | 5 | 예 | 0.089 [0.079, 0.100] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | path-shape LR | 36526 | 6496 | 5 | 예 | 0.346 [0.319, 0.369] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | version LR (게이트) | 36526 | 6496 | 5 | 예 | 0.200 [0.193, 0.208] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | byte-hist LR | 29594 | 1932 | 5 | 예 | 0.016 [-0.006, 0.040] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | char n-gram LR | 29594 | 1932 | 5 | 예 | -0.152 [-0.164, -0.140] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | length LR (게이트) | 29594 | 1932 | 5 | 예 | 0.260 [0.248, 0.273] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | motif-hist LR (3x3) | 29594 | 1932 | 5 | 예 | 0.093 [0.078, 0.108] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | path-shape LR | 29594 | 1932 | 5 | 예 | 0.356 [0.318, 0.392] |
| secondary, Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | version LR (게이트) | 29594 | 1932 | 5 | 예 | 0.260 [0.248, 0.273] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15480 | 8502 | 5 | 예 | -0.070 [-0.109, -0.022] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15480 | 8502 | 5 | 예 | -0.093 [-0.112, -0.070] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.222, 0.264] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 15480 | 4000 | 5 | 예 | 0.115 [0.098, 0.133] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | path-shape LR | 15480 | 8502 | 5 | 예 | -0.036 [-0.081, 0.016] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.222, 0.264] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15698 | 6496 | 5 | 예 | -0.025 [-0.075, 0.046] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15698 | 6496 | 5 | 예 | -0.162 [-0.209, -0.124] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15698 | 6496 | 5 | 예 | 0.202 [0.151, 0.238] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 15698 | 4000 | 5 | 예 | 0.086 [0.068, 0.103] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | path-shape LR | 15698 | 6496 | 5 | 예 | 0.063 [-0.026, 0.179] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15698 | 6496 | 5 | 예 | 0.202 [0.151, 0.238] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7160 | 1932 | 5 | 예 | -0.021 [-0.075, 0.029] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7160 | 1932 | 5 | 예 | -0.121 [-0.180, -0.063] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7160 | 1932 | 5 | 예 | 0.193 [0.153, 0.236] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7160 | 1932 | 5 | 예 | 0.087 [0.047, 0.109] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | path-shape LR | 7160 | 1932 | 5 | 예 | 0.091 [0.017, 0.193] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7160 | 1932 | 5 | 예 | 0.193 [0.153, 0.236] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | byte-hist LR | 7260 | 8502 | 5 | 예 | 0.005 [-0.028, 0.037] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | char n-gram LR | 7260 | 8502 | 5 | 예 | -0.065 [-0.087, -0.042] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | length LR (게이트) | 7260 | 8502 | 5 | 예 | 0.149 [0.126, 0.170] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | motif-hist LR (3x3) | 7260 | 4000 | 5 | 예 | 0.073 [0.052, 0.091] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | path-shape LR | 7260 | 8502 | 5 | 예 | 0.140 [0.106, 0.174] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v2 | version LR (게이트) | 7260 | 8502 | 5 | 예 | 0.149 [0.126, 0.170] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | byte-hist LR | 9496 | 6496 | 5 | 예 | 0.044 [-0.012, 0.098] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | char n-gram LR | 9496 | 6496 | 5 | 예 | -0.195 [-0.236, -0.153] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | length LR (게이트) | 9496 | 6496 | 5 | 예 | 0.139 [0.109, 0.167] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | motif-hist LR (3x3) | 9496 | 4000 | 5 | 예 | 0.080 [0.057, 0.103] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | path-shape LR | 9496 | 6496 | 5 | 예 | 0.244 [0.166, 0.312] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v3 | version LR (게이트) | 9496 | 6496 | 5 | 예 | 0.139 [0.109, 0.167] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | byte-hist LR | 6200 | 1932 | 5 | 예 | -0.015 [-0.076, 0.035] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | char n-gram LR | 6200 | 1932 | 5 | 예 | -0.119 [-0.190, -0.057] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | length LR (게이트) | 6200 | 1932 | 5 | 예 | 0.177 [0.139, 0.224] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | motif-hist LR (3x3) | 6200 | 1932 | 5 | 예 | 0.086 [0.041, 0.110] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | path-shape LR | 6200 | 1932 | 5 | 예 | 0.135 [0.059, 0.259] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort), 길이+경로 매칭 | v4 | version LR (게이트) | 6200 | 1932 | 5 | 예 | 0.177 [0.139, 0.224] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | byte-hist LR | 15480 | 8502 | 5 | 예 | -0.069 [-0.110, -0.021] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | char n-gram LR | 15480 | 8502 | 5 | 예 | -0.093 [-0.112, -0.070] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | length LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.224, 0.265] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | motif-hist LR (3x3) | 15480 | 4000 | 5 | 예 | 0.114 [0.097, 0.131] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | path-shape LR | 15480 | 8502 | 5 | 예 | -0.036 [-0.082, 0.017] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | version LR (게이트) | 15480 | 8502 | 5 | 예 | 0.244 [0.224, 0.265] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | byte-hist LR | 15698 | 6496 | 5 | 예 | -0.025 [-0.076, 0.047] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | char n-gram LR | 15698 | 6496 | 5 | 예 | -0.164 [-0.210, -0.127] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | length LR (게이트) | 15698 | 6496 | 5 | 예 | 0.201 [0.150, 0.235] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | motif-hist LR (3x3) | 15698 | 4000 | 5 | 예 | 0.089 [0.071, 0.103] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | path-shape LR | 15698 | 6496 | 5 | 예 | 0.063 [-0.018, 0.181] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | version LR (게이트) | 15698 | 6496 | 5 | 예 | 0.201 [0.150, 0.235] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | byte-hist LR | 7160 | 1932 | 5 | 예 | -0.020 [-0.076, 0.029] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | char n-gram LR | 7160 | 1932 | 5 | 예 | -0.120 [-0.178, -0.063] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | length LR (게이트) | 7160 | 1932 | 5 | 예 | 0.195 [0.154, 0.240] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | motif-hist LR (3x3) | 7160 | 1932 | 5 | 예 | 0.091 [0.053, 0.112] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | path-shape LR | 7160 | 1932 | 5 | 예 | 0.093 [0.014, 0.194] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | version LR (게이트) | 7160 | 1932 | 5 | 예 | 0.195 [0.154, 0.240] |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | byte-hist LR | 2770 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | char n-gram LR | 2770 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | length LR (게이트) | 2770 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | path-shape LR | 2770 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | version LR (게이트) | 2770 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | byte-hist LR | 1740 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | char n-gram LR | 1740 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | length LR (게이트) | 1740 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | path-shape LR | 1740 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | version LR (게이트) | 1740 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | byte-hist LR | 1014 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | char n-gram LR | 1014 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | length LR (게이트) | 1014 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | path-shape LR | 1014 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | version LR (게이트) | 1014 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | byte-hist LR | 642 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | char n-gram LR | 642 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | length LR (게이트) | 642 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | path-shape LR | 642 | - | - | 예 | - |
| primary, OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | version LR (게이트) | 642 | - | - | 예 | - |
**motif 재현성 (설계 4절)**

| 소스 | cohort | 층 | Spearman ρ (512종) | 부호 일치율 (상위 20) | Jaccard (상위 20) | 판정 |
|---|---|---|---|---|---|---|
| external_primary_2026-09-08 | 고정 | v2 | 0.228 [0.113, 0.256] | 0.850 [0.550, 0.900] | 0.026 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08 | 시드별 | v2 | 0.316 [0.167, 0.385] | 0.800 [0.550, 0.951] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08 | 고정 | v3 | 0.066 [0.019, 0.109] | 0.500 [0.399, 0.700] | 0.053 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08 | 시드별 | v3 | 0.169 [0.074, 0.197] | 0.700 [0.500, 0.750] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08 | 고정 | v4 | 0.073 [-0.018, 0.142] | 0.650 [0.500, 0.800] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08 | 시드별 | v4 | 0.097 [-0.010, 0.149] | 0.650 [0.500, 0.800] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__ccunranked | 고정 | v2 | 0.253 [0.158, 0.266] | 0.800 [0.650, 0.901] | 0.081 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__ccunranked | 고정 | v3 | 0.025 [-0.006, 0.053] | 0.450 [0.350, 0.600] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__ccunranked | 고정 | v4 | 0.061 [-0.026, 0.122] | 0.750 [0.500, 0.850] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v2 | 0.306 [0.196, 0.376] | 0.700 [0.550, 0.950] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v3 | 0.161 [0.068, 0.185] | 0.600 [0.450, 0.701] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v4 | 0.097 [0.002, 0.146] | 0.650 [0.450, 0.750] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v2 | 0.316 [0.167, 0.385] | 0.800 [0.550, 0.951] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v3 | 0.169 [0.074, 0.197] | 0.700 [0.500, 0.750] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v4 | 0.097 [-0.010, 0.149] | 0.650 [0.500, 0.800] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__secondary | 고정 | v2 | 0.324 [0.246, 0.335] | 0.900 [0.750, 1.000] | 0.026 (귀무 상한 0.081) | reproduced |
| external_primary_2026-09-08__secondary | 고정 | v3 | 0.211 [0.138, 0.253] | 0.750 [0.550, 0.850] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__secondary | 고정 | v4 | 0.255 [0.196, 0.275] | 0.650 [0.600, 0.800] | 0.026 (귀무 상한 0.081) | partial |

### 템플릿 누출 쌍체 비교 (G 재설계, 고정 캠페인 T)

| 지표 | v2 | v3 | v4 |
|---|---|---|---|
| AUROC - Model A_sm (누출 허용, 크기 매칭) | 0.840 | 0.775 | 0.814 |
| AUROC - Model B (완전 격리) | 0.851 | 0.774 | 0.828 |
| **ΔAUROC (A_sm-B), 쌍체 95% CI** - 주 지표 | -0.010 [-0.021, 0.001] | +0.001 [-0.010, 0.015] | -0.014 [-0.042, 0.015] |
| 순열 p (클러스터, 양측) | 0.496 | 0.946 | 0.742 |
| ΔAUROC - 누출 행 대 무누출 클래스 대조 (A_sm-B) | +0.017 [-0.047, 0.091] | +0.064 [-0.020, 0.164] | +0.028 [-0.057, 0.093] |
| AUROC - Model A (크기 미매칭, 보조) | 0.840 | 0.787 | 0.838 |
| ΔAUROC (A-B), 쌍체 95% CI - 보조 | -0.011 [-0.021, -0.001] | +0.014 [0.003, 0.024] | +0.009 [-0.010, 0.029] |
| ΔAUROC - 누출 행 대 무누출 클래스 대조 (A-B) | -0.007 [-0.077, 0.083] | -0.005 [-0.146, 0.107] | +0.033 [-0.022, 0.099] |
| Δ char n-gram LR (A_sm-B) | -0.000 [-0.001, 0.000] | +0.001 [0.000, 0.001] | -0.000 [-0.001, 0.000] |
| Δ byte-hist LR (A_sm-B) | +0.002 [-0.000, 0.004] | +0.000 [-0.001, 0.002] | +0.001 [-0.000, 0.003] |
| T 표본 수 | 1688 | 1273 | 378 |
| 그중 A train에 형제가 있는 행 | 38 | 35 | 15 |
| A train의 T 템플릿 중복 행 수 | 44 | 32 | 15 |
| B train의 T 템플릿 중복 행 수 | 0 | 0 | 0 |
| 판정 - 주 지표 (A_sm-B) | 누출 무시 가능 (CI가 0을 포함하거나 하한이 0 이하) | 누출 무시 가능 (CI가 0을 포함하거나 하한이 0 이하) | 누출 무시 가능 (CI가 0을 포함하거나 하한이 0 이하) |
| 판정 - 보조 (A-B) | - | - | - |
| 판정 - 누출 대조 (A_sm-B) | - | - | - |
| 판정 - 누출 대조 (A-B) | - | - | - |

