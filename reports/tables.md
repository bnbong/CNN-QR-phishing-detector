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
| char n-gram LR (디코딩 텍스트 참조 기준) | 0.969 | 0.972 | 0.992 |
| byte-hist LR | 0.861 | 0.766 | 0.771 |
| length LR | 0.500 | 0.500 | 0.500 |
| version LR | 0.500 | 0.500 | 0.500 |
| 참조 기준 − CNN 갭 | +0.097 | +0.158 | +0.187 |

### H4 공간 구조 검정 (CNN × MLP, 정상 배치 × shuffle-pos)

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| CNN · 정상 배치 | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| CNN · shuffle-pos | 0.765 [0.722, 0.803] | 0.671 [0.641, 0.703] | 0.656 [0.626, 0.689] |
| CNN 차이 | -0.107 | -0.143 | -0.149 |
| MLP · 정상 배치 | 0.834 [0.801, 0.862] | 0.732 [0.695, 0.773] | 0.690 [0.649, 0.727] |
| MLP · shuffle-pos | 0.834 [0.801, 0.863] | 0.733 [0.696, 0.772] | 0.706 [0.666, 0.744] |
| MLP 차이 | +0.000 | +0.001 | +0.016 |

### T4 길이 통제 절제 (H3)

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| L-exact (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| L-quantile | 0.853 [0.828, 0.878] | 0.817 [0.779, 0.854] | 0.797 [0.761, 0.829] |
| L-none | 0.877 [0.857, 0.894] | 0.816 [0.777, 0.853] | 0.829 [0.792, 0.863] |
| L-none − L-exact | +0.005 | +0.002 | +0.024 |
| PAD-rand (L-none 위) | 0.862 [0.840, 0.881] | 0.795 [0.755, 0.832] | 0.799 [0.759, 0.837] |
| PAD-rand − L-none | -0.015 | -0.021 | -0.030 |
| length LR @ L-exact | 0.500 | 0.500 | 0.500 |
| length LR @ L-none | 0.543 | 0.544 | 0.610 |
| version LR @ L-none | 0.500 | 0.500 | 0.500 |

### T5 표현·마스크·부분집합 절제

| 조건 | v2 | v3 | v4 |
|---|---|---|---|
| norm · mask-fixed · data_only (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| raw (정규화 해제) | 0.921 [0.906, 0.936] | 0.873 [0.833, 0.912] | 0.892 [0.867, 0.912] |
| raw − norm | +0.049 | +0.060 | +0.087 |
| feat-all (기능 패턴 포함) | 0.873 [0.847, 0.893] | 0.802 [0.766, 0.837] | 0.813 [0.783, 0.841] |
| feat-all − data_only | +0.001 | -0.012 | +0.008 |
| mask-auto | 0.748 [0.723, 0.771] | 0.678 [0.654, 0.702] | 0.675 [0.646, 0.702] |
| mask-auto − mask-fixed | -0.124 | -0.135 | -0.130 |
| maskindex LR @ mask-auto | 0.524 | 0.486 | 0.517 |
| mask-off (언마스킹) | 0.871 [0.843, 0.893] | 0.803 [0.757, 0.845] | 0.816 [0.784, 0.843] |
| mask-off − mask-fixed | -0.001 | -0.011 | +0.011 |
| path+ 부분집합 | 0.727 [0.677, 0.779] | 0.766 [0.713, 0.816] | 0.822 [0.784, 0.853] |
| path+ 참조 기준 (char n-gram LR) | 0.914 | 0.968 | 0.988 |
| EC-M (v3 한정) | — | 0.847 [0.816, 0.871] | — |
| EC-M 참조 기준 (char n-gram LR) | — | 0.961 | — |

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
| char (URL 문자) · enrichment | 1.57 | 1.48 | 1.43 |
| char (URL 문자) · 난수 기준선 | 1.00 | 1.00 | 1.00 |
| pad (패딩) · enrichment | 1.13 | 0.98 | 0.99 |
| pad (패딩) · 난수 기준선 | 1.00 | 1.00 | 1.00 |
| ec (오류정정) · enrichment | 0.68 | 0.68 | 0.61 |
| ec (오류정정) · 난수 기준선 | 1.00 | 1.00 | 1.00 |
| function (기능 패턴) · enrichment | 0.66 | 0.60 | 0.49 |
| function (기능 패턴) · 난수 기준선 | 1.00 | 1.00 | 1.00 |
| length_header · enrichment | — | — | — |
| length_header · 난수 기준선 | — | — | — |
| mode_header · enrichment | — | — | — |
| mode_header · 난수 기준선 | — | — | — |
| remainder · enrichment | 0.18 | 0.34 | 0.19 |
| remainder · 난수 기준선 | 1.00 | 0.99 | 1.00 |
| 무작위화 모델 CAM 상관 (sanity) | 0.076 | 0.021 | -0.067 |

### Bag-of-QR-patches (Q1 직접 측정)

| 표현 / 모델 | v2 | v3 | v4 |
|---|---|---|---|
| patch 2×2 히스토그램 LR | 0.526 [0.494, 0.559] | 0.567 [0.521, 0.614] | 0.573 [0.525, 0.622] |
| patch 3×3 히스토그램 LR | 0.740 [0.705, 0.771] | 0.707 [0.667, 0.743] | 0.715 [0.676, 0.750] |
| patch 3×3 spatial pyramid LR | 0.814 [0.792, 0.834] | 0.712 [0.685, 0.742] | 0.743 [0.708, 0.775] |
| patch 3×3 · within-QR 모듈 셔플 null | 0.498 [0.486, 0.510] | 0.502 [0.489, 0.516] | 0.510 [0.485, 0.535] |
| patch 3×3 · 코드워드 순서 셔플 null | 0.547 [0.534, 0.562] | 0.539 [0.519, 0.561] | 0.555 [0.529, 0.585] |
| patch 3×3 · 라벨 셔플 (바닥) | 0.488 [0.471, 0.508] | 0.488 [0.462, 0.512] | 0.530 [0.500, 0.560] |
| SmallCNN (주 조건) | 0.872 [0.846, 0.892] | 0.814 [0.773, 0.853] | 0.805 [0.771, 0.837] |
| byte-hist LR | 0.861 | 0.766 | 0.771 |
| BitMLP | 0.834 [0.801, 0.862] | 0.732 [0.695, 0.773] | 0.690 [0.649, 0.727] |
| 3×3 창 개수 (평균) | 219 | 395 | 603 |

### 어휘 프로브 (RQ2 · D안)

| 어휘 프로브 | v2 | v3 | v4 |
|---|---|---|---|
| ① 선형으로 접근 가능한가 (`accessible`) | 45 / 91 (49.5%) | 30 / 114 (26.3%) | 6 / 104 (5.8%) |
| ② 학습이 접근성을 높였는가 (`learned_gain`) | 2 / 91 (2.2%) | 0 / 114 (0.0%) | 0 / 104 (0.0%) |
| ③ 둘 다 (`accessible_and_gained`, 옛 significant) | 2 / 91 (2.2%) | 0 / 114 (0.0%) | 0 / 104 (0.0%) |
| ①만 만족 (②는 아님) | 43 | 30 | 6 |
| ④ 실제 분류 결정에 쓰이는가 (`used_in_decision`) | 측정 불가 | 측정 불가 | 측정 불가 |
| 라벨 프로브 (참고 상한선) | 0.872 | 0.813 | 0.804 |

① `accessible` = 전 시드에서 학습 표현 프로브 CI 하한 > 셔플 기준선 CI 상한. ② `learned_gain` = 전 시드에서 학습 표현 프로브 CI 하한 > 무작위 초기화 CNN 기준선 CI 상한. ④는 동결 표현 위의 선형 프로브로는 답할 수 없다 — 프로브는 표현을 읽을 뿐 결정 경로에 개입하지 않는다.


**v2 상위 목표 — ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| ngram:jp. | auroc | 0.909 | 0.497 | 0.849 |
| ngram:.bl | auroc | 0.866 | 0.446 | 0.878 |
| ngram:pot | auroc | 0.855 | 0.468 | 0.862 |
| ngram:x.h | auroc | 0.841 | 0.503 | 0.812 |
| ngram:.ht | auroc | 0.821 | 0.509 | 0.743 |
| keyword:.html | auroc | 0.820 | 0.509 | 0.774 |
| ngram:htm | auroc | 0.817 | 0.494 | 0.736 |
| ngram:ne. | auroc | 0.816 | 0.522 | 0.799 |
| ngram:.ru | auroc | 0.813 | 0.480 | 0.725 |
| ngram:gsp | auroc | 0.810 | 0.434 | 0.846 |

**v2 상위 목표 — ③ 둘 다 만족**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| has_path | auroc | 0.889 | 0.511 | 0.694 |
| path_depth | r2 | 0.245 | -0.059 | 0.078 |

**v3 상위 목표 — ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| keyword:http | auroc | 0.862 | 0.543 | 0.905 |
| has_path | auroc | 0.797 | 0.461 | 0.676 |
| keyword:login | auroc | 0.784 | 0.474 | 0.649 |
| ngram:/lo | auroc | 0.783 | 0.481 | 0.667 |
| ngram:htm | auroc | 0.758 | 0.495 | 0.631 |
| ngram:x.h | auroc | 0.756 | 0.515 | 0.658 |
| ngram:ogi | auroc | 0.756 | 0.477 | 0.645 |
| ngram:.ht | auroc | 0.748 | 0.514 | 0.628 |
| keyword:.php | auroc | 0.745 | 0.497 | 0.636 |
| ngram:.ph | auroc | 0.743 | 0.499 | 0.634 |

**v4 상위 목표 — ①만 만족(정보는 읽히지만 학습 이득은 미확인)**

| 목표 | 지표 | 점수 | 셔플 | 무작위 초기화 |
|---|---|---|---|---|
| ngram:htm | auroc | 0.676 | 0.483 | 0.624 |
| ngram:tml | auroc | 0.670 | 0.507 | 0.618 |
| has_digit | auroc | 0.661 | 0.504 | 0.696 |
| keyword:.html | auroc | 0.655 | 0.505 | 0.621 |
| ngram:.ht | auroc | 0.655 | 0.479 | 0.629 |
| digit_ratio | r2 | 0.227 | -0.117 | 0.028 |

### 인과 motif 절제 (RQ3 · C안)


**v2** (원본 AUROC 0.872)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.016 | — | 0.005 | — | -0.290 | -0.729 | +0.149 | 3.5 |
| random (phishing 개수 맞춤) | -0.002 | — | 0.003 | — | +0.136 | -0.199 | +0.471 | 3.5 |
| benign motif | -0.020 | — | 0.009 | — | +0.566 | +0.121 | +1.011 | 3.8 |
| random (benign 개수 맞춤) | -0.002 | — | 0.002 | — | +0.280 | -0.107 | +0.667 | 3.8 |
| random (phishing topology 맞춤) | -0.002 | — | 0.003 | — | +0.118 | -0.231 | +0.466 | 3.5 |
| random (benign topology 맞춤) | -0.004 | — | 0.002 | — | +0.356 | -0.055 | +0.766 | 3.8 |

**v3** (원본 AUROC 0.814)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.025 | — | 0.011 | — | +0.176 | -0.100 | +0.452 | 6.9 |
| random (phishing 개수 맞춤) | -0.004 | — | 0.001 | — | +0.184 | -0.072 | +0.439 | 6.9 |
| benign motif | -0.027 | — | 0.008 | — | +0.891 | +0.508 | +1.274 | 7.3 |
| random (benign 개수 맞춤) | -0.005 | — | 0.002 | — | +0.262 | -0.041 | +0.564 | 7.3 |
| random (phishing topology 맞춤) | -0.007 | — | 0.004 | — | +0.086 | -0.214 | +0.385 | 6.9 |
| random (benign topology 맞춤) | -0.008 | — | 0.002 | — | +0.342 | +0.008 | +0.677 | 7.3 |

**v4** (원본 AUROC 0.805)

| 조건 | ΔAUROC | 95% CI(시드 층화) | 시드 간 SD | 반복 간 SD | 로짓 Δ | 로짓 Δ(phishing) | 로짓 Δ(benign) | 개입한 모듈 수 |
|---|---|---|---|---|---|---|---|---|
| phishing motif | -0.037 | — | 0.013 | — | -0.691 | -0.962 | -0.420 | 11.9 |
| random (phishing 개수 맞춤) | -0.005 | — | 0.005 | — | -0.060 | -0.242 | +0.122 | 11.9 |
| benign motif | -0.050 | — | 0.015 | — | +0.753 | +0.433 | +1.072 | 15.6 |
| random (benign 개수 맞춤) | -0.008 | — | 0.005 | — | -0.026 | -0.249 | +0.197 | 15.6 |
| random (phishing topology 맞춤) | -0.005 | — | 0.005 | — | -0.142 | -0.328 | +0.045 | 11.9 |
| random (benign topology 맞춤) | -0.008 | — | 0.006 | — | +0.058 | -0.147 | +0.263 | 15.6 |

### 가설 검정 판정 (H1 ~ H4)

| 가설 | 층 | ΔAUROC 추정치 | 95% CI | pseudo-p (Holm) | 순열 p | 판정 |
|---|---|---|---|---|---|---|
| H1. CNN > 라벨 셔플 바닥 | v2 | +0.374 | [+0.344, +0.402] | 0.012 | <0.001 | 기각 |
| H1. CNN > 라벨 셔플 바닥 | v3 | +0.306 | [+0.265, +0.346] | 0.012 | <0.001 | 기각 |
| H1. CNN > 라벨 셔플 바닥 | v4 | +0.299 | [+0.253, +0.339] | 0.012 | <0.001 | 기각 |
| H2. 디코딩 텍스트 참조 기준 > CNN | v2 | +0.097 | [+0.076, +0.122] | 0.012 | — | 기각 |
| H2. 디코딩 텍스트 참조 기준 > CNN | v3 | +0.158 | [+0.124, +0.195] | 0.012 | — | 기각 |
| H2. 디코딩 텍스트 참조 기준 > CNN | v4 | +0.187 | [+0.156, +0.221] | 0.012 | — | 기각 |
| H3. L-none > L-exact | v2 | +0.005 | [-0.023, +0.036] | 1.000 | — | 비기각 |
| H3. L-none > L-exact | v3 | +0.002 | [-0.053, +0.060] | 1.000 | — | 비기각 |
| H3. L-none > L-exact | v4 | +0.024 | [-0.024, +0.071] | 1.000 | — | 비기각 |
| H4. 정상 배치 > shuffle-pos | v2 | +0.107 | [+0.082, +0.135] | 0.012 | <0.001 | 기각 |
| H4. 정상 배치 > shuffle-pos | v3 | +0.143 | [+0.108, +0.179] | 0.012 | <0.001 | 기각 |
| H4. 정상 배치 > shuffle-pos | v4 | +0.149 | [+0.115, +0.181] | 0.012 | <0.001 | 기각 |

### 외부 검증 전이 (F) · motif 재현성

| 외부 세트 | 실행 | 층 | CNN AUROC | 95% CI | 순열 바닥선(97.5%) | 행순열 바닥선(97.5%) | n_perm | n_boot | 평가 n | 플래그 | char n-gram LR | byte-hist LR | motif-hist LR (3x3) | path-shape LR | length LR (게이트) | version LR (게이트) | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.733 | [0.712, 0.755] | 0.545 | — | 200 | 2000 | 14556 | — | 0.853 | 0.817 | 0.645 | — | 0.500 | 0.500 | reproduced_weak |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.675 | [0.620, 0.712] | 0.585 | — | 200 | 2000 | 16290 | — | 0.879 | 0.719 | 0.602 | — | 0.500 | 0.500 | reproduced_weak |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.685 | [0.643, 0.730] | 0.575 | — | 200 | 2000 | 7198 | — | 0.807 | 0.689 | 0.594 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.745 | [0.722, 0.766] | 0.544 | — | 200 | 2000 | 15516 | — | 0.840 | 0.820 | 0.637 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.698 | [0.650, 0.732] | 0.580 | — | 200 | 2000 | 15722 | — | 0.866 | 0.737 | 0.617 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.692 | [0.649, 0.736] | 0.567 | — | 200 | 2000 | 7164 | — | 0.814 | 0.727 | 0.604 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.744 | [0.722, 0.764] | 0.543 | — | 200 | 2000 | 15480 | — | 0.838 | 0.818 | 0.636 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.702 | [0.651, 0.738] | 0.580 | — | 200 | 2000 | 15698 | — | 0.869 | 0.736 | 0.619 | — | 0.500 | 0.500 | reproduced_weak |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.693 | [0.653, 0.736] | 0.566 | — | 200 | 2000 | 7160 | — | 0.813 | 0.722 | 0.604 | — | 0.500 | 0.500 | reproduced_weak |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.714 | [0.703, 0.727] | 0.519 | — | 200 | 2000 | 20304 | — | 0.813 | 0.725 | 0.640 | — | 0.500 | 0.500 | reproduced_weak |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.709 | [0.701, 0.716] | 0.515 | — | 200 | 2000 | 40098 | — | 0.845 | 0.614 | 0.623 | — | 0.500 | 0.500 | reproduced_weak |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.760 | [0.749, 0.772] | 0.522 | — | 200 | 2000 | 30288 | — | 0.907 | 0.737 | 0.670 | — | 0.500 | 0.500 | reproduced_strong |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | 0.744 | [0.722, 0.764] | 0.543 | — | 200 | 2000 | 15480 | — | 0.838 | 0.818 | 0.636 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | 0.702 | [0.651, 0.738] | 0.580 | — | 200 | 2000 | 15698 | — | 0.869 | 0.736 | 0.619 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | 0.693 | [0.653, 0.736] | 0.566 | — | 200 | 2000 | 7160 | — | 0.813 | 0.722 | 0.604 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | 0.744 | [0.724, 0.765] | 0.542 | — | 200 | 2000 | 15480 | — | 0.838 | 0.818 | 0.636 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | 0.701 | [0.650, 0.735] | 0.582 | — | 200 | 2000 | 15698 | — | 0.869 | 0.736 | 0.619 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | 0.695 | [0.654, 0.740] | 0.584 | — | 200 | 2000 | 7160 | — | 0.813 | 0.722 | 0.604 | — | 0.500 | 0.500 | reproduced_weak |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | 0.892 | [0.869, 0.914] | 0.587 | — | 200 | 2000 | 2770 | — | 0.982 | 0.846 | — | — | 0.500 | 0.500 | reproduced_strong |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | 0.918 | [0.891, 0.937] | 0.613 | — | 200 | 2000 | 1740 | — | 0.988 | 0.823 | — | — | 0.500 | 0.500 | reproduced_strong |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | 0.868 | [0.801, 0.919] | 0.617 | — | 200 | 2000 | 1014 | — | 0.910 | 0.734 | — | — | 0.500 | 0.500 | reproduced_strong |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | 0.930 | [0.881, 0.955] | 0.688 | — | 200 | 2000 | 642 | — | 0.969 | 0.918 | — | — | 0.500 | 0.500 | reproduced_unknown_reference |

**기준선 표본 수와 CNN 대비 쌍체 ΔAUROC** (Δ = CNN − 기준선, 시드 층화 그룹 클러스터 부트스트랩)

| 외부 세트 | 실행 | 층 | 기준선 | 평가 n | fit n | 시드 수 | CNN과 같은 행 | ΔAUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 14556 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 14556 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 14556 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 14556 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 16290 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 16290 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 16290 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 16290 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7198 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7198 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7198 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7198 | — | — | 예 | — |
| primary · CC unranked benign(민감도) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7198 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15516 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15516 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15516 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15516 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15722 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15722 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15722 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15722 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7164 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7164 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7164 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7164 | — | — | 예 | — |
| primary · benign 정제 절제: 피싱 도메인 유지 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7164 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15480 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15480 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15480 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15480 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15698 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15698 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15698 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15698 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7160 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7160 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7160 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7160 | — | — | 예 | — |
| primary · benign 정제 절제: 호스팅 블록리스트 미적용 | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7160 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 20304 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 20304 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 20304 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 20304 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 40098 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 40098 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 40098 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 40098 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 30288 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 30288 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 30288 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| secondary · Phishing.Database(robustness) | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 30288 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | byte-hist LR | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | char n-gram LR | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | length LR (게이트) | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v2 | version LR (게이트) | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | byte-hist LR | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | char n-gram LR | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | length LR (게이트) | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v3 | version LR (게이트) | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | byte-hist LR | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | char n-gram LR | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | length LR (게이트) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | motif-hist LR (3x3) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (고정 cohort) | v4 | version LR (게이트) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | byte-hist LR | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | char n-gram LR | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | length LR (게이트) | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v2 | version LR (게이트) | 15480 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | byte-hist LR | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | char n-gram LR | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | length LR (게이트) | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | motif-hist LR (3x3) | 8000 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v3 | version LR (게이트) | 15698 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | byte-hist LR | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | char n-gram LR | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | length LR (게이트) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | motif-hist LR (3x3) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-a zero-shot (WebPhish→외부) (시드별 cohort) | v4 | version LR (게이트) | 7160 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | byte-hist LR | 2770 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | char n-gram LR | 2770 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | length LR (게이트) | 2770 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v2 | version LR (게이트) | 2770 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | byte-hist LR | 1740 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | char n-gram LR | 1740 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | length LR (게이트) | 1740 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v3 | version LR (게이트) | 1740 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | byte-hist LR | 1014 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | char n-gram LR | 1014 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | length LR (게이트) | 1014 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v4 | version LR (게이트) | 1014 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | byte-hist LR | 642 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | char n-gram LR | 642 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | length LR (게이트) | 642 | — | — | 예 | — |
| primary · OpenPhish 90일 × CC×Tranco benign | F-b in-domain (외부→외부) | v5plus | version LR (게이트) | 642 | — | — | 예 | — |
**motif 재현성 (설계 4절)**

| 소스 | cohort | 층 | Spearman ρ (512종) | 부호 일치율 (상위 20) | Jaccard (상위 20) | 판정 |
|---|---|---|---|---|---|---|
| external_primary_2026-09-08 | 고정(=시드별 동일) | v2 | 0.316 [0.167, 0.385] | 0.800 [0.550, 0.951] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08 | 고정(=시드별 동일) | v3 | 0.169 [0.074, 0.197] | 0.700 [0.500, 0.750] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08 | 고정(=시드별 동일) | v4 | 0.097 [-0.010, 0.149] | 0.650 [0.500, 0.800] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__ccunranked | 고정 | v2 | 0.352 [0.230, 0.406] | 0.850 [0.600, 1.000] | 0.111 (귀무 상한 0.081) | reproduced |
| external_primary_2026-09-08__ccunranked | 고정 | v3 | 0.127 [0.041, 0.150] | 0.650 [0.449, 0.700] | 0.111 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__ccunranked | 고정 | v4 | 0.086 [-0.009, 0.137] | 0.700 [0.500, 0.850] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v2 | 0.306 [0.196, 0.376] | 0.700 [0.550, 0.950] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v3 | 0.161 [0.068, 0.185] | 0.600 [0.450, 0.701] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__keep_phish_domains | 고정 | v4 | 0.097 [0.002, 0.146] | 0.650 [0.450, 0.750] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v2 | 0.316 [0.167, 0.385] | 0.800 [0.550, 0.951] | 0.053 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v3 | 0.169 [0.074, 0.197] | 0.700 [0.500, 0.750] | 0.081 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__no_hosting_blocklist | 고정 | v4 | 0.097 [-0.010, 0.149] | 0.650 [0.500, 0.800] | 0.000 (귀무 상한 0.081) | failed |
| external_primary_2026-09-08__secondary | 고정 | v2 | 0.430 [0.355, 0.427] | 1.000 [0.850, 1.000] | 0.111 (귀무 상한 0.081) | reproduced |
| external_primary_2026-09-08__secondary | 고정 | v3 | 0.288 [0.184, 0.327] | 0.800 [0.600, 0.900] | 0.026 (귀무 상한 0.081) | partial |
| external_primary_2026-09-08__secondary | 고정 | v4 | 0.255 [0.191, 0.280] | 0.700 [0.600, 0.800] | 0.000 (귀무 상한 0.081) | partial |

### 템플릿 누출 쌍체 비교 (G 재설계 · 고정 캠페인 T)

| 지표 | v2 | v3 | v4 |
|---|---|---|---|
| AUROC — Model A_sm (누출 허용, 크기 매칭) | 0.840 | 0.775 | 0.814 |
| AUROC — Model B (완전 격리) | 0.851 | 0.774 | 0.828 |
| **ΔAUROC (A_sm−B), 쌍체 95% CI** — 주 지표 | -0.010 [-0.021, 0.001] | +0.001 [-0.010, 0.015] | -0.014 [-0.042, 0.015] |
| 순열 p (클러스터, 양측) | 0.496 | 0.946 | 0.742 |
| ΔAUROC — 누출 행 대 무누출 클래스 대조 (A_sm−B) | +0.017 [-0.047, 0.091] | +0.064 [-0.020, 0.164] | +0.028 [-0.057, 0.093] |
| AUROC — Model A (크기 미매칭, 보조) | 0.840 | 0.787 | 0.838 |
| ΔAUROC (A−B), 쌍체 95% CI — 보조 | -0.011 [-0.021, -0.001] | +0.014 [0.003, 0.024] | +0.009 [-0.010, 0.029] |
| ΔAUROC — 누출 행 대 무누출 클래스 대조 (A−B) | -0.007 [-0.077, 0.083] | -0.005 [-0.146, 0.107] | +0.033 [-0.022, 0.099] |
| Δ char n-gram LR (A_sm−B) | -0.000 [-0.001, 0.000] | +0.001 [0.000, 0.001] | -0.000 [-0.001, 0.000] |
| Δ byte-hist LR (A_sm−B) | +0.002 [-0.000, 0.004] | +0.000 [-0.001, 0.002] | +0.001 [-0.000, 0.003] |
| T 표본 수 | 1688 | 1273 | 378 |
| 그중 A train에 형제가 있는 행 | 38 | 35 | 15 |
| A train의 T 템플릿 중복 행 수 | 44 | 32 | 15 |
| B train의 T 템플릿 중복 행 수 | 0 | 0 | 0 |
| 판정 — 주 지표 (A_sm−B) | `negligible` | `negligible` | `negligible` |
| 판정 — 보조 (A−B) | `negative` | `detected_minor` | `not_detected` |
| 판정 — 누출 대조 (A_sm−B) | `not_detected` | `not_detected` | `not_detected` |
| 판정 — 누출 대조 (A−B) | `not_detected` | `not_detected` | `not_detected` |
판정 정의 (사전 등록). SESOI = **0.02** AUROC — "실질적으로 무시 가능"이라고 부를 수 있는 최대 차이. 결론 수정 임계 **0.05**과는 다른 질문의 임계다.

- `negligible` — 무시 가능 — 쌍체 95% CI 상한 < SESOI(0.02). 누출 이득이 실질적으로 무시할 수 있는 크기임을 등가성 형태로 보였다.
- `not_detected` — 검출 실패 — CI가 0을 포함하지만 상한 ≥ SESOI(0.02). 누출 허용에 따른 AUROC 상승을 검출하지 못했으나, 실질적으로 의미 있는 상승을 배제하지도 못했다.
- `detected_minor` — 이득 검출, 크기 작음 — CI 하한 > 0이고 |Δ| < 0.05.
- `detected_major` — 결론 수정 필요 — CI 하한 > 0이고 |Δ| ≥ 0.05.
- `negative` — 역방향 — CI 상한 < 0. 누출을 허용한 모델이 오히려 더 나쁘다(검정력·교란 점검 필요).
- `undetermined` — 판정 불가 — CI가 유한하지 않다.
- 누출 대조 표본 수 (v2): 시드당 평균 38행
- 누출 대조 표본 수 (v3): 시드당 평균 35행
- 누출 대조 표본 수 (v4): 시드당 평균 15행
- 누출 대조 부집합은 표본이 수십 행뿐이라 CI가 넓다. 구조적으로 `negligible`이 나올 수 없으며, "phishing kit 암기 가능성을 배제했다"는 결론의 근거로 쓸 수 없다.


