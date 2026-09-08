결론부터 말하면, **사용자가 느낀 모호함(두 번째(합성곱 모델이 URL~) 주제는 증명은 한 것 같은데 첫 번째 확인하고 싶은 주제였던 '피싱 사이트의 QR코드에는 공통 시각 패턴이 있을까?'라는 질문의 답은 모호한 느낌)이 맞습니다.** 현재 실험은 상당히 잘 통제되어 있고 흥미로운 결과도 나왔지만, 두 연구 질문 가운데 증거의 강도가 다릅니다.

> **현재 가장 안전하게 말할 수 있는 결론**
>
> **① “피싱 QR에는 공통 시각 패턴이 있는가?” → 약한 의미에서는 Yes, 강한 의미에서는 아직 미검증.**
> 길이·버전 등을 통제한 뒤에도 피싱/정상을 구분하는 **공간적으로 유용한 신호(spatially useful signal)**가 QR 모듈 격자에 존재한다는 증거는 있습니다. 하지만 여러 피싱 QR에서 반복해서 등장하는 **구체적인 공통 시각 motif**가 발견됐다고 말하기에는 현재 실험이 부족합니다.
>
> **② “CNN이 URL 어휘 신호를 디코딩 없이 복원할 수 있는가?” → “복원”보다는 “활용할 수 있다”가 현재 증거에 정확합니다.**
> CNN이 명시적 QR decoding 없이 **URL에서 유래한 판별 신호를 상당히 이용할 수 있다**는 것은 꽤 강하게 입증됐습니다. 하지만 실제 문자·n-gram을 복원한 것은 아니므로 “lexical reconstruction”까지 증명됐다고 쓰면 리뷰어가 공격할 가능성이 큽니다.

## 교수/리뷰어 관점의 전체 평가

제가 이걸 논문 초안으로 받았다면 **“아이디어와 통제 실험은 좋지만, 핵심 claim을 재정의하고 추가 실험이 필요한 Major Revision”**으로 평가하겠습니다.

| 항목                  | 평가    | 코멘트                                                         |
| ------------------- | ----- | ----------------------------------------------------------- |
| 연구 질문               | A-    | 정보이론적으로 자명한 “QR로 피싱 분류 가능?”을 피하고 *무엇을 CNN이 배우는가*로 들어간 점이 좋음 |
| 실험 설계 / confound 통제 | A-    | 길이, 버전, 패딩, 마스크, eTLD+1 분할까지 고려한 것은 강점                      |
| “디코딩 없이 신호 활용” 증거   | A-    | 결과가 상당히 설득력 있음                                              |
| “공통 시각 패턴” 증거       | C+    | 현재 가장 약한 부분                                                 |
| 통계 처리               | B-    | 몇 가지 수정해야 할 부분이 있음                                          |
| 해석가능성 분석            | C+/B- | Grad-CAM만으로 claim하기 어려움                                     |
| 외적 타당성              | C     | WebPhish 단일 데이터셋이 결정적 한계                                    |
| 재현성/코드 구조           | A-    | config, P0 gate, seed, 결과 저장 구조는 좋음                         |
| 논문화 가능성             | 높음    | 다만 현재 claim 그대로보다는 연구 질문을 재구성하는 편이 훨씬 강함                    |

그리고 2026년 현재는 이미 **QR을 디코딩하지 않고 구조/픽셀을 분석해 quishing을 분류하는 연구**가 존재합니다. 예를 들어 Trad & Chehab은 QR 구조/픽셀 분석만으로 AUC 약 0.91을 보고했고, 2026년 systematic review도 pre-decode QR image analysis를 이미 하나의 연구 축으로 분류합니다. 따라서 논문의 novelty를 “QR 이미지로 피싱 탐지가 가능하다”에 두면 약합니다. 반대로 **“그 성능이 어디서 오는지 confound를 하나씩 제거하여 기계적으로 해부한다”**는 방향은 훨씬 차별화됩니다. ([Ifipaiai Repository][1])

---

# 현재 결과가 실제로 무엇을 보여줬는가

주 실험은 꽤 좋습니다. 엄격 길이 매칭 후 SmallCNN의 AUROC가 v2/v3/v4에서 각각 **0.872 / 0.814 / 0.805**이고, label shuffle은 **0.498 / 0.508 / 0.506**입니다. length LR과 version LR도 통제 조건에서 정확히 0.5입니다. 따라서 단순히 “URL이 길어서”, “QR 버전이 달라서” 분류한 결과라고 보기는 어렵습니다.

더 중요한 결과가 `shuffle-pos`입니다.

|              |        v2 |        v3 |        v4 |
| ------------ | --------: | --------: | --------: |
| CNN 정상 QR 배치 |      .872 |      .814 |      .805 |
| CNN 위치 셔플    |      .765 |      .671 |      .656 |
| 하락           | **-.107** | **-.143** | **-.149** |
| MLP 정상       |      .834 |      .732 |      .690 |
| MLP 셔플       |      .834 |      .733 |      .706 |

즉 비트 집합 자체는 그대로인데 원래 QR의 공간 배치를 깨면 CNN만 크게 나빠집니다. **“QR의 원래 공간 배치가 CNN에게 유용하다”**는 주장은 상당히 설득력이 있습니다.

그런데 여기서 바로 **“피싱 QR에 공통 시각 패턴이 있다”**로 점프하면 안 됩니다.

특히 byte-hist LR이 이미 **.861 / .766 / .771**입니다. v2에서는 CNN .872와 불과 0.011 차이입니다. 즉 상당한 판별 정보가 애초에 **어떤 바이트/문자가 얼마나 자주 등장하는가**라는 비공간적 분포로 설명됩니다. CNN이 그보다 추가로 얻는 정보가 있기는 하지만, 현재 결과만으로 그것을 “공통 시각 motif”라고 규정할 수는 없습니다.

그래서 현재 결과를 세 단계로 나누면 훨씬 정확합니다.

| 주장                                  | 현재 증거             |
| ----------------------------------- | ----------------- |
| QR 모듈 격자에 클래스와 연관된 정보가 남는다          | **강하게 지지**        |
| 원래 QR 공간 배치가 CNN의 분류에 도움이 된다        | **강하게 지지**        |
| 여러 피싱 QR에 반복되는 특정 국소 시각 motif가 존재한다 | **아직 미검증**        |
| CNN 내부에서 URL 문자/n-gram 자체가 복원된다     | **부분적 간접 증거만 있음** |
| 이 패턴이 다른 피싱 데이터에서도 일반화된다            | **미검증**           |

이게 사용자가 RESULTS를 읽으며 느낀 찜찜함의 정체라고 봅니다.

---

## 리뷰어가 지적할 가능성이 높은 핵심 문제

1. **char n-gram LR을 “이론적 상한선”이라고 부르면 안 됩니다.** 현재 문서에는 “CNN이 이를 넘으면 누출”이라고까지 되어 있고, 코드도 `below_text_upper_bound` sanity check를 수행합니다. 하지만 char 1–5 gram TF-IDF + Logistic Regression은 **하나의 텍스트 분류기일 뿐** Bayes-optimal classifier가 아닙니다. QR이 URL의 결정적 변환이라는 점에서 *정보론적으로* decoded URL에는 QR보다 적지 않은 정보가 있다는 것은 맞지만, 특정 LR이 그 상한을 구현하는 것은 아닙니다. CNN이 LR 0.97을 넘어 0.98이 나왔다고 해서 데이터 누출이라고 할 수 없습니다. 이름을 **decoded-text reference / strong text baseline**으로 바꾸는 것을 권합니다.

2. **“공통 시각 패턴”이라는 변수가 operationalize되어 있지 않습니다.** 현재 H4는 “원래 공간 배열이 유용한가?”를 검사합니다. 하지만 “여러 phishing QR에 동일하거나 유사한 2×2, 3×3, 5×5 패턴이 반복되는가?”나 “그 패턴이 특정 위치에서 반복되는가?”는 직접 측정하지 않습니다. 따라서 H4는 *spatial structure*의 증거이지 *shared visual motif*의 증거는 아닙니다.

3. **현재 Grad-CAM은 첫 번째 질문에 대한 결정적 증거가 되기 어렵습니다.** 코드상 첫 번째 seed의 test set에서 **앞쪽 128개 샘플만** 가져옵니다. 클래스별 층화 sampling도 아니고 5 seed를 모두 쓰지도 않습니다. 또한 모든 샘플에서 positive/phishing logit을 target으로 하고 ReLU를 사용하므로 benign CAM은 “benign의 근거”가 아니라 “그 benign 샘플에서 phishing logit을 증가시키는 부분”입니다. 각 CAM을 max=1로 정규화하기 때문에 클래스 간 CAM 절대 크기 비교도 해석이 까다롭습니다.   Saliency map의 시각적 설득력과 실제 모델 설명력을 동일시하면 안 된다는 것은 오래전부터 지적된 문제입니다. ([NeurIPS Papers][2])

4. **CAM 질량은 영역의 크기로 정규화할 필요가 있습니다.** 예를 들어 char 영역에 CAM의 60%가 몰렸다고 해도, 전체 데이터 모듈의 70%가 char 영역이라면 오히려 char를 덜 본 것입니다. 현재는 `CAM mass`만 계산합니다. 필요한 것은 `CAM mass fraction / module area fraction` 형태의 **enrichment**와 random attribution baseline입니다. 특히 function 영역에 16~34% CAM이 나온다는 사실 자체가 Grad-CAM의 낮은 공간해상도와 interpolation 번짐을 보여줍니다. 결과 문서에서도 이 문제를 어느 정도 인식하고 있습니다.

5. **통계 CI 집계 방식은 수정하는 편이 좋습니다.** 코드에서 `auroc_ci_pooled`는 5 seed의 bootstrap CI 하한들을 평균하고 상한들을 평균해서 만듭니다. 이것은 aggregate estimator에 대한 정식 confidence interval이 아닙니다. 또 cluster bootstrap percentile CI가 point estimate를 포함하지 않으면 강제로 CI를 넓혀 point estimate를 포함시키는데, 이것도 일반적인 통계 절차는 아닙니다.   설계 문서에서는 H1~H4에 Holm 보정을 하기로 했지만 RESULTS에서도 아직 다음 단계로 남아 있습니다.

6. **H3의 L-none − L-exact가 순수한 “길이 artifact 효과”는 아닙니다.** 예를 들어 v2는 exact N=12,118인데 none N=16,627입니다. 모델이 다른 표본 수·다른 클래스 구성·다른 URL 집합에서 훈련됩니다. 따라서 +0.005라는 차이는 길이 신호만 추가한 intervention이 아니라 **training set composition도 동시에 변경한 비교**입니다. PAD-rand 쪽이 오히려 인과적 ablation에 더 가깝습니다.

7. **가장 큰 외적 타당성 문제는 데이터 출처입니다.** 원 WebPhish 데이터의 정상 샘플은 Alexa 상위 사이트에서 왔고, phishing은 전혀 다른 수집 과정에서 만들어졌습니다. 즉 label과 source가 사실상 결합되어 있습니다. 원 논문의 데이터 구성 자체가 그렇습니다. ([Northumbria University Research Portal][3]) 지금의 `norm`, `path+`, 길이 매칭이 이를 상당히 완화하지만 제거하지는 못합니다. 실제로 `raw → norm`만으로 CNN 성능이 **5~9%p 감소**한다는 결과는 source-specific surface cue가 상당했음을 보여줍니다.  2026년 quishing systematic review도 이 분야에서 데이터셋 일반화 가능성이 주요 문제라고 지적합니다. ([Taylor & Francis Online][4])

추가로 H4 문서의 “BitMLP는 순열 불변”이라는 표현도 기술적으로는 바꾸는 것이 좋습니다. 일반 MLP 자체는 permutation invariant가 아닙니다. 같은 가중치에서 \(f(Px)=f(x)\)가 성립하지 않습니다. 여기서는 **재학습하는 MLP의 hypothesis class가 입력 좌표의 고정 permutation에 대해 동등한 표현력을 갖는다**는 것이 정확한 설명입니다. H4 결과 자체가 틀렸다는 뜻은 아닙니다.

---

# 가장 중요한 개선: Q1을 직접 측정하는 실험을 추가하세요

여기서 CNN을 하나 더 돌리는 것보다 **Bag-of-QR-patches** 실험을 추가하는 것이 훨씬 가치가 큽니다.

QR data module 위를 sliding window로 훑어서 2×2와 3×3 binary patch의 출현 횟수를 세면 됩니다.

3×3 패턴은 가능한 종류가 \(2^9=512\)개뿐이므로 각 QR을 다음처럼 표현할 수 있습니다.

**QR → 512차원 “3×3 motif histogram” → Logistic Regression**

그러면 질문을 직접 물을 수 있습니다.

| 결과                                     | 해석                                     |
| -------------------------------------- | -------------------------------------- |
| patch histogram ≈ 0.5                  | 단순한 “공통 국소 시각 패턴” 증거 없음                |
| patch histogram > 0.5                  | 피싱 QR에 반복적으로 더 자주 등장하는 local motif가 존재 |
| spatial-pyramid patch histogram이 더 좋아짐 | motif의 **위치**도 중요                      |
| CNN만 높고 patch histogram은 낮음            | 단순 motif가 아니라 긴 범위의 순서/구조가 중요          |
| 특정 motif 제거 시 CNN AUROC 감소             | 그 motif가 모델 예측에 **인과적으로 기여**           |

그리고 각 3×3 motif에 대해 phishing/benign 출현율, odds ratio, 그룹 bootstrap CI를 계산할 수 있습니다.

예를 들어 논문에 다음과 같은 결과를 실제로 쓸 수 있게 됩니다.

> “Pattern #173은 benign보다 phishing QR에서 1.84배 자주 나타났으며, 5개 seed와 held-out eTLD+1에서 방향이 일관되었다.”

이것이 **“공통 시각 패턴이 있다”**라는 질문에 훨씬 직접적인 답입니다. Grad-CAM의 예쁜 heatmap보다 학술적으로 훨씬 강합니다.

---

## 제가 권하는 2차 실험 설계

| 단계                           | 해야 할 것                                                                                                | 해결되는 질문                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------- |
| **A. Claim/통계 정리**           | `text upper bound` → `decoded-text reference`; paired bootstrap; Holm H1~H4                           | 기존 결과의 엄밀성                  |
| **B. Visual motif 실험**       | 2×2/3×3 bag-of-patches LR + spatial pyramid + motif enrichment                                        | **Q1 직접 검증**                |
| **C. Causal motif ablation** | 상위 phishing motif가 포함된 patch를 제거/flip하고 같은 크기의 random patch 제거와 비교                                    | motif가 실제 CNN 판단 원인인가       |
| **D. Lexical probe**         | CNN embedding을 freeze하고 URL char n-gram presence, digit ratio, 특수문자, path feature 등을 linear probe로 예측 | **Q2를 “복원” 수준으로 강화**        |
| **E. Counterfactual URL**    | 길이를 유지하며 suspicious token만 삽입/제거한 paired URL 생성 후 CNN score Δ 측정                                      | 어휘 변화→QR→예측의 인과 연결          |
| **F. 외부 검증**                 | WebPhish 학습 → 별도 PhishTank/OpenPhish phishing + Tranco/Common Crawl benign 평가                         | source shortcut 제거          |
| **G. Campaign split**        | eTLD+1뿐 아니라 URL token/path template의 MinHash/클러스터 단위 holdout                                          | phishing kit/template 누출 제거 |
| **H. 실제 QR 조건**              | auto-mask, 여러 EC, rotation/blur/perspective/JPEG/printing simulation                                  | 실제 “시각 QR”로 claim 확장        |

이 중 **B가 지금 연구에서 가장 중요한 추가 실험**입니다. 계산량도 크지 않습니다. CNN을 새로 대량 학습할 필요 없이 현재 생성한 module grid로 대부분 CPU에서 돌릴 수 있습니다.

Colab T4 예산을 고려한다면 먼저 **v3 하나에서 B+C+D를 완성**하고, 효과가 확인된 것만 v2/v4로 replication하는 구조가 좋습니다.

---

# 두 번째 질문도 표현을 약간 낮추는 것이 좋습니다

현재 RESULTS는 사실상 다음을 보여줍니다.

**QR module grid → CNN → phishing classification**

그리고 원래 배치를 깨면 성능이 크게 떨어집니다. 이것은 CNN이 QR 인코딩이 만들어낸 **URL-derived local/order structure**에 접근한다는 좋은 증거입니다.

하지만 “URL 어휘 신호를 복원했다”라고 하려면 저는 리뷰어로서 이런 질문을 하겠습니다.

> “복원된 어휘 신호를 보여주세요. CNN representation으로 `login`, `verify`, `.com`, 숫자 비율, path token 같은 것을 실제로 예측할 수 있습니까?”

따라서 SmallCNN의 GAP 직전 128차원 embedding을 freeze하고 별도의 probe를 붙이면 됩니다.

`QR → CNN embedding → linear probe → "login" n-gram 존재 여부`

이런 probe를 수십~수백 개 URL n-gram에 대해 수행하고 label-shuffle baseline과 비교하면 됩니다.

더 강하게 가고 싶다면:

`QR → CNN encoder → byte/character sequence prediction`

이라는 작은 decoder까지 붙일 수 있습니다. 여기서 실제 문자 복원율이 나온다면 비로소 **“lexical reconstruction without QR decoding”**이라는 강한 표현을 사용할 근거가 생깁니다.

그렇지 않다면 논문에서는 **recover**보다 **exploit / access / utilize URL-derived lexical signals**를 쓰는 것이 안전합니다.

---

# Grad-CAM은 보조 그림으로 내리는 것을 권합니다

Grad-CAM을 버릴 필요는 없습니다. 다만 **Q1을 증명하는 주 분석**에서 **보조 시각화**로 역할을 낮추는 게 좋습니다.

현재처럼 128개 첫 샘플 대신 5 seed × 클래스 balanced random sampling으로 하고, phishing에는 `logit`, benign에는 `-logit`을 target으로 삼거나 signed Integrated Gradients를 사용하는 게 낫습니다. 동시에 attribution randomization sanity check와 patch occlusion을 붙여야 합니다. Geirhos 등이 강조한 shortcut learning 관점에서도, i.i.d. 성능과 보기 좋은 attribution보다 intervention과 OOD 검증이 훨씬 중요합니다. ([DOI][5])

특히 이 연구는 일반 사진을 CNN에 넣는 게 아니라 **완벽하게 정렬된 QR 모듈 격자**를 넣습니다. 그것도 두 번째 채널로 data-module mask까지 제공합니다.  따라서 현재 연구에서 “visual”이라는 단어는 엄밀히 말해 **raw-image visual pattern이 아니라 canonical QR module-grid spatial pattern**입니다. 이 구분을 논문에서 명확히 하는 게 좋습니다.

---

# 연구 질문 자체를 이렇게 고치면 훨씬 강해집니다

현재 두 질문을 억지로 분리하기보다 저는 다음 3개 RQ로 재구성하겠습니다.

| 새 연구 질문                                                                                   | 현재 상태         |
| ----------------------------------------------------------------------------------------- | ------------- |
| **RQ1. 길이·버전·패딩 등의 shortcut을 통제한 후에도 QR 모듈 격자에 phishing-discriminative signal이 존재하는가?**   | 거의 답함         |
| **RQ2. CNN은 명시적 QR decoding 없이 QR의 공간 배치를 이용해 URL-derived lexical/order signal에 접근하는가?**  | 상당 부분 답함      |
| **RQ3. 그 signal은 여러 phishing QR에서 반복되는 local visual motif로 나타나며, 새로운 도메인·데이터셋에서도 재현되는가?** | **다음 실험의 핵심** |

이렇게 바꾸면 현재 결과가 버려지는 게 아니라 오히려 정확한 위치를 갖습니다.

논문 제목도 예를 들어:

**Do Phishing URLs Leave Learnable Spatial Signatures in QR Codes?
A Controlled Study of URL-Derived Signal Learning without Explicit Decoding**

혹은 한국어로는:

**「피싱 URL은 QR 모듈 공간에 반복 가능한 공간 신호를 남기는가? — 명시적 디코딩 없는 CNN의 URL 신호 학습에 대한 통제 실험」**

정도가 연구 내용과 잘 맞습니다.

---

## 최종적으로 첫 번째 질문에 지금 답한다면

제가 논문 Discussion을 쓴다면 현재 단계에서는 이렇게 결론내리겠습니다.

**“길이·QR 버전·패딩 경계와 같은 주요 confound를 통제한 뒤에도 phishing과 benign QR module grids 사이에는 통계적으로 유의한 판별 가능 신호가 남으며, CNN은 QR의 원래 공간 배치를 이용해 이 신호를 활용한다. 그러나 본 실험은 아직 여러 phishing QR에 반복적으로 나타나는 특정 local visual motif의 존재를 직접 입증하지는 않는다.”**

따라서 사용자의 감각대로 **첫 번째 질문은 아직 절반 정도밖에 답하지 못했습니다.** 반대로 이게 연구의 약점이면서 동시에 **2차 실험에서 가장 재미있는 부분**입니다. 지금 결과에 `3×3 motif histogram → motif enrichment → causal occlusion → external dataset replication`만 제대로 붙으면, 단순한 “CNN으로 QR 피싱 분류해봤다”에서 **“CNN이 QR에서 도대체 무엇을 보고 있는지를 해부한 연구”**로 성격이 크게 달라질 수 있습니다.

[1]: https://repository.ifipaiai.org/2026/abstr/261TA331Hk.html "https://repository.ifipaiai.org/2026/abstr/261TA331Hk.html"
[2]: https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html "https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html"
[3]: https://researchportal.northumbria.ac.uk/ws/portalfiles/portal/116653298/1_s2.0_S0957417423016858_main.pdf "https://researchportal.northumbria.ac.uk/ws/portalfiles/portal/116653298/1_s2.0_S0957417423016858_main.pdf"
[4]: https://www.tandfonline.com/doi/full/10.1080/23742917.2026.2696261 "https://www.tandfonline.com/doi/full/10.1080/23742917.2026.2696261"
[5]: https://doi.org/10.1038/s42256-020-00257-z "https://doi.org/10.1038/s42256-020-00257-z"
