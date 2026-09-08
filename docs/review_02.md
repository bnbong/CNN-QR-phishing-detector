**이전 점검 때 지적했던 핵심 문제들은 상당 부분 제대로 개선됐습니다.** 특히 이번 수정은 단순히 실험을 더 붙인 게 아니라, 이전에 과했던 주장 중 일부를 실제 실험으로 **기각하거나 낮춰 쓴 것**이 좋습니다. 현재 상태라면 제가 지난번에 내렸던 “흥미롭지만 핵심 claim이 아직 약한 Major Revision” 평가는 꽤 올라갑니다.

다만 지금 새로 드러난 문제도 있습니다. 특히 **motif 실험의 통계 처리 하나**, **G(template split)의 비교 설계 하나**는 외부 검증이 끝나기 전에 고치는 것을 권합니다.

## 전체 평가

현재 연구의 논리 구조는 이렇게 보는 게 가장 정확합니다.

| 연구 질문                               | 이전             | 현재 평가                                               |
| ----------------------------------- | -------------- | --------------------------------------------------- |
| RQ1. shortcut 통제 후에도 판별 신호가 있는가     | 강하게 지지         | **강하게 지지, 상당히 안정적**                                 |
| RQ2. CNN이 decoding 없이 URL 신호에 접근하는가 | “어휘 복원” 표현이 과함 | **URL-derived signal 활용은 지지, lexical recovery는 기각** |
| RQ3. 공통 local motif가 존재하는가          | 사실상 미검증        | **class-associated 3×3 motif 존재는 지지**               |
| motif가 CNN 판단 원인인가                  | 미검증            | **부분적 인과 증거 확보**                                    |
| 다른 데이터셋에도 일반화되는가                    | 미검증            | **현재 F 실험 진행 중**                                    |
| phishing-kit/template 누출은 없는가       | eTLD+1만 통제     | **G 구현은 됐지만 비교 설계 보완 권장**                           |

특히 RESULTS에서 처음부터 RQ1~RQ3를 구분하고, “motif는 존재하지만 전체 AUROC를 설명하는 몫은 작으며, n-gram/keyword의 선형 복원은 실패했다”고 명시한 것은 이전보다 훨씬 학술적으로 건강한 결론입니다.

---

# 잘 개선된 부분

### 1. `text upper bound` 문제는 제대로 고쳤습니다

이전의 가장 명확한 개념적 오류 중 하나였던

> char n-gram LR = 이론적 상한선
> CNN이 이것을 넘으면 leakage

논리를 완전히 버렸습니다.

현재는 이를 **decoded-text reference / 강한 텍스트 baseline**으로 바꾸고, Bayes-optimal upper bound가 아니라는 점과 CNN이 이를 넘어도 leakage가 아니라는 점까지 명시했습니다. 누출 검증도 label shuffle과 split diagnostic으로 분리했습니다.

이건 거의 그대로 유지하면 됩니다.

---

### 2. Grad-CAM도 지난번 문제를 제대로 수정했습니다

이전에는 제가 “CAM mass만 보면 영역 크기 때문에 의미가 없다”, “benign에도 phishing logit을 target으로 건 문제가 있다”, “128개 한 seed로는 약하다”고 지적했는데, 지금은 상당히 좋아졌습니다.

현재 결과에는 640개 표본을 이용하고, **영역 면적 대비 CAM enrichment**를 계산합니다. char 영역은 약 **1.43~1.57배 enrichment**, function은 0.49~0.66, EC는 0.61~0.68로 나왔고 random-attribution baseline은 약 1입니다. 모델 가중치를 무작위화했을 때 원 CAM과의 상관도 절댓값 0.08 이하였습니다. 이전의 잘못된 `char_position_curve` 해석도 스스로 폐기했습니다.

이 정도면 Grad-CAM은 이제 **“인과 증거가 아니라 모델이 어떤 종류의 영역을 상대적으로 더 보는지 보여주는 보조 분석”**으로 충분히 쓸 만합니다.

---

### 3. RQ3에 Bag-of-QR-patches를 넣은 방향이 정확합니다

이전 실험에서 가장 부족했던 것이 바로 이것이었습니다.

현재는 3×3 local patch histogram만으로

* v2: **0.740**
* v3: **0.706**
* v4: **0.717**

AUROC가 나왔고 label-shuffle은 약 0.49~0.53입니다. 3×3 spatial pyramid도 실험했고 개별 motif의 log odds와 held-out 방향까지 조사했습니다.

그래서 이제는 지난번처럼

> “공통 패턴이라고 했지만 spatial structure가 중요하다는 것만 증명했다”

라고 평가할 단계는 아닙니다.

**“특정 3×3 local binary pattern들의 출현 빈도 분포가 phishing과 benign 사이에서 체계적으로 다르다”**까지는 보여줬습니다.

이건 큰 진전입니다.

---

### 4. lexical recovery 주장을 결과에 맞게 스스로 기각한 것도 좋습니다

이 부분은 연구 신뢰도를 오히려 높입니다.

CNN representation을 freeze한 뒤 n-gram, keyword, digit/path feature 등을 linear probe한 결과 91~114개 target 중 유의한 것이

* v2: 2개
* v3: 0개
* v4: 0개

이고 v2 두 개조차 `has_path`, `path_depth`입니다. 게다가 random-init CNN representation과 비교까지 했습니다.

따라서

> “CNN이 URL 문자열을 사실상 복호화한다”

가 아니라

> “CNN이 URL에서 유래한 판별 신호를 decoding 없이 exploit하지만 개별 lexical feature가 representation에서 선형 복원된다는 증거는 없다”

라는 현재 방향이 맞습니다.

---

### 5. 통계 집계도 이전보다 좋아졌습니다

이전에는 seed별 bootstrap CI의 lower/upper를 각각 평균하는 이상한 `pooled CI`가 있었는데, 지금은 **seed-stratified cluster bootstrap**을 구현해서 bootstrap마다 eTLD+1 그룹을 resample한 뒤 seed별 AUROC를 계산하고 평균합니다.

또 `L-none`과 `L-exact`의 표본 집합이 다르기 때문에 H3을 엄밀한 paired intervention으로 해석할 수 없다는 점도 RESULTS에 명시했습니다.

지난번 지적을 제대로 반영한 부분입니다.

---

# 그런데 지금 반드시 고쳤으면 하는 부분

## 1. 가장 먼저: `motifs.py`에 예전 CI 문제가 하나 남아 있습니다

이건 코드 레벨에서 바로 수정하는 것을 권합니다.

`evaluate.cluster_bootstrap()`에서는 제가 이전에 지적했던

> point estimate가 percentile CI 밖에 있으면 CI를 강제로 넓혀 point를 포함시킨다

는 동작을 제거했습니다. 현재는 순수 percentile CI입니다.

그런데 `motifs.py::motif_enrichment()`에는 아직 이 코드가 남아 있습니다.

> `lo = np.minimum(lo, point)`
> `hi = np.maximum(hi, point)`

심지어 주석도 “evaluate.cluster_bootstrap과 같은 관례”라고 되어 있는데, **현재 evaluate는 더 이상 그렇게 하지 않습니다.**

이건 삭제하세요.

그리고 motif LR의 5-seed 결과도 현재 단순 pooling 방식을 사용하고 있고 RESULTS에서도 CNN과 집계 정의가 다르다고 따로 설명하고 있습니다.

가능하면 motif 결과 역시 `cluster_bootstrap_by_seed()` 방식으로 통일하는 게 좋습니다. “차이가 0.005 이하니까 괜찮다”보다는 **전체 논문이 동일한 estimator를 사용한다**가 훨씬 깔끔합니다.

---

## 2. 현재 “motif가 존재한다”를 한 단계만 더 엄밀하게 만들 필요가 있습니다

현재 3×3 histogram LR이 0.70 이상인 것은 좋은 결과입니다.

그런데 리뷰어는 이렇게 물을 수 있습니다.

> “그 3×3 motif가 정말 공간적 co-occurrence인가요?
> 아니면 phishing URL과 benign URL의 단순 비트/바이트 조성이 달라서 3×3 histogram까지 같이 달라진 것 아닌가요?”

실제로 현재 **byte-hist LR가 motif LR보다 높습니다.**

* byte histogram: 0.861 / 0.766 / 0.771
* 3×3 motif: 0.740 / 0.706 / 0.717

즉 현재 증거는 엄밀히는

> **class-associated local patch-frequency distribution이 존재한다**

입니다.

아직

> **marginal bit/byte composition으로 설명되지 않는 genuine local spatial dependency가 존재한다**

까지는 아닙니다.

여기에는 CNN 재학습도 필요 없습니다. 다음 control 하나를 추가하는 게 좋습니다.

**각 QR마다 data-module의 0/1 개수는 그대로 유지한 채 위치만 sample-wise random shuffle → 다시 3×3 histogram LR.**

그러면:

`실제 3×3 motif AUROC ≫ within-QR shuffled 3×3 AUROC`

가 나오면 “단순 black-module 비율이 아니라 local adjacency가 중요하다”는 훨씬 강한 근거가 됩니다.

가능하면 한 단계 더 나아가 **codeword/byte composition을 최대한 보존하면서 공간 배치만 깨는 null**도 넣으면 좋습니다.

이것이 현재 RQ3을 가장 강하게 만들어줄 작은 추가 실험입니다.

---

## 3. 현재 G(template split)는 그대로 성능 비교하면 해석이 상당히 어렵습니다

이 부분은 이번 점검에서 새로 발견한 가장 중요한 문제입니다.

현재 diagnostics를 보면 v3에서 eTLD+1 group 6,198개가 template union 후 5,733개로 줄고, template-aware split을 하자 **기존 stage-1 test set과 새로운 test set의 Jaccard가 평균 0.062밖에 되지 않습니다.**

더 직접적으로는 기존 test row의 **약 88.4%가 template split에서는 test set을 떠납니다.**  코드에서도 이 값은 실제로 `len(te - tt) / len(te)`, 즉 기존 test 중 새로운 test에 남지 않는 비율로 계산합니다.

따라서 앞으로

> eTLD+1 split AUROC = 0.81
> template split AUROC = 0.73
> → template leakage가 0.08이었다

라고 비교하면 곤란합니다.

왜냐하면 **학습 조건뿐 아니라 평가 대상의 88%가 바뀌었기 때문**입니다.

현재 설계가 이를 unpaired comparison으로 처리하는 것은 정직하지만, causal interpretation은 여전히 약합니다.

제가 권하는 구조는 **campaign/template-disjoint test set을 먼저 하나 고정**하는 것입니다.

그 동일한 test set에 대해:

`Model A`: eTLD+1만 고려해 학습한 모델 — test와 동일 eTLD+1은 없지만 유사 template은 train에 허용
`Model B`: eTLD+1 + template까지 격리한 모델

을 평가하세요.

그러면 같은 test row에서

**A − B**

를 paired comparison할 수 있고, 그 차이가 바로 “template shortcut을 학습에서 허용했을 때 얻는 이득”에 훨씬 가깝습니다.

현재 G 방식보다 훨씬 강합니다.

---

## 4. occlusion 실험도 결론은 괜찮지만 표현을 약간 정확하게 바꿔야 합니다

현재 RESULTS를 읽으면 “상위 3×3 motif를 뒤집었다”는 인상을 줍니다.

실제 코드는 해당 motif가 발견된 **3×3 window의 중심 모듈 하나만 flip**합니다.

이 방식 자체는 괜찮습니다. 중심 비트를 바꾸면 해당 motif는 파괴되니까요.

다만 논문에는

> motif flipping

보다는

> **motif-targeted center-bit intervention**

또는

> **motif-matched local perturbation**

정도로 쓰는 게 정확합니다.

그리고 현재 random control은 flip 개수는 동일하게 맞추지만 **공간적 topology는 맞추지 않습니다.** motif가 나타나는 center들이 특정 영역에 몰려 있고 random center들은 격자 전체에 흩어진다면 차이 일부가 motif identity가 아니라 위치에서 올 수 있습니다.

따라서 random control을 하나 더 넣으면 좋습니다.

**같은 quadrant / 비슷한 local black density / 같은 data-module 후보군에서 matched random center를 선택**하는 식입니다.

현재 결과가 이미 motif intervention이 random의 5~8배 정도 효과가 있어서 결론이 완전히 뒤집힐 가능성은 낮아 보이지만, 이 control이 있으면 “인과적 기여” 표현이 훨씬 안전해집니다.

---

## 5. RQ2의 문구에는 아직 “lexical”이 조금 남아 있습니다

이건 논문 문장 정리 문제입니다.

현재 probe 결과는 개별 n-gram/keyword의 선형 복원 증거가 사실상 없다고 말합니다. 그런데 RESULTS 6-1 제목은 아직

> **“디코딩 없이 어휘 신호의 상당 부분에 접근한다”**

입니다.

이건 probe 결과보다 강합니다.

저라면 전부 다음처럼 통일하겠습니다.

> **URL-derived discriminative signal**

또는 조금 구체적으로

> **URL-derived bit/byte composition and spatial-order signal**

즉 RQ2도

> “URL-derived lexical/order signal”

보다

> **“URL-derived content/order signal”**

이 안전합니다.

“lexical”은 nonlinear probe나 실제 sequence decoder에서 의미 있는 문자 정보가 나오면 그때 다시 올리면 됩니다.

---

## 6. H1~H4의 `p (Holm)`은 논문에서는 조심해서 써야 합니다

코드에서 이 부분을 꽤 솔직하게 작성해 두었습니다.

현재 `bootstrap_p_value()`는 스스로

> “영가설 아래에서 생성한 정식 검정통계량의 p-value가 아니며 명목 수준을 정확히 지키지 않을 수 있다”

고 경고합니다.

그런데 RESULTS 표에는 그냥

> `p (Holm)`

이라고 표시되어 있어서 일반 독자는 정식 hypothesis-test p-value로 읽을 가능성이 있습니다.

선택지는 두 가지입니다.

가장 간단한 것은 **effect size + cluster-bootstrap CI를 본문 주 결과로 두고 p-value를 부록으로 내리면서 `bootstrap-tail pseudo-p`라고 명확하게 명명**하는 것입니다.

정식 p-value가 꼭 필요하다면 H1/H4처럼 paired prediction이 있는 경우에는 **cluster-aware permutation/randomization test**를 쓰는 편이 더 낫습니다.

개인적으로 이 논문에서는 p-value를 전면에 세울 이유가 크지 않습니다. ΔAUROC와 CI가 이미 충분히 강합니다.

---

# 외부 검증 설계는 전반적으로 좋은데, 두 가지는 바꾸는 편이 좋습니다

외부 데이터 구성은 꽤 신경 썼습니다.

특히 외부 데이터에서

* benign `path_depth≥1`: **0.940**
* phishing: **0.761**
* benign median length: 51
* phishing: 46

으로 만들어 **WebPhish의 “benign 짧고 root / phishing 길고 path 있음” 편향을 오히려 뒤집은 것**은 훌륭한 stress test입니다. 스킴도 양쪽 동일하게 제거했습니다.

F-a zero-shot, F-b external→external, F-c reverse transfer, F-d mixed-domain으로 질문을 분리한 구조도 좋습니다. 특히 F-d에서 `origin_lr`까지 두려는 아이디어도 적절합니다.

다만 두 가지는 바꾸는 것을 권합니다.

### 첫째, 저는 OpenPhish를 primary external phishing으로 올리겠습니다

현재 실제 수집 결과는:

* Phishing.Database: 55만+ 사용 가능하지만 해당 ACTIVE 파일 자체의 최종 갱신이 **2025-12-22**
* OpenPhish 최근 90일: **고유 URL 약 43,990개**

입니다.

43,990개면 외부 검증에는 충분히 큽니다.

오히려 연구적으로는

**Primary:** OpenPhish 2026 최근 90일 + CC-MAIN-2026-34 benign
**Secondary robustness:** Phishing.Database + 동일 benign

이 더 깨끗합니다.

이렇게 하면 **수집 시점까지 대략 맞는 진짜 temporal/domain-shift test**가 됩니다.

“789k라서 PhishDB가 주”는 ML 연구에서는 별로 강한 이유가 아닙니다. 외부 test는 크기보다 **독립성·신선도·출처 명확성**이 중요합니다.

### 둘째, Common Crawl × Tranco는 아직 popularity confound를 완전히 없애지 못했습니다

현재 benign은 Common Crawl 페이지이긴 하지만 **Tranco와 조인한 인기 도메인 페이지**입니다.

WebPhish benign도 Alexa 인기 사이트 계열이었습니다.

즉 “root URL vs path URL” bias는 잘 깨졌지만

> **benign = 인기 웹사이트**

라는 축은 일부 유지됩니다.

그래서 외부 검증 결과가 매우 잘 나온다면 리뷰어는 여전히

> “모델이 phishing을 찾는 게 아니라 popular-domain-derived character statistics를 찾는 것 아닌가?”

라고 물을 수 있습니다.

주 결과는 지금처럼 Tranco-joined CC를 사용해도 괜찮지만, 반드시 **Tranco join을 끈 unranked/general Common Crawl benign sensitivity set**을 하나 추가해 보세요.

오염 가능성이 늘어나는 것은 한계로 명시하면 됩니다.

오히려 결과가

`CC×Tranco benign에서는 높음`
`unranked CC benign에서는 급락`

하면 그것 자체가 중요한 연구 결과입니다.

---

# benign cleaning sensitivity는 “권장”이 아니라 필수로 두세요

현재 외부 loader는 benign에서

1. phishing feed에 등장한 eTLD+1 제거
2. 공유 호스팅/shortener blocklist 제거

를 합니다. 코드에서도 이것이 **평가를 낙관적으로 만들 수 있다는 사실을 정확히 알고 있고**, 옵션으로 끌 수 있게 해 두었습니다.

좋은 설계입니다.

그런데 저는 이 sensitivity를 **반드시 실행해야 하는 실험**으로 승격시키겠습니다.

최소한

`clean benign`
`keep benign on phishing domains`
`hosting blocklist off`

세 조건을 비교하세요.

외부 AUROC가 거의 같다면 결과가 매우 강해집니다.

반대로 크게 떨어지면 그 preprocessing 자체가 외부 generalization을 만들어냈다는 뜻입니다.

그리고 문서의

> “benign contamination은 전이 성능을 과소평가하는 방향이다”

는 표현은

> **“일반적으로 성능을 감쇠시키는 방향으로 예상된다”**

정도로 낮추는 게 더 엄밀합니다. label noise가 systematic할 때 항상 단조롭게 AUROC를 낮춘다고 보장할 수는 없습니다.

---

# F-a의 평가 cohort도 가능하면 고정하세요

현재 F-a는 외부에서도 seed마다 group split → split별 exact matching을 다시 하도록 되어 있습니다. F-b와 같은 행 집합으로 비교하려는 의도는 이해됩니다.

하지만 zero-shot transfer의 주 결과로는 저는 다음이 더 깔끔하다고 봅니다.

**외부 평가 cohort를 한 번 고정**
→ 동일한 외부 URL들에 WebPhish seed 0~4의 다섯 모델을 전부 평가.

그러면 모델 seed만 달라지고 test distribution은 완전히 동일합니다.

현재 방식은 seed마다

* WebPhish 모델도 바뀌고
* external matching sample도 일부 바뀌므로

두 종류의 변동이 섞입니다.

`F-a primary = fixed external cohort`

`F-a vs F-b paired secondary = 각 F-b seed test cohort`

로 분리하면 가장 깔끔합니다.

---

# 그리고 G는 외부 검증 못지않게 중요합니다

지금 template diagnostics만 봐도 G를 실제 학습까지 돌릴 가치가 충분합니다.

v3에서 template clustering 자체는 대부분 singleton이지만, eTLD+1과 union하면 group 수가 **6,198 → 5,733**으로 줄고 최대 그룹이 약 **5.14%**까지 커집니다.

즉 “다른 도메인이지만 URL path/template이 비슷한 phishing campaign”이 실제로 어느 정도 존재합니다.

그리고 앞에서 이야기했듯 test cohort가 거의 통째로 재편됩니다.

그래서 외부 F 결과가 좋게 나와도 **G를 안 하고 최종 논문을 닫지는 않는 것**을 권합니다.

---

# 지금 시점에서 연구 결론을 어떻게 쓰면 좋은가

저라면 현재 WebPhish 내부 결과에 대해서는 이렇게 씁니다.

> 길이, QR 버전, padding boundary 및 일부 표면적 URL shortcut을 통제한 뒤에도 QR module grid에는 phishing-discriminative signal이 남는다. CNN은 QR의 원래 공간 배치를 활용하며, 3×3 local patch-frequency distributions 역시 class-associated structure를 포함한다. 특정 motif 위치에 대한 targeted perturbation은 matched random perturbation보다 더 큰 성능 저하를 유발하지만, 그 효과는 전체 CNN 성능의 일부만 설명한다. 반면 CNN representation에서 개별 URL n-gram이나 keyword가 선형적으로 복원된다는 증거는 발견되지 않았다.

이 정도는 현재 데이터가 잘 지지합니다.

반면 아직은

> “피싱 QR에는 보편적인 시각적 signature가 존재한다.”

또는

> “CNN이 URL 어휘를 QR에서 복원한다.”

라고 쓰면 안 됩니다.

외부 F와 G가 모두 살아남고, 제가 위에서 제안한 **within-QR spatial null**까지 통과하면 그때는 상당히 강한 논문이 됩니다.

---

## 우선순위를 정리하면

| 우선도    | 수정/실험                                          | 이유                                        |
| ------ | ---------------------------------------------- | ----------------------------------------- |
| **P0** | `motifs.py` CI 강제 포함 코드 제거                     | 이전 통계 오류가 한 곳 남아 있음                       |
| **P0** | G를 fixed campaign-test 방식으로 비교                 | 현재 test set 86~88%가 바뀌어 효과 분리가 어려움        |
| **P0** | RQ2에서 `lexical` → `URL-derived content/signal` | probe 결과와 claim을 일치시켜야 함                  |
| **P1** | within-QR shuffled motif null                  | “patch frequency”와 “진짜 spatial motif”를 분리 |
| **P1** | OpenPhish를 외부 primary로 사용                      | 2026 temporal validation이 더 강함            |
| **P1** | unranked Common Crawl benign sensitivity       | popularity shortcut 검증                    |
| **P1** | benign-cleaning ablations 필수화                  | 외부 성능의 낙관 편향 검사                           |
| **P1** | motif occlusion topology-matched control       | 인과 주장 강화                                  |
| **P2** | bootstrap pseudo-p 표현 수정                       | 통계적 표현 엄밀화                                |
| **P2** | motif 결과도 seed-stratified CI로 통일               | 논문 전체 estimator 일관성                       |

그리고 문서 자체에도 작은 불일치가 하나 있습니다. `EXPERIMENT_DESIGN.md` 맨 앞은 아직 두 질문 Q1/Q2 구조인데, RESULTS와 README는 새 **RQ1~RQ3** 구조를 사용합니다.   최종 논문화 전에 **RQ1~RQ3를 유일한 canonical research questions로 통일**하는 게 좋습니다.

전체적으로 방향을 잘못 잡은 것은 아닙니다. 오히려 이전의 가장 큰 약점이었던 “공통 패턴이라는 말을 해놓고 실제 motif를 측정하지 않았다”는 문제를 제대로 정면으로 해결했습니다. 이제 중요한 건 **새로 나온 긍정 결과를 너무 빨리 ‘visual signature’로 승격시키지 않고, spatial null + external transfer + campaign holdout으로 한 번 더 압박하는 것**입니다.

