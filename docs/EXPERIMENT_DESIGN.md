# QR 모듈 기반 피싱 URL 분류 — 실험 설계 스펙 v1

대상 독자: 구현 워커. 이 문서만 보고 `qrphish` 패키지를 처음부터 작성할 수 있어야 한다.
범위: 1단계(WebPhish 단일 데이터셋, Colab T4). 2단계(PhishTank/Tranco 신규 수집)는 인터페이스만 정의한다.

---

## 0. 연구 명제와 논증 구조

QR은 URL의 무손실·결정적 인코딩이다. 따라서 "QR 이미지에서 피싱을 분류할 수 있는가"는 정보이론적으로 자명하게 참이고(디코딩 후 텍스트 분류기를 쓰면 됨), 논문의 실제 질문은 아래 셋이다. 이 **RQ1 ~ RQ3가 유일한 canonical 연구 질문**이며 14.1절, `docs/RESULTS.md`, `README.md`가 모두 같은 정의를 쓴다.

- **RQ1 (shortcut 통제 후 신호 존재)** 길이·QR 버전·패딩 경계 같은 shortcut을 모두 통제한 뒤에도 QR 모듈 격자에 phishing-discriminative signal이 남는가. 검증은 그 조건에서 성능이 라벨 셔플 바닥을 유의하게 넘는지로만 한다.
- **RQ2 (디코딩 없는 신호 접근)** CNN이 명시적 QR 디코딩 없이 QR의 공간 배치를 이용해 **URL 유래 내용·순서 신호(URL-derived content/order signal)**에 접근·활용(exploit)하는가. 정량 지표는 **디코딩 텍스트 참조 기준 대비 갭**이다. 문자·n-gram을 실제로 복원(recover)했다는 주장은 하지 않는다. 선형 프로브는 개별 n-gram·키워드가 표현에서 선형으로 복원된다는 증거를 찾지 못했다(14.2 D항, `RESULTS.md` 5-B절).
- **RQ3 (국소 motif)** 그 신호가 여러 피싱 QR에서 반복되는 국소 motif로 나타나며 새 도메인·데이터셋에서도 재현되는가.

논문은 3층 구조로 쓴다.

| 층 | 내용 | 역할 |
|---|---|---|
| 참조 기준 | URL char n-gram TF-IDF LR (디코딩 텍스트 참조 기준) | 디코딩된 텍스트를 직접 쓰는 **강한 베이스라인** |
| 하한선 | 버전 단독 LR, 길이 단독 LR, 바이트 히스토그램 LR | "메타 신호만으로 얼마나 되는가" |
| 본 실험 | 모듈 격자 CNN / MLP | 갭의 위치를 규정 |

**참조 기준의 지위**: char 1–5 gram TF-IDF + Logistic Regression은 **하나의 강한 텍스트 분류기**일 뿐 Bayes 최적 분류기가 아니다. QR이 URL의 결정적 변환이라는 점에서 정보론적으로 디코딩된 URL에 QR보다 적은 정보가 담기지는 않지만, 특정 LR이 그 정보론적 상한을 구현하지는 못한다. 따라서 이 값을 "상한선"이라 부르지 않으며, **CNN이 이를 넘더라도 누출의 증거가 아니다**. 보고하는 것은 부호 있는 갭(`gap_to_decoded_text` = 참조 − CNN) 하나뿐이고, 누출 판정용 sanity check는 두지 않는다. sanity에는 라벨 셔플 AUROC만 남긴다.

---

## 1. 설계 비판과 결정

### 1.1 [치명] 패딩 바이트가 길이를 완벽히 누출한다 — 가장 중요한 결함

`qrcode/util.py:550 create_data`는 데이터 비트 뒤에 종단자(0000) + 바이트 정렬 + `0xEC, 0x11` 교대 패딩으로 남은 데이터 코드워드를 전부 채운다. 고정 버전 안에서 **패딩 시작 코드워드 인덱스는 URL 길이의 결정적 함수**다. `0xEC=11101100`, `0x11=00010001`은 시각적으로 극히 두드러지는 주기 패턴이고, RS 인터리빙으로 격자 전역에 퍼지며, 마스크를 씌워도 주기성이 남는다.

즉 **버전 층화만으로는 길이 누출이 제거되지 않는다.** 층 안에서 benign 평균 20자 / phishing 평균 50자면 CNN은 "패딩이 어디서 시작하나"만 읽고 100%에 가까운 정확도를 낸다. 이전 실험의 "이미지 크기 프록시" 문제가 층 안으로 옮겨온 것뿐이다.

**결정 (필수, 주 조건)**
1. **엄격 길이 매칭(exact-length matching)**: URL 바이트 길이를 1자 단위 버킷으로 쪼개고, 각 버킷에서 두 클래스 중 적은 쪽 개수 `min(n_ben, n_phi)`만큼 양쪽에서 샘플링한다. 결과 데이터셋은 **길이 주변분포가 클래스 간 완전히 동일**하고, 따라서 패딩 경계 분포도 동일하다. 이것이 주 조건 `L-exact`다.

   > **적용 시점(중요)**: 매칭은 **그룹 분할 뒤에 split마다 따로** 건다. 층 전체에서 한 번만 맞추면 1.4의 그룹 분할과 5% 상한 다운샘플이 행을 통째로 옮기거나 지우면서 split 안의 길이 분포를 다시 깨뜨린다. 실제로 재현된다 — 어떤 길이 버킷의 benign이 전부 한 도메인 소속이면 그 버킷은 train에만 남고 test에는 phishing만 남아, test에서 길이가 다시 라벨을 알려준다. 매칭은 행 제거만 하므로 순서를 바꿔도 split 간 그룹 교집합 0은 유지된다. 파이프라인 순서는 **층 필터 → group_split(다운샘플 포함) → split별 match_by_length → split별 (label × length_bucket) 히스토그램 동일성 검사(실패 시 하드 에러)** 다.
2. **완화 매칭(`L-quantile`)**: 표본이 너무 줄면 5자 폭 버킷 매칭을 보조 조건으로 둔다. 이때 패딩 경계는 ±5자 범위로 남으므로 "잔여 누출 있음"을 명시 보고한다.
3. **무통제(`L-none`)**: 층화만 적용. 반드시 함께 보고하되 **주 결과로 쓰지 않는다.** `L-none`과 `L-exact`의 성능 차 자체가 논문의 핵심 수치 중 하나다("보고된 성능의 X%p는 길이 아티팩트였다").
4. **인과 절제 `PAD-rand`**: `create_data`를 대체해 패딩을 `0xEC/0x11` 교대 대신 **URL 해시 시드 기반 의사난수 바이트**로 채운 QR을 별도 생성한다. 규격 위반이지만(디코더 호환은 유지됨 — 패딩 바이트는 길이 필드 밖이라 디코딩에 영향 없음), "패딩 주기 패턴이 성능의 원인인가"를 직접 검증하는 유일한 방법이다. `L-none` 위에서 실행해 `L-none` 대비 성능 하락폭을 보고한다.

> 구현 주의: `PAD-rand`는 `qrcode.util.create_data`를 복사·수정한 `qrphish.qrgen.create_data_randpad`로 구현한다. 라이브러리 monkeypatch 금지(테스트 격리 위해 함수 주입 방식).

### 1.2 [높음] 길이 매칭 후 남는 표본 수 — 층별 최소 표본 규칙

버전(EC=L, 8비트 바이트 모드) 용량 경계: v1≤17, v2≤32, v3≤53, v4≤78, v5≤106바이트(코드로 `qrcode.util` 표에서 계산할 것, 하드코딩 금지).

기존 관측(benign 90%가 v2 ~ v3, 평균 36자 / phishing 평균 60자)에서 예상되는 문제:
- v2(18 ~ 32자)와 v3(33 ~ 53자)는 benign이 다수, phishing이 소수 → 매칭 후 개수는 phishing이 결정.
- v4 이상은 benign이 거의 없음 → **매칭 후 v4·v5+ 층이 소멸할 가능성이 높다.**

**결정: 층 채택 규칙(사전 등록)**
- 매칭 후 **클래스당 ≥ 1,000** → 주 결과(primary).
- 300 ≤ 클래스당 < 1,000 → 보조 결과(secondary). CI 폭 명시, 결론 근거로 단독 사용 금지.
- 클래스당 < 300 → **폐기**. 표에 "표본 부족으로 제외(n=…)"로 남긴다.
- v1은 처음부터 보고 전용(표본 수만 기재).

이 규칙은 **데이터를 보기 전에** config에 박아 넣고, 실제 개수는 `reports/stratum_counts.json`으로 먼저 산출한 뒤 어떤 층이 살아남는지 확정한다. **1차 산출물은 모델이 아니라 이 카운트 표다.** 층이 v2·v3만 살아남는 시나리오를 기본 가정으로 두고 진행하라.

### 1.3 [높음] 버전 층 내에서 EC 레벨을 고정하는 것의 의미

EC=L 고정은 필수다. EC 레벨을 바꾸면 같은 URL이 다른 버전·다른 데이터 용량으로 가므로 층 정의 자체가 흔들린다. 다만 **EC=L 고정은 "패딩 여유 공간이 최대"라는 뜻**이라 1.1의 누출을 최대화한다. 반대로 EC=H는 패딩 여유가 작아 누출이 줄지만 EC 코드워드 비중이 커져 데이터 모듈 해석이 어려워진다.

**결정**: 주 실험 EC=L. **강건성 확인용으로 v3 층 하나에 대해서만 EC=M 재실행**(`EC-M` 조건). EC 전체 격자 탐색은 하지 않는다(계산 낭비, 결론 불변 예상).

### 1.4 [높음] 도메인 그룹 분할 시 클래스 불균형 / 호스팅 집중

phishing URL은 `000webhostapp.com`, `weebly.com`, `blogspot.com`, `duckdns.org` 등 소수 eTLD+1에 대량 집중한다. eTLD+1 그룹 분할은 누출 방지에 **필수**지만 두 부작용이 있다.
- 거대 그룹 하나가 통째로 한 split에 들어가 split 간 클래스 비가 크게 깨진다.
- 그런 대형 그룹을 test에 넣으면 test 성능이 사실상 "그 호스팅 도메인 한 개를 맞히는 문제"가 된다.

**결정**
1. **그룹 정의**: `tldextract`의 `registered_domain`(eTLD+1). 빈 값(IP 주소 URL 등)은 호스트 문자열 자체를 그룹 키로.
2. **분할 알고리즘**: 그룹을 크기 내림차순으로 정렬해, 각 그룹을 "현재 split별 (총량, 클래스별 개수) 목표 대비 결핍이 가장 큰 split"에 배정하는 그리디 빈패킹. 목표 비 70/15/15, 클래스 비는 층 전체 비율을 따라가게 한다. 시드로 동점 처리와 그룹 순서 지터를 준다.
3. **초대형 그룹 상한**: 단일 그룹이 층 전체의 5%를 넘으면 **그 그룹 내부에서 층 전체 비율까지 무작위 다운샘플링**한다(누출 방지 유지, 편중 완화). 다운샘플 전후 개수를 기록한다.
4. **진단 지표 필수 보고**: split 간 eTLD+1 교집합=0 확인, split별 클래스 비, test에서 상위 5개 그룹이 차지하는 비율, **그룹 단위 성능 분산**(그룹별 정확도의 IQR). 특정 그룹 하나가 성능을 끌어올리고 있다면 그것을 본문에 쓴다.
5. **평가는 클러스터 부트스트랩**: 표본이 아니라 **그룹을 리샘플링**해 95% CI를 만든다. 표본 단위 부트스트랩은 CI를 과소평가한다.
6. **분할이 매칭보다 먼저다**: 위 2·3번이 행을 옮기고 지우므로, 1.1의 길이 매칭은 분할이 끝난 뒤 split마다 적용한다. 상세는 1.1의 "적용 시점" 참조.
7. **분할 결과 하드 검증**: 세 split이 모두 비어 있지 않고 두 클래스를 모두 포함해야 한다. 하나라도 어기면 `group_split`이 `ValueError`를 낸다(한 클래스뿐인 val은 AUROC가 NaN이라 조기종료·임계값 선택이 무의미해진다). 학습 루프도 val AUROC가 NaN이면 예외로 끊는다.

### 1.5 [높음] benign=Alexa 상위 도메인, phishing=경로 포함 — 데이터셋 본질 편향

benign은 대부분 `www.example.com` 형태의 맨 도메인이고 phishing은 긴 경로·쿼리를 가진다. 이것은 QR 실험의 결함이 아니라 **WebPhish 데이터셋 자체의 결함**이며, 어떤 인코딩을 써도 남는다. 숨기지 말고 정면으로 다룬다.

**결정**
- 정규화 조건 2종은 예정대로: `raw`(원문) / `norm`(소문자화 + 선행 `www.` 제거 + 말미 `/` 제거). `norm`은 `www.` 프록시(benign 61% vs phishing 13%)를 제거한다.
- **추가 조건 `path+`**: 경로 깊이 ≥1(호스트 뒤에 `/`+문자 존재)인 URL만 사용하는 부분집합. benign 표본이 급감할 수 있으므로 1.2 규칙에 따라 채택 여부를 카운트로 결정. 살아남으면 이것이 **가장 정직한 조건**이며 논문의 핵심 표로 올린다.
- 위협 타당성 절(Limitations)에 "benign 표집이 Alexa 상위이므로 결과는 '피싱 vs 인기 도메인' 구분이며 '피싱 vs 일반 웹' 구분이 아니다"를 명시. 2단계 Tranco 수집이 이 문제를 완화하는 근거로 연결된다.

### 1.6 [중간] 마스크 고정의 부작용, 그리고 언마스킹

`main.py:157` — `mask_pattern=None`이면 `best_mask_pattern()`이 `lost_point`를 최소화하는 마스크를 고른다. **선택된 마스크 인덱스는 콘텐츠의 함수이므로 그 자체가 약한 라벨 프록시**이고, 게다가 층 내 샘플마다 다른 비선형 변환이 걸려 CNN이 배워야 할 사상이 8배로 갈라진다.

**결정 (3조건)**
- `mask-fixed` (**주 조건**): `mask_pattern=0` 고정. 전체가 동일한 결정적 변환을 받으므로 학습·해석이 안정적.
- `mask-auto` (절제): 라이브러리 기본. 부수 산출물로 **마스크 인덱스의 클래스별 분포**를 보고하고, "마스크 인덱스 단독 LR" 베이스라인을 돌려 프록시 강도를 정량화한다.
- `mask-off` (**해석 전용 주력**): 마스크를 XOR로 되돌려 순수 비트 배치 격자를 입력으로 쓴다. 데이터 모듈 값 = 코드워드 비트 그 자체가 되어 Grad-CAM → 비트 → 문자 역추적이 왜곡 없이 성립한다. 규격 QR이 아니므로 "표현 학습 분석용"이라고 명시.

### 1.7 [중간] 기능 패턴 제거 시 입력 형태

데이터 모듈만 남기면 격자가 불규칙 구멍 뚫린 형태가 된다. 1차원으로 펴면 CNN의 2D 귀납편향이 사라지고, 0으로 채우면 "0"이 데이터 값 0과 구분되지 않는다.

**결정: 입력은 항상 `n×n` 2채널 텐서 (float32)**
- 채널 0: 모듈 값 (0/1). 기능 패턴 위치는 조건에 따라 0으로 채움.
- 채널 1: 데이터 모듈 마스크 (1=데이터, 0=기능 패턴). 버전 고정이므로 층 내에서 상수 채널이지만, **명시적으로 넣어 채널 0의 0이 "값 0"인지 "기능 패턴"인지 모델이 구분**할 수 있게 한다.
- 조건 `feat-all`: 채널 0에 기능 패턴 원래 값 포함.
- 조건 `feat-data`(주 조건): 기능 패턴 위치 채널 0 = 0.
- 상수 채널이 낭비라는 반론이 있지만 코드 단순성과 조건 간 텐서 형상 동일성이 더 중요하다. 유지한다.

**추가 절제 `shuffle-pos`**: 층 내 전체 샘플에 **동일한 고정 순열**로 모듈 위치를 섞은 입력. CNN 성능이 이때 급락하고 MLP는 불변이라면 "CNN이 공간적 국소성을 실제로 쓴다"는 증거가 된다. 반대로 CNN이 그대로면 국소 패턴 주장은 성립하지 않는다. **RQ2에 대한 가장 직접적인 검정이므로 필수 조건이다.**

### 1.8 [중간] 통계 절차

- 시드 5개 × (그룹 분할 재생성 + 모델 초기화 + 매칭 서브샘플링 재추출). 서브샘플링도 시드에 묶어야 매칭 우연에 의한 분산이 CI에 포함된다.
- 주 지표 **AUROC**(임계값 무관, 불균형 강건). 보조 F1(양성=phishing, 임계값은 **validation에서 선택**), Accuracy, AUPRC.
- CI: **5개 시드의 test 예측을 모두 모아** 한 번의 그룹 클러스터 부트스트랩 2,000회로 CI 하나를 만든다(`auroc_pooled`, `auroc_pooled_ci`). 시드마다 split이 다르므로 `(seed, row)`를 개별 표본으로 두고, 클러스터 키는 시드와 무관한 eTLD+1 그룹 id를 쓴다. 시드별 CI의 하한/상한을 평균내는 방식은 정식 CI가 아니므로 쓰지 않는다. 백분위 CI는 **순수 백분위**로 두고 점추정을 포함시키려 강제로 넓히지 않는다. 시드 간 표준편차는 그대로 별도 보고한다.
- 조건 간 비교는 **동일 split에서의 쌍체 차이** 분포로 한다(`evaluate.paired_cluster_bootstrap`). 두 조건이 같은 `url_mode`·`length_match`·시드를 쓰면 같은 분할과 같은 test 행을 공유하므로 쌍체가 성립한다. 공유하지 않는 비교(예: L-none 대 L-exact)는 표본 집합 자체가 다르므로 쌍체가 불가능하며, `paired=false`로 표기하고 비쌍체 차이만 보고한다.
- **p값의 지위**: 본문 주 결과는 ΔAUROC 점추정과 클러스터 부트스트랩 CI다. 보고하는 p값은 CI를 역전시켜 정의한 **bootstrap-tail pseudo-p**이고, 영가설 아래에서 재표집한 정식 귀무분포 p값이 아니다. 표·본문에서도 `p`가 아니라 `bootstrap-tail pseudo-p (Holm)`으로 표기하고 보조 지표로만 읽는다. H1·H4처럼 같은 test 행을 공유하는 쌍체 가설에는 **클러스터 인식 순열(randomization) 검정 p값을 추가로 낼 예정**이며, 그 값이 나오면 정식 p값 자리를 그것이 맡는다.
- **사전 등록된 주 가설 목록**(아래 4개)에만 Holm 보정을 적용한다. 나머지는 탐색적(exploratory)으로 표기.
  - H1: `L-exact/feat-data/mask-fixed` CNN AUROC > 0.5 (라벨 셔플 대비). **라벨 셔플은 train+val에만 건다.** test는 원 라벨로 평가해야 "라벨과 무관하게 학습된 모델이 실제 라벨을 맞히지 못한다"를 보는 것이 된다. test까지 섞으면 섞인 라벨을 맞히는지를 보게 되어 검정이 무의미해진다.
  - H2: 위 조건의 CNN AUROC < 디코딩 텍스트 참조 기준 AUROC (갭 존재). 참조 기준은 강한 텍스트 베이스라인이지 천장이 아니므로, 이 검정은 "갭이 남는가"를 묻는 것이지 누출 판정이 아니다.
  - H3: `L-exact` AUROC < `L-none` AUROC (길이 아티팩트 크기). **단서**: `L-none`과 `L-exact`는 표본 수·클래스 구성·URL 집합이 함께 달라지므로(예: v2는 exact N=12,118, none N=16,627) 이 차이는 길이 신호만 더한 순수한 개입이 아니라 훈련·평가 집합 구성까지 동시에 바꾼 비교다. 쌍체 부트스트랩도 성립하지 않는다. 길이 아티팩트의 **인과적 절제**로는 같은 표본 위에서 패딩만 바꾸는 `PAD-rand` 쪽이 더 가깝다.
  - H4: `shuffle-pos` AUROC < 정상 배치 AUROC (공간 구조 사용 여부)

### 1.9 [중간] 중복·근중복

- 정규화 후 **완전 중복 URL 제거**(1,295건 보고됨). 어느 조건의 정규화를 기준으로 할지 문제가 되므로: **`raw`와 `norm` 각각 별도로 dedup**하고 두 파이프라인의 표본 수를 각각 보고한다(공통 부분집합 강제 금지 — 정규화 효과를 죽인다).
- 근중복(같은 eTLD+1 + 같은 경로 앞부분)은 제거하지 않는다. 대신 eTLD+1 그룹 분할이 train/test 누출을 막는다.
- 라벨 충돌(동일 URL이 양쪽 클래스) 발견 시 **양쪽 모두 제거**하고 개수 보고.

### 1.10 [낮음 ~ 중간] 그 밖의 결정

- **잔여 비트(remainder bits)**: 데이터 모듈 수가 총 코드워드 비트 수의 배수가 아닌 버전이 있다. 실측(EC=L)으로 **v2 ~ v6은 7비트, v1과 v7 이상은 0비트**다. 이 비트들은 항상 0으로 배치된 뒤 마스크만 적용되므로 **클래스 정보가 없다**. 역매핑에서 `remainder`로 태깅하고, `feat-data` 조건에서 데이터 마스크에 포함시키되 해석에서는 제외한다.
- **모드 선택**: `add_data(..., optimize=20)` 기본값은 URL을 numeric/alnum/byte 청크로 쪼개 인코딩을 바꾼다. 클래스마다 청크 구조가 달라져 통제 불가능한 혼입이 생기고 역매핑도 청크별로 갈라진다. → **`optimize=0` 고정(전체를 단일 8비트 바이트 청크)**. 이건 재논의 불가 수준의 필수 결정이다.
- **EC 코드워드**: RS 패리티는 데이터의 결정적 함수이므로 정보 누출이 아니라 정보 재배치다. 제거하지 않는다. 역매핑에서 `ec(block=i)`로 태깅.
- **border**: `border=0`. 흰 여백은 상수라 무의미하고 격자 크기만 키운다.

---

## 2. qrcode 8.2 소스 레벨 검토 — 역매핑 가능성 (실측 검증 완료)

scratchpad venv(`qrcode==8.2`)에서 실제로 확인했다.

### 2.1 확인된 사실

- `main.py:473 map_data`의 지그재그 순회는 `col`을 `n-1`부터 2씩 감소(6열 스킵), 행은 상하 교대. 순회 중 `modules[row][c] is None`인 칸만 데이터 비트를 소비한다. 즉 **기능 패턴 마스크 = `map_data` 진입 시점의 non-None 집합**이며, 여기엔 파인더/타이밍/정렬 + `setup_type_info`가 이미 써 넣은 포맷 정보 + (v≥7) 버전 정보 + 다크 모듈이 모두 포함된다.
- 비트 순서는 `data[byteIndex] >> bitIndex`, MSB 우선. 따라서 **전역 비트 인덱스 i ↔ 코드워드 i//8, 비트 7-(i%8)** 이고, i번째 비트의 모듈 좌표는 순회 순서 리스트의 i번째 원소다.
- 마스크는 `util.mask_func(pattern)(row, col)`이 True면 XOR. 좌표만의 함수이므로 완전 가역.
- 실측: `www.google.com` → v1, 데이터 모듈 208개 = 26코드워드×8, 나머지 0. 긴 피싱 URL → v4, 데이터 모듈 807개 = 100코드워드×8 + **잔여 7비트**. 재구성한 격자가 `qr.modules`와 **전 모듈 일치**함을 8개 마스크 전수 대조로 확인했다.
- 인터리빙: `util.create_bytes`가 블록별 데이터 코드워드를 라운드로빈으로, 이어서 EC 코드워드를 라운드로빈으로 쌓는다. `base.rs_blocks(version, ec)`가 블록 (total_count, data_count) 목록을 준다. → **인터리빙 인덱스 → (블록, 블록 내 오프셋) → 프리인터리브 버퍼 바이트 오프셋** 역산이 순수 산술로 가능. v4/L은 단일 블록(100,80)이라 항등에 가깝지만, 코드는 다중 블록 일반형으로 작성해야 한다(EC=L에서는 **v6부터** 블록이 (86,68)×2로 갈린다. v5/L은 아직 단일 블록 (134,108)이다).
- 헤더: 모드 4비트 + 길이 필드(v1 ~ 9 byte 모드 = 8비트) = **12비트**. 따라서 URL의 k번째 문자는 프리인터리브 버퍼의 비트 `12+8k … 12+8k+7`을 차지하며 **바이트 경계에 정렬되지 않는다**(코드워드 2개에 걸침). 역매핑은 반드시 **비트 단위**로 해야 한다.

### 2.2 자체 재구현 범위 (최소)

라이브러리를 포크할 필요는 없다. 다음 세 함수만 자체 구현한다.

1. `bit_placement_order(version, ec) -> list[(row, col)]` — `map_data` 순회를 값 대신 좌표만 기록하도록 복제. **약 25줄.** 기능 패턴 마스크는 `QRCode`를 서브클래싱해 `map_data` 진입 시 non-None 집합을 캡처하는 방식으로 얻는다(순회 로직 복제와 마스크 획득을 분리하면 검증이 쉽다). 버전·EC별로 **한 번만** 계산해 캐시.
2. `interleave_inverse(version, ec) -> list[SourceRef]` — 인터리브 코드워드 인덱스 → `('data', buf_byte_index, block)` 또는 `('ec', None, block)`.
3. `create_data_randpad(...)` — `util.create_data` 복제 + 패딩만 교체 (`PAD-rand` 조건 전용).

마스크 함수, RS 인코딩, rs_blocks 표, 기능 패턴 배치는 **전부 라이브러리 것을 그대로 쓴다.**

### 2.3 정확성 검증 게이트 (테스트로 강제)

`bit_placement_order` + 마스크 + `data_cache`로 격자를 재구성해 `qr.modules`와 **완전 일치**하는지를, 무작위 URL 500개 × 버전 v1 ~ v6 × 마스크 0 ~ 7에 대해 확인한다. 하나라도 불일치하면 파이프라인 전체를 중단한다. 이 테스트가 통과해야 역매핑 결과를 논문에 쓸 수 있다.

---

## 3. 저장소 구조

```
qrphish/
  __init__.py
  config.py          # dataclass 스키마 + yaml 로더/검증
  urls.py            # 정규화, dedup, eTLD+1 추출
  splits.py          # 그룹 그리디 분할, 길이 매칭 서브샘플링
  qrgen.py           # 인코딩 → 모듈 행렬 + 메타
  mapping.py         # 기능 패턴 마스크, 비트 순서, 인터리브 역산, 모듈→문자 역매핑
  dataset.py         # 층 빌드 → npz/parquet, torch Dataset
  models.py          # SmallCNN, BitMLP, LinearProbe
  baselines.py       # char n-gram LR, char-CNN, version-only, length-only, byte-hist, mask-index
  train.py           # 학습 루프, 조기종료, 체크포인트
  evaluate.py        # 지표, 클러스터 부트스트랩 CI, 결과 JSON
  explain.py         # Grad-CAM → 모듈 → 비트 → 문자 기여도
  runner.py          # 실험 매트릭스 실행 오케스트레이션
  cli.py             # typer/argparse 엔트리포인트
tests/
notebooks/colab_run.ipynb
configs/base.yaml, configs/matrix.yaml
```

패키징: `pyproject.toml`, uv, Python 3.11+. 의존성: `torch`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `qrcode==8.*`, `tldextract`, `pyyaml`, `typer`. 이미지 라이브러리(Pillow) 불필요 — **PNG를 만들지 않는다.**

---

## 4. 모듈별 공개 API

```python
# urls.py
def normalize_url(url: str, mode: Literal["raw", "norm"]) -> str: ...
    # raw: 원문 그대로(공백 strip만)
    # norm: 소문자화, 선행 "www." 1회 제거, 말미 "/" 제거
def etld1(url: str, extractor: tldextract.TLDExtract) -> str: ...
    # registered_domain, 비면 host 문자열, 그것도 비면 "<empty>"
def path_depth(url: str) -> int: ...
def load_webphish(csv_path: Path, mode: str) -> pd.DataFrame: ...
    # 컬럼: url, label(int: phishing=1), group, url_len, path_depth
    # 처리: Category 매핑(ham=0, spam=1), normalize, 완전중복 제거,
    #       라벨 충돌 URL 양쪽 제거. 제거 통계를 dict로 함께 반환.
```

```python
# qrgen.py
@dataclass(frozen=True)
class QRArtifact:
    version: int
    ec: int
    mask_pattern: int
    n: int                      # 4*version+17
    modules: np.ndarray         # (n, n) bool, border=0
    data_cache: list[int]       # 인터리브된 코드워드
    n_data_bits: int

def encode(url: str, *, ec: int = ERROR_CORRECT_L, mask_pattern: int | None = 0,
           version: int | None = None, randpad_seed: int | None = None) -> QRArtifact: ...
    # optimize=0 고정. version=None이면 best_fit.
    # randpad_seed가 있으면 create_data_randpad 사용.
def natural_version(url: str, ec: int) -> int: ...   # best_fit 결과만 필요할 때(층 배정용, 격자 생성 없음)
```

```python
# mapping.py  (config 데이터클래스 이름은 Condition이 아니라 ConditionConfig다)
@lru_cache
def function_mask(version: int, ec: int) -> np.ndarray: ...        # (n,n) bool, True=기능 패턴
@lru_cache
def bit_placement_order(version: int, ec: int) -> np.ndarray: ...  # (n_data_bits, 2) int, i번째 비트의 (row,col)
@lru_cache
def interleave_inverse(version: int, ec: int) -> list[tuple[str, int | None, int]]: ...

@dataclass(frozen=True)
class ModuleProvenance:
    kind: np.ndarray   # (n,n) uint8: 0=function 1=char 2=length_header 3=mode_header
                       #              4=pad 5=ec 6=remainder
    char_index: np.ndarray  # (n,n) int32, kind==1일 때 URL 문자 인덱스, 아니면 -1
    block_index: np.ndarray # (n,n) int32, kind==5일 때 RS 블록, 아니면 -1

def provenance(url: str, art: QRArtifact) -> ModuleProvenance: ...
def unmask(modules: np.ndarray, mask_pattern: int) -> np.ndarray: ...
def verify_roundtrip(url: str, art: QRArtifact) -> bool: ...   # 3장 검증 게이트
```

```python
# splits.py
def match_by_length(df, bucket: int | None, seed: int) -> pd.DataFrame: ...
    # bucket=1 -> L-exact, 5 -> L-quantile, None -> L-none(항등)
    # 단일 프레임용 저수준 함수. 파이프라인은 아래 within_splits 쪽을 쓴다.
def group_split(df, ratios=(0.70,0.15,0.15), seed=0, max_group_frac=0.05,
                *, require_valid: bool = True) -> tuple[pd.DataFrame, dict]: ...
    # split 컬럼 추가 + 진단 dict(그룹 교집합, split별 클래스비, 상위그룹 점유율, 다운샘플 내역)
    # require_valid=True면 빈 split / 단일 클래스 split을 ValueError로 막는다.
def match_by_length_within_splits(df, bucket: int | None, seed: int
                                 ) -> tuple[pd.DataFrame, dict]: ...
    # split별로 각각 매칭. group_split **뒤에** 부른다(1.1 적용 시점 참조).
def length_match_report(df, bucket) -> dict: ...      # split별 히스토그램 동일성 진단
def assert_length_matched(df, bucket) -> dict: ...    # 위 진단이 깨졌으면 ValueError
```

```python
# dataset.py
def build_stratum(df, version_spec: str, cond: ConditionConfig, out: Path, *,
                  qr: QRConfig | None = None, condition_id: str = "",
                  rules: StratumRules | None = None, git_sha: str = "") -> StratumMeta: ...
    # version_spec: "v2" | "v3" | "v4" | "v5plus"
    # qr: EC/mask_mode/mask_pattern/pad_mode의 출처. mask_mode="off"의 정본 적용 시점이 여기다.
    # condition_id / rules / git_sha: stratum_meta.json에 기록되는 메타(등급 판정 포함).
def load_stratum(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]: ...
class QRGridDataset(torch.utils.data.Dataset): ...   # __getitem__ -> (2,n,n) float32, label
    # label_shuffle_seed + label_shuffle_rows(bool 마스크)로 셔플 범위를 지정한다.
    # 러너는 train+val 행만 넘긴다(1.8 H1). canonical_data_mask는 층에 등장하는
    # 모든 버전 데이터마스크의 합집합이다(v5plus 혼합 버전 층 대응).
```

```python
# models.py
class SmallCNN(nn.Module):
    """in_ch=2. Conv(32,3)-BN-ReLU x2 -> MaxPool -> Conv(64,3) x2 -> MaxPool
       -> Conv(128,3) -> GlobalAvgPool -> Dropout(0.3) -> Linear(1).
       패딩 same. n이 21~57로 작으므로 풀링 2회까지만."""
class BitMLP(nn.Module):
    """flatten(데이터 모듈 비트만, 배치 순서) -> 512 -> 256 -> 1, Dropout 0.3."""
class LinearProbe(nn.Module): ...
```

```python
# evaluate.py
def evaluate(model, loader, groups: np.ndarray, n_boot: int = 2000, seed: int = 0,
             threshold: float | None = None) -> dict: ...
    # AUROC/AUPRC/F1/Acc + 그룹 클러스터 부트스트랩 95% CI
    # threshold=None이면 같은 확률에서 F1 최대 임계값을 고른다. 학습 때 val에서 고른
    # 임계값을 test에 그대로 쓰려면 명시적으로 넘긴다(러너가 그렇게 한다).
def metrics_from_probs(y, p, groups, n_boot=2000, seed=0, threshold=None) -> dict: ...
    # 확률 배열에서 직접 지표를 낸다. 베이스라인도 이 경로를 쓴다.
```

```python
# explain.py
def gradcam(model, x, target_layer) -> np.ndarray: ...          # (n,n) float, 입력 해상도로 업샘플
def attribute_to_chars(cam: np.ndarray, prov: ModuleProvenance, url: str) -> np.ndarray: ...
    # (len(url),) float — 각 문자에 매핑된 모듈들의 CAM 합 / 모듈 수
    # prov.char_index는 UTF-8 **바이트** 인덱스라 byte_to_char_index로 환산해 귀속한다.
def byte_to_char_index(url: str) -> np.ndarray: ...   # 바이트 오프셋 -> 문자 인덱스
def attribute_by_kind(cam, prov) -> dict[str, float]: ...       # kind별 CAM 질량 비율
```

```python
# stage2 (인터페이스만, NotImplementedError)
def load_external(source: Literal["phishtank","openphish","tranco"], path: Path) -> pd.DataFrame: ...
```

---

## 5. 데이터 아티팩트 포맷

층 하나 = `artifacts/{cond_id}/{stratum}/` 아래 두 파일.

**`grids.npz`** (compressed)
- `X_packed`: `np.packbits`한 uint8, shape `(N, ceil(n*n/8))` — 모듈 값 채널만 저장(마스크 채널은 `function_mask`로 재생성).
- `y`: uint8 `(N,)`
- `n`: int, `version`: int, `ec`: int, `mask_pattern`: int (샘플별로 다르면 `mask_used` 배열 추가)
- `split`: uint8 `(N,)` 0=train 1=val 2=test
- `group_id`: int32 `(N,)`

**`meta.parquet`**: `row_id, url, label, group, url_len, path_depth, split, version, mask_used, n_pad_bytes, first_pad_codeword`
> `first_pad_codeword`는 누출 진단용이다. 반드시 저장하고, **클래스별 분포가 겹치는지 그림으로 확인**한 뒤에야 그 층을 주 결과로 쓴다.

**`stratum_meta.json`**: 조건 id, 원본 개수, 매칭 후 개수, split별 클래스비, 그룹 진단, 생성 시각, qrphish 버전, git sha. `extra.n_by_split_class`에 **split별 매칭 후 클래스별 개수**를, `extra.length_match`에 split별 길이 히스토그램 동일성 진단을 남긴다(1.1의 적용 시점 결정에 대한 증거).

메모리: v3(n=29)에서 N=20,000이면 packed로 약 2.1MB. Colab에서 전부 RAM 상주 가능. **PNG 생성 금지, 리사이즈 금지.**

---

## 6. Config 스키마 (`configs/base.yaml`)

```yaml
seed_list: [0, 1, 2, 3, 4]
data:
  csv_path: "data/webphish.csv"
  category_col: "Category"
  url_col: "Data"
  positive_label: "spam"
qr:
  ec: "L"                 # L|M|Q|H
  optimize: 0             # 고정. 변경 금지
  border: 0
  mask_mode: "fixed"      # fixed|auto|off
  mask_pattern: 0
  pad_mode: "spec"        # spec|random
condition:
  url_mode: "norm"        # raw|norm
  length_match: "exact"   # exact|quantile|none
  length_bucket: 1
  features: "data_only"   # data_only|all
  path_filter: false      # true면 path_depth>=1만
  shuffle_positions: false
strata: ["v2", "v3", "v4", "v5plus"]
stratum_rules:
  primary_min_per_class: 1000
  secondary_min_per_class: 300
split:
  ratios: [0.70, 0.15, 0.15]
  max_group_frac: 0.05
model:
  arch: "small_cnn"       # small_cnn|bit_mlp|linear_probe
  lr: 1.0e-3
  weight_decay: 1.0e-4
  batch_size: 256
  max_epochs: 60
  patience: 8             # val AUROC 기준
  class_weight: "balanced"
  amp: true
eval:
  n_bootstrap: 2000
  primary_metric: "auroc"
output_dir: "artifacts"   # 층 아티팩트·모델·results.json
reports_dir: "reports"    # P0 카운트 표, 진단 json, 집계 csv (러너에 폴백 없음)
```

검증: 로더는 `optimize != 0`, `border != 0`, 알 수 없는 키를 **에러**로 처리한다.

---

## 7. 실험 매트릭스

조건 id 규칙: `{url_mode}-{length_match}-{features}-{mask_mode}-{arch}[-{extra}]`

### P0 — 이것부터. 모델 학습 전에 반드시 끝낸다
| # | 산출물 | 목적 |
|---|---|---|
| P0.1 | `reports/stratum_counts.json` + 표 | 각 조건에서 층별 매칭 후 표본 수. **어떤 층이 살아남는지 여기서 확정** |
| P0.2 | `first_pad_codeword` 클래스별 분포 그림 | `L-exact`에서 완전 중첩 확인 |
| P0.3 | 3장 라운드트립 검증 통과 | 역매핑 신뢰성 게이트 |
| P0.4 | 그룹 분할 진단 리포트 | 교집합 0, split별 클래스비, 상위 그룹 점유율 |

### P1 — 주 실험 (살아남은 각 층 × 5시드)
| id | 설명 |
|---|---|
| `norm-exact-data_only-fixed-small_cnn` | **주 결과** |
| `norm-exact-data_only-fixed-bit_mlp` | 공간 귀납편향 필요성 |
| `norm-exact-data_only-fixed-small_cnn-shufflepos` | H4 |
| `norm-exact-*-labelshuffle` | 영가설 바닥 |
| baselines: `charngram_lr`, `charcnn`, `version_lr`, `length_lr`, `bytehist_lr` | **동일 층 부분집합에서** 실행 |

### P2 — 절제
`L-none` / `L-quantile` (H3) · `raw` vs `norm` · `feat-all` vs `feat-data` · `mask-auto` / `mask-off` + 마스크 인덱스 LR · `PAD-rand` · `path+` · v3에서 `EC-M`

### P3 — 해석
`mask-off` 최고 성능 모델에 Grad-CAM → kind별 CAM 질량 비율 → 문자 위치 기여도 집계(클래스 평균, 상대 위치 정규화) → 상위 기여 문자 n-gram과 char n-gram LR 계수의 상관.

### 계산 예산
층 3개 × P1 5조건 × 5시드 ≈ 75런. 층당 N≈20k, n≤41, SmallCNN 60에폭 → T4에서 런당 2 ~ 5분. **P1 총 4 ~ 6시간.** P2는 조건당 주 층(v3) 하나만 돌려 3 ~ 4시간. 총 10시간 이내로 무료 T4 세션에 맞춘다. 전체 층 × 전체 조건 격자는 하지 않는다.

---

## 8. 결과 JSON 스키마 (`artifacts/{cond}/{stratum}/results.json`)

```json
{
  "schema_version": 1,
  "condition_id": "norm-exact-data_only-fixed-small_cnn",
  "stratum": "v3",
  "tier": "primary",
  "config": { "...전체 config 스냅샷..." },
  "provenance": { "git_sha": "...", "qrphish_version": "0.1.0",
                  "torch": "2.x", "device": "T4", "timestamp": "..." },
  "data": { "n_total": 0, "n_train": 0, "n_val": 0, "n_test": 0,
            "class_ratio": {"train":0.5,"val":0.5,"test":0.5},
            "n_groups": {"train":0,"val":0,"test":0},
            "top5_test_group_frac": 0.0 },
  "per_seed": [
    { "seed": 0, "best_epoch": 0, "threshold": 0.5,
      "test": { "auroc": 0.0, "auprc": 0.0, "f1": 0.0, "acc": 0.0,
                "auroc_ci": [0.0,0.0], "f1_ci": [0.0,0.0] },
      "val":  { "auroc": 0.0 },
      "group_perf_iqr": 0.0 }
  ],
  "aggregate": { "auroc_mean": 0.0, "auroc_sd_across_seeds": 0.0,
                 "auroc_pooled": 0.0, "auroc_pooled_ci": [0.0,0.0],
                 "f1_mean": 0.0, "acc_mean": 0.0,
                 "decoded_text_reference_auroc": 0.0, "gap_to_decoded_text": 0.0 },
  "sanity": { "label_shuffle_auroc": 0.5 },
  "explain": { "cam_mass_by_kind": {"char":0.0,"pad":0.0,"ec":0.0,"remainder":0.0} }
}
```

집계기(`runner.py`)가 모든 `results.json`을 모아 `reports/table_*.csv`를 만든다.

---

## 9. 테스트 항목 (`tests/`, pytest, 로컬 CPU에서 전부 통과해야 함)

**정확성 (필수 게이트)**
1. `test_roundtrip_exact` — 무작위 URL 500 × v1 ~ v6 × 마스크 0 ~ 7: 재구성 격자 == `qr.modules`. (3.3절)
2. `test_bit_count` — `len(bit_placement_order) == (~function_mask).sum()`, 그리고 `8*len(data_cache) + remainder`와 일치.
3. `test_interleave_inverse` — 단일 블록(v4/L)에서 항등, 다중 블록(v6/L 이상)에서 `create_bytes` 결과와 인덱스 대조.
4. `test_provenance_causal` — URL의 k번째 문자를 바꾸면 실제로 변하는 모듈 집합 ⊆ (kind==1 & char_index==k) ∪ (kind==5 & 해당 블록). **가장 강한 역매핑 검증.**
5. `test_unmask_involution` — `unmask(unmask(m, p), p) == m`.

**누출 방지**
6. `test_length_matching` — `L-exact` 후 **split마다** 클래스별 길이 히스토그램이 완전 동일. 회귀 케이스(길이 버킷 하나의 benign이 단일 도메인에 몰린 층)를 함께 고정한다.
7. `test_pad_boundary_overlap` — `L-exact`에서 `first_pad_codeword`의 클래스별 분포가 동일.
8. `test_group_disjoint` — split 간 eTLD+1 교집합 = ∅ (모든 시드).
9. `test_no_dup_url` — split 전체에 중복 URL 없음.
10. `test_max_group_frac` — 어떤 그룹도 층의 5% 초과하지 않음.

**재현성 / 형상**
11. `test_deterministic_pipeline` — 동일 시드 2회 실행 시 `X_packed`, `split`, `group_id` 바이트 동일.
12. `test_tensor_shape` — 모든 조건에서 `(2, n, n)`, `n == 4*version+17`.
13. `test_npz_roundtrip` — packbits/unpackbits 무손실.
14. `test_label_shuffle_auroc` — train+val 라벨만 셔플해 학습한 뒤(**test는 원 라벨**) test AUROC의 95% CI가 0.5를 포함(소형 층으로 빠르게).

---

## 10. Colab 노트북 흐름 (`notebooks/colab_run.ipynb`)

```
[1] !pip -q install "git+https://github.com/<user>/CNN-QR-phishing-detector.git@main"
[2] GPU 확인 (torch.cuda.get_device_name)
[3] 데이터: Drive 마운트 후 CSV 1개 경로 지정 (기본),
    또는 kaggle.json 업로드 후 `kaggle datasets download -d guchiopara/look-before-you-leap`
    → 어느 쪽이든 최종적으로 `DATA_CSV` 변수 하나로 수렴
[4] cfg = load_config("configs/base.yaml"); cfg.data.csv_path = DATA_CSV
[5] P0 실행: qrphish.runner.run_p0(cfg) → 층 카운트 표 + 패딩 경계 그림 + 분할 진단 출력
    ★ 여기서 어떤 층이 primary인지 눈으로 확인하고 진행 여부 결정
[6] qrphish.runner.run_matrix(cfg, "P1") → 진행 표시, results.json들
[7] qrphish.runner.run_matrix(cfg, "P2")
[8] qrphish.runner.run_explain(cfg, best_condition)
[9] 결과 디렉터리를 Drive로 복사 (세션 종료 대비) + reports/*.csv 미리보기
```

원칙: 노트북에는 **로직을 쓰지 않는다.** 전부 패키지 함수 호출이고, 노트북은 설정 + 실행 + 표시만 한다. 셀 5 이후는 중단·재개 가능해야 하므로 `results.json`이 이미 있는 조합은 건너뛴다.

---

## 11. 논문용 표·그림 목록

**표**
- T1 데이터셋 구성과 필터링 흐름(원본 → dedup → 층화 → 길이 매칭, 각 단계 잔존 수)
- T2 층별 표본 수와 채택 등급(primary/secondary/dropped)
- T3 **주 결과**: 층 × {CNN, MLP, char n-gram LR, char-CNN, version LR, length LR, byte-hist LR, label shuffle}, AUROC/F1 + 95% CI
- T4 절제: `L-none` vs `L-quantile` vs `L-exact` 성능 차 (H3)
- T5 절제: `raw`/`norm`, `feat-all`/`feat-data`, `mask-fixed`/`auto`/`off`, `PAD-rand`, `path+`
- T6 결정 요약(12절 표를 논문 부록으로)

**그림**
- F1 파이프라인 도식: URL → 정규화 → 층화 → 길이 매칭 → 모듈 행렬 → 2채널 텐서
- F2 클래스별 URL 길이 분포 (매칭 전/후) — 누출 통제의 시각적 증거
- F3 클래스별 `first_pad_codeword` 분포 (매칭 전/후)
- F4 층별 AUROC 막대 + CI, 디코딩 텍스트 참조 기준과 셔플 바닥을 수평선으로 겹침
- F5 `shuffle-pos` 대 정상 배치 (CNN vs MLP 2×2) — H4
- F6 클래스 평균 Grad-CAM 히트맵 (모듈 격자 위, 기능 패턴 윤곽 오버레이)
- F7 kind별 CAM 질량 비율 스택 막대 (char/pad/ec/remainder)
- F8 URL 상대 위치별 문자 기여도 곡선, 클래스별
- F9 (있으면) 상위 기여 문자 n-gram vs char n-gram LR 계수 산점도

---

## 12. 결정 요약

| # | 쟁점 | 결정 | 등급 |
|---|---|---|---|
| 1 | 패딩 바이트 길이 누출 | 주 조건은 1자 버킷 **엄격 길이 매칭**. `L-none`/`L-quantile`은 절제로만. `PAD-rand`로 인과 검증 | 필수 |
| 2 | 층별 최소 표본 | 클래스당 ≥1000 primary / ≥300 secondary / <300 폐기. 데이터 보기 전 사전 등록 | 필수 |
| 3 | 인코딩 모드 | `optimize=0` 고정 (단일 8비트 바이트 청크) | 필수 |
| 4 | EC 레벨 | 주 실험 L 고정. v3 층에 한해 M 강건성 확인 | 권장 |
| 5 | 마스크 | 주 조건 `mask_pattern=0` 고정. `auto`는 절제 + 마스크인덱스 LR. `mask-off`(언마스킹)는 해석 주력 | 권장 |
| 6 | 기능 패턴 | 입력은 항상 2채널 `(값, 데이터마스크)` `n×n`. 제거는 값 채널 0으로 | 권장 |
| 7 | 공간성 검정 | 고정 순열 `shuffle-pos` 조건을 **필수**로 추가 (H4) | 필수 |
| 8 | 그룹 분할 | eTLD+1 그리디 빈패킹, 단일 그룹 5% 상한 다운샘플, 그룹 클러스터 부트스트랩 CI | 필수 |
| 9 | 데이터셋 본질 편향 | `path+` 부분집합 조건 추가, Limitations에 명시, 2단계로 연결 | 권장 |
| 10 | 정규화 | `raw`/`norm` 각각 독립 dedup, 공통 부분집합 강제 안 함 | 권장 |
| 11 | 통계 | 5시드(분할+서브샘플+초기화), 주 지표 AUROC, 주 가설 4개에만 Holm 보정 | 필수 |
| 12 | 역매핑 구현 | 자체 구현은 `bit_placement_order`, `interleave_inverse`, `create_data_randpad` 3개뿐. 라운드트립 테스트를 게이트로 | 필수 |
| 13 | 잔여 비트 | 정보 없음. `remainder`로 태깅, 해석에서 제외 | 확정 |
| 14 | 산출 순서 | **P0(카운트·누출 진단·검증 게이트)를 통과하기 전에는 모델을 학습하지 않는다** | 필수 |

---

## 13. 열린 질문 (사용자 확인 필요)

1. **`L-exact` 후 v2·v3만 남고 v4·v5+가 폐기되는 시나리오를 수용하는가?** 수용한다면 논문의 주장은 "짧은 URL 구간"으로 한정된다. 대안은 층을 버전이 아니라 길이 구간으로 재정의하고 버전은 보고 변수로 내리는 것인데, 그러면 격자 크기가 층 내에서 달라져 모델 하나로 못 돌린다(패딩 필요). 어느 쪽인가.
2. **`PAD-rand`(비규격 QR) 조건을 논문에 넣는가?** 방법론적으로는 가장 강한 인과 증거지만 "실제 QR이 아니다"라는 리뷰어 반박을 받는다. 부록으로 내릴지 본문에 둘지.
3. **`mask-off`(언마스킹) 격자를 본문 결과로 쓰는가, 해석 부록으로만 쓰는가?** 성능이 `mask-fixed`보다 높게 나올 가능성이 있는데, 그러면 "규격 QR에서의 성능"과 "표현 학습 상한"이 갈린다.
4. **char-CNN 참조 기준을 실제로 구현할 것인가, char n-gram LR만으로 충분한가?** 후자만으로도 논증은 성립하고 예산이 절약된다.
5. **1단계에서 2단계 데이터를 조금이라도 건드릴 것인가?** 지금 스펙은 인터페이스만 두고 완전히 미룬다. 리뷰 대응상 Tranco benign 5k 정도라도 1단계 말미에 넣는 선택지가 있다.
6. **benign 클래스의 `www.` 제거 정책**: `norm`은 선행 `www.`만 1회 제거한다. `m.`, `www2.` 같은 변형도 제거할지.
7. **결과 저장소**: artifacts를 git에 커밋할지, Drive에만 둘지. 층 npz는 층당 수 MB라 커밋 가능한 크기이긴 하다.

### 결정 (2026-09-05)

위 열린 질문 7개에 대한 확정 답이다. 이후 구현·집필은 이 결정을 따른다.

1. **층 소멸 시나리오 수용.** `L-exact` 후 v2·v3만 남더라도 그대로 진행한다. 층은 버전 기준을 유지하고(격자 크기가 층 내에서 고정되어야 모델 하나로 돌릴 수 있다), 논문의 주장 범위를 "짧은 URL 구간"으로 한정해 명시한다. v4는 1.2절의 secondary 기준(클래스당 ≥300)을 충족할 때만 표에 올린다.
2. **`PAD-rand`는 부록.** 인과 증거로서의 가치는 인정하되 비규격 QR이라는 반박 여지가 있으므로 본문이 아니라 부록에 싣는다.
3. **`mask-off`는 해석 부록 전용.** 본문 성능 표는 `mask-fixed`만 쓴다. 언마스킹 격자의 성능 수치는 "표현 학습 상한"으로 부록에 따로 적는다.
4. **디코딩 텍스트 참조 기준은 char n-gram LR만 필수.** char-CNN은 예산이 남을 때만 돌리는 선택 항목이다.
5. **2단계 데이터는 완전히 미룬다.** 1단계에서는 인터페이스(`load_external`)만 두고 실제 수집·학습은 하지 않는다.
6. **`norm`은 선행 `www.`만 1회 제거한다.** `m.`, `www2.` 같은 변형은 건드리지 않는다.
7. **결과 저장 정책.** `results.json`과 `reports/`는 git에 커밋한다. 층 `grids.npz`는 커밋하지 않고 Drive에 둔다.

---

## 14. 2차 실험 (2026-09-07)

연구 질문을 세 개로 재구성하고, 1차 결과의 통계·해석을 정리한다.

### 14.1 재구성된 연구 질문

| 새 연구 질문 | 현재 상태 |
|---|---|
| **RQ1.** 길이·버전·패딩 등의 shortcut을 통제한 후에도 QR 모듈 격자에 phishing-discriminative signal이 존재하는가? | 거의 답함 |
| **RQ2.** CNN은 명시적 QR decoding 없이 QR의 공간 배치를 이용해 URL 유래 내용·순서 신호(URL-derived content/order signal)에 접근하는가? | 상당 부분 답함. 개별 n-gram·키워드의 선형 복원 증거는 없다 |
| **RQ3.** 그 signal은 여러 phishing QR에서 반복되는 local visual motif로 나타나며, 새로운 도메인·데이터셋에서도 재현되는가? | 다음 실험의 핵심 |

증거 단계는 이렇다.

| 주장 | 현재 증거 |
|---|---|
| QR 모듈 격자에 클래스와 연관된 정보가 남는다 | 강하게 지지 |
| 원래 QR 공간 배치가 CNN의 분류에 도움이 된다 | 강하게 지지 |
| 클래스와 연관된 국소 패치 빈도 분포(class-associated local patch-frequency distribution)가 존재한다 | 지지 (`RESULTS.md` 7절) |
| marginal bit/byte composition으로 설명되지 않는 국소 공간 의존성이 존재한다 | 미검증 — within-QR 셔플 null 대기 |
| CNN 내부에서 URL 문자/n-gram 자체가 복원된다 | **기각** — 선형 프로브에서 유의 목표가 거의 없다 (`RESULTS.md` 5-B절) |
| 이 패턴이 다른 피싱 데이터에서도 일반화된다 | 미검증 |

**용어 한정.** 이 연구에서 "visual"은 raw-image의 시각 패턴이 아니라 **정규화된 QR 모듈 격자의 공간 패턴**을 가리킨다. 입력은 완벽히 정렬된 `(2, n, n)` 비트 격자이고 두 번째 채널로 데이터 모듈 마스크까지 준다. 촬영·인쇄·회전이 개입하는 실제 QR 이미지에 대한 주장은 이 실험 범위 밖이다.

### 14.2 1차 구현 항목

**A. Claim·통계 정리.** `text upper bound`를 `decoded-text reference`로 개명하고 "넘으면 누출" 부등식 검사를 없앴다. char n-gram LR은 강한 텍스트 베이스라인이지 Bayes 최적 분류기가 아니므로 부호 있는 갭(`gap_to_decoded_text`)만 보고한다. 시드별 CI 경계를 평균하던 `auroc_ci_pooled`를 폐기하고, 5시드 test 예측을 `(seed, row)`로 풀링해 그룹 클러스터 부트스트랩을 한 번 돌린 `auroc_pooled`/`auroc_pooled_ci`로 바꿨다. 클러스터 키는 시드와 무관한 eTLD+1 그룹이라 같은 그룹은 어느 시드에서 나왔든 함께 리샘플된다. 점추정이 CI 밖으로 나가도 CI를 인위적으로 넓히지 않는 순수 백분위 CI다. 조건 비교는 같은 split의 같은 test 행을 공유할 때만 쌍체 클러스터 부트스트랩을 쓰고, 그렇지 않으면 `paired=False`로 표시해 비쌍체 차이만 보고한다(H3의 `L-none` 대 `L-exact`가 여기 해당한다). H1 ~ H4는 층별로 검정한 뒤 Holm 보정을 걸어 `reports/hypotheses.json`에 쓴다.

**B. Bag-of-QR-patches (RQ3 직접 검정).** 데이터 모듈 위를 stride 1 sliding window로 훑어 2×2(16종)·3×3(512종) 이진 패치의 출현 빈도 히스토그램을 만들고, 그 히스토그램만으로 LR을 학습한다. 창 전체가 데이터 모듈인 경우만 센다. 기능 패턴은 URL과 무관한 상수라 포함시키면 히스토그램을 지배한다. 표현은 셋이다. `patch2`(16차원), `patch3`(512차원), `pyramid3`(전역 + 2×2 사분면 = 2,560차원). 마지막 것이 `patch3`보다 좋아지면 motif의 **위치**도 중요하다는 뜻이다. split은 주 조건의 것을 같은 시드로 재현하므로 CNN 실험과 정확히 같은 train/val/test를 쓰고, C는 val AUROC로 고른다. 라벨 셔플 대조는 test 라벨을 그대로 두고 train·val 라벨만 섞어 바닥선을 만든다. 학습이 없어 로컬 CPU에서 층당 수 분에 끝난다.

motif enrichment는 motif별 존재율 오즈비(Haldane-Anscombe +0.5 보정)로 잰다. **발견은 train에서만** 하고, 부호가 held-out test에서도 유지되는지를 `test_sign_match`로 따로 본다. 상위 목록에 오르려면 네 조건을 모두 만족해야 한다. 시드 5개의 부호 일치, 두 클래스 중 한쪽에서 train QR의 1% 이상 등장, 풀링 CI가 0 배제, 그리고 비포화(두 클래스 모두 존재율 > 0.99가 아님). 거의 안 나오는 motif는 오즈비가 커도 후속 절제에 쓸 표본이 없고, 포화 motif는 존재율 오즈비가 무의미하다. CI는 5시드 train을 `(seed, row)`로 풀링해 그룹 클러스터 부트스트랩을 한 번 돌려 만든다(motif 축이 512개라 비용이 커 부트스트랩 500회, `enrichment.json`의 `ci_n_boot`에 기록). 시드별 CI 경계를 평균하는 것은 A항과 같은 이유로 정당한 절차가 아니다.

**C. 인과 motif 절제 (RQ3) — motif 표적 중심 비트 개입.** B가 찾은 상위 phishing motif와 **매칭된 3×3 창의 중심 모듈 하나만 뒤집고** 원본 대비 ΔAUROC와 평균 로짓 변화를 잰다. 이 개입의 이름은 motif 표적 중심 비트 개입(motif-targeted center-bit intervention)이며, "motif 뒤집기(flipping)"라고 부르지 않는다. 창 전체를 바꾸는 것이 아니라 창당 모듈 하나를 바꾸는 것이기 때문이다. 대조는 둘이다. 상위 benign motif에 같은 개입, 그리고 **각 샘플에서 phishing motif가 맞은 개수와 똑같은 수**의 무작위 데이터 모듈 뒤집기(5회 반복 평균). 후자가 "뒤집은 모듈 수" 자체의 효과를 상쇄한다. phishing과 benign은 매칭 개수가 다르므로 무작위 대조도 둘로 나눈다. `random`은 phishing 매칭 수에, `random_benign_matched`는 benign 매칭 수에 맞춘다. 매칭은 원본 격자 한 벌에서 모두 계산한 뒤 한꺼번에 뒤집는다. 하나씩 순차로 뒤집으면 이웃 창의 id가 바뀌어 결과가 순서에 의존한다. 원본과 개입본이 같은 표본을 공유하므로 비교는 전부 쌍체 그룹 부트스트랩이다. 뒤집은 격자는 더 이상 유효한 QR이 아니며, 이 절제는 모델 입력에 대한 개입일 뿐 실제 QR 변형이 아니다.

현재 무작위 대조는 개입 개수만 맞추고 **위치·국소 밀도는 맞추지 않는다.** motif가 걸리는 중심들이 격자의 특정 영역에 몰려 있다면 차이의 일부가 motif 정체성이 아니라 위치에서 올 수 있다. 따라서 같은 사분면·비슷한 국소 흑색 밀도·같은 데이터 모듈 후보군에서 중심을 고르는 **위치·밀도 매칭 무작위 대조(`random_matched`)를 추가할 예정**이고, 인과 기여 표현은 그 대조가 붙은 뒤에 확정한다.

**D. 어휘 프로브 (RQ2).** SmallCNN의 GAP 직전 128차원 임베딩을 freeze하고 URL 어휘 속성을 선형으로 읽어낸다. 목표는 train URL 상위 char 3-gram 100개, 피싱 문헌의 키워드 16개(`login`·`verify`·`.com` 등), 구조 특성(path·query·숫자·하이픈·IP 호스트), 연속값(숫자 비율·특수문자 비율·점 개수·경로 깊이·서브도메인 수)이다. n-gram 목록은 **라벨을 보지 않고** train URL 빈도로만 고른다. 라벨을 보고 고르면 목표 선택 자체가 누출이다. 이진 목표는 AUROC, 연속 목표는 R²로 잰다.

기준선은 둘이다. **그룹 단위 셔플**(각 그룹에 다른 그룹을 도너로 배정해 목표값을 갈아끼운다. 행 단위로 섞으면 그룹 안에서 목표가 거의 상수인 경우 정보가 남아 기준선이 부풀려진다)과 **무작위 초기화 CNN**(체크포인트에 기록된 같은 arch, 학습만 안 한 모델). 유의 판정은 시드마다 따로 하고, 모든 시드에서 `CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한)`일 때만 유의로 센다. 두 기준선을 CI 수준에서 모두 넘어야 하며, 일부 시드에서 표본 부족으로 건너뛴 목표는 전 시드 일치를 확인할 수 없으므로 유의로 세지 않는다. n-gram 목표 유니버스는 층에서 한 번만 고른다. 시드마다 다시 고르면 "전 시드에서 유의"가 같은 대상을 두고 내린 판정이 아니게 된다. 라벨 프로브(`label:phishing`)는 임베딩이 애초에 라벨을 선형 분리하도록 학습됐으므로 참고 상한선일 뿐 어휘 증거가 아니며 요약 카운트에서 뺀다. 프로브가 유의해도 "문자를 복원했다"가 아니라 "그 속성이 임베딩에서 선형으로 접근 가능하다"는 뜻이다.

**Grad-CAM 보조화.** Grad-CAM을 RQ 증명의 주 분석에서 보조 시각화로 내린다. 표본은 첫 시드의 앞쪽 128개가 아니라 5시드 전부 × test 클래스별 균형 무작위 표본(기본 클래스당 64)이다. 타깃은 부호를 준다. phishing은 `+logit`, benign은 `−logit`이라야 benign CAM이 "benign의 근거"가 된다. 질량 자체가 아니라 **enrichment = CAM 질량 분수 / 영역 면적 분수**를 보고하고, 균일 난수 attribution 기준선(기대값 1)을 나란히 싣는다. 영역 크기로 정규화하지 않으면 char 영역에 60%가 몰린 것이 많이 본 것인지 덜 본 것인지 알 수 없다. 여기에 모델 파라미터 무작위화 sanity check(무작위화 모델의 CAM과 상관이 낮아야 한다)와 옵션으로 signed Integrated Gradients를 붙인다. 인과 주장은 Grad-CAM이 아니라 C의 절제 결과로만 한다.

**유의성 규칙 요약.** 주 가설 H1 ~ H4의 판정은 쌍체 ΔAUROC의 95% 양측 백분위 CI가 0을 배제하는지로 한다. p값은 그 CI를 역전시켜 정의하고(CI가 0을 배제하는 가장 작은 α) Holm 보정을 걸되, 이름은 `bootstrap-tail pseudo-p (Holm)`으로 쓴다. 영가설 아래에서 재표집한 정식 p값이 아니므로 주 결과는 ΔAUROC와 CI로 두고 pseudo-p는 보조로만 읽는다. 쌍체 가설에는 클러스터 순열 검정 p가 추가될 예정이다. motif enrichment는 5시드 부호 일치 + held-out test 부호 일치 + 풀링 CI 0 배제 + 최소 출현율. 프로브는 전 시드에서 두 기준선 CI 상한 초과. 절제는 쌍체 CI가 0을 제외하는 시드 수로 보고한다.

**H2의 쌍체화.** 참조 기준(char n-gram LR)의 test 예측을 `preds_test_charngram.npz`로 함께 저장해 H2를 CNN 대 참조의 쌍체 부트스트랩으로 검정한다. 예측이 저장되지 않은 옛 산출물에서는 참조 AUROC를 상수로 두는 옛 방식으로 되돌아가되 `descriptive_only=True`로 표시해 Holm family에서 제외한다. 참조 기준 자체의 표본 변동이 빠져 CI가 좁게 나오므로 검정으로 쓸 수 없다.

**산출물 경로와 조건 격리.** 2차 phase의 리포트는 `reports/{phase}/{condition_id}/{stratum}/`에 쓴다. 조건 id를 경로에 넣지 않으면 서로 다른 조건(raw 대 norm, mask-auto 등)의 산출물이 같은 층 폴더에서 덮어써진다. 체크포인트에는 `condition`·`qr` 스냅샷을 함께 저장하고, 로드할 때 현재 config와 어긋나면 하드 실패시킨다. 현재 config로 데이터셋을 재구성하면 다른 조건의 체크포인트를 조용히 잘못 읽는다.

### 14.3 2차 백로그

| 단계 | 해야 할 것 | 해결되는 질문 |
|---|---|---|
| **E. Counterfactual URL** | 길이를 유지하며 suspicious token만 삽입·제거한 쌍 URL을 만들고 CNN score Δ를 측정 | 어휘 변화 → QR → 예측의 인과 연결 |
| **F. 외부 검증** | WebPhish 학습 → 별도 PhishTank/OpenPhish phishing + Tranco/Common Crawl benign으로 평가 | source shortcut 제거 |
| **G. Campaign split** | eTLD+1뿐 아니라 URL token·path template의 MinHash 클러스터 단위 holdout | phishing kit/template 누출 제거 |
| **H. 실제 QR 조건** | auto-mask, 여러 EC, 회전·블러·원근·JPEG·인쇄 시뮬레이션 | 실제 "시각 QR"로 주장 확장 |

E ~ H는 이번 실행 범위 밖이다. RQ3의 후반부("새로운 도메인·데이터셋에서도 재현되는가")는 F·G가 붙기 전에는 답할 수 없다.
