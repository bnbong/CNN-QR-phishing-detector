# 결정 (2026-09-08)

아래 절부터가 원래 설계 스펙이고, 이 절은 그 위에 얹는 **실측 기반 확정 사항**이다.
설계 본문과 어긋나는 대목이 있으면 이 절이 우선한다. 근거는 같은 날 수행한 소스 실측 조사이며,
수집 실행 결과는 `reports/external/manifest.json`과 `reports/external/bias_diagnostics.json`에 있다.

## 소스 확정

| 축 | 채택 | 근거 |
|---|---|---|
| phishing 주(primary) | **OpenPhish `public_feed` 최근 90일 커밋 이력 누적** (고유 URL 43,990건) | 수집 시점과 시차가 작아 진짜 temporal/domain-shift 검증이 된다. 외부 test는 크기보다 독립성·신선도·출처 명확성이 중요하다 |
| phishing 보조(secondary robustness) | **Phishing.Database `phishing-links-ACTIVE.txt`** (GitHub raw, MIT) | 무키로 789,054줄, 경로 보유율 91%. 규모는 크지만 파일 자체의 최종 갱신이 2025-12-22인 **역사적 아카이브**이고 다수가 다른 피드의 재집계라 독립 표본이 아니다 |
| benign 주 | **Common Crawl `CC-MAIN-2026-34` columnar index parquet × Tranco `GQJ9K` 조인** | 경로 보유율 99%대. 인기도 축을 Tranco로 통제 |
| benign 민감도(필수) | **Tranco 조인을 끈 unranked/general Common Crawl 표본**(EXT-B3) | benign=인기 웹사이트라는 축이 남아 있는지 검사한다. 오염 위험이 올라가는 것은 한계로 명시 |
| benign 대조 | **Tranco 상위 5만 도메인**(경로 없음, EXT-B2) | "그럼 Tranco benign에서는?"에 같은 파이프라인으로 답하는 대조군 |
| 제외 | PhishTank, URLhaus | 각각 Cloudflare 403(앱 키 발급 중단), malware 전용이라 라벨 정의 불일치 |

**주·보조 배치의 근거.** 43,990건이면 외부 검증에 충분한 규모다. 주 결과(primary)는 OpenPhish 최근 90일
+ CC-MAIN-2026-34 benign으로 내고, Phishing.Database + 같은 benign은 **secondary robustness**로 함께 보고한다.
"789k라서 PhishDB가 주"라는 배치는 쓰지 않는다.

**Phishing.Database 신선도.** 리포 HEAD는 2026-08-23이지만 `phishing-links-ACTIVE.txt` 자체의 마지막
갱신 커밋은 **2025-12-22**다. "현재 살아있는 피싱"이 아니라 **역사적 피싱 URL 아카이브**로 취급해야
하며, 이 사실을 `manifest.json`의 `freshness_note`에 기록했다. 논문에서 "active phishing"이라는
표현을 쓰면 안 된다.

**OpenPhish 수집 방법.** `https://openphish.com/feed.txt`는 `openphish/public_feed`의 GitHub raw로
302 리다이렉트된다. 리포를 clone하지 않고 GitHub API로 최근 90일 `feed.txt` 커밋 목록을 받은 뒤
커밋별 raw 파일을 누적했다(커밋 목록만 API 쿼터를 쓰고 raw는 CDN이라 제한이 느슨하다).
`--openphish-max-commits`로 상한을 두며 실행값은 180커밋이었다. 각 URL의 `first_seen`은
**가장 이른** 커밋 날짜다 — 시간 분리 평가를 하려면 이 컬럼이 필요하다.

**Common Crawl 수집 방법.** S3 버킷 리스팅은 403이라 `cc-index-table.paths.gz`로 part 경로를 얻는다.
UA 헤더가 없으면 503이 온다. fsspec의 http 백엔드는 aiohttp를 요구하는데 이 환경에 없어서,
`external._HttpRangeFile`로 Range 요청 기반 seek 가능 스트림을 직접 만들어 pyarrow에 물렸다.

> **함정**: parquet 파티션은 호스트(SURT) 순으로 정렬되어 있다. 한 part의 row group 하나에는
> 도메인이 수백 개밖에 없어서, row group을 한 part에 몰아 읽으면 전송량만 늘고 도메인 다양성이
> 늘지 않는다. 실제로 처음에는 한 part에서 row group 2개를 읽어 688,363행을 얻고도 도메인 상한
> 적용 후 benign이 3,731건에 그쳤다. `max_row_groups_per_part=1`로 24개 part에 흩뿌린 뒤에야
> 62,520건이 나왔다.

## 열린 질문 8개에 대한 답

1. **PhishTank 앱 키** — 없다. 무키 접근은 Cloudflare 403이고 신규 발급도 닫혀 있어 제외했다.
   `SourceSpec.env_key` 자리는 남겨 뒀으니 나중에 키가 생기면 소스만 추가하면 된다.
2. **OpenPhish 이력 범위** — **90일**(`--openphish-days 90`). 180커밋에서 고유 URL 43,990건을 얻었다.
3. **Common Crawl 샘플 규모** — 크롤 **`CC-MAIN-2026-34`**, `subset=warc` part 24개에서 part당
   row group 1개씩. 원본 35,960,699행 → Tranco 조인 9,447,020행 → 도메인당 8건 상한 후 62,520건.
   benign 후보 10만 목표에는 못 미치지만 조인 후 4만 이상이라는 실질 요건은 넘겼다.
4. **`COLLECT_DATE`** — **2026-09-08**. 모든 소스에 같은 값을 쓰고 파일명에 박았다.
5. **오염률 수동 추정** — **하지 않는다.** benign 300건 수동 검토는 이번 범위 밖이므로
   `contamination_estimate.p_hat`은 null로 두고, 대신 (a) 피싱 피드 eTLD+1 제거,
   (b) 호스팅 블록리스트 제거, (c) VirusTotal/Safe Browsing 미검사 상태임을 한계 절에 명시한다.
   이 오염은 전이 성능을 **일반적으로 감쇠시키는 방향으로 예상되며, 단조성이 보장되지는 않는다**
   (라벨 잡음이 systematic하면 AUROC가 항상 낮아진다고 볼 수 없다). 그래도 예상 방향은
   "전이가 살아남았다"는 결론에 유리하고 "무너졌다"는 결론에 불리하다. 그 방향과 단서를 함께 본문에 적는다.
6. **Phishing.Database 라이선스** — GitHub API `spdx_id`와 README 모두 **MIT**로 확인했다.
   가공·재배포에 제약이 없다. 반면 OpenPhish는 재배포 금지로 읽는 것이 안전하므로
   `data/external/`을 통째로 gitignore하고 `reports/external/`의 메타만 커밋한다.
7. **F-c 포함 여부** — **F-c는 포함**, F-d는 예산이 남을 때만(선택). 우선순위는 F-a → F-b → F-c → F-d.
8. **MinHash 임계** — **0.7 고정**(설계 본문 5.3절의 0.8 대신). 체이닝 방어책
   (`max_template_frac` 초과 시 임계 상향 재실행)은 그대로 유지한다.

## 전처리 규약 (WebPhish와 동일해야 하는 부분)

- **스킴은 제거한다.** `head -3 data/webphish.csv` 결과 WebPhish `Data` 컬럼에는 스킴이 없다.
  따라서 `scheme_policy`는 `strip`으로 확정했고 `manifest.json`에
  `scheme_policy_resolved: "strip"`으로 기록했다. 스킴 유무는 URL 바이트 길이를 7 ~ 8바이트 바꿔
  QR 버전 배정과 길이 매칭을 직접 흔들기 때문에 이 항목이 F의 가장 흔한 실패 모드다.
- 스킴을 떼고 나면 `qrphish.urls.normalize_url(u, mode)`를 **그대로** 호출한다. 새 규칙은 없다.
  로더의 `mode` 인자로 `raw`/`norm`을 고른다.
- 중복 제거는 로더 옵션 `dedup="url" | "etld1"`. 주 결과는 `etld1`, `url`은 표본 확보용 보조다.
  WebPhish와의 겹침은 **양방향으로 보고**하되 삭제는 외부 쪽에서만 한다.
  같은 정규화 URL이 양 클래스에 나타나면 `load_webphish`와 같은 규약으로 **양쪽 모두 제거**한다.

## 편향 진단 게이트 (실행 결과 · 2026-09-09 정정)

> **정정.** 이 절의 옛 판정문("WebPhish와 편향 방향이 반대이므로 stress test로서 통과")은 **철회한다.**
> WebPhish benign을 맨 도메인 코퍼스로 잘못 전제한 데서 나온 서술이다. 단계별 통계를
> `reports/external/bias_stages.json`으로 다시 세어 보니 두 데이터셋의 편향 **부호가 전 층에서 같다.**

`path_depth>=1` 비율(benign / phishing):

| 단계 | WebPhish | 외부 primary |
|---|---|---|
| 원본 | 99.7% / 69.8% | 93.9% / 46.0% |
| 정규화 후 | 99.7% / 69.3% | 93.9% / 46.0% |
| v2 층 · 길이 매칭 후 | 99.75% / 44.3% | 86.8% / 33.7% |

**두 데이터셋 모두 benign 쪽 경로 보유율이 훨씬 높다.** v3·v4도 같은 방향이다. 따라서 외부 세트는
"편향 방향을 뒤집는" 세트가 아니라 **같은 방향의 편향을 공유하는** 세트다. 라벨 매핑
(`spam`→phishing, `ham`→benign)과 원본 컬럼 정의는 재확인했고 문제가 없다. 문제는 데이터가 아니라
그것을 설명한 문서였다.

이 사실이 F-a의 결론을 자동으로 무효화하지는 않는다. 공유 편향은 **차단이 아니라 플래그**로 다루며,
플래그가 켜진 세트에서는 6.1절의 재설계된 게이트를 따른다. 스킴 보유율은 양쪽 0.0으로 규약대로
제거됐고, EXT-B2(Tranco)의 경로 보유율 0.0은 설계상 대조군의 정의 그대로다.

## 남은 한계

- benign 60,562 대 phishing 555,853으로 클래스가 크게 기운다. 주 조건 `L-exact`의 길이 매칭이
  버킷마다 `min(n_ben, n_phi)`를 뽑아 1:1로 맞춰 주므로 실질 문제는 아니지만, `L-none` 조건에서는
  AUPRC와 base rate를 함께 보고해야 한다.
- Tranco 조인 때문에 benign이 **인기 도메인의 페이지**로 한정된다. 경로 편향은 깨지지 않는다 —
  외부 세트도 WebPhish와 같은 방향으로 benign 쪽 경로 보유율이 높다(결정 절 표). 여기에
  "benign = 인기 웹사이트"라는 축도 남는다. 그래서 조인을 끈 unranked/general Common Crawl
  benign(EXT-B3) 민감도 세트를 **필수 실험**으로 둔다(1.3절). 조인을 끄면(`tranco_csv=None`) 오염 위험이
  올라가는 것은 교환 조건이며 한계로 명시한다.
- 라벨 오염을 실측하지 않았다(위 5번).
- **benign에서 피싱 도메인을 제거한 것 자체가 선택 편향이다.** phishing으로 등장한 eTLD+1을
  benign에서 지우면(설계 1.4절 1) benign이 실제 웹보다 깨끗해지고, 그 결과 F-a의 전이 성능이
  낙관 쪽으로 치우칠 수 있다. 실제 웹에는 피싱 페이지를 올린 정상 도메인이 섞여 있는데
  평가 세트에는 없기 때문이다. 이 제거는
  `load_external(drop_benign_on_phish_domains=...)`(기본 True) / 수집 스크립트의
  `--keep-benign-on-phish-domains`로 끌 수 있고, 제거 건수는 항상
  `n_benign_on_phish_domains`·`n_dropped_phish_domain_from_benign`으로 manifest와 stats에
  기록된다. **이 민감도 분석은 권장이 아니라 필수다** — `clean benign` / `keep benign on phishing domains` /
  `hosting blocklist off` 세 조건을 반드시 비교한다(1.3절). 끈 상태의 F-a AUROC가 켠 상태보다 크게
  낮으면 전이 결과의 상당 부분이 이 전처리에서 온 것이다. 다만 실측 수집분에서 `hosting blocklist off`는
  primary benign 제거 건수가 0인 no-op으로 확인됐다(1.3절 3번).


---

# 2차 실험 F(외부 검증) · G(템플릿 단위 분할) 설계 스펙 v1

대상 독자: 구현 워커. 이 문서만 보고 `qrphish/external.py`, `qrphish/transfer.py`,
`qrphish/templates.py`, `scripts/collect_external.py`를 작성할 수 있어야 한다.

상위 문서: `docs/EXPERIMENT_DESIGN.md`(특히 1.4·1.5·5절·14.3절), `docs/RESULTS.md`, `docs/review_01.md`.
이 스펙은 14.3절 백로그의 **F(외부 검증)** 과 **G(Campaign/템플릿 split)** 을 실행 가능한 수준으로 확정한다.
E(counterfactual URL)와 H(실제 QR 조건)는 여전히 범위 밖이다.

작성 시점: 2026-09-08. 외부 소스 접근 조건은 이 날짜 기준이며, 확인하지 못한 항목은 **미확인**으로 표시했다.

---

## 0. 무엇에 답하는 실험인가

`docs/RESULTS.md` 0절의 증거표에서 마지막 줄만 아직 비어 있다.

| 주장 | 현재 증거 |
|---|---|
| 여러 피싱 QR에 반복되는 국소 motif가 존재한다 | 지지 (7절) |
| 그 motif가 CNN 판단에 인과적으로 기여한다 | 지지되지 않음 — 개정된 절제 설계에서 표적 개입과 무작위 개입의 ΔΔAUROC가 구분되지 않는다 (5-A절) |
| **이 패턴이 다른 피싱 데이터에서도 일반화된다** | 약하게 지지 — 길이+경로 매칭 cohort에서 0.649 / 0.639 / 0.677 (10절) |

그리고 `review_01.md` 핵심 문제 7번은 이유를 정확히 짚는다. WebPhish의 benign은 Alexa 상위
사이트에서, phishing은 완전히 다른 수집 과정에서 왔다. **라벨과 출처(source)가 결합**되어 있으므로,
1차 실험이 통제한 길이·버전·패딩·`www.`를 모두 제거하고 남은 신호가 "피싱성"인지 "수집 파이프라인의
지문"인지 원리적으로 구분할 수 없다. `raw → norm`만으로 CNN이 5 ~ 9%p 떨어졌다는 관측이 그 잔재의
크기를 보여준다.

F는 이 질문에 답한다. **WebPhish에서 학습한 모델이, 출처 결합 구조가 다른 데이터에서도 작동하는가.**
G는 보조 질문에 답한다. **eTLD+1 분할이 막지 못하는 피싱 키트 템플릿 누출이 1차 결과를 얼마나 부풀렸는가.**
G는 고정된 캠페인 test 위에서 Model A(eTLD+1만 격리)와 Model B(eTLD+1+템플릿 격리)를 쌍체 비교하는
구조다(5.5절).

두 실험 모두 **새 CNN 대규모 학습을 요구하지 않는다.** F의 (a)는 기존 체크포인트 재사용이고,
(b)(c)(d)와 G만 재학습이 필요하며 그 규모도 1차 매트릭스의 일부다.

---

## 1. 데이터 소스 후보와 선택

### 1.1 phishing 측 후보

| 소스 | 접근 방법 | 라이선스/약관 | 예상 표본 | URL 형태 | 시점 고정 | 판정 |
|---|---|---|---|---|---|---|
| **OpenPhish Community** | `https://openphish.com/feed.txt` (plain text, 6시간 주기 갱신, 스냅샷 약 500 ~ 2,000건). GitHub 미러 `openphish/public_feed`의 `feed.txt` 커밋 이력으로 **과거 스냅샷 소급 수집 가능** | openphish.com/terms.html 동의. 비상업 연구 사용은 통상 허용되나 **재배포 금지**로 읽는 것이 안전 | 스냅샷 1회 1 ~ 2k. GitHub 이력 90일치 누적 시 **30 ~ 80k(중복 제거 후 추정, 미확인)** | 전체 URL, **경로·쿼리 풍부** | GitHub 커밋 해시 + 날짜로 완전 고정 | **채택 (주 소스)** |
| **Phishing.Database** (`Phishing-Database/Phishing.Database`, 구 mitchellkrogza) | GitHub raw: `phishing-links-ACTIVE.txt`, `phishing-links-ACTIVE-today.txt`, `phishing-links-INACTIVE.txt`. 시간당 갱신, PyFunceble로 생존 검증 | 저장소 라이선스 확인 필요(**미확인** — 구현 시 LICENSE 파일 확인해 메타에 기록) | ACTIVE 수십만 건 | 전체 URL, 경로 포함. 단 **다수가 OpenPhish/PhishTank 재집계**라 독립 표본이 아님 | 커밋 해시로 고정 | **채택 (보조/증량용, 출처 중복 경고 기록)** |
| PhishTank | `http://data.phishtank.com/data/<appkey>/online-valid.json.bz2`, 시간당 갱신 | 등록 필요. **2020년 남용 이후 신규 사용자 등록이 닫혀 있고 2026년 6월 기준으로도 닫힌 상태**로 보고됨 | — | 전체 URL | — | **조건부 제외**. 기존 키가 있으면 `PHISHTANK_APP_KEY` 환경변수로 사용, 없으면 스킵 |
| URLhaus (abuse.ch) | CSV 덤프 공개 | 연구 사용 허용 | 대량 | 전체 URL | 가능 | **제외.** malware 배포 URL 위주라 라벨 정의가 "피싱"과 다르다. 라벨 오염 위험이 이득보다 크다 |
| APWG eCrime Exchange | 기관 회원 가입·NDA 필요 | 제한적 | — | — | — | **제외.** 학부·개인 연구 접근성 없음(미확인이나 통상 그렇다) |

### 1.2 benign 측 후보 — 여기가 설계의 핵심이다

**요구 조건**: benign에도 경로·쿼리를 가진 URL이 충분히 포함되어, WebPhish와 **다른 편향**을 가져야 한다.
Tranco 맨 도메인만 쓰면 benign이 전부 경로 없는 루트가 되어 WebPhish benign(경로 보유율 99%대)과도 성격이 달라지고,
그 위에서 전이 성능이 잘 나와도 그것은 **검증이 아니라 편향의 복제**다. 이 점을 놓치면 F 전체가 무의미해진다.

| 소스 | 접근 방법 | 라이선스 | 예상 표본 | URL 형태 | 시점 고정 | 판정 |
|---|---|---|---|---|---|---|
| **Common Crawl URL 인덱스** | (i) CDX API: `https://index.commoncrawl.org/CC-MAIN-2026-34-index?url=<도메인>/*&output=json` — 도메인 지정 질의용. (ii) **대량 무작위 표본은 인덱스 샤드 직접 다운로드**: `https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2026-34/indexes/cdx-00000.gz` (샤드 1개 수백 MB, 스트리밍 파싱으로 필요분만 채집). (iii) Parquet 컬럼 인덱스 `s3://commoncrawl/cc-index/table/cc-main/warc/` (Athena/DuckDB) | CC0 성격의 공개 데이터(약관: commoncrawl.org/terms-of-use) | 무제한. 목표 50k 채집에 샤드 1 ~ 2개면 충분 | **경로·쿼리 풍부. 인기도 순위와 무관한 일반 웹** | 크롤 id(`CC-MAIN-2026-34`) + 샤드 파일명 + 바이트 오프셋으로 완전 고정 | **채택 (주 소스)** |
| **Tranco** | `https://tranco-list.eu/top-1m.csv.zip`, 영구 id는 `https://tranco-list.eu/top-1m-id`로 조회 후 `https://tranco-list.eu/download/<listid>/full` 형태로 고정 인용. `tranco` PyPI 패키지 있음 | 연구 목적 공개 | 1M 도메인 | **맨 도메인만. 경로 없음** | 영구 list id로 완전 고정 | **보조 채택.** 단독 benign으로 쓰지 않는다. 용도는 셋 — (1) CC 샘플의 인기도 편향을 *진단*하는 축, (2) 민감도 분석용 대체 benign(`EXT-B2`), (3) CC 도메인 랭크 부착 |
| Wikipedia 외부 링크 | 덤프 `externallinks.sql.gz` 파싱 | CC BY-SA | 수백만 | 경로 포함, 다만 **위키 편집자가 고른 링크**라 학술·언론 편향 | 덤프 날짜로 고정 | **예비.** CC 접근이 막힐 때의 대안 |
| Majestic Million | `https://downloads.majestic.com/majestic_million.csv` | 무료, 재배포 조건 확인 필요(미확인) | 1M 도메인 | 맨 도메인만 | 날짜 고정만 가능(영구 id 없음) | **제외.** Tranco 대비 이점 없음 |

### 1.3 권장 조합 (확정)

```
EXT-P  (phishing, primary)   = OpenPhish public_feed GitHub 이력 90일 누적
EXT-P2 (phishing, secondary) = Phishing.Database phishing-links-ACTIVE (출처중복·아카이브 플래그)
                               [+ PhishTank online-valid  ← APP KEY 있을 때만]
EXT-B  (benign, 주)          = Common Crawl CC-MAIN-2026-34 × Tranco 조인
EXT-B3 (benign, 필수 민감도) = 같은 크롤에서 Tranco 조인을 끈 unranked 표본
EXT-B2 (benign, 대조)        = Tranco top-1m 상위 도메인 (경로 없음)
```

**필수 실험 목록(권장 아님).** 아래 셋은 결과와 무관하게 반드시 돌린다.

1. **primary vs secondary phishing 소스**: EXT-P(OpenPhish)를 주 결과로, EXT-P2(Phishing.Database)를
   robustness로 각각 F-a에 태운다.
2. **unranked CC benign 민감도**: EXT-B(Tranco 조인)와 EXT-B3(조인 없음)에서 F-a AUROC를 비교한다.
   EXT-B에서 높고 EXT-B3에서 급락하면, 모델이 잡는 것이 피싱성이 아니라 인기 도메인 유래 문자 통계일
   가능성이 커진다. 그 결과 자체가 논문에 쓸 수 있는 발견이다.
3. **benign 정제 절제 3조건**: `clean benign`(기본) / `keep benign on phishing domains`
   (`drop_benign_on_phish_domains=False`) / `hosting blocklist off` 를 같은 파이프라인으로 비교한다.
   셋의 외부 AUROC가 거의 같으면 결과가 강해지고, 크게 떨어지면 외부 일반화의 상당 부분을 전처리가
   만들어냈다는 뜻이다.

   **실측 단서(2026-09-08 수집분).** `no_hosting_blocklist` 조건은 primary benign에서 **한 건도 제거하지
   않는 no-op**이다. `external_primary_no_hosting_blocklist_*.csv`가 `external_primary_*.csv`와 92,408행
   전부 동일하다(benign 60,824 / phishing 31,584). 즉 이 절제는 primary에서 정보량이 0이며, F-a AUROC도
   정의상 기본 조건과 같게 나온다. 이 사실 자체를 결과로 보고하고, **블록리스트를 넓히는 일은 하지
   않는다** — 사전 등록 밖의 사후 선택이라 절제의 해석을 오염시킨다.

**왜 이 조합인가 (2026-09-09 정정).** 처음에는 "WebPhish의 편향축은 benign=인기 도메인 루트 /
phishing=긴 경로이고, EXT-B는 그 부호가 반대"라고 적었다. 이 전제가 틀렸다. 실측하면 WebPhish도
외부도 benign 쪽 경로 보유율이 더 높다(결정 절의 표). **두 세트는 같은 방향의 경로 편향을 공유한다.**
따라서 EXT-B가 통제하는 축은 경로 유무가 아니라 **인기도**다 — Common Crawl은 인기도와 무관한 일반
웹의 임의 페이지이고, EXT-B3(Tranco 조인 해제)가 그 축을 한 번 더 흔든다. 경로 축은 통제되지 않으므로
6.1절의 `path_lr` 기준선과 길이+경로 동시 매칭 cohort로 따로 다룬다. 여기서 WebPhish 학습 모델의
AUROC가 유지되면 신호는 인기도 지문이 아니고, 무너지면 1차 결론의 상당 부분이 출처 지문이었다는 뜻이다.
**두 결과 모두 논문에 쓸 수 있는 답이다.** EXT-B2는 "그럼 Tranco benign에서는?"이라는 리뷰어 질문에
같은 파이프라인으로 답하는 대조군이며, EXT-B2에서만 성능이 살아난다면 그것 자체가
"모델이 배운 것은 인기 도메인 루트 vs 그 외"라는 강한 증거다.

**대안(기록만 남김)**: benign=Tranco 도메인에 CDX API로 실제 크롤된 경로를 붙여 만드는 방식
(`url=<tranco도메인>/*`). 경로 편향은 해소되지만 인기도 편향은 그대로라 WebPhish와 절반 같은 편향이다.
`EXT-B3`로 이름만 예약하고 이번 범위에서는 만들지 않는다.

### 1.4 라벨 잡음 처리 (Common Crawl benign의 최대 약점)

Common Crawl에는 피싱 페이지도 들어 있다. 완전 제거는 불가능하므로 **상한을 걸고 보고**한다.

1. EXT-B에서 **모든 피싱 피드(OpenPhish 전체 이력 + Phishing.Database ACTIVE·INACTIVE + 가능하면 PhishTank)의
   eTLD+1 합집합에 속하는 도메인을 제거**한다. INACTIVE까지 쓰는 이유는 과거 피싱 호스트를 걸러내기 위함이다.
2. 무료 동적 호스팅/단축 도메인 블록리스트를 제거한다(최소: `000webhostapp.com`, `weebly.com`,
   `blogspot.com`, `duckdns.org`, `bit.ly`, `t.co`, `github.io`, `firebaseapp.com`, `web.app`,
   `netlify.app`, `pages.dev`, `r2.dev`, `glitch.me`). 목록은 `qrphish/external.py`에 상수로 두고 메타에 기록.
3. 잔여 오염률을 **추정해 보고**한다. EXT-B에서 무작위 300건을 뽑아 수동 검토하거나, 최소한
   VirusTotal/Google Safe Browsing 무검사 상태임을 명시한다. 추정 오염률 `p`는 AUROC 상한을 대략
   `1 - p` 규모로 깎으므로, `p`가 1%를 넘으면 결론 문장에 그대로 적는다.
4. **이 오염은 F의 결론을 대체로 보수적으로 만든다.** benign에 피싱이 섞이면 성능은 일반적으로 감쇠
   방향으로 예상되지만, 라벨 잡음이 systematic할 때 AUROC가 단조롭게 낮아진다고 보장할 수는 없다.
   예상 방향은 "전이 성능이 살아남았다"는 결론에 유리하고 "무너졌다"는 결론에 불리하다. 방향과
   이 단서를 함께 본문에 명시한다.

---

## 2. 수집 프로토콜

### 2.1 원칙

- **수집 시점을 하나로 고정한다.** `COLLECT_DATE`(YYYY-MM-DD, 실행일)를 한 번 정하고 모든 소스에 같은 값을 쓴다.
  OpenPhish는 GitHub 커밋 해시, Common Crawl은 크롤 id, Tranco는 영구 list id로 **재현 가능한 스냅샷**을 잡는다.
  "오늘 받은 feed.txt"만으로는 재현이 불가능하다.
- **원본 파일을 손대지 않고 저장한 뒤 해시를 기록한다.** `data/external/raw/{source}_{date}.{ext}` +
  `sha256`. 이후 모든 가공은 원본에서 재실행 가능해야 한다.
- **정규화는 WebPhish와 완전히 동일**해야 한다. `qrphish.urls.normalize_url(u, "norm")`을 그대로 호출한다
  (소문자화 + 선행 `www.` 1회 제거 + 말미 `/` 제거). 새 규칙을 추가하면 전이 비교가 깨진다.
  스킴 처리도 동일하다 — `normalize_url`은 스킴을 보존하므로, WebPhish 원본에 스킴이 없다면
  외부 URL도 **동일한 스킴 정책**으로 맞춰야 한다.

  > **구현 워커 필수 확인 사항**: WebPhish CSV의 `Data` 컬럼이 `http://`/`https://` 스킴을 포함하는지
  > 먼저 확인하라(`head -5 data/webphish.csv`). 외부 피드는 거의 항상 스킴을 포함한다. WebPhish가
  > 스킴을 포함하지 않는다면 외부 URL에서 **스킴을 제거**해야 하고, 포함한다면 스킴 분포(http vs https)가
  > 두 데이터셋에서 크게 다른지 진단으로 남긴다. **스킴 유무는 URL 바이트 길이를 7 ~ 8바이트 바꾸므로
  > QR 버전 배정과 길이 매칭을 직접 흔든다.** 이것이 F에서 가장 흔한 실패 모드다.
  > 결정: `external.py`에 `scheme_policy: "strip" | "keep" | "match_webphish"`를 두고 기본값
  > `match_webphish`(WebPhish 첫 1,000행에서 스킴 보유율 > 0.5면 keep, 아니면 strip)로 하되,
  > 실제 판정 결과를 메타 json에 `scheme_policy_resolved`로 기록한다.

### 2.2 중복 제거 (양방향, 필수)

`review_01.md` 7번에 답하려면 외부 데이터가 WebPhish와 겹치지 않아야 한다. 세 층으로 제거한다.

1. **정확 URL 중복**: 정규화 후 WebPhish에 존재하는 URL을 외부에서 제거. 제거 건수 보고.
2. **eTLD+1 중복**: WebPhish에 등장한 eTLD+1(양 클래스 모두)에 속하는 외부 URL을 제거.
   이것이 진짜 "새로운 도메인" 요구를 만족시키는 조건이다. **양방향**으로 보고하되 삭제는 외부 쪽에서만 한다
   (WebPhish는 1차 결과의 기준이라 건드리지 않는다).
   - 다만 이 규칙은 `000webhostapp.com` 같은 공유 호스팅 전체를 날려서 EXT-P를 크게 줄일 수 있다.
     그래서 **두 변형을 모두 만든다**: `dedup=url`(정확 URL만 제거)과 `dedup=etld1`(도메인까지 제거).
     **주 결과는 `dedup=etld1`**, `dedup=url`은 표본 수 확보용 보조. config로 전환한다.
3. **외부 내부 중복**: 소스 간 중복(OpenPhish ∩ Phishing.Database)은 URL 기준 제거하고,
   각 URL의 출처 목록을 `sources` 컬럼에 세미콜론으로 남긴다(독립성 진단용).

**라벨 충돌**: 같은 정규화 URL이 EXT-P와 EXT-B 양쪽에 나타나면 `load_webphish`와 동일하게 **양쪽 모두 제거**하고
건수를 보고한다(`qrphish/urls.py:load_webphish`의 `n_dropped_label_conflict` 규약과 동일).

### 2.3 저장 포맷

```
data/external/
  raw/
    openphish_2026-09-08.txt              # 원본 그대로
    phishdb_active_2026-09-08.txt
    phishtank_2026-09-08.json             # 키 있을 때만
    cc_cdx_2026-09-08.jsonl.gz            # 채집한 CDX 레코드 원본
    tranco_2026-09-08.csv
  openphish_2026-09-08.csv                # 정규화 후 (url,label,group,url_len,url_bytes_len,path_depth,source)
  phishdb_2026-09-08.csv
  commoncrawl_2026-09-08.csv
  tranco_2026-09-08.csv
  external_2026-09-08.csv                 # 병합·중복제거 완료된 최종 평가 세트
  external_2026-09-08.meta.json
```

`external_{date}.meta.json` 스키마:

```json
{
  "collect_date": "2026-09-08",
  "git_sha": "...",
  "qrphish_version": "...",
  "scheme_policy_resolved": "strip",
  "dedup_mode": "etld1",
  "sources": [
    {"name": "openphish", "label": 1,
     "url": "https://raw.githubusercontent.com/openphish/public_feed/<commit>/feed.txt",
     "pinned_id": "<commit sha>", "fetched_at": "2026-09-08T09:12:03Z",
     "sha256": "...", "n_raw": 1834, "n_after_norm": 1791, "license_note": "openphish.com/terms.html"}
  ],
  "dedup": {"n_dropped_exact_url_vs_webphish": 0, "n_dropped_etld1_vs_webphish": 0,
            "n_dropped_cross_source": 0, "n_dropped_label_conflict": 0,
            "n_dropped_phish_domain_from_benign": 0, "n_dropped_hosting_blocklist": 0},
  "counts": {"n_benign": 0, "n_phishing": 0, "by_version": {"v2": 0, "v3": 0, "v4": 0, "v5plus": 0}},
  "bias_diagnostics": {
    "path_depth_ge1_frac": {"benign": 0.0, "phishing": 0.0},
    "median_url_bytes": {"benign": 0, "phishing": 0},
    "www_prefix_frac_before_norm": {"benign": 0.0, "phishing": 0.0},
    "tranco_rank_coverage_frac": {"benign": 0.0, "phishing": 0.0}
  },
  "contamination_estimate": {"method": "manual_sample_300", "p_hat": null, "note": "미실시면 null"}
}
```

`bias_diagnostics`는 **F를 돌리기 전에 반드시 확인하는 게이트**다. `path_depth_ge1_frac`이
benign·phishing에서 비슷하거나(둘 다 높음) benign 쪽이 더 높으면 목적을 달성한 것이다.
WebPhish와 같은 방향(benign 낮음, phishing 높음)이면 **수집을 다시 설계해야 한다** —
그 상태로 돌린 전이 실험은 검증이 아니다.

### 2.4 gitignore 정책

기존 `.gitignore`는 `data/`와 `data/*.csv`를 무시한다. 유지하되 다음을 추가한다.

```gitignore
# 외부 수집 원본·중간 산출물은 커밋하지 않는다 (약관·용량)
data/external/
# 단, 재현에 필요한 메타는 커밋한다
!data/external/*.meta.json
```

메타 json만 커밋하면 URL 본문 재배포 없이 스냅샷 고정 정보(커밋 해시·크롤 id·sha256)를 공유할 수 있다.
`reports/transfer/`와 `reports/template_split/`은 기존 규칙대로 커밋한다.

---

## 3. 평가 프로토콜 (F)

네 가지 실행을 정의한다. 각각이 답하는 질문이 다르므로 **전부 필요하다**.

| id | 학습 | 평가 | 답하는 질문 | 재학습 |
|---|---|---|---|---|
| **F-a** | WebPhish (기존 체크포인트) | 외부 | 신호가 새 데이터에 그대로 전이되는가 (zero-shot) | 없음 |
| **F-b** | 외부 | 외부 (그룹 분할) | 외부 데이터 **자체에** 학습 가능한 QR 신호가 있는가 (전이 실패 시 "신호가 없다"와 "전이가 안 된다"를 구분) | 있음 |
| **F-c** | 외부 | WebPhish test | 역방향 전이. 비대칭이면 어느 쪽 신호가 더 일반적인지 알려준다 | 있음 |
| **F-d** | WebPhish + 외부 혼합 | 양쪽 각각 | 두 도메인 공통 신호가 존재하는가 | 있음 |

우선순위: **F-a → F-b → F-c → F-d**. F-a·F-b만으로 논문의 한계 절을 다시 쓸 수 있다.
F-c·F-d는 예산이 남을 때.

### 3.1 F-a: zero-shot transfer (핵심)

**체크포인트 재사용.** `artifacts/norm-exact-data_only-fixed-small_cnn/{stratum}/seed{k}/model.pt`
(`_load_trained` 경로 규약). 체크포인트에는 `state_dict`, `n`, `arch`, `condition`, `qr` 스냅샷이 들어 있고
`_ckpt_dataset_spec`이 현재 config와 불일치하면 하드 실패한다. **이 검사를 우회하지 마라.**
전이 실행은 반드시 `condition.features=data_only`, `qr.mask_mode=fixed`, `qr.ec=L`,
`qr.optimize=0`, `qr.mask_pattern=0`, `qr.border=0`로 돌린다.

**절차 (층 `s`, 시드 `k`마다)**

1. 외부 프레임을 `_prepare_frame`과 **같은 순서**로 만든다.
   `load_external → (path_filter) → natural_version → 층 필터 → 그룹 분할 → split별 길이 매칭`.
   단 F-a는 학습이 없으므로 분할이 불필요해 보이지만, **분할은 여전히 필요하다**(3.3 참조).
2. `build_stratum`으로 격자를 만든다. 층 디렉터리는
   `artifacts/external/{source_tag}/{cond_id}/{stratum}/seed{k}/`.
   `build_stratum`의 인자는 WebPhish 실행과 동일한 `cond`/`qr` 객체를 넘긴다.
3. **평가 cohort를 한 번 고정한다(F-a primary).** 외부 프레임을 시드 0의 절차로 한 번 만들어 그 행 집합을
   `fixed external cohort`로 얼리고, WebPhish 시드 0 ~ 4의 다섯 모델을 **모두 같은 행**에 평가한다. 그래야
   변동하는 것이 모델 시드뿐이고 test 분포는 완전히 동일하다. 시드마다 외부 매칭 표본까지 다시 뽑으면
   모델 변동과 평가 집합 변동이 섞여 해석이 어려워진다.
   `_load_trained`로 모델을 불러와 `predict_probs`로 이 cohort 전체에 점수를 낸다. F-a에서는 외부 데이터를
   학습에 전혀 쓰지 않으므로 층 전체 행이 평가 대상이다(`eval_rows: "all"`).
   **F-a vs F-b 쌍체 비교는 secondary(F-b)** 로 분리한다. 이때만 시드별 F-b test cohort를 쓰고
   같은 행에서 쌍체 부트스트랩을 건다. 즉 `F-a primary = fixed external cohort`,
   `F-a vs F-b paired secondary = 각 F-b seed test cohort`다.
4. 임계값은 **WebPhish val에서 고른 값을 그대로** 쓴다(`results.json`의 `threshold` 또는 체크포인트 옆
   기록). 외부에서 임계값을 다시 고르면 그것은 zero-shot이 아니다. F1·Acc는 이 임계값에서만 보고하고,
   **주 지표는 임계값 무관한 AUROC**로 둔다.
5. CI는 외부 eTLD+1 그룹 클러스터 부트스트랩. 5시드 예측을 `(seed, row)`로 풀링하고
   `evaluate.cluster_bootstrap_by_seed`를 쓴다(시드 간 점수 척도 차이 문제는 `RESULTS.md` 서두 단서 참조 —
   **반드시 `_by_seed` 계열 함수를 쓸 것**, 단순 풀링은 이미 알려진 왜곡을 재현한다).
6. 바닥선: **라벨 셔플 대신 점수 순열 검정**을 쓴다. 모델이 고정이므로 라벨을 재학습 없이 섞을 수 있다.
   외부 라벨을 **그룹 단위로** 순열해 얻은 AUROC 분포의 97.5 백분위를 `null_ci_upper`로 기록한다.
   행 단위 순열은 그룹 내 라벨 상관을 깨서 바닥선을 낮게 잡으므로 주 바닥선으로 쓰지 않는다.

   **순열 횟수.** 설계 목표 2,000회를 재실행에서 채웠다(결과 JSON의 `n_perm` = 2000).

   **귀무 모형의 정체 (2026-09-09).** 이 순열은 크기가 다른 그룹 사이에서 라벨 블록을 복사·교환하므로
   **양성 라벨 총수가 보존되지 않는다.** 따라서 정식 순열 검정이 아니라 **그룹 블록 교환 근사**로
   부르고, 무엇을 교환 가능하다고 가정했는지를 본문에 적는다. 대조를 위해 행 단위 순열 바닥선도
   **함께** 보고한다(그룹 상관을 무시한 하한선 역할).

### 3.2 F-b / F-c / F-d

- **F-b**: 외부 데이터를 `run_matrix`와 같은 경로로 학습한다. 조건은 주 조건과 동일
  (`norm-exact-data_only-fixed-small_cnn`), 시드 5개, eTLD+1 그룹 분할 70/15/15, 5% 상한 다운샘플.
  `condition_id`는 그대로 두고 출력 루트만 `artifacts/external/{source_tag}/`로 바꾼다.
- **F-c**: F-b에서 학습한 체크포인트로 **WebPhish의 각 시드 test 행**을 평가. WebPhish 층 아티팩트는
  이미 있으므로 재생성 불필요. 쌍체 비교가 가능하다(같은 test 행) — `paired_cluster_bootstrap_by_seed`로
  "WebPhish 자체 학습 대 외부 학습"의 ΔAUROC를 낸다.
- **F-d**: 학습 프레임 = WebPhish train ∪ 외부 train. **그룹 분할을 두 데이터셋 합집합 위에서 한 번에**
  수행해야 한다(각각 나눠 붙이면 같은 eTLD+1이 한쪽 train, 다른 쪽 test에 갈라질 수 있다).
  평가는 WebPhish test와 외부 test를 **따로** 보고한다. 합쳐서 하나의 AUROC를 내면
  두 도메인의 난이도 차이가 섞여 해석 불가능해진다. `dataset_origin` 컬럼(0=webphish, 1=external)을
  meta에 넣고, 그 컬럼만으로 LR을 돌린 `origin_lr` 베이스라인을 **필수로** 보고한다 —
  이 값이 높으면 혼합 모델이 "어느 데이터셋에서 왔는지"를 배우고 있다는 직접 증거다.

### 3.3 층화·길이 매칭을 외부에도 적용하는가 — **적용한다**

**논거.** F-a에서 외부 평가 세트에 길이 매칭을 걸지 않으면, 외부에서도 "benign은 짧다"(또는 반대)가
성립할 때 AUROC가 길이만으로 올라간다. 그러면 전이 성공/실패 판정이 **신호 전이가 아니라 길이 분포의
우연한 일치**를 재는 것이 된다. 1차 실험의 주 조건이 `L-exact`인 이상 외부 평가도 `L-exact`여야
같은 척도의 비교다. 층화(버전 고정)도 같은 이유로 필수다 — 격자 크기가 층 안에서 고정되어야
모델 입력 형상이 맞고, 버전이 라벨 프록시가 되지 않는다.

**보고는 세 조건 모두.** `L-exact`(주), `L-quantile`, `L-none`을 나란히 낸다.
외부에서 `L-none`과 `L-exact`의 격차가 WebPhish보다 크면 "외부 데이터의 길이 편향이 더 세다"는
사실 자체가 보고할 만한 진단이다.

**길이+경로 동시 매칭 cohort `a_fixed_lenpath` (2026-09-09 추가).** 두 데이터셋이 같은 방향의 경로
편향을 공유하므로(결정 절), 길이만 맞춘 cohort에서는 전이 AUROC에 경로 모양의 기여가 그대로 남는다.
그래서 1바이트 길이 버킷과 **경로 유무**를 동시에 맞춘 cohort를 추가로 만든다. 6.1절의 경로 편향
플래그가 켜진 세트에서는 이 cohort가 F-a의 **주 판정**이고 일반 cohort는 보조다. 표본이 줄어드는
대가는 층별 n을 병기해 드러낸다.

**적용 순서는 WebPhish와 동일하다**: 층 필터 → `group_split` → `match_by_length_within_splits` →
`assert_length_matched`. F-a에서 학습이 없어도 이 순서를 지켜야 F-b와 정확히 같은 행 집합이 되어
F-a 대 F-b의 쌍체 비교가 성립한다. **이것이 F-a에서도 분할을 도는 이유다.**

### 3.4 외부 데이터의 버전 분포가 다를 때 (층 소멸)

외부 피싱 URL은 WebPhish보다 길 가능성이 높고, Common Crawl benign도 경로가 있어 길다.
결과적으로 **v2가 비고 v4·v5+가 두꺼워질 수 있다.** 대응:

1. **층 채택 규칙은 1차와 동일하게 사전 등록된 값을 쓴다**(`stratum_rules`: primary ≥1000/class,
   secondary ≥300/class, <300 폐기). 규칙을 외부 데이터에 맞춰 완화하지 않는다.
2. **전이 결론은 두 데이터셋에서 모두 살아남은 층에서만 낸다.** WebPhish에서 primary이고 외부에서
   최소 secondary인 층의 교집합을 `comparable_strata`로 계산해 메타에 기록하고, 그 층들에 대해서만
   H-F 가설을 검정한다. 나머지 층은 표에 "외부 표본 부족(n=…)"으로 남긴다.
3. 교집합이 **공집합이면 F-a는 실행 불가**다. 이 경우의 대체 경로(사전 등록):
   `v5plus`를 층으로 살리는 대신, **길이 구간 재층화는 하지 않는다**(격자 크기가 달라져 같은 CNN을
   못 쓴다). 대신 WebPhish 쪽에서 `L-none` 조건의 v4 체크포인트를 쓰고 외부도 `L-none` v4로 맞춘 뒤
   `descriptive_only=True`로 표시해 가설 검정에서 제외한다. **이 폴백을 쓰면 반드시 그 사실을 표에 적는다.**
4. `v5plus`는 층 안에 여러 버전이 섞이는 특수 층이다. `QRGridDataset.canonical_data_mask`가
   버전 합집합 마스크를 쓰므로 동작은 하지만, **WebPhish 체크포인트의 `n`과 외부 층의 `n`이 다르면
   모델을 못 쓴다.** 전이 전에 `ds.n == ckpt["n"]`을 하드 검사하고 불일치면 명확한 에러를 낸다.

### 3.5 클래스 불균형

외부 수집은 1:1이 되지 않는다(benign은 무한, phishing은 유한). 정책:

- **AUROC를 주 지표로 유지**한다. 불균형에 강건하다.
- 길이 매칭(`match_by_length`)이 버킷마다 `min(n_ben, n_phi)`를 뽑으므로 `L-exact`에서는 **자동으로 1:1이 된다.**
  따라서 주 조건에서는 불균형이 사실상 해소된다. 이 사실을 메타에 확인 기록한다.
- `L-none` 조건에서는 불균형이 남으므로 **AUPRC를 함께** 보고하고, 양성 비율(`base_rate`)을 표에 병기한다.
  AUPRC는 base rate에 의존하므로 base rate 없이 층 간·데이터셋 간 비교를 하면 안 된다.
- 수집 단계에서 benign을 무제한 모으지 말고 **phishing 목표 수의 3배**로 캡을 건다(계산량·`L-exact` 후
  낭비 방지). 목표: EXT-P 20k, EXT-B 60k(중복 제거 전).

### 3.6 그룹 분할은 eTLD+1 유지

외부에서도 그룹 키는 `qrphish.urls.etld1`(번들 PSL 스냅샷 사용, 네트워크 접근 없음)이다.
`max_group_frac=0.05` 상한도 유지한다. 외부 피싱은 공유 호스팅 집중이 WebPhish보다 심할 수 있으므로
`split_diagnostics`(교집합 0, test 상위 5 그룹 점유율)를 반드시 낸다. 상위 5 그룹 점유율이
40%를 넘으면 그 층의 결과는 `descriptive_only`로 강등한다(사전 등록 규칙).

### 3.7 베이스라인도 동일하게 전이 평가 — **필수**

`review_01.md`가 지적한 핵심 구분이다. **CNN만 무너지는가, 텍스트 기준선도 함께 무너지는가.**
`qrphish/baselines.py`의 다음 넷을 F-a·F-b에서 동일하게 돌린다.

| 베이스라인 | 함수 | 전이 시 의미 |
|---|---|---|
| char n-gram TF-IDF LR (디코딩 텍스트 참조 기준) | `charngram_lr` | 텍스트 수준 신호가 전이되는가. **이것도 무너지면 문제는 QR/CNN이 아니라 데이터셋 간 어휘 분포 차이다** |
| byte-hist LR | `bytehist_lr` | 비공간 바이트 분포 신호의 전이 |
| length LR | `length_lr` | `L-exact`에서 0.5여야 한다. 아니면 매칭이 깨진 것 (게이트) |
| version LR | `version_lr` | 층 안에서 0.5여야 한다 (게이트) |
| **motif 히스토그램 LR** | `motifs.bag_of_patches_lr` (`patch3`, `pyramid3`) | RQ3 직접 검정의 전이판 |
| **`path_lr`** (2026-09-09 추가) | 경로 유무·경로 깊이·구분자 개수 세 특징만 | 경로 모양만으로 라벨이 갈리는가. 6.1절 게이트의 판정 입력 |

**표본 수를 표에 병기한다.** 재실행에서 기준선은 CNN과 같은 평가 행에서 계산했고, `fit n`(기준선을
학습시킨 WebPhish train 행 수)과 `평가 n`을 표에 나란히 적는다. motif 히스토그램 기준선만 fit 표본이
4,000행으로 잘려 있어 그 열은 별도로 읽는다.

**기준선도 5시드 쌍체로 낸다.** 재실행에서 시드별 학습 자료와 test를 맞춘 뒤 5시드 쌍체 Δ와 그 CI로
교체했다. 주 cohort의 Δ(CNN − char n-gram LR)는 −0.065 / −0.195 / −0.119로 세 층 모두 CI가 0을 배제하고,
byte-hist LR과의 Δ는 세 층 모두 CI가 0을 포함한다.

**중요**: 이 베이스라인들은 `fit`이 필요하므로 F-a에서는 "WebPhish train에서 fit한 벡터라이저·계수를
외부에 그대로 적용"해야 한다. 현재 `charngram_lr(urls, y, split, ...)` 시그니처는 한 프레임 안에서
fit·predict를 모두 한다. **따라서 `baselines.py`에 fit/apply 분리 경로를 추가해야 한다**(7.2 참조).
분리 없이 외부에서 다시 fit하면 그것은 F-a가 아니라 F-b다.

**해석 매트릭스(사전 등록)**

| CNN 전이 | 텍스트 기준선 전이 | 해석 |
|---|---|---|
| 유지 | 유지 | 신호는 실제 피싱성. RQ3 후반 **긍정** |
| 붕괴 | 유지 | QR 격자 표현이 데이터셋 특이적. CNN이 배운 것은 출처 지문 쪽에 가깝다 |
| 붕괴 | 붕괴 | 두 데이터셋의 어휘 분포 자체가 다르다. QR/CNN의 문제가 아니라 **도메인 시프트**. F-b에서 외부 자체 신호가 있으면 이 해석이 확정된다 |
| 유지 | 붕괴 | 예상 밖. 구현 오류를 먼저 의심하고 라벨 누출을 재점검 |

---

## 4. motif 재현성

`docs/RESULTS.md` 7절과 5-A절이 WebPhish train에서 찾은 상위 motif를 외부에서 다시 본다.
`qrphish/motifs.py`의 `motif_enrichment`, `combine_seed_enrichments`를 그대로 쓴다.

**측정 항목 (모두 층별·시드별로 낸 뒤 집계)**

1. **부호 일치율.** WebPhish train 상위 K개 motif(K=20, 2×2와 3×3 각각)의 log OR 부호가
   외부 데이터에서 같은 부호인지. 지표는 `sign_match_rate = (일치 개수)/K`.
   귀무 기대값은 0.5이므로 이항검정 또는 그룹 부트스트랩 CI를 붙인다.
   **motif 선택은 WebPhish train에서만** 한다(외부를 보고 고르면 순환 논증).
2. **독립 발견 겹침.** 외부 데이터에서 동일 절차로 독립 발견한 상위 K motif 집합과의 **Jaccard 유사도**.
   귀무 분포는 512개(3×3) 중 K개를 무작위로 두 번 뽑았을 때의 Jaccard로, 해석적으로도 계산 가능하고
   1,000회 시뮬레이션으로 CI를 잡아도 된다. K=20, 512종이면 기대 Jaccard ≈ 0.02 수준이라
   실측 0.3 이상이면 강한 재현이다.
3. **효과 크기 상관.** WebPhish log OR 벡터와 외부 log OR 벡터의 **Spearman 상관**(전체 512종).
   부호 일치율보다 정보량이 많고 상위 K 선택에 의존하지 않아 **주 지표로 삼는다.**
   포화 motif(양 클래스 존재율 > 0.99)와 희소 motif(어느 클래스에서도 출현율 < 1%)는 1차와 동일 기준으로 제외.
4. **외부에서의 인과 절제 반복 — 실행하지 않는다.** 내부(5-A절)에서 표적 개입과 무작위 개입이 CI
   수준에서 구분되지 않으므로, 외부에서 같은 절제를 돌려도 답할 질문이 남지 않는다. 아래는 실행하지
   않기로 한 설계의 기록이다.

   `qrphish/occlusion.py`의 절차를 외부 층에 그대로 적용하는 안이었다.
   - 대상 모델: **F-a의 WebPhish 체크포인트**(이 조합이 가장 의미 있다 — WebPhish에서 배운 모델이
     외부 QR에서도 같은 motif에 의존하는가). 여력이 되면 F-b 모델로도 반복.
   - 개입 대상 motif: WebPhish train 상위 phishing motif(1차와 동일 목록). 개입은 1차와 같은
     **motif 표적 중심 비트 개입(motif-targeted center-bit intervention)** — 매칭된 3×3 창의 중심 모듈
     하나만 뒤집는다.
   - 대조: 1차와 동일하게 상위 benign motif에 같은 개입 + **매칭 개수 동일 무작위 개입**
     (`random`, `random_benign_matched`), 5회 반복 평균. 1차에 추가될 위치·밀도 매칭 대조
     (`random_matched`)가 준비되면 외부에서도 같이 돌린다.
   - 지표: ΔAUROC와 평균 로짓 변화, 쌍체 그룹 부트스트랩 CI.
   - 판정: WebPhish에서 관측된 ΔAUROC(0.016 ~ 0.050)의 **부호가 같고**, 무작위 대조보다 크면 재현.

**보고 형식**: `reports/transfer/{source}/motif_replication/{stratum}/replication.json`
```json
{"K": 20, "size": 3,
 "spearman_logodds": {"rho": 0.0, "ci": [0.0, 0.0], "n_motifs_used": 0},
 "sign_match_rate": {"value": 0.0, "ci": [0.0, 0.0], "null": 0.5},
 "jaccard_topk": {"value": 0.0, "null_mean": 0.0, "null_ci": [0.0, 0.0]},
 "ablation": {"delta_auroc": 0.0, "ci": [0.0, 0.0],
              "random_control_delta": 0.0, "ratio_to_random": 0.0}}
```

---

## 5. 템플릿 단위 분할 (G)

### 5.1 목적

eTLD+1 분할은 같은 도메인이 train/test에 갈리는 것만 막는다. 피싱 키트는 **여러 도메인에 같은 경로
구조를 복제**하므로(`/wp-admin/secure/login.php?id=<random>` 같은 골격), eTLD+1이 달라도 test에
train과 사실상 같은 URL이 남는다. G는 그 누출의 크기를 잰다.

**이것은 외부 데이터 없이 WebPhish만으로 실행 가능하며, 계산량도 작다.** F보다 먼저 해도 된다.

### 5.2 템플릿 키 생성

URL 하나에서 **경로 골격 문자열**을 만든다.

```
입력:  http://secure-login.example.com/wp-admin/user/verify2.php?id=8a3f1c&ref=44
1) 정규화 URL에서 host, path, query를 분리 (urlsplit; 스킴 없으면 http:// 부착)
2) path 세그먼트별 토큰화:
   - 순수 숫자 세그먼트            -> "#"
   - [0-9a-f]{8,} 16진 문자열      -> "H"
   - base64/난수스러운 문자열
     (길이>=12 이고 (숫자비율>=0.3 또는 대소문자혼합)) -> "R"
   - 그 외                          -> 소문자 원문. 단 세그먼트 내부의 숫자런은 "#"로,
                                      16진런(>=8)은 "H"로 치환
3) 확장자는 보존한다 (.php/.html은 키트 식별에 유용)
4) query는 **키 집합만** 정렬해 사용, 값은 버린다: "?id&ref"
5) 골격 = "/".join(치환된 세그먼트) + "?" + ",".join(sorted(query keys))

결과: "/wp-admin/user/verify#.php?id,ref"
```

**호스트는 골격에 넣지 않는다.** 도메인은 이미 eTLD+1 그룹이 담당하고, 여기서 잡으려는 것은
도메인을 넘어 반복되는 경로 구조다. 다만 **서브도메인 라벨 패턴**(`secure-login`)은 키트 신호이므로
보조 필드로 남겨 진단에만 쓰고 클러스터링에는 넣지 않는다(1차 구현 범위 밖).

**경로가 없는 URL**(맨 도메인, WebPhish benign의 다수)은 골격이 `""`가 되어 전부 한 덩어리로 뭉친다.
이를 방지하기 위해 **`path_depth == 0`인 행은 템플릿 클러스터링에서 제외하고 각자 단독 클러스터**로 둔다.
이 처리를 하지 않으면 benign 절반이 하나의 그룹이 되어 `group_split`이 5% 상한에 걸려 폭발한다.
**이것이 G 구현에서 가장 흔한 실패 모드다.**

#### 5.2.1 과병합 게이트 `min_template_tokens` (2026-09-08 추가)

빈 골격을 막아도 두 번째 실패 모드가 남는다. `/index.html`, `/index.htm`, `/about` 같은
**범용 웹 골격**은 캠페인이 아니라 웹 전반의 기본 파일명인데, 정확 일치 지름길이 이들을 통째로
한 클러스터로 묶는다. v2 실측에서 `/index.html` 하나가 무관한 도메인 276건을 한 덩어리로 만들었다.
반대로 같은 층의 최대 클러스터(2,913건, 15.6%)는 `tools.ietf.org/html/rfc###` → `/html/rfc#`의
정확 일치라서 정당하다. 토큰 수만으로는 이 둘을 가를 수 없다(둘 다 토큰 2개다).

그래서 골격은 다음 중 **하나라도** 만족할 때만 클러스터링 대상이다.

1. 토큰이 `min_template_tokens`개 이상 (기본 **3**)
2. 치환 자리표시자 `#`/`H`/`R`를 포함 — 숫자·해시·난수 자리는 키트 골격의 지문이다
3. 쿼리 키를 포함 (`?id&ref`)

통과하지 못한 골격은 `path_depth == 0`과 똑같이 **행마다 단독 클러스터**(`solo:{row}`)로 둔다.
구현은 `qrphish.templates.is_clusterable_template`이고, config는 `split.min_template_tokens`다.
`min_template_tokens=1`로 두면 이 게이트가 사실상 꺼진다(자리표시자·쿼리 조건은 남는다).

실측 효과 (`reports/template_split/`, 주 조건 5시드):

| 층 | 템플릿 클러스터 수 (게이트 전 → 후) | 최대 클러스터 점유율 | 병합 후 그룹 수 | 1차 test 행 중 이탈 비율 |
|---|---|---|---|---|
| v2 | 13,417 → **15,225** | 0.1556 (변화 없음, `/html/rfc#`) | 9,558 → 9,248 | 0.859 |
| v3 | (게이트 전 미측정) → **10,544** | 0.018 | 6,198 → 5,733 | 0.884 |
| v4 | (게이트 전 미측정) → **3,191** | 0.015 | 2,035 → 1,905 | 0.882 |

v2에서 2위 클러스터가 276건(`/index.html`)에서 70건으로 내려갔고, 정당한 최대 클러스터는 그대로다.

### 5.3 MinHash + 연결 요소 클러스터링

> **사전 등록 파라미터 변경 (2026-09-08 결정).** 최초 초안은 문자 4-gram / `num_perm=128` /
> `threshold=0.8`이었다. 구현하면서 세 값을 모두 바꿨고, 아래 본문은 **실제 구현 값**이다.
> 사유: (1) 골격 문자열은 이미 `#`/`H`/`R`로 마스킹돼 있어 문자 4-gram이 마스크 문자 위를
> 미끄러지며 서로 다른 캠페인을 같은 shingle로 묶었다. 토큰 단위(구분자 `/ ? & .`)
> 3-shingle이 "경로 세그먼트 구성이 같은가"라는 원래 의도에 맞는다. (2) `datasketch` 의존성을
> 추가하지 않고 numpy로 직접 구현했고(`minhash_signatures`), 토큰 shingle은 집합 크기가
> 문자 4-gram보다 훨씬 작아 `num_perm=64`로도 추정 분산이 충분했다 — 128은 60만 행에서
> 비용만 두 배였다. (3) shingle 단위가 커지면 같은 캠페인 안에서도 Jaccard가 낮게 나와
> `0.8`은 과하게 보수적이었고, `0.7`로 내렸다. 대신 `min_template_tokens=3` 게이트로
> `/index.html` 같은 범용 골격을 클러스터링에서 아예 빼서 체이닝을 막는다.

- **shingle**: 골격 문자열의 **토큰 3-shingle**(`shingle_size=3`, 토큰 구분자 `/ ? & .`).
  토큰 수가 3보다 적으면 전체를 단일 shingle로 쓴다(짧은 골격이 사라지지 않게).
- **MinHash**: `num_perm=64`. 외부 의존성 없이 numpy로 구현한다(32bit 해시를 소수
  `2**31 - 1` 위에서 아핀 변환 — `qrphish.templates.minhash_signatures`).
  **시드 고정 필수**: 해시 계열을 config seed에 묶어 재현성을 보장한다.
- **LSH 임계값**: `threshold=0.7` (Jaccard). 밴드 LSH로 후보쌍을 뽑고 추정 Jaccard가
  임계값 이상인 쌍만 union한다.
- **연결 요소**: union-find로 간선을 합쳐 `template_id`를 부여한다.
  > 주의: 연결 요소는 전이적이라 **체이닝**으로 거대 클러스터가 생길 수 있다. 방어책 둘을
  > 구현하고 진단으로 고른다. (1) 최대 클러스터 크기가 전체의 `max_template_frac=0.05`를 넘으면
  > 경고를 내고 `threshold`를 0.85, 0.9로 올려 재실행(자동 3회까지). (2) 최종 클러스터 크기 분포를
  > `template_diagnostics.json`에 기록. 이 진단을 보고 임계값을 확정한 뒤 config에 박는다.
- **정확 일치 지름길**: 골격 문자열이 **완전히 같은** URL은 MinHash 없이 먼저 union한다(대부분의 신호가
  여기서 잡힌다). MinHash는 근사 변형을 잡는 보조 단계다. 이 최적화로 실행 시간이 크게 준다.

### 5.4 eTLD+1과의 합집합

최종 그룹 키는 **union-find로 두 관계를 모두 합친 연결 요소**다.

```
uf = UnionFind(n_rows)
for rows sharing same etld1:      uf.union(...)
for rows sharing same template_id: uf.union(...)
group_campaign = uf.find(row)
```

즉 "같은 도메인" 또는 "같은 템플릿"이면 같은 그룹. 이 키를 `df["group"]`에 넣고 기존
`group_split`을 **그대로** 호출한다. `splits.py`는 수정하지 않는다 — 그룹 키만 갈아끼운다.
`max_group_frac=0.05` 다운샘플도 그대로 작동한다.

> 합집합은 반드시 필요하다. 템플릿만으로 분할하면 같은 도메인이 train/test에 갈려
> 1차보다 **약한** 통제가 된다. 두 제약을 동시에 만족해야 한다.

### 5.5 실행과 판정 — 고정 캠페인 test 위의 쌍체 비교

**설계를 바꿨다.** 분할 방식을 통째로 갈아끼운 뒤 성능을 비교하면 학습 조건뿐 아니라 평가 대상까지
함께 바뀐다. 실측 진단이 그 규모를 보여준다. v3에서 eTLD+1 그룹 6,198개가 템플릿 union 후 5,733개로
줄고, 1차 test 행의 **86 ~ 88%가 template split에서는 test를 떠난다**(v3 기준 88.4% = `len(te - tt) / len(te)`).
1차 test와 새 test의 Jaccard도 평균 0.06 ~ 0.10에 그친다. 이 상태에서 두 AUROC를 빼면 그 차이가 템플릿
누출 때문인지 평가 집합이 바뀐 탓인지 분리할 수 없다.

**새 설계**: campaign/template-disjoint **test set을 먼저 하나 고정**하고, 그 동일한 test 위에서 학습 조건만
다른 두 모델을 쌍체 비교한다.

```
고정 test  = template ∪ eTLD+1 연결 요소 단위로 잘라낸 held-out cohort (한 번 만들어 얼린다)
Model A    = eTLD+1만 격리해 학습 (test와 같은 eTLD+1은 없지만 유사 template은 train에 허용)
Model B    = eTLD+1 + template까지 격리해 학습
핵심 수치  = ΔAUROC_G = AUROC(A) − AUROC(B)   (같은 test 행, 쌍체)
```

**형제(sibling) 도메인 — 구현에서 반드시 필요한 한 겹.** 위 상자를 그대로 코드로 옮기면 ΔAUROC가
항등적으로 0이 된다. 템플릿 클러스터와 eTLD+1을 union-find로 합친 `combined_group_key` 성분은
**전이적 폐포**이기 때문이다. 성분 단위로 test를 배정하는 순간 "T와 템플릿이 겹치는 행"은 하나도
남김없이 T 안으로 들어가고, 그러면 A의 train 후보가 B와 완전히 같아진다. 다운샘플로 성분 밖에
떨어지는 행을 A에 돌려주는 우회도 성립하지 않는다 — v3에서 그렇게 빠지는 행은 11,996 중 19개뿐이다.

그래서 **test로 배정된 성분 안에서 eTLD+1 단위로 한 번 더 쪼갠다**(`qrphish/campaign.py`의
`build_campaign_split`):

- **T-도메인** → 고정 test 집합 T.
- **형제(sibling) 도메인** → T와 같은 캠페인(템플릿 클러스터)의 **다른 도메인**. A의 train에만 넣는다.
  목표 비율은 `sibling_frac=0.5`(다중 도메인 test 성분의 eTLD+1 그룹 중 절반을 형제로 돌린다).
- 형제가 없는 단일 도메인 성분은 전부 T로 간다. 그 행에는 누출 기회 자체가 없으며, 비율을 진단으로 남긴다.
- **A-sizematched(`A_sm`) — 주 지표는 이쪽이다.** `train_a = train_b ∪ siblings`이므로 A는 B보다 정확히
  `|siblings|`만큼 크고, 그 크기 차이만으로도 AUROC가 오를 수 있다. `A_sm`은 **형제 행을 전부 유지한 채**
  비형제 행에서 `|siblings|`개를 무작위 제거해 `|train_A_sm| = |train_B|`로 맞춘다. 그러면 Δ에 남는 것은
  "형제 행이 비형제 행을 대체했을 때의 이득"뿐이다. **판정은 `ΔAUROC = A_sm − B`로 하고**, 크기를 맞추지
  않은 `A − B`는 보조 지표로 함께 보고한다(`qrphish.campaign.sizematched_train_a`).
- 결과적으로 `train_b ⊆ train_a`이고 `train_a = train_b ∪ siblings`다(길이 매칭 전). 이것이 "train 후보 =
  T·val에 속하지 않는 모든 행 중 eTLD+1이 T와 겹치지 않는 행"이라는 리뷰 문장을 만족시키는 유일한 구성이다.
- **val 성분은 쪼개지 않는다.** val을 쪼개면 형제 행이 A의 조기종료·임계값 선택에만 val 누출을 주어,
  A 대 B 비교에 템플릿과 무관한 교란이 낀다.

**실데이터 진단 — 검정력이 빠듯하다.** 형제를 실제로 보유한 test 행의 비율
(`frac_test_rows_with_sibling_in_A`)은 층에 따라 **0.3 ~ 6%**에 그친다. 나머지 행은 A와 B가 같은 조건에서
보는 행이므로 전체 ΔAUROC는 구조적으로 희석되고, 특히 **v4는 표본이 작아 검정력이 부족하다**. 그래서
**누출 행만 따로 본 ΔAUROC를 `cnn_leaky_vs_contrast` 태그로 부차 지표**로 함께 낸다. 이름이 "부분집합"이
아닌 이유가 있다 — 형제 행은 거의 전부 피싱이라 누출 행만 남기면 단일 클래스가 되어 AUROC가 정의되지
않는다. 그래서 이 지표는 **"누출 행 + 누출 행이 하나도 없는 클래스의 행 전부(대조)"** 위에서 계산한다.
표본 구성이 전체 T와 다르므로 두 Δ의 크기를 직접 비교하면 안 된다. 주 지표는 어디까지나 T 전체 위의 ΔAUROC다.

**해석 프레임(사전 등록).** ΔAUROC_G가 0 근처로 나오는 것은 실패가 아니다. 그것은 **옛 G가 보던 차이가
템플릿 누출이 아니라 test 집합이 교체된 효과였다는 확인**이다. 이 문장을 데이터를 보기 전에 박아 두어,
0에 가까운 결과를 뒤늦게 "검정력 부족"으로만 돌리는 사후 해석을 막는다.

**판정 기준 (2026-09-09 개정 — review_03 5절).** 주 판정은 **`A_sm − B` 쌍체 ΔAUROC의 95% 양측 CI**로
내리고 alpha는 **0.05**다. 쌍체 예측이 있으므로 그룹 단위 교환 클러스터 순열 p값도 함께 낸다.
`A − B`는 같은 규칙으로 판정을 붙이되 **주 판정에는 쓰지 않는다**(학습 표본 수 효과가 섞여 있다).

옛 규칙은 "CI 하한이 0 이하 → 누출 무시 가능"이었다. 그 규칙은 CI가 `[-0.10, +0.30]`이어도 같은 판정을
내린다 — 즉 **"차이를 검출하지 못했다"와 "차이가 없다"를 구분하지 못한다.** "무시 가능"을 주장하려면
실질적으로 중요한 최소 차이(SESOI)를 미리 정하고 **CI 상한이 그보다 작다**는 등가성 형태의 근거가 있어야
한다. 그래서 사전 등록 상수를 하나 더 둔다.

> **SESOI = 0.02 AUROC** (`qrphish.campaign.SESOI`). "실질적으로 무시 가능"이라고 부를 수 있는 최대
> 누출 이득. 1차 실험의 층 간 AUROC 변동(v2 0.872 → v4 0.805)의 약 1/3, 시드 간 표준편차 수준이다.
> 6.4의 **결론 수정 임계 0.05**(`MAJOR_THRESHOLD`)와는 **다른 질문의 임계**다 — 0.02는 "무시해도
> 되는가", 0.05는 "결론을 고쳐야 하는가".

판정 코드는 다섯 가지이며, 네 CNN 태그(`cnn_sm`·`cnn`과 두 `*_leaky_vs_contrast` 부집합) 전부에 같은
규칙으로 붙는다. 정의와 SESOI 값은 결과 JSON의 `aggregate.verdict_rule`에 함께 기록하고, 표에는 각주로
싣는다.

**누출 대조 부집합의 표본 수 경고 (필수).** `*_leaky_vs_contrast`의 검정력은 "A train에 형제가 있는 T 행"
개수에 통째로 걸려 있다. 실측은 시드당 평균 **v2 38행 / v3 35행 / v4 15행**뿐이다. 이 CI로는 구조적으로
`negligible`이 나올 수 없으므로, **"phishing kit 암기 가능성을 배제했다"는 결론의 근거로 쓸 수 없다.**
이 경고 문장과 시드별 행 수를 결과 JSON(`sample_size_warning`)과 표 각주에 항상 함께 싣는다.

**규칙이 바뀌면 재학습 없이 다시 판정한다.** `runner.recompute_campaign_verdicts(cfg)`가 저장된
`delta_auroc`·`delta_ci`만 읽어 판정을 갈아끼우고 옛 판정과 새 판정을 나란히 보고한다.

- A와 B의 train 집합만 다르고 평가 행은 완전히 같으므로 `paired_cluster_bootstrap_by_seed`를 쓴다.
  ΔAUROC_G는 "template shortcut을 학습에서 허용했을 때 얻는 이득"에 훨씬 가깝다.
- A의 train에는 T의 형제 행이 들어가고, B의 train에서는 그 형제 행까지 제거한다. **두 조건의 train 크기가
  달라지므로** 위의 `A_sm`(A-sizematched)을 주 지표로 쓰고, A와 B의 train 크기·B의 감소분도 함께 보고한다.
  그러지 않으면 차이의 일부가 학습 표본 수 효과다.
- **길이 매칭은 A와 B를 독립으로 걸지 않는다.** 따로 걸면 각자 다른 비형제 행을 버려
  `train_b ⊆ train_a`가 깨지고, Δ에 "비형제 train 행이 서로 다르다"는 교란이 섞인다. 그래서 **B를 먼저
  매칭하고 형제를 전부 그 위에 얹어 A를 만든다**(`train_a = train_b ∪ siblings`). 매칭 후에도 부분집합
  관계가 정확히 성립한다.
- A와 B를 **동시에** 정확히 매칭할 수는 없다. 형제 행이 거의 전부 피싱이라, A가 형제를 전부 가지는 한
  A의 클래스별 길이 주변분포는 반드시 틀어진다. 깨끗한 기준선인 **B를 정확히 맞추고**, A의 잔여 불균형은
  진단 `length_match["A"]`로 그대로 보고한다. 그 불균형이 만드는 크기 효과는 `A_sm`이 걷어낸다.
  (형제 프레임만 따로 `match_by_length`에 넣는 방식은 쓸 수 없다 — 단일 클래스라 클래스 균형이 형제를
  전부 지워 A와 B가 같아진다.)
- 시드 5개, 층 v2/v3/v4, 조건은 주 조건(`norm` · `L-exact` · `data_only` · `mask-fixed` · SmallCNN) 그대로.
  길이 매칭은 1차와 같이 split별로 건다.
- 함께 보고할 진단: 병합 후 그룹 수, 최대 그룹 크기 비율, 고정 test의 상위 5 그룹 점유율,
  A와 B의 train 크기, B에서 추가로 제거된 행 수.
- 베이스라인(char n-gram LR, byte-hist LR)도 같은 A/B 구조로 돌린다. 텍스트 기준선이 더 크게 떨어지면
  누출은 어휘 수준이고, CNN이 더 크게 떨어지면 QR 격자 표현이 템플릿을 특히 잘 외웠다는 뜻이다.

**부록으로 내리는 기존 방식.** 1차 분할과 template 분할을 각각 돌려 비쌍체로 빼는 원래 설계는
폐기하고 부록 서술로만 남긴다. 폐기 사유는 위의 두 수치다 — 1차 test 행의 86 ~ 88%가 교체되고
test cohort Jaccard가 0.06 ~ 0.10이라 효과 분리가 되지 않는다. 부록에는 이 진단 수치와 함께
"왜 쌍체 설계로 바꿨는가"를 적는다.

### 5.6 F와 G의 결합

**시간이 있으면** F-b(외부 자체 학습)에도 같은 A/B 쌍체 구조를 적용한다. 외부 피싱은 키트 중복이
WebPhish보다 심할 가능성이 높아 G의 효과가 더 클 수 있다. 우선순위는 낮다.

---

## 6. 성공·실패 판정 기준 (사전 등록 — 데이터를 보기 전에 확정)

**이 절은 실행 전에 확정하고, 결과를 본 뒤에 고치지 않는다.** 고쳐야 할 이유가 생기면
고친 사실과 이유를 논문에 적는다.

### 6.1 F-a (zero-shot transfer)

| 판정 | 기준 |
|---|---|
| **신호 재현 (강)** | `comparable_strata` 전부에서 전이 AUROC의 부트스트랩 CI 하한 > 그룹 순열 바닥선 CI 상한, **그리고** 전이 AUROC ≥ (in-domain AUROC − 0.10) |
| **신호 재현 (약)** | CI 하한 > 바닥선 CI 상한이지만 하락폭이 0.10 초과 0.20 이하 |
| **부분 붕괴** | 하락폭 0.20 초과이면서 CI 하한은 여전히 바닥선 초과 |
| **붕괴** | 어느 층에서든 CI 하한 ≤ 바닥선 CI 상한 (= 우연과 구분 불가) |

> 임계 0.10/0.20의 근거: 1차의 층 간 변동(v2 0.872 → v4 0.805, 0.067)과 `shuffle-pos` 절제 효과
> (0.107 ~ 0.149)를 척도로 삼았다. 0.10 이내면 층 간 변동 수준, 0.20 초과면 공간 구조를 통째로
> 파괴한 것보다 큰 하락이라는 뜻이다.

**추가 필수 조건 (게이트) — 2026-09-09 재설계 (review_03 4절).**

옛 게이트는 "두 데이터셋의 경로 편향이 같은 방향이면 실격"이었다. 실측 결과 두 세트는 실제로 같은
방향의 편향을 공유하므로(결정 절 표), 이 규칙을 그대로 두면 모든 층이 한꺼번에 탈락한다. 그러나
공유 편향은 결론을 자동으로 무효화하는 조건이 아니라 **주 판정 대상을 바꿔야 할 조건**이다. 그래서
차단 대신 **플래그**로 바꾼다.

| 게이트 | 기준 | 위반 시 |
|---|---|---|
| 길이·버전 | `length LR`·`version LR` 전이 AUROC ∈ [0.48, 0.52] | `descriptive_only` |
| 그룹 집중도 | 외부 test 상위 5 그룹 점유율 ≤ 40% | `descriptive_only` |
| 경로 편향 방향 | 두 데이터셋의 `path_depth>=1` 부호가 같음 | **플래그** (아래 규칙 적용) |
| 경로 shortcut 지배 | `path_lr` 기준선 AUROC ≥ 0.80 | **플래그** → `descriptive_only` |

- **경로 편향 플래그가 켜진 세트**에서는 길이와 경로 유무를 **함께** 맞춘 cohort(`a_fixed_lenpath`)의
  결과가 F-a의 **주 판정**이고, 길이만 맞춘 일반 cohort는 보조로 내린다.
- **`path_lr`**은 경로 유무·경로 깊이·구분자 개수 세 특징만 쓰는 로지스틱 회귀다. URL 어휘도 길이도
  보지 않으므로, 이 기준선이 높다는 것은 곧 "경로 모양만으로 라벨이 갈린다"는 뜻이다. 전이 AUROC가
  0.80 이상이면 그 층의 전이 성능은 경로 shortcut에 지배된다고 보고 `descriptive_only`로 강등한다.
- 두 플래그의 상태는 층마다 결과 JSON에 기록하고 표에 그대로 싣는다.

> **재실행 결과.** 재설계된 게이트와 `a_fixed_lenpath` cohort, `path_lr` 기준선을 넣어 다시 돌렸다.
> primary 주 판정은 v2 `partial_collapse`, v3·v4 `reproduced_weak`이고, 길이만 맞춘 cohort는 세 층 모두
> `reported_unmatched_path`다. F-b는 v2가 경로 shortcut 플래그로 `descriptive_only`, v3·v4가
> `reproduced_strong`이다.

### 6.2 CNN 대 텍스트 기준선

`transfer_gap = AUROC_charngram(전이) − AUROC_CNN(전이)`를 WebPhish 내부의
`gap_to_decoded_text`(+0.097 / +0.158 / +0.187)와 비교한다.
- 전이 갭이 내부 갭보다 **작거나 비슷**하면 CNN이 텍스트 기준선만큼은 일반화한 것.
- 전이 갭이 내부 갭보다 0.10 이상 **커지면** CNN 쪽이 특이적으로 무너진 것 → 해석 매트릭스 2행.

### 6.3 motif 재현

| 지표 | 재현 판정 기준 |
|---|---|
| Spearman ρ (log OR, 512종, 주 지표) | ρ의 95% CI 하한 > 0.2 |
| 부호 일치율 (상위 K=20) | CI 하한 > 0.5 (귀무 배제), 강한 재현은 ≥ 0.75 |
| Jaccard (상위 20) | 무작위 귀무 CI 상한 초과 |
| 외부 절제 ΔAUROC | 부호가 WebPhish와 동일 **그리고** 표적−무작위 ΔΔ의 쌍체 95% CI가 0을 배제 (2026-09-09 개정: 배수 기준 폐기 — 무작위 대조의 하락폭이 0에 가까워 비율이 불안정하다) |

**셋 중 둘 이상 충족 → "motif 재현"**, 하나만 → "부분 재현", 전부 실패 → "재현 실패".

### 6.4 G (템플릿 분할)

고정 캠페인 test 위의 쌍체 차이 `ΔAUROC_G = AUROC(Model A) − AUROC(Model B)`로 판정한다.

**2026-09-09 개정 (review_03 5절).** 아래 표가 사전 등록 규칙이다. 옛 표의 첫 줄("쌍체 CI가 0을 포함
→ 누출 무시 가능")은 폐기한다 — 그 규칙은 CI가 `[-0.10, +0.30]`이어도 "무시 가능"을 내주어 비유의성을
효과 부재로 읽게 만든다. 두 임계(SESOI 0.02, 결론 수정 0.05)의 정의는 5.5절 상자에 있다.

| 판정 코드 | `ΔAUROC_G = A_sm − B` | 읽는 법 |
|---|---|---|
| `negligible` | CI 상한 < **0.02**(SESOI) | 누출 이득이 실질적으로 무시 가능함을 등가성 형태로 보였다 |
| `not_detected` | CI가 0을 포함하고 상한 ≥ 0.02 | **상승을 검출하지 못했으나 배제하지도 못했다.** "누출이 없다"고 쓰면 안 된다 |
| `detected_minor` | CI 하한 > 0, \|Δ\| < 0.05 | 이득은 있으나 크기가 작다 |
| **`detected_major`** | CI 하한 > 0, \|Δ\| ≥ 0.05 | **결론 수정 필요** |
| `negative` | CI 상한 < 0 | 역방향 — 누출 허용 모델이 오히려 나쁘다. 검정력·교란을 점검한다 |
| `undetermined` | CI가 유한하지 않음 | 판정 불가 |
| **결론 무효** | Model B AUROC의 CI 하한 ≤ 라벨 셔플 바닥 CI 상한 | 위 판정과 무관하게 우선한다 |

마지막 줄이 나오면 `RESULTS.md`의 RQ1 답을 "eTLD+1 통제 하에서는 그렇다"로 한정해야 한다.

같은 규칙을 `cnn`(A−B, 보조)과 두 `*_leaky_vs_contrast` 부집합에도 적용해 판정을 명시하되, **주 판정은
`cnn_sm` 하나뿐이다.** 부집합 판정에는 5.5절의 표본 수 경고(시드당 평균 38 / 35 / 15행)를 반드시 함께
싣는다.

**서술 규약.** `not_detected`인 층을 서술할 때는 "정의한 템플릿 분리 방식과 평가 집합에서, 누출 허용에
따른 전체 AUROC 상승을 검출하지 못했다"까지만 쓴다. "누출은 무시 가능하다", "kit 암기 가능성을
배제했다"는 `negligible`이 나온 지표에 한해서만 쓸 수 있고, 그때에도 주장 범위는 그 지표가 실제로 덮는
집합(전체 T)까지다.

### 6.5 다중 비교

F의 주 가설은 층별 **H-F1**(전이 AUROC > 순열 바닥선) 하나뿐이다. G의 주 가설은 **H-G1**
(`ΔAUROC_G` ≠ 0) 하나. 두 family를 합쳐 층 수만큼(최대 6개) Holm 보정을 걸고
`reports/hypotheses_stage2.json`에 쓴다. 나머지(F-b/c/d, motif 재현 지표)는 **탐색적**으로 표기하고
보정에 넣지 않는다. `evaluate.holm`을 그대로 쓴다.

---

## 7. 구현 스펙

### 7.1 새 모듈

#### `qrphish/external.py`

```python
__all__ = ["SOURCES", "fetch_source", "parse_source", "build_external_frame",
           "load_external", "dedup_against", "bias_diagnostics", "HOSTING_BLOCKLIST"]

@dataclass(frozen=True)
class SourceSpec:
    name: str                 # "openphish" | "phishdb" | "phishtank" | "commoncrawl" | "tranco"
    label: int                # 1=phishing, 0=benign
    kind: Literal["text_lines", "json", "cdxj", "csv_rank"]
    url_template: str
    env_key: str | None       # 필요한 API 키의 환경변수 이름 (없으면 None)
    license_note: str

SOURCES: dict[str, SourceSpec]

def parse_source(spec: SourceSpec, raw_path: Path) -> pd.DataFrame:
    """원본 파일 -> DataFrame(url_src, label, source). 네트워크 접근 없음. 테스트 가능."""

def build_external_frame(
    parts: list[pd.DataFrame], *, mode: Literal["raw","norm"] = "norm",
    scheme_policy: Literal["strip","keep","match_webphish"] = "match_webphish",
    webphish_df: pd.DataFrame | None = None,
    dedup_mode: Literal["url","etld1"] = "etld1",
    hosting_blocklist: frozenset[str] = HOSTING_BLOCKLIST,
    extractor=None,
) -> tuple[pd.DataFrame, dict]:
    """load_webphish와 **완전히 같은 컬럼**을 내는 것이 계약이다.
    반환 컬럼: url, label, group, url_len, url_bytes_len, path_depth (+ source, dataset_origin).
    두 번째 반환값은 2.3절 meta json의 dedup/counts/bias_diagnostics 블록.
    """

def load_external(source: str, path: Path) -> pd.DataFrame:
    """urls.py의 스텁을 대체하는 정식 구현. 병합된 external_{date}.csv를 읽어
    load_webphish와 동일한 컬럼으로 돌려준다. 시그니처 확장:
      load_external(source: str, path: Path, *, mode="norm") -> tuple[pd.DataFrame, dict]
    urls.py의 기존 스텁은 external.load_external로 위임하도록 고치고,
    Literal 타입 힌트는 str로 넓힌다(소스가 늘어난다)."""
```

> `urls.py`의 `load_external` 스텁(현재 `NotImplementedError`)은 지우지 말고
> `from qrphish.external import load_external as _impl` 위임으로 바꾼다. 순환 import를 피하려고
> **함수 안에서** import한다(`external.py`가 `urls.py`의 `normalize_url`/`etld1`을 쓰므로).

#### `qrphish/transfer.py`

```python
__all__ = ["prepare_external_frame", "build_external_stratum", "transfer_eval",
           "permutation_null", "transfer_baselines"]

def prepare_external_frame(cfg, ext_df, stratum, seed) -> tuple[pd.DataFrame, dict]:
    """runner._prepare_frame과 **같은 순서**의 외부판.
    path_filter -> natural_version -> 층 필터 -> group_split -> match_by_length_within_splits
    -> assert_length_matched. _prepare_frame에서 load_webphish 호출부만 갈아끼운 형태이므로,
    **runner._prepare_frame을 리팩터해 프레임 소스를 주입 가능하게 만드는 편이 낫다**:
        _prepare_frame(cfg, stratum, seed, *, frame: pd.DataFrame | None = None)
    frame이 주어지면 load_webphish를 건너뛴다. 중복 구현을 만들지 마라 — 두 경로가
    갈라지면 F-a 대 in-domain 비교가 조용히 무의미해진다."""

def build_external_stratum(cfg, df, stratum, seed, out_dir) -> StratumMeta:
    """dataset.build_stratum 래퍼. cond/qr는 **체크포인트 스냅샷과 동일**해야 한다."""

def transfer_eval(cfg, ckpt_cid: str, stratum: str, seeds: list[int],
                  ext_df, out_dir: Path, *, eval_rows: str = "all",
                  n_boot: int = 2000) -> dict:
    """시드마다: prepare -> build_stratum -> _load_trained -> predict_probs.
    ds.n != ckpt['n']이면 ValueError. 임계값은 WebPhish val 임계값을 재사용.
    CI는 evaluate.cluster_bootstrap_by_seed."""

def permutation_null(y, p, groups, seeds, n_perm: int = 2000, seed: int = 0) -> dict:
    """**그룹 단위** 라벨 순열 -> AUROC 분포 -> {mean, ci}. 행 단위 순열 금지."""

def transfer_baselines(cfg, webphish_frames, ext_frames, seed) -> dict:
    """baselines를 WebPhish train에서 fit하고 외부에 apply."""
```

#### `qrphish/templates.py`

```python
# 실제 구현된 이름·기본값 (2026-09-08 결정 반영. 5.3절 상단 박스 참조)
__all__ = ["url_template", "template_shingles", "minhash_signature", "minhash_signatures",
           "template_groups", "combined_group_key", "template_diagnostics", "UnionFind"]

def url_template(url: str) -> str: ...               # 5.2절 규칙. 순수 함수 -> 테스트 쉬움
def template_shingles(tpl: str, k: int = 3) -> set[str]: ...   # 토큰 shingle (`/ ? & .`)
def minhash_signature(shingles, n_perm: int = 64, seed: int = 0) -> np.ndarray: ...
def template_groups(urls: Sequence[str], threshold: float = 0.7, k: int = 3, *,
                    n_perm: int = 64, rows: int = 4, seed: int = 0,
                    min_template_tokens: int = 3) -> np.ndarray:
    """path_depth==0인 행과 범용 골격은 각자 단독 클러스터. 정확 골격 일치는 MinHash 전에 union.
    반환: template_id 배열."""
def combined_group_key(df: pd.DataFrame, ...) -> np.ndarray:
    """etld1 그룹과 template_id를 union-find로 합친 최종 그룹 키."""
```

### 7.2 기존 모듈 수정

| 파일 | 수정 | 이유 |
|---|---|---|
| `qrphish/urls.py` | `load_external` 스텁 → `external.load_external` 위임 | 스펙 4절 인터페이스 유지 |
| `qrphish/baselines.py` | 각 베이스라인에 **fit/apply 분리 경로** 추가. 최소 `charngram_lr`·`bytehist_lr`에 대해 `fit_*(urls_tr, y_tr, seed) -> 객체`, `apply_*(obj, urls) -> p` | F-a는 WebPhish에서 fit한 모델을 외부에 적용해야 한다. 기존 `(urls, y, split, meta, seed)` 시그니처는 **깨지 않고** 그대로 둔다(1차 재현성) |
| `qrphish/runner.py` | `_prepare_frame(cfg, stratum, seed, *, frame=None)` 주입 인자 추가 | 외부/WebPhish 경로 통일 |
| `qrphish/runner.py` | `run_transfer(cfg, external_csv, ...)`, `run_template_split(cfg, ...)` 추가 | 아래 7.3 |
| `qrphish/config.py` | `ExternalConfig`, `TemplateConfig` 추가 + `_validate` 확장 | 아래 7.4 |
| `qrphish/cli.py` | `transfer`, `template-split`, `collect-external` 서브커맨드 | 실행 진입점 |
| `.gitignore` | 2.4절 규칙 추가 | — |

**금지**: `splits.py`는 수정하지 않는다. G는 `group` 컬럼 값만 바꿔 기존 `group_split`을 재사용한다.

### 7.3 러너 함수

```python
def run_transfer(cfg, external_csv: str | Path, *,
                 phase: str = "transfer",
                 mode: Literal["a","b","c","d"] = "a",
                 strata: list[str] | None = None) -> dict:
    """
    mode="a": zero-shot. 학습 없음. 기존 체크포인트 필요.
    mode="b": 외부 자체 학습·평가. run_matrix와 같은 경로, output_dir만 external 하위.
    mode="c": mode="b" 체크포인트로 WebPhish test 평가.
    mode="d": 혼합 학습. dataset_origin 컬럼 + origin_lr 베이스라인 필수.
    반환: 층별 결과 dict 목록. 부작용으로 결과 JSON을 아래 경로에 쓴다.
    """

def run_template_split(cfg, *, phase: str = "template_split",
                       strata: list[str] | None = None) -> dict:
    """campaign_groups로 group 컬럼을 대체한 뒤 run_matrix의 주 조건 엔트리를 재실행.
    1차 결과와의 비교(unpaired delta)까지 계산해 저장한다."""
```

**주의 (조건 격리)**: `EXPERIMENT_DESIGN.md` 14.2절의 산출물 경로 규약을 따른다.
`reports/{phase}/{condition_id}/{stratum}/`. F는 여기에 `{source_tag}`가 더 붙는다.
아티팩트는 `artifacts/external/{source_tag}/{condition_id}/{stratum}/seed{k}/`.
**`artifacts/{condition_id}/` 아래에 외부 층을 절대 쓰지 마라** — 1차 체크포인트를 덮어쓴다.

### 7.4 config 확장

```yaml
# configs/base.yaml 에 추가
external:
  csv_path: "data/external/external_2026-09-08.csv"
  meta_path: "data/external/external_2026-09-08.meta.json"
  source_tag: "openphish_cc_2026-09-08"     # 결과 경로에 쓰이는 짧은 식별자
  dedup_mode: "etld1"                        # url|etld1
  scheme_policy: "match_webphish"            # strip|keep|match_webphish
  eval_rows: "all"                           # all|test
  benign_cap_ratio: 3.0
  max_top5_group_frac: 0.40                  # 넘으면 descriptive_only
template:
  enabled: false
  shingle_size: 3      # 토큰 shingle (2026-09-08 결정: 문자 4-gram → 토큰 3-shingle)
  num_perm: 64         # 2026-09-08 결정: 128 → 64
  threshold: 0.7       # 2026-09-08 결정: 0.8 → 0.7
  max_template_frac: 0.05
  auto_raise_threshold: true
  seed: 0
```

`_validate` 추가 검사: `dedup_mode`/`scheme_policy`/`eval_rows` 열거값,
`0 < threshold <= 1`, `num_perm >= 16`, `shingle_size >= 2`,
`0 < max_template_frac <= 1`. `external.csv_path`는 **파일 존재를 검증하지 않는다**
(수집 전에도 config가 로드되어야 한다).

### 7.5 결과 JSON 스키마

`reports/transfer/{source_tag}/{mode}/{condition_id}/{stratum}/results.json`

```json
{
  "phase": "transfer", "mode": "a",
  "source_tag": "openphish_cc_2026-09-08",
  "condition_id": "norm-exact-data_only-fixed-small_cnn",
  "stratum": "v3",
  "git_sha": "...", "qrphish_version": "...", "created_at": "...",
  "external_meta_sha256": "...",

  "data": {
    "n_total": 0, "n_benign": 0, "n_phishing": 0, "base_rate": 0.0,
    "n_groups": 0, "tier": "primary|secondary|dropped",
    "eval_rows": "all",
    "length_match": {"bucket": 1, "all_identical": true},
    "split_diagnostics": {"groups_disjoint": true, "top5_test_group_frac": 0.0}
  },

  "model": {
    "auroc_mean": 0.0, "auroc_sd": 0.0, "auroc_per_seed": [],
    "auroc_pooled": 0.0, "auroc_pooled_ci": [0.0, 0.0],
    "auprc_mean": 0.0, "f1_at_val_threshold": 0.0, "threshold_source": "webphish_val",
    "n_boot": 2000
  },
  "null_permutation": {"auroc_mean": 0.0, "ci": [0.0, 0.0], "n_perm": 2000,
                       "unit": "etld1_group"},

  "baselines": {
    "charngram_lr": {"auroc": 0.0, "ci": [0.0, 0.0], "fit_on": "webphish_train"},
    "bytehist_lr": {"auroc": 0.0, "ci": [0.0, 0.0], "fit_on": "webphish_train"},
    "length_lr": {"auroc": 0.0}, "version_lr": {"auroc": 0.0},
    "motif_patch3_lr": {"auroc": 0.0}, "motif_pyramid3_lr": {"auroc": 0.0}
  },

  "comparison_to_in_domain": {
    "in_domain_auroc": 0.0, "delta": 0.0, "paired": false,
    "delta_ci": [0.0, 0.0], "transfer_gap_to_text": 0.0
  },

  "verdict": {
    "label": "reproduced_strong|reproduced_weak|partial_collapse|collapse|descriptive_only",
    "gates_passed": {"length_lr_neutral": true, "version_lr_neutral": true,
                     "bias_direction_ok": true, "group_concentration_ok": true},
    "criteria_version": "prereg_v1"
  }
}
```

`reports/template_split/{condition_id}/{stratum}/results.json`은 위의 `model`·`baselines` 블록에
다음을 더한다.

```json
{"template": {"n_clusters": 0, "threshold_used": 0.8,
              "largest_cluster_frac": 0.0,
              "cluster_size_quantiles": {"p50": 1, "p90": 0, "p99": 0, "max": 0},
              "n_groups_after_union": 0,
              "frac_stage1_test_rows_now_in_train": 0.0},
 "delta_vs_etld1": {"value": 0.0, "ci": [0.0, 0.0], "paired": false,
                    "n_boot": 2000}}
```

집계 산출물: `reports/transfer_summary.csv`, `reports/template_split_summary.csv`
(`scripts/make_results_tables.py`에 생성 함수 추가).

#### 7.5.1 구현과의 차이 (2026-09-08 갱신 — 아래가 실제 구현이다)

| 항목 | 위 초안 | **실제 구현** |
|---|---|---|
| F 결과 경로 | `reports/transfer/{source_tag}/{mode}/{condition_id}/{stratum}/results.json` | 동일 (`runner._transfer_report_dir`) |
| 외부 로더 통계 | 없음 | `reports/transfer/{source_tag}/external_stats.json` (층과 무관해 한 번만 쓴다) |
| 외부 아티팩트 | 명시 없음 | `artifacts/external/{source_tag}/{condition_id}/{stratum}/seed{n}/` (`runner._external_cond_dir`). 1차 체크포인트 `artifacts/{condition_id}/`를 침범하지 않는다 |
| G 진단 경로 | `reports/template_split/{condition_id}/{stratum}/results.json` | 진단은 `reports/template_split/{stratum}/diagnostics.json`과 층 전체 요약 `reports/template_split/diagnostics.json` (`run_template_split`). **재학습 결과**는 `template_split` 매트릭스 항목이 조건 id `...-templatesplit`로 일반 경로에 쓴다 |
| `null_permutation` | `{auroc_mean, ci, n_perm, unit}` | `+ ci_upper`(= 판정에 쓰는 바닥선), `n_valid`. 기본 `n_perm=200`(2000이 아니다) |
| `verdict.label` | 5종 | `+ reproduced_unknown_reference` (in-domain 기준값이 없을 때) |
| 최상위 키 | 표에 없음 | `+ schema_version`, `git_sha`, `qrphish_version`, `created_at`, `config`, `runtime_sec`, `motif_replication`(F-a, `motif=True`일 때) |
| `data` 블록 | 표에 없음 | `+ length_match`, `external_loader`, `bias_diagnostics`, `baseline_seed` |
| `template` 블록 | `threshold_used` 등 | `run_template_split`의 `template` 블록은 `{clusters, combined, n_groups_etld1, n_groups_after_union, threshold, shingle_k, min_template_tokens}`이고, 1차 test 이탈 비율은 층 요약의 `mean.frac_stage1_test_rows_left_test`다 |

`REQUIRED_TRANSFER_KEYS`(`qrphish/transfer.py`)가 최상위 키 계약의 단일 진실 원천이고
`tests/test_transfer.py`가 이를 검사한다.

### 7.6 노트북 셀

`notebooks/colab_run.ipynb`에 셀 3개를 추가한다(기존 `[12]`까지 뒤).

- **`[13]` 외부 데이터 업로드·검증**: `data/external/external_*.csv`와 `.meta.json`을 Drive에서 복사하고,
  `external_meta` 요약(수집일, 소스, 표본 수, `bias_diagnostics`)을 표로 출력한다.
  **`path_depth_ge1_frac`이 WebPhish와 같은 방향이면 빨간 경고를 찍고 셀을 중단**한다.
  수집 자체는 노트북에서 하지 않는다(네트워크·약관·시간).
- **`[14]` F-a 전이 평가**: `run_transfer(cfg, ext_csv, mode="a")`. 체크포인트가 Drive에 있어야 하므로
  `artifacts/` 마운트를 먼저 확인. GPU 불필요(추론만) — 런타임이 CPU여도 돌아가야 한다.
  이어서 `mode="b"` 실행(여기서만 GPU 필요, 층당 5시드 × 수 분).
- **`[15]` G 템플릿 분할**: `run_template_split(cfg)` + motif 재현(`4절`) + 표 생성
  (`make_results_tables.py`의 신규 함수) → `reports/*.csv` 갱신.

각 셀은 기존 셀과 같은 규약을 지킨다. 결과 JSON을 Drive에 백업하고, 실패 시 예외를 삼키지 않는다.

### 7.7 테스트 (`tests/`)

| 테스트 | 내용 |
|---|---|
| `test_external_parse` | 각 소스 형식(고정 fixture 문자열)에서 `parse_source`가 기대 URL 수·라벨을 낸다. **네트워크 접근 없음** |
| `test_external_columns` | `build_external_frame` 출력 컬럼·dtype이 `load_webphish`와 **정확히 일치**. 계약 테스트 |
| `test_external_dedup` | WebPhish와 URL 1건·도메인 1건이 겹치는 fixture에서 `dedup_mode` 두 값이 각각 기대대로 제거 |
| `test_external_label_conflict` | 같은 URL이 양쪽 라벨이면 양쪽 제거, 카운트 일치 |
| `test_scheme_policy` | `strip`/`keep`/`match_webphish` 각각에서 `url_bytes_len`이 기대대로 변하고, 그에 따라 `natural_version` 배정이 바뀌는 것을 명시적으로 확인 |
| `test_transfer_shape_guard` | `ds.n != ckpt["n"]`이면 ValueError |
| `test_transfer_prepare_parity` | 같은 프레임을 `_prepare_frame(frame=...)`과 WebPhish 경로에 넣으면 동일 결과(주입 리팩터가 경로를 갈라놓지 않았음) |
| `test_permutation_null_group_unit` | 그룹 단위 순열이 행 단위 순열보다 넓은 귀무 분포를 낸다 |
| `test_path_skeleton` | 5.2절 예시 표(최소 10케이스: 숫자·16진·난수·확장자·쿼리키·빈 경로·IP 호스트·유니코드) |
| `test_template_clusters` | 같은 키트 5개 변형이 한 클러스터, 무관 URL 5개가 각각 단독. `path_depth==0` 행이 뭉치지 않음 |
| `test_campaign_groups_superset` | campaign 그룹이 eTLD+1 그룹의 **coarsening**임(같은 eTLD+1이면 반드시 같은 campaign 그룹) |
| `test_template_determinism` | 같은 seed로 두 번 돌리면 `template_id`가 비트 단위로 동일 |
| `test_config_external` | 새 config 섹션의 검증 규칙 |

`test_external_parse`의 fixture는 실제 피드에서 5 ~ 10줄만 떼어 `tests/fixtures/`에 둔다
(재배포 우려가 있으면 도메인을 `example-<n>.test`로 치환한 합성 데이터로 만든다 — **이 편을 권한다**).

### 7.8 수집 스크립트 `scripts/collect_external.py`

**로컬 맥북에서 실행한다**(Colab은 네트워크 정책·세션 수명 때문에 부적합).

```
사용법:
  python scripts/collect_external.py --date 2026-09-08 --out data/external \
      --sources openphish,phishdb,commoncrawl,tranco \
      --n-phishing 20000 --n-benign 60000 \
      --cc-crawl CC-MAIN-2026-34 --openphish-history-days 90 \
      [--dry-run]

환경 변수 (없으면 해당 소스를 **건너뛰고 경고만** 남긴다 — 실패시키지 않는다):
  PHISHTANK_APP_KEY   PhishTank 다운로드용. 2026년 기준 신규 등록이 닫혀 있으므로
                      대부분의 실행에서 미설정 상태가 정상이다.
  GITHUB_TOKEN        OpenPhish public_feed 커밋 이력 API의 rate limit 완화용(선택).

동작:
  1) 소스별 원본을 data/external/raw/{source}_{date}.{ext}로 저장, sha256 기록
  2) OpenPhish는 GitHub API로 feed.txt의 최근 N일 커밋 목록을 받아
     각 커밋의 raw 파일을 받아 합집합을 만든다 (스냅샷 1회로는 표본이 부족하다).
     각 커밋 sha와 커밋 일시를 메타에 전부 남긴다.
  3) Common Crawl은 인덱스 샤드를 스트리밍하며 CDXJ 라인을 읽고,
     status==200 & mime이 text/html인 레코드만, 도메인당 최대 3개까지 채집,
     목표 수에 도달하면 중단. 시드 고정 reservoir sampling으로 무작위성 확보.
     (샤드 전체를 내려받지 말 것 — Range 요청 없이 스트리밍하며 조기 종료하면 된다)
  4) Tranco는 top-1m.csv.zip을 받고, 동시에 https://tranco-list.eu/top-1m-id 로
     영구 list id를 조회해 메타에 기록
  5) parse_source -> build_external_frame -> external_{date}.csv + .meta.json
  6) --dry-run이면 네트워크 요청 수와 예상 파일 경로만 출력하고 종료

정중함 규칙:
  - User-Agent에 연구 목적과 연락처를 명시
  - 요청 간 최소 1초 sleep, 실패 시 지수 백오프 3회
  - robots/약관을 존중. 수집한 URL을 **재배포하지 않는다**(gitignore가 강제)
  - 피싱 URL에 실제로 접속하지 않는다. **URL 문자열만 다루고 HTTP GET을 하지 않는다.**
    (Phishing.Database의 ACTIVE 판정을 그대로 신뢰한다)
```

---

## 8. 실행 순서와 예산

| 단계 | 산출물 | 위치 | 예상 시간 |
|---|---|---|---|
| 0 | `templates.py` + 테스트 | 로컬 CPU | 0.5일 |
| 1 | **G 실행** (`run_template_split`, v2/v3/v4 × 5시드) | Colab T4 | 학습 60셀, 2 ~ 3시간 |
| 2 | `collect_external.py` 실행 | 로컬 맥북 | 1 ~ 3시간(네트워크) |
| 3 | `bias_diagnostics` 검토 → **게이트** | 로컬 | 30분 |
| 4 | F-a (zero-shot) + 베이스라인 전이 | 로컬 CPU 가능 | 1시간 |
| 5 | motif 재현 (4절) | 로컬 CPU | 1시간 |
| 6 | F-b (외부 자체 학습) | Colab T4 | 2 ~ 3시간 |
| 7 | F-c / F-d | Colab T4 | 3 ~ 4시간 (선택) |

**G를 먼저 두는 이유**: 외부 데이터 수집 결과와 무관하게 실행 가능하고, 1차 결론의 유효 범위를
바로 갱신하기 때문이다. 3단계 게이트에서 막히면 수집 설계를 고쳐 2단계로 돌아간다 —
그동안 G 결과는 이미 손에 있다.

---

## 9. 결정 요약

| # | 쟁점 | 결정 | 등급 |
|---|---|---|---|
| 1 | phishing 외부 소스 | OpenPhish public_feed(GitHub 이력 누적)가 주, Phishing.Database가 증량. PhishTank는 키 있을 때만 | 필수 |
| 2 | benign 외부 소스 | **Common Crawl URL 인덱스 무작위 표본**이 주. Tranco는 대조군(EXT-B2)일 뿐 주 benign이 아니다 | 필수 |
| 3 | benign 경로 편향 | benign에 경로 있는 URL이 충분해야 한다. `path_depth_ge1_frac`이 WebPhish와 같은 방향이면 **F 실행 중단** | 필수 |
| 4 | URLhaus / APWG | 제외 (라벨 정의 불일치 / 접근성) | 확정 |
| 5 | 정규화·스킴 | `normalize_url(mode="norm")` 그대로. 스킴 정책은 WebPhish에 맞춰 자동 판정하고 메타에 기록 | 필수 |
| 6 | 중복 제거 | 정확 URL + **eTLD+1** 양방향, 주 결과는 `dedup=etld1` | 필수 |
| 7 | benign 라벨 오염 | 피싱 피드 도메인 합집합 + 호스팅 블록리스트 제거, 잔여 오염률 추정·보고. 방향은 보수적 | 권장 |
| 8 | 외부 평가의 길이 매칭 | **적용한다.** 안 하면 길이 분포의 우연한 일치를 신호로 오독 | 필수 |
| 9 | 층 소멸 | 사전 등록 규칙 유지. `comparable_strata` 교집합에서만 결론. 공집합이면 `L-none` 폴백 + descriptive | 필수 |
| 10 | zero-shot 임계값 | WebPhish val 임계값 재사용. 외부에서 재선택 금지 | 필수 |
| 11 | 바닥선 | 재학습 없는 **그룹 단위 라벨 순열** 검정 | 필수 |
| 12 | 베이스라인 전이 | char n-gram LR·byte-hist LR을 WebPhish train에서 fit → 외부에 apply. `baselines.py`에 fit/apply 분리 추가 | 필수 |
| 13 | motif 재현 주 지표 | 512종 log OR의 **Spearman ρ**. 부호 일치율·Jaccard·외부 절제는 보조 | 권장 |
| 14 | 템플릿 키 | 경로 골격(숫자→#, 16진→H, 난수→R) + 쿼리 키 집합. 호스트 미포함 | 필수 |
| 15 | 템플릿 클러스터링 | 정확 골격 일치 union → 토큰 3-shingle → MinHash(64) → LSH(0.7) → 연결 요소. `min_template_tokens=3` 게이트로 체이닝 방어 (파라미터는 2026-09-08 결정으로 초안에서 변경) | 권장 |
| 16 | `path_depth==0` 행 | 템플릿 클러스터링에서 제외, 각자 단독 클러스터 | 필수 |
| 17 | campaign 그룹 | eTLD+1 ∪ template의 union-find 연결 요소. `splits.py`는 수정하지 않는다 | 필수 |
| 18 | G의 비교 | 표본 집합이 달라 **비쌍체**. `unpaired_delta_bootstrap_by_seed` | 필수 |
| 19 | 산출 경로 | `reports/{phase}/{source_tag}/{mode}/{condition_id}/{stratum}/`, 아티팩트는 `artifacts/external/...`. 1차 체크포인트 경로 침범 금지 | 필수 |
| 20 | 다중 비교 | H-F1·H-G1만 Holm family. 나머지 탐색적 | 권장 |

---

## 10. 열린 질문 (사용자 확인 필요)

1. **PhishTank API 키를 보유하고 있는가?** 2026년 기준 신규 등록이 닫혀 있는 것으로 확인된다.
   없다면 EXT-P는 OpenPhish + Phishing.Database 둘로 간다. 이 경우 두 소스의 **출처 독립성이 낮다**는
   한계를 논문에 적어야 한다(Phishing.Database가 OpenPhish를 재집계할 가능성). 수용하는가?
2. **OpenPhish GitHub 이력을 며칠치 소급할 것인가?** 기본 제안 90일. 길수록 표본이 늘지만
   "수집 시점 고정"의 의미가 흐려지고 오래된 URL은 사라진 캠페인이 된다. 30/90/180 중 선택.
3. **Common Crawl 샘플 크기와 크롤 회차.** 기본 제안: `CC-MAIN-2026-34`, benign 목표 60,000건
   (도메인당 최대 3건). 더 늘리면 `L-exact` 후 층이 두꺼워지지만 수집 시간이 선형으로 는다.
4. **수집 시점(`COLLECT_DATE`)을 언제로 잡는가?** 구현 착수일에 한 번만 정하고 이후 재수집하지 않는 것을 권한다.
   재수집하면 F 결과 전체를 다시 돌려야 한다.
5. **benign 라벨 오염률을 수동 표본 검토로 추정할 것인가?** 300건 검토에 1 ~ 2시간. 하지 않으면
   메타의 `contamination_estimate.p_hat`을 `null`로 두고 한계 절에 "미추정"으로 적는다.
6. **Phishing.Database의 라이선스**를 확인하지 못했다(미확인). 구현 시 LICENSE 파일을 읽고,
   연구 사용에 제약이 있으면 OpenPhish 단독으로 축소할 것인가?
7. **F-c·F-d를 이번 회차에 포함할 것인가?** F-a·F-b만으로 RQ3 후반에 답할 수 있다.
   F-c·F-d는 Colab 예산 3 ~ 4시간을 더 쓴다.
8. **G에서 `threshold`(현 구현 0.7)를 고정할 것인가, 진단 후 확정할 것인가?** 제안은 후자
   (클러스터 크기 분포를 먼저 보고 확정한 뒤 config에 박기). 다만 그것은 데이터를 본 뒤의 선택이므로,
   **임계값 선택은 라벨을 보지 않고 클러스터 크기 분포만 보고** 한다는 점을 명시해야 한다.

---

## 부록 A. 외부 소스 접근 조건 (2026-09-08 확인)

| 항목 | 값 | 확인 상태 |
|---|---|---|
| OpenPhish community feed | `https://openphish.com/feed.txt`, 6시간 주기, GitHub 미러 `openphish/public_feed` (`feed.txt` 커밋 이력) | 확인 |
| OpenPhish 약관 | `https://openphish.com/terms.html` 동의 필요, 재배포 제한 | 확인(본문 조항 세부는 미확인) |
| PhishTank 데이터 URL | `http://data.phishtank.com/data/<appkey>/online-valid.json.bz2`, 시간당 갱신 | 확인 |
| PhishTank 신규 등록 | 2020년 남용 이후 중단, **2026년 6월 기준 여전히 닫힘**으로 보고됨 | 확인(2차 출처) |
| Phishing.Database | `Phishing-Database/Phishing.Database`(구 mitchellkrogza), `phishing-links-ACTIVE.txt` 등, 시간당 갱신, PyFunceble 검증 | 확인 |
| Phishing.Database 라이선스 | — | **미확인** |
| Tranco 다운로드 | `https://tranco-list.eu/top-1m.csv.zip`, 영구 id는 `https://tranco-list.eu/top-1m-id` | 확인 |
| Common Crawl CDX API | `https://index.commoncrawl.org/CC-MAIN-2026-34-index?url=...&output=json` | 확인 |
| Common Crawl 최신 크롤 | CC-MAIN-2026-34(2026-08), 2026-30, 2026-25, 2026-21, 2026-17 | 확인 |
| Common Crawl 인덱스 샤드 | `https://data.commoncrawl.org/cc-index/collections/<crawl>/indexes/cdx-000NN.gz` | 확인(경로 규약), 샤드 개수·크기는 미확인 |
| Common Crawl Parquet 인덱스 | `s3://commoncrawl/cc-index/table/cc-main/warc/` (Athena/DuckDB/Spark) | 확인 |
| Majestic Million | `https://downloads.majestic.com/majestic_million.csv` | 미확인(제외 결정이라 확인 불필요) |
| APWG eCrime Exchange | 기관 회원 필요 | 미확인 |

**출처**
- https://phishtank.org/api_info.php , https://www.phishtank.com/developer_info.php
- https://github.com/openphish/public_feed , https://openphish.com/phishing_feeds.html
- https://github.com/mitchellkrogza/Phishing.Database
- https://tranco-list.eu/
- https://commoncrawl.org/columnar-index , https://commoncrawl.org/cdxj-index , https://index.commoncrawl.org/collinfo.json
