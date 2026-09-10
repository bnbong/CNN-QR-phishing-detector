# backups

프로젝트를 처음부터 다시 만들기 전에 남겨둔 자료 모음이다.

- `data/URL.xlsx`, `data/images/`: 이전 데이터셋 원본(URL 목록과 QR 이미지). 재구축 시 참고하거나 재사용하려고 보존했다.
- `evaluate_outputs/`: 이전 모델의 평가 결과(혼동행렬, ROC, Grad-CAM 히트맵 등). 당시 결과를 확인하기 위한 기록이다. 현재 실험과는 지표 및 데이터 구성이 달라 직접 비교하지 않는다.
- `README.legacy.md`: 이전 버전 프로젝트 설명 문서. 구조 재설계 시 참고용.
- `notebooks/`: 데이터 생성, 시각화, 예측, 평가에 쓰인 노트북 전체. 로직을 재사용할 수 있어 보존했다.
- `training_log.csv`: 이전 학습 로그. 새 학습과 비교할 참고 자료다.

`data/`(URL.xlsx, images)는 용량이 커서 `.gitignore`에서 계속 제외되며, `evaluate_outputs/`의 이미지들은 예외 처리되어 계속 추적된다.
