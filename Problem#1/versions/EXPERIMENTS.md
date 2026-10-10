# Problem#1 레이더 이동 표적 식별 — 실험 기록

- 지표: Accuracy (최대화), 평가 단위 sample_id
- split: RepeatedStratifiedKFold(5-fold × 3 반복, random_state=42), sample_id 단위 (1 sample = 1 행이므로 그룹 누수 없음)
- 환경: 개발 세션 Python 3.13 / sklearn 1.9.1 / lightgbm 4.7.0 / 4코어 (대회 서버 2코어, sklearn 1.1.3에서는 미실행)

| 버전 | 구성 | CV 평균 ± std (반복 간) | 결정 | 파일 |
|---|---|---|---|---|
| v1 | 사용자 초안: 7채널 × (mean/std/min/max/first/last, diff, rolling5 std, gap) + RF(200, balanced) | 0.816 ± 0.001 | 기준 | v1_baseline_rf.csv |
| v2 | 물리 피처 105개(detection 수·프레임 수·프레임당 점 수, 프레임 단위 크기/속도 퍼짐/RCS, 선형 RCS 합, 중심 궤적 속도·잔차) + ExtraTrees(800) + LightGBM 확률 평균 | **0.858 ± 0.003** | **메인 채택** | v2_physfeat_et_lgb.csv = submission.csv |
| v2b | v2 피처 + ExtraTrees + sklearn HGB (lightgbm 설치 실패 시 대체) | 0.852 ± 0.002 | 대체 버전 보관 | v2b_physfeat_et_hgb_sklearn_only.csv |
| v3 | v2 + 진행 방향 기준 길이/폭, 속도 분포 형태(kurt/skew/gap) 15개 | ET+LGB 0.855 | 기각 (노이즈 수준, 복잡도 증가) | - |
| v4 | v2 + 프레임 내 무리 수 16개 (scipy 단일 연결, 거리 0.5/1/2m, 위치+속도 공간, 궤적 제거 누적 점구름) | ET+LGB 0.861 ± 0.005 (반복별 +0.001/+0.004/+0.006), ET 단독 0.847로 하락 | 기각: 노이즈 수준, 목표였던 PERSONNEL↔GROUP 혼동 66→66건 그대로, 피처 생성 +9초 | - |
| v4s | v2 + 무리 피처 4개만 (cl0.5_mean, cl1.0_mean, cl1.0_bigfrac, acc_cl1.0) | ET+LGB 0.857 ± 0.006 | 기각 (v2와 동일) | - |
| v2-et-mf | ET max_features 0.3 / 0.5 | 0.856 / 0.857 (앙상블) | 기각 (노이즈 수준) | - |

단일 모델 (v2 피처, 5×3 CV): RF 0.839, ET 0.854, HGB 0.847, LGBM 0.852.

## 서버 버전 환경 검증 (2026-10-10)
- 환경: Python 3.9.25 / numpy 1.23.5 / pandas 1.5.2 / scikit-learn 1.1.3 / scipy 1.9.3 venv, `taskset` CPU 2코어 제한, nbconvert로 노트북 전체 실행.
- 메인 경로(첫 셀이 lightgbm 4.6.0 설치): 실행 약 42초, 오류·stderr 출력 0, 실행 번호 1~6 순서. 노트북 5-fold CV 0.852 ± 0.039. submission.csv가 v2_physfeat_et_lgb.csv와 해시 동일(md5 60c48396…).
- 대체 경로(lightgbm 설치 실패, HGB): 실행 약 49초, 오류 0. 노트북 5-fold CV 0.844 ± 0.035. v2b CSV를 이 서버 버전 실행 결과로 교체(md5 805bc258…).
- 첫 셀 수정: pip 출력이 stderr로 셀에 남지 않도록 캡처하고 성공/실패만 출력.

## 메모
- 노트북 단일 5-fold(seed 42) 결과: fold 0.885/0.805/0.805/0.870/0.895, 평균 0.852 ± 0.039. fold 간 편차가 커서 200개 홀드아웃 1회 점수(초안 0.86)는 ±4%p 수준으로 흔들린다.
- 주 오류: PERSONNEL ↔ PERSONNEL_GROUP(약 68건), HEAVY → LIGHT(21건). detection 6개 이하 sample 정확도 0.74, 31개 이상 0.94 → 정보량 부족이 천장 요인.
- test 예측 비율: HEAVY 0.17, LIGHT 0.207, PERSONNEL 0.207, GROUP 0.237, TWO_WHEELER 0.18 (train 균등 0.2). test sample당 detection 수 분포는 train과 유사.
- v2 vs v2b 예측 일치 95.3%, v2 vs v1 84.7%.
