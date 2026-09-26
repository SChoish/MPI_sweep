# IQL 게시 로그 안내

자동 생성: python3 scripts/build_log_index.py. 원본 CSV와 상태 파일의 위치는 유지합니다.
상태 완료 수는 게시자가 보고한 진행 상황입니다. 본문 채택 수는 build_main_results.py의 최종 actor·mean 평가·중복 검사 결과입니다.

| 게시 폴더 | 실험 | 머신·출처 | 상태 완료/계획 | CSV 원시 행 | 본문 채택 | 상태 갱신 |
|---|---|---|---:|---:|---:|---|
| [STATUS.json](iql_awr_fr/STATUS.json) | AWR–FR | iisl-server02, svcho | 275/756 | 307 | 297 | — |
| [STATUS.json](iql_awr_fr_hopper_expert_k3/STATUS.json) | AWR–FR | choi (계정) | 16/16 | 48 | 0 | 2026-09-26 04:30:03 KST |
| [STATUS.json](iql_gauss_fr_loco/STATUS.json) | Gaussian Q+BC–FR (본문 제외) | choi (계정) | 6/3924 | 6 | 0 | 2026-09-26 04:30:03 KST |
| [STATUS.json](iql_gauss_fr_s0/STATUS.json) | Gaussian Q+BC–FR (본문 제외) | ext_csh (계정) | 124/1188 | 124 | 0 | — |
| [STATUS.json](iql_gauss_v5/STATUS.json) | Gaussian Q+BC–W2 | choi (계정) | 360/360 | 1680 | 360 | 2026-09-26 04:30:03 KST |
| [STATUS.json](iql_k4_t6/STATUS.json) | AWR–FR, Deterministic Q+BC–W2 (본문 제외), Gaussian Q+BC–W2 | shchoi (계정) | 396/540 | 3678 | 753 | 2026-09-26 11:11:52 UTC+09:00 |
| [STATUS.json](iql_qbc_deterministic_w2_hscale_seed0/STATUS.json) | Deterministic Q+BC–W2 (본문 제외) | ext_csv (계정) | 282/1620 | 0 | 0 | 2026-09-20T01:57:44Z |
| [STATUS.json](iql_qbc_deterministic_w2_seed0/STATUS.json) | Deterministic Q+BC–W2 (본문 제외) | ext_csv (계정) | 24/162 | 0 | 0 | 2026-09-19T05:08:26Z |
| [STATUS.json](iql_qbc_gaussian_w2_hc_walker_s0123/STATUS.json) | Gaussian Q+BC–W2 | DGX-H200-02, ext_csv | 720/720 | 720 | 720 | 2026-09-26T02:12:26Z |

## 파일 역할

- STATUS.json: 머신이 보고한 계획·진행 상태. 완료 수만으로 점수를 만들지 않습니다.
- scores_verified.csv + EXPORT.json: 게시자의 검증된 최종점수 export와 근거.
- scores.csv / scores_long.csv: 원래 게시 형식. 중간 actor, sample 평가, 다른 변형이 섞일 수 있습니다.
- queue.log: 큐의 실행 기록. 점수나 학습 완료의 근거로 사용하지 않습니다.
- [MAIN_RESULTS.md](../MAIN_RESULTS.md): 채택 점수. [main_results_audit.json](../reports/main_results_audit.json): 후보별 선택·충돌·출처.
- 상태 완료는 run 수이고 본문 채택은 방법·환경·T·K·시드별 점수 수입니다. 공유 K=4 run에 여러 방법의 점수와 구형 full-T K=1 baseline이 있으면 본문 채택 수가 완료 run 수보다 클 수 있습니다.

## 파일별 원시 행과 채택 점수

| 게시 폴더 | 점수 파일 (원시 행) | 본문 후보 → 채택 |
|---|---|---:|
| sweep_results/iql_awr_fr | [scores.csv](iql_awr_fr/scores.csv) (275), [scores_verified.csv](iql_awr_fr/scores_verified.csv) (32), [EXPORT.json](iql_awr_fr/EXPORT.json) | 307 → 297 |
| sweep_results/iql_awr_fr_hopper_expert_k3 | [scores.csv](iql_awr_fr_hopper_expert_k3/scores.csv) (48) | 0 → 0 |
| sweep_results/iql_gauss_fr_loco | [scores.csv](iql_gauss_fr_loco/scores.csv) (6) | 0 → 0 |
| sweep_results/iql_gauss_fr_s0 | [scores.csv](iql_gauss_fr_s0/scores.csv) (124) | 0 → 0 |
| sweep_results/iql_gauss_v5 | [scores.csv](iql_gauss_v5/scores.csv) (1680) | 360 → 360 |
| sweep_results/iql_k4_t6 | [scores_long.csv](iql_k4_t6/scores_long.csv) (3678) | 1196 → 753 |
| sweep_results/iql_qbc_deterministic_w2_hscale_seed0 | 점수 CSV 없음 | 0 → 0 |
| sweep_results/iql_qbc_deterministic_w2_seed0 | 점수 CSV 없음 | 0 → 0 |
| sweep_results/iql_qbc_gaussian_w2_hc_walker_s0123 | [scores_verified.csv](iql_qbc_gaussian_w2_hc_walker_s0123/scores_verified.csv) (720), [EXPORT.json](iql_qbc_gaussian_w2_hc_walker_s0123/EXPORT.json) | 720 → 720 |

Gaussian Q+BC–FR와 deterministic Q+BC–W2는 본문에서 제외합니다. K=4는 다른 머신에서 실행돼도 같은 방법의 K=4로 합칩니다.

## 파서 확인 필요

- sweep_results/iql_awr_fr_hopper_expert_k3/scores.csv: step 없는 신규 소스는 집계 보류
