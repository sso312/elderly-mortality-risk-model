# 노인 ICU 조기 사망 위험 예측 프로젝트 (MIMIC-IV)

고령 ICU 환자의 시간대별 임상 데이터를 기반으로, **미래 사망 위험을 조기 경보** 형태로 예측하는 프로젝트입니다.  
중간보고서(PDF)의 문제의식(초고령화, ICU 자원 한계, 조기 개입 필요)을 실험 코드와 서비스(대시보드)까지 연결하는 형태로 구현했습니다.

## 1. 프로젝트 개요

- 프로젝트명: 노인 조기 사망 예측 및 중환자 위험 모니터링
- 데이터셋: MIMIC-IV
- 대상: `age >= 65` 고령 ICU 환자
- 핵심 목표: 환자별(stay-level) 조기 위험도 산출 및 우선순위 의사결정 지원
- 최종 산출물:
  - 전처리/학습 파이프라인 코드
  - 모델 실험 스크립트(LSTM, BoXHED, 비교 베이스라인)
  - Streamlit 모니터링 앱

## 2. 배경 및 문제 정의 (PDF 반영)

- 고령 환자 증가로 중환자실 위험관리 난이도 상승
- 동일 인력/자원 대비 모니터링 대상이 많아져 선제 대응 필요
- "현재 상태 분류"보다 **"앞으로 H시간 내 악화/사망 가능성"** 예측이 임상적으로 더 유의미

본 프로젝트는 이를 위해, 시점 `t` 기준 미래 구간 `(t, t+H]` 라벨을 정의해 조기 경보형 모델을 설계했습니다.

## 3. 데이터 구성

사용 테이블:

- `icustays`
- `admissions`
- `patients`
- `chartevents`
- `labevents`

코호트 정의:

- 입원 시점 나이 재계산 후 `age >= 65`
- ICU 체류 구간(`intime ~ outtime`) 내 사망 시 `icu_mortality=1`
- 관측 창: 최대 `120h` (`t=0~119`)

## 4. 전처리 파이프라인

### 4.1 시간축 정렬 및 변수 매핑

- `chartevents`는 `stay_id` 기반 직접 매핑
- `labevents`는 `subject_id/hadm_id` + ICU 체류시간 필터로 매핑
- 1시간 버킷화: `t = floor((charttime - intime)/1h)`
- 변수별 itemid 매핑 테이블 구성 (HR, RR, BP, SpO2, FiO2, GCS, Glucose, pH 등)

### 4.2 정제 규칙

- 변수별 임상 범위 기반 이상치 제거 (`bounds`)
- Temp 단위 보정(화씨 입력 추정 시 섭씨 변환)
- 변수별 집계 방식 적용 (`min/max` rule)
- wide pivot 후 stay 단위 forward fill

### 4.3 파생 피처

- `GCS_Total` 생성 (Eye/Verbal/Motor 동시 측정 시)
- 추세 피처 30개 생성:
  - `{var}_diff`
  - `{var}_mean_6h`
  - `{var}_std_6h`

### 4.4 누수 방지

- `subject_id` 단위 stratified split (train/valid/test)
- split 간 `subject_id`, `stay_id` 완전 분리 assert
- 이벤트 이후 구간 제거 옵션(`DROP_AFTER_EVENT`)
- 우측 검열 처리(`_label_observable`)

## 5. 라벨 설계

### 5.1 이벤트 라벨

- `event`, `delta`: 실제 사망 이벤트 시점 row에만 1

### 5.2 조기경보 라벨 (핵심)

- `_future_label`: 시점 `t`에서 `(t, t+H]` 내 이벤트 발생 시 1
- 기본 실험 설정:
  - `HORIZON_HOURS = 6`
  - `CUTOFF_HOURS = 24`
  - `TARGET_RECALL = 0.80`

## 6. 모델링 전략

실험 대상:

- XGBoost (baseline)
- LightGBM (baseline)
- LSTM Timewise (시계열 모델)
- BoXHED (이벤트/생존형 확장 파이프라인)

평가 단위:

- row-level score -> stay-level score 집계(`max/mean/last`)
- 지표: ROC-AUC, PR-AUC, Precision, Recall, F1
- 임계값 선택: validation에서 `target_recall` 이상 중 precision 최대

## 7. 실험 설정 및 결과 요약 (노트북 실행 로그 기준)

### 7.1 데이터/피처 스냅샷

- 로드된 분할 크기:
  - train: `(3,987,360, 15)`
  - valid: `(976,560, 15)`
  - test: `(1,243,080, 15)`
- LSTM 입력 피처 수: `12`

### 7.2 성능 결과

| 모델 | 평가셋 | Threshold | ROC-AUC | PR-AUC(AP) | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | TEST | 0.5000 | 0.9439 | 0.4781 | 0.6283 | 0.1055 | 0.1807 |
| LightGBM | TEST | 0.5000 | 0.9648 | 0.5635 | 0.6000 | 0.0134 | 0.0262 |
| LSTM | VALID | 0.0269 | 0.8721 | 0.3519 | 0.1321 | 0.8049 | - |
| LSTM | TEST | 0.0269 | 0.8575 | 0.3861 | 0.1459 | 0.7784 | - |

주의:

- XGBoost/LightGBM 결과는 `thr=0.5` 고정 결과
- LSTM 결과는 `target_recall` 기준으로 선택된 임계값 결과
- 임계값 정책이 다르므로, 운영 의사결정 관점에서는 recall 제약 기반 비교가 핵심

## 8. 모델 선택 근거 

최종 운영 모델 우선순위:

- **1순위: LSTM Timewise**
  - 목표인 조기경보 성격(높은 recall)에 부합
  - TEST에서 `Recall 0.7784`로 임상 선별 목적에 적합
  - 시간축 패턴을 직접 학습 가능

보조/비교 모델:

- XGBoost, LightGBM
  - 분류 성능(ROC-AUC/PR-AUC)은 높지만 고정 threshold에서 재현율이 낮음
  - high-recall 운영 정책에서는 threshold 재설계가 필수

확장 모델:

- BoXHED
  - 이벤트/생존 관점 분석 확장용으로 코드 파이프라인 유지

## 9. 서비스(대시보드) 연계

- Streamlit 앱: `streamlit/app.py`
- 기능:
  - 환자별 최신 위험도 조회
  - 바이탈/위험도 추이 시각화
  - 위험군 필터링(critical/stable)
- 모델 출력 -> 서비스 입력 변환 스크립트 제공
  - `scripts/export_streamlit_data.py`

## 10. 디렉토리 구조

- `src/elderly_mortality/`
  - 전처리, 라벨링, 피처생성, 분할, 평가, 모델 유틸
- `scripts/`
  - 전체 전처리 / LSTM / BoXHED / Streamlit export 실행
- `streamlit/`
  - 앱 코드 및 데이터 폴더

## 11. 실행 방법

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_full_preprocessing.py \
  --data-root "C:/path/to/mimic/csv_root" \
  --out-dir "./artifacts/boxhed_io"
```

```bash
python scripts/run_lstm_experiment.py \
  --input-dir "./artifacts/boxhed_io" \
  --out-dir "./artifacts/lstm" \
  --horizon-hours 6
```

```bash
python scripts/run_boxhed_experiment.py \
  --input-dir "./artifacts/boxhed_io" \
  --out-dir "./artifacts/boxhed"
```

```bash
python scripts/export_streamlit_data.py \
  --source-csv "./artifacts/boxhed_io/test_df.csv" \
  --out-dir "./streamlit/data" \
  --n-patients 20
```

```bash
streamlit run streamlit/app.py
```

## 12. 기술 스택

- Python, Pandas, NumPy
- DuckDB
- Scikit-learn, TensorFlow
- BoXHED
- Streamlit, Altair, Plotly


