# ⚡ 전력 수요 예측 시스템

Streamlit 기반의 전력/가스 수요 예측 웹 애플리케이션입니다. Google Sheets와 연동해 데이터를 로딩·편집하고, RandomForest/LightGBM 모델로 예측하며, Step 0~6 단계형 UI로 작업 흐름을 제공합니다.

## 🚀 주요 기능

- **Google Sheets 연동**: 자동 로딩, 인라인 편집, 변경분만 업데이트
- **머신러닝 예측**:
  - 최대수요: RandomForestRegressor
  - 가스수요: LightGBM (단일 모델)
- **모델 성능 평가**: MAE, R² 지표 (Step 5 익스팬더 내부에서 출력)
- **예측 실행**: 날짜 기반/입력 기반 폼으로 예측 및 결과 시각화
- **전역 로딩 배너**: Step 4~5 진행 시 상단 배너로 상태 표시

## 🧭 UI 단계(Flow)

- Step 0: 데이터 로딩 및 편집 — `st.header("📁 Step 0: ...")`
- Step 1: 데이터 준비 — `with st.expander("📋 Step 1: ...")`
- Step 2: 특징 공학 및 데이터 정제 — `with st.expander("🔧 Step 2: ...")`
- Step 3: 모델 변수 및 데이터 분리 — `with st.expander("🎯 Step 3: ...")`
- Step 4: 모델 학습 — `with st.expander("🤖 Step 4: ...")`
- Step 5: 모델 성능 평가 — `with st.expander("📊 Step 5: ...")`
- Step 6: 전력 수요 예측 — 입력 폼 및 결과 표시

## 📋 설치 및 실행

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

### 2) Google Sheets 연동 설정

- 서비스 계정 JSON 키를 환경변수로 제공 (권장)
  - Windows PowerShell 예시:
    ```powershell
    $env:GOOGLE_CREDENTIALS_JSON='{"type":"service_account", ... }'
    ```
- 또는 코드 내 하드코딩(개발/테스트용 백업 경로) 사용
- 시트 공유 설정: 서비스 계정 이메일을 대상 시트 공유자에 추가(편집 권한)
- 시트 정보: 코드의 `sheet_name`, `sheet_id` 값 확인/수정

### 3) 애플리케이션 실행

```bash
streamlit run streamlit_app.py
```

## 🗂️ 데이터 형식(예)

- `날짜` (YYYY-MM-DD)
- `최대수요` 또는 `전력 수요` (MW)
- `최고기온`, `평균기온`, `최저기온`, `체감온도`
- 파생 변수: `태양광최대`, `잔여부하`, `최대수요대비_태양광비율`, `최대수요대비_잔여부하비율` 등

## 🤖 모델 및 지표

- 최대수요: RandomForestRegressor (간단 튜닝 포함)
- 가스수요: LightGBM (단일 모델)
- 평가 지표: MAE, R²
- 모델 성능 출력: Step 5 익스팬더 내부에서 `st.metric`으로 표시

## 🔮 예측 기능

- Step 6에서 날짜 기반/입력 기반 예측 지원
- 전역 진행 배너: Step 4 시작 시 표시, Step 5 종료 시 제거

## 🛠️ 기술 스택

- Frontend/UI: Streamlit, Plotly
- Data: Pandas, NumPy
- ML: scikit-learn (RandomForest), LightGBM
- Storage/Sync: Google Sheets (gspread, Google OAuth2)

## 🧯 트러블슈팅

- 인증/권한: `GOOGLE_CREDENTIALS_JSON` 설정 및 시트 공유 권한 확인
- API 한도: 호출 제한 시 대기/재시도, 편집 변경분만 업데이트
- 데이터: 날짜 파싱/결측 처리/필수 컬럼 검증

## 📝 라이선스

MIT License
