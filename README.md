# ⚡ 전력 수요 예측 시스템

Streamlit을 사용한 전력 수요 예측 웹 애플리케이션입니다.

## 🚀 주요 기능

- **구글 시트 연동**: 실시간 데이터 로딩 및 편집
- **머신러닝 예측**: Random Forest를 사용한 최대/최저 수요 예측
- **인터랙티브 시각화**: Plotly를 활용한 동적 차트
- **데이터 편집**: 웹 인터페이스를 통한 실시간 데이터 수정

## 📋 설치 및 실행

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 구글 시트 연동 설정

#### 방법 1: 환경변수 사용 (권장)

1. **`.env` 파일 생성** (이미 생성됨)
   ```bash
   # .env 파일에 Google 서비스 계정 JSON 키 설정
   GOOGLE_CREDENTIALS_JSON={"type":"service_account",...}
   ```

2. **환경변수 직접 설정** (Windows PowerShell)
   ```powershell
   $env:GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}'
   ```

#### 방법 2: 하드코딩 (백업 옵션)

환경변수가 설정되지 않은 경우 자동으로 하드코딩된 값 사용

### 3. 구글 시트 권한 설정

1. **구글 시트 공유**: `firebase-adminsdk-fbsvc@test-92f50.iam.gserviceaccount.com`에게 편집자 권한 부여
2. **시트 ID 확인**: `1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4`

### 4. 애플리케이션 실행

```bash
streamlit run streamlit_app.py
```

## 🔧 인증 방식 비교

### 환경변수 방식 (권장)
- **장점**: 보안성, 유연성, 안정성
- **설정**: `.env` 파일 또는 시스템 환경변수
- **우선순위**: 최고

### 하드코딩 방식 (백업)
- **장점**: 간단한 설정
- **단점**: 보안 위험, 코드 노출
- **우선순위**: 환경변수 없을 때만 사용

## 📊 데이터 형식

필수 컬럼:
- `날짜`: YYYY-MM-DD 형식
- `최고기온`, `평균기온`, `최저기온`: 섭씨 온도
- `최대수요`, `최저수요`: MW 단위
- `요일`: 요일명 (월요일, 화요일, ...)
- `평일`: 평일 여부 (평일/주말)

## 🤖 예측 모델

### 최대수요 예측 모델
- **특징**: 최고기온, 평균기온, 월, 어제의_최대수요, 요일, 평일
- **알고리즘**: Random Forest Regressor

### 최저수요 예측 모델
- **특징**: 최저기온, 평균기온, 월, 어제의_최저수요, 요일, 평일
- **알고리즘**: Random Forest Regressor

## 📈 성능 평가

- **MAE (Mean Absolute Error)**: 예측 오차의 절대값 평균
- **R² (결정 계수)**: 모델의 설명력 (0~1, 높을수록 좋음)

## 🔗 관련 링크

- **기상청 기상자료개방포털**: https://data.kma.go.kr/stcs/grnd/grndTaList.do?pgmNo=70
- **한국전력거래소**: https://www.kpx.or.kr/powerinfoSubmain.es?mid=a10606030000

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python
- **ML**: scikit-learn (Random Forest)
- **Visualization**: Plotly
- **Data Storage**: Google Sheets
- **Authentication**: Google Service Account

## 📝 라이선스

MIT License 
