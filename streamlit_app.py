import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import io
import gspread
from google.oauth2.service_account import Credentials
import os
import json
from functools import partial
import holidays

# 성능 관련 상수
APPLY_SHEET_FORMATTING = False  # 구글시트 업데이트 시 서식 적용 여부 (속도 개선을 위해 기본 비활성화)
QUICK_SHEET_CONNECT = True      # 구글시트 연결 시 검증 호출 생략하여 초기 로딩 가속

# 학습 캐싱 함수들
@st.cache_resource(show_spinner=False)
def train_rf_model(X: pd.DataFrame, y: pd.Series, *, n_estimators: int, random_state: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model

def tune_rf_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
):
    """간단한 시계열 CV 기반 RandomForest 튜닝."""
    # 시계열 분할 (인덱스 순서를 시간 순서로 가정)
    tscv = TimeSeriesSplit(n_splits=3)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 8, 12, 16],
        'min_samples_leaf': [1, 2, 4],
    }
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=6,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    *,
    test_size: float,
):
    """시간 순서(오름차순)로 학습/평가 세트를 분할합니다.

    마지막 test_size 비율 구간을 테스트로 사용합니다.
    """
    try:
        # 정렬용 인덱스
        sort_idx = dates.sort_values().index
        X_sorted = X.loc[sort_idx]
        y_sorted = y.loc[sort_idx]
        split_idx = int(len(X_sorted) * (1 - test_size))
        split_idx = max(1, min(split_idx, len(X_sorted) - 1))
        return (
            X_sorted.iloc[:split_idx],
            X_sorted.iloc[split_idx:],
            y_sorted.iloc[:split_idx],
            y_sorted.iloc[split_idx:],
        )
    except Exception:
        # 실패 시 안전하게 전체를 학습으로 반환
        return X, X.iloc[0:0], y, y.iloc[0:0]

def align_features_for_model(model, df: pd.DataFrame) -> pd.DataFrame:
    """모델 학습 시 사용한 컬럼 집합(feature_names_in_)에 입력을 정렬.
    - 학습 시 있었던 컬럼이 예측 시 없으면 0으로 생성
    - 학습 시 없던 컬럼은 드롭
    - 컬럼 순서 일치
    """
    try:
        feature_names = list(getattr(model, 'feature_names_in_', []))
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            # 여분 컬럼 제거 및 순서 정렬
            df = df[feature_names]
    except Exception:
        pass
    return df

@st.cache_resource(show_spinner=False)
def train_lgbm_gas_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    monotone_constraints: list,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    random_state: int
):
    # 중요한 하이퍼파라미터만 캐시 키에 반영하도록 인자 유지
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        n_jobs=-1,
        monotone_constraints=monotone_constraints,
    )
    model.fit(X, y)
    return model

# 페이지 설정 (반드시 첫 번째 Streamlit 명령어여야 함)
st.set_page_config(
    page_title="전력 수요 예측 시스템",
    page_icon="⚡",
    layout="wide"
)

# 제목
st.title("⚡ 전력 수요 예측 시스템")
st.markdown("---")

# 전역 변수 초기화
if 'mae_max' not in st.session_state:
    st.session_state.mae_max = None
if 'r2_max' not in st.session_state:
    st.session_state.r2_max = None

# 구글 시트 설정
@st.cache_resource(show_spinner=False)
def setup_google_sheets():
    """구글 시트 연결 설정"""
    try:
        # 구글 시트 API 스코프 설정
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # 방법 1: Streamlit secrets에서 JSON 키 읽기 (우선순위)
        try:
            google_credentials_json = st.secrets.get('GOOGLE_CREDENTIALS_JSON')
            if google_credentials_json:
                # JSON 문자열을 딕셔너리로 변환
                credentials_data = json.loads(google_credentials_json)
                
                # private_key 형식 검증 및 수정
                if 'private_key' in credentials_data:
                    private_key = credentials_data['private_key']
                    # 개행 문자 정규화
                    if '\\n' in private_key:
                        credentials_data['private_key'] = private_key.replace('\\n', '\n')
                
                # 필수 필드 확인
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in credentials_data]
                if missing_fields:
                    st.error(f"❌ 필수 필드가 누락되었습니다: {missing_fields}")
                    return None
                
                # 인증 정보 생성
                creds = Credentials.from_service_account_info(
                    credentials_data, 
                    scopes=scope
                )
                
                # gspread 클라이언트 생성
                client = gspread.authorize(creds)
                
                # 연결 테스트(옵션)
                if not QUICK_SHEET_CONNECT:
                    try:
                        test_sheet = client.open_by_key("1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4")
                        return client
                    except Exception as test_error:
                        st.error(f"❌ 구글 시트 접근 테스트 실패: {str(test_error)}")
                        st.info("""
                        **구글 시트 접근 권한 확인:**
                        1. 서비스 계정 이메일: power-supply@flash-zenith-453703-p6.iam.gserviceaccount.com
                        2. 구글 시트 ID: 1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4
                        3. 구글 시트에 서비스 계정 이메일을 편집자로 추가했는지 확인
                        """)
                        return None
                return client
            else:
                st.warning("⚠️ Streamlit secrets에서 GOOGLE_CREDENTIALS_JSON을 찾을 수 없습니다.")
        except Exception as e:
            st.warning(f"⚠️ Streamlit secrets 접근 오류: {str(e)}")
        
        # 방법 2: 새로운 서비스 계정 키 파일 읽기 (대안)
        new_key_file = 'new-service-account-key.json'
        
        if os.path.exists(new_key_file):
            try:
                # JSON 파일에서 직접 읽기
                with open(new_key_file, 'r', encoding='utf-8') as f:
                    credentials_data = json.load(f)
                
                # private_key 형식 검증 및 수정
                if 'private_key' in credentials_data:
                    private_key = credentials_data['private_key']
                    # 개행 문자 정규화
                    if '\\n' in private_key:
                        credentials_data['private_key'] = private_key.replace('\\n', '\n')
                
                # 필수 필드 확인
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in credentials_data]
                if missing_fields:
                    st.error(f"❌ 필수 필드가 누락되었습니다: {missing_fields}")
                    return None
                
                # 인증 정보 생성
                creds = Credentials.from_service_account_info(
                    credentials_data, 
                    scopes=scope
                )
                
                # gspread 클라이언트 생성
                client = gspread.authorize(creds)
                
                if not QUICK_SHEET_CONNECT:
                    try:
                        test_sheet = client.open_by_key("1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4")
                        return client
                    except Exception as test_error:
                        st.error(f"❌ 구글 시트 접근 테스트 실패: {str(test_error)}")
                        st.info("""
                        **구글 시트 접근 권한 확인:**
                        1. 서비스 계정 이메일: 새로운_서비스_계정_이메일@test-92f50.iam.gserviceaccount.com
                        2. 구글 시트 ID: 1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4
                        3. 구글 시트에 서비스 계정 이메일을 편집자로 추가했는지 확인
                        """)
                        return None
                return client
            except Exception as e:
                st.error(f"❌ 새로운 키 파일 인증 오류: {str(e)}")
                return None
            except Exception as e:
                st.error(f"❌ 새로운 키 파일 읽기 오류: {str(e)}")
                return None
        
        # 방법 2: 환경변수에서 JSON 키 읽기 (대안)
        google_credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        
        if google_credentials_json:
            try:
                # JSON 문자열을 딕셔너리로 변환
                credentials_data = json.loads(google_credentials_json)
                
                # private_key 형식 검증 및 수정
                if 'private_key' in credentials_data:
                    private_key = credentials_data['private_key']
                    # 개행 문자 정규화
                    if '\\n' in private_key:
                        credentials_data['private_key'] = private_key.replace('\\n', '\n')
                
                # 필수 필드 확인
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in credentials_data]
                if missing_fields:
                    st.error(f"❌ 필수 필드가 누락되었습니다: {missing_fields}")
                    return None
                
                # 인증 정보 생성
                creds = Credentials.from_service_account_info(
                    credentials_data, 
                    scopes=scope
                )
                
                # gspread 클라이언트 생성
                client = gspread.authorize(creds)
                
                if not QUICK_SHEET_CONNECT:
                    try:
                        test_sheet = client.open_by_key("1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4")
                        return client
                    except Exception as test_error:
                        st.error(f"❌ 구글 시트 접근 테스트 실패: {str(test_error)}")
                        st.info("""
                        **구글 시트 접근 권한 확인:**
                        1. 서비스 계정 이메일: firebase-adminsdk-fbsvc@test-92f50.iam.gserviceaccount.com
                        2. 구글 시트 ID: 1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4
                        3. 서비스 계정이 구글 시트에 편집자 권한으로 공유되어 있는지 확인
                        """)
                        return None
                return client
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON 파싱 오류: {str(e)}")
                st.info("환경변수 GOOGLE_CREDENTIALS_JSON의 형식이 올바른지 확인해주세요.")
                return None
            except Exception as e:
                st.error(f"❌ 인증 정보 생성 오류: {str(e)}")
                st.info("""
                **PEM 파일 오류 해결 방법:**
                1. private_key의 개행 문자 확인
                2. JSON 키 파일이 올바른 형식인지 확인
                3. 서비스 계정 권한 확인
                4. 네트워크 연결 상태 확인
                """)
                return None
        
        # 방법 2: Streamlit secrets에서 JSON 키 읽기 (백업)
        if hasattr(st, 'secrets') and 'GOOGLE_CREDENTIALS_JSON' in st.secrets:
            try:
                st.info("🔍 Streamlit secrets에서 인증 정보를 읽는 중...")
                google_credentials_json = st.secrets['GOOGLE_CREDENTIALS_JSON']
                
                # JSON 문자열을 딕셔너리로 변환
                credentials_data = json.loads(google_credentials_json)
                
                # private_key 형식 검증 및 수정
                if 'private_key' in credentials_data:
                    private_key = credentials_data['private_key']
                    # 개행 문자 정규화
                    if '\\n' in private_key:
                        credentials_data['private_key'] = private_key.replace('\\n', '\n')
                        st.info("✅ private_key 개행 문자 정규화 완료")
                
                # 필수 필드 확인
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in credentials_data]
                if missing_fields:
                    st.error(f"❌ 필수 필드가 누락되었습니다: {missing_fields}")
                    return None
                
                st.info("🔍 인증 정보 생성 중...")
                
                # 인증 정보 생성
                creds = Credentials.from_service_account_info(
                    credentials_data, 
                    scopes=scope
                )
                
                st.info("🔍 gspread 클라이언트 생성 중...")
                
                # gspread 클라이언트 생성
                client = gspread.authorize(creds)
                
                # 연결 테스트
                st.info("🔍 구글 시트 연결 테스트 중...")
                try:
                    # 간단한 테스트로 연결 확인
                    test_sheet = client.open_by_key("1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4")
                    st.success("✅ 구글 시트 연결 성공!")
                    return client
                except Exception as test_error:
                    st.error(f"❌ 구글 시트 접근 테스트 실패: {str(test_error)}")
                    st.info("""
                    **구글 시트 접근 권한 확인:**
                    1. 서비스 계정 이메일: firebase-adminsdk-fbsvc@test-92f50.iam.gserviceaccount.com
                    2. 구글 시트 ID: 1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4
                    3. 서비스 계정이 구글 시트에 편집자 권한으로 공유되어 있는지 확인
                    """)
                    return None
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON 파싱 오류: {str(e)}")
                st.info("Streamlit secrets의 GOOGLE_CREDENTIALS_JSON 형식이 올바른지 확인해주세요.")
                return None
            except Exception as e:
                st.error(f"❌ Streamlit secrets 인증 오류: {str(e)}")
                return None
        
        # 방법 3: JSON 파일에서 직접 읽기
        json_file_path = "test-92f50-a704ebe1984f.json"
        if os.path.exists(json_file_path):
            try:
                st.info(f"🔍 JSON 파일에서 인증 정보를 읽는 중: {json_file_path}")
                
                creds = Credentials.from_service_account_file(
                    json_file_path,
                    scopes=scope
                )
                
                # gspread 클라이언트 생성
                client = gspread.authorize(creds)
                return client
            except Exception as e:
                st.error(f"❌ JSON 파일 인증 오류: {str(e)}")
                st.info(f"파일 경로: {json_file_path}")
                return None
        
        # 방법 4: 기본 인증 정보 사용 (개발용)
        st.warning("⚠️ 환경변수, Streamlit secrets, JSON 파일을 찾을 수 없습니다.")
        st.info("""
        **구글 시트 연결 설정 방법:**
        
        1. **환경변수 설정 (권장):**
           - GOOGLE_CREDENTIALS_JSON 환경변수 설정
        
        2. **Streamlit secrets 설정:**
           - .streamlit/secrets.toml 파일에 GOOGLE_CREDENTIALS_JSON 설정
        
        3. **JSON 파일 사용:**
           - test-92f50-a704ebe1984f.json 파일이 프로젝트 루트에 있는지 확인
        
        4. **서비스 계정 설정:**
           - 구글 클라우드 콘솔에서 서비스 계정 키 생성
           - 구글 시트에 서비스 계정 이메일 공유
        """)
        
        # 개발용 더미 클라이언트 반환 (실제 연결은 안됨)
        return None
            
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 오류: {str(e)}")
        st.info("""
        **일반적인 해결 방법:**
        1. JSON 키 파일이 올바른 형식인지 확인
        2. 서비스 계정이 구글 시트에 접근 권한이 있는지 확인
        3. 네트워크 연결 상태 확인
        4. private_key의 개행 문자 형식 확인
        """)
        return None

def load_data_from_sheet(client, sheet_name="power_data", sheet_id=None):
    """구글 시트에서 데이터 로드 (비캐시 원본)"""
    try:
        # 시트 열기 (ID가 제공된 경우 ID로, 아니면 이름으로)
        if sheet_id and sheet_id.strip():
            sheet = client.open_by_key(sheet_id).sheet1
        else:
            sheet = client.open(sheet_name).sheet1
        
        # 모든 데이터 가져오기
        all_values = sheet.get_all_values()
        
        if len(all_values) == 0:
            st.error("❌ 시트에 데이터가 없습니다.")
            return None
        
        # 첫 번째 행을 헤더로 사용
        headers = all_values[0]
        data_rows = all_values[1:]
        
        # 데이터프레임 생성
        df = pd.DataFrame(data_rows, columns=headers)
        
        # 수치형 컬럼 변환
        numeric_columns = ['최고기온', '평균기온', '최저기온', '최대수요', '체감온도']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 날짜 컬럼을 년월일까지만 표시하도록 변환
        if '날짜' in df.columns:
            try:
                # 날짜 컬럼을 datetime으로 변환 후 년월일까지만 표시
                df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception as e:
                st.warning(f"날짜 변환 중 오류: {e}")
        
        return df
    except Exception as e:
        st.error(f"❌ 시트 데이터 로드 오류: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=300)
def load_data_from_sheet_cached(sheet_name="power_data", sheet_id=None):
    """구글 시트에서 데이터 로드 (캐시) - 5분 TTL"""
    client = setup_google_sheets()
    if client is None:
        return None
    return load_data_from_sheet(client, sheet_name, sheet_id)

def save_data_to_sheet(client, data, sheet_name="power_data", sheet_id=None, original_data=None):
    """구글 시트에 데이터 저장 (변경된 부분만 업데이트)"""
    try:
        # 시트 열기 (ID가 제공된 경우 ID로, 아니면 이름으로)
        if sheet_id and sheet_id.strip():
            sheet = client.open_by_key(sheet_id).sheet1
        else:
            sheet = client.open(sheet_name).sheet1
        
        # 월 컬럼 제거 (내부 계산용이므로 구글 시트에 저장하지 않음)
        data_to_save = data.copy()
        if '월' in data_to_save.columns:
            data_to_save = data_to_save.drop(columns=['월'])
        
        # 원본 데이터도 월 컬럼 제거하여 비교
        original_data_to_compare = None
        if original_data is not None:
            original_data_to_compare = original_data.copy()
            if '월' in original_data_to_compare.columns:
                original_data_to_compare = original_data_to_compare.drop(columns=['월'])
        
        # 원본 데이터가 제공된 경우 변경된 부분만 감지
        if original_data_to_compare is not None:
            # 변경된 행과 열 감지
            changed_rows = []
            changed_columns = []
            
            # 데이터 타입 통일을 위해 문자열로 변환하여 비교
            data_str = data_to_save.astype(str)
            original_str = original_data_to_compare.astype(str)
            
            # 변경된 행 감지
            for idx in range(len(data_to_save)):
                if idx < len(original_str) and not data_str.iloc[idx].equals(original_str.iloc[idx]):
                    changed_rows.append(idx + 2)  # +2는 헤더(1)와 0-based 인덱스(1) 때문
            
            # 변경된 열 감지
            for col in data_to_save.columns:
                if col in original_str.columns and not data_str[col].equals(original_str[col]):
                    changed_columns.append(col)
            
            # 변경된 부분만 업데이트 (최적화된 배치 방식)
            if changed_rows:
                # 변경된 행들을 하나의 연속된 범위로 그룹화
                changed_rows.sort()  # 행 번호 정렬
                
                # 연속된 행들을 그룹으로 나누기
                row_groups = []
                current_group = [changed_rows[0]]
                
                for i in range(1, len(changed_rows)):
                    if changed_rows[i] == changed_rows[i-1] + 1:
                        # 연속된 행
                        current_group.append(changed_rows[i])
                    else:
                        # 불연속된 행 - 새 그룹 시작
                        row_groups.append(current_group)
                        current_group = [changed_rows[i]]
                
                row_groups.append(current_group)  # 마지막 그룹 추가
                
                # 각 그룹을 하나의 범위로 업데이트
                for group in row_groups:
                    start_row = group[0]
                    end_row = group[-1]
                    
                    # 해당 범위의 데이터 추출
                    group_data = data_to_save.iloc[start_row-2:end_row-1]  # -2는 인덱스 조정
                    
                    # 각 행을 문자열로 변환
                    group_values = []
                    for _, row in group_data.iterrows():
                        row_values = []
                        for val in row:
                            if pd.isna(val):
                                row_values.append('')
                            elif isinstance(val, pd.Timestamp):
                                row_values.append(val.strftime('%Y-%m-%d'))
                            elif isinstance(val, str) and 'T' in val:
                                try:
                                    date_obj = pd.to_datetime(val)
                                    row_values.append(date_obj.strftime('%Y-%m-%d'))
                                except:
                                    row_values.append(str(val))
                            else:
                                row_values.append(str(val))
                        group_values.append(row_values)
                    
                    # 범위 업데이트 (연속된 행들을 한 번에)
                    range_name = f'A{start_row}:{chr(65 + len(group_values[0]) - 1)}{end_row}'
                    
                    # 서식 복사: 바로 위 행의 서식을 따라가도록 설정
                    try:
                        if APPLY_SHEET_FORMATTING and start_row > 2:  # 옵션: 서식 적용
                            format_range = f'A{start_row}:{chr(65 + len(group_values[0]) - 1)}{end_row}'
                            sheet.format(format_range, {
                                "textFormat": {
                                    "fontSize": 11,
                                    "fontFamily": "Arial"
                                }
                            })
                    except Exception as e:
                        st.warning(f"⚠️ 서식 적용 실패: {str(e)}")
                    
                    # 데이터 업데이트
                    # 빠른 업데이트(배치) 모드
                    sheet.update(range_name, group_values, value_input_option='RAW')
                
                return True, f"✅ {len(changed_rows)}개 행이 {len(row_groups)}개 그룹으로 업데이트되었습니다."
        
        # 원본 데이터가 없거나 전체 업데이트가 필요한 경우
        # 날짜 컬럼을 년월일까지만 표시하도록 변환
        for col in data_to_save.columns:
            if data_to_save[col].dtype == 'datetime64[ns]':
                data_to_save[col] = data_to_save[col].dt.strftime('%Y-%m-%d')
            elif data_to_save[col].dtype == 'object':
                # 문자열 컬럼에서 날짜 형식인지 확인
                try:
                    # 첫 번째 유효한 값으로 날짜 형식 확인
                    first_valid = data_to_save[col].dropna().iloc[0] if len(data_to_save[col].dropna()) > 0 else None
                    if first_valid and isinstance(first_valid, str) and ('T' in first_valid or '-' in first_valid):
                        # 날짜 형식으로 변환 시도
                        data_to_save[col] = pd.to_datetime(data_to_save[col], errors='coerce').dt.strftime('%Y-%m-%d')
                except:
                    pass  # 변환 실패 시 원본 유지
        
        # 모든 데이터를 한 번에 업데이트 (API 호출 최소화)
        all_values = [data_to_save.columns.tolist()]  # 헤더
        for _, row in data_to_save.iterrows():
            # 각 값을 문자열로 변환
            row_values = [str(val) if val is not None else '' for val in row.tolist()]
            all_values.append(row_values)
        
        # 시트를 한 번에 업데이트
        sheet.clear()
        sheet.update('A1', all_values, value_input_option='RAW')
        
        return True, "✅ 전체 데이터가 업데이트되었습니다."
        
    except Exception as e:
        st.error(f"❌ 시트 데이터 저장 오류: {str(e)}")
        return False, f"❌ 저장 실패: {str(e)}"

# 사이드바 제거됨 (요청에 따라 비표시)

# --- 0. 데이터 로딩 및 편집 ---
st.header("📁 Step 0: 데이터 로딩 및 편집")

# 구글 시트 설정
st.subheader("🔐 구글 시트 연결 설정")

client = setup_google_sheets()
if client is None:
    st.warning("⚠️ 구글 시트 연결이 일시적으로 지연됩니다. 캐시된 데이터를 불러옵니다.")

# 구글 시트 설정 정보
sheet_name = "시트1"
sheet_id = "1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4"

# 데이터 자동 로딩 (캐시 우선)
if 'data' not in st.session_state:
    with st.spinner("데이터 로딩 중..."):
        data = load_data_from_sheet_cached(sheet_name, sheet_id)
        if data is None and client is not None:
            data = load_data_from_sheet(client, sheet_name, sheet_id)
        if data is not None:
            st.session_state.data = data
            st.session_state.original_data = data.copy()
        else:
            st.error("❌ 데이터 로딩에 실패했습니다.")
            st.stop()

# 데이터가 로드되었는지 확인
if 'data' not in st.session_state:
    st.error("❌ 데이터를 로드할 수 없습니다.")
    st.stop()

data = st.session_state.data

# 데이터 편집 기능
st.subheader("📊 데이터 미리보기 및 편집")

# 데이터 정보 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 행 수", f"{len(data):,}개")
with col2:
    st.metric("총 컬럼 수", f"{len(data.columns)}개")
with col3:
    # 날짜 컬럼이 있으면 년월일까지만 표시
    if '날짜' in data.columns:
        try:
            # 날짜 컬럼을 datetime으로 변환
            date_data = pd.to_datetime(data['날짜'], errors='coerce')
            start_date = date_data.min().strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e:
            st.warning(f"날짜 변환 오류: {e}")
            start_date = "N/A"
    else:
        start_date = "N/A"
    st.metric("시작일", start_date)
with col4:
    if '날짜' in data.columns:
        try:
            date_data = pd.to_datetime(data['날짜'], errors='coerce')
            end_date = date_data.max().strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e:
            st.warning(f"날짜 변환 오류: {e}")
            end_date = "N/A"
    else:
        end_date = "N/A"
    st.metric("종료일", end_date)

# 데이터 편집 탭
tab1, tab2, tab3 = st.tabs(["📊 데이터 미리보기", "✏️ 데이터 편집", "📈 통계 정보"])

with tab1:
    st.subheader("전체 데이터 미리보기")
    
    # 날짜 컬럼이 있으면 년월일까지만 표시하도록 변환
    display_data = data.copy()
    if '날짜' in display_data.columns:
        try:
            # datetime으로 변환 후 년월일까지만 표시
            display_data['날짜'] = pd.to_datetime(display_data['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception as e:
            st.warning(f"날짜 표시 변환 중 오류: {e}")
    
    st.dataframe(display_data, use_container_width=True)
    
    # 데이터 다운로드
    csv = data.to_csv(index=False)
    st.download_button(
        label="📥 데이터를 CSV로 다운로드",
        data=csv,
        file_name="power_data_edited.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("데이터 편집")
    st.info("아래에서 데이터를 직접 편집하거나 구글시트에서 편집할 수 있습니다. 편집 후 '변경사항 적용' 버튼을 클릭하세요.")
    # 편집용 데이터 준비 (세션 상태 사용하여 안정성 확보)
    if 'edit_data' not in st.session_state:
        # 처음 로드할 때만 편집용 데이터 준비
        edit_data = data.copy()
        
        # 월 컬럼이 있으면 제거 (내부 계산용이므로 편집 불가)
        if '월' in edit_data.columns:
            edit_data = edit_data.drop(columns=['월'])
        
        if '날짜' in edit_data.columns:
            try:
                # datetime으로 변환 후 년월일까지만 표시
                edit_data['날짜'] = pd.to_datetime(edit_data['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception as e:
                st.warning(f"날짜 편집 변환 중 오류: {e}")
        
        # 세션 상태에 저장
        st.session_state.edit_data = edit_data
    else:
        # 세션 상태에서 편집용 데이터 가져오기
        edit_data = st.session_state.edit_data
    
    # 편집 가능한 데이터프레임
    edited_data = st.data_editor(
        edit_data,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    
    # 변경사항 적용 버튼
    if st.button("✅ 변경사항 적용", type="primary"):
        with st.spinner("구글 시트에 저장 중... (변경된 부분만 업데이트)"):
            # 편집된 데이터를 전역 변수에 반영
            data = edited_data.copy()
            
            # 날짜 컬럼을 datetime으로 변환 (편집 시 문자열로 표시되었으므로)
            if '날짜' in data.columns:
                try:
                    data['날짜'] = pd.to_datetime(data['날짜'], errors='coerce')
                except Exception as e:
                    st.warning(f"날짜 변환 중 오류: {e}")
            
            # 월 컬럼 다시 추가 (내부 계산용)
            if '날짜' in data.columns:
                try:
                    # 날짜에서 월 추출하여 월 컬럼 추가
                    data['월'] = pd.to_datetime(data['날짜']).dt.month
                except Exception as e:
                    st.warning(f"월 컬럼 계산 중 오류: {e}")
            
            st.session_state.data = data
            
            # 편집용 데이터 세션 상태 초기화 (다음 편집을 위해)
            if 'edit_data' in st.session_state:
                del st.session_state.edit_data
            
            # 원본 데이터 가져오기 (세션에 저장된 원본 데이터)
            original_data = st.session_state.get('original_data', None)
            
            # 구글 시트에 저장 (변경된 부분만 업데이트)
            success, message = save_data_to_sheet(client, data, sheet_name, sheet_id, original_data)
            
            if success:
                st.success(message)
                
                # 원본 데이터 업데이트 (다음 편집을 위해)
                st.session_state.original_data = data.copy()
                
                # 페이지 새로고침을 위한 세션 상태 업데이트
                st.session_state.data_updated = True
                st.rerun()
            else:
                st.error("❌ 구글 시트 업데이트에 실패했습니다.")
                st.info("💡 API 한도 초과로 인한 오류일 수 있습니다. 잠시 후 다시 시도해주세요.")
        
        # 업데이트된 데이터 다운로드
        csv_updated = data.to_csv(index=False)
        st.download_button(
            label="📥 업데이트된 데이터 다운로드",
            data=csv_updated,
            file_name="power_data_updated.csv",
            mime="text/csv"
        )
        
        # 엑셀 파일로 저장 옵션
        st.subheader("💾 엑셀 파일 저장")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 엑셀 파일로 저장", type="secondary"):
                try:
                    # 임시 엑셀 파일 생성
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        data.to_excel(writer, index=False, sheet_name='Power_Data')
                    
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📥 엑셀 파일 다운로드",
                        data=excel_buffer.getvalue(),
                        file_name="power_data_updated.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("✅ 엑셀 파일이 준비되었습니다!")
                except Exception as e:
                    st.error(f"❌ 엑셀 파일 생성 오류: {str(e)}")
        
        with col2:
            st.success("✅ 구글 시트가 자동으로 업데이트됩니다!")
    
    # 원본 구글시트 주소
    st.subheader("📊 원본 구글시트")
    st.markdown("[🔗 전력 수요 예측 데이터 바로 가기](https://docs.google.com/spreadsheets/d/1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4/edit?gid=0#gid=0)")

with tab3:
    st.subheader("데이터 통계 정보")
    
    # 수치형 데이터 통계
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**수치형 데이터 통계:**")
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)
    
    # 범주형 데이터 통계
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write("**범주형 데이터 통계:**")
        for col in categorical_cols:
            st.write(f"**{col}:**")
            value_counts = data[col].value_counts()
            st.dataframe(value_counts, use_container_width=True)
    
    # 결측값 정보
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.write("**결측값 정보:**")
        st.dataframe(missing_data[missing_data > 0], use_container_width=True)
    else:
        st.success("✅ 결측값이 없습니다!")

st.markdown("---")

# --- 1. 데이터 준비 ---
st.header("📋 Step 1: 데이터 준비")
with st.spinner("데이터를 전처리 중..."):
    # 날짜 컬럼 변환
    if '날짜' in data.columns:
        data['날짜'] = pd.to_datetime(data['날짜'])
    else:
        st.error("❌ '날짜' 컬럼이 없습니다. 데이터를 확인해주세요.")
        st.stop()
    
    # 요일/평일 파생 (없으면 생성)
    try:
        weekday_map = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
        if '요일' not in data.columns:
            data['요일'] = data['날짜'].dt.weekday.map(weekday_map)
        if '평일' not in data.columns:
            data['평일'] = np.where(data['날짜'].dt.weekday < 5, '평일', '주말')
    except Exception:
        pass
    
    # 필수 컬럼 확인 (유연하게 처리)
    required_columns = ['최고기온', '평균기온', '최저기온', '최대수요', '평일', '체감온도']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.warning(f"⚠️ 일부 컬럼이 누락되었습니다: {missing_columns}")
        st.info("누락된 컬럼이 있어도 가능한 기능만 제공됩니다.")
        
        # 최소한의 필수 컬럼만 확인 (최저수요 제외)
        essential_columns = ['날짜', '최대수요']
        essential_missing = [col for col in essential_columns if col not in data.columns]
        
        if essential_missing:
            st.error(f"❌ 핵심 컬럼이 누락되었습니다: {essential_missing}")
            st.info("최소한 날짜, 최대수요 컬럼은 필요합니다.")
            st.stop()
    
    # 가스수요 데이터 확인
    gas_columns = ['가스수요', '태양광최대']
    available_gas_columns = [col for col in gas_columns if col in data.columns]
    
    if available_gas_columns:
        st.success(f"✅ 가스수요 예측 가능: {', '.join(available_gas_columns)} 컬럼 발견")
        if len(available_gas_columns) == 2:
            st.info("🔥 가스수요 예측 모델 학습 가능")
        else:
            st.warning(f"⚠️ 가스수요 예측을 위해 추가 컬럼 필요: {[col for col in gas_columns if col not in available_gas_columns]}")
    else:
        st.info("ℹ️ 가스수요 예측을 위한 컬럼이 없습니다 (가스수요, 태양광최대)")
    
    # 데이터 정보 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 데이터 수", f"{len(data):,}개")
    with col2:
        st.metric("시작일", data['날짜'].min().strftime('%Y-%m-%d'))
    with col3:
        st.metric("종료일", data['날짜'].max().strftime('%Y-%m-%d'))

st.markdown("---")

# --- 2. 특징 공학 및 데이터 정제 ---
st.header("🔧 Step 2: 특징 공학 및 데이터 정제")
with st.spinner("특징 공학을 수행 중..."):
    data['월'] = data['날짜'].dt.month
    data['일'] = data['날짜'].dt.day
    data['연도'] = data['날짜'].dt.year
    # 공휴일 플래그는 사용하지 않음 (평일 컬럼 '평일/휴일'에 통합)
    
    # 요일 더미 생성, 평일은 수동 이진 플래그로 처리(값: '평일' 또는 '휴일')
    data_processed = pd.get_dummies(data, columns=['요일'], drop_first=True)
    try:
        if '평일' in data.columns:
            data_processed['평일_평일'] = (data['평일'].astype(str) == '평일').astype(int)
        else:
            # 백업: 요일로 추정 (주말이면 0)
            data_processed['평일_평일'] = (data['날짜'].dt.weekday < 5).astype(int)
    except Exception:
        data_processed['평일_평일'] = 0
    # 어제 수요 래그
    data_processed['어제의_최대수요'] = data_processed['최대수요'].shift(1)
    
    # 계절별 온도 특징 생성 (유연하게 처리)
    try:
        is_summer_mask = data_processed['월'].isin([5, 6, 7, 8, 9])
        is_winter_mask = data_processed['월'].isin([10, 11, 12, 1, 2, 3, 4])
        
        # 개선된 계절별 온도 특징: 냉방강도/난방강도
        st.subheader("🌡️ 계절별 온도 특징 최적화")
        cooling_base_temp = 25.0
        heating_base_temp = 10.0
        # 냉방강도: 체감온도(있으면) 없으면 최고기온 기준으로 계산
        if '체감온도' in data_processed.columns:
            temp_for_cooling = data_processed['체감온도']
        elif '최고기온' in data_processed.columns:
            temp_for_cooling = data_processed['최고기온']
        else:
            st.error("❌ 온도 관련 컬럼이 부족합니다. 최소한 최고기온 또는 체감온도가 필요합니다.")
            st.stop()
        data_processed['냉방강도'] = (pd.to_numeric(temp_for_cooling, errors='coerce') - cooling_base_temp).clip(lower=0)
        
        # 난방강도: 최저기온 기준으로 계산
        if '최저기온' in data_processed.columns:
            temp_for_heating = pd.to_numeric(data_processed['최저기온'], errors='coerce')
            data_processed['난방강도'] = (heating_base_temp - temp_for_heating).clip(lower=0)
        else:
            st.warning("⚠️ 최저기온이 없어 난방강도는 0으로 대체됩니다.")
            data_processed['난방강도'] = 0.0
        st.success("✅ 냉방/난방 강도를 반영한 계절별 온도 특징 생성 완료!")
        # 계절 마스크 적용: 여름엔 냉방강도만, 겨울엔 난방강도만 사용
        data_processed['냉방강도'] = data_processed['냉방강도'] * is_summer_mask.astype(int)
        data_processed['난방강도'] = data_processed['난방강도'] * is_winter_mask.astype(int)

        # 추가 온도 파생: 일교차(최고-최저) (가능할 때)
        try:
            if '최고기온' in data_processed.columns and '최저기온' in data_processed.columns:
                data_processed['일교차'] = pd.to_numeric(data_processed['최고기온'], errors='coerce') - pd.to_numeric(data_processed['최저기온'], errors='coerce')
        except Exception:
            pass

        # 이동평균(누수 방지: shift(1) 후 rolling)
        try:
            data_processed['7일평균_최대수요'] = data_processed['최대수요'].shift(1).rolling(window=7, min_periods=1).mean()
            data_processed['14일평균_최대수요'] = data_processed['최대수요'].shift(1).rolling(window=14, min_periods=1).mean()
            data_processed['전주동일요일_최대수요'] = data_processed['최대수요'].shift(7)
        except Exception:
            pass

        # 최신 관측 기반 동적 입력 기본값 저장 (예측 시 사용)
        try:
            st.session_state.dynamic_max_features = {}
            # 재귀 예측을 위한 최근 14일 타깃 시계열 저장
            try:
                st.session_state.max_series_tail = list(pd.to_numeric(data_processed['최대수요'], errors='coerce').dropna().tail(14).values)
            except Exception:
                st.session_state.max_series_tail = []
        except Exception:
            st.session_state.dynamic_max_features = {}
            st.session_state.max_series_tail = []
            
    except Exception as e:
        st.error(f"❌ 온도 특징 생성 중 오류: {e}")
        st.stop()
    
    # 가스수요 특징 공학
    if '가스수요' in data_processed.columns and '태양광최대' in data_processed.columns:
        # 가스수요 데이터를 숫자로 변환
        data_processed['가스수요'] = pd.to_numeric(data_processed['가스수요'], errors='coerce')
        data_processed['태양광최대'] = pd.to_numeric(data_processed['태양광최대'], errors='coerce')
        # 잔여부하(최대수요 - 태양광최대)
        if '최대수요' in data_processed.columns:
            try:
                data_processed['잔여부하'] = pd.to_numeric(data_processed['최대수요'], errors='coerce') - data_processed['태양광최대']
            except Exception:
                pass
            # 최대수요 대비 비율 특징들 (가스 제외, 누설 방지)
            try:
                denom = data_processed['최대수요'].replace(0, np.nan)
                data_processed['최대수요대비_태양광비율'] = (data_processed['태양광최대'] / denom).fillna(0.0)
                data_processed['최대수요대비_잔여부하비율'] = (data_processed['잔여부하'] / denom).fillna(0.0)
            except Exception:
                pass

        # (가스+태양광)/최대수요 총비율의 평일/주말 평균을 계산하여 예산형 가스 기준치 생성
        try:
            denom_total = data_processed['최대수요'].replace(0, np.nan)
            total_ratio = (data_processed['가스수요'] + data_processed['태양광최대']) / denom_total
            # 평일 플래그 파생 (원-핫이 있으면 사용, 없으면 원본에서 유도)
            if '평일_평일' in data_processed.columns:
                is_weekday_series = data_processed['평일_평일']
            else:
                is_weekday_series = (data['평일'] == '평일').astype(int) if '평일' in data.columns else pd.Series(0, index=data_processed.index)

            weekday_mean = total_ratio[is_weekday_series == 1].mean()
            weekend_mean = total_ratio[is_weekday_series == 0].mean()
            global_mean = total_ratio.mean()

            if pd.isna(weekday_mean):
                weekday_mean = global_mean
            if pd.isna(weekend_mean):
                weekend_mean = global_mean

            # 세션에 저장 (예측 시 사용)
            st.session_state.gas_total_ratio_weekday = float(weekday_mean) if not pd.isna(weekday_mean) else 0.0
            st.session_state.gas_total_ratio_weekend = float(weekend_mean) if not pd.isna(weekend_mean) else 0.0

            # 행별 예산 비율 선택 후 목표 가스량 계산: max*ratio - solar
            ratio_used = np.where(is_weekday_series == 1, st.session_state.gas_total_ratio_weekday, st.session_state.gas_total_ratio_weekend)
            data_processed['목표가스_예산'] = (data_processed['최대수요'] * ratio_used - data_processed['태양광최대']).clip(lower=0)
        except Exception:
            # 실패 시 컬럼 미생성
            pass
        
        # 결측값 제거 후 특징 공학
        gas_data_clean = data_processed[['가스수요', '태양광최대']].dropna()
        if len(gas_data_clean) > 0:
            data_processed['어제의_가스수요'] = data_processed['가스수요'].shift(1)
            # 누설 방지: 변화율은 t시점이 아니라 (t-1,t-2)로 계산
            data_processed['어제의_가스수요_변화율'] = data_processed['가스수요'].pct_change().shift(1)
            # 예측 시 사용하기 위한 최신 관측 래그 보관
            try:
                last_two_gas = pd.to_numeric(gas_data_clean['가스수요'], errors='coerce').dropna().tail(2).values
                if len(last_two_gas) >= 1:
                    st.session_state.last_gas = float(last_two_gas[-1])
                if len(last_two_gas) == 2:
                    st.session_state.prev_gas = float(last_two_gas[0])
            except Exception:
                pass
            st.success("✅ 전력수요 및 가스수요 데이터 정제 완료!")
        else:
            st.warning("⚠️ 가스수요 데이터가 숫자로 변환되지 않습니다.")
            st.success("✅ 전력수요 데이터 정제 완료!")
    else:
        st.success("✅ 전력수요 데이터 정제 완료!")
    
    # 핵심 학습 피처 위주로 결측 제거 (불필요한 전체 드랍 방지)
    essential_cols = ['최대수요','태양광최대','최고기온','평균기온','체감온도','월','어제의_최대수요','7일평균_최대수요','14일평균_최대수요','전주동일요일_최대수요']
    essential_cols = [c for c in essential_cols if c in data_processed.columns]
    data_processed.dropna(subset=essential_cols, inplace=True)
    
    # 처리된 데이터 정보
    col1, col2 = st.columns(2)
    with col1:
        st.metric("정제 후 데이터 수", f"{len(data_processed):,}개")
    with col2:
        st.metric("특징 변수 수", f"{len(data_processed.columns)}개")
    
    # 처리된 데이터 미리보기
    with st.expander("🔍 처리된 데이터 미리보기"):
        st.dataframe(data_processed.head(10), use_container_width=True)

st.markdown("---")

# --- 3. 모델 변수 및 데이터 분리 ---
st.header("🎯 Step 3: 모델 변수 및 데이터 분리")

# 평균기온을 모델 특징에 사용하지 않음 (향후 필요 시 True로 변경할 수 있도록 변수만 유지)
include_avg_temp_feature = False

# [최대수요 모델] (여름철에는 체감온도 사용)
_base_max = ['냉방강도', '난방강도', '월', '어제의_최대수요', '7일평균_최대수요', '14일평균_최대수요', '전주동일요일_최대수요']
if include_avg_temp_feature:
    _base_max.insert(1, '평균기온')

# 추가로 최저/최고/체감/일교차까지 함께 사용 (존재하는 경우만)
_temp_extras = [f for f in ['최고기온', '최저기온', '체감온도', '일교차'] if f in data_processed.columns]

# 요일, 평일 더미 포함
_dummies = [col for col in data_processed if col.startswith('요일_') or col.startswith('평일_')]

features_max = _base_max + _temp_extras + _dummies
X_max = data_processed[features_max]
y_max = data_processed['최대수요']

# 최저수요 모델 제거 - 최대수요 모델만 사용

# 고정된 파라미터 사용
test_size = 0.2
n_estimators = 100
random_state = 42

# 단일 모델용 데이터 분할 - 시간순 분할 (평일/주말 분리 제거)
X_max_train, X_max_test, y_max_train, y_max_test = chronological_split(
    X_max, y_max, data_processed['날짜'], test_size=test_size
)

# 변수 정보 표시
st.subheader("📈 최대수요 모델 변수")
st.write(f"특징 변수: {len(features_max)}개")
# 표시용 이름 매핑: '냉방강도'/'난방강도'를 직관적으로 표시
_display_name_map = {
    '냉방강도': '냉방강도(>25°C)',
    '난방강도': '난방강도(<10°C)',
}
display_features_max = [_display_name_map.get(name, name) for name in features_max]
# 헤더 행을 사용한 한 줄 표
max_vars_df = pd.DataFrame([display_features_max], columns=[f'변수{i+1}' for i in range(len(display_features_max))])
st.dataframe(max_vars_df, use_container_width=True)



# 가스수요 모델 변수 (가능한 경우)
if '가스수요' in data_processed.columns and '태양광최대' in data_processed.columns:
    st.subheader("🔥 가스수요 모델 변수")
    # 최소·핵심 피처 위주 구성 (다중공선성/누설 위험 낮춤)
    features_gas = [
        '최대수요',          # 총 스케일
        '태양광최대',        # 대체관계 핵심
        '잔여부하',          # 잔여 총량
        '최대수요대비_태양광비율',
        '최대수요대비_잔여부하비율',
        '목표가스_예산',      # 평일/주말 총비율 예산
        '어제의_가스수요',     # 래그
        '어제의_가스수요_변화율',# 래그 변화율(누설 방지)
        '평일_평일'          # 평일/주말 효과
    ]
    available_gas_features = [col for col in features_gas if col in data_processed.columns]
    
    if len(available_gas_features) >= 2:  # 최소 2개 변수 필요
        X_gas = data_processed[available_gas_features]
        y_gas = data_processed['가스수요']
        
        # 가스수요 데이터 분할 - 시간순 분할
        X_gas_train, X_gas_test, y_gas_train, y_gas_test = chronological_split(
            X_gas, y_gas, data_processed.loc[X_gas.index, '날짜'], test_size=test_size
        )
        
        st.write(f"특징 변수: {len(available_gas_features)}개")
        gas_vars_df = pd.DataFrame([available_gas_features], columns=[f'변수{i+1}' for i in range(len(available_gas_features))])
        st.dataframe(gas_vars_df, use_container_width=True)
        
        # 세션 상태에 저장 (단일 전체 세트)
        st.session_state.X_gas_train = X_gas_train
        st.session_state.X_gas_test = X_gas_test
        st.session_state.y_gas_train = y_gas_train
        st.session_state.y_gas_test = y_gas_test
        st.session_state.features_gas = available_gas_features

        # 평일/주말 분리 세트 생성
        if '평일_평일' in data_processed.columns:
            try:
                mask_weekday = data_processed['평일_평일'] == 1
            except Exception:
                mask_weekday = pd.Series(False, index=data_processed.index)
        else:
            # 원본 '평일'에서 유도
            mask_weekday = (data['평일'] == '평일') if '평일' in data.columns else pd.Series(False, index=data_processed.index)

        try:
            X_gas_wd = X_gas[mask_weekday]
            y_gas_wd = y_gas[mask_weekday]
            X_gas_we = X_gas[~mask_weekday]
            y_gas_we = y_gas[~mask_weekday]

            # 최소 표본 확인 후 분할
            if len(X_gas_wd) >= 20 and len(X_gas_we) >= 20:
                X_gas_wd_tr, X_gas_wd_te, y_gas_wd_tr, y_gas_wd_te = chronological_split(
                    X_gas_wd, y_gas_wd, data_processed.loc[X_gas_wd.index, '날짜'], test_size=test_size
                )
                X_gas_we_tr, X_gas_we_te, y_gas_we_tr, y_gas_we_te = chronological_split(
                    X_gas_we, y_gas_we, data_processed.loc[X_gas_we.index, '날짜'], test_size=test_size
                )

                st.session_state.X_gas_train_weekday = X_gas_wd_tr
                st.session_state.X_gas_test_weekday = X_gas_wd_te
                st.session_state.y_gas_train_weekday = y_gas_wd_tr
                st.session_state.y_gas_test_weekday = y_gas_wd_te

                st.session_state.X_gas_train_weekend = X_gas_we_tr
                st.session_state.X_gas_test_weekend = X_gas_we_te
                st.session_state.y_gas_train_weekend = y_gas_we_tr
                st.session_state.y_gas_test_weekend = y_gas_we_te
            else:
                st.warning("⚠️ 평일/주말 분리 학습을 위한 표본 수가 부족합니다. 단일 모델로 학습합니다.")
        except Exception:
            st.warning("⚠️ 평일/주말 분리 데이터 생성 중 오류가 발생하여 단일 모델로 진행합니다.")
    else:
        st.warning("⚠️ 가스수요 예측을 위한 충분한 변수가 없습니다.")
else:
    st.info("ℹ️ 가스수요 예측을 위한 컬럼이 없습니다.")

st.markdown("---")

# --- 4. 모델 학습 (단일 모델) ---
st.header("🤖 Step 4: 모델 학습")
with st.spinner("모델을 학습 중..."):
    st.subheader("📈 단일 모델 학습")
    # 단일 모델 학습 (간단 튜닝 적용)
    try:
        rf_max = tune_rf_model(X_max_train, y_max_train, random_state=random_state)
    except Exception:
        rf_max = train_rf_model(X_max_train, y_max_train, n_estimators=n_estimators, random_state=random_state)
    
    # 가스수요 모델 학습 (단일 모델로 고정)
    if hasattr(st.session_state, 'features_gas'):
        features_for_constraints = st.session_state.features_gas
        constraint_map = {
            '최대수요': 1,
            '태양광최대': -1,
            '잔여부하': 1,
            '최대수요대비_태양광비율': -1,
            '최대수요대비_잔여부하비율': 1,
            '목표가스_예산': 1,
            '어제의_가스수요': 0,
            '어제의_가스수요_변화율': 0,
            '평일_평일': 0,
        }
        monotone_constraints = [constraint_map.get(f, 0) for f in features_for_constraints]

        # 단일 모델로 학습
        st.session_state.gas_model = train_lgbm_gas_model(
            st.session_state.X_gas_train,
            st.session_state.y_gas_train,
            monotone_constraints=monotone_constraints,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=random_state,
        )
        st.success("✅ 전력수요 및 가스수요 모델 학습 완료! (단일)")
    else:
        st.success("✅ 전력수요 모델 학습 완료!")

st.markdown("---")

# --- 5. 모델 성능 평가 ---
st.header("📊 Step 5: 모델 성능 평가")
with st.spinner("성능을 평가 중..."):
    st.subheader("📈 단일 모델 성능 (검증 세트)")
    y_pred = rf_max.predict(X_max_test)
    st.session_state.mae_max = mean_absolute_error(y_max_test, y_pred)
    st.session_state.r2_max = r2_score(y_max_test, y_pred)
    
    # 가스수요 단일 모델 성능 평가
    if hasattr(st.session_state, 'gas_model') and hasattr(st.session_state, 'X_gas_test'):
        y_gas_pred = st.session_state.gas_model.predict(st.session_state.X_gas_test)
        st.session_state.mae_gas = mean_absolute_error(st.session_state.y_gas_test, y_gas_pred)
        st.session_state.r2_gas = r2_score(st.session_state.y_gas_test, y_gas_pred)

# 성능 결과 표시 (최대수요 / 가스수요 나란히)
if hasattr(st.session_state, 'mae_gas') and hasattr(st.session_state, 'r2_gas'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 최대수요 예측 모델 성능")
        st.metric("검증 MAE", f"{st.session_state.mae_max:,.0f} MW")
        st.metric("검증 R²", f"{st.session_state.r2_max:.4f}")
    with col2:
        st.subheader("🔥 가스수요 예측 모델 성능")
        st.metric("검증 MAE", f"{st.session_state.mae_gas:,.0f} MW")
        st.metric("검증 R²", f"{st.session_state.r2_gas:.4f}")
else:
    st.subheader("📈 최대수요 예측 모델 성능")
    st.metric("검증 MAE", f"{st.session_state.mae_max:,.0f} MW")
    st.metric("검증 R²", f"{st.session_state.r2_max:.4f}")

# 그래프 비표시(요청에 따라 검증 라인차트 생략)

st.markdown("---")

# --- 6. 전력 수요 예측 ---
st.header("🔮 Step 6: 전력 수요 예측")
st.info("요일과 체감온도를 입력하고 예측 기간을 선택하여 최대수요를 예측합니다.")

# 예측 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 예측 조건 입력")
    
    # 요일 선택
    weekday_options = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    selected_weekday = st.selectbox("요일 선택", weekday_options, index=0)
    
    # 체감온도 입력 (여름/겨울 적용)
    feels_like_simple = st.number_input("체감온도 (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
    
    # 월 선택 (계절성 고려)
    month_options = list(range(1, 13))
    selected_month = st.selectbox("월 선택", month_options, index=4)  # 5월 기본값
    
    # 예측 기간은 1일로 고정
    horizon = 1
    # 예측 버튼
    predict_button = st.button("🔮 예측 실행", type="primary")

with col2:
    st.subheader("📊 입력 정보")
    st.write(f"**선택된 요일:** {selected_weekday}")
    st.write(f"**체감온도:** {feels_like_simple}°C")
    st.write(f"**선택된 월:** {selected_month}월")

# 예측 실행
if predict_button:
    try:
        with st.spinner("예측을 수행 중..."):
            # 평일/주말 + 요일 더미 구성 (예측 입력)
            weekday_dummies = {}
            # 평일/휴일 판단: 요일 기반(월~금=평일). 추후 UI로 직접 선택 가능
            is_weekday = 1 if selected_weekday in ['월요일','화요일','수요일','목요일','금요일'] else 0
            weekday_dummies['평일_평일'] = is_weekday
            weekday_dummies.update({f'요일_{w}': (1 if w == selected_weekday else 0) for w in ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']})
            
            # 계절 판별
            is_summer_sel = selected_month in [5, 6, 7, 8, 9]
            is_winter_sel = selected_month in [10, 11, 12, 1, 2, 3, 4]

            # 최대수요 예측을 위한 특징 생성 (여름: 체감온도, 그 외: 체감온도 대용)
            est_high = feels_like_simple
            # 모델 선택(단일 모델)
            model_max = rf_max
            
            def predict_one_day(feels_like_val: float, month_val: int, weekday_name: str, min_temp_val=None, max_temp_val=None) -> float:
                dummies = {'평일_평일': 1 if weekday_name in ['월요일','화요일','수요일','목요일','금요일'] else 0}
                dummies.update({f'요일_{w}': (1 if w == weekday_name else 0) for w in ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']})
                feats = {
                    # 냉방강도/난방강도 계산
                    '냉방강도': max(0.0, (feels_like_val if max_temp_val is None else max_temp_val) - 25.0),
                    '난방강도': max(0.0, 10.0 - (min_temp_val if min_temp_val is not None else feels_like_val)),
                    '월': month_val,
                    # 계절별 추가 온도: 여름=최고기온, 겨울=최저기온
                    '최고기온': (max_temp_val if max_temp_val is not None else feels_like_val) if month_val in [5,6,7,8,9] else 0.0,
                    '최저기온': (min_temp_val if min_temp_val is not None else feels_like_val) if month_val in [10,11,12,1,2,3,4] else 0.0,
                    '체감온도': feels_like_val,
                    '일교차': ( (max_temp_val - min_temp_val) if (max_temp_val is not None and min_temp_val is not None) else 0.0 ),
                }
                feats.update(dummies)
                frame = pd.DataFrame([feats])
                # 훈련 피처 집합과 정렬/보정
                frame = align_features_for_model(model_max, frame)
                return float(model_max.predict(frame)[0])

            # 단일일 예측 또는 재귀 7일 예측
            if horizon == 1:
                predicted_max = predict_one_day(feels_like_simple, selected_month, selected_weekday)
                forecast_series = [predicted_max]
            else:
                # 7일 재귀 예측: 매 스텝에서 래그/평균 업데이트
                weekday_cycle = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
                start_idx = weekday_cycle.index(selected_weekday)
                # 시드 시계열 준비
                tail = st.session_state.get('max_series_tail', [])
                buf = list(tail)
                if len(buf) < 14:
                    buf = ([buf[0]] * (14 - len(buf)) + buf) if buf else [0.0]*14
                forecast_series = []
                dyn_work = dyn.copy()
                for step in range(7):
                    wd_name = weekday_cycle[(start_idx + step) % 7]
                    y_hat = predict_one_day(feels_like_simple, selected_month, wd_name, dyn_work)
                    forecast_series.append(y_hat)
                    # 버퍼 업데이트 (최대 14개 유지)
                    buf.append(y_hat)
                    if len(buf) > 14:
                        buf.pop(0)
                    # 동적 특징 업데이트
                    dyn_work['어제의_최대수요'] = buf[-1]
                    dyn_work['7일평균_최대수요'] = float(pd.Series(buf[-7:]).mean())
                    dyn_work['14일평균_최대수요'] = float(pd.Series(buf[-14:]).mean())
                    dyn_work['전주동일요일_최대수요'] = buf[-7] if len(buf) >= 7 else dyn_work.get('전주동일요일_최대수요', 0.0)
                predicted_max = forecast_series[0]
            
            st.success("✅ 예측 완료!")
            
            # 예측 결과 표시
            st.subheader("🎯 예측 결과")
            
            if horizon == 1:
                st.metric("예측 최대수요", f"{predicted_max:,.0f} MW")
            else:
                st.write("**예측 최대수요(7일):**")
                st.dataframe(pd.DataFrame({
                    '일차': list(range(1, len(forecast_series)+1)),
                    '예측 최대수요': [f"{v:,.0f}" for v in forecast_series]
                }), use_container_width=True)
            
            # 예측 결과 상세 정보
            st.subheader("📋 예측 상세 정보")
            base_items = ['요일', '체감온도', '월']
            base_vals = [selected_weekday, f"{feels_like_simple}°C", f"{selected_month}월"]
            if horizon == 1:
                base_items.append('예측 최대수요')
                base_vals.append(f"{predicted_max:,.0f} MW")
            else:
                base_items.append('예측 기간')
                base_vals.append(f"{horizon}일")
            prediction_info = pd.DataFrame({'항목': base_items, '값': base_vals})
            st.dataframe(prediction_info, use_container_width=True)
            
            # 예측 신뢰도 (모델 성능 기반)
            confidence_max = min(95, max(60, st.session_state.r2_max * 100))
            
            st.subheader("📊 예측 신뢰도")
            st.metric("최대수요 예측 신뢰도", f"{confidence_max:.1f}%")
            
            # 예측 결과 시각화
            st.subheader("📈 예측 결과 시각화")
            
            fig_prediction = go.Figure()
            
            # 최대수요만 막대 그래프로 표시
            fig_prediction.add_trace(go.Bar(
                x=['최대수요'],
                y=[predicted_max],
                name='예측 최대수요',
                marker_color=['red']
            ))
            
            fig_prediction.update_layout(
                title=f"{selected_weekday} (체감 {feels_like_simple}°C) 최대수요 예측",
                yaxis_title="전력 수요 (MW)",
                showlegend=True
            )
            
            st.plotly_chart(fig_prediction, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ 예측 중 오류가 발생했습니다: {str(e)}")
        st.info("모델 학습이 완료되지 않았거나 입력 데이터에 문제가 있을 수 있습니다.")

# --- 새로운 예측 기능: 최저기온/최고기온 입력 ---
st.markdown("---")
st.subheader("🌡️ 상세 기온 기반 예측")
st.info("최저기온, 최고기온, 체감온도를 직접 입력하여 더 정확한 최대수요 예측을 수행합니다.")

# 새로운 예측 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 상세 예측 조건 입력")
    
    # 요일 선택 (상세)
    selected_weekday_detailed = st.selectbox("요일 선택", ['월요일','화요일','수요일','목요일','금요일','토요일','일요일'], index=0, key="weekday_detailed")
    
    # 최저기온 입력
    min_temp = st.number_input("최저기온 (°C)", min_value=-50.0, max_value=50.0, value=15.0, step=0.1, key="min_temp")
    
    # 최고기온 입력
    max_temp = st.number_input("최고기온 (°C)", min_value=-50.0, max_value=50.0, value=25.0, step=0.1, key="max_temp")
    
    # 체감온도 입력
    feels_like_detailed = st.number_input("체감온도 (°C)", min_value=-50.0, max_value=50.0, value=(min_temp + max_temp) / 2, step=0.1, key="feels_like_detailed")

    # 월 선택 (계절성 고려)
    selected_month_detailed = st.selectbox("월 선택", month_options, index=4, key="month_detailed")  # 5월 기본값
    
    # 상세 예측 버튼
    predict_detailed_button = st.button("🔮 상세 예측 실행", type="primary", key="predict_detailed")

with col2:
    st.subheader("📊 상세 입력 정보")
    st.write(f"**선택된 요일:** {selected_weekday_detailed}")
    st.write(f"**최저기온:** {min_temp}°C")
    st.write(f"**최고기온:** {max_temp}°C")
    st.write(f"**체감온도:** {feels_like_detailed}°C")

    st.write(f"**선택된 월:** {selected_month_detailed}월")

# 상세 예측 실행
if predict_detailed_button:
    try:
        with st.spinner("상세 예측을 수행 중..."):

            
            # 평일/주말 + 요일 더미 구성 (상세 예측 입력)
            weekday_dummies_detailed = {}
            # 평일/휴일 판단: 요일 기반
            is_weekday_detailed = 1 if selected_weekday_detailed in ['월요일','화요일','수요일','목요일','금요일'] else 0
            weekday_dummies_detailed['평일_평일'] = is_weekday_detailed
            weekday_dummies_detailed.update({f'요일_{w}': (1 if w == selected_weekday_detailed else 0) for w in ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']})
            
            # 계절 판별
            is_summer_detailed = selected_month_detailed in [5, 6, 7, 8, 9]
            is_winter_detailed = selected_month_detailed in [10, 11, 12, 1, 2, 3, 4]

            # 최대수요 예측을 위한 특징 생성 (여름: 체감온도, 그 외: 실제 최고기온)
            dyn = st.session_state.get('dynamic_max_features', {})
            max_features_detailed = {
                # 냉방/난방 강도 사용
                '냉방강도': max(0.0, max_temp - 25.0),
                '난방강도': max(0.0, 10.0 - min_temp),
                '월': selected_month_detailed,
                '어제의_최대수요': dyn.get('어제의_최대수요', 0.0),
                '7일평균_최대수요': dyn.get('7일평균_최대수요', 0.0),
                '14일평균_최대수요': dyn.get('14일평균_최대수요', 0.0),
                '전주동일요일_최대수요': dyn.get('전주동일요일_최대수요', 0.0),
                # 계절별 추가 온도: 여름=최고기온 포함, 겨울=최저기온 포함
                '최고기온': max_temp if is_summer_detailed else 0.0,
                '최저기온': min_temp if is_winter_detailed else 0.0,
                '체감온도': feels_like_detailed,
                '일교차': max_temp - min_temp,
            }
            max_features_detailed.update(weekday_dummies_detailed)
            
            # 단일 모델 선택 후 피처 정렬/보정
            model_max_detailed = rf_max
            max_input_detailed = pd.DataFrame([max_features_detailed])
            max_input_detailed = align_features_for_model(model_max_detailed, max_input_detailed)
            
            # 예측 실행
            predicted_max_detailed = model_max_detailed.predict(max_input_detailed)[0]
            
            st.success("✅ 상세 예측 완료!")
            
            # 예측 결과 표시
            st.subheader("🎯 상세 예측 결과")
            
            st.metric("예측 최대수요", f"{predicted_max_detailed:,.0f} MW")
            
            # 예측 결과 상세 정보
            st.subheader("📋 상세 예측 상세 정보")
            prediction_info_detailed = pd.DataFrame({
                '항목': ['요일', '최저기온', '최고기온', '체감온도', '월', '예측 최대수요'],
                '값': [selected_weekday_detailed, f"{min_temp}°C", f"{max_temp}°C", f"{feels_like_detailed:.1f}°C", f"{selected_month_detailed}월", 
                      f"{predicted_max_detailed:,.0f} MW"]
            })
            st.dataframe(prediction_info_detailed, use_container_width=True)
            
            # 예측 신뢰도 (모델 성능 기반)
            confidence_max_detailed = min(95, max(60, st.session_state.r2_max * 100))
            
            st.subheader("📊 상세 예측 신뢰도")
            st.metric("최대수요 예측 신뢰도", f"{confidence_max_detailed:.1f}%")
            
            # 상세 예측 결과 시각화
            st.subheader("📈 상세 예측 결과 시각화")
            
            fig_prediction_detailed = go.Figure()
            
            # 최대수요만 막대 그래프로 표시
            fig_prediction_detailed.add_trace(go.Bar(
                x=['최대수요'],
                y=[predicted_max_detailed],
                name='상세 예측 최대수요',
                marker_color=['red']
            ))
            
            fig_prediction_detailed.update_layout(
                title=f"{selected_weekday_detailed} (최저:{min_temp}°C, 최고:{max_temp}°C) 최대수요 예측",
                yaxis_title="전력 수요 (MW)",
                showlegend=True
            )
            
            st.plotly_chart(fig_prediction_detailed, use_container_width=True)
            

            
    except Exception as e:
        st.error(f"❌ 상세 예측 중 오류가 발생했습니다: {str(e)}")
        st.info("모델 학습이 완료되지 않았거나 입력 데이터에 문제가 있을 수 있습니다.")

# --- 가스수요 예측 섹션 ---
st.markdown("---")
st.subheader("🔥 가스수요 예측")
st.info("최대수요와 태양광최대를 기반으로 가스수요를 예측합니다.")

# 가스수요 예측 (가능한 경우)
if hasattr(st.session_state, 'gas_model'):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 가스수요 예측 조건 입력")
        
        # 요일 선택 (평일/주말 반영)
        gas_weekday = st.selectbox("요일 선택", ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'], index=0, key="gas_weekday")
        gas_is_weekday = 1 if gas_weekday in ['월요일', '화요일', '수요일', '목요일', '금요일'] else 0

        # 최대수요 입력
        max_demand_input = st.number_input(
            "최대수요 (MW)",
            min_value=0.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0
        )
        
        # 태양광최대 입력
        solar_max_input = st.number_input(
            "태양광최대 (MW)",
            min_value=0.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0
        )
        
        # 가스수요 예측 버튼
        predict_gas_button = st.button("🔥 가스수요 예측", type="primary")
    
    with col2:
        st.subheader("📊 가스수요 입력 정보")
        st.write(f"**요일:** {gas_weekday} ({'평일' if gas_is_weekday else '주말'})")
        st.write(f"**최대수요:** {max_demand_input:,.0f} MW")
        st.write(f"**태양광최대:** {solar_max_input:,.0f} MW")
    
    # 가스수요 예측 실행
    if predict_gas_button:
        try:
            with st.spinner("가스수요 예측을 수행 중..."):
                # 예측 입력 데이터 준비 (학습 시 사용한 특징과 정합)
                last_gas = st.session_state.get('last_gas', None)
                prev_gas = st.session_state.get('prev_gas', None)

                # 변화율 계산 (가능하면), 불가 시 0.0
                if last_gas is not None and prev_gas is not None and prev_gas != 0:
                    gas_rate = (last_gas - prev_gas) / prev_gas
                else:
                    gas_rate = 0.0

                # 입력 기반 파생
                residual_load_input = max_demand_input - solar_max_input
                denom_total = max_demand_input if max_demand_input != 0 else 1.0
                solar_ratio_total = solar_max_input / denom_total
                residual_ratio_total = residual_load_input / denom_total

                # 안전한 가스/태양광 비율 (최근 가스가 없다면 0)
                # 필요 없는 파생 제거: 태양광_가스_비율 사용 안 함

                # 예측 입력을 학습 특징에 맞춰 구성 (누락 컬럼은 0으로 채움)
                input_dict = {f: 0.0 for f in st.session_state.features_gas}
                input_dict.update({
                    '최대수요': max_demand_input,
                    '태양광최대': solar_max_input,
                    '잔여부하': residual_load_input,
                    '최대수요대비_태양광비율': solar_ratio_total,
                    '최대수요대비_잔여부하비율': residual_ratio_total,
                    '목표가스_예산': max_demand_input * (st.session_state.get('gas_total_ratio_weekday', 0.0) if gas_is_weekday else st.session_state.get('gas_total_ratio_weekend', 0.0)) - solar_max_input,
                    '어제의_가스수요': last_gas if last_gas is not None else 0.0,
                    '어제의_가스수요_변화율': gas_rate,
                    '평일_평일': float(gas_is_weekday),
                })

                prediction_input_gas = pd.DataFrame([input_dict])
                
                # Step 5에서 학습된 모델의 특징 변수와 동일하게 맞춤
                if hasattr(st.session_state, 'features_gas'):
                    prediction_input_gas = prediction_input_gas[st.session_state.features_gas]
                    
                    # 가스수요 예측 (단일 모델)
                    predicted_gas_demand = st.session_state.gas_model.predict(prediction_input_gas)[0]
                    # 물리적 클리핑: 0 ≤ 가스 ≤ 최대수요
                    predicted_gas_demand = max(0.0, min(predicted_gas_demand, max_demand_input))
                    
                    st.success("✅ 가스수요 예측 완료!")
                    
                    # 예측 결과 표시
                    st.subheader("📊 가스수요 예측 결과")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("입력 최대수요", f"{max_demand_input:,.0f} MW")
                    with col2:
                        st.metric("입력 태양광최대", f"{solar_max_input:,.0f} MW")
                    with col3:
                        st.metric("예측 가스수요", f"{predicted_gas_demand:,.0f} MW")
                    
                    # 예측 신뢰도
                    confidence_gas = min(95, max(60, st.session_state.r2_gas * 100)) if hasattr(st.session_state, 'r2_gas') else 60
                    st.metric("예측 신뢰도", f"{confidence_gas:.1f}%")
                    
                    # 예측 결과 시각화
                    st.subheader("📈 가스수요 예측 시각화")
                    
                    fig_prediction_gas = go.Figure()
                    
                    fig_prediction_gas.add_trace(go.Bar(
                        x=['최대수요', '태양광최대', '예측 가스수요'],
                        y=[max_demand_input, solar_max_input, predicted_gas_demand],
                        name='입력값 및 예측값',
                        marker_color=['red', 'orange', 'green']
                    ))
                    
                    fig_prediction_gas.update_layout(
                        title="가스수요 예측 결과",
                        yaxis_title="값 (MW)",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_prediction_gas, use_container_width=True)
                    
                    # 예측 근거 설명
                    st.subheader("📋 예측 근거")
                    # 모델 중요도
                    feature_importance = st.session_state.gas_model.feature_importances_
                    
                    # Step 5에서 학습된 모델의 실제 특징 변수 사용
                    if hasattr(st.session_state, 'features_gas'):
                        if feature_importance is not None:
                            importance_df = pd.DataFrame({
                                '특성': st.session_state.features_gas,
                                '중요도': feature_importance
                            }).sort_values('중요도', ascending=False)
                            st.info(f"💡 주요 영향 요인: {importance_df.iloc[0]['특성']} ({importance_df.iloc[0]['중요도']:.1%})")
                            if len(importance_df) > 1:
                                st.info(f"💡 보조 영향 요인: {importance_df.iloc[1]['특성']} ({importance_df.iloc[1]['중요도']:.1%})")
                    else:
                        st.info("💡 모델의 특징 중요도 정보를 확인할 수 없습니다.")
                    
                else:
                    st.error("❌ 가스수요 예측을 위한 충분한 특성이 없습니다.")
                    
        except Exception as e:
            st.error(f"❌ 가스수요 예측 중 오류가 발생했습니다: {str(e)}")
            st.info("가스수요 모델 학습이 완료되지 않았거나 입력 데이터에 문제가 있을 수 있습니다.")

st.markdown("---")

# --- 7. 관련 링크 ---
st.header("🔗 관련 링크")
st.info("전력 수요 예측 검증에 사용수 있는 데이터 소스입니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌤️ 기상청 기상자료개방포털")
    st.write("기온, 습도 등 기상 데이터를 제공합니다.")
    st.markdown(
        "[기상청 기상자료개방포털 바로가기](https://data.kma.go.kr/stcs/grnd/grndTaList.do?pgmNo=70)",
        help="기상청에서 제공하는 기상 관측 데이터를 확인할 수 있습니다."
    )

with col2:
    st.subheader("⚡ 한국전력거래소")
    st.write("실시간 전력 수요 및 공급 현황을 확인할 수 있습니다.")
    st.markdown(
        "[한국전력거래소 바로가기](https://www.kpx.or.kr/powerinfoSubmain.es?mid=a10606030000)",
        help="한국전력거래소에서 제공하는 전력 수요 정보를 확인할 수 있습니다."
    )

st.markdown("---")
