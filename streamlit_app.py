import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import io
import gspread
from google.oauth2.service_account import Credentials
import json

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
if 'mae_min' not in st.session_state:
    st.session_state.mae_min = None
if 'r2_min' not in st.session_state:
    st.session_state.r2_min = None

# --- [수정된 부분 1] ---
# 구글 시트 설정 함수 수정
def setup_google_sheets():
    """구글 시트 연결 설정 (st.secrets 사용)"""
    try:
        # Streamlit Cloud의 Secrets에서 직접 JSON 정보 읽기
        # st.secrets는 딕셔너리처럼 작동하여 TOML 파일을 자동으로 파싱해줍니다.
        creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
        
        # JSON 문자열을 딕셔너리로 변환
        credentials_data = json.loads(creds_json_str)
        
        # 구글 시트 API 스코프 설정
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # 인증 정보 생성
        creds = Credentials.from_service_account_info(
            credentials_data, 
            scopes=scope
        )
        
        # gspread 클라이언트 생성
        client = gspread.authorize(creds)
        return client
            
    except KeyError:
        st.error("❌ Streamlit Secrets에 'GOOGLE_CREDENTIALS_JSON'이 설정되지 않았습니다.")
        return None
    except json.JSONDecodeError:
        st.error("❌ Secrets에 저장된 GOOGLE_CREDENTIALS_JSON 값이 올바른 JSON 형식이 아닙니다.")
        st.info("값이 `{\"type\": \"service_account\", ...}` 로 시작하는지, 그리고 전체가 `\"\"\"`로 감싸여 있는지 확인하세요.")
        return None
    except Exception as e:
        st.error(f"❌ 구글 시트 연결 오류: {str(e)}")
        st.info("서비스 계정 이메일이 구글 시트에 편집자로 공유되었는지 확인하세요.")
        return None
# --- [수정 완료] ---


def load_data_from_sheet(client, sheet_name="power_data", sheet_id=None):
    """구글 시트에서 데이터 로드"""
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
        numeric_columns = ['최고기온', '평균기온', '최저기온', '최대수요', '최저수요']
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

def save_data_to_sheet(client, data, sheet_name="power_data", sheet_id=None, original_data=None):
    """구글 시트에 데이터 저장 (변경된 부분만 업데이트)"""
    try:
        # 시트 열기 (ID가 제공된 경우 ID로, 아니면 이름으로)
        if sheet_id and sheet_id.strip():
            sheet = client.open_by_key(sheet_id).sheet1
        else:
            sheet = client.open(sheet_name).sheet1
        
        # 원본 데이터가 제공된 경우 변경된 부분만 감지
        if original_data is not None:
            # 변경된 행과 열 감지
            changed_rows = []
            changed_columns = []
            
            # 데이터 타입 통일을 위해 문자열로 변환하여 비교
            data_str = data.astype(str)
            original_str = original_data.astype(str)
            
            # 변경된 행 감지
            for idx in range(len(data)):
                if not data_str.iloc[idx].equals(original_str.iloc[idx]):
                    changed_rows.append(idx + 2)  # +2는 헤더(1)와 0-based 인덱스(1) 때문
            
            # 변경된 열 감지
            for col in data.columns:
                if not data_str[col].equals(original_str[col]):
                    changed_columns.append(col)
            
            # 변경된 부분만 업데이트
            if changed_rows:
                # 변경된 행들만 업데이트
                for row_idx in changed_rows:
                    # 해당 행의 데이터 준비
                    row_data = data.iloc[row_idx - 2]  # -2는 위의 +2와 상쇄
                    
                    # 각 값을 문자열로 변환 (날짜는 년월일까지만)
                    row_values = []
                    for val in row_data:
                        if pd.isna(val):
                            row_values.append('')
                        elif isinstance(val, pd.Timestamp):
                            row_values.append(val.strftime('%Y-%m-%d'))
                        elif isinstance(val, str) and 'T' in val:  # ISO 형식 날짜 문자열
                            try:
                                date_obj = pd.to_datetime(val)
                                row_values.append(date_obj.strftime('%Y-%m-%d'))
                            except:
                                row_values.append(str(val))
                        else:
                            row_values.append(str(val))
                    
                    # 해당 행 업데이트 (A2부터 시작하므로 row_idx 사용)
                    range_name = f'A{row_idx}:{chr(65 + len(row_values) - 1)}{row_idx}'
                    sheet.update(range_name, [row_values])
                
                return True, f"✅ {len(changed_rows)}개 행이 업데이트되었습니다."
        
        # 원본 데이터가 없거나 전체 업데이트가 필요한 경우
        data_to_save = data.copy()
        
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
        sheet.update('A1', all_values)
        
        return True, "✅ 전체 데이터가 업데이트되었습니다."
        
    except Exception as e:
        st.error(f"❌ 시트 데이터 저장 오류: {str(e)}")
        return False, f"❌ 저장 실패: {str(e)}"

# 사이드바 - 데이터 정보
with st.sidebar:
    st.header("📊 데이터 정보")
    
    # 성능 평가 메시지 추가
    st.markdown("---")
    st.header("📊 모델 성능 평가")
    
    if st.session_state.r2_max is not None and st.session_state.r2_min is not None:
        if st.session_state.r2_max > 0.8 and st.session_state.r2_min > 0.8:
            st.success("🎉 두 모델 모두 우수한 성능을 보입니다!")
        elif st.session_state.r2_max > 0.6 and st.session_state.r2_min > 0.6:
            st.warning("⚠️ 모델 성능이 보통 수준입니다. 개선이 필요할 수 있습니다.")
        else:
            st.error("❌ 모델 성능이 낮습니다. 특징 공학이나 모델 튜닝이 필요합니다.")
    else:
        st.info("모델 학습 후 성능 평가가 표시됩니다.")

# --- 0. 데이터 로딩 및 편집 ---
st.header("📁 Step 0: 데이터 로딩 및 편집")

# 구글 시트 설정
st.subheader("🔐 구글 시트 연결 설정")

# 구글 시트 클라이언트 설정
client = setup_google_sheets()

if client is None:
    st.error("❌ 구글 시트 연결에 실패했습니다. 위의 안내에 따라 설정을 확인해주세요.")
    st.stop()
else:
    st.success("✅ 구글 시트 연결 성공!")

# 구글 시트 설정 정보
sheet_name = "시트1"
sheet_id = "1xyL8hCNBtf7Xo5jyIFEdoNoVJWEMSkgxMZ4nUywSBH4"

# 데이터 로딩
if st.button("📊 데이터 로드", type="primary"):
    with st.spinner("구글 시트에서 데이터를 로딩 중..."):
        data = load_data_from_sheet(client, sheet_name, sheet_id)
        
        if data is not None:
            st.session_state.data = data
            # 원본 데이터 저장 (변경 감지를 위해)
            st.session_state.original_data = data.copy()
            st.success("✅ 구글 시트에서 데이터 로딩 성공!")
        else:
            st.error("❌ 데이터 로딩에 실패했습니다.")
            st.stop()

# 데이터가 로드되었는지 확인
if 'data' not in st.session_state:
    st.info("👆 위의 '데이터 로드' 버튼을 클릭하여 구글 시트에서 데이터를 가져오세요.")
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
            # 이미 문자열인 경우 그대로 사용, 아니면 변환
            if display_data['날짜'].dtype == 'object':
                # 이미 YYYY-MM-DD 형식인지 확인
                sample_date = display_data['날짜'].iloc[0] if len(display_data) > 0 else ''
                if isinstance(sample_date, str) and len(sample_date) == 10 and '-' in sample_date:
                    pass  # 이미 올바른 형식
                else:
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
    st.info("아래에서 데이터를 직접 편집할 수 있습니다. 편집 후 '변경사항 적용' 버튼을 클릭하세요.")
    
    # 편집용 데이터 준비 (날짜는 년월일까지만 표시)
    edit_data = data.copy()
    if '날짜' in edit_data.columns:
        try:
            # 이미 문자열인 경우 그대로 사용, 아니면 변환
            if edit_data['날짜'].dtype == 'object':
                # 이미 YYYY-MM-DD 형식인지 확인
                sample_date = edit_data['날짜'].iloc[0] if len(edit_data) > 0 else ''
                if isinstance(sample_date, str) and len(sample_date) == 10 and '-' in sample_date:
                    pass  # 이미 올바른 형식
                else:
                    # datetime으로 변환 후 년월일까지만 표시
                    edit_data['날짜'] = pd.to_datetime(edit_data['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception as e:
            st.warning(f"날짜 편집 변환 중 오류: {e}")
    
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
            
            st.session_state.data = data
            
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
    
    # 필수 컬럼 확인
    required_columns = ['최고기온', '평균기온', '최저기온', '최대수요', '최저수요', '요일', '평일']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"❌ 필수 컬럼이 누락되었습니다: {missing_columns}")
        st.info("필수 컬럼: 날짜, 최고기온, 평균기온, 최저기온, 최대수요, 최저수요, 요일, 평일")
        st.stop()
    
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
    data_processed = pd.get_dummies(data, columns=['요일', '평일'], drop_first=True)
    data_processed['어제의_최대수요'] = data_processed['최대수요'].shift(1)
    data_processed['어제의_최저수요'] = data_processed['최저수요'].shift(1)
    data_processed.dropna(inplace=True)
    
    st.success("✅ 데이터 정제 완료!")
    
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

# --- 3. 모델별 변수 및 데이터 분리 ---
st.header("🎯 Step 3: 모델별 변수 및 데이터 분리")

# [최대수요 모델]
features_max = [
    '최고기온', '평균기온', '월', '어제의_최대수요'
] + [col for col in data_processed if '요일_' in col or '평일_' in col]
X_max = data_processed[features_max]
y_max = data_processed['최대수요']

# [최저수요 모델]
features_min = [
    '최저기온', '평균기온', '월', '어제의_최저수요'
] + [col for col in data_processed if '요일_' in col or '평일_' in col]
X_min = data_processed[features_min]
y_min = data_processed['최저수요']

# 고정된 파라미터 사용
test_size = 0.2
n_estimators = 100
random_state = 42

# train_test_split을 사용한 랜덤 분할
X_max_train, X_max_test, y_max_train, y_max_test = train_test_split(
    X_max, y_max, test_size=test_size, random_state=random_state
)
X_min_train, X_min_test, y_min_train, y_min_test = train_test_split(
    X_min, y_min, test_size=test_size, random_state=random_state
)

# 변수 정보 표시
st.subheader("📈 최대수요 모델 변수")
st.write(f"특징 변수: {len(features_max)}개")
# 헤더 행을 사용한 한 줄 표
max_vars_df = pd.DataFrame([features_max], columns=[f'변수{i+1}' for i in range(len(features_max))])
st.dataframe(max_vars_df, use_container_width=True)

st.subheader("📉 최저수요 모델 변수")
st.write(f"특징 변수: {len(features_min)}개")
# 헤더 행을 사용한 한 줄 표
min_vars_df = pd.DataFrame([features_min], columns=[f'변수{i+1}' for i in range(len(features_min))])
st.dataframe(min_vars_df, use_container_width=True)

st.markdown("---")

# --- 4. 모델 학습 ---
st.header("🤖 Step 4: 모델 학습")
with st.spinner("모델을 학습 중..."):
    rf_max = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_max.fit(X_max_train, y_max_train)
    
    rf_min = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_min.fit(X_min_train, y_min_train)
    
    st.success("✅ 모델 학습 완료!")

st.markdown("---")

# --- 5. 모델 성능 평가 ---
st.header("📊 Step 5: 모델 성능 평가")
with st.spinner("성능을 평가 중..."):
    y_max_pred = rf_max.predict(X_max_test)
    y_min_pred = rf_min.predict(X_min_test)
    
    # 전역 변수 업데이트
    st.session_state.mae_max = mean_absolute_error(y_max_test, y_max_pred)
    st.session_state.r2_max = r2_score(y_max_test, y_max_pred)
    st.session_state.mae_min = mean_absolute_error(y_min_test, y_min_pred)
    st.session_state.r2_min = r2_score(y_min_test, y_min_pred)

# 성능 결과 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 최대수요 예측 모델 성능")
    st.metric("평균 절대 오차 (MAE)", f"{st.session_state.mae_max:,.0f} MW")
    st.metric("결정 계수 (R²)", f"{st.session_state.r2_max:.4f}")

with col2:
    st.subheader("📉 최저수요 예측 모델 성능")
    st.metric("평균 절대 오차 (MAE)", f"{st.session_state.mae_min:,.0f} MW")
    st.metric("결정 계수 (R²)", f"{st.session_state.r2_min:.4f}")

st.markdown("---")

# --- 6. 전력 수요 예측 ---
st.header("🔮 Step 6: 전력 수요 예측")
st.info("요일과 평균기온을 입력하여 최대/최저 수요를 예측합니다.")

# 예측 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 예측 조건 입력")
    
    # 요일 선택
    weekday_options = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    selected_weekday = st.selectbox("요일 선택", weekday_options, index=0)
    
    # 평균기온 입력
    avg_temp = st.number_input("평균기온 (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
    
    # 월 선택 (계절성 고려)
    month_options = list(range(1, 13))
    selected_month = st.selectbox("월 선택", month_options, index=4)  # 5월 기본값
    
    # 예측 버튼
    predict_button = st.button("🔮 예측 실행", type="primary")

with col2:
    st.subheader("📊 입력 정보")
    st.write(f"**선택된 요일:** {selected_weekday}")
    st.write(f"**평균기온:** {avg_temp}°C")
    st.write(f"**선택된 월:** {selected_month}월")

# 예측 실행
if predict_button:
    try:
        with st.spinner("예측을 수행 중..."):
            # 요일 원핫 인코딩
            weekday_dummies = {}
            for day in weekday_options:
                if day == selected_weekday:
                    weekday_dummies[f'요일_{day}'] = 1
                else:
                    weekday_dummies[f'요일_{day}'] = 0
            
            # 평일 여부 (주말이면 0, 평일이면 1)
            is_weekday = 1 if selected_weekday in ['월요일', '화요일', '수요일', '목요일', '금요일'] else 0
            
            # 평일 원핫 인코딩
            weekday_dummies['평일_평일'] = is_weekday
            
            # 최대수요 예측을 위한 특징 생성
            max_features = {
                '최고기온': avg_temp + 5,  # 평균기온 + 5도로 추정
                '평균기온': avg_temp,
                '월': selected_month,
                '어제의_최대수요': 50000  # 기본값 (실제로는 이전 데이터 필요)
            }
            max_features.update(weekday_dummies)
            
            # 최저수요 예측을 위한 특징 생성
            min_features = {
                '최저기온': avg_temp - 5,  # 평균기온 - 5도로 추정
                '평균기온': avg_temp,
                '월': selected_month,
                '어제의_최저수요': 30000  # 기본값 (실제로는 이전 데이터 필요)
            }
            min_features.update(weekday_dummies)
            
            # 특징 순서 맞추기
            max_input = pd.DataFrame([max_features])
            min_input = pd.DataFrame([min_features])
            
            # 필요한 컬럼만 선택
            max_input = max_input[features_max]
            min_input = min_input[features_min]
            
            # 예측 실행
            predicted_max = rf_max.predict(max_input)[0]
            predicted_min = rf_min.predict(min_input)[0]
            
            st.success("✅ 예측 완료!")
            
            # 예측 결과 표시
            st.subheader("🎯 예측 결과")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("예측 최대수요", f"{predicted_max:,.0f} MW")
            with col2:
                st.metric("예측 최저수요", f"{predicted_min:,.0f} MW")
            
            # 예측 결과 상세 정보
            st.subheader("📋 예측 상세 정보")
            prediction_info = pd.DataFrame({
                '항목': ['요일', '평균기온', '월', '예측 최대수요', '예측 최저수요', '수요 차이'],
                '값': [selected_weekday, f"{avg_temp}°C", f"{selected_month}월", 
                      f"{predicted_max:,.0f} MW", f"{predicted_min:,.0f} MW", 
                      f"{predicted_max - predicted_min:,.0f} MW"]
            })
            st.dataframe(prediction_info, use_container_width=True)
            
            # 예측 신뢰도 (모델 성능 기반)
            confidence_max = min(95, max(60, st.session_state.r2_max * 100))
            confidence_min = min(95, max(60, st.session_state.r2_min * 100))
            
            st.subheader("📊 예측 신뢰도")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("최대수요 예측 신뢰도", f"{confidence_max:.1f}%")
            with col2:
                st.metric("최저수요 예측 신뢰도", f"{confidence_min:.1f}%")
            
            # 예측 결과 시각화
            st.subheader("📈 예측 결과 시각화")
            
            fig_prediction = go.Figure()
            
            # 최대수요와 최저수요를 막대 그래프로 표시
            fig_prediction.add_trace(go.Bar(
                x=['최대수요', '최저수요'],
                y=[predicted_max, predicted_min],
                name='예측 수요',
                marker_color=['red', 'blue']
            ))
            
            fig_prediction.update_layout(
                title=f"{selected_weekday} ({avg_temp}°C) 전력 수요 예측",
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
st.info("최저기온과 최고기온을 직접 입력하여 더 정확한 예측을 수행합니다.")

# 새로운 예측 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 상세 예측 조건 입력")
    
    # 요일 선택 (기존과 동일)
    selected_weekday_detailed = st.selectbox("요일 선택", weekday_options, index=0, key="weekday_detailed")
    
    # 최저기온 입력
    min_temp = st.number_input("최저기온 (°C)", min_value=-50.0, max_value=50.0, value=15.0, step=0.1, key="min_temp")
    
    # 최고기온 입력
    max_temp = st.number_input("최고기온 (°C)", min_value=-50.0, max_value=50.0, value=25.0, step=0.1, key="max_temp")
    
    # 월 선택 (계절성 고려)
    selected_month_detailed = st.selectbox("월 선택", month_options, index=4, key="month_detailed")  # 5월 기본값
    
    # 상세 예측 버튼
    predict_detailed_button = st.button("🔮 상세 예측 실행", type="primary", key="predict_detailed")

with col2:
    st.subheader("📊 상세 입력 정보")
    st.write(f"**선택된 요일:** {selected_weekday_detailed}")
    st.write(f"**최저기온:** {min_temp}°C")
    st.write(f"**최고기온:** {max_temp}°C")
    st.write(f"**평균기온:** {(min_temp + max_temp) / 2:.1f}°C")
    st.write(f"**선택된 월:** {selected_month_detailed}월")

# 상세 예측 실행
if predict_detailed_button:
    try:
        with st.spinner("상세 예측을 수행 중..."):
            # 평균기온 계산
            avg_temp_detailed = (min_temp + max_temp) / 2
            
            # 요일 원핫 인코딩
            weekday_dummies_detailed = {}
            for day in weekday_options:
                if day == selected_weekday_detailed:
                    weekday_dummies_detailed[f'요일_{day}'] = 1
                else:
                    weekday_dummies_detailed[f'요일_{day}'] = 0
            
            # 평일 여부 (주말이면 0, 평일이면 1)
            is_weekday_detailed = 1 if selected_weekday_detailed in ['월요일', '화요일', '수요일', '목요일', '금요일'] else 0
            
            # 평일 원핫 인코딩
            weekday_dummies_detailed['평일_평일'] = is_weekday_detailed
            
            # 최대수요 예측을 위한 특징 생성 (실제 최고기온 사용)
            max_features_detailed = {
                '최고기온': max_temp,  # 실제 최고기온 사용
                '평균기온': avg_temp_detailed,
                '월': selected_month_detailed,
                '어제의_최대수요': 50000  # 기본값
            }
            max_features_detailed.update(weekday_dummies_detailed)
            
            # 최저수요 예측을 위한 특징 생성 (실제 최저기온 사용)
            min_features_detailed = {
                '최저기온': min_temp,  # 실제 최저기온 사용
                '평균기온': avg_temp_detailed,
                '월': selected_month_detailed,
                '어제의_최저수요': 30000  # 기본값
            }
            min_features_detailed.update(weekday_dummies_detailed)
            
            # 특징 순서 맞추기
            max_input_detailed = pd.DataFrame([max_features_detailed])
            min_input_detailed = pd.DataFrame([min_features_detailed])
            
            # 필요한 컬럼만 선택
            max_input_detailed = max_input_detailed[features_max]
            min_input_detailed = min_input_detailed[features_min]
            
            # 예측 실행
            predicted_max_detailed = rf_max.predict(max_input_detailed)[0]
            predicted_min_detailed = rf_min.predict(min_input_detailed)[0]
            
            st.success("✅ 상세 예측 완료!")
            
            # 예측 결과 표시
            st.subheader("🎯 상세 예측 결과")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("예측 최대수요", f"{predicted_max_detailed:,.0f} MW")
            with col2:
                st.metric("예측 최저수요", f"{predicted_min_detailed:,.0f} MW")
            
            # 예측 결과 상세 정보
            st.subheader("📋 상세 예측 상세 정보")
            prediction_info_detailed = pd.DataFrame({
                '항목': ['요일', '최저기온', '최고기온', '평균기온', '월', '예측 최대수요', '예측 최저수요', '수요 차이'],
                '값': [selected_weekday_detailed, f"{min_temp}°C", f"{max_temp}°C", f"{avg_temp_detailed:.1f}°C", f"{selected_month_detailed}월", 
                      f"{predicted_max_detailed:,.0f} MW", f"{predicted_min_detailed:,.0f} MW", 
                      f"{predicted_max_detailed - predicted_min_detailed:,.0f} MW"]
            })
            st.dataframe(prediction_info_detailed, use_container_width=True)
            
            # 예측 신뢰도 (모델 성능 기반)
            confidence_max_detailed = min(95, max(60, st.session_state.r2_max * 100))
            confidence_min_detailed = min(95, max(60, st.session_state.r2_min * 100))
            
            st.subheader("📊 상세 예측 신뢰도")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("최대수요 예측 신뢰도", f"{confidence_max_detailed:.1f}%")
            with col2:
                st.metric("최저수요 예측 신뢰도", f"{confidence_min_detailed:.1f}%")
            
            # 상세 예측 결과 시각화
            st.subheader("📈 상세 예측 결과 시각화")
            
            fig_prediction_detailed = go.Figure()
            
            # 최대수요와 최저수요를 막대 그래프로 표시
            fig_prediction_detailed.add_trace(go.Bar(
                x=['최대수요', '최저수요'],
                y=[predicted_max_detailed, predicted_min_detailed],
                name='상세 예측 수요',
                marker_color=['red', 'blue']
            ))
            
            fig_prediction_detailed.update_layout(
                title=f"{selected_weekday_detailed} (최저:{min_temp}°C, 최고:{max_temp}°C) 전력 수요 예측",
                yaxis_title="전력 수요 (MW)",
                showlegend=True
            )
            
            st.plotly_chart(fig_prediction_detailed, use_container_width=True)
            
            # 두 예측 방식 비교
            st.subheader("📊 예측 방식 비교")
            comparison_data = pd.DataFrame({
                '예측 방식': ['평균기온 기반', '상세 기온 기반'],
                '최대수요 예측': [f"{predicted_max:,.0f} MW", f"{predicted_max_detailed:,.0f} MW"],
                '최저수요 예측': [f"{predicted_min:,.0f} MW", f"{predicted_min_detailed:,.0f} MW"],
                '수요 차이': [f"{predicted_max - predicted_min:,.0f} MW", f"{predicted_max_detailed - predicted_min_detailed:,.0f} MW"]
            })
            st.dataframe(comparison_data, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ 상세 예측 중 오류가 발생했습니다: {str(e)}")
        st.info("모델 학습이 완료되지 않았거나 입력 데이터에 문제가 있을 수 있습니다.")

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
