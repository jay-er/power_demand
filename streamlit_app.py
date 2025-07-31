import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import io # Added for Excel file saving

# 페이지 설정
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

# 데이터 로딩
try:
    file_path = 'power_data.xlsx' 
    data = pd.read_excel(file_path)
    st.success("✅ 기본 파일 로딩 성공!")
except Exception as e:
    st.error(f"❌ 기본 파일 로딩 오류: {str(e)}")
    st.stop()

# 데이터 편집 기능
st.subheader("📊 데이터 미리보기 및 편집")

# 데이터 정보 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 행 수", f"{len(data):,}개")
with col2:
    st.metric("총 컬럼 수", f"{len(data.columns)}개")
with col3:
    # 날짜 컬럼이 있고 datetime 타입인 경우에만 strftime 사용
    if '날짜' in data.columns:
        try:
            # 날짜 컬럼을 datetime으로 변환
            data['날짜'] = pd.to_datetime(data['날짜'])
            start_date = data['날짜'].min().strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e:
            st.warning(f"날짜 변환 오류: {e}")
            start_date = "N/A"
    else:
        start_date = "N/A"
    st.metric("시작일", start_date)
with col4:
    if '날짜' in data.columns:
        try:
            end_date = data['날짜'].max().strftime('%Y-%m-%d')
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
    st.dataframe(data, use_container_width=True)
    
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
    
    # 편집 가능한 데이터프레임
    edited_data = st.data_editor(
        data,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    
    # 변경사항 적용 버튼
    if st.button("✅ 변경사항 적용", type="primary"):
        data = edited_data.copy()
        st.success("✅ 데이터가 성공적으로 업데이트되었습니다!")
        
        # 원본 엑셀 파일 직접 수정
        try:
            data.to_excel('power_data.xlsx', index=False, engine='openpyxl')
            st.success("💾 원본 엑셀 파일이 성공적으로 업데이트되었습니다!")
        except Exception as e:
            st.error(f"❌ 원본 파일 수정 오류: {str(e)}")
            st.info("파일이 다른 프로그램에서 열려있거나 권한이 없을 수 있습니다.")
        
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
            st.success("✅ 원본 파일이 자동으로 업데이트됩니다!")

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
    '최저기온', '월', '어제의_최저수요'
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
