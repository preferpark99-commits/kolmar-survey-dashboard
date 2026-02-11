"""
============================================================
맨즈케어 데이&나이트 듀얼 샴푸 - 설문 분석 대시보드
============================================================
Streamlit + Plotly로 구현한 인터랙티브 시각화
HK Gothic 폰트 적용
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import base64
import os

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="데이&나이트 듀얼 샴푸 분석",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HK Gothic 폰트 로드 (Base64 인코딩)
# ============================================================
@st.cache_data
def load_font_as_base64(font_path):
    """폰트 파일을 Base64로 인코딩"""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 폰트 파일 경로
font_dir = os.path.dirname(os.path.abspath(__file__))
font_bold_path = os.path.join(font_dir, "HK Gothic Bold.ttf")
font_extrabold_path = os.path.join(font_dir, "HK Gothic ExtraBold.ttf")

# Base64 인코딩
font_bold_b64 = load_font_as_base64(font_bold_path)
font_extrabold_b64 = load_font_as_base64(font_extrabold_path)

# 커스텀 CSS (HK Gothic 폰트 적용)
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    /* Material Symbols 폰트 (Streamlit 아이콘용) */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    /* HK Gothic Bold 폰트 정의 */
    @font-face {{
        font-family: 'HK Gothic';
        src: url(data:font/truetype;charset=utf-8;base64,{font_bold_b64}) format('truetype');
        font-weight: 700;
        font-style: normal;
    }}
    
    /* HK Gothic ExtraBold 폰트 정의 */
    @font-face {{
        font-family: 'HK Gothic';
        src: url(data:font/truetype;charset=utf-8;base64,{font_extrabold_b64}) format('truetype');
        font-weight: 800;
        font-style: normal;
    }}
    
    /* 전체 폰트 적용 (Material Symbols 제외) */
    *:not(.material-symbols-outlined) {{
        font-family: 'HK Gothic', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    /* Material Symbols 폰트 명시적 적용 */
    .material-symbols-outlined {{
        font-family: 'Material Symbols Outlined' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }}
    
    /* Streamlit 기본 요소들 */
    .stMarkdown, .stText, p, span, div, label {{
        font-family: 'HK Gothic', sans-serif !important;
        font-weight: 700;
    }}
    
    /* 메인 헤더 - ExtraBold */
    .main-header {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    
    /* 서브 헤더 - Bold */
    .sub-header {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 1.1rem;
        font-weight: 700;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    /* 섹션 타이틀 - ExtraBold */
    h1, h2, h3 {{
        font-family: 'HK Gothic', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }}
    
    /* 메트릭 값 - ExtraBold */
    .metric-value, [data-testid="stMetricValue"] {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 800 !important;
    }}
    
    /* 메트릭 라벨 - Bold */
    .metric-label, [data-testid="stMetricLabel"] {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 0.9rem;
        font-weight: 700;
        opacity: 0.9;
    }}
    
    /* 인사이트 박스 */
    .insight-box {{
        font-family: 'HK Gothic', sans-serif !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #667eea;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 0.8rem 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .insight-box strong {{
        font-weight: 800;
        color: #1a1a2e;
    }}
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 1rem;
        font-weight: 800;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }}
    
    /* 사이드바 */
    [data-testid="stSidebar"] {{
        font-family: 'HK Gothic', sans-serif !important;
    }}
    
    [data-testid="stSidebar"] h2 {{
        font-weight: 800 !important;
    }}
    
    /* 버튼 */
    .stButton > button {{
        font-family: 'HK Gothic', sans-serif !important;
        font-weight: 700;
    }}
    
    /* 셀렉트박스 */
    .stSelectbox label {{
        font-family: 'HK Gothic', sans-serif !important;
        font-weight: 700;
    }}
    
    /* 카드 스타일 */
    .card {{
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }}
    
    .card-title {{
        font-family: 'HK Gothic', sans-serif !important;
        font-weight: 800;
        font-size: 1.2rem;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }}
    
    /* 강조 텍스트 */
    .highlight {{
        font-weight: 800;
        color: #667eea;
    }}
    
    /* 숫자 강조 */
    .big-number {{
        font-family: 'HK Gothic', sans-serif !important;
        font-size: 3rem;
        font-weight: 800;
        color: #667eea;
        line-height: 1;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def load_data():
    df_raw = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', header=None)
    df = df_raw[df_raw[0].str.contains('2026', na=False)].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타1', '기타2']
    
    # 데이터 전처리
    df['구매의향'] = (df['Q8'] == '있다').astype(int)
    df['Q7_score'] = pd.to_numeric(df['Q7'], errors='coerce').fillna(3)
    df['하루2번샴푸'] = df['머리감는시간'].str.contains('아침&저녁', na=False)
    
    return df

df = load_data()

# ============================================================
# Plotly 차트 폰트 설정 (시스템 폰트 사용 - 웹에서는 CSS가 적용됨)
# ============================================================
# Plotly 기본 템플릿 설정
plotly_font = "HK Gothic, AppleGothic, Malgun Gothic, sans-serif"

chart_layout = dict(
    font=dict(family=plotly_font, size=13),
    legend_font=dict(family=plotly_font, size=11),
    hoverlabel=dict(font=dict(family=plotly_font, size=11)),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title_text="",  # 빈 문자열로 타이틀 제거
    margin=dict(l=20, r=80, t=40, b=60),  # 여백 조정 (왼쪽, 오른쪽, 위, 아래)
    autosize=True,
)

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ 필터 설정")
    
    # 성별 필터
    gender_options = ['전체'] + list(df['성별'].unique())
    selected_gender = st.selectbox("성별", gender_options)
    
    # 연령대 필터 (순서 정렬 + 20&30대 옵션 추가)
    age_order = ['전체', '10대', '20대', '30대', '40대', '50대 이상', '20&30대']
    age_options = [age for age in age_order if age == '전체' or age == '20&30대' or age in df['연령대'].unique()]
    selected_age = st.selectbox("연령대", age_options)
    
    # 머리 감는 시간 필터
    time_options = ['전체'] + list(df['머리감는시간'].unique())
    selected_time = st.selectbox("머리 감는 시간", time_options)
    
    st.markdown("---")
    st.markdown("### 📊 데이터 정보")
    st.markdown(f"**전체 응답자:** {len(df)}명")
    
# 필터 적용
filtered_df = df.copy()
if selected_gender != '전체':
    filtered_df = filtered_df[filtered_df['성별'] == selected_gender]
if selected_age == '20&30대':
    filtered_df = filtered_df[filtered_df['연령대'].isin(['20대', '30대'])]
elif selected_age != '전체':
    filtered_df = filtered_df[filtered_df['연령대'] == selected_age]
if selected_time != '전체':
    filtered_df = filtered_df[filtered_df['머리감는시간'] == selected_time]

# ============================================================
# 메인 헤더
# ============================================================
st.markdown('<h1 class="main-header">🧴 데이&나이트 듀얼 샴푸</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">설문 분석 대시보드 | 맨즈케어 제품 기획 근거</p>', unsafe_allow_html=True)

# ============================================================
# 핵심 지표 카드
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📋 필터된 응답자",
        value=f"{len(filtered_df)}명",
        delta=f"전체의 {len(filtered_df)/len(df)*100:.1f}%"
    )

with col2:
    purchase_rate = filtered_df['구매의향'].mean() * 100
    st.metric(
        label="💰 구매 의향",
        value=f"{purchase_rate:.1f}%",
        delta=f"{purchase_rate - df['구매의향'].mean()*100:+.1f}%p vs 전체"
    )

with col3:
    avg_q7 = filtered_df['Q7_score'].mean()
    st.metric(
        label="🌙 아침/밤 두피 변화 체감도",
        value=f"{avg_q7:.2f}점",
        delta=f"{avg_q7 - df['Q7_score'].mean():+.2f} vs 전체"
    )

with col4:
    twice_daily = filtered_df['하루2번샴푸'].mean() * 100
    st.metric(
        label="🚿 하루 2번 샴푸",
        value=f"{twice_daily:.1f}%"
    )

st.markdown("---")

# ============================================================
# 탭 구성
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 기본 분석", "🎯 타겟 분석", "🤖 구매 의향 예측 모델", "🧴 제품 소개", "📝 설문조사"])

# ============================================================
# Tab 1: 기본 분석
# ============================================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 성별 분포")
        gender_counts = filtered_df['성별'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            color_discrete_sequence=['#667eea', '#f093fb'],
            hole=0.4
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14))
        fig_gender.update_layout(**chart_layout, showlegend=False, height=350)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 연령대 분포")
        age_counts = filtered_df['연령대'].value_counts().sort_index()
        # 연령대별 연속적인 색상 (밝은 → 진한 그라데이션)
        age_colors = ['#A8E6CF', '#7BD3EA', '#5B9BD5', '#3A6EA5', '#1E3A5F']
        fig_age = px.bar(
            x=age_counts.index,
            y=age_counts.values,
            color=age_counts.index,
            color_discrete_sequence=age_colors
        )
        fig_age.update_layout(
            **chart_layout,
            xaxis_title="연령대",
            yaxis_title="응답자 수",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🕐 머리 감는 시간대")
        time_counts = filtered_df['머리감는시간'].value_counts()
        # 시간대별 색상 매핑 (아침: 노란색, 저녁: 보라색, 아침&저녁: 초록색)
        time_color_map = {
            '아침(하루 1번)': '#FFD93D',      # 노란색 (아침 햇살)
            '저녁(하루 1번)': '#6C5CE7',      # 보라색 (저녁 밤)
            '아침&저녁(하루 2번)': '#00B894'  # 초록색 (둘 다)
        }
        fig_time = px.bar(
            y=time_counts.index,
            x=time_counts.values,
            orientation='h',
            color=time_counts.index,
            color_discrete_map=time_color_map
        )
        fig_time.update_layout(
            **chart_layout,
            xaxis_title="응답자 수",
            yaxis_title="",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col4:
        st.markdown("### 💡 구매 의향")
        purchase_counts = filtered_df['Q8'].value_counts()
        fig_purchase = px.pie(
            values=purchase_counts.values,
            names=purchase_counts.index,
            color=purchase_counts.index,
            color_discrete_map={'있다': '#2ecc71', '없다': '#e74c3c'},
            hole=0.4
        )
        fig_purchase.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14))
        fig_purchase.update_layout(**chart_layout, height=350)
        st.plotly_chart(fig_purchase, use_container_width=True)

# ============================================================
# Tab 2: 타겟 분석 (순서 변경됨)
# ============================================================
with tab3:
    st.markdown("### 🤖 구매 의향 예측 - Feature Importance 분석")
    
    # Random Forest 설명 (접을 수 있는 expander)
    with st.expander("ℹ️ Random Forest 모델이란? (클릭하여 펼치기)"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
            <h4 style="color: #1a1a2e; margin-top: 0;">🌲 Random Forest (랜덤 포레스트)</h4>
            <p style="color: #333; line-height: 1.7;">
                여러 개의 <strong>의사결정나무(Decision Tree)</strong>를 만들어 
                <strong>다수결 투표</strong>로 최종 예측을 하는 앙상블 머신러닝 모델입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 작동 원리 시각화 (이미지 사용)
        st.markdown("#### 🔄 작동 원리")
        
        col_exp1, col_exp2, col_exp3 = st.columns([1, 3, 1])
        with col_exp2:
            st.image("random_forest.png", use_container_width=True)
        
        st.markdown("""
        <p style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 10px;">
            여러 개의 Decision Tree가 각각 예측 → 다수결(Majority Voting)로 최종 결정
        </p>
        """, unsafe_allow_html=True)
        
        # 장점 카드
        st.markdown("#### ✅ 왜 Random Forest를 사용하나요?")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("""
            <div style="background: #e8f5e9; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <strong style="color: #2e7d32;">🛡️ 과적합 방지</strong><br>
                <span style="font-size: 0.9rem;">여러 트리의 평균을 사용해 안정적인 결과</span>
            </div>
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 0.5rem;">
                <strong style="color: #1565c0;">📊 Feature Importance</strong><br>
                <span style="font-size: 0.9rem;">어떤 변수가 중요한지 자동으로 계산</span>
            </div>
            """, unsafe_allow_html=True)
        
        with adv_col2:
            st.markdown("""
            <div style="background: #fff3e0; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <strong style="color: #ef6c00;">🔀 비선형 관계 학습</strong><br>
                <span style="font-size: 0.9rem;">복잡한 패턴도 잡아낼 수 있음</span>
            </div>
            <div style="background: #fce4ec; padding: 1rem; border-radius: 0.5rem;">
                <strong style="color: #c2185b;">💪 결측치에 강함</strong><br>
                <span style="font-size: 0.9rem;">일부 데이터가 없어도 잘 작동</span>
            </div>
            """, unsafe_allow_html=True)
        
        # 분석 방법론 설명
        st.markdown("#### 🔬 본 분석의 신뢰성 확보 방법")
        st.markdown("""
        <table style="width: 100%; border-collapse: collapse; text-align: center;">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 12px; border: 1px solid #ddd;">방법</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">설명</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">목적</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Train/Test 분리</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">데이터를 75% 학습용, 25% 테스트용으로 분리</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">실제 예측 성능 측정</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>교차 검증 (5-Fold CV)</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">데이터를 5등분하여 5번 반복 검증</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">결과의 안정성 확인</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Permutation Importance</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">변수 값을 섞어서 성능 저하 측정</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">더 정확한 중요도 산출</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>클래스 균형 처리</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">구매 있다/없다 비율 보정</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">편향 없는 학습</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>💡 분석 방법:</strong> Random Forest 모델을 사용하여 구매 의향에 영향을 미치는 요인을 분석합니다.<br>
    <strong>💡 개선된 분석:</strong> Train/Test 분리, 교차 검증, Permutation Importance를 통해 신뢰성을 확보했습니다.
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Engineering (개선됨)
    df_ml = filtered_df.copy()
    
    # 성별 인코딩 (남성=1, 여성=0)
    df_ml['성별_encoded'] = (df_ml['성별'] == '남성').astype(int)
    
    # 연령대 순서형 인코딩 (수동 매핑으로 순서 보장)
    age_mapping = {'10대': 1, '20대': 2, '30대': 3, '40대': 4, '50대 이상': 5}
    df_ml['연령대_encoded'] = df_ml['연령대'].map(age_mapping).fillna(3)
    
    df_ml['하루2번샴푸_encoded'] = df_ml['하루2번샴푸'].astype(int)
    
    scalp_concerns = ['두피 열감', '유분 과다', '건조함', '가려움', '탈모', '민감성']
    for concern in scalp_concerns:
        df_ml[f'고민_{concern}'] = df_ml['두피고민'].str.contains(concern, na=False).astype(int)
    
    feature_cols = ['성별_encoded', '연령대_encoded', 'Q7_score', '하루2번샴푸_encoded'] + \
                   [f'고민_{c}' for c in scalp_concerns]
    feature_names_kr = ['성별', '연령대', '아침/밤 두피 변화 체감도', '하루2번샴푸',
                        '고민:두피열감', '고민:유분과다', '고민:건조함', '고민:가려움', 
                        '고민:탈모', '고민:민감성']
    
    X = df_ml[feature_cols]
    y = df_ml['구매의향']
    
    if len(filtered_df) >= 30:  # 최소 샘플 수 증가 (Train/Test 분리를 위해)
        
        # ============================================================
        # 1. Train/Test 분리
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # ============================================================
        # 2. 모델 학습 (클래스 불균형 처리 포함)
        # ============================================================
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42,
            class_weight='balanced'  # 클래스 불균형 처리
        )
        rf_model.fit(X_train, y_train)
        
        # ============================================================
        # 3. 모델 성능 평가
        # ============================================================
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = None
        
        # ============================================================
        # 4. 교차 검증
        # ============================================================
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
        
        # ============================================================
        # 5. Permutation Importance (더 신뢰성 있는 중요도)
        # ============================================================
        perm_importance = permutation_importance(
            rf_model, X_test, y_test, n_repeats=10, random_state=42
        )
        
        # Gini Importance (기존 방식)
        gini_importance_df = pd.DataFrame({
            '피처': feature_names_kr,
            'Gini 중요도': rf_model.feature_importances_
        }).sort_values('Gini 중요도', ascending=True)
        
        # Permutation Importance (개선된 방식)
        perm_importance_df = pd.DataFrame({
            '피처': feature_names_kr,
            'Permutation 중요도': perm_importance.importances_mean
        }).sort_values('Permutation 중요도', ascending=True)
        
        # ============================================================
        # 시각화
        # ============================================================
        
        # 모델 성능 지표 표시
        st.markdown("#### 📊 모델 성능 지표")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("테스트 정확도", f"{accuracy*100:.1f}%")
        with perf_col2:
            if auc_score:
                st.metric("AUC Score", f"{auc_score:.3f}")
            else:
                st.metric("AUC Score", "N/A")
        with perf_col3:
            st.metric("교차검증 평균", f"{cv_scores.mean()*100:.1f}%")
        with perf_col4:
            st.metric("교차검증 표준편차", f"±{cv_scores.std()*100:.1f}%")
        
        st.markdown("---")
        
        # Feature Importance 비교 (Gini vs Permutation)
        st.markdown("#### 🔬 Feature Importance 비교")
        st.markdown("""
        <div class="insight-box">
        <strong>Gini Importance:</strong> 트리 분할 시 불순도 감소량 기반 (빠르지만 편향 가능)<br>
        <strong>Permutation Importance:</strong> 피처 값을 섞었을 때 성능 저하 정도 (더 신뢰성 있음)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Gini Importance")
            fig_gini = px.bar(
                gini_importance_df,
                x='Gini 중요도',
                y='피처',
                orientation='h',
                color='Gini 중요도',
                color_continuous_scale='Purples'
            )
            fig_gini.update_layout(
                font=dict(family=plotly_font, size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="중요도",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False,
                height=400,
                margin=dict(l=120, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_gini, use_container_width=True)
        
        with col2:
            st.markdown("##### Permutation Importance")
            fig_perm = px.bar(
                perm_importance_df,
                x='Permutation 중요도',
                y='피처',
                orientation='h',
                color='Permutation 중요도',
                color_continuous_scale='Greens'
            )
            fig_perm.update_layout(
                font=dict(family=plotly_font, size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="중요도",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False,
                height=400,
                margin=dict(l=120, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_perm, use_container_width=True)
        
        st.markdown("---")
        
        # Top 5 비교 (카드 스타일)
        st.markdown("#### 🏆 Top 5 중요 피처 비교")
        
        top5_col1, top5_col2 = st.columns(2)
        
        # 순위별 메달 이모지
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        
        with top5_col1:
            top5_gini = gini_importance_df.tail(5).iloc[::-1]
            gini_items = []
            for i, (_, row) in enumerate(top5_gini.iterrows()):
                medal = medals[i]
                feature = row['피처']
                score = row['Gini 중요도']
                gini_items.append((medal, feature, score, i == 0))
            
            # Gini 카드 HTML 생성
            gini_card_html = '''<div style="background: white; padding: 20px; border-radius: 12px; 
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15); border: 1px solid #e0e0e0;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="background: #667eea; color: white; padding: 6px 16px; 
                                border-radius: 20px; font-weight: 800; font-size: 0.9rem;">
                        Gini Importance
                    </span>
                </div>'''
            
            for medal, feature, score, is_first in gini_items:
                if is_first:
                    gini_card_html += f'''
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;
                                display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: white; font-weight: 800; font-size: 1rem;">
                            {medal} {feature}
                        </span>
                        <span style="background: rgba(255,255,255,0.2); color: white; padding: 4px 10px; 
                                    border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                            {score:.4f}
                        </span>
                    </div>'''
                else:
                    gini_card_html += f'''
                    <div style="background: #f8f9fa; padding: 10px 16px; border-radius: 8px; 
                                margin-bottom: 6px; border-left: 4px solid #667eea;
                                display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #333; font-weight: 700;">
                            {medal} {feature}
                        </span>
                        <span style="color: #667eea; font-weight: 700; font-size: 0.9rem;">
                            {score:.4f}
                        </span>
                    </div>'''
            
            gini_card_html += '</div>'
            st.markdown(gini_card_html, unsafe_allow_html=True)
        
        with top5_col2:
            top5_perm = perm_importance_df.tail(5).iloc[::-1]
            perm_items = []
            for i, (_, row) in enumerate(top5_perm.iterrows()):
                medal = medals[i]
                feature = row['피처']
                score = row['Permutation 중요도']
                perm_items.append((medal, feature, score, i == 0))
            
            # Permutation 카드 HTML 생성
            perm_card_html = '''<div style="background: white; padding: 20px; border-radius: 12px; 
                        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.15); border: 1px solid #e0e0e0;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="background: #2ecc71; color: white; padding: 6px 16px; 
                                border-radius: 20px; font-weight: 800; font-size: 0.9rem;">
                        Permutation Importance
                    </span>
                </div>'''
            
            for medal, feature, score, is_first in perm_items:
                if is_first:
                    perm_card_html += f'''
                    <div style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                                padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;
                                display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: white; font-weight: 800; font-size: 1rem;">
                            {medal} {feature}
                        </span>
                        <span style="background: rgba(255,255,255,0.2); color: white; padding: 4px 10px; 
                                    border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                            {score:.4f}
                        </span>
                    </div>'''
                else:
                    perm_card_html += f'''
                    <div style="background: #f8f9fa; padding: 10px 16px; border-radius: 8px; 
                                margin-bottom: 6px; border-left: 4px solid #2ecc71;
                                display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #333; font-weight: 700;">
                            {medal} {feature}
                        </span>
                        <span style="color: #2ecc71; font-weight: 700; font-size: 0.9rem;">
                            {score:.4f}
                        </span>
                    </div>'''
            
            perm_card_html += '</div>'
            st.markdown(perm_card_html, unsafe_allow_html=True)
        
        # 핵심 인사이트 - 동적으로 생성
        top_feature_gini = gini_importance_df.iloc[-1]['피처']
        top_feature_perm = perm_importance_df.iloc[-1]['피처']
        
        if top_feature_gini == top_feature_perm:
            insight_html = f'''
            <div style="background: #f8f9fa; border: 2px solid #667eea; 
                        padding: 1.2rem 1.8rem; border-radius: 0.8rem; margin-top: 1.5rem;
                        display: flex; align-items: center; gap: 1rem;">
                <div style="background: #667eea; color: white; padding: 0.6rem 1rem; 
                            border-radius: 0.5rem; font-weight: 800; font-size: 0.9rem; white-space: nowrap;">
                    🎯 핵심 인사이트
                </div>
                <p style="color: #1a1a2e; font-size: 1rem; font-weight: 700; margin: 0; line-height: 1.5;">
                    두 방법 모두에서 <span style="color: #667eea; font-weight: 800;">{top_feature_gini}</span>가 
                    가장 중요한 변수로 나타났습니다. → <span style="color: #e74c3c; font-weight: 800;">높은 신뢰도!</span>
                </p>
            </div>
            '''
        else:
            insight_html = f'''
            <div style="background: #f8f9fa; border: 2px solid #667eea; 
                        padding: 1.2rem 1.8rem; border-radius: 0.8rem; margin-top: 1.5rem;
                        display: flex; align-items: center; gap: 1rem;">
                <div style="background: #667eea; color: white; padding: 0.6rem 1rem; 
                            border-radius: 0.5rem; font-weight: 800; font-size: 0.9rem; white-space: nowrap;">
                    🎯 핵심 인사이트
                </div>
                <p style="color: #1a1a2e; font-size: 1rem; font-weight: 700; margin: 0; line-height: 1.5;">
                    Gini: <span style="color: #667eea; font-weight: 800;">{top_feature_gini}</span> / 
                    Permutation: <span style="color: #2ecc71; font-weight: 800;">{top_feature_perm}</span>이 
                    각각 1위입니다. 두 결과를 종합적으로 해석하세요.
                </p>
            </div>
            '''
        
        st.markdown(insight_html, unsafe_allow_html=True)
        
        # 분석 신뢰도 안내
        st.markdown("---")
        st.markdown("#### 📋 분석 신뢰도 체크리스트")
        
        checks = []
        checks.append(("✅" if accuracy > 0.6 else "⚠️", f"테스트 정확도: {accuracy*100:.1f}% {'(양호)' if accuracy > 0.6 else '(주의 필요)'}"))
        checks.append(("✅" if cv_scores.std() < 0.15 else "⚠️", f"교차검증 안정성: ±{cv_scores.std()*100:.1f}% {'(안정적)' if cv_scores.std() < 0.15 else '(변동 큼)'}"))
        checks.append(("✅" if len(filtered_df) >= 50 else "⚠️", f"샘플 수: {len(filtered_df)}명 {'(충분)' if len(filtered_df) >= 50 else '(더 많으면 좋음)'}"))
        checks.append(("✅", "Train/Test 분리: 적용됨 (25% 테스트)"))
        checks.append(("✅", "클래스 불균형 처리: 적용됨 (class_weight='balanced')"))
        
        for icon, text in checks:
            st.markdown(f"{icon} {text}")
        
    else:
        st.warning("⚠️ 신뢰성 있는 분석을 위해 최소 30명 이상의 데이터가 필요합니다. 필터를 조정해주세요.")

# ============================================================
# Tab 3: Feature Importance (순서 변경됨)
# ============================================================
with tab2:
    st.markdown("### 🎯 세그먼트별 구매 의향 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 두피 변화 체감도별 구매 의향")
        q7_purchase = df.groupby('Q7_score')['구매의향'].agg(['mean', 'count']).reset_index()
        q7_purchase = q7_purchase[q7_purchase['Q7_score'].between(1, 5)]
        q7_purchase['구매의향_pct'] = q7_purchase['mean'] * 100
        
        fig_q7 = px.bar(
            q7_purchase,
            x='Q7_score',
            y='구매의향_pct',
            color='구매의향_pct',
            color_continuous_scale='RdYlGn',
            text=q7_purchase.apply(lambda x: f"{x['구매의향_pct']:.0f}%<br>(n={int(x['count'])})", axis=1)
        )
        fig_q7.update_traces(textposition='outside', textfont=dict(size=11))
        fig_q7.update_layout(
            font=dict(family=plotly_font, size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="아침/밤 두피 변화 체감도",
            yaxis_title="구매 의향 비율 (%)",
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=60, r=40, t=60, b=80),
            yaxis=dict(range=[0, 95]),
        )
        st.plotly_chart(fig_q7, use_container_width=True)
    
    with col2:
        st.markdown("#### 머리 감는 시간대별 구매 의향")
        time_purchase = df.groupby('머리감는시간')['구매의향'].agg(['mean', 'count']).reset_index()
        time_purchase['구매의향_pct'] = time_purchase['mean'] * 100
        time_purchase = time_purchase.sort_values('구매의향_pct', ascending=True)
        
        fig_time = px.bar(
            time_purchase,
            y='머리감는시간',
            x='구매의향_pct',
            orientation='h',
            color='구매의향_pct',
            color_continuous_scale='Blues',
            text=time_purchase.apply(lambda x: f"{x['구매의향_pct']:.0f}% (n={int(x['count'])})", axis=1)
        )
        fig_time.update_traces(textposition='outside', textfont=dict(size=10))
        fig_time.update_layout(
            font=dict(family=plotly_font, size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="구매 의향 비율 (%)",
            yaxis_title="",
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=150, r=140, t=60, b=80),
            xaxis=dict(range=[0, 110]),
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # 인사이트 박스 - 두 개를 나란히 배치
    twice_rate = df[df['하루2번샴푸']]['구매의향'].mean() * 100
    once_rate = df[~df['하루2번샴푸']]['구매의향'].mean() * 100
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div style="background: #f8f9fa; border: 2px solid #667eea; 
                    padding: 1rem 1.5rem; border-radius: 0.8rem; height: 120px;
                    display: flex; align-items: center; gap: 1rem;">
            <div style="background: #667eea; color: white; padding: 0.5rem 0.8rem; 
                        border-radius: 0.5rem; font-weight: 800; font-size: 0.85rem; white-space: nowrap;">
                💡 해석
            </div>
            <p style="color: #1a1a2e; font-size: 0.95rem; font-weight: 700; margin: 0; line-height: 1.5;">
                아침/밤 두피 변화 체감도가 높을수록 구매 의향이 높아지는 경향이 있습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f"""
        <div style="background: #f8f9fa; border: 2px solid #667eea; 
                    padding: 1rem 1.5rem; border-radius: 0.8rem; height: 120px;
                    display: flex; align-items: center; gap: 1rem;">
            <div style="background: #667eea; color: white; padding: 0.5rem 0.8rem; 
                        border-radius: 0.5rem; font-weight: 800; font-size: 0.85rem; white-space: nowrap;">
                💡 핵심 타겟
            </div>
            <div style="color: #1a1a2e; font-size: 0.9rem; font-weight: 700; margin: 0; line-height: 1.6;">
                • 하루 2번 샴푸: <span style="color: #667eea; font-weight: 800;">{twice_rate:.1f}%</span> 구매 의향<br>
                • 하루 1번 샴푸: <span style="color: #667eea; font-weight: 800;">{once_rate:.1f}%</span> 구매 의향<br>
                → <span style="color: #e74c3c; font-weight: 800;">하루 2번 샴푸 고객이 주요 타겟!</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 두피 변화 체감도와 구매 의향 관계 시각화 (제품 컨셉 검증에서 이동)
    st.markdown("#### 📈 두피 변화 체감도와 구매 의향의 관계")
    
    fig_boxplot = px.box(
        df,
        x='Q8',
        y='Q7_score',
        color='Q8',
        color_discrete_map={'있다': '#2ecc71', '없다': '#e74c3c'},
        points='all'
    )
    fig_boxplot.update_layout(
        font=dict(family=plotly_font, size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="구매 의향",
        yaxis_title="아침/밤 두피 변화 체감도",
        showlegend=False,
        height=400,
        margin=dict(l=60, r=40, t=40, b=60)
    )
    st.plotly_chart(fig_boxplot, use_container_width=True)
    
    # 통계 요약
    q7_yes_avg = df[df['Q8'] == '있다']['Q7_score'].mean()
    q7_no_avg = df[df['Q8'] == '없다']['Q7_score'].mean()
    
    box_col1, box_col2, box_col3 = st.columns(3)
    with box_col1:
        st.metric("구매 의향 있음 - 체감도 평균", f"{q7_yes_avg:.2f}점")
    with box_col2:
        st.metric("구매 의향 없음 - 체감도 평균", f"{q7_no_avg:.2f}점")
    with box_col3:
        st.metric("평균 차이", f"{q7_yes_avg - q7_no_avg:+.2f}점")
    
    st.markdown("""
    <div class="insight-box">
    <strong>✅ 결론:</strong><br>
    "아침과 밤 두피 상태가 다르다고 느끼는 소비자일수록 데이&나이트 듀얼 샴푸에 대한 구매 의향이 높다"<br>
    → <strong>제품 컨셉이 소비자 니즈와 정확히 매칭됨!</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 두피 고민별 구매 의향")
    
    concerns = ['탈모', '유분 과다', '두피 열감', '건조함', '가려움', '민감성']
    concern_data = []
    
    for concern in concerns:
        concern_temp_df = df[df['두피고민'].str.contains(concern, na=False)]
        if len(concern_temp_df) >= 5:
            concern_data.append({
                '두피 고민': concern,
                '구매의향_pct': concern_temp_df['구매의향'].mean() * 100,
                '응답자수': len(concern_temp_df)
            })
    
    concern_result_df = pd.DataFrame(concern_data).sort_values('구매의향_pct', ascending=True)
    
    fig_concern = px.bar(
        concern_result_df,
        y='두피 고민',
        x='구매의향_pct',
        orientation='h',
        color='구매의향_pct',
        color_continuous_scale='Oranges',
        text=concern_result_df.apply(lambda x: f"{x['구매의향_pct']:.0f}% (n={int(x['응답자수'])})", axis=1)
    )
    fig_concern.update_traces(textposition='outside', textfont=dict(size=11))
    fig_concern.update_layout(
        font=dict(family=plotly_font, size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="구매 의향 비율 (%)",
        yaxis_title="",
        showlegend=False,
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=100, r=120, t=40, b=60),
        xaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_concern, use_container_width=True)
    
    # 두피 고민별 구매 의향 해석 박스
    # 상위 2개, 하위 2개 고민 추출
    top_concerns = concern_result_df.tail(2).iloc[::-1]
    bottom_concerns = concern_result_df.head(2)
    
    top1_concern = top_concerns.iloc[0]['두피 고민']
    top1_pct = top_concerns.iloc[0]['구매의향_pct']
    top2_concern = top_concerns.iloc[1]['두피 고민']
    top2_pct = top_concerns.iloc[1]['구매의향_pct']
    
    concern_insight_col1, concern_insight_col2 = st.columns(2)
    
    with concern_insight_col1:
        st.markdown(f'''
        <div style="background: #f8f9fa; border: 2px solid #667eea; 
                    padding: 1rem 1.5rem; border-radius: 0.8rem; height: 140px;
                    display: flex; align-items: center; gap: 1rem;">
            <div style="background: #667eea; color: white; padding: 0.5rem 0.8rem; 
                        border-radius: 0.5rem; font-weight: 800; font-size: 0.85rem; white-space: nowrap;">
                💡 해석
            </div>
            <p style="color: #1a1a2e; font-size: 0.95rem; font-weight: 700; margin: 0; line-height: 1.6;">
                <span style="color: #667eea; font-weight: 800;">{top1_concern}</span>, 
                <span style="color: #667eea; font-weight: 800;">{top2_concern}</span> 고민을 가진 고객이<br>
                데이&나이트 샴푸에 가장 높은 관심을 보입니다.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with concern_insight_col2:
        st.markdown(f'''
        <div style="background: #f8f9fa; border: 2px solid #667eea; 
                    padding: 1rem 1.5rem; border-radius: 0.8rem; height: 140px;
                    display: flex; align-items: center; gap: 1rem;">
            <div style="background: #667eea; color: white; padding: 0.5rem 0.8rem; 
                        border-radius: 0.5rem; font-weight: 800; font-size: 0.85rem; white-space: nowrap;">
                💡 제품 전략
            </div>
            <div style="color: #1a1a2e; font-size: 0.9rem; font-weight: 700; margin: 0; line-height: 1.6;">
                • {top1_concern} 고민: <span style="color: #667eea; font-weight: 800;">{top1_pct:.0f}%</span> 구매 의향<br>
                • {top2_concern} 고민: <span style="color: #667eea; font-weight: 800;">{top2_pct:.0f}%</span> 구매 의향<br>
                → <span style="color: #e74c3c; font-weight: 800;">나이트 샴푸에 해당 기능 강조!</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# ============================================================
# Tab 4: 제품 소개
# ============================================================
with tab4:
    st.markdown("### 🌙☀️ 데이&나이트 듀얼 샴푸 제품 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 1rem;">
            <h3 style="color: #f39c12;">🌙 나이트 샴푸</h3>
            <p><strong>컨셉:</strong> 세정력 + 보습 + 탈모 완화</p>
            <ul>
                <li>세정력 + 멘톨/살리실산 유효성분</li>
                <li>계면활성제 (sulfate계)</li>
                <li>보습 (판테놀, 오일 등)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 나이트 샴푸 타겟 니즈
        night_needs = df['샴푸선택이유'].str.contains('탈모 완화|세정력|두피 케어', na=False, regex=True).sum()
        night_complaints = df['샴푸아쉬운점'].str.contains('두피 케어 효과|세정력', na=False, regex=True).sum()
        
        st.markdown(f"""
        **📊 니즈 검증:**
        - 탈모 완화/세정력/두피케어 중시: **{night_needs}명** ({night_needs/len(df)*100:.1f}%)
        - 두피케어/세정력 불만: **{night_complaints}명** ({night_complaints/len(df)*100:.1f}%)
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                    padding: 2rem; border-radius: 1rem; color: #333; margin-bottom: 1rem;">
            <h3 style="color: #e74c3c;">☀️ 모닝 샴푸</h3>
            <p><strong>컨셉:</strong> 저자극 + 가볍게 유분기 제거</p>
            <ul>
                <li>컨디셔닝제</li>
                <li>천연 계면활성제</li>
                <li>저자극 포뮬러</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 모닝 샴푸 타겟 니즈
        morning_needs = df['두피고민'].str.contains('유분', na=False).sum()
        morning_complaints = df['샴푸아쉬운점'].str.contains('유분|자극', na=False, regex=True).sum()
        
        st.markdown(f"""
        **📊 니즈 검증:**
        - 유분 과다 고민: **{morning_needs}명** ({morning_needs/len(df)*100:.1f}%)
        - 유분/자극 불만: **{morning_complaints}명** ({morning_complaints/len(df)*100:.1f}%)
        """)

# ============================================================
# Tab 5: 설문조사 (디자인 개선)
# ============================================================
with tab5:
    st.markdown("### 📝 헤어·두피 케어 제품 설문조사")
    
    # 안내 배너
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem 2rem; border-radius: 1rem; margin-bottom: 2rem; color: white;">
        <p style="font-size: 1.1rem; margin: 0; line-height: 1.8;">
            안녕하세요 😊 본 설문조사는 <strong>일상 속 헤어·두피 케어 제품에 대한 소비자 수요와 사용 경험</strong>을 
            알아보기 위해 진행됩니다.<br>
            응답해주신 내용은 설문 목적에 한해 활용되며, 모든 응답은 <strong>익명으로 처리</strong>됩니다 🔒<br>
            <span style="opacity: 0.9;">🕒 소요 시간: 약 1분</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="survey_form", clear_on_submit=True):
        
        # ========== Q1, Q2: 기본 정보 ==========
        st.markdown("""
        <div style="background: #f0f4ff; padding: 1rem 1.5rem; border-radius: 0.8rem; 
                    border-left: 5px solid #667eea; margin-bottom: 1.5rem;">
            <h4 style="color: #667eea; margin: 0; font-size: 1.1rem;">📋 기본 정보</h4>
        </div>
        """, unsafe_allow_html=True)
        
        q1_col, q2_col = st.columns(2)
        
        with q1_col:
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin-bottom: 1rem;">
                <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                    <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q1</span>
                    성별을 선택해 주세요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            q1_gender = st.radio("성별", options=["남성", "여성"], horizontal=True, label_visibility="collapsed")
        
        with q2_col:
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin-bottom: 1rem;">
                <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                    <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q2</span>
                    연령대를 선택해 주세요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            q2_age = st.radio("연령대", options=["10대", "20대", "30대", "40대", "50대 이상"], horizontal=True, label_visibility="collapsed")
        
        # Q3: 머리 감는 시간
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin-bottom: 1.5rem;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q3</span>
                평소 머리를 감는 시간대는 언제인가요?
            </p>
        </div>
        """, unsafe_allow_html=True)
        q3_time = st.radio("머리감는시간", options=["아침(하루 1번)", "저녁(하루 1번)", "아침&저녁(하루 2번)"], horizontal=True, label_visibility="collapsed")
        
        # ========== Q4, Q5, Q6: 두피 고민 및 샴푸 사용 ==========
        st.markdown("""
        <div style="background: #fff5f0; padding: 1rem 1.5rem; border-radius: 0.8rem; 
                    border-left: 5px solid #f39c12; margin: 1.5rem 0;">
            <h4 style="color: #f39c12; margin: 0; font-size: 1.1rem;">🧴 두피 고민 및 샴푸 사용</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Q4: 두피 고민 (체크박스 형태)
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin-bottom: 1rem;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.5rem;">
                <span style="background: #f39c12; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q4</span>
                현재 가장 고민되는 두피 상태는 무엇인가요? 
                <span style="color: #e74c3c; font-size: 0.9rem;">(최대 2개 선택)</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        q4_options = ["두피 열감", "유분 과다 / 떡짐", "건조함 / 각질", "가려움", "탈모 / 모발 가늘어짐", "민감성 / 자극감", "특별한 고민 없음"]
        q4_cols = st.columns(4)
        q4_concerns = []
        for i, option in enumerate(q4_options):
            with q4_cols[i % 4]:
                if st.checkbox(option, key=f"q4_{i}"):
                    q4_concerns.append(option)
        
        if len(q4_concerns) > 2:
            st.warning("⚠️ 최대 2개까지만 선택 가능합니다.")
        
        # Q5: 샴푸 선택 이유 (체크박스 형태)
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin: 1rem 0;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.5rem;">
                <span style="background: #f39c12; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q5</span>
                현재 사용 중인 샴푸를 선택하게 된 가장 큰 이유는 무엇인가요? 
                <span style="color: #667eea; font-size: 0.9rem;">(복수 선택 가능)</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        q5_options = ["두피 케어 효과를 기대해서", "탈모 완화 기능이 있어서", "세정력이 좋아서", "향이 마음에 들어서", "가격이 합리적이어서", "브랜드 신뢰도 / 인지도"]
        q5_cols = st.columns(3)
        q5_reasons = []
        for i, option in enumerate(q5_options):
            with q5_cols[i % 3]:
                if st.checkbox(option, key=f"q5_{i}"):
                    q5_reasons.append(option)
        
        # Q6: 샴푸 아쉬운 점 (체크박스 형태)
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin: 1rem 0;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.5rem;">
                <span style="background: #f39c12; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q6</span>
                현재 사용 중인 샴푸에 대해 가장 아쉬운 점은 무엇인가요? 
                <span style="color: #667eea; font-size: 0.9rem;">(복수 선택 가능)</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        q6_options = ["세정력이 부족하다", "유분이 빨리 올라온다", "향이 부담스럽다", "자극적이다", "두피 케어 효과를 느끼기 어렵다"]
        q6_cols = st.columns(3)
        q6_complaints = []
        for i, option in enumerate(q6_options):
            with q6_cols[i % 3]:
                if st.checkbox(option, key=f"q6_{i}"):
                    q6_complaints.append(option)
        
        # ========== Q7, Q8: 제품 관심도 ==========
        st.markdown("""
        <div style="background: #f0fff4; padding: 1rem 1.5rem; border-radius: 0.8rem; 
                    border-left: 5px solid #2ecc71; margin: 1.5rem 0;">
            <h4 style="color: #2ecc71; margin: 0; font-size: 1.1rem;">🌙☀️ 제품 관심도</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Q7: 두피 상태 차이 (라디오 버튼 1~5)
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin-bottom: 1rem;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                <span style="background: #2ecc71; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q7</span>
                하루 중 아침과 밤, 두피 상태가 다르다고 느낀 적이 있나요?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        q7_label_col1, q7_radio_col, q7_label_col2 = st.columns([1.5, 4, 1.5])
        with q7_label_col1:
            st.markdown("<p style='text-align: right; color: #888; font-size: 0.9rem; margin-top: 8px;'>매우 그렇지 않다</p>", unsafe_allow_html=True)
        with q7_radio_col:
            q7_score = st.radio("Q7점수", options=[1, 2, 3, 4, 5], horizontal=True, label_visibility="collapsed", index=2)
        with q7_label_col2:
            st.markdown("<p style='text-align: left; color: #888; font-size: 0.9rem; margin-top: 8px;'>매우 그렇다</p>", unsafe_allow_html=True)
        
        # Q8: 구매 의향
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e0e0e0; margin: 1rem 0;">
            <p style="font-size: 1rem; font-weight: 700; color: #333; margin-bottom: 0.5rem;">
                <span style="background: #2ecc71; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-right: 8px;">Q8</span>
                만약 아침용(데이) / 밤용(나잇)으로 구분된 두피 케어 샴푸가 출시된다면, 구매 의향이 있나요?
            </p>
            <div style="background: #f8f9fa; padding: 0.8rem 1rem; border-radius: 0.5rem; margin-top: 0.5rem; font-size: 0.9rem; color: #555;">
                <strong>*참고:</strong><br>
                ☀️ 아침용(데이): 저자극, 순한 성분으로 가볍게 유분기만 제거<br>
                🌙 밤용(나잇): 세정력, 보습력, 탈모 완화↑
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        q8_purchase = st.radio("구매의향", options=["있다", "없다"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 제출 버튼
        submit_button = st.form_submit_button(
            label="📮 설문 제출하기",
            use_container_width=True,
            type="primary"
        )
        
        if submit_button:
            # 최대 선택 개수 검증
            if len(q4_concerns) > 2:
                st.error("❌ Q4에서 최대 2개까지만 선택 가능합니다. 다시 선택해주세요.")
            else:
                # 응답 데이터 생성
                from datetime import datetime
                
                new_response = {
                    '타임스탬프': datetime.now().strftime('%Y/%m/%d %p %I:%M:%S').replace('AM', '오전').replace('PM', '오후'),
                    '성별': q1_gender,
                    '연령대': q2_age,
                    '머리감는시간': q3_time,
                    '두피고민': ', '.join(q4_concerns) if q4_concerns else '',
                    '샴푸선택이유': ', '.join(q5_reasons) if q5_reasons else '',
                    '샴푸아쉬운점': ', '.join(q6_complaints) if q6_complaints else '',
                    'Q7': str(q7_score),
                    'Q8': q8_purchase,
                    '기타1': '',
                    '기타2': ''
                }
                
                # CSV 파일에 추가 저장 시도
                try:
                    csv_path = '헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv'
                    new_df = pd.DataFrame([new_response])
                    new_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                    
                    st.success("✅ 설문이 성공적으로 제출되었습니다! 감사합니다 💗")
                    st.balloons()
                    st.cache_data.clear()
                    st.info("🔄 페이지를 새로고침하면 새로운 응답이 분석에 반영됩니다.")
                    
                except Exception as e:
                    st.warning("⚠️ 설문이 접수되었습니다! (서버 환경에서는 실시간 저장이 제한될 수 있습니다)")
                    st.info(f"📋 응답 내용: {q1_gender}, {q2_age}, {q3_time}, Q7={q7_score}, Q8={q8_purchase}")
    
    # 감사 메시지
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 1rem; color: white;">
        <p style="font-size: 1.2rem; font-weight: 800; margin: 0;">💗 설문에 참여해주셔서 감사합니다 💗</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; font-family: 'HK Gothic', sans-serif;">
    <p style="font-weight: 800; font-size: 1.1rem; margin-bottom: 0.5rem;">🧴 맨즈케어 데이&나이트 듀얼 샴푸 | 설문 분석 대시보드</p>
    <p style="font-size: 0.85rem; font-weight: 700; opacity: 0.7;">데이터 기반 제품 기획 | Streamlit + Plotly | HK Gothic Font</p>
</div>
""", unsafe_allow_html=True)
