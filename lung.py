import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- 페이지 설정 ----------------
st.set_page_config(
    page_title="LUNG AI",
    page_icon="🫁",
    layout="wide"
)

# ---------------- 스타일 ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
    background-color: #F1F5F9;
    color: #0F172A;
}

/* 전체 배경 */
.stApp {
    background: linear-gradient(
        135deg,
        #F8FAFC,
        #E2E8F0
    );
}

/* 메인 제목 */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: #1E3A8A;
    margin-top: 10px;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

/* 서브 제목 */
.sub-title {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 40px;
}

/* 카드 */
.card {
    background: rgba(255,255,255,0.9);
    border: 1px solid #CBD5E1;
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

/* 입력창 */
[data-testid="stNumberInput"] {
    background-color: white;
    border-radius: 14px;
    padding: 8px;
    border: 1px solid #CBD5E1;
}

/* 버튼 */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;

    background: #2563EB;
    color: white;

    font-size: 18px;
    font-weight: 600;

    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #1D4ED8;
}

/* 결과 박스 */
.result-box {
    background: white;
    border: 1px solid #CBD5E1;
    border-radius: 18px;
    padding: 22px;

    text-align: center;

    font-size: 28px;
    font-weight: 600;

    color: #0F172A;

    margin-bottom: 25px;

    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

/* 소제목 */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 20px;
    color: #0F172A;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 제목 ----------------
st.markdown(
    '<div class="main-title">LUNG AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">폐 건강 환자 군집 분석 시스템</div>',
    unsafe_allow_html=True
)

# ---------------- 데이터 ----------------
df = pd.DataFrame({
    '흡연': [1,2,3,8,9,10,4,5,6,7],
    '음주': [1,2,1,8,9,10,5,6,7,8],
    '나이': [20,22,25,55,60,58,35,40,45,50]
})

# ---------------- 모델 ----------------
X = df[['흡연', '음주', '나이']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=2, random_state=42)
df['cluster'] = model.fit_predict(X_scaled)

# ---------------- 레이아웃 ----------------
col1, col2 = st.columns([1, 1.4])

# ---------------- 입력 영역 ----------------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">환자 정보 입력</div>',
        unsafe_allow_html=True
    )

    Smokes = st.number_input("흡연 수치", min_value=0.0)
    Alkhol = st.number_input("음주 수치", min_value=0.0)
    Age = st.number_input("나이", min_value=0.0)

    predict_btn = st.button("군집 분석 시작")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 결과 영역 ----------------
with col2:

    if predict_btn:

        new_patient = pd.DataFrame(
            [[Smokes, Alkhol, Age]],
            columns=['흡연', '음주', '나이']
        )

        # 스케일링
        new_patient_scaled = scaler.transform(new_patient)

        # 예측
        pred_cluster = model.predict(new_patient_scaled)

        # 결과 표시
        st.markdown(
            f"""
            <div class="result-box">
                분석 결과 : {pred_cluster[0]}번 군집
            </div>
            """,
            unsafe_allow_html=True
        )

        # 그래프
        fig = px.scatter(
            df,
            x='흡연',
            y='음주',
            color=df['cluster'].astype(str),
            size='나이',
            opacity=0.8,

            labels={
                '흡연': '흡연',
                '음주': '음주',
                'cluster': '군집'
            }
        )

        # 새 환자 표시
        fig.add_scatter(
            x=[Smokes],
            y=[Alkhol],
            mode='markers',
            marker=dict(
                size=22,
                color='red',
                symbol='x'
            ),
            name='새 환자'
        )

        # 그래프 디자인
        fig.update_layout(
            title='환자 군집 분석 그래프',

            paper_bgcolor='#F8FAFC',
            plot_bgcolor='white',

            font=dict(
                family='Pretendard',
                size=15,
                color='#0F172A'
            ),

            legend_title='군집',

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(fig, use_container_width=True)
