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

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1120;
    color: white;
}

/* 전체 배경 */
.stApp {
    background: linear-gradient(
        135deg,
        #0B1120,
        #111827
    );
}

/* 메인 제목 */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 10px;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

/* 서브 제목 */
.sub-title {
    text-align: center;
    color: #94A3B8;
    font-size: 18px;
    margin-bottom: 40px;
}

/* 카드 */
.card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid #1E293B;
    border-radius: 22px;
    padding: 30px;
}

/* 입력창 */
[data-testid="stNumberInput"] {
    background-color: #111827;
    border-radius: 14px;
    padding: 8px;
    border: 1px solid #334155;
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
    background: #111827;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px;

    text-align: center;

    font-size: 28px;
    font-weight: 600;

    color: #F8FAFC;

    margin-bottom: 25px;
}

/* 소제목 */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 20px;
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 제목 ----------------
st.markdown(
    '<div class="main-title">LUNG AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">환자 군집 분석 시스템</div>',
    unsafe_allow_html=True
)

# ---------------- 데이터 ----------------
df = pd.DataFrame({
    'Smoking': [1,2,3,8,9,10,4,5,6,7],
    'Alcohol': [1,2,1,8,9,10,5,6,7,8],
    'Age': [20,22,25,55,60,58,35,40,45,50]
})

# ---------------- 모델 ----------------
X = df[['Smoking', 'Alcohol', 'Age']]

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

    Smokes = st.number_input("Smoking", min_value=0.0)
    Alkhol = st.number_input("Alcohol", min_value=0.0)
    Age = st.number_input("Age", min_value=0.0)

    predict_btn = st.button("군집 분석")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 결과 영역 ----------------
with col2:

    if predict_btn:

        new_patient = pd.DataFrame(
            [[Smokes, Alkhol, Age]],
            columns=['Smoking', 'Alcohol', 'Age']
        )

        # 스케일링
        new_patient_scaled = scaler.transform(new_patient)

        # 예측
        pred_cluster = model.predict(new_patient_scaled)

        # 결과 표시
        st.markdown(
            f"""
            <div class="result-box">
                예측 결과 : {pred_cluster[0]}번 군집
            </div>
            """,
            unsafe_allow_html=True
        )

        # 그래프
        fig = px.scatter(
            df,
            x='Smoking',
            y='Alcohol',
            color=df['cluster'].astype(str),
            size='Age',
            template='plotly_dark',
            opacity=0.75
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
            name='Patient'
        )

        # 그래프 디자인
        fig.update_layout(
            title='Patient Cluster Map',

            paper_bgcolor='#0B1120',
            plot_bgcolor='#111827',

            font=dict(
                family='Inter',
                size=15,
                color='white'
            ),

            legend_title='Cluster',

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(fig, use_container_width=True)
