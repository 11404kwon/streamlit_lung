import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- 페이지 설정 ----------------
st.set_page_config(
    page_title="CYBER LUNG AI",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    background-color: #050816;
    color: white;
}

.stApp {
    background: linear-gradient(135deg,#050816,#0f172a,#111827);
}

.main-title {
    text-align:center;
    font-size:60px;
    font-weight:700;
    color:#00F5FF;
    text-shadow: 0 0 20px #00F5FF;
    margin-bottom:10px;
}

.sub-title{
    text-align:center;
    color:#94A3B8;
    margin-bottom:40px;
}

.block {
    background: rgba(15,23,42,0.7);
    border: 1px solid #00F5FF;
    border-radius:20px;
    padding:25px;
    box-shadow: 0 0 25px rgba(0,245,255,0.3);
}

.stButton>button {
    width:100%;
    background: linear-gradient(90deg,#00F5FF,#8B5CF6);
    color:white;
    border:none;
    border-radius:15px;
    height:60px;
    font-size:20px;
    font-weight:bold;
    transition:0.3s;
    box-shadow:0 0 20px rgba(0,245,255,0.5);
}

.stButton>button:hover{
    transform:scale(1.03);
    box-shadow:0 0 30px rgba(139,92,246,0.8);
}

[data-testid="stNumberInput"]{
    background-color:#0f172a;
    border-radius:15px;
    padding:10px;
    border:1px solid #334155;
}

.result-box{
    background: rgba(0,245,255,0.08);
    border:1px solid #00F5FF;
    border-radius:20px;
    padding:20px;
    text-align:center;
    font-size:30px;
    color:#00F5FF;
    margin-top:20px;
    box-shadow:0 0 20px rgba(0,245,255,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------- 제목 ----------------
st.markdown('<div class="main-title">CYBER LUNG AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">환자 군집 분석 시스템</div>', unsafe_allow_html=True)

# ---------------- 예시 데이터 ----------------
df = pd.DataFrame({
    '흡연': [1,2,3,8,9,10,4,5,6,7],
    '음주': [1,2,1,8,9,10,5,6,7,8],
    '나이': [20,22,25,55,60,58,35,40,45,50]
})

# ---------------- 모델 학습 ----------------
X = df[['흡연','음주','나이']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=2, random_state=42)
df['cluster'] = model.fit_predict(X_scaled)

# ---------------- 레이아웃 ----------------
col1, col2 = st.columns([1,1.3])

# ---------------- 입력창 ----------------
with col1:

    st.markdown('<div class="block">', unsafe_allow_html=True)

    st.subheader("🧬 환자 데이터 입력")

    Smokes = st.number_input("흡연 수치", min_value=0.0)
    Alkhol = st.number_input("음주 수치", min_value=0.0)
    Age = st.number_input("나이", min_value=0.0)

    predict_btn = st.button("AI 군집 분석 시작")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 결과 ----------------
with col2:

    if predict_btn:

        new_patient = pd.DataFrame(
            [[Smokes, Alkhol, Age]],
            columns=['흡연','음주','나이']
        )

        new_patient_scaled = scaler.transform(new_patient)

        pred_cluster = model.predict(new_patient_scaled)

        st.markdown(
            f'''
            <div class="result-box">
                🔥 환자는 <b>{pred_cluster[0]}번 군집</b>으로 분석되었습니다.
            </div>
            ''',
            unsafe_allow_html=True
        )

        # ---------------- 그래프 ----------------
        fig = px.scatter(
            df,
            x='흡연',
            y='음주',
            color=df['cluster'].astype(str),
            size='나이',
            template='plotly_dark',
            opacity=0.75
        )

        # 새 환자 추가
        fig.add_scatter(
            x=[Smokes],
            y=[Alkhol],
            mode='markers',
            marker=dict(
                size=24,
                color='red',
                symbol='x'
            ),
            name='NEW PATIENT'
        )

        fig.update_layout(
            paper_bgcolor='#050816',
            plot_bgcolor='#050816',
            font=dict(color='white'),
            title='CYBER CLUSTER MAP'
        )

        st.plotly_chart(fig, use_container_width=True)
