import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- 데이터 ----------------
# 예시 데이터
df = pd.DataFrame({
    '흡연': [1, 2, 3, 8, 9, 10],
    '음주': [1, 2, 1, 8, 9, 10],
    '나이': [20, 22, 25, 55, 60, 58]
})

# ---------------- 스케일링 ----------------
X = df[['흡연', '음주', '나이']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- 모델 학습 ----------------
model = KMeans(n_clusters=2, random_state=42)
df['cluster'] = model.fit_predict(X_scaled)

# ---------------- Streamlit UI ----------------
st.title("환자 군집 예측 시스템")

Smokes = st.number_input("흡연 입력", min_value=0.0)
Alkhol = st.number_input("음주 입력", min_value=0.0)
Age = st.number_input("나이 입력", min_value=0.0)

if st.button("군집 예측"):

    # 새 환자 데이터
    new_patient = pd.DataFrame(
        [[Smokes, Alkhol, Age]],
        columns=['흡연', '음주', '나이']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # ---------------- 그래프 ----------------
    fig = px.scatter(
        df,
        x='흡연',
        y='음주',
        color=df['cluster'].astype(str),
        opacity=0.6,
        title='환자 군집 시각화'
    )

    # 새 환자 위치 표시
    fig.add_scatter(
        x=[Smokes],
        y=[Alkhol],
        mode='markers',
        marker=dict(size=18, color='black', symbol='x'),
        name='새 환자'
    )

    st.plotly_chart(fig, use_container_width=True)
