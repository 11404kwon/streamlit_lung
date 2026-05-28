import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("환자 군집 예측 시스템")

st.write("흡연, 음주, 나이를 입력하면 환자의 군집을 예측합니다.")

# 입력 받기
Smokes = st.number_input("흡연 입력", min_value=0.0, step=1.0)
Alkhol = st.number_input("음주 입력", min_value=0.0, step=1.0)
Age = st.number_input("나이 입력", min_value=0.0, step=1.0)

# 버튼 클릭 시 실행
if st.button("군집 예측"):

    # 새로운 환자 데이터 생성
    new_patient = pd.DataFrame(
        [[Smokes, Alkhol, Age]],
        columns=['흡연', '음주', '나이']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # ---------------- 그래프 ----------------
    fig, ax = plt.subplots(figsize=(8,6))

    # 기존 데이터
    scatter = ax.scatter(
        df['흡연'],
        df['음주'],
        c=df['cluster'],
        alpha=0.5
    )

    # 새 환자 표시
    ax.scatter(
        Smokes,
        Alkhol,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel('흡연')
    ax.set_ylabel('음주')
    ax.set_title('환자 군집 시각화')
    ax.legend()

    # Streamlit에 그래프 출력
    st.pyplot(fig)