import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# ---------------- IMPROVED DATASET ----------------
data = pd.DataFrame({
    "Age":[45,30,50,25,40,35,60,55,20,70],
    "Smoking":[1,0,1,0,1,0,1,1,0,1],
    "Exercise":[0,1,0,1,0,1,0,0,1,0],
    "BP":[1,0,1,0,1,0,1,1,0,1],
    "Cholesterol":[1,0,1,0,0,1,1,1,0,1],
    "Fever":[1,0,1,0,1,0,1,1,0,1],
    "Cough":[1,0,0,0,1,1,1,1,0,1],
    "Fatigue":[1,0,1,0,0,0,1,1,0,1],
    "Disease":[1,0,1,0,1,0,1,1,0,1]
})

X = data.drop("Disease", axis=1)
y = data["Disease"]

model = GaussianNB()
model.fit(X, y)

# ---------------- UI ----------------
st.title(" Health Risk Prediction System")

st.subheader("Enter Patient Details")

# Name input
name = st.text_input("👤 Enter your name")

age = st.slider("👤 Age", 1, 100)

smoking = st.selectbox("🚬 Do you smoke?", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

exercise = st.selectbox("🏃 Do you exercise regularly?", ["Yes", "No"])
exercise = 1 if exercise == "Yes" else 0

bp = st.selectbox("💉 Do you have high blood pressure?", ["No", "Yes"])
bp = 1 if bp == "Yes" else 0

chol = st.selectbox("🧪 Do you have high cholesterol?", ["No", "Yes"])
chol = 1 if chol == "Yes" else 0

fever = st.selectbox("🌡️ Are you experiencing fever?", ["No", "Yes"])
fever = 1 if fever == "Yes" else 0

cough = st.selectbox("🤧 Do you have cough?", ["No", "Yes"])
cough = 1 if cough == "Yes" else 0

fatigue = st.selectbox("😴 Do you feel fatigue?", ["No", "Yes"])
fatigue = 1 if fatigue == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Health Risk"):

    if name == "":
        st.warning("⚠️ Please enter your name")
    else:
        patient = [[age, smoking, exercise, bp, chol, fever, cough, fatigue]]
        prob = model.predict_proba(patient)[0][1]

        st.subheader("📊 Prediction Result")

        st.write(f"### 👤 Patient: {name}")
        st.write(f"### Disease Probability: {round(prob*100,2)}%")

        # ----------- LOGICAL RISK SCORE -----------
        risk_score = smoking + cough + fever + bp + chol + fatigue

        if risk_score >= 4:
            st.error(f"⚠️ {name}, High Risk! Please consult a doctor.")
        elif risk_score >= 2:
            st.warning(f"⚠️ {name}, Moderate Risk. Take precautions.")
        else:
            st.success(f"✅ {name}, Low Risk. You are healthy.")

        # ---------------- GRAPH ----------------
        st.subheader("📈 Risk Comparison")

        values = [prob, 1 - prob]
        labels = ["Disease Risk", "No Disease"]

        plt.figure()
        plt.bar(labels, values)
        plt.ylabel("Probability")

        st.pyplot(plt)