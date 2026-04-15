import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ---------------- DATASET ----------------
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

# ---------------- MODEL ----------------
model = GaussianNB()
model.fit(X, y)

# Accuracy
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>🏥 Health Risk Prediction System</h1>", unsafe_allow_html=True)

st.write("This app predicts health risk based on lifestyle and symptoms using Machine Learning.")

st.subheader("👤 Enter Patient Details")

name = st.text_input("Enter your name")

age = st.slider("Age", 1, 100)

smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
exercise = 1 if exercise == "Yes" else 0

bp = st.selectbox("High blood pressure?", ["No", "Yes"])
bp = 1 if bp == "Yes" else 0

chol = st.selectbox("High cholesterol?", ["No", "Yes"])
chol = 1 if chol == "Yes" else 0

fever = st.selectbox("Fever?", ["No", "Yes"])
fever = 1 if fever == "Yes" else 0

cough = st.selectbox("Cough?", ["No", "Yes"])
cough = 1 if cough == "Yes" else 0

fatigue = st.selectbox("Fatigue?", ["No", "Yes"])
fatigue = 1 if fatigue == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Health Risk"):

    if name.strip() == "":
        st.warning("⚠️ Please enter your name")
    else:
        patient = np.array([[age, smoking, exercise, bp, chol, fever, cough, fatigue]])
        prob = model.predict_proba(patient)[0][1]

        st.subheader("📊 Prediction Result")

        st.write(f"### 👤 Patient: {name}")
        st.write(f"### Disease Probability: {round(prob*100,2)}%")

        # Risk logic
        risk_score = smoking + cough + fever + bp + chol + fatigue

        if risk_score >= 4:
            risk_level = "High Risk"
            st.error(f"⚠️ {name}, High Risk! Please consult a doctor.")
        elif risk_score >= 2:
            risk_level = "Moderate Risk"
            st.warning(f"⚠️ {name}, Moderate Risk. Take precautions.")
        else:
            risk_level = "Low Risk"
            st.success(f"✅ {name}, Low Risk. You are healthy.")

        # ---------------- ACCURACY ----------------
        st.subheader("📊 Model Accuracy")
        st.write(f"{accuracy * 100:.2f}%")

        # ---------------- GRAPH ----------------
        st.subheader("📈 Risk Comparison")

        values = [prob, 1 - prob]
        labels = ["Disease Risk", "No Disease"]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
        HEALTH REPORT
        ------------------------
        Name: {name}
        Age: {age}

        Disease Probability: {round(prob*100,2)}%

        Risk Level: {risk_level}

        Model Accuracy: {accuracy * 100:.2f}%
        """

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="health_report.txt",
            mime="text/plain"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("💡 Developed as a Mini Project using Streamlit & Machine Learning")
