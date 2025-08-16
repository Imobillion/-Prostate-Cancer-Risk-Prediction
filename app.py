import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("prostate_risk_model.pkl")

st.set_page_config(page_title="Prostate Cancer Risk Prediction", layout="wide")

st.title("🔬 Prostate Cancer Risk Prediction")
st.write("Fill in the details below to predict your prostate cancer risk.")

# Collect user input
age = st.number_input("Age", 18, 100, 55)
bmi = st.number_input("BMI", 10.0, 50.0, 27.5)
smoker = st.selectbox("Smoker", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ["Low", "Moderate", "High"])
diet = st.selectbox("Diet Type", ["Healthy", "Mixed", "Fatty"])
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
family = st.selectbox("Family History of Cancer", ["Yes", "No"])
stress = st.selectbox("Mental Stress Level", ["Low", "Medium", "High"])
sleep = st.number_input("Sleep Hours per Night", 0.0, 12.0, 7.0)
checkup = st.selectbox("Regular Health Checkup", ["Yes", "No"])
exam = st.selectbox("Prostate Exam Done", ["Yes", "No"])

# Prediction button
if st.button("Predict Risk"):
    # ✅ FIX: Add dummy ID column so model doesn’t crash
    input_data = pd.DataFrame([{
        "id": 0,  # Dummy placeholder
        "age": age,
        "bmi": bmi,
        "smoker": smoker,
        "alcohol_consumption": alcohol,
        "diet_type": diet,
        "physical_activity_level": activity,
        "family_history": family,
        "mental_stress_level": stress,
        "sleep_hours": sleep,
        "regular_health_checkup": checkup,
        "prostate_exam_done": exam
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Show results
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Prostate Cancer (Probability: {probabilities[1]:.2f})")
    else:
        st.success(f"✅ Low Risk of Prostate Cancer (Probability: {probabilities[0]:.2f})")

    # Interpretation
    st.info("ℹ️ Interpretation: This prediction is **not a medical diagnosis**. "
            "It is based on statistical patterns from the training data. "
            "Consult a medical professional for proper screening and advice.")
