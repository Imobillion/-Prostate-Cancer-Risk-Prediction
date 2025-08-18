import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
model = joblib.load("prostate_risk_model_pipeline.pkl")

st.title("🧑‍⚕️ Prostate Cancer Risk Prediction App")

st.markdown("Fill in the details below to predict prostate cancer risk.")

# Sidebar user inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=50)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
alcohol = st.sidebar.selectbox("Alcohol Consumption", [0, 1, 2])  # 0=none,1=moderate,2=high
diet = st.sidebar.selectbox("Diet Type", [0, 1, 2])  # e.g. 0=poor,1=average,2=healthy
activity = st.sidebar.selectbox("Physical Activity Level", [0, 1, 2])  # 0=low,1=moderate,2=high
family_history = st.sidebar.selectbox("Family History of Cancer", [0, 1])
stress = st.sidebar.slider("Mental Stress Level", min_value=0, max_value=3, value=1)
sleep_hours = st.sidebar.number_input("Average Sleep Hours", min_value=3.0, max_value=12.0, value=7.0)
checkup = st.sidebar.selectbox("Regular Health Checkup", [0, 1])
exam_done = st.sidebar.selectbox("Prostate Exam Done", [0, 1])

# Collect inputs into DataFrame
user_input = pd.DataFrame([{
    "id": 0,  # dummy column required by your trained model
    "age": age,
    "bmi": bmi,
    "smoker": smoker,
    "alcohol_consumption": alcohol,
    "diet_type": diet,
    "physical_activity_level": activity,
    "family_history": family_history,
    "mental_stress_level": stress,
    "sleep_hours": sleep_hours,
    "regular_health_checkup": checkup,
    "prostate_exam_done": exam_done
}])

# Ensure column order matches training
expected_features = [
    "id", "age", "bmi", "smoker", "alcohol_consumption", "diet_type",
    "physical_activity_level", "family_history", "mental_stress_level",
    "sleep_hours", "regular_health_checkup", "prostate_exam_done"
]
user_input = user_input[expected_features]

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0]

    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success(f"Low Risk (Probability: {prediction_proba[0]:.2f}) ✅")
    elif prediction == 1:
        st.warning(f"Medium Risk (Probability: {prediction_proba[1]:.2f}) ⚠️")
    else:
        st.error(f"High Risk (Probability: {prediction_proba[2]:.2f}) 🚨")

# PREDICTION
# ------------------------------
if st.button("🔍 Predict Prostate Cancer Risk"):
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0]

    risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

    st.subheader("🔮 Prediction Result")
    st.success(f"**Predicted Risk Level:** {risk_levels[prediction]}")

    st.write("### 📊 Risk Probability Distribution")
    st.progress(int(prediction_proba[prediction] * 100))
    st.write(f"- Low Risk: {prediction_proba[0]*100:.2f}%")
    st.write(f"- Medium Risk: {prediction_proba[1]*100:.2f}%")
    st.write(f"- High Risk: {prediction_proba[2]*100:.2f}%")

    # ------------------------------
    # INTERPRETATION
    # ------------------------------
    st.write("### 🧠 Interpretation")
    if prediction == 0:
        st.info("✅ The patient is at **low risk**. Encourage maintaining a healthy lifestyle and regular checkups.")
    elif prediction == 1:
        st.warning("⚠️ The patient is at **medium risk**. Lifestyle adjustments and frequent screenings are recommended.")
    else:
        st.error("🚨 The patient is at **high risk**. Immediate medical attention and advanced diagnostic tests are advised.")