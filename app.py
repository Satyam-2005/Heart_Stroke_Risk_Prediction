import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("Columns.pkl")

# --- Page Config ---
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8f9fa, #e3f2fd, #f1f8e9);
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 26px;
        transition: 0.3s;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        transform: scale(1.07);
    }
    .input-card {
        background: rgba(255,255,255,0.9);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
        margin-bottom: 18px;
    }
    .result-box {
        background: rgba(255,255,255,0.95);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
        font-size: 22px;
        font-weight: bold;
        animation: fadeIn 1.2s;
    }
    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        color: #555;
    }
    .footer b {
        color: #ff416c;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    💖 **Heart Stroke Risk Prediction App**  
    Enter your health details and let AI estimate your risk.  

    ⚠️ *Disclaimer*: This is not medical advice.  
    Please consult a doctor for a professional diagnosis.
    """
)

# --- Title ---
st.title("💖 Heart Stroke Risk Prediction")
st.markdown("#### Provide your health details below and check your **risk level** 👇")

# --- Layout with Cards ---
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        age = st.slider("🧑 Age", 18, 100, 40)
        sex = st.selectbox("⚧ Sex", ["M", "F"])
        chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("🥓 Cholesterol (mg/dL)", 100, 600, 200)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("❤️ Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("📊 ST Slope", ["Up", "Flat", "Down"])
        st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Button ---
if st.button("🔍 Predict Risk"):

    # Create input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    # --- Show Result ---
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="result-box" style="color:#b30000;">
            ⚠️ **High Risk of Heart Disease**  
            🔴 Risk Probability: <b>{prob*100:.2f}%</b>  
            👉 Please consult a doctor immediately.
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(prob*100))
    else:
        st.markdown(f"""
        <div class="result-box" style="color:#006600;">
            ✅ **Low Risk of Heart Disease**  
            🟢 Risk Probability: <b>{prob*100:.2f}%</b>  
            🎉 Keep maintaining a healthy lifestyle! 💪
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(prob*100))

# --- Footer ---
st.markdown("""
    <div class="footer">
        🚀 Developed by <b>Satyam Raj Mahakul</b>
    </div>
""", unsafe_allow_html=True)
