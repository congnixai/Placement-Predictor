import streamlit as st
import pickle
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered",
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.main {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 2rem;
}
.title-text {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #4b0082;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.card {
    background: #f9f9ff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title-text">🎓 Placement Prediction System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Predict your campus placement chances using AI</div>',
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        iq = st.number_input("🧠 IQ Score", min_value=50, max_value=200, step=1)

    with col2:
        cgpa = st.number_input("📚 CGPA", min_value=0.0, max_value=10.0, step=0.1)

    st.write("")

    predict_btn = st.button("🚀 Predict Placement")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:
    with st.spinner("🔍 Analyzing your profile..."):
        time.sleep(1.5)

    input_data = np.array([[iq, cgpa]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.write("")
    st.markdown("### 📊 Prediction Result")

    progress_bar = st.progress(0)
    for i in range(int(probability)):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    if prediction == 1:
        st.success(f"🎉 **Congratulations! You are likely to be PLACED**")
        st.metric(label="Placement Probability", value=f"{probability:.2f}%")
        st.balloons()
    else:
        st.error(f"⚠️ **Placement Not Likely at the Moment**")
        st.metric(label="Placement Probability", value=f"{probability:.2f}%")
        st.info("💡 Improve skills, projects & internships to increase chances.")

# ---------------- FOOTER ----------------
st.markdown(
    """
<div class="footer">
Made with ❤️ using Machine Learning & Streamlit<br>
© 2026 Placement Predictor
</div>
""",
    unsafe_allow_html=True,
)
