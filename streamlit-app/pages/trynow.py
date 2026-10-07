import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predict Customer Churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "models" / "Logistic_Regression.pkl",
    "Decision Tree": BASE_DIR / "models" / "DecisionTreeModel.pkl",
    "K-Nearest Neighbors": BASE_DIR / "models" / "KNN.pkl",
    "Random Forest": BASE_DIR / "models" / "RandomForest.pkl",
    "XGBoost": BASE_DIR / "models" / "XGBoost.pkl",
}


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(37, 99, 235, 0.16), transparent 32%),
        linear-gradient(160deg, #07111f 0%, #0b1424 55%, #0a1728 100%);
}
[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.72);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b192b, #091321);
    border-right: 1px solid rgba(148, 163, 184, 0.16);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #a9bbd0;
}
[data-testid="stSidebarNav"] {
    padding-top: 0.75rem;
}
[data-testid="stSidebarNav"] a {
    border-radius: 9px;
    margin: 0.12rem 0.55rem;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(14, 165, 233, 0.17);
    color: #7dd3fc;
}
.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(15, 35, 61, 0.96), rgba(13, 25, 43, 0.9));
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.2);
}
.hero-eyebrow {
    color: #7dd3fc;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.hero-title {
    color: #f8fafc;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    font-weight: 750;
    line-height: 1.15;
    margin: 0.55rem 0;
}
.hero-copy {
    color: #a9bbd0;
    font-size: 1rem;
    margin-bottom: 0;
}
.form-heading {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 650;
    margin: 0.3rem 0 0.9rem;
}
[data-testid="stForm"] {
    padding: 1.2rem 1.35rem;
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    background: rgba(11, 23, 40, 0.76);
}
[data-testid="stFormSubmitButton"] button {
    border: 1px solid rgba(125, 211, 252, 0.35);
    border-radius: 10px;
    background: linear-gradient(110deg, #0ea5e9, #2563eb);
    color: white;
    font-weight: 650;
    min-height: 2.8rem;
}
[data-testid="stFormSubmitButton"] button:hover {
    border-color: #7dd3fc;
    color: white;
    filter: brightness(1.1);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="padding: 0.6rem 0 1rem;">
        <div style="color:#7dd3fc;font-size:0.72rem;font-weight:700;letter-spacing:0.15em;">CHURN / INSIGHTS</div>
        <div style="color:#f8fafc;font-size:1.05rem;font-weight:700;margin-top:0.4rem;">Customer analytics</div>
        <div style="color:#8fa3ba;font-size:0.8rem;margin-top:0.3rem;">Project by Manikanta Palla</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Prediction studio</div>
        <div class="hero-title">Estimate churn risk</div>
        <p class="hero-copy">Enter a customer profile and choose a trained model to estimate the likelihood of churn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== User Form ============================
with st.form("Try Now Form"):
    st.markdown('<div class="form-heading">Customer profile</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        Name = st.text_input("Enter Your Name", placeholder="Enter Your Name Here")
        
        SeniorCitizen = st.selectbox(
            "Are you a senior citizen (60+ years)?",
            [0, 1],
            index=None,
            placeholder="Select Option"
        )

        tenure = st.number_input(
            "How many months have you been using the service?",
            min_value=0,
            max_value=100,
            step=1,
            value=None,
            placeholder="Enter number of months"
        )

        PhoneService = st.selectbox(
            "Do you have phone service?",
            ["Yes", "No"],
            index=None,
            placeholder="Select Option"
        )

        InternetService = st.selectbox(
            "What type of internet service do you use?",
            ["DSL", "Fiber optic", "No"],
            index=None,
            placeholder="Select Internet Service"
        )

        OnlineBackup = st.selectbox(
            "Do you have online backup service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )

        StreamingTV = st.selectbox(
            "Do you use TV streaming service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )

        Contract = st.selectbox(
            "What type of contract do you have?",
            ["Month-to-month", "One year", "Two year"],
            index=None,
            placeholder="Select Contract Type"
        )

        PaymentMethod = st.selectbox(
            "Which payment method do you use?",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=None,
            placeholder="Select Payment Method"
        )

        OnlineSecurity = st.selectbox(
            "Do you have online security service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )


    with col2:
        st.markdown('<div class="form-heading">Services and billing</div>', unsafe_allow_html=True)
        gender = st.selectbox(
            "What is your gender?",
            ["Male", "Female"],
            index=None,
            placeholder="Select Gender"
        )

        Partner = st.selectbox(
            "Do you have a partner/spouse?",
            ["Yes", "No"],
            index=None,
            placeholder="Select Option"
        )

        Dependents = st.selectbox(
            "Do you have dependents?",
            ["Yes", "No"],
            index=None,
            placeholder="Select Option"
        )

        MultipleLines = st.selectbox(
            "Do you use multiple phone lines?",
            ["Yes", "No", "No phone service"],
            index=None,
            placeholder="Select Option"
        )

        DeviceProtection = st.selectbox(
            "Do you have device protection service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )

        TechSupport = st.selectbox(
            "Do you have technical support service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )

        StreamingMovies = st.selectbox(
            "Do you use movie streaming service?",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select Option"
        )

       
        PaperlessBilling = st.selectbox(
            "Do you receive bills electronically (paperless billing)?",
            ["Yes", "No"],
            index=None,
            placeholder="Select Option"
        )

        MonthlyCharges = st.number_input(
            "What is your monthly bill amount ($)?",
            min_value=0.0,
            step=0.01,
            value=None,
            placeholder="Enter monthly charges"
        )
        algorithms = st.selectbox(
            "Choose a prediction model",
            [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors",
                "Random Forest",
                "XGBoost",
            ],
            index=None,
            placeholder="Select Model"
        )

    btn = st.form_submit_button("Submit")




required_fields = [
    gender, SeniorCitizen, Partner, Dependents,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod, algorithms,
    Name, MonthlyCharges, tenure
]

if btn:

    if any(x is None for x in required_fields):
        st.error("Please fill all fields before prediction.")
        st.stop()

    # Prediction

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": tenure * MonthlyCharges
    })

    try:
        selected_model = load_model(algorithms)
    except ModuleNotFoundError as exc:
        if exc.name != "xgboost":
            raise
        st.error(
            "The XGBoost model requires the `xgboost` package. Install it in the "
            f"Python environment running Streamlit: `{sys.executable} -m pip install xgboost`."
        )
        st.stop()

    prediction = selected_model.predict(input_df)[0]
    probability = selected_model.predict_proba(input_df)[0][1]

    st.subheader(f"Prediction for {Name}")
    st.caption(f"Model used: {algorithms}")
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(float(probability), text="Churn probability")

    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")