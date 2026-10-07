import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Machine Learning-Based Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_PATH = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_PATH / "data" / "Churn-Prediction-dataset.csv")

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
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
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
    font-size: clamp(1.8rem, 3vw, 2.65rem);
    font-weight: 750;
    line-height: 1.15;
    margin: 0.55rem 0;
}

.hero-copy {
    color: #a9bbd0;
    font-size: 1rem;
    margin-bottom: 0;
}

.section-heading {
    color: #e2e8f0;
    font-size: 1.18rem;
    font-weight: 650;
    margin: 1.5rem 0 0.2rem;
}

.section-copy {
    color: #8fa3ba;
    font-size: 0.9rem;
    margin: 0 0 0.8rem;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(17, 34, 56, 0.98), rgba(12, 25, 43, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-left: 3px solid var(--accent);
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 104px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.16);
    transition: transform 160ms ease, border-color 160ms ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(125, 211, 252, 0.38);
}

.kpi-title {
    color: #9fb1c6;
    font-size: 0.77rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

.kpi-value {
    color: #f8fafc;
    font-size: clamp(1.45rem, 2vw, 1.9rem);
    font-weight: 700;
    line-height: 1.1;
}

.kpi-icon {
    align-items: center;
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    border-radius: 12px;
    color: var(--accent);
    display: flex;
    font-size: 1.35rem;
    height: 44px;
    justify-content: center;
    width: 44px;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 14px;
    background: rgba(11, 23, 40, 0.72);
    padding: 0.35rem;
}

[data-testid="stButton"] > button {
    border: 1px solid rgba(125, 211, 252, 0.35);
    border-radius: 10px;
    background: linear-gradient(110deg, #0ea5e9, #2563eb);
    color: white;
    font-weight: 650;
    min-height: 2.8rem;
    transition: filter 150ms ease, transform 150ms ease;
}

[data-testid="stButton"] > button:hover {
    border-color: #7dd3fc;
    color: white;
    filter: brightness(1.1);
    transform: translateY(-1px);
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
        <div class="hero-eyebrow">Customer retention · Analytics</div>
        <div class="hero-title">Customer Churn Prediction</div>
        <p class="hero-copy">Explore customer trends, compare machine-learning models, and identify customers who may be at risk of leaving.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">Portfolio overview</div>'
    '<div class="section-copy">A snapshot of the Telco customer dataset.</div>',
    unsafe_allow_html=True,
)

total_customers = len(df)
churned_customers = int(df["Churn"].eq("Yes").sum())
active_customers = int(df["Churn"].eq("No").sum())
churn_rate = churned_customers / total_customers if total_customers else 0
average_tenure = pd.to_numeric(df["tenure"], errors="coerce").mean()
average_monthly_charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce").mean()

cards = [
    ("Total customers", f"{total_customers:,}", "#38bdf8", "01"),
    ("Churned customers", f"{churned_customers:,}", "#fb7185", "02"),
    ("Churn rate", f"{churn_rate:.1%}", "#fbbf24", "03"),
    ("Active customers", f"{active_customers:,}", "#34d399", "04"),
    ("Average tenure", f"{average_tenure:.1f} mo.", "#a78bfa", "05"),
    ("Avg. monthly charges", f"${average_monthly_charges:.2f}", "#60a5fa", "06"),
]

for row_start in range(0, len(cards), 3):
    cols = st.columns(3, gap="medium")
    for col, (title, value, color, icon) in zip(cols, cards[row_start:row_start + 3]):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card" style="--accent:{color}">
                    <div>
                        <div class="kpi-title">{title}</div>
                        <div class="kpi-value">{value}</div>
                    </div>
                    <div class="kpi-icon" aria-hidden="true">{icon}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="section-heading">Customer patterns</div>'
    '<div class="section-copy">Explore churn distribution and customer attributes.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    fig = px.pie(
        df,
        names="Churn",
        hole=0.35,
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.80,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Churn Distribution",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )
    st.plotly_chart(fig, width="stretch")


with col2:
    fig = px.bar(
        df,
        y=["Contract","Churn"],
        x=df["Churn"],
        barmode="group",
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.85,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Contract Type vs Churn",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )

    st.plotly_chart(fig,width="stretch", key="fig1")


with col3:
    fig = px.bar(
        df,
        x=df["Churn"],
        y=["gender","Churn"],
        barmode="group",
        color="Churn",
        color_discrete_map={
            "Yes": "#EF553B",
            "No": "#2f6df6"
        }
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#04142d",
        plot_bgcolor="#04142d",
        font_color="white",
        legend=dict(
            x=0.85,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        title={
            "text" : "Gender vs Churn",
            "x" : 0.05
        },
        margin=dict(
            l=50,
            r=50,
            t=40,
            b=10
        )
    )

    st.plotly_chart(fig,width="stretch", key="fig3")


st.markdown(
    '<div class="section-heading">Model performance</div>'
    '<div class="section-copy">Compare accuracy, precision, recall, F1, and probability-based ROC-AUC.</div>',
    unsafe_allow_html=True,
)

evaluation = pd.DataFrame({
    "Models": [
        "Logistic Regression",
        "Random Forest (Tuned)",
        "KNN (45, Euclidean)",
        "Decision Tree (Tuned)",
        "XGBoost",
    ],
    "Accuracy": [0.8034, 0.8070, 0.7842, 0.7956, 0.8020],
    "Precision": [0.6520, 0.6759, 0.6882, 0.6243, 0.6547],
    "Recall": [0.5561, 0.5241, 0.3422, 0.5775, 0.5374],
    "F1 Score": [0.6003, 0.5904, 0.4571, 0.6000, 0.5903],
    "ROC-AUC": [0.8428, 0.8433, 0.7899, 0.8215, 0.8416]
})

st.caption("All models were evaluated on the same stratified 80/20 test split (random state 42). ROC-AUC uses predicted probabilities.")

model_fig = px.bar(
    evaluation,
    x="Models",
    y=["Accuracy",'Precision','Recall','F1 Score','ROC-AUC'],
    barmode="group",
    labels={
        "Models" : "Machine Leaning Models",
        "value" : "Score",
        "variable" : "Matrics"
    },
    color="variable",
    color_discrete_map={
    "Accuracy": "#2f6df6",   # Plotly Blue
    "Precision": "#EF553B",  # Plotly Orange/Red
    "Recall": "#00CC96",     # Plotly Green
    "F1 Score": "#AB63FA",   # Plotly Purple
    "ROC-AUC": "#FFA15A"     # Plotly Orange
}
)
model_fig.update_layout(
    height=350,
    paper_bgcolor="#04142d",
    plot_bgcolor="#04142d",
    font_color="white",
    legend=dict(
        x=1,
        y=1,
        xanchor="left",
        yanchor="top"
    ),
    title={
        "text" : "Model Evaluation Matrics",
        "x" : 0.05
    },
    margin=dict(
        l=50,
        r=50,
        t=40,
        b=10
    )
)

st.plotly_chart(model_fig, width="stretch")

st.caption("Percentages may not total 100% because of rounding.")

dataset, try_now = st.columns([1,1])

with dataset:
    st.markdown("Explore the source customer records.")
    if st.button("View Dataset"):
        st.switch_page("pages/datasetpage.py")


with try_now:
    st.markdown("Estimate churn risk for a customer profile.")
    if st.button("Try Now"):
        st.switch_page("pages/trynow.py")