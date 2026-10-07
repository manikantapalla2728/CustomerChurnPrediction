import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Customer Dataset",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_PATH = Path(__file__).resolve().parents[2]
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
    max-width: 1500px;
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
[data-testid="stMetric"] {
    padding: 0.8rem 1rem;
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    background: rgba(11, 23, 40, 0.76);
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button {
    border: 1px solid rgba(125, 211, 252, 0.35);
    border-radius: 10px;
    background: linear-gradient(110deg, #0ea5e9, #2563eb);
    color: white;
    font-weight: 650;
    min-height: 2.7rem;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover {
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
        <div class="hero-eyebrow">Data explorer</div>
        <div class="hero-title">Telco customer dataset</div>
        <p class="hero-copy">Search customer records, filter by churn status, and download the visible rows for further analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_customers = len(df)
churned_customers = int(df["Churn"].eq("Yes").sum())
churn_rate = churned_customers / total_customers if total_customers else 0
column_count = len(df.columns)

metric_columns = st.columns(3, gap="medium")
metric_columns[0].metric("Customer records", f"{total_customers:,}")
metric_columns[1].metric("Churn rate", f"{churn_rate:.1%}")
metric_columns[2].metric("Dataset fields", f"{column_count}")

st.markdown("### Browse records")
filter_columns = st.columns([1, 2], gap="medium")
with filter_columns[0]:
    churn_filter = st.selectbox(
        "Churn status",
        ["All customers", "Churned", "Active"],
    )
with filter_columns[1]:
    search_text = st.text_input(
        "Search records",
        placeholder="Search any field, e.g. contract, payment method, or customer ID",
    )

filtered_df = df
if churn_filter == "Churned":
    filtered_df = filtered_df.loc[filtered_df["Churn"].eq("Yes")]
elif churn_filter == "Active":
    filtered_df = filtered_df.loc[filtered_df["Churn"].eq("No")]

if search_text.strip():
    matching_rows = filtered_df.astype(str).apply(
        lambda column: column.str.contains(search_text.strip(), case=False, na=False)
    ).any(axis=1)
    filtered_df = filtered_df.loc[matching_rows]

st.caption(f"Showing {len(filtered_df):,} of {total_customers:,} customer records.")
st.dataframe(filtered_df, width="stretch", hide_index=True)

download_column, navigation_column = st.columns([1, 5], vertical_alignment="center")
with download_column:
    st.download_button(
        "Download visible rows",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="telco-customer-churn-filtered.csv",
        mime="text/csv",
    )
with navigation_column:
    if st.button("Back to dashboard"):
        st.switch_page("app.py")
