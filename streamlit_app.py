# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import os

# -----------------------
# 1. Streamlit page config
# -----------------------
st.set_page_config(page_title="AI-Powered Data Quality & Anomaly Detection Dashboard",
                   layout="wide")
sns.set(style="whitegrid")
st.title("AI-Powered Data Quality & Anomaly Detection Dashboard")

# -----------------------
# 2. CSV Input: Upload or fixed path
# -----------------------
st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_fixed_path = st.sidebar.checkbox("Use fixed CSV path instead", value=False)

# Fixed path (update this to your file)
fixed_file_path = r"C:\Users\SURYA\Downloads\synthetic_sample_data_for_automation_framework.csv"

# Load the CSV
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_fixed_path:
    if os.path.exists(fixed_file_path):
        df = pd.read_csv(fixed_file_path)
        st.sidebar.success(f"Loaded fixed file: {fixed_file_path}")
    else:
        st.error(f"Fixed file not found: {fixed_file_path}")
        st.stop()
else:
    st.info("Please upload a CSV or enable 'Use fixed CSV path' checkbox.")
    st.stop()

st.subheader("Raw Data")
st.dataframe(df.head(10))

# -----------------------
# 3a. Null Counts
# -----------------------
st.subheader("Null Counts per Column")
null_counts = df.isnull().sum()
null_pdf = pd.DataFrame({"Column": null_counts.index, "NullCount": null_counts.values})
st.dataframe(null_pdf)

# Plot null counts
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x="Column", y="NullCount", data=null_pdf, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# -----------------------
# 3b. Deduplication
# -----------------------
if "id" in df.columns and "join_date" in df.columns:
    df = df.sort_values("join_date", ascending=False).drop_duplicates(subset=["id"])

# -----------------------
# 3c. Business Rules
# -----------------------
dq_col = "_dq_failed_rules"
df[dq_col] = ""

required_cols = ["id", "user_id"]
for rc in required_cols:
    df.loc[df[rc].isnull() | (df[rc] == ""), dq_col] = dq_col + rc + "_required"

numeric_cols = ["amount", "price", "qty", "fare", "distance", "value"]
for nc in numeric_cols:
    if nc in df.columns:
        df.loc[df[nc] < 0, dq_col] = dq_col + nc + "_non_negative"

# Separate clean and failed rows
clean_df = df[df[dq_col] == ""].copy()
failed_df = df[df[dq_col] != ""].copy()

st.subheader("Business Rule Failures")
st.write(f"Total Rows: {len(df)}, Failing Business Rules: {len(failed_df)}")
st.dataframe(failed_df.head(10))

# -----------------------
# 4. AI Anomaly Detection
# -----------------------
st.subheader("Anomaly Detection (IsolationForest)")
feature_cols = [c for c in numeric_cols if c in clean_df.columns]

if len(feature_cols) > 0 and len(clean_df) > 0:
    pdf = clean_df[feature_cols].fillna(0)

    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(pdf)

    pdf["__anomaly_score"] = clf.decision_function(pdf[feature_cols])
    pdf["__is_anomaly"] = clf.predict(pdf[feature_cols]) == -1

    st.subheader("Anomaly Scores Distribution")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(pdf["__anomaly_score"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplots for numeric columns
    st.subheader("Boxplots with Anomalies Highlighted")
    for col_name in feature_cols:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.boxplot(x="__is_anomaly", y=col_name, data=pdf, ax=ax)
        ax.set_xlabel("Is Anomaly")
        st.pyplot(fig)

    # Top anomalies
    st.subheader("Top 10 Most Anomalous Records")
    top_anomalies = pdf.sort_values("__anomaly_score").head(10)
    st.dataframe(top_anomalies)

# -----------------------
# 5. Save outputs (optional)
# -----------------------
if st.sidebar.button("Save Outputs"):
    output_dir = "outputs_dashboard"
    os.makedirs(output_dir, exist_ok=True)
    clean_df.to_parquet(os.path.join(output_dir, "clean.parquet"), index=False)
    failed_df.to_parquet(os.path.join(output_dir, "dq_failed.parquet"), index=False)
    if len(feature_cols) > 0 and len(clean_df) > 0:
        pdf.to_parquet(os.path.join(output_dir, "scored.parquet"), index=False)
    st.success(f"Outputs saved under {output_dir}/")
