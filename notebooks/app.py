import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="E-commerce Analytics System", layout="wide")
st.title("E-commerce Analytics Dashboard")
st.markdown("---")

# -------------------------------
# 2. LOAD & CLEAN DATA
# -------------------------------
@st.cache_data
def load_and_fix_data():
    df = pd.read_csv('cleaned_customer_data.csv')
    df.columns = df.columns.str.strip().str.lower()

    numeric_keywords = ['spend', 'item', 'age', 'satisfaction', 'score', 'rating']

    for col in df.columns:
        if any(key in col for key in numeric_keywords):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9.]', '', regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = load_and_fix_data()
st.write(df.columns)  # Debug: Check column names after loading and fixing data

# Identify key columns
spend_col = next((c for c in df.columns if 'spend' in c), None)
age_col = next((c for c in df.columns if 'age' in c), None)
item_col = next((c for c in df.columns if 'item' in c), None)
sat_col = 'satisfaction_score' if 'satisfaction_score' in df.columns else None
if sat_col:
    df[sat_col] = pd.to_numeric(df[sat_col], errors='coerce')

# -------------------------------
# 3. SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filters")

if 'city' in df.columns:
    cities = st.sidebar.multiselect(
        "Select Cities",
        df['city'].dropna().unique(),
        default=df['city'].dropna().unique()
    )
    df = df[df['city'].isin(cities)]

# -------------------------------
# 4. KPIs
# -------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", len(df))

if spend_col:
    c2.metric("Avg Spend", f"${df[spend_col].dropna().mean():,.2f}")

if sat_col in df.columns:
    avg_sat = df[sat_col].dropna().mean()
    
    if pd.isna(avg_sat):
        c3.metric("Customer Satisfaction", "No valid data")
    else:
        c3.metric("Customer Satisfaction", f"{avg_sat:.2f} / 5")

# -------------------------------
# 5. MAIN SCATTER
# -------------------------------
st.subheader("Customer Behavior")

fig = px.scatter(
    df,
    x=item_col,
    y=spend_col,
    color=sat_col,
    size=df[age_col] if age_col else None,
    title="Spending vs Purchase Volume"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 6. 📈 TREND ANALYSIS
# -------------------------------
st.subheader("Spending Trend")

if spend_col:
    trend_df = df[[spend_col]].dropna()
    trend_df['index'] = range(len(trend_df))

    fig_trend = px.line(
        trend_df,
        x='index',
        y=spend_col,
        title="Spending Trend Over Customers"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------------
# 7. 🧠 CUSTOMER SEGMENTATION (K-Means)
# -------------------------------
st.subheader("Customer Segmentation")

if spend_col and age_col:
    cluster_df = df[[spend_col, age_col]].dropna()

    k = st.slider("Select Number of Clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_df['cluster'] = kmeans.fit_predict(cluster_df)

    fig_cluster = px.scatter(
        cluster_df,
        x=age_col,
        y=spend_col,
        color=cluster_df['cluster'].astype(str),
        title="Customer Segments"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

# -------------------------------
# 8. 🎯 SPENDING PREDICTION (ML)
# -------------------------------
st.subheader("Predict Customer Spend")

if age_col and spend_col:
    model_df = df[[age_col, spend_col]].dropna()

    X = model_df[[age_col]]
    y = model_df[spend_col]

    model = LinearRegression()
    model.fit(X, y)

    age_input = st.slider("Select Age", 18, 70, 25)

    prediction = model.predict([[age_input]])

    st.success(f"Predicted Spend: ${prediction[0]:,.2f}")

# -------------------------------
# 9. DATA PREVIEW
# -------------------------------
with st.expander("View Dataset"):
    st.dataframe(df)