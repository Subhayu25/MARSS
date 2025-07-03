import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    layout="wide",
    page_title="Marketing Survey & Campaign Dashboard",
    page_icon="📊"
)

# Sidebar: Uploads & Filters
st.sidebar.title("🔧 Uploads & Filters")

uploaded_csv = st.sidebar.file_uploader(
    "Upload survey CSV file",
    type=["csv"],
    help="Upload your own survey data in CSV format. If not provided, the default dataset is used."
)

uploaded_excel = st.sidebar.file_uploader(
    "Upload campaign Excel file (xlsx)",
    type=["xlsx"],
    help="Must contain sheet 'marketing_campaign_dataset' with a Date column."
)

@st.cache_data
def load_survey(uploaded_csv):
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return pd.DataFrame()
    else:
        try:
            df = pd.read_csv("synthetic_consumer_marketing_survey.csv")
        except FileNotFoundError:
            st.error("File 'synthetic_consumer_marketing_survey.csv' not found and no CSV uploaded.")
            return pd.DataFrame()
    if "Purchase_Last_3mo" not in df.columns:
        st.error("'Purchase_Last_3mo' column missing from CSV.")
        return pd.DataFrame()
    df = df.dropna(subset=["Purchase_Last_3mo"])
    return df

survey_df = load_survey(uploaded_csv)

if survey_df.empty:
    st.stop()

# Display the raw data (with expander)
with st.expander("See Raw Survey Data"):
    st.dataframe(survey_df)

# Show available columns in the sidebar for transparency & debugging
st.sidebar.write("**Available columns in dataset:**")
st.sidebar.write(list(survey_df.columns))

# Filters: Only show if columns are present!
if "Age" in survey_df.columns and "Gender" in survey_df.columns:
    st.sidebar.header("Survey Filters")
    age_min, age_max = int(survey_df['Age'].min()), int(survey_df['Age'].max())
    age_filter = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=survey_df["Gender"].unique(),
        default=list(survey_df["Gender"].unique())
    )
    filtered_df = survey_df[
        (survey_df["Age"] >= age_filter[0]) &
        (survey_df["Age"] <= age_filter[1]) &
        (survey_df["Gender"].isin(gender_filter))
    ]
else:
    st.sidebar.warning("Your dataset must contain columns named 'Age' and 'Gender' for filtering. Filters are disabled.")
    filtered_df = survey_df.copy()  # No filters applied

st.markdown("## 📈 Dashboard: Key Insights")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Responses", len(filtered_df))
with col2:
    if "Satisfaction_Score" in filtered_df.columns:
        st.metric("Avg Satisfaction", round(filtered_df["Satisfaction_Score"].mean(),2))
    else:
        st.metric("Avg Satisfaction", "N/A")
with col3:
    st.metric("Purchase Rate (Last 3mo)", f"{filtered_df['Purchase_Last_3mo'].mean()*100:.1f}%")

# Basic Visualizations
if "Gender" in filtered_df.columns:
    st.markdown("### Demographic Breakdown")
    dem_col1, dem_col2 = st.columns(2)

    with dem_col1:
        fig1, ax1 = plt.subplots()
        gender_counts = filtered_df["Gender"].value_counts()
        ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Gender Distribution")
        st.pyplot(fig1)

    if "Age" in filtered_df.columns:
        with dem_col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(filtered_df["Age"], kde=True, bins=20, ax=ax2)
            ax2.set_title("Age Distribution")
            st.pyplot(fig2)

if "Satisfaction_Score" in filtered_df.columns:
    st.markdown("### Satisfaction Score Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df["Satisfaction_Score"], kde=True, bins=10, ax=ax3)
    ax3.set_title("Customer Satisfaction Score")
    st.pyplot(fig3)

# Purchase behavior by channel (if 'Acquisition_Channel' exists)
if 'Acquisition_Channel' in filtered_df.columns:
    st.markdown("### Purchase by Acquisition Channel")
    channel_purchase = filtered_df.groupby("Acquisition_Channel")["Purchase_Last_3mo"].mean().reset_index()
    fig4 = px.bar(channel_purchase, x="Acquisition_Channel", y="Purchase_Last_3mo", labels={"Purchase_Last_3mo": "Purchase Rate"})
    st.plotly_chart(fig4, use_container_width=True)

# Example: Cluster Analysis (if enough features are present)
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 3:
    st.markdown("### Customer Segmentation (KMeans)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data = filtered_df[numeric_cols].fillna(0)
    clusters = kmeans.fit_predict(cluster_data)
    pca = PCA(n_components=2)
    cluster_2d = pca.fit_transform(cluster_data)
    fig5 = px.scatter(
        x=cluster_2d[:,0], y=cluster_2d[:,1], color=clusters.astype(str),
        labels={"x":"PCA1","y":"PCA2","color":"Cluster"}
    )
    st.plotly_chart(fig5, use_container_width=True)

# If Excel file uploaded: process campaign data
if uploaded_excel is not None:
    try:
        campaign_df = pd.read_excel(uploaded_excel, sheet_name="marketing_campaign_dataset")
        st.markdown("## 📊 Marketing Campaign Data")
        st.dataframe(campaign_df.head())
        # Example: Show campaigns over time if 'Date' column exists
        if "Date" in campaign_df.columns:
            campaign_df["Date"] = pd.to_datetime(campaign_df["Date"])
            st.line_chart(campaign_df.set_index("Date").select_dtypes(include=np.number))
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

st.markdown("---")
st.info("Enhance this dashboard with more filters, charts, and ML predictions as needed!")
