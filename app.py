import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
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
    help="Upload your own survey data in CSV format."
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
    if "Purchase_Last_3mo" in df.columns:
        df = df.dropna(subset=["Purchase_Last_3mo"])
    return df

survey_df = load_survey(uploaded_csv)

if survey_df.empty:
    st.warning("No data loaded. Please upload a CSV or ensure the default dataset is available.")
    st.stop()

# Display the raw data (with expander)
with st.expander("See Raw Survey Data"):
    st.dataframe(survey_df)

# Show available columns in the sidebar
st.sidebar.write("**Available columns in dataset:**")
st.sidebar.write(list(survey_df.columns))

# Filters (check columns first)
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
    if "Age" not in survey_df.columns or "Gender" not in survey_df.columns:
        st.sidebar.warning("No 'Age' and/or 'Gender' column(s) for filtering. All rows shown.")
    filtered_df = survey_df.copy()

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
    if "Purchase_Last_3mo" in filtered_df.columns:
        filtered_df['Purchase_Last_3mo_numeric'] = pd.to_numeric(filtered_df['Purchase_Last_3mo'], errors='coerce')
        if filtered_df['Purchase_Last_3mo_numeric'].isna().all():
            mapping = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}
            filtered_df['Purchase_Last_3mo_numeric'] = filtered_df['Purchase_Last_3mo'].map(mapping)
        purchase_rate = filtered_df['Purchase_Last_3mo_numeric'].mean()
        if purchase_rate is not None and not np.isnan(purchase_rate):
            st.metric("Purchase Rate (Last 3mo)", f"{purchase_rate*100:.1f}%")
        else:
            st.metric("Purchase Rate (Last 3mo)", "N/A")
    else:
        st.metric("Purchase Rate (Last 3mo)", "N/A")

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
elif "Age" in filtered_df.columns:
    st.markdown("### Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df["Age"], kde=True, bins=20, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)
else:
    st.info("No 'Age' or 'Gender' columns for demographic breakdown.")

if "Satisfaction_Score" in filtered_df.columns:
    st.markdown("### Satisfaction Score Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df["Satisfaction_Score"], kde=True, bins=10, ax=ax3)
    ax3.set_title("Customer Satisfaction Score")
    st.pyplot(fig3)
else:
    st.info("No 'Satisfaction_Score' column for satisfaction distribution.")

# Purchase behavior by channel
if 'Acquisition_Channel' in filtered_df.columns and 'Purchase_Last_3mo_numeric' in filtered_df.columns:
    st.markdown("### Purchase by Acquisition Channel")
    channel_purchase = filtered_df.groupby("Acquisition_Channel")["Purchase_Last_3mo_numeric"].mean().reset_index()
    fig4 = px.bar(channel_purchase, x="Acquisition_Channel", y="Purchase_Last_3mo_numeric", labels={"Purchase_Last_3mo_numeric": "Purchase Rate"})
    st.plotly_chart(fig4, use_container_width=True)

# KMeans clustering
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
else:
    st.info("Not enough numeric columns for KMeans clustering.")

# REGRESSION SECTION
st.markdown("---")
st.header("🔬 Regression Modeling (Lasso, Ridge, Decision Tree)")

if len(numeric_cols) >= 2:
    reg_candidates = [col for col in numeric_cols if filtered_df[col].nunique() > 5]
    if reg_candidates:
        reg_target = st.selectbox(
            "Select target variable for regression",
            options=reg_candidates,
            help="Target variable should be continuous/numeric."
        )
        reg_features = [col for col in numeric_cols if col != reg_target]
        if reg_target and reg_features:
            X = filtered_df[reg_features].fillna(0)
            y = filtered_df[reg_target].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            results = {}
            models = {
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "Decision Tree Regression": DecisionTreeRegressor(random_state=42)
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = {
                    "r2": r2_score(y_test, preds),
                    "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                    "y_test": y_test,
                    "preds": preds
                }
            res_df = pd.DataFrame({
                "Model": list(results.keys()),
                "R² Score": [results[m]["r2"] for m in results],
                "RMSE": [results[m]["rmse"] for m in results]
            })
            st.dataframe(res_df)
            for name in models.keys():
                st.subheader(f"{name}: Actual vs. Predicted")
                fig, ax = plt.subplots()
                ax.scatter(results[name]["y_test"], results[name]["preds"], alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{name} - {reg_target}")
                st.pyplot(fig)
        else:
            st.info("Not enough numeric features for regression modeling.")
    else:
        st.info("No suitable numeric column with enough unique values for regression modeling.")
else:
    st.info("Not enough numeric columns for regression modeling.")

# MARKETING CAMPAIGN DATA
if uploaded_excel is not None:
    try:
        campaign_df = pd.read_excel(uploaded_excel, sheet_name="marketing_campaign_dataset")
        st.markdown("## 📊 Marketing Campaign Data")
        st.dataframe(campaign_df.head())
        if "Date" in campaign_df.columns:
            campaign_df["Date"] = pd.to_datetime(campaign_df["Date"])
            st.line_chart(campaign_df.set_index("Date").select_dtypes(include=np.number))
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

st.markdown("---")
st.info("Enhance this dashboard with more filters, charts, and ML predictions as needed!")
