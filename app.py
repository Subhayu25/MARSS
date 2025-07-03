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

st.sidebar.write("**Available columns in dataset:**")
st.sidebar.write(list(survey_df.columns))

# --- Multi-filter: for all categorical columns ---
cat_cols = survey_df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = survey_df.select_dtypes(include=[np.number]).columns.tolist()
filters = {}
if cat_cols:
    st.sidebar.markdown("**Advanced Filters:**")
    for col in cat_cols:
        if survey_df[col].nunique() <= 25: # Only show filters for columns with not too many unique values
            opts = st.sidebar.multiselect(
                f"Filter by {col}", 
                options=survey_df[col].unique(), 
                default=list(survey_df[col].unique())
            )
            filters[col] = opts
# Numeric range sliders
if "Age" in survey_df.columns:
    age_min, age_max = int(survey_df['Age'].min()), int(survey_df['Age'].max())
    age_filter = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
else:
    age_filter = None

filtered_df = survey_df.copy()
for col, vals in filters.items():
    if col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col].isin(vals)]
if age_filter:
    filtered_df = filtered_df[(filtered_df["Age"] >= age_filter[0]) & (filtered_df["Age"] <= age_filter[1])]

# Prepare numeric purchase column for later use
if "Purchase_Last_3mo" in filtered_df.columns:
    filtered_df['Purchase_Last_3mo_numeric'] = pd.to_numeric(filtered_df['Purchase_Last_3mo'], errors='coerce')
    if filtered_df['Purchase_Last_3mo_numeric'].isna().all():
        mapping = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}
        filtered_df['Purchase_Last_3mo_numeric'] = filtered_df['Purchase_Last_3mo'].map(mapping)

# ---- TABS ----
tab_titles = [
    "Dashboard",
    "Demographics",
    "Satisfaction",
    "Correlation & Outliers",
    "Channel Analysis",
    "Segmentation",
    "ML Predictions",
    "Campaign Data",
    "Raw Data"
]
tabs = st.tabs(tab_titles)

with tabs[0]:
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
        if "Purchase_Last_3mo_numeric" in filtered_df.columns:
            purchase_rate = filtered_df['Purchase_Last_3mo_numeric'].mean()
            if purchase_rate is not None and not np.isnan(purchase_rate):
                st.metric("Purchase Rate (Last 3mo)", f"{purchase_rate*100:.1f}%")
            else:
                st.metric("Purchase Rate (Last 3mo)", "N/A")
        else:
            st.metric("Purchase Rate (Last 3mo)", "N/A")

with tabs[1]:
    st.markdown("### Demographic Breakdown")
    if "Gender" in filtered_df.columns or "Age" in filtered_df.columns:
        col1, col2 = st.columns(2)
        if "Gender" in filtered_df.columns:
            with col1:
                fig1, ax1 = plt.subplots()
                gender_counts = filtered_df["Gender"].value_counts()
                ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
                ax1.set_title("Gender Distribution")
                st.pyplot(fig1)
        if "Age" in filtered_df.columns:
            with col2:
                fig2, ax2 = plt.subplots()
                sns.histplot(filtered_df["Age"], kde=True, bins=20, ax=ax2)
                ax2.set_title("Age Distribution")
                st.pyplot(fig2)
    else:
        st.info("No 'Age' or 'Gender' columns for demographic breakdown.")

with tabs[2]:
    st.markdown("### Satisfaction Score Distribution")
    if "Satisfaction_Score" in filtered_df.columns:
        fig3, ax3 = plt.subplots()
        sns.histplot(filtered_df["Satisfaction_Score"], kde=True, bins=10, ax=ax3)
        ax3.set_title("Customer Satisfaction Score")
        st.pyplot(fig3)
    else:
        st.info("No 'Satisfaction_Score' column for satisfaction distribution.")

with tabs[3]:
    st.markdown("### Correlation & Outliers")
    if len(num_cols) >= 2:
        st.subheader("Correlation Heatmap")
        corr = filtered_df[num_cols].corr()
        figc, axc = plt.subplots(figsize=(8,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axc)
        st.pyplot(figc)
        st.subheader("Pairplot (sampled for speed)")
        sample_size = min(200, len(filtered_df))
        sns.pairplot(filtered_df[num_cols].sample(sample_size))
        st.pyplot(plt.gcf())
        st.subheader("Outlier Detection")
        # Example: IQR-based for numeric columns
        outlier_cols = []
        for col in num_cols:
            q1 = filtered_df[col].quantile(0.25)
            q3 = filtered_df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = filtered_df[(filtered_df[col] < q1 - 1.5 * iqr) | (filtered_df[col] > q3 + 1.5 * iqr)]
            if not outliers.empty:
                outlier_cols.append(col)
        if outlier_cols:
            st.write("Columns with outliers detected:", outlier_cols)
        else:
            st.write("No strong outliers detected.")
    else:
        st.info("Not enough numeric columns for correlation/outlier analysis.")

with tabs[4]:
    st.markdown("### Purchase by Acquisition Channel")
    if 'Acquisition_Channel' in filtered_df.columns and 'Purchase_Last_3mo_numeric' in filtered_df.columns:
        channel_purchase = filtered_df.groupby("Acquisition_Channel")["Purchase_Last_3mo_numeric"].mean().reset_index()
        fig4 = px.bar(channel_purchase, x="Acquisition_Channel", y="Purchase_Last_3mo_numeric", labels={"Purchase_Last_3mo_numeric": "Purchase Rate"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No suitable columns for channel analysis.")

with tabs[5]:
    st.markdown("### Customer Segmentation (KMeans)")
    if len(num_cols) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_data = filtered_df[num_cols].fillna(0)
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

with tabs[6]:
    st.header("🤖 ML Predictions (Lasso, Ridge, Decision Tree)")
    if len(num_cols) >= 2:
        reg_candidates = [col for col in num_cols if filtered_df[col].nunique() > 5]
        if reg_candidates:
            reg_target = st.selectbox(
                "Select target variable for regression",
                options=reg_candidates,
                help="Target variable should be continuous/numeric."
            )
            reg_features = [col for col in num_cols if col != reg_target]
            model_choice = st.radio("Choose regression model", ["Lasso", "Ridge", "Decision Tree"])
            if reg_target and reg_features:
                X = filtered_df[reg_features].fillna(0)
                y = filtered_df[reg_target].fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                if model_choice == "Lasso":
                    model = Lasso()
                elif model_choice == "Ridge":
                    model = Ridge()
                else:
                    model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                st.markdown(f"**R² Score:** {r2:.3f}")
                st.markdown(f"**RMSE:** {rmse:.3f}")
                fig, ax = plt.subplots()
                ax.scatter(y_test, preds, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{model_choice} Regression: Actual vs. Predicted ({reg_target})")
                st.pyplot(fig)
                # Show feature importances if Decision Tree
                if model_choice == "Decision Tree":
                    st.subheader("Feature Importances")
                    importances = pd.Series(model.feature_importances_, index=reg_features)
                    st.bar_chart(importances.sort_values(ascending=False))
            else:
                st.info("Not enough numeric features for regression modeling.")
        else:
            st.info("No suitable numeric column with enough unique values for regression modeling.")
    else:
        st.info("Not enough numeric columns for regression modeling.")

with tabs[7]:
    st.markdown("## 📊 Marketing Campaign Data")
    if uploaded_excel is not None:
        try:
            campaign_df = pd.read_excel(uploaded_excel, sheet_name="marketing_campaign_dataset")
            st.dataframe(campaign_df.head())
            if "Date" in campaign_df.columns:
                campaign_df["Date"] = pd.to_datetime(campaign_df["Date"])
                st.line_chart(campaign_df.set_index("Date").select_dtypes(include=np.number))
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
    else:
        st.info("Upload an Excel file to see campaign data.")

with tabs[8]:
    st.markdown("## Raw Data Table")
    st.dataframe(filtered_df)

st.markdown("---")
st.info("You can enhance this dashboard further with more filters, charts, and ML predictions as needed. 🚀")
