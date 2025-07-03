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

@st.cache_data
def load_survey():
    df = pd.read_csv("synthetic_consumer_marketing_survey.csv")
    df = df.dropna(subset=["Purchase_Last_3mo"])
    return df

survey_df = load_survey()

st.sidebar.title("🔧 Uploads & Filters")

uploaded_excel = st.sidebar.file_uploader(
    "Upload campaign Excel file (xlsx)",
    type=["xlsx"],
    help="Must contain sheet 'marketing_campaign_dataset' with a Date column."
)

with st.sidebar.expander("Consumer Survey Filters", expanded=True):
    age_sel    = st.multiselect("Age Group", survey_df["Age_Group"].unique(), default=survey_df["Age_Group"].unique())
    gender_sel = st.multiselect("Gender", survey_df["Gender"].unique(), default=survey_df["Gender"].unique())
    region_sel = st.multiselect("Region", survey_df["Region"].unique(), default=survey_df["Region"].unique())
    edu_sel    = st.multiselect("Education", survey_df["Education"].unique(), default=survey_df["Education"].unique())
    rev_input  = st.number_input("Avg Revenue per Purchase", value=100.0, step=10.0)

if uploaded_excel:
    camp_df_raw = pd.read_excel(
        uploaded_excel,
        sheet_name="marketing_campaign_dataset",
        engine="openpyxl",
        parse_dates=["Date"]
    )
    with st.sidebar.expander("Campaign Data Filters", expanded=False):
        chan_sel  = st.multiselect("Channel", camp_df_raw["Channel_Used"].unique(), default=camp_df_raw["Channel_Used"].unique())
        ctype_sel = st.multiselect("Campaign Type", camp_df_raw["Campaign_Type"].unique(), default=camp_df_raw["Campaign_Type"].unique())
        dmin = camp_df_raw["Date"].min().date()
        dmax = camp_df_raw["Date"].max().date()
        date_range = st.date_input("Date Range", [dmin, dmax], min_value=dmin, max_value=dmax)

survey_mask = (
    survey_df["Age_Group"].isin(age_sel) &
    survey_df["Gender"].isin(gender_sel) &
    survey_df["Region"].isin(region_sel) &
    survey_df["Education"].isin(edu_sel)
)
survey_f = survey_df[survey_mask].copy()

if uploaded_excel:
    camp_mask = (
        camp_df_raw["Channel_Used"].isin(chan_sel) &
        camp_df_raw["Campaign_Type"].isin(ctype_sel) &
        (camp_df_raw["Date"].dt.date >= date_range[0]) &
        (camp_df_raw["Date"].dt.date <= date_range[1])
    )
    camp_f = camp_df_raw[camp_mask].copy()
else:
    camp_f = None

tabs = st.tabs([
    "🏠 Overview",
    "🧮 Descriptive",
    "📊 Segmentation",
    "🤖 Models",
    "🚀 Campaign Analysis",
    "✨ Advanced Insights"
])

with tabs[0]:
    st.title("Marketing Survey & Campaign Dashboard")
    st.markdown("""
    **Survey:** individual-level conversion & ROI proxy  
    **Campaign:** upload your Excel to unlock channel & campaign KPIs  
    Use the sidebar to filter both datasets.
    """)
    c1, c2, c3 = st.columns(3)
    conv_rate = (survey_f["Purchase_Last_3mo"] == "Yes").mean()
    avg_spend = survey_f["Monthly_Online_Spend"].mean()
    roi_proxy = conv_rate * rev_input - avg_spend
    c1.metric("Conversion Rate", f"{conv_rate:.1%}")
    c2.metric("Avg Monthly Spend", f"${avg_spend:,.2f}")
    c3.metric("ROI Proxy", f"${roi_proxy:,.2f}")

with tabs[1]:
    st.header("🔍 Descriptive Analytics")
    st.subheader("Survey: Summary Statistics")
    desc = survey_f.select_dtypes(include="number").describe().T
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["LB"]  = desc["25%"] - 1.5 * desc["IQR"]
    desc["UB"]  = desc["75%"] + 1.5 * desc["IQR"]
    st.dataframe(desc.style.format({
        "mean": "{:.2f}",
        "std": "{:.2f}",
        "IQR": "{:.2f}",
        "LB": "{:.2f}",
        "UB": "{:.2f}"
    }), use_container_width=True)
    if camp_f is not None:
        st.subheader("Campaign: Summary Statistics")
        num_cols = [
            "Conversion_Rate", "Acquisition_Cost", "ROI", "Cost/Click",
            "Impressions", "Click-Through Rate", "Engagement_Score"
        ]
        desc2 = camp_f[num_cols].describe().T
        desc2["IQR"] = desc2["75%"] - desc2["25%"]
        desc2["LB"]  = desc2["25%"] - 1.5 * desc2["IQR"]
        desc2["UB"]  = desc2["75%"] + 1.5 * desc2["IQR"]
        st.dataframe(desc2.style.format({
            "mean": "{:.2f}",
            "std": "{:.2f}",
            "IQR": "{:.2f}",
            "LB": "{:.2f}",
            "UB": "{:.2f}"
        }), use_container_width=True)

with tabs[2]:
    st.header("🔖 Segmentation Performance")
    seg_dim = st.selectbox("Segment by", ["Age_Group", "Gender", "Region", "Education"])
    conv_seg = survey_f.groupby(seg_dim)["Purchase_Last_3mo"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Conversion Rate")
    fig = px.bar(conv_seg, x=seg_dim, y="Conversion Rate", text=conv_seg["Conversion Rate"].map("{:.1%}".format), labels={"Conversion Rate": "Conv. Rate"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("ROI Proxy by Segment")
    roi_seg = survey_f.groupby(seg_dim).apply(lambda x: ((x["Purchase_Last_3mo"] == "Yes").mean() * rev_input - x["Monthly_Online_Spend"].mean())).reset_index(name="ROI")
    fig2 = px.bar(roi_seg, x=seg_dim, y="ROI", text=roi_seg["ROI"].map("${:,.2f}".format))
    st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.header("🎯 Predictive Models")
    df_clf = survey_f.copy()
    df_clf["Purchase_Last_3mo"] = df_clf["Purchase_Last_3mo"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    for col in df_clf.select_dtypes(include="object"):
        df_clf[col] = le.fit_transform(df_clf[col].astype(str))
    X = df_clf.drop(columns=["Purchase_Last_3mo"])
    y = df_clf["Purchase_Last_3mo"]
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notnull().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    logr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    ypred = logr.predict(Xte)
    clf_metrics = {
        "Accuracy":  accuracy_score(yte, ypred),
        "Precision": precision_score(yte, ypred),
        "Recall":    recall_score(yte, ypred),
        "F1":        f1_score(yte, ypred),
        "ROC AUC":   roc_auc_score(yte, logr.predict_proba(Xte)[:, 1])
    }
    st.subheader("Logistic Regression")
    st.json(clf_metrics)

    # --- Linear Regression (Spend) ---
    st.subheader("Linear Regression (Spend)")
    Xl = pd.get_dummies(
        survey_f.drop(columns=["Monthly_Online_Spend", "Purchase_Last_3mo"]),
        drop_first=True
    )
    yl = survey_f["Monthly_Online_Spend"]
    df_reg = pd.concat([Xl, yl.rename("Monthly_Online_Spend")], axis=1)
    df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna()
    Xl_clean = df_reg.drop(columns=["Monthly_Online_Spend"])
    yl_clean = df_reg["Monthly_Online_Spend"]
    Xltr, Xlte, yltr, ylte = train_test_split(Xl_clean, yl_clean, test_size=0.3, random_state=42)
    lr = LinearRegression().fit(Xltr, yltr)
    ylpred = lr.predict(Xlte)
    mse = mean_squared_error(ylte, ylpred)
    rmse = np.sqrt(mse)
    st.metric("RMSE", f"{rmse:.2f}")

    # --- KMeans Clustering (SAFE!) ---
    st.subheader("K-Means Clustering")
    num_features = X.select_dtypes(include="number")
    scaler = StandardScaler().fit(num_features)
    Xs = scaler.transform(num_features)
    k = st.slider("Number of Clusters", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(Xs)
    pca = PCA(n_components=2).fit_transform(Xs)
    fig3 = px.scatter(x=pca[:, 0], y=pca[:, 1], color=km.labels_.astype(str), labels={"x": "PC1", "y": "PC2", "color": "Cluster"})
    st.plotly_chart(fig3, use_container_width=True)

    # Assign cluster labels ONLY to the rows used in clustering!
    clust_df = survey_f.loc[X.index].copy()
    clust_df["Cluster"] = km.labels_

    prof = clust_df.groupby("Cluster").agg({
        "Purchase_Last_3mo": lambda x: (x == "Yes").mean(),
        "Monthly_Online_Spend": "mean"
    }).rename(columns={
        "Purchase_Last_3mo": "Conv_Rate",
        "Monthly_Online_Spend": "Avg_Spend"
    })
    st.write(prof)

with tabs[4]:
    st.header("🚀 Campaign Analysis")
    if camp_f is None:
        st.warning("Upload your Excel on the left to unlock Campaign Analysis.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Campaigns", len(camp_f))
        c2.metric("Avg Conv. Rate", f"{camp_f['Conversion_Rate'].mean():.1%}")
        c3.metric("Avg ROI",    f"{camp_f['ROI'].mean():.2f}")
        c4.metric("Total Impr.", f"{int(camp_f['Impressions'].sum()):,}")

        st.subheader("Conv. Rate by Channel")
        br1 = camp_f.groupby("Channel_Used")["Conversion_Rate"].mean().reset_index()
        fig4 = px.bar(
            br1,
            x="Channel_Used",
            y="Conversion_Rate",
            text=br1["Conversion_Rate"].map("{:.1%}".format)
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("ROI by Campaign Type")
        fig5 = px.box(
            camp_f,
            x="Campaign_Type",
            y="ROI",
            points="all"
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Acquisition Cost Distribution")
        fig6 = px.histogram(
            camp_f,
            x="Acquisition_Cost",
            nbins=25,
            marginal="box"
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("ROI vs Click-Through Rate")
        fig7 = px.scatter(
            camp_f,
            x="Click-Through Rate",
            y="ROI",
            size="Impressions",
            color="Channel_Used",
            hover_data=["Campaign_ID", "Campaign_Type"]
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Customer Segment Share")
        fig8 = px.pie(
            camp_f,
            names="Customer_Segment",
            title="Segments"
        )
        st.plotly_chart(fig8, use_container_width=True)

        st.subheader("Conversion Rate Over Time")
        ts = camp_f.groupby("Date")["Conversion_Rate"].mean().reset_index()
        fig9 = px.line(ts, x="Date", y="Conversion_Rate")
        st.plotly_chart(fig9, use_container_width=True)

with tabs[5]:
    st.header("✨ Advanced Insights")

    st.subheader("Scatter Matrix (Campaign Metrics)")
    if camp_f is not None:
        sel_cols = [
            "Conversion_Rate",
            "Acquisition_Cost",
            "ROI",
            "Cost/Click",
            "Click-Through Rate"
        ]
        fig10 = px.scatter_matrix(camp_f[sel_cols])
        fig10.update_traces(diagonal_visible=False)
        st.plotly_chart(fig10, use_container_width=True)

    st.subheader("Survey Feature Correlations")
    num_corr = survey_f.select_dtypes(include="number").corr()
    if num_corr.isnull().all().all() or num_corr.shape[0] == 0:
        st.warning("No numeric data available for correlation heatmap. Try adjusting filters.")
    else:
        fig11, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(num_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig11)

    if camp_f is not None:
        st.subheader("Campaign Metric Correlations")
        camp_corr = camp_f[sel_cols].corr()
        if camp_corr.isnull().all().all() or camp_corr.shape[0] == 0:
            st.warning("No numeric campaign data available for correlation heatmap. Try adjusting campaign filters.")
        else:
            corr2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(camp_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            st.pyplot(corr2)
