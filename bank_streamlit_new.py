import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

# Page configuration
st.set_page_config(page_title="🔍 Bank Marketing Insights", layout="wide")

# Load dataset
file_path = "C:\\Users\\Rajiv Ranjan\\OneDrive\\Documents\\bank_marketing_data_new.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

# Sidebar filters
st.sidebar.header("🔧 Filter Options")
st.sidebar.markdown("Use these filters to refine your data view.")
selected_job = st.sidebar.selectbox("Select Job Type", ["All"] + sorted(df['job'].unique().astype(str)))
selected_education = st.sidebar.multiselect("Select Education Levels", df['education'].unique())

filtered_df = df.copy()
if selected_job != "All":
    filtered_df = filtered_df[filtered_df['job'] == int(selected_job)]
if selected_education:
    filtered_df = filtered_df[filtered_df['education'].isin(selected_education)]

# Header with branding
st.markdown("""
    <style>
        .main-header { text-align: center; font-size: 36px; color: #2E86C1; font-weight: bold; }
        .sub-header { text-align: center; font-size: 20px; color: #5D6D7E; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='main-header'>📊 Bank Marketing Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>An interactive platform to explore customer marketing insights</div>", unsafe_allow_html=True)
st.markdown("---")

# Metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{filtered_df.shape[0]}")
col2.metric("Total Features", f"{filtered_df.shape[1]}")
col3.metric("Subscription Rate", f"{round(filtered_df['y'].mean() * 100, 2)}%")

# Tabs layout
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Preview", "📈 Visual Analytics", "🔁 Correlation Map", "📊 Demographic Distributions"])

# Tab 1: Data Overview
with tab1:
    st.subheader("🔍 Preview of Filtered Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)

# Tab 2: Visual Analytics
with tab2:
    st.subheader("🎯 Subscription Status")
    fig1 = px.histogram(filtered_df, x='y', color='y', title="Target Subscription (Yes/No)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💼 Job Category vs Subscription")
    fig2 = px.histogram(filtered_df, x='job', color='y', barmode='group', title="Subscription by Job", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3: Correlation Matrix
with tab3:
    st.subheader("🔁 Feature Correlation Matrix")
    corr_matrix = filtered_df.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale='Viridis')
    st.plotly_chart(fig3, use_container_width=True)

# Tab 4: Demographic Plots
with tab4:
    st.subheader("👤 Age Distribution")
    fig4 = px.histogram(filtered_df, x='age', nbins=30, title="Customer Age Distribution", template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🏠 Housing vs Subscription")
    fig5 = px.histogram(filtered_df, x='housing', color='y', barmode='group', title="Housing Status vs Subscription", template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Rajiv Ranjan** 🚀")
