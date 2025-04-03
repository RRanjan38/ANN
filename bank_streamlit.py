import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        h1, h2, h3 { color: #2E86C1; text-align: center; }
        .metric-box { border-radius: 10px; padding: 10px; background-color: #ffffff; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Load Data
file_path = "bank_marketing_data_new.csv"  # Replace with actual filename in GitHub repo
data = pd.read_csv(file_path)

# Encode categorical features (fixes ValueError issue)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category').cat.codes  

# Sidebar: Hyperparameter Filters
st.sidebar.header("🔧 Hyperparameter Filters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
epochs = st.sidebar.slider("Epochs", 10, 100, 50, step=10)
batch_size = st.sidebar.slider("Batch Size", 16, 256, 32, step=16)

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Performance", "📉 Visualizations"])

# Tab 1: Data Overview
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(data.head(10))
    st.write(f"**Rows:** {data.shape[0]}, **Columns:** {data.shape[1]}")
    
    # Correlation Heatmap
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Tab 2: Model Performance Metrics
with tab2:
    st.subheader("📊 Model Performance Metrics")
    accuracy = round(np.random.uniform(0.85, 0.95), 4)  # Replace with actual accuracy
    loss = round(np.random.uniform(0.05, 0.15), 4)  # Replace with actual loss
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
    with col2:
        st.metric(label="Model Loss", value=f"{loss:.4f}")
    
    st.write("---")
    st.subheader("Hyperparameter Settings")
    st.write(f"- **Learning Rate:** {learning_rate}")
    st.write(f"- **Epochs:** {epochs}")
    st.write(f"- **Batch Size:** {batch_size}")

# Tab 3: Visualizations
with tab3:
    st.subheader("📈 Training Loss & Accuracy Over Time")
    
    # Simulated data (Replace with actual training logs)
    epochs_list = list(range(1, epochs+1))
    loss_values = np.exp(-0.1 * np.array(epochs_list)) + np.random.uniform(0, 0.02, size=len(epochs_list))
    accuracy_values = 0.5 + (1 - np.exp(-0.1 * np.array(epochs_list))) + np.random.uniform(0, 0.02, size=len(epochs_list))
    
    fig = px.line(x=epochs_list, y=loss_values, labels={'x': 'Epochs', 'y': 'Loss'}, title="Loss Curve", line_shape='spline')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.line(x=epochs_list, y=accuracy_values, labels={'x': 'Epochs', 'y': 'Accuracy'}, title="Accuracy Curve", line_shape='spline')
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.text("🚀 Developed by Rajiv Ranjan")
