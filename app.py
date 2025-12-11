import streamlit as st

from modules.data_loader import load_raw_data
from modules.preprocess import preprocess
from modules.model import train_models
from ui.tab_input import render_input_tab
from ui.tab_result import render_result_tab
from ui.tab_analysis import render_analysis_tab

st.set_page_config(page_title="Income Predictor", page_icon="💵", layout="wide")

# Load CSS
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
from modules.texts import TEXT

# Load & preprocess dataset
df_raw = load_raw_data()
df_processed, X_train, X_test, y_train, y_test, scaler, num_cols = preprocess(df_raw)

# Train models
log_model, knn_model = train_models(X_train, y_train)

# Tabs
tabs = st.tabs(["📥 Input", "📤 AI Result", "📊 Data & Model Analysis"])

with tabs[0]:
    render_input_tab(df_raw, df_processed, scaler, num_cols, log_model, knn_model)

with tabs[1]:
    render_result_tab(df_raw)

with tabs[2]:
    render_analysis_tab(log_model, knn_model, X_test, y_test, df_processed)
