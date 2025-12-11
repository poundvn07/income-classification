import streamlit as st
import pandas as pd
from modules.model import evaluate
from modules.charts import plot_confusion_matrix, plot_roc_curve
from modules.texts import TEXT

def render_analysis_tab(log_model, knn_model, X_test, y_test, df):
    t = TEXT

    if "prediction" not in st.session_state:
        st.info(t["not_ready"])
        return

    st.markdown("### 📊 Model Performance Comparison")

    log_metrics = evaluate(log_model, X_test, y_test)
    knn_metrics = evaluate(knn_model, X_test, y_test)

    metrics = pd.DataFrame([log_metrics, knn_metrics], index=["Logistic Regression", "KNN"])
    st.dataframe(metrics.style.format("{:.4f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix (Logistic)")
        st.plotly_chart(plot_confusion_matrix(log_model, X_test, y_test))
    with col2:
        st.subheader("Confusion Matrix (KNN)")
        st.plotly_chart(plot_confusion_matrix(knn_model, X_test, y_test))

    st.subheader("ROC Curve")
    st.plotly_chart(plot_roc_curve(log_model, knn_model, X_test, y_test))

