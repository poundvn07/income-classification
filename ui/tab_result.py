import streamlit as st
from modules.texts import TEXT

def render_result_tab(df_raw):
    t = TEXT

    if "prediction" not in st.session_state:
        st.info(t["not_ready"])
        return

    st.header(t["result_title"])

    if st.session_state.prediction == 1:
        st.success(t["high"])
    else:
        st.error(t["low"])

    st.subheader("📘 Dataset Preview")
    st.dataframe(df_raw, use_container_width=True)
