import streamlit as st


def init_session_state() -> None:
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = {}
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = "Lahore"
