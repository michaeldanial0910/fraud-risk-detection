"""Streamlit app: card-transaction fraud risk, as a fraud team would use it.

    streamlit run app.py        ->  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Fraud risk review", page_icon="🛡️", layout="wide")

from app_lib.ui import setup  # noqa: E402  (set_page_config must run first)

setup()
pages = [
    st.Page("app_pages/queue.py", title="Review queue", url_path="queue", default=True),
    st.Page("app_pages/thresholds.py", title="Thresholds", url_path="thresholds"),
    st.Page("app_pages/results.py", title="Results", url_path="results"),
    st.Page("app_pages/mistakes.py", title="Mistakes", url_path="mistakes"),
    st.Page("app_pages/data_story.py", title="The data", url_path="data"),
]
st.navigation(pages, position="top").run()
