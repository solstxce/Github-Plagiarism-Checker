import streamlit as st

def render_file_uploader():
    return st.file_uploader("Upload CSV file", type="csv")