import streamlit as st
from components.file_uploader import render_file_uploader
from components.results_display import render_results

def main():
    st.title("Project Verification System")
    
    uploaded_file = render_file_uploader()
    
    if uploaded_file is not None:
        results = process_file(uploaded_file)
        render_results(results)

if __name__ == "__main__":
    main()
