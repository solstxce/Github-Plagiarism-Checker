import streamlit as st
import pandas as pd
import requests
from components.file_uploader import render_file_uploader
from components.results_display import render_results

def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    results = []
    for _, row in df.iterrows():
        response = requests.post("http://localhost:5000/api/verify", 
                                 json={"roll_no": row["Roll.No."], "github_link": row["Github link"]})
        results.append(response.json())
    return results

def main():
    st.title("Project Verification System")
    
    uploaded_file = render_file_uploader()
    
    if uploaded_file is not None:
        st.write("Processing file...")
        results = process_file(uploaded_file)
        render_results(results)

if __name__ == "__main__":
    main()
