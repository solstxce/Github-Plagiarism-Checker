import streamlit as st
import pandas as pd

def render_results(results):
    st.write(pd.DataFrame(results))
    
    st.download_button(
        label="Download Results",
        data=pd.DataFrame(results).to_csv(index=False),
        file_name="verification_results.csv",
        mime="text/csv"
    )
