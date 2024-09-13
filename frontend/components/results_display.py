import streamlit as st
import pandas as pd

def render_results(results):
    if not results:
        st.write("No results to display.")
        return

    df = pd.DataFrame(results)
    st.write(df)
    
    st.download_button(
        label="Download Results",
        data=df.to_csv(index=False),
        file_name="verification_results.csv",
        mime="text/csv"
    )

    # Display detailed reports
    for result in results:
        with st.expander(f"Detailed Report for Roll No: {result['roll_no']}"):
            st.text(result['report'])