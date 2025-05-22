import streamlit as st
import pandas as pd

def clean_column_names(df):
    """Fix column name spelling errors"""
    with st.form("rename_columns"):
        rename_map = {}
        for col in df.columns:
            new_name = st.text_input(f"Rename '{col}'? (Press enter to keep or type new name): ", value=col)
            if new_name != col:
                rename_map[col] = new_name
        submitted = st.form_submit_button("Rename Columns")
        if submitted:
            return df.rename(columns=rename_map)
    return df

# Upload DataFrame
st.title("Clean Column Names")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Current Columns:", df.columns.tolist())
    cleaned_df = clean_column_names(df)
    st.write(cleaned_df)