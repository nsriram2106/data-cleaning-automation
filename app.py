import streamlit as st
from data_processing import load_data, preprocess_data
import io
from eda_test import comprehensive_statistical_analysis
from eda_Graphical_Analysis import perform_eda
from simple_Linear_Regression import linear_regression
from Multiple_Linear_Regression import multiple_linear_regression

# Streamlit UI
st.title("Data Preprocessing and EDA")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = load_data(uploaded_file)
    st.write("### Raw Data Preview")
    st.write(df.head())

    # Preprocessing
    st.write("### processing Data")
    cleaned_df = preprocess_data(df)

    #Graphical analysis
    perform_eda(cleaned_df)

    # Run analysis (specify target column for hypothesis testing)
    comprehensive_statistical_analysis(cleaned_df)

    #Simple linear regression
    linear_regression(cleaned_df)

    #multiple linear regression
    multiple_linear_regression(cleaned_df)

    # 1. Convert DataFrame to CSV string:
    csv_buffer = io.StringIO()  # Use StringIO for in-memory file-like object
    cleaned_df.to_csv(csv_buffer, index=False)  # Write DataFrame to CSV string
    csv_data = csv_buffer.getvalue()  # Get the CSV string

    # or if you want bytes:
    csv_bytes = cleaned_df.to_csv(index=False).encode()


    st.download_button(
        label="Download cleaned data as CSV",
        data=csv_data,  # Pass the CSV string
        file_name="cleaned_data.csv",
        mime="text/csv",
    )


