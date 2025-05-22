import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def statistical_analysis(df, target_column, num_cols, cat_cols, alpha=0.05):
    st.subheader("Descriptive Statistics")
    st.write("### Numerical Data Summary")
    st.write(df[num_cols].describe())

    st.write("### Categorical Data Summary")
    st.write(df[cat_cols].describe(include='object'))

    # Normalization & Standardization
    st.subheader("Data Preprocessing")
    if num_cols:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df[num_cols])
        st.write("Normalized Data:", pd.DataFrame(normalized, columns=num_cols).head())

        scaler = StandardScaler()
        standardized = scaler.fit_transform(df[num_cols])
        st.write("Standardized Data:", pd.DataFrame(standardized, columns=num_cols).head())

    # Hypothesis Testing
    if target_column:
        st.subheader("Inferential Statistics")
        if df[target_column].nunique() == 2:
            st.write("### T-Test")
            for col in num_cols:
                group1 = df[df[target_column] == df[target_column].unique()[0]][col].dropna()
                group2 = df[df[target_column] == df[target_column].unique()[1]][col].dropna()
                t_stat, p_val = stats.ttest_ind(group1, group2)
                st.write(f"{col}: T-stat = {t_stat:.4f}, P-value = {p_val:.4f}")

        if df[target_column].nunique() > 2:
            st.write("### ANOVA Test")
            for col in num_cols:
                groups = [df[df[target_column] == g][col].dropna() for g in df[target_column].unique()]
                f_stat, p_val = stats.f_oneway(*groups)
                st.write(f"{col}: F-stat = {f_stat:.4f}, P-value = {p_val:.4f}")

        st.write("### Chi-Squared Test")
        for col in cat_cols:
            if col != target_column:
                contingency_table = pd.crosstab(df[target_column], df[col])
                chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                st.write(f"{col}: Chi2 = {chi2:.4f}, P-value = {p_val:.4f}")



def comprehensive_statistical_analysis(df):
    st.title("Comprehensive Statistical Analysis")
    df = df
    if df is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        target_column = st.selectbox("Select Target Column", [None] + df.columns.tolist())
        statistical_analysis(df, target_column, num_cols, cat_cols)


