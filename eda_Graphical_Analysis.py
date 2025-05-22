
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(df):
    st.title("Interactive Data Analysis Tool")
    st.markdown("## Exploratory Data Analysis (EDA)")
    st.markdown("### Graphical Data Analysis")

    if st.checkbox("Show Data Preview"):
        st.write(df.head())

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("#### Univariate Analysis")

    if numerical_cols:
        st.markdown("##### Numerical Variables")
        selected_num = st.selectbox("Select a numerical column:", numerical_cols, key="num_col_uni")  # Unique key
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num], ax=ax)
        plt.title(f"Histogram of {selected_num}")
        st.pyplot(fig)

    if categorical_cols:
        st.markdown("##### Categorical Variables")
        selected_cat = st.selectbox("Select a categorical column:", categorical_cols, key="cat_col_uni")  # Unique key
        fig, ax = plt.subplots()
        sns.countplot(x=selected_cat, data=df, ax=ax)
        plt.title(f"Count Plot of {selected_cat}")
        st.pyplot(fig)

    st.markdown("#### Bivariate Analysis")

    if len(numerical_cols) >= 2:
        st.markdown("##### Two Numerical Columns")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", numerical_cols)
        with col2:
            y_axis = st.selectbox("Y-axis:", numerical_cols)
        fig, ax = plt.subplots()
        plt.scatter(df[x_axis], df[y_axis])
        plt.title(f"Scatter plot of {x_axis} vs {y_axis}")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot(fig)

    if numerical_cols and categorical_cols:  # Numerical vs. Categorical
        st.markdown("##### Numerical vs. Categorical")
        num_col = st.selectbox("Select a numerical column:", numerical_cols, key="num_col_bi")  # Unique key
        cat_col = st.selectbox("Select a categorical column:", categorical_cols, key="cat_col_bi")  # Unique key

        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)  # Or violinplot, etc.
        plt.title(f"Box plot of {num_col} by {cat_col}")
        st.pyplot(fig)



    st.markdown("#### Multivariate Analysis")
    if len(numerical_cols) >= 2 and len(categorical_cols) >= 1:
        st.markdown("##### Numerical vs. Numerical vs. Categorical")
        num_col1 = st.selectbox("Select a numerical column for x-axis:", numerical_cols,
                                key="num_col1_multi")  # Unique key
        num_col2 = st.selectbox("Select a numerical column for y-axis:", numerical_cols,
                                key="num_col2_multi")  # Unique key
        cat_col_multi = st.selectbox("Select a categorical column for hue:", categorical_cols, key="cat_col_multi")

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better visualization
        sns.scatterplot(x=num_col1, y=num_col2, hue=cat_col_multi, data=df, ax=ax)
        plt.title(f"Scatter plot of {num_col1} vs {num_col2} by {cat_col_multi}")
        st.pyplot(fig)


        st.markdown("##### Categorical vs. Categorical vs. Numerical")
        cat_col1_multi = st.selectbox("Select a categorical column for x-axis:", categorical_cols,
                                      key="cat_col1_multi")  # Unique key
        cat_col2_multi = st.selectbox("Select a categorical column for y-axis:", categorical_cols,
                                      key="cat_col2_multi")  # Unique key
        num_col_multi = st.selectbox("Select a numerical column for hue:", numerical_cols,
                                     key="num_col_multi_2")  # Unique key

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better visualization
        sns.barplot(x=cat_col1_multi, y=num_col_multi, hue=cat_col2_multi, data=df, ax=ax)
        plt.title(f"Bar plot of {cat_col1_multi} vs {num_col_multi} by {cat_col2_multi}")
        st.pyplot(fig)


    st.markdown("#### Correlation Analysis")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numerical_cols) >= 2:  # Only show correlation if at least 2 numerical columns
        try:
            corr_matrix = df[numerical_cols].corr(method='pearson')  # Calculate correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size for better visualization
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Generate a mask for the upper triangle
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, mask=mask,
                        ax=ax)  # Plot heatmap
            plt.title("Correlation Matrix")
            st.pyplot(fig)

            # Optional: Display correlation values as a table
            if st.checkbox("Show Correlation Table"):
                st.write(corr_matrix)

        except Exception as e:
            st.error(f"Error calculating or displaying correlation: {e}")
    else:
        st.warning("Correlation analysis requires at least two numerical columns.")
