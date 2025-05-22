import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression App")



def linear_regression(df):
    try:
        df = df

        # Display DataFrame
        st.title("Simple Linear Regression")
        if st.checkbox("Show DataFrame"):
            st.dataframe(df)

        # Column Selection
        target_column = st.selectbox("Select Target Variable Column:", df.columns)
        feature_column = st.selectbox("Select Feature Column:", df.columns)

        if target_column and feature_column:  # Ensure both are selected
            X = df[[feature_column]]
            y = df[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Display Coefficients
            st.subheader("Regression Coefficients")
            st.write(f"Coefficient: {model.coef_[0]:.4f}")
            st.write(f"Intercept: {model.intercept_:.4f}")

            # Display Metrics
            st.subheader("Model Evaluation")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"R-squared: {r2_score(y_test, y_pred):.4f}")

            # Visualization
            st.subheader("Regression Line")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, y, color='blue', label='Actual Data')
            ax.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
            ax.set_title(f'Relationship between {feature_column} and {target_column}')
            ax.set_xlabel(feature_column)
            ax.set_ylabel(target_column)
            ax.legend()
            st.pyplot(fig)

            # Prediction Example
            st.subheader("Prediction")
            sample_value = st.number_input("Enter a sample value for prediction:", value=0.0) #Better input
            if st.button("Predict"): #Button to trigger prediction
                prediction = model.predict([[sample_value]])
                st.write(f"Predicted {target_column}: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")