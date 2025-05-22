import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def multiple_linear_regression(df):
    st.title("Multiple Linear Regression ")

    data =df

    st.write("Preview of your dataset:", data.head())

    # Select target variable
    target = st.selectbox("Select the target variable", data.columns)

    # Select features
    features = st.multiselect("Select features (include categorical columns)",
                              data.columns.drop(target))

    # Select categorical features
    categorical_features = st.multiselect("Select categorical features",
                                          [col for col in features if data[col].dtype == 'object'])

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        test_size = st.slider("Select test size ratio", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Preprocessing
        numeric_features = [col for col in features if col not in categorical_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ])

        # Create pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        st.subheader("Model Evaluation")

        # Training set performance
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test set performance
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write(f"Training R² Score: {train_r2:.3f}")
        st.write(f"Test R² Score: {test_r2:.3f}")

        # Show coefficients
        # Modified coefficients section
        st.subheader("Model Coefficients")

        try:
            # Get numeric coefficients
            numeric_coefs = model.named_steps['regressor'].coef_[:len(numeric_features)]

            # Initialize categorical coefficients list
            categorical_coefs = []
            cat_feature_names = []

            # Check if categorical features exist and encoder was used
            if categorical_features:
                encoder = model.named_steps['preprocessor'].named_transformers_['cat']
                cat_feature_names = encoder.get_feature_names_out(categorical_features)
                categorical_coefs = model.named_steps['regressor'].coef_[len(numeric_features):]

            # Create coefficients DataFrame
            coefficients = pd.DataFrame({
                'Feature': numeric_features + list(cat_feature_names),
                'Coefficient': list(numeric_coefs) + list(categorical_coefs)
            })

            st.write(coefficients)

        except Exception as e:
            st.error(f"Error displaying coefficients: {str(e)}")
            st.write("Please ensure you've selected appropriate features and categorical columns")
        '''
        st.subheader("Model Coefficients")
        numeric_coefs = model.named_steps['regressor'].coef_[:len(numeric_features)]
        categorical_coefs = model.named_steps['regressor'].coef_[len(numeric_features):]

        coefficients = pd.DataFrame({
            'Feature': numeric_features + list(model.named_steps['preprocessor']
                                               .named_transformers_['cat']
                                               .get_feature_names_out(categorical_features)),
            'Coefficient': list(numeric_coefs) + list(categorical_coefs)
        })
        st.write(coefficients)
        '''

        # Visualization
        st.subheader("Visualizations")

        # Actual vs Predicted plot
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)

        # Residual plot
        fig2, ax2 = plt.subplots()
        sns.residplot(x=y_test_pred, y=y_test - y_test_pred, lowess=True,
                      line_kws={'color': 'red', 'lw': 1}, ax=ax2)
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)

        '''
        # Save model
        if st.button("Download Model"):
            joblib.dump(model, 'regression_model.pkl')
            with open('regression_model.pkl', 'rb') as f:
                st.download_button(
                    label="Download trained model",
                    data=f,
                    file_name="regression_model.pkl",
                    mime="application/octet-stream"
                )
        '''

