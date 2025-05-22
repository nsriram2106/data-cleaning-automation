import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats


def load_data(file_path):
    """Loads CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)

def generate_summary(df):
    """Generates summary statistics for the dataset."""
    return df.describe()

def mot(df):#measure of tendency


    st.write("### mean of Age")
    st.write(df.Age.mean())

    st.write("### median of Age")
    st.write(df.Age.median())

    st.write("### mode of sex")
    st.write(df.Sex.mode()[0])

    st.write("### Measurement of spread")
    st.write("### Variance of age")
    st.write(df.Age.var())

    st.write("### IQR of Fare")
    st.write(df.Age.max() - df.Age.min())

    st.write("### range of age")
    st.write(df.Fare.quantile(0.75) - df.Fare.quantile(0.25))

    st.write("### describing data")
    st.write(df.describe())

    st.write("### skewness of fare")
    st.write(df.Fare.skew())

    sns.histplot(df['Fare'])
    plt.title("Histogram of Fare column")
    fig1=plt.show()
    st.pyplot(fig1)

    st.write("### kurtosis of Fare")
    st.write(df.Fare.kurt())
    st.write(df.dtypes)

    # Normalization
    st.write("### Normalization")
    scaler = MinMaxScaler()

    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')  # Converts non-numeric values to NaN
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Replace NaN with median

    num_df = ['Fare']
    scaler.fit(num_df)
    scaler.fit_transform(num_df)

    scaler = StandardScaler()
    scaler.fit_transform(num_df)

    st.write("### Inferential statistics")
    st.write("### t_test")
    st.write(df.dtypes)

    t_statistic, p_value = stats.ttest_ind(df[df['Survived'] == 1]['Fare'].dropna(),
                                           df[df['Survived'] == 0]['Fare'].dropna())
    st.write("### T-Test:")
    st.write("T-Statistic:", t_statistic)
    st.write("P-Value:", p_value)
    st.write(df.dtypes)


    return df