import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def load_and_inspect_data(file_path):
    """Load dataset and perform initial inspection"""
    df = pd.read_csv(file_path)

    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    #display(df.head())
    print("\nLast 5 Rows:")
    #display(df.tail())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    #display(df.describe(include='all').T)

    return df


def clean_column_names(df):
    """Fix column name spelling errors"""
    print("\nCurrent Columns:", df.columns.tolist())
    rename_map = {}
    for col in df.columns:
        new_name = input(f"Rename '{col}'? (Press enter to keep or type new name): ").strip()
        if new_name:
            rename_map[col] = new_name
    return df.rename(columns=rename_map)


def handle_data_types(df):
    """Automatically detect and convert data types"""
    for col in df.columns:
        # Attempt numeric conversion for object columns
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric")
            except:
                pass
    return df


def handle_categorical_values(df):
    """Check and clean categorical columns"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        print(f"\nColumn: {col}")
        print("Unique Values:", df[col].unique())

        # Show value counts and get user input for corrections
        print("\nValue Counts:")
        print(df[col].value_counts(dropna=False))

        if input(f"Clean values in {col}? (y/n): ").lower() == 'y':
            replacements = {}
            for val in df[col].unique():
                new_val = input(f"Replace '{val}' with: (press enter to keep) ").strip()
                if new_val:
                    replacements[val] = new_val or val
            df[col] = df[col].replace(replacements)

    return df


def check_negative_values(df):
    """Identify and handle negative values in numeric columns"""
    num_cols = df.select_dtypes(include=np.number).columns
    negatives = df[num_cols].lt(0).any()

    print("\nNegative Value Check:")
    for col in negatives[negatives].index:
        print(f"Column '{col}' contains negative values")
        print("Values:", df[col][df[col] < 0].unique())

        action = input(f"Handle negatives in {col}? (abs/zero/drop/skip): ").lower()
        if action == 'abs':
            df[col] = df[col].abs()
        elif action == 'zero':
            df[col] = df[col].clip(lower=0)
        elif action == 'drop':
            df = df[df[col] >= 0]

    return df


def main():
    file_path = input("Enter path to your CSV file: ")
    df = load_and_inspect_data(file_path)

    # Data Cleaning Pipeline
    df = clean_column_names(df)
    df = handle_data_types(df)
    df = handle_categorical_values(df)
    df = check_negative_values(df)

    # Final Report
    print("\nCleaning Complete! Final Dataset Summary:")
    print("Missing Values:", df.isnull().sum().sum())
    print("Data Types:\n", df.dtypes)
    print("\nSample Data:")


    # Save cleaned data
    if input("Save cleaned data? (y/n): ").lower() == 'y':
        df.to_csv('cleaned_data.csv', index=False)
        print("Data saved to cleaned_data.csv")


if __name__ == "__main__":
    main()
