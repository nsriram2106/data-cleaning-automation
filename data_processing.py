import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler



def load_data(file_path):
    """Loads CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    # Let user select target column
    target_column = st.selectbox("Select target column", df.columns)  # GOOD

    # Initialize cleaned_df
    cleaned_df = clean_column_names(df)  # GOOD

    st.write("###  Data Types Preview")
    st.write(df.head())



    # Modified data type handling section with symbol cleaning
    with st.expander("Data Type Handling"):
        st.write("### Current Data Types")
        st.write(cleaned_df.dtypes)

        type_changes = {}
        for col in cleaned_df.columns:
            current_type = cleaned_df[col].dtype
            new_type = st.selectbox(
                f"{col} ({current_type})",
                ["keep", "numeric", "category", "datetime"],
                key=f"type_{col}"
            )
            if new_type != "keep":
                type_changes[col] = new_type

        # Apply type changes with symbol cleaning for numeric columns
        for col, new_type in type_changes.items():
            if new_type == "numeric":
                # Clean symbols before conversion
                cleaned_df[col] = (
                    cleaned_df[col]
                    .astype(str)
                    .str.replace(r'[^0-9.-]', '', regex=True)  # Remove non-numeric characters
                    .replace(r'^\.$', np.nan, regex=True)  # Handle standalone decimal points
                    .replace(r'^$', np.nan, regex=True)  # Handle empty strings
                )
                # Convert to numeric
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

                # Show cleaning results
                st.write(f"Cleaned column '{col}':")
                st.write(f"Original values: {df[col].unique()[:10]}")
                st.write(f"Cleaned values: {cleaned_df[col].unique()[:10]}")

            elif new_type == "category":
                cleaned_df[col] = cleaned_df[col].astype('category')
            elif new_type == "datetime":
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')


    st.write("### verifying updated data types")
    st.write(cleaned_df.dtypes)
    st.write(df.head())

    # New Negative Value Handling section
    with st.expander("Negative Value Handling"):
        numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        st.write("### Negative Value Treatment")

        for col in numeric_cols:
            if (cleaned_df[col] < 0).any():  # Check if column has negatives
                neg_count = (cleaned_df[col] < 0).sum()
                st.markdown(f"**{col}** has {neg_count} negative values")

                # Show sample values before
                sample_before = cleaned_df[col].head(5).to_list()

                # User choice
                if st.checkbox(f"Convert negatives to positives in {col}?", key=f"neg_{col}"):
                    # Convert to absolute values
                    cleaned_df[col] = cleaned_df[col].abs()

                    # Show conversion results
                    sample_after = cleaned_df[col].head(5).to_list()

                    st.write("Sample conversion:")
                    st.json({
                        "Before": sample_before,
                        "After": sample_after
                    })
                    st.success(f"Converted {neg_count} negative values in {col}")
            else:
                st.write(f"**{col}** has no negative values")

    # Handle missing values
    '''
    with st.expander("Missing Values Handling"):
        st.write("### Missing Values Summary")
        missing = cleaned_df.isnull().sum()
        st.write(missing[missing > 0])

        # Create two-column layout
        col_left, col_right, col_btn = st.columns([3, 3, 2])

        # Left dropdown - Select column with missing values
        missing_cols = [col for col in cleaned_df.columns if cleaned_df[col].isnull().any()]
        selected_col = col_left.selectbox(
            "Select Column with Missing Values",
            options=missing_cols,
            index=0,
            label_visibility="collapsed"
        )

        # Right dropdown - Select handling strategy
        strategy = col_right.selectbox(
            "Handling Strategy",
            options=["drop", "fill"],
            index=1,
            label_visibility="collapsed"
        )

        # Apply button
        if col_btn.button("Apply", key="missing_apply"):
            if strategy == "fill":
                if pd.api.types.is_numeric_dtype(cleaned_df[selected_col]):
                    fill_value = cleaned_df[selected_col].mean()
                else:
                    fill_value = cleaned_df[selected_col].mode()[0]
                cleaned_df[selected_col].fillna(fill_value, inplace=True)
                st.success(f"Filled missing values in {selected_col} with {fill_value}")
            else:
                cleaned_df.dropna(subset=[selected_col], inplace=True)
                st.success(f"Dropped rows with missing values in {selected_col}")

            # Refresh the page to update the missing values summary
            st.rerun()

        # Show current data type
        st.caption(f"Selected column type: {cleaned_df[selected_col].dtype}")

    missing = cleaned_df.isnull().sum()
    st.write(missing[missing > 0])
    '''
    with st.expander("Missing Values Handling"):
        st.write("### Missing Values Summary")
        missing = cleaned_df.isnull().sum()
        st.write(missing[missing > 0])

        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                strategy = st.selectbox(
                    f"Handling for {col} ({cleaned_df[col].dtype})",
                    ["drop", "fill"],
                    key=f"missing_{col}"
                )
                if strategy == "fill":
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        fill_value = cleaned_df[col].mean()
                    else:
                        fill_value = cleaned_df[col].mode()[0]
                    cleaned_df[col].fillna(fill_value, inplace=True)
                else:
                    cleaned_df.dropna(subset=[col], inplace=True)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    fig01 = plt.show()
    st.pyplot(fig01)

    # Outlier handling
    with st.expander("Outlier Handling"):
        numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
        selected_cols = st.multiselect("Select columns for outlier removal", numeric_cols)

        for col in selected_cols:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            st.write(f"Removed outliers from {col}")


    # Streamlit App
    st.title("Target Variable Engineering")

    with st.expander("Categorical Binning"):
        # Let the user choose a numerical column
        num_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()

        if num_cols:
            selected_col = st.selectbox("Select a Numerical Column", num_cols)

            min_val = cleaned_df[selected_col].min()
            max_val = cleaned_df[selected_col].max()

            # User-defined bin thresholds
            low_med = st.slider(f"Low/Medium Threshold for {selected_col}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=min_val + (max_val - min_val) / 3)
            med_high = st.slider(f"Medium/High Threshold for {selected_col}",
                                 min_value=float(low_med),
                                 max_value=float(max_val),
                                 value=min_val + 2 * (max_val - min_val) / 3)

            # Create bins and labels dynamically
            bins = [min_val - 1, low_med, med_high, max_val + 1]
            labels = ['Low', 'Medium', 'High']

            # Create categorical column
            cat_col_name = f"{selected_col}_Category"
            cleaned_df[cat_col_name] = pd.cut(
                cleaned_df[selected_col],
                bins=bins,
                labels=labels,
                include_lowest=True
            ).astype('category')

            # Set category ordering
            cleaned_df[cat_col_name] = cleaned_df[cat_col_name].cat.reorder_categories(
                ['Low', 'Medium', 'High'], ordered=True)

            # Show distribution
            st.write(f"### {cat_col_name} Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=cat_col_name, data=cleaned_df, order=labels)
            st.pyplot(fig)

            # Show value counts
            st.write("Value Counts:")
            st.write(cleaned_df[cat_col_name].value_counts().sort_index())

        else:
            st.warning("No numerical columns found in the dataset.")

    st.write("### Raw Data Preview")
    st.write(df.head())



    with st.expander("Categorical Value Cleaning"):
        cat_cols = cleaned_df.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            st.write(f"### Column: {col}")
            unique_values = cleaned_df[col].astype(str).unique()
            st.write("Current unique values:", unique_values)

            if st.checkbox(f"Clean values in {col}?", key=f"clean_{col}"):
                # Initialize session state for mappings
                if f'mappings_{col}' not in st.session_state:
                    st.session_state[f'mappings_{col}'] = []

                # Add new mapping
                if st.button("➕ Add Replacement Rule", key=f"add_rule_{col}"):
                    st.session_state[f'mappings_{col}'].append({
                        'from': [],
                        'to': ''
                    })

                # Display existing mappings
                for i, rule in enumerate(st.session_state[f'mappings_{col}']):
                    st.markdown("---")
                    cols = st.columns([3, 1])

                    # Target value input
                    new_value = cols[0].text_input(
                        f"Replace with value:",
                        value=rule['to'],
                        key=f"to_{col}_{i}"
                    )

                    # Values to replace
                    old_values = cols[1].multiselect(
                        "Select values to replace:",
                        options=unique_values,
                        default=rule['from'],
                        key=f"from_{col}_{i}"
                    )

                    # Update rule
                    st.session_state[f'mappings_{col}'][i] = {
                        'from': old_values,
                        'to': new_value.strip()
                    }

                    # Remove rule button
                    if st.button("❌ Remove Rule", key=f"remove_{col}_{i}"):
                        del st.session_state[f'mappings_{col}'][i]
                        st.rerun()

                # Apply replacements
                if st.session_state[f'mappings_{col}']:
                    df_copy = cleaned_df.copy()
                    df_copy[col] = df_copy[col].astype(str)

                    # Create replacement dictionary
                    replace_dict = {}
                    for rule in st.session_state[f'mappings_{col}']:
                        if rule['to'] and rule['from']:
                            for old_val in rule['from']:
                                replace_dict[old_val] = rule['to']

                    # Apply replacements
                    if replace_dict:
                        df_copy[col] = df_copy[col].replace(replace_dict)
                        cleaned_df[col] = df_copy[col]

                        st.success("Replacements applied!")
                        st.write("### Before vs After")
                        st.dataframe(pd.DataFrame({
                            "Original": df_copy[col].value_counts().index,
                            "New Value": df_copy[col].value_counts().values
                        }))

    # Handle class imbalance
    if pd.api.types.is_categorical_dtype(cleaned_df[target_column]):
        with st.expander("Class Balance Handling"):
            st.write("### Class Distribution")
            class_dist = cleaned_df[target_column].value_counts()
            st.write(class_dist)

            if st.checkbox("Balance classes using undersampling?"):
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(
                    cleaned_df.drop(target_column, axis=1),
                    cleaned_df[target_column]
                )
                cleaned_df = pd.concat([
                    pd.DataFrame(X_resampled),
                    pd.DataFrame({target_column: y_resampled})
                ], axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(cleaned_df[col], ax=axes[i], kde=True)  # kernel density estimate
        axes[i].set_title(col)

    plt.tight_layout()
    fig2 = plt.show()
    st.pyplot(fig2)

    # Correlation handling
    with st.expander("Correlation Analysis"):
        numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            corr_matrix = cleaned_df[numeric_cols].corr()
            st.write("### Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

            threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.8)
            high_corr = np.where((corr_matrix.abs() > threshold) & (corr_matrix.abs() < 1))

            cols_to_drop = set()
            for i, j in zip(*high_corr):
                if i != j and i < j:
                    cols_to_drop.add(corr_matrix.columns[j])

            if cols_to_drop:
                st.write("Columns to consider dropping:", cols_to_drop)
                if st.checkbox("Drop high-correlation columns?"):
                    cleaned_df = cleaned_df.drop(columns=cols_to_drop)

    return cleaned_df


def clean_column_names(df):
    """Fix column name spelling errors"""
    with st.form("rename_form"):
        col1, col2, col3 = st.columns([3, 1, 3])

        # Left column - Dropdown of existing columns
        selected_col = col1.selectbox(
            "Select column to rename",
            options=df.columns,
            index=0,
            label_visibility="collapsed",
            key="column_selector"
        )

        # Middle column - Arrow separator
        col2.markdown("<div style='text-align: center; margin-top: 15px;'>→</div>", unsafe_allow_html=True)

        # Right column - Text input for new name
        new_name = col3.text_input(
            "New column name",
            value=selected_col,
            label_visibility="collapsed",
            key="new_name_input",
            placeholder="Enter new name"
        )

        # Rename button
        if st.form_submit_button("Rename"):
            if new_name != selected_col:
                df = df.rename(columns={selected_col: new_name})
                st.success(f"Column renamed: {selected_col} → {new_name}")
            else:
                st.warning("No changes made. New name matches existing name.")

    return df



def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def split_data(df, target_column):
    """Splits data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)