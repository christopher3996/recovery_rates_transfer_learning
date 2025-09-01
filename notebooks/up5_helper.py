import warnings
warnings.filterwarnings("ignore")

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def timestamp():
    """Return a string of the current date and time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def add_src_to_path():
    """Add the 'src' directory located one level up into sys.path."""
    project_root = os.path.abspath("..")
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

# It's safe to add your src path immediately if you need it throughout your project.
add_src_to_path()

def get_36_columns():
    """
    Return the list of the 36 specific columns to be used if only_36 is True.
    """
    return [
        "currency_",
        "domicile_country_",
        "exchange_country_",
        "Industry_sector_",
        "event_type_",
        "consumer_opinion_confidence_indicator",
        "consumer_sentiment_index",
        "1_month_financial_uncertainty",
        "3_month_financial_certainty",
        "12_month_financial_certainty",
        "US Corporate Bond Yield Spread",
        "US Corporate Bond Yield Spread(3-5 year)",
        "US Corporate Bond Yield Spread(5-7 year)",
        "US Corporate Bond Yield Spread(7-10 year)",
        "US Corporate Bond Yield Spread(10+ year)",
        "US Generic Govt 3 Month Yield",
        "US Generic Govt 6 Month Yield",
        "US Generic Govt 12 Month Yield",
        "US Generic Govt 2 Year Yield",
        "US Generic Govt 5 Year Yield",
        "US Generic Govt 7 Year Yield",
        "US Generic Govt 10 Year Yield",
        "1-year_GDP_growth",
        "1-year_CPI_growth",
        "1-year_PPI_growth",
        "1-year_M1_growth",
        "1-year_M2_growth",
        "employment_rate",
        "interest_rate",
        "Ratio to MA",
        "SP500 MD",
        "duration_in_years",
        "defaulted_in_last_5_years",
        "treasury_rate",
        "economic_policy_uncertainty",
        "1-year_housing",
    ]

def load_dataset(data_dir, only_36=True):
    """
    Load train/test features and labels from Excel files.
    If only_36=True, select the 36 columns from get_36_columns().
    Otherwise, use all intersecting columns between train/test.
    
    Args:
        data_dir (str): Path to the directory containing the Excel files.
        only_36 (bool): Whether to only select the 36 specified columns (True)
                        or use all intersecting columns (False).
    
    Returns:
        X_train (pd.DataFrame): Training features.
        X_test  (pd.DataFrame): Testing features.
        y_train (np.array): Training labels (ravel).
        y_test  (np.array): Testing labels (ravel).
    """
    # Load data from Excel files
    train_features_df = pd.read_excel(os.path.join(data_dir, "train_features2.xlsx"))
    test_features_df = pd.read_excel(os.path.join(data_dir, "test_features2.xlsx"))
    train_labels_df = pd.read_excel(os.path.join(data_dir, "train_labels2.xlsx"))
    test_labels_df = pd.read_excel(os.path.join(data_dir, "test_labels2.xlsx"))
    
    if only_36:
        # Select only the 36 specified columns, preserving order and intersection
        columns_36 = get_36_columns()
        common_36_train = list(set(train_features_df.columns).intersection(columns_36))
        common_36_test = list(set(test_features_df.columns).intersection(columns_36))
    
        # Final list to ensure the same ordering and intersection
        final_columns_train = [col for col in columns_36 if col in common_36_train]
        final_columns_test = [col for col in columns_36 if col in common_36_test]
    
        # Make sure train and test columns match in order
        common_final_columns = list(set(final_columns_train).intersection(final_columns_test))
        common_final_columns.sort(key=lambda x: columns_36.index(x))  # preserve original order
    
        X_train = train_features_df[common_final_columns].copy()
        X_test = test_features_df[common_final_columns].copy()
    else:
        # Use all intersecting columns between train and test datasets
        common_columns = list(set(train_features_df.columns).intersection(set(test_features_df.columns)))
        X_train = train_features_df[common_columns].copy()
        X_test = test_features_df[common_columns].copy()
    
    # Clean the training set: drop rows with missing features and align labels
    X_train = X_train.dropna()
    train_labels_df = train_labels_df.loc[X_train.index].dropna()
    X_train = X_train.loc[train_labels_df.index]
    
    # Clean the testing set similarly
    X_test = X_test.dropna()
    test_labels_df = test_labels_df.loc[X_test.index].dropna()
    X_test = X_test.loc[test_labels_df.index]
    
    y_train = train_labels_df.values.ravel()
    y_test = test_labels_df.values.ravel()
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test

def reconstruct_categories(df_input):
    
    prefix_list = ['Industry_group', 'Industry_sector', 'Industry_subgroup', 
            'currency', 'domicile_country', 'exchange_country', 
            'event_type', 'event_type_subcategory_sum', 'seniorioty_adj']

    df = df_input.copy()
    
    for prefix in prefix_list:
        # Select all dummy columns for this prefix
        dummy_cols = [col for col in df.columns if col.startswith(prefix + "_")]
        if dummy_cols:
            # Create the original column by checking which dummy is True
            df[prefix] = df[dummy_cols].idxmax(axis=1).str[len(prefix)+1:]
            # Drop the dummy columns if you don't need them anymore
            df = df.drop(columns=dummy_cols)
    return df

# If you want this module to run as a standalone script for testing or demo purposes,
# wrap the execution code in the following block.
if __name__ == "__main__":
    # Adjust the directory as needed
    X_train, X_test, y_train, y_test = load_dataset("../data/UP5/", only_36=False)
