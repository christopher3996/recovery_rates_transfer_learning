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

from experiments.up5.data import get_36_columns, load_dataset

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
