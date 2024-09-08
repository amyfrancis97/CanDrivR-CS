from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import time
import random

def prepare_data(df, features=None):
    """
    Prepare the data for training.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the dataset.
        features (list, optional): List of feature columns. If None, default columns are used. Default is None.

    Returns:
        tuple: A tuple containing X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test.
    """
    if features is None:
        X = df.drop(["chrom", "pos", "ref_allele", "alt_allele", "driver_stat"], axis=1)
    elif isinstance(features, list):
        if len(features) > 1:
            X = df[features + ["grouping"]]
        elif len(features) == 1:
            X = df[features + ["grouping"]]
    y = df["driver_stat"]
    groups = df["grouping"].values
    #groups = df["chrom"].values
    # Get unique groups
    unique_groups = np.unique(groups)

    random.shuffle(unique_groups)

    # Define the test group
    test_group = unique_groups[0]

    # Define train_val groups
    train_val_groups = unique_groups[1:]

    # Filter test and train_val groups
    test_indices = groups == test_group
    train_val_indices = np.isin(groups, train_val_groups)

    X = X.drop("grouping", axis = 1)

    # Define test and train_val data
    X_test, y_test = X[test_indices], y[test_indices]
    X_train_val, y_train_val = X[train_val_indices], y[train_val_indices]
    groups_test, groups_train_val = groups[test_indices], groups[train_val_indices]

    return X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test

