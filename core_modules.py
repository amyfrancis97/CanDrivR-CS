# core_modules.py
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score)
from optimisation.selected_features import features
import numpy as np

def load_data(file_path, compression='gzip', features = features, sep='\t'):
    """ 
    Load dataset from a file path and ensure all features are present. 
    If any feature from the `features` list is missing, it is added with NaN values.
    """
    # Load the dataset
    df = pd.read_csv(file_path, compression=compression, sep=sep)
    
    # Check for missing features in the dataframe
    missing_features = [feature for feature in features if feature not in df.columns]
    
    # Add missing features as NaN
    for feature in missing_features:
        df[feature] = np.nan
    
    return df

def check_duplicates_between_datasets(df_train, df_test):
    # Create a unique identifier in both datasets
    df_train['id'] = df_train['chrom'].astype(str) + "_" + df_train['pos'].astype(str) + "_" + df_train['ref_allele'] + "_" + df_train['alt_allele']
    df_test['id'] = df_test['chrom'].astype(str) + "_" + df_test['pos'].astype(str) + "_" + df_test['ref_allele'] + "_" + df_test['alt_allele']

    # Find duplicates by checking the intersection of the 'id' columns in both datasets
    duplicates = set(df_train['id']).intersection(set(df_test['id']))

    # If there are duplicates, print a warning and remove them from df_test
    if duplicates:
        print(f"Warning: {len(duplicates)} test variants are present in the training dataset and will be removed.")
        for dup in duplicates:
            print(f"Duplicate variant: {dup}")

        # Filter df_test to exclude duplicates
        df_test_filtered = df_test[~df_test['id'].isin(duplicates)].copy()
    else:
        print("No duplicate variants found between training and test datasets.")
        df_test_filtered = df_test.copy()  # No changes needed

    # Drop the 'id' column from df_test_filtered
    df_test_filtered.drop(columns=['id'], inplace=True)
    
    return df_test_filtered

def save_data(df, file_path, compression='gzip', sep='\t'):
    """ Save dataframe to a file path. """
    df.to_csv(file_path, sep=sep, compression=compression, index=False)

def load_pickle(file_path):
    """ Load pickle file. """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def standardize_data(data):
    """ Standardize data using StandardScaler. """
    scaler = StandardScaler()
    return scaler.fit_transform(data)



