# data_processing.py
import pandas as pd
from core_modules import load_data, save_data, load_pickle
import pickle

def split_dataset_cancer(df, output_dir):
    """
    Splits a dataset into separate datasets based on different cancer types identified by 'study_id'.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data from multiple cancer studies.

    This function does not return any value; instead, it saves the split datasets into a file for later use.
    """

    # Extract unique cancer types from the 'study_id' column
    cancer_types = df['study_id'].unique()
    
    # Create a dictionary where each key is a cancer type, and the value is the subset of the dataframe corresponding to that type
    cancer_datasets = {cancer: df[df['study_id'] == cancer] for cancer in cancer_types}

    # Save the dictionary containing the split datasets to a pickle file
    with open(f'{output_dir}/cancer_datasets.pkl', 'wb') as f:
        pickle.dump(cancer_datasets, f)

def filter_cancer_data(cancer_data, donor_count=4):
    """ Filter cancer data based on donor count. """
    rare = cancer_data[cancer_data["donor_count"] == 1]
    recurrent = cancer_data[cancer_data["donor_count"] > donor_count]
    filtered_data = pd.concat([rare.sample(len(recurrent)), recurrent]).reset_index(drop=True)
    return filtered_data

def prepare_cancer_specific_datasets(data_dir):
    """ Load and prepare cancer-specific datasets. """
    return load_pickle(f'{data_dir}/cancer_datasets.pkl')

def filter_skcm_dataset(cancer_datasets, output_path):
    """ Filter SKCM dataset for rare and recurrent cases, then save to file. """
    SKCM_data = cancer_datasets["SKCM"]
    SKCM_rare = SKCM_data[SKCM_data["donor_count"] == 1]
    SKCM_recurrent = SKCM_data[SKCM_data["donor_count"] > 4]
    SKCM_filtered = pd.concat([SKCM_rare.sample(len(SKCM_recurrent)), SKCM_recurrent]).reset_index(drop=True)
    SKCM_filtered.to_csv(output_path, sep="\t", index=None, compression="gzip")
    return SKCM_filtered

def add_missing_columns(df, features):
    """ Add missing columns to a DataFrame with NaN values. """
    missing_features = [feature for feature in features if feature not in df.columns]
    for feature in missing_features:
        df[feature] = pd.NA
    return df
