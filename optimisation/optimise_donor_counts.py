import sys
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR-TS')

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
import time
import importlib
from sklearn.metrics import roc_curve, auc
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn import metrics, model_selection, preprocessing, datasets, svm
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_squared_error, confusion_matrix, balanced_accuracy_score)
from sklearn.model_selection import (train_test_split, GridSearchCV, RepeatedStratifiedKFold, 
                                     LeaveOneGroupOut, KFold, cross_val_predict)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tensorflow.keras.models import clone_model
from xgboost import XGBClassifier, plot_importance
from keras import models, layers, regularizers
from models.prepare_training_data import prepare_data

from models.train_evaluate import train_and_evaluate_model
from models.run_models import run_model

from models.metric_results_table import get_results_table
from models.train_classifier import train_classifier
from optimisation.selected_features import features
from optimisation.params import *
from optimisation.best_donor_count import cancer_best_donor_count
import pickle
from plots.donor_count_vs_accuracy import *
import scipy.stats as stats
#%%
def optimise_cancer_spec_donor_count(cancer_datasets, df_tissue_spec1):
    """
    Runs an optimisation process for donor count thresholds on cancer-specific models and logs the results.

    Parameters:
    cancer_datasets (dict): A dictionary with keys as cancer types and values as DataFrames specific to each cancer.
    df_tissue_spec1 (pd.DataFrame): The DataFrame containing feature data used for model training.

    Returns:
    list: A list containing the results of optimisation for each cancer type.
    """
    optimised_cancer_results = []
    cancer_dict_donor_counts = Counter() # Initialise an empty dictionary for storing optimised donor counts for each cancer
    # Run a general cancer-specific model using XGB
    # Optimise the donor count threshold for each cancer type then run XGB
    for cancer_dataset in list(cancer_datasets.keys()):
        cancer_dict_donor_counts[cancer_dataset] = 0
        tracker = 0
        for dc in range(1, 10):
            try:
                XGB_results_metrics, ICGC, features, test_results, _, _ = run_model(df_tissue_spec1, positive_dataset_donor_count = dc, study_id = cancer_dataset)
                f1 = XGB_results_metrics["f1"].str.split(" ± ", expand=True)[0].astype("float").item()
                if f1 > tracker:
                    cancer_dict_donor_counts[cancer_dataset] = dc # Update the donor count if the f1 score is higher
                    tracker = f1 # Update f1 tracker
                optimised_cancer_results.append([cancer_dataset, dc, len(ICGC), XGB_results_metrics, test_results, ICGC, cancer_dict_donor_counts]) # Create a list for plotting
            except:
                dataset_results = np.nan
    
    return optimised_cancer_results, cancer_dict_donor_counts

def compile_results(optimised_cancer_results):
    """
    Compiles and organises the optimised results into a DataFrame, and creates a list of DataFrames filtered by cancer type.

    Parameters:
    optimised_cancer_results (list): A list containing results from the optimisation process for each cancer type.

    Returns:
    tuple: A tuple containing:
        - A DataFrame with all combined results.
        - A list of DataFrames, each filtered to contain results specific to a single cancer type.
    """

    cancers = [optimised_cancer_results[i][0] for i in range(len(optimised_cancer_results))]
    donor_counts = [optimised_cancer_results[i][1] for i in range(len(optimised_cancer_results))]
    dataset_size = [optimised_cancer_results[i][2] for i in range(len(optimised_cancer_results))]
    F1 = [optimised_cancer_results[i][3]["f1"].item().split(" ± ")[0] for i in range(len(optimised_cancer_results))]
    F1_test = [optimised_cancer_results[i][4]["f1"].item() for i in range(len(optimised_cancer_results))]
    results_cancer_spec = pd.DataFrame(data = {"cancer": cancers, "donor_count": donor_counts, "size": dataset_size, "f1": F1, "f1_test": F1_test})
    cancer_specific_results = [results_cancer_spec[results_cancer_spec["cancer"] == cancer] for cancer in results_cancer_spec["cancer"].unique()]

    return cancer_specific_results, results_cancer_spec

def get_test_cross_val_res(optimised_cancer_results, results_cancer_spec, cancer_dict_donor_counts, cross_val_path, test_path):
    F1 = [optimised_cancer_results[i][3]["f1"].item() for i in range(len(optimised_cancer_results))]
    res = [optimised_cancer_results[i][3] for i in range(len(optimised_cancer_results))]
    res = pd.concat([results_cancer_spec.reset_index(drop = True).drop("f1", axis = 1), pd.concat(res).reset_index(drop = True)], axis = 1)
    res = res.rename(columns = {"balanced_accuracy": "accuracy"})
    optimised_res = []
    for cancer_name in list(cancer_dict_donor_counts.keys()):
        opt_donor_count = cancer_dict_donor_counts[cancer_name]
        result = res[(res["cancer"] == cancer_name) & (res["donor_count"] == opt_donor_count)]
        optimised_res.append(result)
    optimised_metrics_cross_val = pd.concat(optimised_res).reset_index(drop = True)
    F1 = [optimised_cancer_results[i][4]["f1"].item() for i in range(len(optimised_cancer_results))]
    res = [optimised_cancer_results[i][4] for i in range(len(optimised_cancer_results))]
    res = pd.concat([results_cancer_spec.reset_index(drop = True).drop("f1", axis = 1), pd.concat(res).reset_index(drop = True)], axis = 1)
    res = res.rename(columns = {"balanced_accuracy": "accuracy"})
    optimised_res = []
    for cancer_name in list(cancer_dict_donor_counts.keys()):
        opt_donor_count = cancer_dict_donor_counts[cancer_name]
        result = res[(res["cancer"] == cancer_name) & (res["donor_count"] == opt_donor_count)]
        optimised_res.append(result)
    optimised_metrics_test = pd.concat(optimised_res).reset_index(drop = True)
    optimised_metrics_cross_val = optimised_metrics_cross_val[optimised_metrics_cross_val["size"] > 100].reset_index(drop = True)
    optimised_metrics_test = optimised_metrics_test[optimised_metrics_test["size"] > 100].reset_index(drop = True)

    print("The optimised cancer-specific cross validation results are: ")
    print(optimised_metrics_cross_val)
    print("Results saved to: ", cross_val_path)
    optimised_metrics_cross_val.to_csv(cross_val_path, sep = "\t", index = None)

    print("The optimised cancer-specific test results are: ")
    print(optimised_metrics_test)
    print("Results saved to: ", test_path)
    optimised_metrics_cross_val.to_csv(test_path, sep = "\t", index = None)

    return optimised_metrics_cross_val, optimised_metrics_test

# %%
