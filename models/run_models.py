#%%
import sys
import os 

# Check if running in a script or interactive environment
if '__file__' in globals():
    # Automatically find the root directory based on the location of this script
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
else:
    # If in an interactive environment, manually set the root directory
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))

# Add the root directory to the system path
sys.path.append(ROOT_DIR)

#%%
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

from models.metric_results_table import get_results_table
from models.train_classifier import train_classifier
from optimisation.selected_features import features
from optimisation.params import *
#from optimisation.best_donor_count import cancer_best_donor_count
import pickle
from plots.donor_count_vs_accuracy import *
import scipy.stats as stats
#%%
def run_model(df: pd.DataFrame, results_file: str = "test_results.tsv", study_id: str = None, positive_dataset_donor_count: int = 2, features: list = features, test_dataset_cancer_name: str = None, test_dataset_df = pd.DataFrame(), feature_importance: bool = False ) -> tuple:
    """
    Runs a baseline gradient boosting model on a balanced ICGC dataset.

    Parameters:
    df (pd.DataFrame): The dataframe containing the ICGC dataset.
    study_id (str, optional): The study identifier to filter the dataframe. If provided, only data from this study will be used.
    positive_dataset_donor_count (int, optional): The threshold for considering variants as 'positive' based on donor count.
    features (list, optional): A list of features to include in the model.

    Returns:
    tuple: Contains the metrics of the XGB model, the modified ICGC dataframe, the list of updated features, and the model itself.
    """
    
    # Filter the dataframe by study_id if provided, otherwise use the entire dataframe
    if study_id:
        ICGC = df[df["study_id"] == study_id]
    else:
        ICGC = df

    # Balance the dataset by sampling non-positive cases to match the number of positive cases
    ICGC = pd.concat([
        ICGC[ICGC["donor_count"] == 1].sample(len(ICGC[ICGC["donor_count"] > positive_dataset_donor_count])), 
        ICGC[ICGC["donor_count"] > positive_dataset_donor_count]
    ], axis=0)

    # Reset the index of the dataframe
    ICGC = ICGC.reset_index(drop=True)

    # Identify features present both in the provided list and the dataframe's columns
    features_updated = list(set(ICGC.columns.tolist()) & set(features))

    if feature_importance:
        # Train and evaluate the classifier using the updated feature set and XGBClassifier
        actual_predicted_targets, feature_importance, final_model, results = train_classifier(ICGC, features=features_updated, classifier=XGBClassifier(random_state=42), feature_importance = True)

    else:
        # Train and evaluate the classifier using the updated feature set and XGBClassifier
        actual_predicted_targets, feature_importance, final_model, results = train_classifier(ICGC, features=features_updated, classifier=XGBClassifier(random_state=42))
        
    # Get results table for the model evaluation
    XGB_results_metrics = get_results_table(results[0], model_name="XGB")

    if test_dataset_cancer_name:
        if isinstance(test_dataset_cancer_name, list):
            cancer_testing_results = []
            for cancer_type in test_dataset_cancer_name:
                to_test = df[df["study_id"] == cancer_type]
                X_test = to_test[final_model[1]]
                y_test = to_test["driver_stat"]

                # Predict and evaluate on the test set
                y_test_pred = final_model[0].predict(X_test)
                test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

                test_results = pd.DataFrame(
                    {
                        "balanced_accuracy": [test_accuracy],
                        "precision": [precision_score(y_test, y_test_pred)],
                        "recall": [recall_score(y_test, y_test_pred)],
                        "f1": [f1_score(y_test, y_test_pred)],
                        "roc_auc": [roc_auc_score(y_test, y_test_pred)],
                        "test_dataset": [cancer_type]
                    }
                )
                cancer_testing_results.append(test_results)
    else:
        cancer_testing_results = list()

    if not test_dataset_df.empty:
        to_test = test_dataset_df.copy()

        # Remove any variants from test data that are present in training data
        ICGC["id"] = ICGC["chrom"] + "_" + ICGC["pos"].astype(str) + "_" + ICGC["ref_allele"] + "_" + ICGC["alt_allele"]
        to_test["id"] = to_test["chrom"] + "_" + to_test["pos"].astype(str) + "_" + to_test["ref_allele"] + "_" + to_test["alt_allele"]
        to_test = to_test[~to_test['id'].isin(list(set(ICGC["id"]) & set(to_test["id"])))].drop("id", axis = 1).reset_index(drop = True)
        to_test =  pd.concat([to_test[to_test["driver_stat"] == 0], to_test[to_test["driver_stat"] == 1].sample(len(to_test[to_test["driver_stat"] == 0]))]).reset_index(drop = True)
        driver_status = to_test["driver_stat"]
        X_test = to_test[final_model[1]]
        y_test = to_test["driver_stat"]

        # Predict and evaluate on the test set
        y_test_pred = final_model[0].predict(X_test)
        y_test_pred_proba = final_model[0].predict_proba(X_test)[:, 1]
        test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

        test_results = pd.DataFrame(
            {
                "balanced_accuracy": [test_accuracy],
                "precision": [precision_score(y_test, y_test_pred)],
                "recall": [recall_score(y_test, y_test_pred)],
                "f1": [f1_score(y_test, y_test_pred)],
                "roc_auc": [roc_auc_score(y_test, y_test_pred)]
            }
        )
        
        fpr, tpr, thresholds = roc_curve(driver_status, y_test_pred_proba)
        roc_auc = auc(fpr, tpr)
        curve_plotting = (fpr, tpr, thresholds, roc_auc)
        cancer_testing_results = (test_results, curve_plotting, X_test)
#    print("Evaluation completed!")
    return XGB_results_metrics, ICGC, features_updated, results[1], cancer_testing_results, feature_importance
#%%
def test_cancers_for_model(cancer_to_model, cancers_to_test, cancer_dict_donor_counts, df):
    cancers_to_test2 = cancers_to_test.copy()
    cancers_to_test2.remove(cancer_to_model)
    print("cancer_to_model:", cancer_to_model)
    print("cancer_to_test:", cancers_to_test2)
    
    try:
        _, _, _, _, test_results, _ = run_model(
            df=df, 
            study_id=cancer_to_model, 
            positive_dataset_donor_count=cancer_dict_donor_counts[cancer_to_model], 
            features=features, 
            test_dataset_cancer_name=cancers_to_test2
        )
        result = pd.concat(test_results)
    except Exception as e:
        print(f"Error: {e}")  # Print the exception for debugging
        result = pd.DataFrame()  # Handle exceptions

    result["modeled_cancer"] = cancer_to_model
    return result

# %%
