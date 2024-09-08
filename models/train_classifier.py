#%%
import sys
sys.path.append('/Users/uw20204/Documents/scripts/CanDrivR-TS')
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import time
import sys
from models.prepare_training_data import prepare_data
from models.train_evaluate import train_and_evaluate_model

def train_classifier(df, features=None, classifier=None, feature_importance=False):
    """
    Train a classifier on the provided dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the dataset.
        features (list, optional): List of feature columns. If None, default columns are used. Default is None.
        classifier: Classifier object. If None, a default classifier is used.
    Returns:
        tuple: A tuple containing actual_predicted_targets, results_df, final_model_tuple, and results_tuple.
    """
    if classifier is None:
        classifier = XGBClassifier(random_state=42)

    X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test = prepare_data(df, features)

    start = time.time()
    metrics, final_model, test_results, actual_predicted = train_and_evaluate_model(
        classifier, X_train_val, y_train_val, groups_train_val, X_test, y_test
    )
    end = time.time()
    total_time = end - start
    test_results["time"] = format(total_time, ".3f")

    # Prepare final_model tuple with the model and list of feature columns
    final_model_tuple = (final_model, X_test.columns.tolist())

    # Prepare results tuple with validation_results and test_results
    validation_results = pd.DataFrame(metrics)

    results_tuple = (validation_results, test_results)

    if feature_importance:
        # Extract feature importance
        if isinstance(classifier, XGBClassifier):
            feature_importance = final_model.get_booster().get_score(importance_type="weight")
            feature_importance = pd.DataFrame(feature_importance.items()).sort_values(1, ascending=False)
            feature_importance.columns = ["feature", "importance"]
        else:
            # Calculate permutation importance
            perm_importance = permutation_importance(final_model, X_test, y_test, n_repeats=30, random_state=42)
            # Get the feature importance scores
            feature_importance = pd.DataFrame(
                {"feature": X_test.columns, "importance": perm_importance["importances_mean"]}
            )
            feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    else:
        feature_importance = np.nan

    return actual_predicted, feature_importance, final_model_tuple, results_tuple



# %%
