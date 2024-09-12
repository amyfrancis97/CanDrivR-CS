#%%
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
def train_and_evaluate_model(classifier, X_train_val, y_train_val, groups_train_val, X_test, y_test):
    """
    Train the model using Leave-One-Group-Out cross-validation and evaluate its performance.

    Parameters:
        classifier: The classifier model object.
        X_train_val (pd.DataFrame): Features of the training/validation set.
        y_train_val (pd.Series): Target variable of the training/validation set.
        groups_train_val (pd.Series): Groups corresponding to the training/validation set.
        X_test (pd.DataFrame): Features of the test set.
        y_test (pd.Series): Target variable of the test set.

    Returns:
        tuple: A tuple containing metrics, actual_targets, predicted_targets, final_model, test_accuracy, y_test_pred.
    """
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    # Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    actual_targets = np.array([])
    predicted_targets = np.array([])
    for train_index, val_index in logo.split(X_train_val, y_train_val, groups_train_val):
        model = classifier
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
        # Random undersampling to balance the training data
        sampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

        # Model fitting
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on validation set
        y_val_pred = model.predict(X_val)
        actual_targets = np.append(actual_targets, y_val)
        predicted_targets = np.append(predicted_targets, y_val_pred)

        # Evaluate and store metrics
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics["accuracy"].append(accuracy_score(y_val, y_val_pred))
        metrics["precision"].append(precision_score(y_val, y_val_pred))
        metrics["recall"].append(recall_score(y_val, y_val_pred))
        metrics["f1"].append(f1_score(y_val, y_val_pred))
        metrics["roc_auc"].append(roc_auc_score(y_val, y_val_pred_proba))

    # Fit the final model on the entire training/validation set
    final_model = classifier
    final_sampler = RandomUnderSampler(random_state=42)
    X_train_val_resampled, y_train_val_resampled = final_sampler.fit_resample(X_train_val, y_train_val)
    final_model.fit(X_train_val_resampled, y_train_val_resampled)

    # Predict and evaluate on the test set
    y_test_pred = final_model.predict(X_test[X_train_val_resampled.columns.tolist()])
    test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    test_results = pd.DataFrame(
        {
            "balanced_accuracy": [test_accuracy],
            "precision": [precision_score(y_test, y_test_pred)],
            "recall": [recall_score(y_test, y_test_pred)],
            "f1": [f1_score(y_test, y_test_pred)],
            "roc_auc": [roc_auc_score(y_test, y_test_pred)],
        }
    )

    actual_predicted_targets_cross_val = (actual_targets, predicted_targets)
    actual_predicted_targets_test = (y_test, y_test_pred)
    actual_predicted = (actual_predicted_targets_cross_val, actual_predicted_targets_test)

    return metrics, final_model, test_results, actual_predicted

# %%
