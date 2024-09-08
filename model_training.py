# model_training.py
import pandas as pd
from xgboost import XGBClassifier
from core_modules import load_data, save_data, standardize_data
from models.train_classifier import train_classifier
from models.metric_results_table import get_results_table
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
from data_processing import *
from models.run_models import *

def train_baseline_model(ICGC, features, save_path_test = "baseline_test_results.tsv", save_path_cross_val = "baseline_cross_val_results.tsv"):
    """ Train baseline model on ICGC data using XGBClassifier and return the trained model. """
    actual_predicted_targets, feature_importance, final_model, results = train_classifier(
        ICGC.drop("study_id", axis=1), features=features, classifier=XGBClassifier(random_state=42), feature_importance=True
    )
    
    # Save the model metrics
    XGB_results_metrics = get_results_table(results[0], model_name="XGB")
    XGB_results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
    save_data(XGB_results_metrics, save_path_cross_val, compression = None)
    save_data(results[1], save_path_test, compression = None)
    print("Baseline model completed!")
    print("Here's the cross-validation results:")
    print(XGB_results_metrics)
    print("File saved as: ", save_path_cross_val)
    print("Here's the test results:")
    print(results[1])
    print("File saved as: ", save_path_test)
    # Return the trained model and feature importance
    return final_model, feature_importance

def evaluate_model_on_cosmic(final_model, cosmic_data, shared_features, save_path = "baseline_COSMIC_evaluation.tsv"):
    """ Evaluate model on COSMIC data. """
    X_test = cosmic_data[shared_features]
    y_test = cosmic_data["driver_stat"]
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
    print("Model evaluation on COSMIC data completed!")
    print("Here's the evaluation results: ")
    print(test_results)
    print("File saved as: ", "baseline_COSMIC_evaluation.tsv")
    test_results.to_csv(save_path, sep = "\t", index = None)

def run_and_test_tcga_model(ICGC, results_file, TCGA_df, study_id, positive_dataset_donor_count, features):
    """ Run the model on ICGC and test on the provided TCGA dataset. """
    # Ensure missing columns are handled before model training
    TCGA_df = add_missing_columns(TCGA_df, features)
    
    # Run the model
    XGB_results_metrics, ICGC, features_updated, results, cancer_testing_results, feature_importance = run_model(ICGC, results_file, study_id=study_id, positive_dataset_donor_count=positive_dataset_donor_count,
                     features=features, test_dataset_df=TCGA_df)

    print("Evaluation completed! Here's the results: ")
    print(results)
    results.to_csv(results_file, sep = "\t", index = None)
    print("Results saved to: ", results_file)


def run_cancer_specific_tests(ICGC, cancer_dict_donor_counts, cancers_to_test, df):
    """ Test cancer-specific models on all other cancers. """
    return [test_cancers_for_model(cancer, cancers_to_test, cancer_dict_donor_counts, df) for cancer in cancers_to_test]
