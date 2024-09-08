
from core_modules import load_data, save_data, check_duplicates_between_datasets
from data_processing import prepare_cancer_specific_datasets, split_dataset_cancer
from model_training import train_baseline_model, evaluate_model_on_cosmic
from visualisation import *
from config import *
from optimisation.selected_features import features
from collections import Counter
from optimisation.optimise_donor_counts import optimise_cancer_spec_donor_count, compile_results, get_test_cross_val_res
from data_processing import filter_skcm_dataset, add_missing_columns
from model_training import run_and_test_tcga_model, run_cancer_specific_tests
import math
from plots.donor_count_vs_accuracy import plot_grid_of_curves
from plots.plot_cancer_specific_heatmap import plot_cancer_spec_heatmap
from plots.heatmap_feature_importance import *
from plots.plot_dna_shape_position_means import *
from collections import Counter

def main():
    """
    Main function to run the CanDrivR-CS cancer model training, evaluation and visualisation pipeline.
    """
    # Step 1: Load datasets
    ICGC = load_data(f"{DATA_DIR}/ICGC.tsv.gz")
    COSMIC = load_data(f"{DATA_DIR}/COSMIC.tsv.gz")

    # Step 2: Check to ensure no training variants from ICGC are found in the COSMIC test data.
    print("Checking for duplicate variants between ICGC and COSMIC datasets...")
    COSMIC_updated = check_duplicates_between_datasets(ICGC, COSMIC)

    # Step 3: Train and evaluate baseline pan-cancer model on ICGC
    print("Training baseline pan-cancer model...")
    final_model, feature_importance = train_baseline_model(ICGC, features, 
        save_path_test=f"{OUTPUT_DIR}/baseline_test_results.tsv", 
        save_path_cross_val=f"{OUTPUT_DIR}/baseline_cross_val_results.tsv")

    # Step 4: Evaluate the baseline model on the COSMIC dataset
    print("Evaluating baseline model on COSMIC data...")
    baseline_accuracy = evaluate_model_on_cosmic(final_model, COSMIC_updated, features, 
        save_path=f"{OUTPUT_DIR}/COSMIC_eval_results.tsv")

    # Step 5: Split ICGC training data into cancer-specific datasets
    print("Splitting ICGC dataset into cancer-specific datasets...")
    split_dataset_cancer(ICGC, DATA_DIR)
    cancer_datasets = prepare_cancer_specific_datasets(DATA_DIR)

    # Step 6: Optimise cancer-specific donor counts and compile results
    print("Optimising donor counts for cancer-specific models...")
    print("This may take some time since we are optimising 50 cancer models...")
    optimised_cancer_results, cancer_dict_donor_counts = optimise_cancer_spec_donor_count(cancer_datasets, ICGC)

    # Step 7: Compile results for visualisation
    cancer_specific_list, cancer_specific_df = compile_results(optimised_cancer_results)

    # Step 8: Plot donor counts for each cancer type
    print("Plotting optimised donor counts for each cancer dataset...")
    plot_grid_of_curves(cancer_specific_list, f"{OUTPUT_DIR}/", cancer_specific_df["cancer"].unique())

    # Step 9: Cross-validation and test results for heatmap
    print("Compiling cross-validation and test results for heatmap plotting...")
    optimised_metrics_cross_val, optimised_metrics_test = get_test_cross_val_res(
        optimised_cancer_results, cancer_specific_df, cancer_dict_donor_counts,
        cross_val_path=f"{OUTPUT_DIR}/cancer_specific_cross_val_results.tsv", 
        test_path=f"{OUTPUT_DIR}/cancer_specific_test_results.tsv"
    )

    # Step 10: Plot heatmaps for test and validation metrics
    print("Plotting heatmaps for test and cross-validation metrics...")
    plot_cancer_spec_heatmap(optimised_metrics_test.sort_values("size", ascending=False), f"{OUTPUT_DIR}/test_heatmap.png")
    plot_cancer_spec_heatmap(optimised_metrics_cross_val.sort_values("size", ascending=False), f"{OUTPUT_DIR}/cross_validation_heatmap.png")

    # Step 11: Plot feature importance heatmap
    print("Plotting feature importance heatmap...")
    plot_feature_importance_heatmap(ICGC, optimised_metrics_cross_val, cancer_dict_donor_counts, OUTPUT_DIR, features)

    # Step 12: Test baseline model on TCGA UCEC dataset
    print("Running model on TCGA UCEC dataset...")
    TCGA_UCEC = load_data("/Volumes/Seagate5TB/data/CanDrivR-data-final/TCGA_UCEC_test.txt.gz")
    UCEC_updated = check_duplicates_between_datasets(ICGC, TCGA_UCEC)
    run_and_test_tcga_model(
        ICGC=ICGC, 
        results_file=f"{OUTPUT_DIR}/TCGA_UCEC_results.tsv", 
        TCGA_df=UCEC_updated, 
        study_id='UCEC', 
        positive_dataset_donor_count=3, 
        features=features
    )

    # Step 13: Test baseline model on TCGA SKCM dataset
    print("Running model on TCGA SKCM dataset...")
    TCGA_SKCM = load_data("/Volumes/Seagate5TB/data/CanDrivR-data-final/TCGA_SKCM_test.txt.gz")
    SKCM_updated = check_duplicates_between_datasets(ICGC, TCGA_SKCM)
    run_and_test_tcga_model(
        ICGC=ICGC, 
        results_file=f"{OUTPUT_DIR}/TCGA_SKCM_results.tsv", 
        TCGA_df=SKCM_updated, 
        study_id='SKCM', 
        positive_dataset_donor_count=4, 
        features=features
    )


if __name__ == "__main__":
    main()
