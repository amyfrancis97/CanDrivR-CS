import pandas as pd
import matplotlib.pyplot as plt 
from models.run_models import run_model
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import numpy as np

#%%
def plot_feature_importance_heatmap(df, optimised_metrics_cross_val, cancer_dict_donor_counts, output_dir, features):
    """
    Extracts feature importance for each cancer type and plots the heatmap.
    
    Parameters:
    df (pd.DataFrame): The main dataset.
    optimised_metrics_cross_val (pd.DataFrame): Optimised cross-validation metrics.
    cancer_dict_donor_counts (dict): Dictionary of donor counts for each cancer.
    output_dir (str): Path to save the heatmap.
    features (list): List of features to include in the model.
    
    Returns:
    pd.DataFrame: Standardised heatmap data for plotting.
    """

    # Extract feature importances for the top 10 cancers
    cancers_to_test = optimised_metrics_cross_val.sort_values("f1", ascending=False)["cancer"].tolist()[:10]
    
    # Initialise an empty dictionary to store feature importances for each cancer type
    features_dict = {}

    for cancer in cancers_to_test:
        # Run the model to get feature importance for each cancer type
        _, _, _, _, _, feature_importance = run_model(df, study_id=cancer, 
                                                      positive_dataset_donor_count=cancer_dict_donor_counts[cancer], 
                                                      feature_importance=True, features=features)
        
        # Store the full feature importance data, indexed by feature
        features_dict[cancer] = feature_importance.set_index('feature')

    # Collect top 15 features for each cancer
    all_top_features = []
    for cancer in cancers_to_test:
        top_features = features_dict[cancer]['importance'].nlargest(15).index.tolist()
        all_top_features.extend(top_features)

    # Unique set of top features
    unique_top_features = list(set(all_top_features))

    # Initialise an empty DataFrame with these features as columns
    heatmap_data = pd.DataFrame(index=cancers_to_test, columns=unique_top_features)

    # Fill the DataFrame with actual importance values where they exist
    for cancer in cancers_to_test:
        available_features = features_dict[cancer].index
        for feature in unique_top_features:
            if feature in available_features:
                heatmap_data.at[cancer, feature] = features_dict[cancer].loc[feature, 'importance']

    # Clean up long column names
    heatmap_data = heatmap_data.rename(columns={
        'mutant_AA_Free_energy_change_of_alpha(Ri)_to_alpha(Rh)_(Wertz-Scheraga,_1978)': 'mutant_AA_energy_change_(Ri/Rh)_(Wertz-Scheraga,_1978)',
        'WT_AA_Normalized_frequency_of_N-terminal_helix_(Chou-Fasman,_1978b)': 'WT_AA_freq_N-term_helix_(Chou-Fasman,_1978b)',
        'mutant_AA_Normalized_frequency_of_alpha-helix_in_alpha/beta_class_(Palau_et_al.,_1981)': 'mutant_AA_freq_a-helix_(Palau_et_al.,_1981)',
        'WT_AA_Average_relative_fractional_occurrence_in_A0(i-1)_(Rackovsky-Scheraga,_1982)': 'WT_AA_occ_in_A0(i-1)_(Rackovsky-Scheraga,_1982)',
        'WT_AA_occurrence_in_A0(i-1)(Rackovsky-Scheraga,_1982)': 'WT_AA_occ_A0(i-1)(Rackovsky-Scheraga,_1982)'
    })

    # Remove duplicated columns and fill NaN values with 0
    heatmap_data = heatmap_data.loc[:, ~heatmap_data.columns.duplicated()]
    heatmap_data.fillna(0, inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    standardised_data = scaler.fit_transform(heatmap_data.fillna(0))

    # Convert standardized data back to DataFrame
    standardised_heatmap_data = pd.DataFrame(standardised_data, index=heatmap_data.index, columns=heatmap_data.columns)

    # Save and plot the heatmap
    feature_importance_top_cancers(standardised_heatmap_data, output_dir)

    return standardised_heatmap_data

#%%
def feature_importance_top_cancers(standardized_heatmap_data, output_dir):
    """
    Plot and save the feature importance heatmap for top cancers.

    Parameters:
    standardized_heatmap_data (pd.DataFrame): Standardized feature importance data.
    output_dir (str): Path to save the heatmap.
    """
    # Plot the heatmap
    plt.figure(figsize=(18, 16))
    heatmap = sns.heatmap(standardized_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 13}, cbar=False)

    # Adjust y-axis ticks
    yticks_positions = np.arange(len(standardized_heatmap_data.columns)) + 0.5  # Calculate tick positions
    #yticks_labels = standardized_heatmap_data.columns  # Get tick labels

    # Truncate y-axis labels that are longer than 30 characters
    yticks_labels = [label[:26] + '...' if len(label) > 30 else label for label in standardized_heatmap_data.columns]

    plt.yticks(yticks_positions, yticks_labels, fontsize=16)  # Set tick positions and labels
    plt.xticks(fontsize=15)

    # Set title and labels
    plt.title('Feature Importance Heatmap for Top Cancers', fontsize=18)
    plt.xlabel('Cancer', fontsize=18)
    plt.ylabel('Features', fontsize=18)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_heatmap_cancer_spec.png', dpi=300)
    plt.show()
# %%

