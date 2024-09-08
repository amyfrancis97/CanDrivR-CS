import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
#%%
def plot_cancer_spec_heatmap(optimised_metrics, output_dir):
    # Only keep cancers where dataset size is > 100
    optimised_metrics = optimised_metrics[optimised_metrics["size"] > 100]
    optimised_metrics = optimised_metrics.rename(columns = {'accuracy': 'Acc', 'precision': 'Prec', 'recall': 'Rec', 'f1': 'F1', 'roc_auc': 'AUC'})
    # Assuming 'df' is your DataFrame
    metrics = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
    # Create a new DataFrame for heatmap data
    heatmap_data = pd.DataFrame()

    
    for metric in metrics:
        try:
            # Split the metric and its standard deviation, and only keep the metric values
            heatmap_data[metric] = optimised_metrics[metric].apply(lambda x: float(x.split(' ± ')[0]))
        except:
            heatmap_data[metric] = optimised_metrics[metric]

    # Set the index to cancer types for better visualization
    heatmap_data.set_index(optimised_metrics['cancer'] + " " + "(" + optimised_metrics['size'].astype(str) + ")", inplace=True)
    plt.figure(figsize=(10, 13))  # Adjust size as necessary
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 14}, cbar=False)

    plt.title('Cross Validation Metrics Heatmap by Cancer Type', fontsize = 18)
    plt.ylabel('Cancer Type', fontsize = 16)
    #plt.xlabel('Metrics', fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 16)
    #plt.xticks(rotation=45)  # Rotate metric labels for better readability
    plt.tight_layout()  # Adjust layout to make sure nothing is clipped
    plt.savefig(output_dir, dpi=300)  # Save the figure
    plt.show()
