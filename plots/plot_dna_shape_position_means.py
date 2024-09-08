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
from scipy.interpolate import make_interp_spline
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
#%%

def plot_position_means(ax, cancer_data, dna_shape_features):
    custom_palette = sns.color_palette("Dark2", 4)
    sns.set(style="whitegrid")

    lines = {}
    legend_labels = []

    for index, feature in enumerate(dna_shape_features):
        print(feature)
        prot_cols = [col for col in cancer_data.dropna().columns if feature in col]
        data = cancer_data.dropna()[prot_cols + ["driver_stat"]]

        colmeans_negative = data[data["driver_stat"] == 0].mean().tolist()
        colmeans_positive = data[data["driver_stat"] == 1].mean().tolist()

        to_plot = pd.DataFrame({
            "feature": prot_cols,
            "mean_value_negative": colmeans_negative[:len(colmeans_negative) - 1],
            "mean_value_positive": colmeans_positive[:len(colmeans_positive) - 1]
        })

        to_plot['sort_key'] = to_plot['feature'].str.extract('(\d+)').astype(int)
        to_plot.sort_values('sort_key', inplace=True)

        x = np.linspace(to_plot['sort_key'].min(), to_plot['sort_key'].max(), 300)
        spl_negative = make_interp_spline(to_plot['sort_key'], to_plot['mean_value_negative'], k=3)
        spl_positive = make_interp_spline(to_plot['sort_key'], to_plot['mean_value_positive'], k=3)

        smooth_negative = spl_negative(x)
        smooth_positive = spl_positive(x)

        line_neg, = ax.plot(x, smooth_negative, label=f'{feature.split("_")[1]} Rec', marker='', color=custom_palette[index * 2])
        line_pos, = ax.plot(x, smooth_positive, label=f'{feature.split("_")[1]} Rare', marker='', color=custom_palette[index * 2 + 1])

        lines[f'{feature}_negative'] = smooth_negative
        lines[f'{feature}_positive'] = smooth_positive
        
        legend_labels.extend([line_neg, line_pos])

    ax.fill_between(x, lines[f'{dna_shape_features[0]}_negative'], lines[f'{dna_shape_features[1]}_negative'], color='gray', alpha=0.2)
    ax.fill_between(x, lines[f'{dna_shape_features[0]}_positive'], lines[f'{dna_shape_features[1]}_positive'], color='gray', alpha=0.2)

    ax.set_xlabel('Nucleotide Number', fontsize=16)
    ax.set_xticklabels(list(range(-9, 9, 2)), fontsize = 18)
    ax.set_ylabel('Mean Value', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    # Reorder legend labels
    reordered_labels = [legend_labels[i] for i in [1, 3, 0, 2]]
    print(reordered_labels[0])
    #reordered_labels = ['WT Rare', 'Mut Rare', 'WT Rec', 'Mut Rec']
    ax.legend(handles=reordered_labels, title=None, title_fontsize='13', fontsize='14', loc='best')
    #ax.legend(labs = reordered_labels, title='Group', title_fontsize='13', fontsize='11', loc='best')
    sns.despine()
# %%

def plot_DNA_shapes(cancer_dict_donor_counts, optimised_cancer_results, output_figure_dir):
    # Creates a dictionary of feature datasets for each optimised cancer dataset
    cancer_feature_sets = Counter()
    for cancer in list(cancer_dict_donor_counts.keys()):
        for result in optimised_cancer_results:
            dict_key = cancer
            dict_donor_count = cancer_dict_donor_counts[dict_key]
            if result[0] == dict_key and result[1] == dict_donor_count:
                cancer_feature_sets[dict_key] = result[5]

    # Plot the mean DNA Shape values
    for features_to_plot in [["EP_WT", "EP_Mut"], ["ProT_WT", "ProT_Mut"], ["MGW_WT", "MGW_Mut"], ["HelT_WT", "HelT_Mut"]]:
        if "EP" in features_to_plot[0]:
            name = "EP"
        elif "ProT" in features_to_plot[0]:
            name = "ProT"
        elif "MGW" in features_to_plot[0]:
            name = "MGW"
        elif "HelT" in features_to_plot[0]:
            name = "HelT"
        cancer_types = ["SKCM", "SKCA"]
        num_plots = len(cancer_types)
        grid_rows = math.ceil(num_plots / 2)  # more flexible row calculation
        grid_cols = 2  # or a fixed number of columns

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, grid_rows * 5))
        axes = axes.flatten()

        for i, cancer in enumerate(cancer_types):
            plot_position_means(axes[i], cancer_feature_sets[cancer], features_to_plot)
            axes[i].set_title(f"{cancer} - {features_to_plot[0].split('_')[0]}", fontsize = 18)

        for j in range(i + 1, grid_rows * grid_cols):
            axes[j].set_visible(False)

        plt.tight_layout()
        #plt.subplots_adjust(bottom=0.1)  # Adjust bottom margin
        plt.savefig(f'{output_figure_dir}/dna_shape_{name}.png', dpi=300)
        plt.show()
