#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_curve_with_dual_y_axis(data, axs):
    """
    Plot curve with dual y-axis.

    Parameters:
        data (pd.DataFrame): DataFrame containing donor count, F1 score, and size of the dataset.
        axs (matplotlib.axes.Axes): Axes object to plot on.
    """
    # Sort dataframe based on donor count
    data["f1"] = data["f1"].str.split(" ± ", expand=True)[0].astype("float")

    # Convert metrics to floats
    data["f1"] = data["f1"].astype("float")
    data["size"] = data["size"].astype("float")

    # Extracting  data
    x = data["donor_count"].values
    y1 = data["f1"].values
    y2 = data["size"].values

    # Define custom palette
    custom_palette = sns.color_palette("Dark2", 2)

    # Plot F1 score on the left y-axis
    smoothed_f1 = lowess(y1, x, frac=0.2)
    axs.plot(smoothed_f1[:, 0], smoothed_f1[:, 1], label="F1 Score", color=custom_palette[0], linewidth=2)
    axs.set_xlabel("Donor Count Threshold for Positives", fontsize=14)
    axs.set_xticks(data["donor_count"])  # Set the ticks to match donor counts exactly
    axs.set_xticklabels(">" + data["donor_count"].astype(str))  # Apply your custom labels
    axs.set_ylabel("F1 Score", fontsize=14)

    # Create a twin Axes sharing the xaxis for plotting dataset size on the right y-axis
    ax2 = axs.twinx()
    smoothed_size = lowess(y2, x, frac=0.2)
    ax2.plot(smoothed_size[:, 0], smoothed_size[:, 1], label="Dataset Size", color=custom_palette[1], linewidth=2, linestyle = "--")
    ax2.set_ylabel("Dataset Size", fontsize=14)

    # Combine legends
    lines1, labels1 = axs.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=12)

    # Further enhance the title, labels, and ticks for clarity
    axs.set_title("F1 Score and Dataset Size vs Donor Count", fontsize=16)
    axs.tick_params(axis="both", which="major", labelsize=12)
    ax2.tick_params(axis="y", which="major", labelsize=12)

def plot_grid_of_curves(datasets, outputDir, fileNames):
    """
    Plot a grid of curves with dual y-axis.

    Parameters:
        datasets (list of pd.DataFrame): List of DataFrames containing donor count, F1 score, and size of the dataset.
        outputDir (str): Directory to save the plots.
        fileNames (list of str): List of file names for the plots.
    """
    # Calculate the grid dimensions
    n_plots = len(datasets)
    n_cols = 2
    n_rows = -(-n_plots // n_cols)  # Round up division

    # Break into three grids
    n_plots_per_grid = -(-n_plots // 4)

    for g in range(4):
        start = g * n_plots_per_grid
        end = min((g + 1) * n_plots_per_grid, n_plots)
        
        # Create the grid
        fig, axs = plt.subplots(8, n_cols, figsize=(15, 20))
        axs = axs.ravel()

        for i, (data, fileName) in enumerate(zip(datasets[start:end], fileNames[start:end])):
            plot_curve_with_dual_y_axis(data, axs[i])
            axs[i].set_title(fileName)

        # Hide any empty subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.savefig(f"{outputDir}/cancer_spec_donor_plots_{g+1}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

# %%
