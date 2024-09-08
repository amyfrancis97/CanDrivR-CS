
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
import pandas as pd

def plot_boxplot(df, column_name):
    """
    Plots a boxplot for a specified column in the DataFrame, grouped by 'driver_stat'.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.

    """
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Initialize the matplotlib figure
        plt.figure(figsize=(10, 6))

        # Create a boxplot
        sns.boxplot(x='driver_stat', y=column_name, data=df)

        # Set title and labels
        plt.title(f'Boxplot of {column_name} by Driver Stat')
        plt.xlabel('Driver Stat')
        plt.ylabel(column_name)

        # Show the plot
        plt.show()
    else:
        print(f"The column '{column_name}' does not exist in the DataFrame.")

def plot_multi_boxplot(df, column_names, width=0.5):
    """
    Plots boxplots for specified columns in the DataFrame, grouped by 'driver_stat', all on one plot.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_names (list): The list of column names to plot.
        width (float): The width of the boxes in the boxplot.
    """
    # Filter to ensure all column names exist in the DataFrame
    column_names = [col for col in column_names if col in df.columns]
    if not column_names:
        print("None of the specified columns exist in the DataFrame.")
        return

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 6))

    # Melt the DataFrame to long format for easier plotting with seaborn
    df_long = df.melt(id_vars=['driver_stat'], value_vars=column_names, 
                      var_name='Feature', value_name='Value')

    # Create a boxplot
    sns.boxplot(x='Feature', y='Value', hue='driver_stat', data=df_long, width=width)

    # Set title and labels
    plt.title('Distribution of Features by Driver Stat')
    plt.xlabel('Feature')
    plt.ylabel('Value')

    # Adjust the legend
    plt.legend(title='Driver Stat')

    # Enhance style for publication
    sns.set_style("whitegrid")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)  # Rotate the x labels for better readability

    # Show the plot
    plt.show()
