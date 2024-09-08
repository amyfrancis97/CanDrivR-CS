import pandas as pd

def get_results_table(results, model_name):
    """
    Get a results table for means of cross validation runs.

    Parameters:
        results (pd.DataFrame): The results table for each run of cross validation
        model name (str): Name of model used, for labeling column.
    Returns:
        pd.DataFrame: A dataframe containing the mean ± standard deviation for each of the metrics captured in the model.
    """

    # Calculate the means of the metrics
    means = pd.DataFrame(results.mean()).transpose()
    means = [format(means.loc[0, :][i], ".3f") for i in range(0, len(list(means)))]

    # Calculate the stds of the metrics
    stds = pd.DataFrame(results.std()).transpose()
    stds = [format(stds.loc[0, :][i], ".3f") for i in range(0, len(list(stds)))]

    # Combine means and standard deviations in a single row, separated by ±
    mean_df = pd.DataFrame([means[i] + " ± " + stds[i] for i in range(0, len(means))]).transpose()
    mean_df.columns = results.columns.tolist()
    mean_df["model"] = model_name
    return mean_df
