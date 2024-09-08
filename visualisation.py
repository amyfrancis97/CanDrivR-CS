# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from core_modules import save_data

def plot_metrics(metrics_data, save_path):
    """ Plot a barplot for evaluation metrics. """
    sns.barplot(data=metrics_data, x='cancer', y='f1')
    plt.xticks(rotation=90)
    plt.title('F1 Scores for Different Cancers')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

