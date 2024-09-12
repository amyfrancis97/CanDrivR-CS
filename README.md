# CanDrivR-CS: A Cancer-Specific Machine Learning Framework for Distinguishing Recurrent and Rare Variants

## Introduction and Overview
Welcome to CanDrivR-CS! This repository contains the scripts used for modeling in our recent CanDrivR-CS paper [](). The project includes functionality for managing ICGC, COSMIC, and TCGA datasets, applying machine learning models, and visualising the results.

## Directory Structure
```bash
.
├── README.md                      # Project overview and usage instructions
├── config.py                      # Configuration settings for the pipeline
├── core_modules.py                # Core utility functions used throughout the pipeline
├── data/                          # Folder containing raw and processed data
│   ├── COSMIC_rare.tsv.gz         # COSMIC rare dataset (gzipped)
│   ├── COSMIC_recurrent.tsv.gz    # COSMIC recurrent dataset (gzipped)
│   ├── ICGC.tsv.gz                # ICGC dataset (gzipped)
│   ├── TCGA_SKCM_test.txt.gz      # TCGA SKCM test dataset
│   ├── TCGA_UCEC_test.txt.gz      # TCGA UCEC test dataset
│   ├── cancer_datasets.pkl        # Cancer-specific dataset pickle file
│   └── get_train_test_data/       # Scripts for fetching and processing raw data
│       ├── MANIFEST.txt
│       ├── config.R
│       ├── convert_icgc_coords.py # Script for converting ICGC coordinates
│       ├── get_ICGC.sh            # Shell script for downloading ICGC data
│       ├── get_TCGA.R             # R script for fetching TCGA data
│       ├── get_COSMIC.sh          # Shell script for downloading COSMIC data
│       └── merge_cosmic_class.py  # Python script for merging cancer types onto COSMIC data
├── data_processing.py             # Data pre-processing logic
├── main.py                        # Main entry point for the pipeline
├── model_training.py              # Script for training and evaluating machine learning models
├── models/                        # Directory containing model-related scripts
│   ├── metric_results_table.py    # Script for generating result metrics
│   ├── prepare_training_data.py   # Script for preparing the training data
│   ├── run_models.py              # Script for running the models
│   ├── train_classifier.py        # Core model training logic
│   └── train_evaluate.py          # Script for training and evaluating the model
├── optimisation/                  # Directory for optimisation scripts
│   ├── optimise_donor_counts.py   # Script for optimising donor counts
│   ├── params.py                  # Parameters for the pipeline
│   └── selected_features.py       # Feature selection logic
│
├── plots/                         # Directory for plotting scripts
│   ├── donor_count_vs_accuracy.py # Script for plotting donor counts vs accuracy
│   ├── heatmap_feature_importance.py # Script for plotting heatmaps of feature importance
│   └── plot_dna_shape_position_means.py # DNA shape position plotting logic
└── visualisation.py               # Other core visualisation functions

```
## Setup

1. Clone the repository:

```bash
git clone git@github.com:amyfrancis97/CanDrivR-CS.git
or
git clone https://github.com/amyfrancis97/CanDrivR-CS.git

cd CanDrivR-CS
```

2. Install dependencies:

```bash
conda env create --name envname --file=CanDrivR.yml 
conda activate envname
```

3. Configure environment: Modify config.py to set up paths, parameters, and environment-specific settings for running the pipeline.

## Usage

1. Running the main script:

```bash
python main.py
```

The script will train and evaluate the model on the datasets specified in config.py and produce output results in the outputs folder. The default output will include evaluation metrics, plots, and results for cancer-specific models.

## Contributing
Contributions are welcome! Follow the guidelines in [CONTRIBUTING.md](https://github.com/amyfrancis97/CanDrivR-CS/blob/main/CONTRIBUTING.md).

## License
This project is licensed under [LICENSE](https://github.com/amyfrancis97/CanDrivR-CS/blob/main/LICENSE).

## Acknowledgments
This work was carried out by Amy Francis in the UK Medical Research Council Integrative Epidemiology Unit (MC\_UU\_00032/03) and using the computational facilities of the Advanced Computing Research Centre, University of Bristol.

Special thanks to:
* Tom Gaunt, Bristol Medical School (PHS), University of Bristol
* Colin Campbell, Intelligent Systems Laboratory, University of Bristol

## Funding
This work was funded by Cancer Research UK [C18281/A30905]. 

## Contact
For enquiries, contact us at [amy.francis@bristol.ac.uk](mailto:amy.francis@bristol.ac.uk).





