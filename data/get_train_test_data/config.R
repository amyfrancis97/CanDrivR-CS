# Configuration for file paths and thresholds

# Base directory where data is downloaded and stored
DATA_DIR <- "/Volumes/Seagate5TB/data/"

# Directory where output files are saved
OUTPUT_DIR <- "/Users/uw20204/Documents/data/TCGA_data/"

# Donor count threshold for filtering mutations
DONOR_COUNT_THRESHOLD <- 3

# Specify the dataset to process (e.g., "LGG")
DATASET <- "LGG"

# Specify which genome build to use
GENOME_BUILD <- "GRCh38"

# Output filenames
BASIC_FILENAME <- paste0(DATA_DIR, DATASET, "_TCGA.txt")
DETAILED_FILENAME <- paste0(DATA_DIR, DATASET, "_TCGA_with_driver_stat.txt")

