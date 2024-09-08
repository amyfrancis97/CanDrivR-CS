#!/bin/bash
# --------------------------------------------------------------
# COSMIC Data Processing Script
# --------------------------------------------------------------
# This script filters COSMIC variant data, keeping only coding and SNV variants.
# It then merges this data with classification data.
#
# Files needed for this process:
# - Cosmic_GenomeScreensMutant_v99_GRCh38.tsv.gz
# - Cosmic_Classification_v99_GRCh38.tsv.gz
#
# You can download the files from the COSMIC website:
# https://cancer.sanger.ac.uk/cosmic/download/cosmic
#
# After downloading the files, place them in the same directory where this script is executed.
# --------------------------------------------------------------

# Paths to the COSMIC data files
COSMIC_GENOME="Cosmic_GenomeScreensMutant_v99_GRCh38.tsv.gz"
COSMIC_CLASSIFICATION="Cosmic_Classification_v99_GRCh38.tsv.gz"
OUTPUT_DIR="/user/home/uw20204/CanDrivR_data/cosmic_tissue_spec"

# Ensure we are in the correct directory
cd $OUTPUT_DIR

# 1. Filter for coding variants
echo "Filtering for coding variants..."
zcat $COSMIC_GENOME | awk -F"\t" '{
    if ($12 ~ /missense_variant|protein_altering_variant/) 
        print $0 
}' > cosmic_coding.tsv

# 2. Filter for SNVs only (Single Nucleotide Variants)
echo "Filtering for SNVs..."
awk -F"\t" '{if(length($24) == 1 && length($25) == 1) print $0}' cosmic_coding.tsv > cosmic_coding_snv.tsv

# 3. Add header back to the filtered file
echo "Adding header to the filtered SNV data..."
cat header.tsv cosmic_coding_snv.tsv > cosmic_coding_snv.tmp && mv cosmic_coding_snv.tmp cosmic_coding_snv.tsv

# 4. Clean up intermediate files
echo "Tidying up temporary files..."
rm cosmic_coding.tsv cosmic_coding_snv.tmp

# 5. Merge with COSMIC classification using a Python script and pass file paths as arguments
echo "Merging with COSMIC classification data..."
python merge_cosmic_class.py "$OUTPUT_DIR/cosmic_coding_snv.tsv" "$OUTPUT_DIR/$COSMIC_CLASSIFICATION" "$OUTPUT_DIR/cosmic_with_classification.tsv"

echo "Process completed!"

