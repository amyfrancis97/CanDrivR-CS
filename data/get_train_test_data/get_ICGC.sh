#!/bin/bash

# Define a working directory for downloaded and processed files
WORKING_DIR="/Volumes/Seagate5TB/data"
ICGC_URL="https://dcc.icgc.org/api/v1/download?fn=/current/Summary/simple_somatic_mutation.aggregated.vcf.gz"
ICGC_VCF="simple_somatic_mutation.aggregated.vcf.gz"
BED_FILE="simple_somatic_mutation.aggregated.bed"
SNV_FILE="simple_somatic_mutation.aggregated.snv.bed"
AUTOSOME_SNV_FILE="simple_somatic_mutation.aggregated.snv.autosome.bed"
FILTERED_FILE="filtered_somatic_mutation.bed"

echo "This might take some time to run since the ICGC file is very large..."

# Navigate to working directory
cd "$WORKING_DIR"

# Step 1: Download the ICGC VCF file
#echo "Downloading ICGC data..."
#wget -O "$ICGC_VCF" "$ICGC_URL"

# Step 2: Unzip the VCF file
#echo "Unzipping the downloaded VCF..."
#gunzip "$ICGC_VCF"

# Rename the unzipped file to remove query parameters from the filename
#echo "Renaming the unzipped VCF file..."
#mv "${ICGC_VCF%.gz}" "simple_somatic_mutation.aggregated.vcf"

# Step 3: Remove header lines from the VCF and convert it to BED format
#echo "Converting VCF to BED format (removing VCF headers)..."
#grep -v "^#" simple_somatic_mutation.aggregated.vcf > "$BED_FILE"

# Step 4: Filter to only Single Nucleotide Variants (SNVs)
#echo "Filtering to SNVs (Single Nucleotide Variants)..."
#awk -F"\t" '(length($4) == 1 && length($5) == 1)' "$BED_FILE" > "$SNV_FILE"

# Step 5: Remove variants located on sex chromosomes (X, Y)
echo "Filtering out sex chromosomes (X and Y)..."
awk -F"\t" '($1 != "X" && $1 != "Y")' "$SNV_FILE" > "$AUTOSOME_SNV_FILE"

# Step 6: Filter to coding variants only (based on functional impact)
# Note: Adjust column index if the annotation format differs.
echo "Filtering to coding variants..."
awk -F"|" '{
    if ($7 ~ /missense_variant|protein_altering_variant|coding_sequence_variant/)
        print $0
}' "$AUTOSOME_SNV_FILE" > "$FILTERED_FILE"

# Step 7: Notify user and remind to convert coordinates from hg19 to hg38
echo "Processing complete. The final filtered file is: $FILTERED_FILE"
echo "Reminder: This file uses hg19 coordinates. Please convert to hg38 using the provided Python script."

# Convert coordinates from hg19 to hg38 using the Python script
# Pass the path of the filtered file and specify the output path
python3 convert_icgc_coords.py "$WORKING_DIR/$FILTERED_FILE" "$WORKING_DIR/filtered_somatic_mutation_hg38.bed"

# Final message
echo "Coordinate conversion complete. The hg38 converted file is: filtered_somatic_mutation_hg38.bed"

