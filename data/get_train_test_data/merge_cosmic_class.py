import pandas as pd
import sys
import os

# Ensure correct usage
if len(sys.argv) != 4:
    print("Usage: python merge_cosmic_class.py <cosmic_coding_snv_path> <cosmic_classification_path> <output_path>")
    sys.exit(1)

# Parse command-line arguments
cosmic_coding_snv_path = sys.argv[1]
cosmic_classification_path = sys.argv[2]
output_path = sys.argv[3]

# Load COSMIC coding data
print(f"Loading COSMIC coding SNV data from {cosmic_coding_snv_path}...")
cosmic_coding = pd.read_csv(cosmic_coding_snv_path, sep="\t")

print("COSMIC coding data loaded:")
print(cosmic_coding.head())

# Initialise an empty DataFrame for merged data
df = pd.DataFrame()

# Process the classification data in chunks to manage memory efficiently
print(f"Merging with classification data from {cosmic_classification_path} in chunks...")
for chunk in pd.read_csv(cosmic_classification_path, compression="gzip", sep="\t", chunksize=100000):
    if df.empty:
        df = pd.merge(cosmic_coding, chunk, on="COSMIC_PHENOTYPE_ID", how="left")
    else:
        df = pd.merge(df, chunk, on="COSMIC_PHENOTYPE_ID", how="left")

# Save the merged data to the specified output path
df.to_csv(output_path, sep="\t", index=False)
print(f"Merged data saved to {output_path}.")

