# Import necessary libraries
import pandas as pd
import numpy as np
from pyliftover import LiftOver
import argparse

def get_icgc_GRCh38(file_path):
    """
    Processes the ICGC somatic mutation file and converts coordinates from hg19 to hg38 using LiftOver.
    
    Parameters:
    file_path (str): The file path of the input somatic mutation file (hg19 coordinates).

    Returns:
    pd.DataFrame: A DataFrame with updated chromosome and position (hg38) along with relevant mutation information.
    """
    # Load the data
    df = pd.read_csv(file_path, sep="\t", header=None)
    print(f"Original dataset length: {len(df)}")

    # Filter out non-standard chromosomes X, Y, and MT
    icgc_donor_count = df[(df[0] != "X") & (df[0] != "Y") & (df[0] != "MT")]

    # Prefix chromosome numbers with "chr" to match LiftOver format
    icgc_donor_count[0] = "chr" + icgc_donor_count[0].astype(str)

    # Initialise LiftOver object for converting from hg19 to hg38
    lo = LiftOver('hg19', 'hg38')

    # Perform coordinate conversion for all rows
    results = [lo.convert_coordinate(icgc_donor_count[0][i], icgc_donor_count[1][i]) for i in range(len(icgc_donor_count))]

    # Extract new chromosome positions from LiftOver results
    chromosomes = []
    positions = []
    for res in results:
        try:
            chromosomes.append(res[0][0])  # Get the new chromosome
            positions.append(res[0][1])    # Get the new position
        except IndexError:
            chromosomes.append(np.nan)     # Handle missing conversions
            positions.append(np.nan)

    # Add the new chromosome and position to the DataFrame
    icgc_donor_count["new_chroms"] = chromosomes
    icgc_donor_count["new_positions"] = positions
    icgc_donor_count["new_positions"] = icgc_donor_count["new_positions"].astype('Int64')

    # Create a unique identifier based on the updated chromosome, position, reference and alternate alleles
    icgc_donor_count["id"] = (icgc_donor_count["new_chroms"].astype(str) + "_" + 
                              icgc_donor_count["new_positions"].astype(str) + "_" + 
                              icgc_donor_count[3].astype(str) + "_" + 
                              icgc_donor_count[4].astype(str))

    # Extract donor count from the 7th column
    icgc_donor_count["donor_count"] = icgc_donor_count[7].str.split("affected_donors=", expand=True)[1].str.split(";", expand=True)[0]

    # Extract the study ID from the 7th column (e.g., OCCURRENCE=BOCA-UK|)
    studies = icgc_donor_count[7].str.split("OCCURRENCE=", expand=True)[1].str.split("|", expand=True)[0].str.split("-", expand=True)[0]
    icgc_donor_count["studies"] = studies

    # Remove duplicate mutations by keeping the first instance for each mutation and study combination
    icgc_donor_count = icgc_donor_count.drop_duplicates(subset=["id", "studies"], keep="first")

    # Create the final cleaned DataFrame with relevant columns
    icgc_donor_count_cleaned = icgc_donor_count[["new_chroms", "new_positions", 3, 4, "studies", "donor_count"]]
    icgc_donor_count_cleaned.columns = ["chrom", "pos", "ref_allele", "alt_allele", "studyID", "donor_count"]

    return icgc_donor_count_cleaned

def main(input_file, output_file):
    """
    Main function to process the input file, convert the coordinates, and save the processed DataFrame to the specified output path.
    
    Parameters:
    input_file (str): Path to the input ICGC somatic mutation file.
    output_file (str): Path to save the processed file with hg38 coordinates.
    """
    # Process the input file and get the updated DataFrame
    processed_df = get_icgc_GRCh38(input_file)

    # Save the processed data to the specified output path
    processed_df.to_csv(output_file, sep="\t", index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    # Define argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Convert ICGC somatic mutation coordinates from hg19 to hg38.")
    parser.add_argument("input_file", help="Path to the input ICGC somatic mutation file.")
    parser.add_argument("output_file", help="Path to save the output file with updated hg38 coordinates.")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.input_file, args.output_file)

