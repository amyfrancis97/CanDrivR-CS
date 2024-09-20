# Load configuration file
source("config.R")

# Load necessary libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("TCGAbiolinks", force = TRUE)

library(TCGAbiolinks)
library(dplyr)
library(readr)
library(stringr)
library(purrr)

# Function to process MAF data
# dataset: TCGA dataset (e.g., "LGG")
# donor_count_threshold: Threshold for filtering mutations based on sample count
# data_dir: Directory for data storage
# output_dir: Directory for saving processed files
process_maf <- function(dataset, donor_count_threshold, data_dir, output_dir, genome_build, basic_filename, detailed_filename) {
  
  # Query GDC for somatic mutation data for the specified TCGA dataset
  query <- GDCquery(project = paste("TCGA", dataset, sep = "-"), 
                    data.category = "Simple Nucleotide Variation",
                    data.type = "Masked Somatic Mutation",
                    workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking")
  
  # Download the queried data
  GDCdownload(query, dir = data_dir)
  
  # Prepare the data for analysis
  data <- GDCprepare(query, dir = data_dir)
  
  # Filter the data to include only GRCh38 genome build
  data <- data[data$NCBI_Build == genome_build, ]
  
  # Filter to include only SNPs (single nucleotide polymorphisms)
  data <- data[data$Variant_Type == "SNP", ]
  
  # Filter to include only missense variants
  coding_data <- subset(data, One_Consequence == "missense_variant")
  
  # Create a list of chromosome names 1-22
  chromosome_list <- paste0("chr", as.character(1:22))
  
  # Filter to keep only variants on chromosomes 1-22
  filtered_data <- coding_data %>%
    filter(Chromosome %in% chromosome_list)
  
  # Create a unique identifier for each mutation
  maf_df <- filtered_data %>%
    mutate(Mutation = paste(Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2, sep = ":")) %>%
    group_by(Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
    summarise(Sample_Count = n_distinct(Tumor_Sample_Barcode), .groups = 'drop')
  
  # Filter mutations based on donor count threshold
  greater_than_threshold <- maf_df[maf_df$Sample_Count > donor_count_threshold, ]
  
  # Filter mutations with sample count equal to 1
  equal_to_1 <- maf_df[maf_df$Sample_Count == 1, ]
  
  # Randomly sample mutations with a sample count of 1 to match those greater than the threshold
  equal_to_1 <- sample_n(equal_to_1, nrow(greater_than_threshold))
  
  # Combine both sets of filtered mutations
  combined_df <- rbind(greater_than_threshold, equal_to_1)
  
  # Save basic mutation data (no sample count)
  write.table(combined_df[c("Chromosome", "Start_Position", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2")],
              basic_filename, sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)
  
  # Save detailed mutation data including donor sample counts
  write.table(combined_df, detailed_filename, sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)
  
  # Return the processed mutation data
  return(combined_df)
}

# Load data and process
df <- process_maf(DATASET, DONOR_COUNT_THRESHOLD, DATA_DIR, OUTPUT_DIR, GENOME_BUILD, BASIC_FILENAME, DETAILED_FILENAME)

# Print the processed mutation data
print(df)

