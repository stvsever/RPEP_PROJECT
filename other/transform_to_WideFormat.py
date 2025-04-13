import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python conversions
pandas2ri.activate()
numpy2ri.activate()

# Import R packages
tidyr = importr('tidyr')
dplyr = importr('dplyr')

# Specify the file path to your CSV
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Read the CSV file in R (set stringsAsFactors = FALSE for easier handling)
ro.r(f'data <- read.csv("{file_path}", stringsAsFactors = FALSE)')

# Transform from long to wide format with summarization to avoid list-cols.
ro.r('''
library(tidyr)
library(dplyr)
data_wide <- pivot_wider(data, 
                         id_cols = participant_identification, 
                         names_from = experimental_condition, 
                         values_from = c(score_MCQ, reaction_time_MCQ, score_TEST, reaction_time_TEST, affinity_score),
                         values_fn = list(score_MCQ = first, 
                                          reaction_time_MCQ = first, 
                                          score_TEST = first, 
                                          reaction_time_TEST = first, 
                                          affinity_score = first))
''')

# Convert the resulting R data frame back into a pandas DataFrame
data_wide = pandas2ri.rpy2py(ro.r('data_wide'))

# Display the first few rows to verify the wide format
print(data_wide.head())

# Save the wide format to a new CSV file
output_file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data_wide.csv"
data_wide.to_csv(output_file_path, index=False)
print(f"Wide format data saved to {output_file_path}")

# TODO: not correct yet