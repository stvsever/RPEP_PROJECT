import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python conversions
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')

# 1) Read CSV in R
csv_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
ro.r(f"df <- read.csv('{csv_path}', header=TRUE)")
