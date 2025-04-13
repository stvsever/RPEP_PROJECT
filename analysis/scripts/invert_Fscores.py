import pandas as pd

column_name_1 = 'affinity_score'
column_name_2 = 'experimental_condition'

file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Invert the scores (only when experimental_condition is 'F')
def invert_Fscores(row):
    if row[column_name_2] == 'F':
        return 8 - row[column_name_1] # --> LikertScore from 1 - 7 ; this inverts the score correctly
    return row[column_name_1]

df[column_name_1] = df.apply(invert_Fscores, axis=1)

# Save back to the same CSV (overwriting the original column)
df.to_csv(file_path, index=False)
