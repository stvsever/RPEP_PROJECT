import pandas as pd

file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Filter for only RAT task_type
df_rat = df[df['task_type'] == 'RAT']

# Calculate the proportion of correct answers per participant_identification (score_TEST==1)
proportion_correct = df_rat.groupby('participant_identification')['score_TEST'].apply(lambda x: (x == 1).mean())

print(proportion_correct)

# Print descriptive statistics
print(proportion_correct.describe())

# Print average proportion of correct answers
print(proportion_correct.mean())