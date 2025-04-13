import pandas as pd

# File path to your CSV
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# 1. Load the CSV file
df = pd.read_csv(file_path)

# 2. Calculate the proportion of 'correct' responses per participant
df['score_MCQ'] = df.groupby('participant_identification')['response_1'] \
                    .transform(lambda x: (x == 'correct').mean())

# 3. Save back to the same CSV (adding the new column)
df.to_csv(file_path, index=False)
