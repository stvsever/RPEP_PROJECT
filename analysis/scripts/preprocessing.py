import os
import pandas as pd

data_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/raw"
preprocessed_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed"


# Count number of csv files in data directory
def count_files(data_dir):
    count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                count += 1
    return count

print(count_files(data_dir))


# Concatenate all CSV files vertically so that:
# 1) Only one header row appears
# 2) No rows are missed
# 3) Column 'belief_id' is removed
# 4) Column 'critique_id' is renamed to 'condition_item'
def concatenate_files(data_dir, save=False):
    dataframes = []

    # Gather all CSV files into dataframes
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                dataframes.append(df)

    # Concatenate all dataframes into one (with a single header)
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Drop 'belief_id' column if it exists
    if 'belief_id' in concatenated_df.columns:
        concatenated_df.drop(columns=['belief_id'], inplace=True)

    # Drop 'mc_id' column if it exists
    if 'mc_id' in concatenated_df.columns:
        concatenated_df.drop(columns=['mc_id'], inplace=True)

    # Rename 'critique_id' to 'condition_item' if it exists
    if 'critique_id' in concatenated_df.columns:
        concatenated_df.rename(columns={'critique_id': 'condition_item'}, inplace=True)

    # Optionally save to file
    if save:
        output_file = os.path.join(preprocessed_path, "concatenated_data_MCQ.csv")
        concatenated_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    #def count_number_participants(concatenated_df):
    #    return concatenated_df['participant_identification'].nunique()
    #print(count_number_participants(concatenated_df))

concatenate_files(data_dir, save=True)

# TODO: excel file voor analyse moet volgende kolommen hebben: participant_id, block_number, experimental_condition, condition_item, task_type, task_id, score_MCQ, reaction_time_MCQ, score_TEST, reaction_time_TEST, affinity_score, age, gender, education_level, social_media_hours, verbal_intelligence

