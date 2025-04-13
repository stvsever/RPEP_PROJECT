import pandas as pd

# Original file path
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Read the dataset
df = pd.read_csv(file_path)

# 1) Create a mapping from UUID (participant_identification) to incremental subject IDs (starting at 1)
unique_ids = sorted(df["participant_identification"].unique())
subject_map = {old_id: i + 1 for i, old_id in enumerate(unique_ids)}

# 2) Apply the mapping to create a new "subject_id" column
df["subject_id"] = df["participant_identification"].map(subject_map)


def aggregate_data(dataframe, task_filter=None):
    """
    If task_filter is 'Insight' or 'RAT', filter on that.
    Otherwise, keep all rows (Both).
    Then group/aggregate by subject_id & experimental_condition.
    The aggregation includes average values for various metrics including the four BI_score columns.
    """
    if task_filter in ['Insight', 'RAT']:
        dataframe = dataframe[dataframe['task_type'] == task_filter]

    # Base aggregation dictionary for standard metrics
    agg_dict = {
        'score_MCQ': 'mean',
        'reaction_time_MCQ': 'mean',
        'score_TEST': 'mean',
        'reaction_time_TEST': 'mean',
        'affinity_score': 'mean',
        'age': 'first',
        'gender': 'first',
        'education_level (third grade)': 'first',
        'social_media_hours (daily)': 'first',
        'verbal_intelligence (self-perceived)': 'first'
    }

    # Add the four BI_score columns if they exist in the dataframe
    bi_scores = ["BI_score", "BI_score_TaskType", "BI_score_TaskID", "BI_score_ParticipantID", "BI_score_TaskType_ParticipantID"]
    for col in bi_scores:
        if col in dataframe.columns:
            agg_dict[col] = 'mean'

    agg_df = (
        dataframe
        .groupby(['subject_id', 'experimental_condition'], as_index=False)
        .agg(agg_dict)
    )

    # Rename columns to the desired final column names
    agg_df.rename(columns={
        'education_level (third grade)': 'education_level',
        'social_media_hours (daily)': 'social_media_usage',
        'verbal_intelligence (self-perceived)': 'verbal_intelligence'
    }, inplace=True)

    # Define final order of columns
    final_cols = [
        'subject_id',
        'experimental_condition',
        'age',
        'gender',
        'education_level',
        'social_media_usage',
        'verbal_intelligence',
        'score_MCQ',
        'reaction_time_MCQ',
        'score_TEST',
        'reaction_time_TEST',
        'affinity_score'
    ]

    # Append BI_score columns if they exist in the aggregated dataframe
    for col in bi_scores:
        if col in agg_df.columns:
            final_cols.append(col)

    agg_df = agg_df[final_cols]
    return agg_df


# Create separate dataframes for aggregated data
df_both = aggregate_data(df, task_filter=None)  # Aggregated for both task types
df_insight = aggregate_data(df, task_filter='Insight')  # Aggregated for Insight only
df_rat = aggregate_data(df, task_filter='RAT')  # Aggregated for RAT only

# Write each to a new Excel file
output_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed"

df_both.to_excel(f"{output_dir}/aggregated_by_subjectID_BOTH.xlsx", index=False)
df_insight.to_excel(f"{output_dir}/aggregated_by_subjectID_INSIGHT.xlsx", index=False)
df_rat.to_excel(f"{output_dir}/aggregated_by_subjectID_RAT.xlsx", index=False)
