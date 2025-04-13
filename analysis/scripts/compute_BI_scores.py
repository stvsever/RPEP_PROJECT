import pandas as pd
import numpy as np


def compute_BI_score_by_group(df, new_col_name, group_cols, w=0.5):
    """
    Computes the BI_score for each row using optional group-based z-score standardization.

    Parameters:
        df: pandas DataFrame with at least the columns 'score_TEST' and 'reaction_time_TEST'.
        new_col_name: str, the name of the new BI_score column.
        group_cols: list of column names to group by for computing z-scores.
                    If None or empty, the computation is done globally.
        w: float (0 <= w <= 1) controlling the weight given to reaction time (default 0.5).

    Returns:
        df: the original DataFrame with the new BI_score column added.
    """
    # Normalization factor to maintain the scale
    norm_factor = np.sqrt((1 - w) ** 2 + w ** 2)

    # Safe function to compute z-scores; avoids division by zero if std is zero.
    safe_z = lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1)

    if group_cols:
        z_score = df.groupby(group_cols)['score_TEST'].transform(safe_z)
        z_reaction = df.groupby(group_cols)['reaction_time_TEST'].transform(safe_z)
    else:
        z_score = safe_z(df['score_TEST'])
        z_reaction = safe_z(df['reaction_time_TEST'])

    # Compute the BI_score for each row:
    # BI_score = [ (1 - w)*z(score_TEST) - w*z(reaction_time_TEST) ] / norm_factor
    df[new_col_name] = ((1 - w) * z_score - w * z_reaction) / norm_factor
    return df


if __name__ == '__main__':
    # Define the CSV file path (update if necessary)
    csv_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
    )

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Check for required columns
    required_cols = [
        'score_TEST',
        'reaction_time_TEST',
        'participant_identification',
        'task_type',
        'task_id'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the input CSV.")

    # Compute the BI_score columns:
    # 1. Regular BI_score (global, no grouping)
    df = compute_BI_score_by_group(df, "BI_score", group_cols=None, w=0.5)
    # 2. BI_score grouped by participant_identification
    df = compute_BI_score_by_group(df, "BI_score_ParticipantID", group_cols=['participant_identification'], w=0.5)
    # 3. BI_score grouped by task_type
    df = compute_BI_score_by_group(df, "BI_score_TaskType", group_cols=['task_type'], w=0.5)
    # 4. BI_score grouped by task_id
    df = compute_BI_score_by_group(df, "BI_score_TaskID", group_cols=['task_id'], w=0.5)
    # 5. BI_score grouped by task_type and participant_identification
    df = compute_BI_score_by_group(df, "BI_score_TaskType_ParticipantID", group_cols=['task_type', 'participant_identification'], w=0.5)

    # Save the DataFrame with the new BI_score columns to a new CSV file
    output_csv_path = csv_path.replace(".csv", "_BI_score.csv")
    df.to_csv(output_csv_path, index=False)

    print(f"BI_scores have been computed using various grouping methods and saved to: {output_csv_path}")
