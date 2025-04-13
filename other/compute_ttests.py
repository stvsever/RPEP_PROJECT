import pandas as pd
from scipy.stats import ttest_rel
import numpy as np


def run_within_subject_ttests(file_path: str):
    """
    Reads the Excel file with columns:
      ['participant_id', 'experimental_condition', 'proportion_correct',
       'classical_insight_score', 'classical_reaction_time', 'rat_score',
       'rat_reaction_time', 'response_3_avg']

    Performs a paired t-test (within-subject design) on each of the following metrics:
      1. proportion_correct
      2. classical_insight_score
      3. classical_reaction_time
      4. rat_score
      5. rat_reaction_time
      6. response_3_avg

    For each metric, it prints:
      - t-statistic (in decimal format)
      - p-value (in decimal format)
      - Cohen's d (normalized effect size for paired data)

    We assume the experimental_condition has exactly two values (e.g. "DFU" and "F"),
    and that each participant_id appears exactly once per condition.
    """
    # Load data
    df = pd.read_excel(file_path)

    # Verify required columns
    required_cols = [
        'participant_id', 'experimental_condition', 'proportion_correct',
        'classical_insight_score', 'classical_reaction_time', 'rat_score',
        'rat_reaction_time', 'response_3_avg'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in the Excel file: {missing_cols}")

    # List of metrics to test
    metrics = [
        'proportion_correct',
        'classical_insight_score',
        'classical_reaction_time',
        'rat_score',
        'rat_reaction_time',
        'response_3_avg'
    ]

    # Check experimental conditions
    conditions = df['experimental_condition'].unique()
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, found: {conditions}")
    cond1, cond2 = conditions[0], conditions[1]

    print(f"Within-subject T-tests comparing '{cond1}' vs '{cond2}'\n")

    # For each metric, pivot data and perform t-test and compute Cohen's d
    for metric in metrics:
        pivot_df = df.pivot_table(
            index='participant_id',
            columns='experimental_condition',
            values=metric
        )

        # Drop participants with missing data in either condition
        pivot_df = pivot_df.dropna(subset=[cond1, cond2], how='any')
        if pivot_df.empty:
            print(f"Metric '{metric}': No valid data for both conditions.\n")
            continue

        # Paired t-test
        t_stat, p_val = ttest_rel(pivot_df[cond1], pivot_df[cond2])

        # Calculate Cohen's d for paired samples:
        # d = mean(diff) / std(diff)
        differences = pivot_df[cond1] - pivot_df[cond2]
        mean_diff = differences.mean()
        std_diff = differences.std(ddof=1)
        cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan

        # Print results
        print(f"Metric: {metric}")
        print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.8f}")
        print(f"  Cohen's d = {cohens_d:.4f}\n")


if __name__ == "__main__":
    # Run within-subject t-tests
    excel_file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/aggregated_by_participant.xlsx"

    run_within_subject_ttests(excel_file_path)
