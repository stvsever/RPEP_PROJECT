import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

def analyze_scores_and_reaction_times(file_path):
    """
    1. Reads the CSV file and cleans key columns.
    2. Aggregates data per participant:
       - score_DFU: Mean of 'response_3' for condition 'DFU'.
       - score_F:   Mean of 'response_3' for condition 'F'.
       - inverted_F_score: An inverted scale of score_F (e.g. 8 - score_F).
       - affinity_index: Average of score_DFU and score_F.
       - overall_rt: Mean of 'reaction_time_2' across ALL rows (irrespective of condition).
    3. Computes Pearson correlations of (score_DFU, score_F, affinity_index) each with overall_rt.
    4. Saves the aggregated data to CSV (with DFU score, F score, inverted F score, affinity index, overall_rt).
    5. Plots all three (DFU, F, and affinity_index) vs overall_rt in a single figure (3 subplots).
    """

    # Read CSV file
    df = pd.read_csv(file_path)

    # Clean key columns: trim and standardize
    df['participant_identification'] = df['participant_identification'].astype(str).str.strip()
    df['experimental_condition'] = df['experimental_condition'].astype(str).str.strip().str.upper()

    # Convert to numeric (coerce invalid entries to NaN)
    df['response_3'] = pd.to_numeric(df['response_3'], errors='coerce')
    df['reaction_time_2'] = pd.to_numeric(df['reaction_time_2'], errors='coerce')

    # --- 1) Compute DFU Score (score_DFU) ---
    # Filter only rows with DFU condition, group by participant, average 'response_3'
    df_DFU = df[df['experimental_condition'] == 'DFU']
    grouped_scores_DFU = df_DFU.groupby('participant_identification', as_index=False)['response_3'].mean()
    grouped_scores_DFU.rename(columns={'response_3': 'score_DFU'}, inplace=True)

    # --- 2) Compute F Score (score_F) ---
    # Filter only rows with F condition, group by participant, average 'response_3'
    df_F = df[df['experimental_condition'] == 'F']
    grouped_scores_F = df_F.groupby('participant_identification', as_index=False)['response_3'].mean()
    grouped_scores_F.rename(columns={'response_3': 'score_F'}, inplace=True)

    # --- 3) Compute Overall Reaction Time (overall_rt) ---
    # For ALL rows, group by participant, average 'reaction_time_2'
    grouped_rt = df.groupby('participant_identification', as_index=False)['reaction_time_2'].mean()
    grouped_rt.rename(columns={'reaction_time_2': 'overall_rt'}, inplace=True)

    # --- Merge all partial dataframes ---
    merged_df = pd.merge(grouped_scores_DFU, grouped_scores_F, on='participant_identification', how='inner')
    merged_df = pd.merge(merged_df, grouped_rt, on='participant_identification', how='inner')

    # --- 4) Compute inverted F score ---
    # (Assuming a 1-7 scale, for example 8 - score_F)
    merged_df['inverted_F_score'] = 8 - merged_df['score_F']

    # --- 5) Compute Affinity Index ---
    # Average of DFU score and (original) F score
    merged_df['affinity_index'] = (merged_df['score_DFU'] + merged_df['score_F']) / 2.0

    # --- Save aggregated data to CSV ---
    out_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed"
    out_file = os.path.join(out_dir, "aggregated_data.csv")
    merged_df.to_csv(out_file, index=False)
    print(f"Aggregated data saved to: {out_file}")

    # --- Check sample size for correlation ---
    #if len(merged_df) < 2:
    #    print("Warning: Not enough participants to compute correlations.")
    #    return

    # --- 6) Compute Pearson correlations ---
    # We'll compute correlation for DFU, F, and Affinity Index each with overall_rt
    score_vars = ['score_DFU', 'score_F', 'affinity_index']
    for var in score_vars:
        corr, p_value = pearsonr(merged_df[var], merged_df['overall_rt'])
        print(f"=== Correlation ({var} vs Overall RT) ===")
        print(f"Pearson r = {corr:.3f}, p-value = {p_value:.4f}\n")

    # --- 7) Scatterplots with regression lines (3 subplots in one figure) ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Colors to distinguish each plot
    colors = ['blue', 'green', 'red']
    titles = ["DFU Score vs Overall RT",
              "(Inverted) F Score vs Overall RT",
              "Affinity Index vs Overall RT"]

    for i, var in enumerate(score_vars):
        sns.regplot(ax=axes[i],
                    x=var,
                    y='overall_rt',
                    data=merged_df,
                    scatter_kws={'alpha': 0.7, 'color': colors[i]},
                    line_kws={'color': colors[i]})
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Overall Reaction Time (ms)")

    plt.tight_layout()
    plt.show()

    # Compute correlation between DFU and inverted F score
    def compute_correlation_DFU_inverted_F(merged_df):
        """
        Computes the Pearson correlation between DFU score and inverted F score.
        """
        corr, p_value = pearsonr(merged_df['score_DFU'], merged_df['inverted_F_score'])
        print(f"=== Correlation (DFU Score vs Inverted F Score) ===")
        print(f"Pearson r = {corr:.3f}, p-value = {p_value:.4f}\n")
        return corr, p_value

    compute_correlation_DFU_inverted_F(merged_df)

if __name__ == "__main__":
    file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
    analyze_scores_and_reaction_times(file_path)
