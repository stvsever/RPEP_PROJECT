import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.cm as cm

# ---------------------------------------------------------
# Step 1: Load Data and Apply Exclusion Logic
# ---------------------------------------------------------
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
df = pd.read_csv(file_path)

# Exclusion logic:
# For each participant and block, number the trials and compute the cumulative count of correct MCQ responses.
# Exclude trials until the first correct response is given and keep only trials with trial_index 1 to 4.
df['trial_index'] = df.groupby(['participant_identification', 'block_number']).cumcount() + 1
df['cum_correct'] = df.groupby(['participant_identification', 'block_number'])['MCQ_response_unprocessed'].transform(lambda x: (x=="correct").cumsum())
df_filtered = df[(df['cum_correct'] > 0) & (df['trial_index'] <= 4)]

# Use the filtered dataset for all further analysis.
df = df_filtered.copy()

base_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/priming_effects/participant_id"

# ---------------------------------------------------------
# Helper Function: Get significance code from p-value
# ---------------------------------------------------------
def significance_code(p):
    """
    Returns significance code for a given p-value based on:
      p <= 0.001: '***'
      p <= 0.01 : '**'
      p <= 0.05 : '*'
      p <= 0.1  : '.'
      Otherwise:  ' '
    """
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    elif p <= 0.1:
        return '.'
    else:
        return ' '

# =============================================================================
#          ANALYSIS ACROSS PARTICIPANTS: BI_score
# =============================================================================
# For DFU: Priming effect = (participant mean in DFU) - (global mean in F)
df_DFU_BI = df[df['experimental_condition'] == 'DFU']
df_F_BI   = df[df['experimental_condition'] == 'F']

participant_means_DFU_BI = df_DFU_BI.groupby('participant_identification')['BI_score'].mean()
global_mean_F_BI = df_F_BI['BI_score'].mean()
priming_effect_DFU_BI = participant_means_DFU_BI - global_mean_F_BI

overall_effect_DFU_BI = participant_means_DFU_BI.mean() - global_mean_F_BI
t_stat, p_val = ttest_1samp(participant_means_DFU_BI, global_mean_F_BI)
overall_annotation_DFU_BI = f"Overall: {overall_effect_DFU_BI:.2f}, p = {p_val:.4f}"
print(f"\nBI_score DFU: Overall priming effect = {overall_effect_DFU_BI:.2f}, p = {p_val:.4f}")

# For F: Priming effect = (participant mean in F) - (global mean in DFU)
participant_means_F_BI = df_F_BI.groupby('participant_identification')['BI_score'].mean()
global_mean_DFU_BI = df_DFU_BI['BI_score'].mean()
priming_effect_F_BI = participant_means_F_BI - global_mean_DFU_BI

overall_effect_F_BI = participant_means_F_BI.mean() - global_mean_DFU_BI
t_stat_f, p_val_f = ttest_1samp(participant_means_F_BI, global_mean_DFU_BI)
overall_annotation_F_BI = f"Overall: {overall_effect_F_BI:.2f}, p = {p_val_f:.4f}"
print(f"\nBI_score F: Overall priming effect = {overall_effect_F_BI:.2f}, p = {p_val_f:.4f}")

# Create sequential x-values for participants
x_vals_DFU = np.arange(1, len(participant_means_DFU_BI) + 1)
x_vals_F   = np.arange(1, len(participant_means_F_BI) + 1)

# Plot DFU BI_score priming effects (using Blues colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_DFU_BI)
colors = [cm.Blues(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_DFU, priming_effect_DFU_BI.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (DFU): DFU mean - Global F mean", fontsize=12)
plt.title('Priming Effect Across Participants (BI_score, DFU)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_DFU_BI, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_DFU_BI, overall_annotation_DFU_BI,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_DFU_BI_score.png"
plt.savefig(output_file, dpi=300)
plt.show()

# Plot F BI_score priming effects (using Reds colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_F_BI)
colors = [cm.Blues(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_F, priming_effect_F_BI.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (F): F mean - Global DFU mean", fontsize=12)
plt.title('Priming Effect Across Participants (BI_score, F)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_F_BI, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_F_BI, overall_annotation_F_BI,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_F_BI_score.png"
plt.savefig(output_file, dpi=300)
plt.show()

# =============================================================================
#         ANALYSIS ACROSS PARTICIPANTS: score_TEST
# =============================================================================
# For DFU: Priming effect = (participant mean in DFU) - (global mean in F)
df_DFU_test = df[df['experimental_condition'] == 'DFU']
df_F_test   = df[df['experimental_condition'] == 'F']

participant_means_DFU_test = df_DFU_test.groupby('participant_identification')['score_TEST'].mean()
global_mean_F_test = df_F_test['score_TEST'].mean()
priming_effect_DFU_test = participant_means_DFU_test - global_mean_F_test

overall_effect_DFU_test = participant_means_DFU_test.mean() - global_mean_F_test
t_stat, p_val = ttest_1samp(participant_means_DFU_test, global_mean_F_test)
overall_annotation_DFU_test = f"Overall: {overall_effect_DFU_test:.2f}, p = {p_val:.4f}"
print(f"\nscore_TEST DFU: Overall priming effect = {overall_effect_DFU_test:.2f}, p = {p_val:.4f}")

# For F: Priming effect = (participant mean in F) - (global mean in DFU)
participant_means_F_test = df_F_test.groupby('participant_identification')['score_TEST'].mean()
global_mean_DFU_test = df_DFU_test['score_TEST'].mean()
priming_effect_F_test = participant_means_F_test - global_mean_DFU_test

overall_effect_F_test = participant_means_F_test.mean() - global_mean_DFU_test
t_stat_f, p_val_f = ttest_1samp(participant_means_F_test, global_mean_DFU_test)
overall_annotation_F_test = f"Overall: {overall_effect_F_test:.2f}, p = {p_val_f:.4f}"
print(f"\nscore_TEST F: Overall priming effect = {overall_effect_F_test:.2f}, p = {p_val_f:.4f}")

x_vals_DFU_test = np.arange(1, len(participant_means_DFU_test) + 1)
x_vals_F_test   = np.arange(1, len(participant_means_F_test) + 1)

# Plot DFU score_TEST priming effects (using Greens colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_DFU_test)
colors = [cm.Greens(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_DFU_test, priming_effect_DFU_test.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (DFU): DFU mean - Global F mean", fontsize=12)
plt.title('Priming Effect Across Participants (score_TEST, DFU)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_DFU_test, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_DFU_test, overall_annotation_DFU_test,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_DFU_score_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# Plot F score_TEST priming effects (using Purples colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_F_test)
colors = [cm.Greens(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_F_test, priming_effect_F_test.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (F): F mean - Global DFU mean", fontsize=12)
plt.title('Priming Effect Across Participants (score_TEST, F)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_F_test, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_F_test, overall_annotation_F_test,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_F_score_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# =============================================================================
#       ANALYSIS ACROSS PARTICIPANTS: reaction_time_TEST
# =============================================================================
# For DFU: Priming effect = (participant mean in DFU) - (global mean in F)
df_DFU_rt = df[df['experimental_condition'] == 'DFU']
df_F_rt   = df[df['experimental_condition'] == 'F']

participant_means_DFU_rt = df_DFU_rt.groupby('participant_identification')['reaction_time_TEST'].mean()
global_mean_F_rt = df_F_rt['reaction_time_TEST'].mean()
priming_effect_DFU_rt = participant_means_DFU_rt - global_mean_F_rt

overall_effect_DFU_rt = participant_means_DFU_rt.mean() - global_mean_F_rt
t_stat_rt, p_val_rt = ttest_1samp(participant_means_DFU_rt, global_mean_F_rt)
overall_annotation_DFU_rt = f"Overall: {overall_effect_DFU_rt:.2f}, p = {p_val_rt:.4f}"
print(f"\nreaction_time_TEST DFU: Overall priming effect = {overall_effect_DFU_rt:.2f}, p = {p_val_rt:.4f}")

# For F: Priming effect = (participant mean in F) - (global mean in DFU)
participant_means_F_rt = df_F_rt.groupby('participant_identification')['reaction_time_TEST'].mean()
global_mean_DFU_rt = df_DFU_rt['reaction_time_TEST'].mean()
priming_effect_F_rt = participant_means_F_rt - global_mean_DFU_rt

overall_effect_F_rt = participant_means_F_rt.mean() - global_mean_DFU_rt
t_stat_rt_f, p_val_rt_f = ttest_1samp(participant_means_F_rt, global_mean_DFU_rt)
overall_annotation_F_rt = f"Overall: {overall_effect_F_rt:.2f}, p = {p_val_rt_f:.4f}"
print(f"\nreaction_time_TEST F: Overall priming effect = {overall_effect_F_rt:.2f}, p = {p_val_rt_f:.4f}")

x_vals_DFU_rt = np.arange(1, len(participant_means_DFU_rt) + 1)
x_vals_F_rt   = np.arange(1, len(participant_means_F_rt) + 1)

# Plot DFU reaction_time_TEST priming effects (using Oranges colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_DFU_rt)
colors = [cm.Oranges(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_DFU_rt, priming_effect_DFU_rt.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (DFU): DFU mean - Global F mean", fontsize=12)
plt.title('Priming Effect Across Participants (reaction_time_TEST, DFU)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_DFU_rt, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_DFU_rt, overall_annotation_DFU_rt,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_DFU_reaction_time_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# Plot F reaction_time_TEST priming effects (using Copper colormap)
plt.figure(figsize=(10, 6))
n = len(priming_effect_F_rt)
colors = [cm.Oranges(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(x_vals_F_rt, priming_effect_F_rt.values, color=colors)
plt.xlabel('Participant (sequential)', fontsize=12)
plt.ylabel("Priming Effect (F): F mean - Global DFU mean", fontsize=12)
plt.title('Priming Effect Across Participants (reaction_time_TEST, F)', fontsize=14)
plt.xticks(ticks=np.arange(1, n+1, 5))
plt.yticks(fontsize=10)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height if height>=0 else 0,
             f'{height:.2f}', ha='center', va='bottom', fontsize=3, color='black')
plt.axhline(y=overall_effect_F_rt, color='black', linestyle='--', linewidth=2)
plt.text(0.95, overall_effect_F_rt, overall_annotation_F_rt,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
output_file = f"{base_dir}/Participants_F_reaction_time_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()
