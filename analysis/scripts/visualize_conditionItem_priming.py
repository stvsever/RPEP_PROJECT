import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.cm as cm

# ---------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
df = pd.read_csv(file_path)

base_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/priming_effects/condition_items"

# ---------------------------------------------------------
# Helper Function: Extract numeric part using re
# ---------------------------------------------------------
def extract_number(label):
    """
    Extracts the first sequence of digits from a label string.
    Returns the number as an integer.
    """
    match = re.search(r'\d+', label)
    return int(match.group()) if match else 0

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
#                   ANALYSIS FOR BI_score
# =============================================================================
# ---------------------------------------------------------
# Global Means for BI_score
# ---------------------------------------------------------
f_means_BI = df[df['experimental_condition'] == 'F'].groupby('condition_item')['BI_score'].mean()
f_global_mean_BI = f_means_BI.mean()

dfu_means_BI = df[df['experimental_condition'] == 'DFU'].groupby('condition_item')['BI_score'].mean()
dfu_global_mean_BI = dfu_means_BI.mean()

# ---------------------------------------------------------
# Process DFU Data for BI_score
# ---------------------------------------------------------
df_dfu = df[df['experimental_condition'] == 'DFU'].copy()
df_dfu['item_number'] = df_dfu['condition_item'].apply(extract_number)
df_dfu = df_dfu.sort_values('item_number')
dfu_item_means_BI = df_dfu.groupby('condition_item')['BI_score'].mean()
dfu_item_means_BI = dfu_item_means_BI.sort_index(key=lambda x: x.map(extract_number))
dfu_priming_effect_BI = dfu_item_means_BI - f_global_mean_BI

# ---------------------------------------------------------
# Compute per-item p-values for DFU (BI_score)
# ---------------------------------------------------------
dfu_pvalues_BI = {}
print("DFU Condition Items p-values (BI_score, comparing each item's BI_scores to F global mean):")
for item in sorted(df_dfu['condition_item'].unique(), key=extract_number):
    group = df_dfu[df_dfu['condition_item'] == item]['BI_score']
    t_stat, p_val = ttest_1samp(group, f_global_mean_BI)
    dfu_pvalues_BI[item] = p_val
    print(f"DFU {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for DFU (BI_score)
# ---------------------------------------------------------
dfu_overall_mean = df[df['experimental_condition'] == 'DFU']['BI_score'].mean()
dfu_overall_effect_BI = dfu_overall_mean - f_global_mean_BI
t_stat_all, p_val_all = ttest_1samp(df[df['experimental_condition'] == 'DFU']['BI_score'], f_global_mean_BI)
dfu_overall_annotation = f"Overall: {dfu_overall_effect_BI:.2f}, p = {p_val_all:.4f}"
print(f"\nDFU Overall BI_score: priming effect = {dfu_overall_effect_BI:.2f}, p-value = {p_val_all:.4f}")

# ---------------------------------------------------------
# Plot DFU BI_score with horizontal line (using Blues colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(dfu_priming_effect_BI.index)
colors = [cm.Reds(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(dfu_priming_effect_BI.index, dfu_priming_effect_BI.values, color=colors)
plt.xlabel('Condition Item (DFU)', fontsize=12)
plt.ylabel("Priming Effect: 'DFU mean - Global mean of F means'", fontsize=12)
plt.title('Priming Effect per DFU Condition Item (BI_score)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(dfu_priming_effect_BI.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(dfu_pvalues_BI.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=dfu_overall_effect_BI, color='black', linestyle='--', linewidth=2)
plt.text(0.95, dfu_overall_effect_BI, dfu_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/DFU_items_BI_score.png"
plt.savefig(output_file, dpi=300)
plt.show()

# ---------------------------------------------------------
# Process F Data for BI_score
# ---------------------------------------------------------
df_f = df[df['experimental_condition'] == 'F'].copy()
df_f['item_number'] = df_f['condition_item'].apply(extract_number)
df_f = df_f.sort_values('item_number')
f_item_means_BI = df_f.groupby('condition_item')['BI_score'].mean()
f_item_means_BI = f_item_means_BI.sort_index(key=lambda x: x.map(extract_number))
f_priming_effect_BI = f_item_means_BI - dfu_global_mean_BI

# ---------------------------------------------------------
# Compute per-item p-values for F (BI_score)
# ---------------------------------------------------------
f_pvalues_BI = {}
print("\nF Condition Items p-values (BI_score, comparing each item's BI_scores to DFU global mean):")
for item in sorted(df_f['condition_item'].unique(), key=extract_number):
    group = df_f[df_f['condition_item'] == item]['BI_score']
    t_stat, p_val = ttest_1samp(group, dfu_global_mean_BI)
    f_pvalues_BI[item] = p_val
    print(f"F {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for F (BI_score)
# ---------------------------------------------------------
f_overall_mean = df[df['experimental_condition'] == 'F']['BI_score'].mean()
f_overall_effect_BI = f_overall_mean - dfu_global_mean_BI
t_stat_all_f, p_val_all_f = ttest_1samp(df[df['experimental_condition'] == 'F']['BI_score'], dfu_global_mean_BI)
f_overall_annotation = f"Overall: {f_overall_effect_BI:.2f}, p = {p_val_all_f:.4f}"
print(f"\nF Overall BI_score: priming effect = {f_overall_effect_BI:.2f}, p-value = {p_val_all_f:.4f}")

# ---------------------------------------------------------
# Plot F BI_score with horizontal line (using Reds colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(f_priming_effect_BI.index)
colors = [cm.Reds(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(f_priming_effect_BI.index, f_priming_effect_BI.values, color=colors)
plt.xlabel('Condition Item (F)', fontsize=12)
plt.ylabel("Priming Effect: 'F mean - Global mean of DFU means'", fontsize=12)
plt.title('Priming Effect per F Condition Item (BI_score)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(f_priming_effect_BI.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(f_pvalues_BI.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=f_overall_effect_BI, color='black', linestyle='--', linewidth=2)
plt.text(0.95, f_overall_effect_BI, f_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/Inverse_F_items_BI_score.png"
plt.savefig(output_file, dpi=300)
plt.show()

# =============================================================================
#                   ANALYSIS FOR score_TEST
# =============================================================================
# ---------------------------------------------------------
# Global Means for score_TEST
# ---------------------------------------------------------
f_means_test = df[df['experimental_condition'] == 'F'].groupby('condition_item')['score_TEST'].mean()
f_global_mean_test = f_means_test.mean()

dfu_means_test = df[df['experimental_condition'] == 'DFU'].groupby('condition_item')['score_TEST'].mean()
dfu_global_mean_test = dfu_means_test.mean()

# ---------------------------------------------------------
# Process DFU Data for score_TEST
# ---------------------------------------------------------
df_dfu_test = df[df['experimental_condition'] == 'DFU'].copy()
df_dfu_test['item_number'] = df_dfu_test['condition_item'].apply(extract_number)
df_dfu_test = df_dfu_test.sort_values('item_number')
dfu_item_means_test = df_dfu_test.groupby('condition_item')['score_TEST'].mean()
dfu_item_means_test = dfu_item_means_test.sort_index(key=lambda x: x.map(extract_number))
dfu_priming_effect_test = dfu_item_means_test - f_global_mean_test

# ---------------------------------------------------------
# Compute per-item p-values for DFU (score_TEST)
# ---------------------------------------------------------
dfu_pvalues_test = {}
print("\nDFU Condition Items p-values (score_TEST, comparing each item's score_TEST to F global mean):")
for item in sorted(df_dfu_test['condition_item'].unique(), key=extract_number):
    group = df_dfu_test[df_dfu_test['condition_item'] == item]['score_TEST']
    t_stat, p_val = ttest_1samp(group, f_global_mean_test)
    dfu_pvalues_test[item] = p_val
    print(f"DFU {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for DFU (score_TEST)
# ---------------------------------------------------------
dfu_overall_mean_test = df[df['experimental_condition'] == 'DFU']['score_TEST'].mean()
dfu_overall_effect_test = dfu_overall_mean_test - f_global_mean_test
t_stat_all_test, p_val_all_test = ttest_1samp(df[df['experimental_condition'] == 'DFU']['score_TEST'], f_global_mean_test)
dfu_overall_annotation = f"Overall: {dfu_overall_effect_test:.2f}, p = {p_val_all_test:.4f}"
print(f"\nDFU Overall score_TEST: priming effect = {dfu_overall_effect_test:.2f}, p-value = {p_val_all_test:.4f}")

# ---------------------------------------------------------
# Plot DFU score_TEST with horizontal line (using Greens colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(dfu_priming_effect_test.index)
colors = [cm.Purples(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(dfu_priming_effect_test.index, dfu_priming_effect_test.values, color=colors)
plt.xlabel('Condition Item (DFU)', fontsize=12)
plt.ylabel("Priming Effect: 'DFU mean - Global mean of F means' (score_TEST)", fontsize=12)
plt.title('Priming Effect per DFU Condition Item (score_TEST)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(dfu_priming_effect_test.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(dfu_pvalues_test.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=dfu_overall_effect_test, color='black', linestyle='--', linewidth=2)
plt.text(0.95, dfu_overall_effect_test, dfu_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/DFU_items_score_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# ---------------------------------------------------------
# Process F Data for score_TEST
# ---------------------------------------------------------
df_f_test = df[df['experimental_condition'] == 'F'].copy()
df_f_test['item_number'] = df_f_test['condition_item'].apply(extract_number)
df_f_test = df_f_test.sort_values('item_number')
f_item_means_test = df_f_test.groupby('condition_item')['score_TEST'].mean()
f_item_means_test = f_item_means_test.sort_index(key=lambda x: x.map(extract_number))
f_priming_effect_test = f_item_means_test - dfu_global_mean_test

# ---------------------------------------------------------
# Compute per-item p-values for F (score_TEST)
# ---------------------------------------------------------
f_pvalues_test = {}
print("\nF Condition Items p-values (score_TEST, comparing each item's score_TEST to DFU global mean):")
for item in sorted(df_f_test['condition_item'].unique(), key=extract_number):
    group = df_f_test[df_f_test['condition_item'] == item]['score_TEST']
    t_stat, p_val = ttest_1samp(group, dfu_global_mean_test)
    f_pvalues_test[item] = p_val
    print(f"F {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for F (score_TEST)
# ---------------------------------------------------------
f_overall_mean_test = df[df['experimental_condition'] == 'F']['score_TEST'].mean()
f_overall_effect_test = f_overall_mean_test - dfu_global_mean_test
t_stat_all_f_test, p_val_all_f_test = ttest_1samp(df[df['experimental_condition'] == 'F']['score_TEST'], dfu_global_mean_test)
f_overall_annotation = f"Overall: {f_overall_effect_test:.2f}, p = {p_val_all_f_test:.4f}"
print(f"\nF Overall score_TEST: priming effect = {f_overall_effect_test:.2f}, p-value = {p_val_all_f_test:.4f}")

# ---------------------------------------------------------
# Plot F score_TEST with horizontal line (using Purples colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(f_priming_effect_test.index)
colors = [cm.Purples(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(f_priming_effect_test.index, f_priming_effect_test.values, color=colors)
plt.xlabel('Condition Item (F)', fontsize=12)
plt.ylabel("Priming Effect: 'F mean - Global mean of DFU means' (score_TEST)", fontsize=12)
plt.title('Priming Effect per F Condition Item (score_TEST)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(f_priming_effect_test.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(f_pvalues_test.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=f_overall_effect_test, color='black', linestyle='--', linewidth=2)
plt.text(0.95, f_overall_effect_test, f_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/Inverse_F_items_score_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# =============================================================================
#                ANALYSIS FOR reaction_time_TEST
# =============================================================================
# ---------------------------------------------------------
# Global Means for reaction_time_TEST
# ---------------------------------------------------------
f_means_rt = df[df['experimental_condition'] == 'F'].groupby('condition_item')['reaction_time_TEST'].mean()
f_global_mean_rt = f_means_rt.mean()

dfu_means_rt = df[df['experimental_condition'] == 'DFU'].groupby('condition_item')['reaction_time_TEST'].mean()
dfu_global_mean_rt = dfu_means_rt.mean()

# ---------------------------------------------------------
# Process DFU Data for reaction_time_TEST
# ---------------------------------------------------------
df_dfu_rt = df[df['experimental_condition'] == 'DFU'].copy()
df_dfu_rt['item_number'] = df_dfu_rt['condition_item'].apply(extract_number)
df_dfu_rt = df_dfu_rt.sort_values('item_number')
dfu_item_means_rt = df_dfu_rt.groupby('condition_item')['reaction_time_TEST'].mean()
dfu_item_means_rt = dfu_item_means_rt.sort_index(key=lambda x: x.map(extract_number))
dfu_priming_effect_rt = dfu_item_means_rt - f_global_mean_rt

# ---------------------------------------------------------
# Compute per-item p-values for DFU (reaction_time_TEST)
# ---------------------------------------------------------
dfu_pvalues_rt = {}
print("\nDFU Condition Items p-values (reaction_time_TEST, comparing each item's reaction_time_TEST to F global mean):")
for item in sorted(df_dfu_rt['condition_item'].unique(), key=extract_number):
    group = df_dfu_rt[df_dfu_rt['condition_item'] == item]['reaction_time_TEST']
    t_stat, p_val = ttest_1samp(group, f_global_mean_rt)
    dfu_pvalues_rt[item] = p_val
    print(f"DFU {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for DFU (reaction_time_TEST)
# ---------------------------------------------------------
dfu_overall_mean_rt = df[df['experimental_condition'] == 'DFU']['reaction_time_TEST'].mean()
dfu_overall_effect_rt = dfu_overall_mean_rt - f_global_mean_rt
t_stat_all_rt, p_val_all_rt = ttest_1samp(df[df['experimental_condition'] == 'DFU']['reaction_time_TEST'], f_global_mean_rt)
dfu_overall_annotation = f"Overall: {dfu_overall_effect_rt:.2f}, p = {p_val_all_rt:.4f}"
print(f"\nDFU Overall reaction_time_TEST: priming effect = {dfu_overall_effect_rt:.2f}, p-value = {p_val_all_rt:.4f}")

# ---------------------------------------------------------
# Plot DFU reaction_time_TEST with horizontal line (using Oranges colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(dfu_priming_effect_rt.index)
colors = [cm.Blues(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(dfu_priming_effect_rt.index, dfu_priming_effect_rt.values, color=colors)
plt.xlabel('Condition Item (DFU)', fontsize=12)
plt.ylabel("Priming Effect: 'DFU mean - Global mean of F means' (reaction_time_TEST)", fontsize=12)
plt.title('Priming Effect per DFU Condition Item (reaction_time_TEST)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(dfu_priming_effect_rt.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(dfu_pvalues_rt.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=dfu_overall_effect_rt, color='black', linestyle='--', linewidth=2)
plt.text(0.95, dfu_overall_effect_rt, dfu_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/DFU_items_reaction_time_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()

# ---------------------------------------------------------
# Process F Data for reaction_time_TEST
# ---------------------------------------------------------
df_f_rt = df[df['experimental_condition'] == 'F'].copy()
df_f_rt['item_number'] = df_f_rt['condition_item'].apply(extract_number)
df_f_rt = df_f_rt.sort_values('item_number')
f_item_means_rt = df_f_rt.groupby('condition_item')['reaction_time_TEST'].mean()
f_item_means_rt = f_item_means_rt.sort_index(key=lambda x: x.map(extract_number))
f_priming_effect_rt = f_item_means_rt - dfu_global_mean_rt

# ---------------------------------------------------------
# Compute per-item p-values for F (reaction_time_TEST)
# ---------------------------------------------------------
f_pvalues_rt = {}
print("\nF Condition Items p-values (reaction_time_TEST, comparing each item's reaction_time_TEST to DFU global mean):")
for item in sorted(df_f_rt['condition_item'].unique(), key=extract_number):
    group = df_f_rt[df_f_rt['condition_item'] == item]['reaction_time_TEST']
    t_stat, p_val = ttest_1samp(group, dfu_global_mean_rt)
    f_pvalues_rt[item] = p_val
    print(f"F {item}: p-value = {p_val:.4f}")

# ---------------------------------------------------------
# Compute overall priming effect for F (reaction_time_TEST)
# ---------------------------------------------------------
f_overall_mean_rt = df[df['experimental_condition'] == 'F']['reaction_time_TEST'].mean()
f_overall_effect_rt = f_overall_mean_rt - dfu_global_mean_rt
t_stat_all_f_rt, p_val_all_f_rt = ttest_1samp(df[df['experimental_condition'] == 'F']['reaction_time_TEST'], dfu_global_mean_rt)
f_overall_annotation = f"Overall: {f_overall_effect_rt:.2f}, p = {p_val_all_f_rt:.4f}"
print(f"\nF Overall reaction_time_TEST: priming effect = {f_overall_effect_rt:.2f}, p-value = {p_val_all_f_rt:.4f}")

# ---------------------------------------------------------
# Plot F reaction_time_TEST with horizontal line (using Copper colormap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
n = len(f_priming_effect_rt.index)
colors = [cm.Blues(0.3 + 0.7*(i/n)) for i in range(n)]
bars = plt.bar(f_priming_effect_rt.index, f_priming_effect_rt.values, color=colors)
plt.xlabel('Condition Item (F)', fontsize=12)
plt.ylabel("Priming Effect: 'F mean - Global mean of DFU means' (reaction_time_TEST)", fontsize=12)
plt.title('Priming Effect per F Condition Item (reaction_time_TEST)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for item, bar in zip(f_priming_effect_rt.index, bars):
    height = bar.get_height()
    y_coord = height if height >= 0 else 0
    code = significance_code(f_pvalues_rt.get(item, 1))
    plt.text(bar.get_x() + bar.get_width()/2, y_coord, f'{height:.2f} {code}',
             ha='center', va='bottom', fontsize=10, color='black')
plt.axhline(y=f_overall_effect_rt, color='black', linestyle='--', linewidth=2)
plt.text(0.95, f_overall_effect_rt, f_overall_annotation,
         transform=plt.gca().get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='black')
plt.tight_layout()
output_file = f"{base_dir}/Inverse_F_items_reaction_time_TEST.png"
plt.savefig(output_file, dpi=300)
plt.show()
