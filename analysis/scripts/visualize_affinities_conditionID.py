import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the CSV file path (update if necessary)
csv_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Base directory for saving images (all plots will be saved here; no subdirectories will be created)
base_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/affinity_levels/condition_id"
os.makedirs(base_dir, exist_ok=True)


# Helper function: extract numeric part from condition_item for sorting
def extract_item_number(item):
    match = re.search(r'\d+', str(item))
    return int(match.group()) if match else 0


# Function to aggregate by condition_item and plot a single colorful bar chart
def plot_condition_items(df_subset, title, filename, cmap=plt.cm.viridis):
    # Aggregate average affinity_score per condition_item
    agg_data = df_subset.groupby('condition_item')['affinity_score'].mean().reset_index()
    agg_data['item_num'] = agg_data['condition_item'].apply(extract_item_number)
    agg_data = agg_data.sort_values(by='item_num')

    # Create a colorful bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cmap(np.linspace(0, 1, len(agg_data)))
    bars = ax.bar(agg_data['condition_item'], agg_data['affinity_score'], color=colors)

    ax.set_xlabel("Condition Item")
    ax.set_ylabel("Average Affinity Score")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    # Annotate each bar with its average value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    full_path = os.path.join(base_dir, filename)
    plt.savefig(full_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {full_path}")


# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# 1. Plot for DFU condition items only
df_DFU = df[df['experimental_condition'] == "DFU"]
plot_condition_items(df_DFU,
                     "Avg. Affinity Score per Condition Item (DFU)",
                     "DFU_condition_item_affinity_bar.png",
                     cmap=plt.cm.viridis)

# 2. Plot for F condition items only
df_F = df[df['experimental_condition'] == "F"]
plot_condition_items(df_F,
                     "Avg. Affinity Score per Condition Item (F)",
                     "F_condition_item_affinity_bar.png",
                     cmap=plt.cm.plasma)

# 3. Plot for the full dataset (both DFU and F)
plot_condition_items(df,
                     "Avg. Affinity Score per Condition Item (Full Dataset)",
                     "Full_condition_item_affinity_bar.png",
                     cmap=plt.cm.cividis)

# 4. Combined plot: Two subplots (top: DFU, bottom: F) with independent x-axis labels
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

# Function to aggregate data for plotting (used in combined plot)
def aggregate_data(df_subset):
    agg = df_subset.groupby('condition_item')['affinity_score'].mean().reset_index()
    agg['item_num'] = agg['condition_item'].apply(extract_item_number)
    return agg.sort_values(by='item_num')

# Top subplot: DFU condition items
agg_DFU = aggregate_data(df_DFU)
colors_DFU = plt.cm.viridis(np.linspace(0, 1, len(agg_DFU)))
bars = axes[0].bar(agg_DFU['condition_item'], agg_DFU['affinity_score'], color=colors_DFU)
axes[0].set_ylabel("Average Affinity Score")
axes[0].set_title("DFU Condition Items")
axes[0].tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Bottom subplot: F condition items
agg_F = aggregate_data(df_F)
colors_F = plt.cm.plasma(np.linspace(0, 1, len(agg_F)))
bars = axes[1].bar(agg_F['condition_item'], agg_F['affinity_score'], color=colors_F)
axes[1].set_xlabel("Condition Item")
axes[1].set_ylabel("Average Affinity Score")
axes[1].set_title("F Condition Items")
axes[1].tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    axes[1].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
combined_filename = os.path.join(base_dir, "Combined_DFU_F_condition_item_affinity_bar.png")
plt.savefig(combined_filename, dpi=300)
plt.close(fig)
print(f"Saved combined plot: {combined_filename}")
