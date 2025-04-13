import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the CSV file path (update if necessary)
csv_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Base directory for saving images
base_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/affinity_levels/task_id"
os.makedirs(base_dir, exist_ok=True)


# Helper function: extract numeric part from task_id for sorting
def extract_task_number(task_id):
    match = re.search(r'\d+', task_id)
    return int(match.group()) if match else 0


# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# -----------------------------
# 1. Four specific subsets: DFU_RAT, F_RAT, DFU_Insight, F_Insight
subsets = {
    "DFU_RAT": {"experimental_condition": "DFU", "task_type": "RAT"},
    "F_RAT": {"experimental_condition": "F", "task_type": "RAT"},
    "DFU_Insight": {"experimental_condition": "DFU", "task_type": "Insight"},
    "F_Insight": {"experimental_condition": "F", "task_type": "Insight"}
}

for sub_name, conditions in subsets.items():
    sub_dir = os.path.join(base_dir, sub_name)
    os.makedirs(sub_dir, exist_ok=True)

    # Filter data for the given experimental condition and task type
    df_sub = df[(df['experimental_condition'] == conditions["experimental_condition"]) &
                (df['task_type'] == conditions["task_type"])]

    # Group by task_id and calculate average affinity_score
    agg_data = df_sub.groupby('task_id')['affinity_score'].mean().reset_index()
    # Sort using the numeric part of the task_id
    agg_data['task_num'] = agg_data['task_id'].apply(extract_task_number)
    agg_data = agg_data.sort_values(by='task_num')

    # Create colorful bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(agg_data)))
    bars = ax.bar(agg_data['task_id'], agg_data['affinity_score'], color=colors)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Average Affinity Score")
    ax.set_title(
        f"Avg. Affinity Score per Task ID\nCondition: {conditions['experimental_condition']} | Task: {conditions['task_type']}")

    # Annotate bars with average value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    filename = os.path.join(sub_dir, f"{sub_name}_affinity_bar.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filename}")

# -----------------------------
# 2. Additional subsets: Insight_All, RAT_All, DFU_All, F_All

extra_subsets = {
    "Insight_All": {"filter_col": "task_type", "filter_val": "Insight"},
    "RAT_All": {"filter_col": "task_type", "filter_val": "RAT"},
    "DFU_All": {"filter_col": "experimental_condition", "filter_val": "DFU"},
    "F_All": {"filter_col": "experimental_condition", "filter_val": "F"}
}

for sub_name, params in extra_subsets.items():
    sub_dir = os.path.join(base_dir, sub_name)
    os.makedirs(sub_dir, exist_ok=True)

    df_sub = df[df[params["filter_col"]] == params["filter_val"]]

    agg_data = df_sub.groupby('task_id')['affinity_score'].mean().reset_index()
    agg_data['task_num'] = agg_data['task_id'].apply(extract_task_number)
    agg_data = agg_data.sort_values(by='task_num')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(agg_data)))
    bars = ax.bar(agg_data['task_id'], agg_data['affinity_score'], color=colors)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Average Affinity Score")
    ax.set_title(f"Avg. Affinity Score per Task ID\nSubset: {params['filter_val']} (by {params['filter_col']})")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    filename = os.path.join(sub_dir, f"{sub_name}_affinity_bar.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filename}")

# -----------------------------
# 3. Full data across 24 task IDs (both Insight and RAT) with two rows:
#    Top row: Insight (task_id starts with 'I')
#    Bottom row: RAT (task_id starts with 'R')

full_dir = os.path.join(base_dir, "Full_Data")
os.makedirs(full_dir, exist_ok=True)

# Filter Insight and RAT tasks separately
df_insight = df[df['task_id'].str.startswith("I")]
df_rat = df[df['task_id'].str.startswith("R")]


def aggregate_and_sort(df_subset):
    agg = df_subset.groupby('task_id')['affinity_score'].mean().reset_index()
    agg['task_num'] = agg['task_id'].apply(extract_task_number)
    return agg.sort_values(by='task_num')


agg_insight = aggregate_and_sort(df_insight)
agg_rat = aggregate_and_sort(df_rat)

# Create a figure with two subplots (rows)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot for Insight (top row)
colors_insight = plt.cm.cividis(np.linspace(0, 1, len(agg_insight)))
axes[0].bar(agg_insight['task_id'], agg_insight['affinity_score'], color=colors_insight)
axes[0].set_ylabel("Average Affinity Score")
axes[0].set_title("Insight Tasks (Task IDs starting with 'I')")
for bar in axes[0].patches:
    height = bar.get_height()
    axes[0].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot for RAT (bottom row)
colors_rat = plt.cm.cividis(np.linspace(0, 1, len(agg_rat)))
axes[1].bar(agg_rat['task_id'], agg_rat['affinity_score'], color=colors_rat)
axes[1].set_xlabel("Task ID")
axes[1].set_ylabel("Average Affinity Score")
axes[1].set_title("RAT Tasks (Task IDs starting with 'R')")
for bar in axes[1].patches:
    height = bar.get_height()
    axes[1].annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
filename = os.path.join(full_dir, "Full_Data_affinity_bar.png")
plt.savefig(filename, dpi=300)
plt.close(fig)
print(f"Saved full data plot: {filename}")
