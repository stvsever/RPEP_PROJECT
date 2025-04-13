import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.4):
    """
    Lightens the given color by mixing it with white.
    Input can be matplotlib color string, hex string, or RGB tuple.
    The amount specifies how much to lighten (0 is original, 1 is white).
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = mc.to_rgb(c)
    # Convert to HLS to adjust lightness
    h, l, s = colorsys.rgb_to_hls(*c)
    # Increase the lightness; ensure it doesn't exceed 1
    l_new = min(1, l + (1 - l) * amount)
    return colorsys.hls_to_rgb(h, l_new, s)


# ---------------------------
# Settings & Directory Setup
# ---------------------------
csv_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
base_plot_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/priming_effects"

# Aggregated directories for box and violin plots (by task type)
aggregated_box_dir = os.path.join(base_plot_dir, "aggregated", "box_plots")
aggregated_violin_dir = os.path.join(base_plot_dir, "aggregated", "violin_plots")

# Task_id specific directories
taskid_box_dir = os.path.join(base_plot_dir, "task_id", "box_plots")
combined_box_dir = os.path.join(base_plot_dir, "task_id", "combined_box_plots")
combined_violin_dir = os.path.join(base_plot_dir, "task_id", "combined_violin_plots")

# Directory for full dataset plots (across both task_types)
full_dataset_dir = os.path.join(base_plot_dir, "full_dataset")
for d in [aggregated_box_dir, aggregated_violin_dir, taskid_box_dir, combined_box_dir, combined_violin_dir,
          full_dataset_dir]:
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Data Import & Preprocessing
# ---------------------------
df = pd.read_csv(csv_path)

# Exclude outliers (> 3SD) for each measure (trial-based)
for measure in ["reaction_time_TEST", "score_TEST"]:
    mean_val = df[measure].mean()
    std_val = df[measure].std()
    df = df[(df[measure] >= mean_val - 3 * std_val) & (df[measure] <= mean_val + 3 * std_val)]

# Exclude trials until first correct answer in each block
df = df.sort_values(by=["participant_identification", "block_number"])
df["trial_in_block"] = df.groupby(["participant_identification", "block_number"]).cumcount() + 1
df["cum_correct"] = df.groupby(["participant_identification", "block_number"])["MCQ_response_unprocessed"] \
    .transform(lambda x: (x == "correct").cumsum())
df_filtered = df[df["cum_correct"] > 0].copy()

# Report exclusion counts
total_trials = len(df)
filtered_trials = len(df_filtered)
print("\n=== Exclusion Report ===")
print(f"Total trials: {total_trials}")
print(f"Trials kept: {filtered_trials}")
print(f"Trials excluded: {total_trials - filtered_trials}")

# Use the filtered dataset for visualization
df_vis = df_filtered

# ---------------------------
# Aggregated Plot Annotation Info (by task type)
# ---------------------------
# Hard-coded descriptive stats from your previous analysis (by task type)
annotation_dict = {
    "RAT": {
        "score_TEST": {
            "DFU": {"mean": 0.102941, "sd": 0.3043299, "n": 340},
            "F": {"mean": 0.103774, "sd": 0.3054472, "n": 318},
            "p": 0.203, "sig": "n.s."
        },
        "reaction_time_TEST": {
            "DFU": {"mean": 15.17944, "sd": 4.69849, "n": 340},
            "F": {"mean": 14.39825, "sd": 4.59702, "n": 318},
            "p": 0.03716, "sig": "*"
        },
        "BI_score": {
            "DFU": {"mean": -0.092037, "sd": 1.11354, "n": 340},
            "F": {"mean": 0.082022, "sd": 1.11007, "n": 318},
            "p": 0.0699, "sig": "."
        }
    },
    "Insight": {
        "score_TEST": {
            "DFU": {"mean": 0.5739645, "sd": 0.4952321, "n": 338},
            "F": {"mean": 0.5110410, "sd": 0.5006684, "n": 317},
            "p": 0.271, "sig": "n.s."
        },
        "reaction_time_TEST": {
            "DFU": {"mean": 54.47702, "sd": 35.01172, "n": 338},
            "F": {"mean": 52.78301, "sd": 34.40638, "n": 317},
            "p": 0.0305, "sig": "*"
        },
        "BI_score": {
            "DFU": {"mean": 0.02001, "sd": 1.16515, "n": 338},
            "F": {"mean": -0.01962, "sd": 1.17818, "n": 317},
            "p": 0.347, "sig": "n.s."
        }
    }
}


def add_diff_annotation(ax, y, h, diff_text):
    """
    Draws a horizontal bracket between the two conditions (x=0 and x=1)
    and annotates it with the provided difference text.
    """
    ax.plot([0, 0], [y, y + h], color='black', lw=1.5)
    ax.plot([1, 1], [y, y + h], color='black', lw=1.5)
    ax.plot([0, 1], [y + h, y + h], color='black', lw=1.5)
    ax.annotate(diff_text, xy=((0 + 1) / 2, y + h), xytext=((0 + 1) / 2, y + h + 0.02 * (y + h)),
                ha='center', va='bottom', fontsize=12, color='black')


# Create a light-saturated two-color palette based on the "colorblind" palette.
base_colors = sns.color_palette("colorblind", 2)
palette_agg = [lighten_color(c, amount=0.7) for c in base_colors]

# For the full dataset plots: use palette_agg for reaction time and accuracy,
# and a distinct palette for BI_score using a different base (here from "Set1").
base_colors_bi = sns.color_palette("Set1", 2)
palette_bi = [lighten_color(c, amount=0.7) for c in base_colors_bi]

# ---------------------------
# Visualization: By Task Type
# ---------------------------
for task in ["RAT", "Insight"]:
    df_task = df_vis[df_vis["task_type"] == task]

    # --- Aggregated Plots (by participant) ---
    # Compute average per participant & condition
    agg_data = df_task.groupby(["participant_identification", "experimental_condition"]).agg({
        "reaction_time_TEST": "mean",
        "score_TEST": "mean",
        "BI_score": "mean"
    }).reset_index()

    for measure, label in [("reaction_time_TEST", "Reaction Time (ms)"),
                           ("score_TEST", "Accuracy Score"),
                           ("BI_score", "BI Score")]:
        info = annotation_dict[task][measure]
        diff = info["F"]["mean"] - info["DFU"]["mean"]
        diff_text = f"Δ={diff:.3f}   p={info['p']} {info['sig']}"

        title = "Remote Associates Test" if task == "RAT" else "Classical Insight Test"

        # Aggregated Box Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="experimental_condition", y=measure, data=agg_data, palette=palette_agg, ax=ax)
        ax.set_title(f"{title}", fontsize=14)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        y_max = agg_data[measure].max()
        y_min = agg_data[measure].min()
        y = y_max + 0.05 * (y_max - y_min)
        h = 0.03 * (y_max - y_min)
        add_diff_annotation(ax, y, h, diff_text)
        plt.tight_layout()
        box_file = os.path.join(aggregated_box_dir, f"{task}_aggregated_{measure}_box.png")
        plt.savefig(box_file, dpi=300)
        plt.close()

        # Aggregated Violin Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(x="experimental_condition", y=measure, data=agg_data, palette=palette_agg, ax=ax)
        ax.set_title(f"{title}", fontsize=14)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        y_max = agg_data[measure].max()
        y_min = agg_data[measure].min()
        y = y_max + 0.05 * (y_max - y_min)
        h = 0.03 * (y_max - y_min)
        add_diff_annotation(ax, y, h, diff_text)
        plt.tight_layout()
        violin_file = os.path.join(aggregated_violin_dir, f"{task}_aggregated_{measure}_violin.png")
        plt.savefig(violin_file, dpi=300)
        plt.close()

    # --- Task_id Specific Plots (Individual Box Plots) ---
    task_ids = df_task["task_id"].unique()
    for t_id in task_ids:
        df_taskid = df_task[df_task["task_id"] == t_id]
        for measure, label in [("reaction_time_TEST", "Reaction Time (ms)"),
                               ("score_TEST", "Accuracy Score"),
                               ("BI_score", "BI Score")]:
            plt.figure(figsize=(8, 6))
            df_taskid.boxplot(column=measure, by="experimental_condition", grid=False)
            plt.title(f"{task}: {label} (Task {t_id})", fontsize=14)
            plt.suptitle("")
            plt.xlabel("Condition", fontsize=12)
            plt.ylabel(label, fontsize=12)
            plt.tight_layout()
            taskid_box_file = os.path.join(taskid_box_dir, f"{task}_task_{t_id}_{measure}_box.png")
            plt.savefig(taskid_box_file, dpi=300)
            plt.close()

    # --- Combined Plots for All Task IDs ---
    for measure, label in [("reaction_time_TEST", "Reaction Time (ms)"),
                           ("score_TEST", "Accuracy Score"),
                           ("BI_score", "BI Score")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        palette_rev_set2 = list(sns.color_palette("Set2", n_colors=len(df_task["task_id"].unique())))[::-1]
        sns.boxplot(x="experimental_condition", y=measure, hue="task_id", data=df_task, palette=palette_rev_set2, ax=ax)
        ax.set_title(f"{task}: {label} (Combined Box)", fontsize=14)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(title="Task ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        combined_box_file = os.path.join(combined_box_dir, f"{task}_combined_{measure}_box.png")
        plt.savefig(combined_box_file, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x="experimental_condition", y=measure, hue="task_id", data=df_task, palette=palette_rev_set2,
                       split=True, ax=ax)
        ax.set_title(f"{task}: {label} (Combined Violin)", fontsize=14)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(title="Task ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        combined_violin_file = os.path.join(combined_violin_dir, f"{task}_combined_{measure}_violin.png")
        plt.savefig(combined_violin_file, dpi=300)
        plt.close()

# ---------------------------
# Full Dataset Analysis Visualization (Both Task Types Combined)
# ---------------------------
# Here we compute descriptive statistics across the entire filtered dataset
df_full = df_vis.copy()
desc_score = df_full.groupby("experimental_condition")["score_TEST"].agg(['mean', 'std']).reset_index()
desc_rt = df_full.groupby("experimental_condition")["reaction_time_TEST"].agg(['mean', 'std']).reset_index()
desc_bi = df_full.groupby("experimental_condition")["BI_score"].agg(['mean', 'std']).reset_index()

print("score_TEST Descriptive Statistics (Full Filtered Dataset):")
print(desc_score)
print("reaction_time_TEST Descriptive Statistics (Full Filtered Dataset):")
print(desc_rt)
print("BI_score Descriptive Statistics (Full Filtered Dataset):")
print(desc_bi)

# Hard-coded annotation info from the printed analysis results for full dataset
annotation_full = {
    "score_TEST": {
        "DFU": {"mean": 0.3377581, "sd": 0.4732948},
        "F": {"mean": 0.3070866, "sd": 0.4616493},
        "p": 0.273, "sig": "n.s."
    },
    "reaction_time_TEST": {
        "DFU": {"mean": 34.77027, "sd": 31.74729},
        "F": {"mean": 33.56041, "sd": 31.13721},
        "p": 0.0249, "sig": "*"
    },
    "BI_score": {
        "DFU": {"mean": -0.03617801, "sd": 1.14010},
        "F": {"mean": 0.03128343, "sd": 1.14480},
        "p": 0.0845, "sig": "."
    }
}

# Aggregated plots for full dataset (not split by task type)
for measure, label in [("reaction_time_TEST", "Reaction Time (ms)"),
                       ("score_TEST", "Accuracy Score"),
                       ("BI_score", "BI Score")]:
    agg_full = df_full.groupby("experimental_condition").agg({measure: "mean"}).reset_index()
    info = annotation_full[measure]
    diff = info["F"]["mean"] - info["DFU"]["mean"]
    diff_text = f"Δ={diff:.3f}   p={info['p']} {info['sig']}"

    # For reaction time and accuracy, use dark blue for DFU and red for F.
    # For BI_score, use dark purple for DFU and dark red for F.
    if measure in ["reaction_time_TEST", "score_TEST"]:
        full_palette = [lighten_color('#8B2500', amount=0.8), lighten_color('#000080', amount=0.8)]
    elif measure == "BI_score":
        full_palette = [lighten_color('#FF0000', amount=0.6), lighten_color('#00008B', amount=0.6)]

    # Full dataset aggregated Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="experimental_condition", y=measure, data=df_full, palette=full_palette, ax=ax)
    ax.set_title(f"Full Dataset - {label}", fontsize=14)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    y_max = df_full[measure].max()
    y_min = df_full[measure].min()
    y = y_max + 0.05 * (y_max - y_min)
    h = 0.03 * (y_max - y_min)
    add_diff_annotation(ax, y, h, diff_text)
    plt.tight_layout()
    full_box_file = os.path.join(full_dataset_dir, f"full_dataset_{measure}_box.png")
    plt.savefig(full_box_file, dpi=300)
    plt.close()

    # Full dataset aggregated Violin Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x="experimental_condition", y=measure, data=df_full, palette=full_palette, ax=ax)
    ax.set_title(f"Full Dataset - {label}", fontsize=14)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    y_max = df_full[measure].max()
    y_min = df_full[measure].min()
    y = y_max + 0.05 * (y_max - y_min)
    h = 0.03 * (y_max - y_min)
    add_diff_annotation(ax, y, h, diff_text)
    plt.tight_layout()
    full_violin_file = os.path.join(full_dataset_dir, f"full_dataset_{measure}_violin.png")
    plt.savefig(full_violin_file, dpi=300)
    plt.close()

# Close all figures
plt.close("all")
