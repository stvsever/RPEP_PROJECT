import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri
import pandas as pd

# Activate R <-> Python data conversions
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')  # For linear mixed models
base = importr('base')          # Base R
stats = importr('stats')        # For lm, anova, etc.
utils = importr('utils')        # For reading CSV etc.
dplyr = importr('dplyr')         # For data manipulation

# Path to your CSV file
csv_path = (
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
)

# Read the data into a pandas DataFrame
df = pd.read_csv(csv_path)

# Make sure we have the required columns for the analysis
required_cols = [
    "participant_identification",            # participant ID column
    "task_type",                             # task type: Insight or RAT
    "experimental_condition",                # independent variable
    "BI_score_regular",                              # global BI_score
    "BI_score_TaskType",                     # BI_score computed by grouping on task_type
    "BI_score_TaskID",                       # BI_score computed by grouping on task_id
    "BI_score_ParticipantID",                # BI_score computed by grouping on participant_identification
    "BI_score_TaskType_ParticipantID",       # BI_score computed by grouping on task_type & participant_identification
    "block_number",                          # block number for exclusion logic
    "MCQ_response_unprocessed"               # response column to check for "correct"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from the CSV file.")

# Convert the pandas DataFrame to an R DataFrame and assign it as 'df' in R
rdf = pandas2ri.py2rpy(df)
ro.globalenv["df"] = rdf

# Exclusion logic: Exclude trials until the first correct answer in each block.
print("\n=== EXCLUSION OF TRIALS UNTIL FIRST CORRECT ANSWER IN EACH BLOCK AND ANALYSIS ON FILTERED DATA ===\n")
exclusion_code = r"""
library(dplyr)
# Add a trial sequence number within each block and compute cumulative correct responses
df_with_seq <- df %>%
    group_by(participant_identification, block_number) %>%
    mutate(trial_in_block = row_number(),
           cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
    ungroup()

# Exclude trials until the first correct answer is given in each block:
# Keep only trials where at least one correct response has occurred (cum_correct > 0)
df_filtered <- df_with_seq %>% filter(cum_correct > 0)

# Report the exclusion counts
total_trials <- nrow(df)
filtered_trials <- nrow(df_filtered)
cat("Total trials in original dataset:", total_trials, "\n")
cat("Trials kept after excluding trials until first correct in each block:", filtered_trials, "\n")
cat("Trials excluded:", total_trials - filtered_trials, "\n")
"""
ro.r(exclusion_code)

# Assign the filtered dataset as the new full dataset
ro.r('rdf_full <- df_filtered')

# Create subsets based on task_type:
# For RAT
ro.r('rdf_rat <- subset(df_filtered, task_type == "RAT")')
# For Insight
ro.r('rdf_insight <- subset(df_filtered, task_type == "Insight")')

# Utility function to run R code and capture printed output, while catching specific errors
def run_r_code(code, model_label=""):
    """
    Runs a string of R code using rpy2, captures console output,
    and prints it in Python. If an error occurs due to insufficient grouping
    levels, a message is printed and the script continues.
    """
    cmd = f"""
    capture.output({{
        {code}
    }})
    """
    try:
        result = ro.r(cmd)
        for line in result:
            print(line)
    except Exception as e:
        if "grouping factors must have > 1 sampled level" in str(e):
            print(f"Skipping model {model_label} due to insufficient grouping factor levels. Error: {e}")
        else:
            raise e

# Function to print means, SDs, and n for each experimental_condition for a given BI score column
def print_condition_stats(bi_col, rdf_name):
    stats_code = f"""
    library(dplyr)
    cat("\\nMean, SD, and N for {bi_col} by experimental_condition:\\n")
    df_stats <- {rdf_name} %>%
      group_by(experimental_condition) %>%
      summarise(
        mean = mean({bi_col}, na.rm=TRUE),
        sd = sd({bi_col}, na.rm=TRUE),
        n = n()
      )
    print(df_stats)
    """
    run_r_code(stats_code)

# Function to run all models for a given BI score column and dataset
def run_all_models_for_bi_score(bi_col, rdf_name, dataset_label):
    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_label}")
    print(f"BI-SCORE COLUMN: {bi_col}")
    print("=" * 60)

    # Print condition stats first
    print_condition_stats(bi_col, rdf_name)

    # 1) Linear Model: bi_col ~ experimental_condition
    code_lm = f"""
    library(lmerTest)
    model_lm <- lm({bi_col} ~ experimental_condition, data={rdf_name})
    cat("\\n--- Linear Model: {bi_col} ~ experimental_condition ---\\n")
    print(summary(model_lm))
    """
    run_r_code(code_lm, model_label=f"Linear Model: {bi_col} on {dataset_label}")

    # 2) Mixed Model: (1|participant_identification)
    code_lmm_participant = f"""
    library(lmerTest)
    model_lmm_p <- lmer({bi_col} ~ experimental_condition + (1|participant_identification), data={rdf_name})
    cat("\\n--- Mixed Model: {bi_col} ~ experimental_condition + (1|participant_identification) ---\\n")
    print(summary(model_lmm_p))
    """
    run_r_code(code_lmm_participant,
               model_label=f"Mixed Model (participant_identification): {bi_col} on {dataset_label}")

    # 3) Mixed Model: (1|task_type)
    code_lmm_tasktype = f"""
    library(lmerTest)
    model_lmm_ttype <- lmer({bi_col} ~ experimental_condition + (1|task_type), data={rdf_name})
    cat("\\n--- Mixed Model: {bi_col} ~ experimental_condition + (1|task_type) ---\\n")
    print(summary(model_lmm_ttype))
    """
    run_r_code(code_lmm_tasktype, model_label=f"Mixed Model (task_type): {bi_col} on {dataset_label}")

    # 4) Mixed Model: (1|task_id)
    code_lmm_taskid = f"""
    library(lmerTest)
    model_lmm_tid <- lmer({bi_col} ~ experimental_condition + (1|task_id), data={rdf_name})
    cat("\\n--- Mixed Model: {bi_col} ~ experimental_condition + (1|task_id) ---\\n")
    print(summary(model_lmm_tid))
    """
    run_r_code(code_lmm_taskid, model_label=f"Mixed Model (task_id): {bi_col} on {dataset_label}")

    # 5) Mixed Model: (1|participant_identification) + (1|task_id)
    code_lmm_participant_taskid = f"""
    library(lmerTest)
    model_lmm_ptid <- lmer({bi_col} ~ experimental_condition + (1|participant_identification) + (1|task_id), data={rdf_name})
    cat("\\n--- Mixed Model: {bi_col} ~ experimental_condition + (1|participant_identification) + (1|task_id) ---\\n")
    print(summary(model_lmm_ptid))
    """
    run_r_code(code_lmm_participant_taskid,
               model_label=f"Mixed Model (participant_identification + task_id): {bi_col} on {dataset_label}")

# List of BI_score columns to analyze, now including the new variant
bi_score_cols = [
    "BI_score_regular",
    "BI_score_TaskType",
    "BI_score_TaskID",
    "BI_score_ParticipantID",
    "BI_score_TaskType_ParticipantID"
]

# Run models on the three datasets: FULL, RAT, and INSIGHT (all using the filtered data)
datasets = [
    ("rdf_full", "FULL FILTERED DATA"),
    ("rdf_rat", "RAT SUBSET (FILTERED)"),
    ("rdf_insight", "INSIGHT SUBSET (FILTERED)")
]

for rdf_name, dataset_label in datasets:
    for col in bi_score_cols:
        run_all_models_for_bi_score(col, rdf_name, dataset_label)

# FINAL PART 1: Interaction Models on FULL FILTERED DATA
print("\n" + "=" * 60)
print("FINAL: Interaction Models on FULL FILTERED DATA (for each BI_score column)")
print("=" * 60)
for col in bi_score_cols:
    print_condition_stats(col, "rdf_full")
    code_interaction = f"""
    library(lmerTest)
    model_inter <- lmer({col} ~ experimental_condition * task_type + (1|participant_identification) + (1|task_id), data=rdf_full)
    cat("\\n=== Interaction Model for {col}: {col} ~ experimental_condition * task_type + (1|participant_identification) + (1|task_id) ===\\n")
    print(summary(model_inter))
    """
    run_r_code(code_interaction, model_label=f"Interaction Model: {col} on FULL FILTERED DATA")

# FINAL PART 2: Non-Interaction Models on FULL FILTERED DATA
print("\n" + "=" * 60)
print("FINAL: Non-Interaction Models on FULL FILTERED DATA (for each BI_score column)")
print("=" * 60)
for col in bi_score_cols:
    print_condition_stats(col, "rdf_full")
    code_non_interaction = f"""
    library(lmerTest)
    model_non_inter <- lmer({col} ~ experimental_condition + task_type + (1|participant_identification) + (1|task_id), data=rdf_full)
    cat("\\n=== Non-Interaction Model for {col}: {col} ~ experimental_condition + task_type + (1|participant_identification) + (1|task_id) ===\\n")
    print(summary(model_non_inter))
    """
    run_r_code(code_non_interaction, model_label=f"Non-Interaction Model: {col} on FULL FILTERED DATA")

print("\nAnalysis complete.")
