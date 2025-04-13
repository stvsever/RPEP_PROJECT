import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Activate R <-> Python data conversion
import rpy2.robjects.pandas2ri as pandas2ri
pandas2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')
base = importr('base')
utils = importr('utils')

# Set the CSV file path (update if necessary)
csv_path = ("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
            "Master 1/SEMESTER 2/Research Project Experimental Psychology/"
            "RPEP_experiment/data/preprocessed/concatenated_data.csv")

# Read the CSV into R and remove rows with missing values in key columns
ro.r(f"""
df <- read.csv('{csv_path}', header=TRUE, stringsAsFactors=FALSE)
df <- df[!is.na(df$affinity_score) & !is.na(df$score_TEST) & !is.na(df$reaction_time_TEST), ]
""")

# Define filtering options for experimental_condition and task_type.
exp_conditions = ["DFU", "F", "Both"]
task_types = ["RAT", "Insight", "Both"]

# Define DV specifications.
# Each tuple is (dv_label, dv_formula)
# For the multivariate model the dv_formula will be ignored in favor of brms.
dv_specs = [
    ("score_TEST", "score_TEST"),
    ("reaction_time_TEST", "reaction_time_TEST"),
    ("Both (multivariate)", "cbind(score_TEST, reaction_time_TEST)")
]

# R code template for running one model.
# For univariate models we use lmerTest::lmer();
# For the multivariate model we use brms::brm().
model_r_script_template = r'''
# --- Subset the data ---
{exp_filter}
{task_filter}

cat("-----------------------------------------------------\n")
cat("Running model for:\n")
cat("  Experimental Condition: {exp_label}\n")
cat("  Task Type: {task_label}\n")
cat("  Dependent Variable: {dv_label}\n")
cat("Number of rows in subset: ", nrow(df_sub), "\n")
mean_affinity <- mean(df_sub$affinity_score, na.rm=TRUE)
cat("Mean Affinity Score: ", round(mean_affinity,2), "\n\n")

if("{dv_label}" == "Both (multivariate)") {{
    # Load brms for the multivariate model
    if(!require(brms)) {{
      install.packages("brms", repos="http://cran.r-project.org")
      library(brms)
    }} else {{
      library(brms)
    }}
    cat("Fitting multivariate model using brms...\n")
    model <- brm(
      bf(score_TEST ~ affinity_score + (1|participant_identification) + (1|task_id)) +
      bf(reaction_time_TEST ~ affinity_score + (1|participant_identification) + (1|task_id)),
      data = df_sub, chains = 2, iter = 2000, silent=TRUE
    )
    print(summary(model))
    # Visualize conditional effects for affinity_score
    cat("\nPlotting conditional effects for affinity_score...\n")
    ce <- conditional_effects(model, effects = "affinity_score")
    print(ce)
    plot(ce)  # This will open a plot window in R (or be rendered if using an R-enabled environment)
}} else {{
    # Fit univariate model using lmerTest::lmer
    model <- lmerTest::lmer({dv_formula} ~ affinity_score + (1|participant_identification) + (1|task_id), data = df_sub)
    print(summary(model))
    cat("\n--- ANOVA Table (Type III) ---\n")
    print(anova(model))
}}
cat("-----------------------------------------------------\n\n")
'''

# Counter for total models
model_count = 0

# Loop over experimental_condition, task_type, and DV specifications to run the models.
for exp in exp_conditions:
    # Build experimental condition filter and label
    if exp == "Both":
        exp_filter = "df_sub <- df"  # no filtering; use all experimental_condition values
        exp_label = "Both experimental conditions"
    else:
        exp_filter = f'df_sub <- subset(df, experimental_condition == "{exp}")'
        exp_label = exp
    for task in task_types:
        # Build task_type filter and label
        if task == "Both":
            task_filter = ""  # no filtering; use all task_type values
            task_label = "Both task types"
        else:
            task_filter = f'df_sub <- subset(df_sub, task_type == "{task}")'
            task_label = task
        for dv_label, dv_formula in dv_specs:
            # Format the R code with the proper filters, labels, and DV formula
            model_script = model_r_script_template.format(
                exp_filter=exp_filter,
                task_filter=task_filter,
                exp_label=exp_label,
                task_label=task_label,
                dv_label=dv_label,
                dv_formula=dv_formula
            )
            # Run the R script
            ro.r(model_script)
            model_count += 1

# After running all models, print total count and explanation.
print(f"\nTotal number of models tested: {model_count}")
print("Explanation: 3 experimental_condition options (DFU, F, Both) x 3 task_type options (RAT, Insight, Both) x 3 DV specifications (score_TEST, reaction_time_TEST, and multivariate cbind(score_TEST, reaction_time_TEST)) = 27 models.")
print("\nFor multivariate models, brms provides posterior summaries and conditional_effects plots, which can be interpreted using credible intervals (if the 95% CI for affinity_score does not include zero, the effect is significant).")
