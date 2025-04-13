import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Activate R <-> Python data conversion
import rpy2.robjects.pandas2ri as pandas2ri

pandas2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')
lme4 = importr('lme4')
base = importr('base')
utils = importr('utils')

# Set the CSV file path (update if necessary)
csv_path = (
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
    "Master 1/SEMESTER 2/Research Project Experimental Psychology/"
    "RPEP_experiment/data/preprocessed/concatenated_data.csv"
)

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
# For the multivariate model the dv_formula is ignored; we use MMeM_henderson3.
dv_specs = [
    ("score_TEST", "score_TEST"),
    ("reaction_time_TEST", "reaction_time_TEST"),
    ("Both (multivariate)", "c(score_TEST, reaction_time_TEST)")
]

# Template for running one model in R.
# For univariate models: lmerTest::lmer + bootstrap CIs for affinity_score.
# For the multivariate model: MMeM_henderson3 (no built-in p-values or boot CI).
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
    # Load MMeM package for multivariate mixed effects model
    if(!require(MMeM)) {{
      install.packages("MMeM", repos="http://cran.r-project.org")
      library(MMeM)
    }} else {{
      library(MMeM)
    }}
    cat("Fitting multivariate model using MMeM_henderson3...\n")
    # Note: MMeM only supports one random effect. We'll use participant_identification.
    # No built-in function for bootstrap intervals for the fixed effect in MMeM.
    model <- MMeM_henderson3(fml = c(score_TEST, reaction_time_TEST) ~ affinity_score + (1|participant_identification),
                             data = df_sub, factor_X = TRUE)
    print(model)
    cat("NOTE: MMeM does not provide p-values or direct bootstrapped CI for the fixed effect.\n")
}} else {{
    # Univariate model with lmerTest
    # Fit the model
    library(lmerTest)
    library(lme4)
    model <- lmerTest::lmer({dv_formula} ~ affinity_score + (1|participant_identification) + (1|task_id), data = df_sub)

    # Print summary (includes fixed-effect estimates, SE, t-values, p-values)
    cat("--- Model Summary (univariate) ---\n")
    print(summary(model))

    # Print ANOVA table
    cat("\n--- ANOVA Table (Type III) ---\n")
    print(anova(model))

    # Now compute bootstrap 95% CI for the affinity_score parameter
    # We do parametric bootstrap using confint from lme4
    cat("\n--- Bootstrapped 95% CI for 'affinity_score' ---\n")
    # We have to identify the row name for the fixed effect
    # confint will produce intervals for all parameters, but we just print the one for affinity_score
    # We'll do ~ "affinity_score"
    # We'll do 'method="boot"' with a certain number of simulations
    bs_ci <- confint(model, parm="beta_", method="boot", nsim=500)  # 500 = # of bootstrap reps
    # Note: We use "beta_" to get all fixed effects. We'll then extract the row matching affinity_score.

    # Identify the row index for the affinity_score (2nd fixed effect typically)
    # Alternatively, we can do a partial match on the row name
    rownames_ci <- rownames(bs_ci)
    idx_affinity <- which(grepl("affinity_score", rownames_ci))
    if(length(idx_affinity) == 1) {{
      cat("95% CI for affinity_score:\n")
      print(bs_ci[idx_affinity, , drop=FALSE])
    }} else {{
      cat("Could not find row for 'affinity_score' in confint output.\n")
    }}
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
print(
    "Explanation: 3 experimental_condition options (DFU, F, Both) x 3 task_type options (RAT, Insight, Both) x 3 DV specifications (score_TEST, reaction_time_TEST, and multivariate c(score_TEST, reaction_time_TEST)) = 27 models.")
print(
    "\nAll models have been fitted. For univariate models, you see a bootstrapped 95% CI for the 'affinity_score' fixed effect. For the multivariate MMeM model, p-values/CIs are not provided by the package.")
