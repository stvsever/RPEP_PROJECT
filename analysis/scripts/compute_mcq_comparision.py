import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects import pandas2ri

# Activate conversion between numpy, pandas and R objects
numpy2ri.activate()
pandas2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')
base = importr('base')
stats = importr('stats')
utils = importr('utils')
# Load additional libraries in R
ro.r('library(ggplot2)')
ro.r('library(emmeans)')
ro.r('library(effectsize)')

# CSV file path
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data_MCQ.csv"

# 1. Load the concatenated data
df = pd.read_csv(file_path)

# 2. Convert 'response_1' to a binary 'accuracy' column (1 = correct, 0 = otherwise)
df['accuracy'] = (
    df['response_1']
    .fillna("")    # Replace NaN with an empty string
    .astype(str)   # Ensure all values are strings
    .apply(lambda x: 1 if x.strip().lower() == "correct" else 0)
)

# 3. Push the pandas DataFrame into R using the new localconverter method
with localconverter(default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)
ro.globalenv['df_R'] = r_df

# 4. Fit Mixed-Effects Models in R

# Logistic Mixed Model (Accuracy)
model_acc_code = """
model_acc <- glmer(accuracy ~ experimental_condition + (experimental_condition | participant_identification) + (1 | condition_id),
                   data = df_R,
                   family = 'binomial')
cat('--- Logistic Mixed Model for Accuracy ---\\n')
print(summary(model_acc))

cat('\\n--- Pairwise Comparisons (emmeans) for Accuracy ---\\n')
acc_emm <- emmeans(model_acc, pairwise ~ experimental_condition, type='response')
print(acc_emm)
"""

# Linear Mixed Model (Reaction Time)
model_rt_code = """
model_rt <- lmer(reaction_time_1 ~ experimental_condition + (experimental_condition | participant_identification) + (1 | condition_id),
                 data = df_R)
cat('\\n--- Linear Mixed Model for Reaction Time ---\\n')
print(summary(model_rt))

cat('\\n--- Pairwise Comparisons (emmeans) for Reaction Time ---\\n')
rt_emm <- emmeans(model_rt, pairwise ~ experimental_condition)
print(rt_emm)
"""

# Execute the R code for mixed models
ro.r(model_acc_code)
ro.r(model_rt_code)

# 5. Aggregated Descriptive Statistics and Paired t-tests:
# Aggregate data per participant per condition and then reshape the data for paired t-tests.
t_test_code = """
cat('\\n=== Aggregated Descriptive Statistics and Paired t-tests ===\\n')

# Aggregate data: mean per participant per condition
agg_data <- aggregate(cbind(accuracy, reaction_time_1) ~ participant_identification + experimental_condition, 
                      data = df_R, FUN = mean)

# Reshape data to wide format using base R reshape()
agg_wide <- reshape(agg_data, timevar = "experimental_condition", idvar = "participant_identification", direction = "wide")

# Descriptive stats for Accuracy by condition
cat('\\n--- Descriptive Statistics for Accuracy (MCQ)---\\n')
acc_DFU <- agg_data[agg_data$experimental_condition == 'DFU', "accuracy"]
acc_F <- agg_data[agg_data$experimental_condition == 'F', "accuracy"]
cat('DFU: mean =', mean(acc_DFU), ', SD =', sd(acc_DFU), ', min =', min(acc_DFU), ', max =', max(acc_DFU), '\\n')
cat('F: mean =', mean(acc_F), ', SD =', sd(acc_F), ', min =', min(acc_F), ', max =', max(acc_F), '\\n')

# Paired t-test for Accuracy using reshaped data
cat('\\n--- Paired t-test for Accuracy ---\\n')
t_test_acc <- t.test(agg_wide$accuracy.DFU, agg_wide$accuracy.F, paired = TRUE)
print(t_test_acc)

# Descriptive stats for Reaction Time by condition
cat('\\n--- Descriptive Statistics for Reaction Time (MCQ)---\\n')
rt_DFU <- agg_data[agg_data$experimental_condition == 'DFU', "reaction_time_1"]
rt_F <- agg_data[agg_data$experimental_condition == 'F', "reaction_time_1"]
cat('DFU: mean =', mean(rt_DFU), ', SD =', sd(rt_DFU), ', min =', min(rt_DFU), ', max =', max(rt_DFU), '\\n')
cat('F: mean =', mean(rt_F), ', SD =', sd(rt_F), ', min =', min(rt_F), ', max =', max(rt_F), '\\n')

# Paired t-test for Reaction Time using reshaped data
cat('\\n--- Paired t-test for Reaction Time ---\\n')
t_test_rt <- t.test(agg_wide$reaction_time_1.DFU, agg_wide$reaction_time_1.F, paired = TRUE)
print(t_test_rt)
"""

# Execute the R code for aggregated descriptive stats and paired t-tests
ro.r(t_test_code)
