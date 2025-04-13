import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python data conversions
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')  # For linear mixed models
base = importr('base')          # Base R
stats = importr('stats')        # For statistical tests
utils = importr('utils')        # For read.csv, etc.
car = importr('car')            # For Type III ANOVA

# Load additional R libraries
ro.r('library(ggplot2)')
ro.r('library(emmeans)')
ro.r('library(effectsize)')
ro.r('library(dplyr)')

def analyze_interaction(file_path):
    """
    This script reads the CSV file, applies exclusion logic to remove all trials before the first correct
    MCQ response in each block, and then tests the interaction between experimental_condition and trial_index
    on three dependent variables: score_TEST, reaction_time_TEST, and BI_score.

    For each measure, two models are fit (with and without the interaction term) using sum contrasts
    (i.e., contr.sum for factors and contr.poly for ordered factors) to properly test the interaction effect.
    Type III ANOVAs are performed on the interaction models and the models are compared.
    Additionally, a summary table is printed for each measure showing the difference in means (DFU minus F),
    along with the number of items (N) for each trial_index level (1 to 4). A clearly structured output is produced.
    """
    # Set sum contrasts for factors (important for proper testing of interactions)
    ro.r('options(contrasts = c("contr.sum", "contr.poly"))')

    # --- Read data and apply exclusion logic ---
    ro.r(f"df <- read.csv('{file_path}', header=TRUE, stringsAsFactors=FALSE)")
    ro.r("df <- na.omit(df)")  # Remove rows with any NA

    # Convert necessary columns to factors
    ro.r('df$participant_identification <- as.factor(df$participant_identification)')
    ro.r('df$experimental_condition <- as.factor(df$experimental_condition)')
    ro.r('df$block_number <- as.factor(df$block_number)')
    ro.r('df$task_type <- as.factor(df$task_type)')
    ro.r('df$task_id <- as.factor(df$task_id)')

    # --- Exclusion Logic: Remove trials until the first correct MCQ response in each block ---
    ro.r('''
        # Add a trial sequence number within each block and compute cumulative correct responses
        df_with_seq <- df %>%
            group_by(participant_identification, block_number) %>%
            mutate(trial_in_block = row_number(),
                   cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
            ungroup()
        # Exclude trials until the first correct answer in each block (keep trials with cum_correct > 0)
        df <- df_with_seq %>% filter(cum_correct > 0)
    ''')

    # --- New Interaction Analysis Script ---
    ro.r('''
        library(dplyr)
        library(car)  # For Type III ANOVA

        #################################################################
        # ANALYSES ON FULL FILTERED DATASET
        #################################################################
        cat("\n================== FULL DATASET ANALYSES ==================\n")

        # Define trial_index within each block (ensure values 1 to 4)
        df_filtered <- df %>%
          group_by(participant_identification, block_number) %>%
          mutate(trial_index = row_number()) %>%
          ungroup() %>%
          filter(trial_index <= 4)

        ##############################
        # score_TEST Models
        ##############################
        cat("\n--- score_TEST Models ---\n")
        # Print the summary table ("4x4 nibble") for trial_index: means, diff, and N
        cat("\n[Summary Table] Difference in means (DFU - F) for score_TEST by trial_index:\n")
        score_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(score_diff)

        cat("\n[Interaction Model] Formula: score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id)\n")
        model_score_inter <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                              (1|participant_identification) + (1|task_id),
                                            data = df_filtered)
        print(summary(model_score_inter))
        cat("\nType III ANOVA (Interaction Model):\n")
        print(car::Anova(model_score_inter, type="III"))

        cat("\n[No Interaction Model] Formula: score_TEST ~ experimental_condition + trial_index + (1|participant_identification) + (1|task_id)\n")
        model_score_nointer <- lmerTest::lmer(score_TEST ~ experimental_condition + trial_index +
                                                (1|participant_identification) + (1|task_id),
                                              data = df_filtered)
        print(summary(model_score_nointer))
        cat("\nModel Comparison (score_TEST):\n")
        print(anova(model_score_nointer, model_score_inter))

        ##############################
        # reaction_time_TEST Models
        ##############################
        cat("\n--- reaction_time_TEST Models ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for reaction_time_TEST by trial_index:\n")
        rt_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(rt_diff)

        cat("\n[Interaction Model] Formula: reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id)\n")
        model_rt_inter <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                           (1|participant_identification) + (1|task_id),
                                         data = df_filtered)
        print(summary(model_rt_inter))
        cat("\nType III ANOVA (Interaction Model):\n")
        print(car::Anova(model_rt_inter, type="III"))

        cat("\n[No Interaction Model] Formula: reaction_time_TEST ~ experimental_condition + trial_index + (1|participant_identification) + (1|task_id)\n")
        model_rt_nointer <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition + trial_index +
                                             (1|participant_identification) + (1|task_id),
                                           data = df_filtered)
        print(summary(model_rt_nointer))
        cat("\nModel Comparison (reaction_time_TEST):\n")
        print(anova(model_rt_nointer, model_rt_inter))

        ##############################
        # BI_score Models
        ##############################
        cat("\n--- BI_score Models ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for BI_score by trial_index:\n")
        bi_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(bi_diff)

        cat("\n[Interaction Model] Formula: BI_score ~ experimental_condition * trial_index + (1|task_id)\n")
        model_bi_inter <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                           (1|task_id),
                                         data = df_filtered)
        print(summary(model_bi_inter))
        cat("\nType III ANOVA (Interaction Model):\n")
        print(car::Anova(model_bi_inter, type="III"))

        cat("\n[No Interaction Model] Formula: BI_score ~ experimental_condition + trial_index + (1|task_id)\n")
        model_bi_nointer <- lmerTest::lmer(BI_score ~ experimental_condition + trial_index +
                                             (1|task_id),
                                           data = df_filtered)
        print(summary(model_bi_nointer))
        cat("\nModel Comparison (BI_score):\n")
        print(anova(model_bi_nointer, model_bi_inter))

        #################################################################
        # ANALYSES ON RAT SUBSET
        #################################################################
        cat("\n================== RAT SUBSET ANALYSES ==================\n")
        df_RAT <- df_filtered %>% filter(task_type == "RAT")

        cat("\n--- score_TEST Models (RAT) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for score_TEST by trial_index (RAT):\n")
        score_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(score_diff_RAT)

        cat("\n[Interaction Model] Formula: score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) (RAT)\n")
        model_score_trial_RAT <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                                  (1|participant_identification) + (1|task_id),
                                                data = df_RAT)
        print(summary(model_score_trial_RAT))
        cat("\nType III ANOVA (RAT, score_TEST):\n")
        print(car::Anova(model_score_trial_RAT, type="III"))

        cat("\n--- reaction_time_TEST Models (RAT) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for reaction_time_TEST by trial_index (RAT):\n")
        rt_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(rt_diff_RAT)

        cat("\n[Interaction Model] Formula: reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) (RAT)\n")
        model_rt_trial_RAT <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                               (1|participant_identification) + (1|task_id),
                                             data = df_RAT)
        print(summary(model_rt_trial_RAT))
        cat("\nType III ANOVA (RAT, reaction_time_TEST):\n")
        print(car::Anova(model_rt_trial_RAT, type="III"))

        cat("\n--- BI_score Models (RAT) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for BI_score by trial_index (RAT):\n")
        bi_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(bi_diff_RAT)

        cat("\n[Interaction Model] Formula: BI_score ~ experimental_condition * trial_index + (1|task_id) (RAT)\n")
        model_bi_trial_RAT <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                               (1|task_id),
                                             data = df_RAT)
        print(summary(model_bi_trial_RAT))
        cat("\nType III ANOVA (RAT, BI_score):\n")
        print(car::Anova(model_bi_trial_RAT, type="III"))

        #################################################################
        # ANALYSES ON INSIGHT SUBSET
        #################################################################
        cat("\n================== INSIGHT SUBSET ANALYSES ==================\n")
        df_Insight <- df_filtered %>% filter(task_type == "Insight")

        cat("\n--- score_TEST Models (Insight) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for score_TEST by trial_index (Insight):\n")
        score_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(score_diff_Insight)

        cat("\n[Interaction Model] Formula: score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) (Insight)\n")
        model_score_trial_Insight <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                                      (1|participant_identification) + (1|task_id),
                                                    data = df_Insight)
        print(summary(model_score_trial_Insight))
        cat("\nType III ANOVA (Insight, score_TEST):\n")
        print(car::Anova(model_score_trial_Insight, type="III"))

        cat("\n--- reaction_time_TEST Models (Insight) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for reaction_time_TEST by trial_index (Insight):\n")
        rt_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(rt_diff_Insight)

        cat("\n[Interaction Model] Formula: reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) (Insight)\n")
        model_rt_trial_Insight <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                                   (1|participant_identification) + (1|task_id),
                                                 data = df_Insight)
        print(summary(model_rt_trial_Insight))
        cat("\nType III ANOVA (Insight, reaction_time_TEST):\n")
        print(car::Anova(model_rt_trial_Insight, type="III"))

        cat("\n--- BI_score Models (Insight) ---\n")
        cat("\n[Summary Table] Difference in means (DFU - F) for BI_score by trial_index (Insight):\n")
        bi_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(N = n(),
                    mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        print(bi_diff_Insight)

        cat("\n[Interaction Model] Formula: BI_score ~ experimental_condition * trial_index + (1|task_id) (Insight)\n")
        model_bi_trial_Insight <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                                   (1|task_id),
                                                 data = df_Insight)
        print(summary(model_bi_trial_Insight))
        cat("\nType III ANOVA (Insight, BI_score):\n")
        print(car::Anova(model_bi_trial_Insight, type="III"))
    ''')

if __name__ == "__main__":
    csv_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/"
        "School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/"
        "RPEP_experiment/data/preprocessed/concatenated_data.csv"
    )
    analyze_interaction(csv_path)
