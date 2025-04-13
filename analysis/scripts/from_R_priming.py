import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python data conversions
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')  # For linear mixed models
lme4 = importr('lme4')  # In case glmer is needed
base = importr('base')  # Base R functions
stats = importr('stats')  # For statistical tests and anova
utils = importr('utils')  # For data import utilities

ro.r("""

    library(dplyr)
    library(lme4)  # Provides glmer and lmer functions
    
    # Read the data
    file_path <- "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
    df <- read.csv(file_path)
    
    # Ensure experimental_condition is a factor with two levels ("DFU" and "F")
    df$experimental_condition <- factor(df$experimental_condition, levels = c("DFU", "F"))
    
    # Add trial sequence number within each block and compute cumulative correct responses
    df_with_seq <- df %>%
      group_by(participant_identification, block_number) %>%
      mutate(trial_in_block = row_number(),
             cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
      ungroup()
    
    # Exclude trials until the first correct answer is given in each block
    df_filtered <- df_with_seq %>% filter(cum_correct > 0)
    
    cat("Total trials in original dataset:", nrow(df), "\n")
    cat("Trials kept after exclusion:", nrow(df_filtered), "\n")
    cat("Trials excluded:", nrow(df) - nrow(df_filtered), "\n")
    
    # Create binary variable score_TEST_bin (1 if score_TEST >= 0.5, else 0)
    df_filtered$score_TEST_bin <- ifelse(df_filtered$score_TEST >= 0.5, 1, 0)
    
    # Create subsets for RAT and Insight tasks
    df_rat <- df_filtered %>% filter(task_type == "RAT")
    df_insight <- df_filtered %>% filter(task_type == "Insight")
    
    #############################
    # FULL FILTERED DATASET ANALYSIS
    #############################
    cat("\n\n----- FULL FILTERED DATASET ANALYSIS -----\n")
    
    # --- score_TEST_bin Analysis (Descriptive Stats & Logistic Mixed Effects Model)
    cat("\nscore_TEST_bin Descriptive Statistics (Full Filtered Dataset):\n")
    df_desc_score_bin_full <- aggregate(df_filtered$score_TEST_bin,
                                        by = list(df_filtered$experimental_condition),
                                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_score_bin_full) <- c("Condition", "score_TEST_bin_stats")
    print(df_desc_score_bin_full)
    
    cat("\nMixed Logistic Model for score_TEST_bin (Full Filtered Dataset):\n")
    model_score_bin_full <- glmer(score_TEST_bin ~ experimental_condition + task_type + experimental_condition:task_type +
                                    (experimental_condition + task_type + experimental_condition:task_type | participant_identification) +
                                    (1 | task_id),
                                  data = df_filtered, family = binomial)
    print(summary(model_score_bin_full))
    
    # --- reaction_time_TEST Analysis (Descriptive Stats & LME)
    cat("\nreaction_time_TEST Descriptive Statistics (Full Filtered Dataset):\n")
    df_desc_rt_full <- aggregate(df_filtered$reaction_time_TEST,
                                 by = list(df_filtered$experimental_condition),
                                 FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_rt_full) <- c("Condition", "reaction_time_TEST_stats")
    print(df_desc_rt_full)
    
    cat("\nMixed Linear Model for reaction_time_TEST (Full Filtered Dataset):\n")
    model_rt_full <- lmer(reaction_time_TEST ~ experimental_condition + task_type + experimental_condition:task_type +
                            (experimental_condition + task_type + experimental_condition:task_type | participant_identification) +
                            (1 | task_id),
                          data = df_filtered)
    print(summary(model_rt_full))
    
    # --- BI_score Analysis (Descriptive Stats & LME)
    cat("\nBI_score Descriptive Statistics (Full Filtered Dataset):\n")
    df_desc_bi_full <- aggregate(df_filtered$BI_score_TaskType_ParticipantID,
                                 by = list(df_filtered$experimental_condition),
                                 FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_bi_full) <- c("Condition", "BI_score_stats")
    print(df_desc_bi_full)
    
    cat("\nMixed Linear Model for BI_score (Full Filtered Dataset):\n")
    model_bi_full <- lmer(BI_score_TaskType_ParticipantID ~ experimental_condition +
                            (experimental_condition | participant_identification) +
                            (1 | task_id),
                          data = df_filtered)
    print(summary(model_bi_full))
    
    #############################
    # RAT FILTERED DATASET ANALYSIS
    #############################
    cat("\n\n----- RAT FILTERED DATASET ANALYSIS -----\n")
    
    # --- score_TEST_bin Analysis (Descriptive Stats & Logistic Mixed Effects Model)
    cat("\nscore_TEST_bin Descriptive Statistics (RAT Filtered Dataset):\n")
    df_desc_score_bin_rat <- aggregate(df_rat$score_TEST_bin,
                                       by = list(df_rat$experimental_condition),
                                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_score_bin_rat) <- c("Condition", "score_TEST_bin_stats")
    print(df_desc_score_bin_rat)
    
    cat("\nMixed Logistic Model for score_TEST_bin (RAT Filtered Dataset):\n")
    model_score_bin_rat <- glmer(score_TEST_bin ~ experimental_condition +
                                   (experimental_condition | participant_identification) +
                                   (1 | task_id),
                                 data = df_rat, family = binomial)
    print(summary(model_score_bin_rat))
    
    # --- reaction_time_TEST Analysis (Descriptive Stats & LME)
    cat("\nreaction_time_TEST Descriptive Statistics (RAT Filtered Dataset):\n")
    df_desc_rt_rat <- aggregate(df_rat$reaction_time_TEST,
                                by = list(df_rat$experimental_condition),
                                FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_rt_rat) <- c("Condition", "reaction_time_TEST_stats")
    print(df_desc_rt_rat)
    
    cat("\nMixed Linear Model for reaction_time_TEST (RAT Filtered Dataset):\n")
    model_rt_rat <- lmer(reaction_time_TEST ~ experimental_condition +
                           (1 | participant_identification) +
                           (1 | task_id),
                         data = df_rat)
    print(summary(model_rt_rat))
    
    # --- BI_score Analysis (Descriptive Stats & LME)
    cat("\nBI_score Descriptive Statistics (RAT Filtered Dataset):\n")
    df_desc_bi_rat <- aggregate(df_rat$BI_score_TaskType_ParticipantID,
                                by = list(df_rat$experimental_condition),
                                FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_bi_rat) <- c("Condition", "BI_score_stats")
    print(df_desc_bi_rat)
    
    cat("\nMixed Linear Model for BI_score (RAT Filtered Dataset):\n")
    model_bi_rat <- lmer(BI_score_TaskType_ParticipantID ~ experimental_condition +
                           (experimental_condition | participant_identification) +
                           (1 | task_id),
                         data = df_rat)
    print(summary(model_bi_rat))
    
    #############################
    # INSIGHT FILTERED DATASET ANALYSIS
    #############################
    cat("\n\n----- INSIGHT FILTERED DATASET ANALYSIS -----\n")
    
    # --- score_TEST_bin Analysis (Descriptive Stats & Logistic Mixed Effects Model)
    cat("\nscore_TEST_bin Descriptive Statistics (Insight Filtered Dataset):\n")
    df_desc_score_bin_insight <- aggregate(df_insight$score_TEST_bin,
                                           by = list(df_insight$experimental_condition),
                                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_score_bin_insight) <- c("Condition", "score_TEST_bin_stats")
    print(df_desc_score_bin_insight)
    
    cat("\nMixed Logistic Model for score_TEST_bin (Insight Filtered Dataset):\n")
    model_score_bin_insight <- glmer(score_TEST_bin ~ experimental_condition +
                                       (experimental_condition | participant_identification) +
                                       (1 | task_id),
                                     data = df_insight, family = binomial)
    print(summary(model_score_bin_insight))
    
    # --- reaction_time_TEST Analysis (Descriptive Stats & LME)
    cat("\nreaction_time_TEST Descriptive Statistics (Insight Filtered Dataset):\n")
    df_desc_rt_insight <- aggregate(df_insight$reaction_time_TEST,
                                    by = list(df_insight$experimental_condition),
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_rt_insight) <- c("Condition", "reaction_time_TEST_stats")
    print(df_desc_rt_insight)
    
    cat("\nMixed Linear Model for reaction_time_TEST (Insight Filtered Dataset):\n")
    model_rt_insight <- lmer(reaction_time_TEST ~ experimental_condition +
                               (experimental_condition | participant_identification) +
                               (1 | task_id),
                             data = df_insight)
    print(summary(model_rt_insight))
    
    # --- BI_score Analysis (Descriptive Stats & LME)
    cat("\nBI_score Descriptive Statistics (Insight Filtered Dataset):\n")
    df_desc_bi_insight <- aggregate(df_insight$BI_score_TaskType_ParticipantID,
                                    by = list(df_insight$experimental_condition),
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
    colnames(df_desc_bi_insight) <- c("Condition", "BI_score_stats")
    print(df_desc_bi_insight)
    
    cat("\nMixed Linear Model for BI_score (Insight Filtered Dataset):\n")
    model_bi_insight <- lmer(BI_score_TaskType_ParticipantID ~ experimental_condition +
                               (experimental_condition | participant_identification) +
                               (1 | task_id),
                             data = df_insight)
    print(summary(model_bi_insight))

    """)
