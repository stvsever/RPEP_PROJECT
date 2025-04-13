import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python conversions
pandas2ri.activate()

# Import required R packages
stats = importr('stats')
base = importr('base')
utils = importr('utils')
ro.r("library(dplyr)")

def paired_ttest(file_path):
    # Read CSV and remove rows with NA
    ro.r(f"df <- read.csv('{file_path}', header=TRUE, stringsAsFactors=FALSE)")
    ro.r("df <- na.omit(df)")

    # Convert relevant columns to factors
    ro.r("df$participant_identification <- as.factor(df$participant_identification)")
    ro.r("df$experimental_condition <- as.factor(df$experimental_condition)")

    # -----------------------------
    # Analysis by participant (aggregated within task type) on full dataset
    # -----------------------------
    ro.r('''
        cat("\n=== Analysis by participant (aggregated within task type) - Full Dataset ===\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df, task_type == task)

            ## SCORE_TEST analysis
            cat("\nScore_TEST averages and SD by condition:\n")
            summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                       data = df_sub, 
                                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_score)

            # Aggregate by participant for paired analysis
            df_score <- aggregate(score_TEST ~ participant_identification + experimental_condition, 
                                  data = df_sub, FUN = mean)
            df_score_wide <- reshape(df_score, 
                                     idvar = "participant_identification", 
                                     timevar = "experimental_condition", 
                                     direction = "wide")
            if (all(c("score_TEST.DFU", "score_TEST.F") %in% colnames(df_score_wide))) {
                x <- na.omit(df_score_wide$`score_TEST.DFU`)
                y <- na.omit(df_score_wide$`score_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough paired observations for Score_TEST (DFU vs F).\n")
                } else {
                    cat("\nPaired t-test for Score_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = TRUE))
                }
            } else {
                cat("\nNot enough data for paired t-test for Score_TEST (DFU vs F).\n")
            }

            ## Reaction_time_TEST analysis
            cat("\nReaction_time_TEST averages and SD by condition:\n")
            summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                    data = df_sub, 
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_rt)

            # Aggregate by participant for paired analysis
            df_rt <- aggregate(reaction_time_TEST ~ participant_identification + experimental_condition, 
                               data = df_sub, FUN = mean)
            df_rt_wide <- reshape(df_rt, 
                                  idvar = "participant_identification", 
                                  timevar = "experimental_condition", 
                                  direction = "wide")
            if (all(c("reaction_time_TEST.DFU", "reaction_time_TEST.F") %in% colnames(df_rt_wide))) {
                x <- na.omit(df_rt_wide$`reaction_time_TEST.DFU`)
                y <- na.omit(df_rt_wide$`reaction_time_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough paired observations for Reaction_time_TEST (DFU vs F).\n")
                } else {
                    cat("\nPaired t-test for Reaction_time_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = TRUE))
                }
            } else {
                cat("\nNot enough data for paired t-test for Reaction_time_TEST (DFU vs F).\n")
            }
        }
    ''')

    # -----------------------------
    # Analysis by task_id (within each task type) – using unpaired test on full dataset
    # -----------------------------
    ro.r('''
        cat("\n=== Analysis by task_id (within task type) - Full Dataset ===\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df, task_type == task)
            unique_tasks <- unique(df_sub$task_id)
            for (t_id in unique_tasks) {
                cat("\n--- Task_id:", t_id, "---\n")
                df_item <- subset(df_sub, task_id == t_id)

                ## SCORE_TEST analysis for task_id
                cat("\nScore_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                           data = df_item, 
                                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_score)

                # Extract vectors for DFU and F (unpaired)
                x <- df_item$score_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$score_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Score_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }

                ## Reaction_time_TEST analysis for task_id
                cat("\nReaction_time_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                        data = df_item, 
                                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_rt)
                x <- df_item$reaction_time_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$reaction_time_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }
            }
        }
    ''')

    # ============================================================================
    # Additional Analysis Part 1: Using only trials where the answer was correct
    # For this additional analysis, use unpaired t-tests to accommodate different lengths
    # ============================================================================
    ro.r('''
        cat("\n=== Analysis on Dataset with Only Correct Trials ===\n")
        # Filter the dataset to include only trials with a correct response
        df_correct <- subset(df, MCQ_response_unprocessed == 'correct')
        cat("Total trials in full dataset:", nrow(df), "\n")
        cat("Trials with correct response:", nrow(df_correct), "\n")
        cat("Trials excluded (incorrect):", nrow(df) - nrow(df_correct), "\n")

        # Analysis by participant (aggregated within task type) on correct trials
        cat("\n--- Analysis by participant (aggregated within task type) - Correct Trials ---\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_correct, task_type == task)

            ## SCORE_TEST analysis
            cat("\nScore_TEST averages and SD by condition:\n")
            summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                       data = df_sub, 
                                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_score)

            # Aggregate by participant for unpaired analysis (to handle unequal lengths)
            df_score <- aggregate(score_TEST ~ participant_identification + experimental_condition, 
                                  data = df_sub, FUN = mean)
            df_score_wide <- reshape(df_score, 
                                     idvar = "participant_identification", 
                                     timevar = "experimental_condition", 
                                     direction = "wide")
            if (all(c("score_TEST.DFU", "score_TEST.F") %in% colnames(df_score_wide))) {
                x <- na.omit(df_score_wide$`score_TEST.DFU`)
                y <- na.omit(df_score_wide$`score_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Score_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Score_TEST (DFU vs F).\n")
            }

            ## Reaction_time_TEST analysis
            cat("\nReaction_time_TEST averages and SD by condition:\n")
            summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                    data = df_sub, 
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_rt)

            # Aggregate by participant for unpaired analysis
            df_rt <- aggregate(reaction_time_TEST ~ participant_identification + experimental_condition, 
                               data = df_sub, FUN = mean)
            df_rt_wide <- reshape(df_rt, 
                                  idvar = "participant_identification", 
                                  timevar = "experimental_condition", 
                                  direction = "wide")
            if (all(c("reaction_time_TEST.DFU", "reaction_time_TEST.F") %in% colnames(df_rt_wide))) {
                x <- na.omit(df_rt_wide$`reaction_time_TEST.DFU`)
                y <- na.omit(df_rt_wide$`reaction_time_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Reaction_time_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Reaction_time_TEST (DFU vs F).\n")
            }
        }

        # Analysis by task_id (within each task type) on correct trials
        cat("\n=== Analysis by task_id (within task type) - Correct Trials ===\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_correct, task_type == task)
            unique_tasks <- unique(df_sub$task_id)
            for (t_id in unique_tasks) {
                cat("\n--- Task_id:", t_id, "---\n")
                df_item <- subset(df_sub, task_id == t_id)

                ## SCORE_TEST analysis for task_id
                cat("\nScore_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                           data = df_item, 
                                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_score)

                x <- df_item$score_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$score_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Score_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }

                ## Reaction_time_TEST analysis for task_id
                cat("\nReaction_time_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                        data = df_item, 
                                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_rt)
                x <- df_item$reaction_time_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$reaction_time_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }
            }
        }
    ''')

    # ============================================================================
    # Additional Analysis Part 2: Using only trials from blocks that contained at least one correct response
    # For this additional analysis, also use unpaired t-tests
    # ============================================================================
    ro.r('''
        cat("\n=== Analysis on Dataset with Blocks Having At Least One Correct Response ===\n")
        # Create a flag for correct response
        df$correct_flag <- ifelse(df$MCQ_response_unprocessed == 'correct', 1, 0)
        # Summarize by participant and block to count correct responses in each block
        block_summary <- aggregate(correct_flag ~ participant_identification + block_number, 
                                   data = df, FUN = sum)
        # Identify valid blocks: those with at least one correct trial
        valid_blocks <- subset(block_summary, correct_flag >= 1)
        # Merge to keep only trials from valid blocks
        df_valid_blocks <- merge(df, valid_blocks[, c('participant_identification', 'block_number')], 
                                 by = c('participant_identification', 'block_number'))
        cat("Total trials in full dataset:", nrow(df), "\n")
        cat("Trials in valid blocks:", nrow(df_valid_blocks), "\n")
        cat("Trials excluded (blocks with no correct):", nrow(df) - nrow(df_valid_blocks), "\n")

        # Analysis by participant (aggregated within task type) on valid blocks dataset
        cat("\n--- Analysis by participant (aggregated within task type) - Valid Blocks ---\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_valid_blocks, task_type == task)

            ## SCORE_TEST analysis
            cat("\nScore_TEST averages and SD by condition:\n")
            summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                       data = df_sub, 
                                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_score)

            # Aggregate by participant for unpaired analysis
            df_score <- aggregate(score_TEST ~ participant_identification + experimental_condition, 
                                  data = df_sub, FUN = mean)
            df_score_wide <- reshape(df_score, 
                                     idvar = "participant_identification", 
                                     timevar = "experimental_condition", 
                                     direction = "wide")
            if (all(c("score_TEST.DFU", "score_TEST.F") %in% colnames(df_score_wide))) {
                x <- na.omit(df_score_wide$`score_TEST.DFU`)
                y <- na.omit(df_score_wide$`score_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Score_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Score_TEST (DFU vs F).\n")
            }

            ## Reaction_time_TEST analysis
            cat("\nReaction_time_TEST averages and SD by condition:\n")
            summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                    data = df_sub, 
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_rt)

            # Aggregate by participant for unpaired analysis
            df_rt <- aggregate(reaction_time_TEST ~ participant_identification + experimental_condition, 
                               data = df_sub, FUN = mean)
            df_rt_wide <- reshape(df_rt, 
                                  idvar = "participant_identification", 
                                  timevar = "experimental_condition", 
                                  direction = "wide")
            if (all(c("reaction_time_TEST.DFU", "reaction_time_TEST.F") %in% colnames(df_rt_wide))) {
                x <- na.omit(df_rt_wide$`reaction_time_TEST.DFU`)
                y <- na.omit(df_rt_wide$`reaction_time_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Reaction_time_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Reaction_time_TEST (DFU vs F).\n")
            }
        }

        # Analysis by task_id (within each task type) on valid blocks dataset
        cat("\n=== Analysis by task_id (within task type) - Valid Blocks ===\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_valid_blocks, task_type == task)
            unique_tasks <- unique(df_sub$task_id)
            for (t_id in unique_tasks) {
                cat("\n--- Task_id:", t_id, "---\n")
                df_item <- subset(df_sub, task_id == t_id)

                ## SCORE_TEST analysis for task_id
                cat("\nScore_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_score <- aggregate(score_TEST ~ experimental_condition, 
                                           data = df_item, 
                                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_score)

                x <- df_item$score_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$score_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Score_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }

                ## Reaction_time_TEST analysis for task_id
                cat("\nReaction_time_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition, 
                                        data = df_item, 
                                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_rt)
                x <- df_item$reaction_time_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$reaction_time_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }
            }
        }
    ''')

    ro.r('''
        cat("\n=== Additional Analysis Part 3: Exclude Trials Until First Correct Answer in Each Block ===\n")
        library(dplyr)
        # Add a trial sequence number within each block and compute cumulative correct responses
        df_with_seq <- df %>%
            group_by(participant_identification, block_number) %>%
            mutate(trial_in_block = row_number(),
                   cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
            ungroup()

        # Keep only trials that occur at or after the first correct trial in each block
        df_filtered <- df_with_seq %>% filter(cum_correct > 0)

        cat("Total trials in full dataset:", nrow(df), "\n")
        cat("Trials kept after excluding trials before the first correct in each block:", nrow(df_filtered), "\n")
        cat("Trials excluded:", nrow(df) - nrow(df_filtered), "\n")

        # ============================================================================
        # Now perform the full analysis (by participant and by task_id) on the filtered dataset
        # ============================================================================

        # Analysis by participant (aggregated within task type) on filtered dataset
        cat("\n--- Analysis by participant (aggregated within task type) - Filtered Dataset ---\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_filtered, task_type == task)

            ## SCORE_TEST analysis
            cat("\nScore_TEST averages and SD by condition:\n")
            summary_score <- aggregate(score_TEST ~ experimental_condition,
                                       data = df_sub,
                                       FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_score)

            # Aggregate by participant for unpaired analysis
            df_score <- aggregate(score_TEST ~ participant_identification + experimental_condition,
                                  data = df_sub, FUN = mean)
            df_score_wide <- reshape(df_score,
                                     idvar = "participant_identification",
                                     timevar = "experimental_condition",
                                     direction = "wide")
            if (all(c("score_TEST.DFU", "score_TEST.F") %in% colnames(df_score_wide))) {
                x <- na.omit(df_score_wide$`score_TEST.DFU`)
                y <- na.omit(df_score_wide$`score_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Score_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Score_TEST (DFU vs F).\n")
            }

            ## Reaction_time_TEST analysis
            cat("\nReaction_time_TEST averages and SD by condition:\n")
            summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition,
                                    data = df_sub,
                                    FUN = function(x) c(mean = mean(x), sd = sd(x)))
            print(summary_rt)

            # Aggregate by participant for unpaired analysis
            df_rt <- aggregate(reaction_time_TEST ~ participant_identification + experimental_condition,
                               data = df_sub, FUN = mean)
            df_rt_wide <- reshape(df_rt,
                                  idvar = "participant_identification",
                                  timevar = "experimental_condition",
                                  direction = "wide")
            if (all(c("reaction_time_TEST.DFU", "reaction_time_TEST.F") %in% colnames(df_rt_wide))) {
                x <- na.omit(df_rt_wide$`reaction_time_TEST.DFU`)
                y <- na.omit(df_rt_wide$`reaction_time_TEST.F`)
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F).\n")
                } else {
                    cat("\nUnpaired t-test for Reaction_time_TEST (DFU vs F):\n")
                    print(t.test(x, y, paired = FALSE))
                }
            } else {
                cat("\nNot enough data for t-test for Reaction_time_TEST (DFU vs F).\n")
            }
        }

        # Analysis by task_id (within each task type) on filtered dataset
        cat("\n=== Analysis by task_id (within task type) - Filtered Dataset ===\n")
        for (task in c("RAT", "Insight")) {
            cat("\n------ Task type:", task, "------\n")
            df_sub <- subset(df_filtered, task_type == task)
            unique_tasks <- unique(df_sub$task_id)
            for (t_id in unique_tasks) {
                cat("\n--- Task_id:", t_id, "---\n")
                df_item <- subset(df_sub, task_id == t_id)

                ## SCORE_TEST analysis for task_id
                cat("\nScore_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_score <- aggregate(score_TEST ~ experimental_condition,
                                           data = df_item,
                                           FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_score)

                x <- df_item$score_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$score_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Score_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Score_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }

                ## Reaction_time_TEST analysis for task_id
                cat("\nReaction_time_TEST averages and SD by condition for task_id ", t_id, ":\n", sep = "")
                summary_rt <- aggregate(reaction_time_TEST ~ experimental_condition,
                                        data = df_item,
                                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
                print(summary_rt)
                x <- df_item$reaction_time_TEST[df_item$experimental_condition == 'DFU']
                y <- df_item$reaction_time_TEST[df_item$experimental_condition == 'F']
                if (length(x) < 2 | length(y) < 2) {
                    cat("\nNot enough observations for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ".\n", sep = "")
                } else {
                    cat("\nUnpaired t-test (Welch) for Reaction_time_TEST (DFU vs F) for task_id ", t_id, ":\n", sep = "")
                    print(t.test(x, y, paired = FALSE))
                }
            }
        }
    ''')

if __name__ == "__main__":
    csv_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
    paired_ttest(csv_path)
