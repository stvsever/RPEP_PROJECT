import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python data conversions
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')  # For linear mixed models
base = importr('base')  # Base R
stats = importr('stats')  # For cor.test, anova, etc.
utils = importr('utils')  # For read.csv, etc.

# We'll also load ggplot2, emmeans, effectsize in R
ro.r('library(ggplot2)')
ro.r('library(emmeans)')
ro.r('library(effectsize)')


def analyze_experiment(file_path):
    """
    1) Read CSV into R, remove missing rows.
    2) For RAT vs. Insight tasks:
       - Print mean/SD by experimental_condition & run mixed models (for score_MCQ, reaction_time_MCQ, score_TEST,
         reaction_time_TEST, affinity_score, and BI_score). Note that for BI_score the model uses (1|task_id) only.
       - Print the model formula in parentheses under each result.
       - For score_MCQ, reaction_time_MCQ, score_TEST, reaction_time_TEST, and BI_score: produce boxplots with
         p-value & Cohen's d annotation (if applicable).
       3) Correlate RAT vs. Insight average score_TEST and reaction_time_TEST.
       4) Avoid crashing if the model is singular or unidentifiable (skip model).
    """

    # Directory to save plots
    plot_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/experimental_manipulation"
    ro.r(f'plot_dir <- "{plot_dir}"')

    # --- 1) Read data into R, remove missing rows ---
    ro.r(f"df <- read.csv('{file_path}', header=TRUE, stringsAsFactors=FALSE)")
    ro.r("df <- na.omit(df)")  # remove rows with any NA

    # Print columns so you can confirm correct names
    ro.r("cat('Column names in df:\\n')")
    ro.r("print(colnames(df))")

    # Convert to factors
    ro.r('df$participant_identification <- as.factor(df$participant_identification)')
    ro.r('df$experimental_condition <- as.factor(df$experimental_condition)')
    ro.r('df$block_number <- as.factor(df$block_number)')
    ro.r('df$task_type <- as.factor(df$task_type)')
    ro.r('df$task_id <- as.factor(df$task_id)')

    # All measures of interest (for most analyses)
    all_measures = ["score_MCQ", "reaction_time_MCQ", "score_TEST", "reaction_time_TEST", "affinity_score"]

    ro.r("""
        library(dplyr)
        cat("\n=== FULL DATASET BLOCK NUMBER ANALYSIS WITH EXCLUSION LOGIC (Exclude Trials Until First Correct in Each Block) ===\n")
        
        # Add a trial sequence number within each block and compute cumulative correct responses
        df_with_seq <- df %>%
            group_by(participant_identification, block_number) %>%
            mutate(trial_in_block = row_number(),
                   cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
            ungroup()
        
        # Exclude trials until the first correct answer is given in each block
        df_filtered <- df_with_seq %>% filter(cum_correct > 0)
        
        cat("Total trials in original dataset:", nrow(df), "\n")
        cat("Trials kept after excluding trials until first correct in each block:", nrow(df_filtered), "\n")
        cat("Trials excluded:", nrow(df) - nrow(df_filtered), "\n")
        
        # Perform Block Number Analysis on the filtered (full) dataset
        measures <- c("score_MCQ", "reaction_time_MCQ", "score_TEST", "reaction_time_TEST", "affinity_score", "BI_score")
        for (m in measures) {
            cat(paste0("\nFull Dataset - ", m, " Descriptive Statistics by Block (Filtered):\n"))
            df_summary_block <- aggregate(df_filtered[[m]],
                                          by = list(df_filtered$block_number),
                                          FUN = function(x) c(mean = mean(x), sd = sd(x)))
            colnames(df_summary_block) <- c("Block", paste0(m, "_stats"))
            print(df_summary_block)
        
            # Build and print the mixed linear model for each measure
            if (m == "BI_score") {
                cat(paste0("\nFull Dataset - ", m, " Mixed Linear Model (Filtered):\n"))
                cat("(Model: ", m, " ~ block_number + (1|task_id) + (1|condition_item))\n")
                suppressWarnings(
                    tryCatch(
                        {
                            model <- lmerTest::lmer(as.formula(paste(m, "~ block_number + (1|task_id) + (1|condition_item)")),
                                                    data = df_filtered)
                            print(anova(model))
                        },
                        error = function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\n")
                        }
                    )
                )
            } else if (m %in% c("score_MCQ", "reaction_time_MCQ", "affinity_score")) {
                cat(paste0("\nFull Dataset - ", m, " Mixed Linear Model (Filtered):\n"))
                cat("(Model: ", m, " ~ block_number + (1|participant_identification) + (1|condition_item))\n")
                suppressWarnings(
                    tryCatch(
                        {
                            model <- lmerTest::lmer(as.formula(paste(m, "~ block_number + (1|participant_identification) + (1|condition_item)")),
                                                    data = df_filtered)
                            print(anova(model))
                        },
                        error = function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\n")
                        }
                    )
                )
            } else {
                cat(paste0("\nFull Dataset - ", m, " Mixed Linear Model (Filtered):\n"))
                cat("(Model: ", m, " ~ block_number + (1|participant_identification) + (1|task_id) + (1|condition_item))\n")
                suppressWarnings(
                    tryCatch(
                        {
                            model <- lmerTest::lmer(as.formula(paste(m, "~ block_number + (1|participant_identification) + (1|task_id) + (1|condition_item)")),
                                                    data = df_filtered)
                            print(anova(model))
                        },
                        error = function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\n")
                        }
                    )
                )
            }
            cat("-----------------------------------------------------\n")
        }
    """)

    print("\n--- LINEAR MIXED-EFFECTS ANALYSES (R via rpy2) ---\n")

    # We'll do two subsets: RAT and Insight
    for current_task_type in ["RAT", "Insight"]:
        print(f"\n===== ANALYSES FOR TASK TYPE: {current_task_type} =====\n")
        ro.r(f"df_sub <- subset(df, task_type == '{current_task_type}')")

        # If df_sub is empty, skip
        ro.r(f"""
            if (nrow(df_sub) == 0) {{
                cat("No rows found for task_type = '{current_task_type}'. Skipping.\\n")
                quit(status=0)
            }}
        """)

        # Check levels of participant_id and task_id
        ro.r("""
            n_participants <- length(unique(df_sub$participant_identification))
            n_items <- length(unique(df_sub$task_id))
        """)

        # === CONDITION ANALYSIS (F vs. DFU) for standard measures ===
        print("=== CONDITION ANALYSIS (F vs. DFU) for standard measures ===")
        for m in all_measures:
            ro.r(f"""
                # Print group means & SD
                df_summary_cond <- aggregate(df_sub${m},
                                             by=list(df_sub$experimental_condition),
                                             FUN=function(x) c(mean=mean(x), sd=sd(x)))
                colnames(df_summary_cond) <- c('Condition', '{m}')
                cat('\\nMeasure: {m} - Means & SD by Condition\\n')
                print(df_summary_cond)

                if (n_participants < 2 || n_items < 2) {{
                    cat("Skipping model for {m} due to only one level in participant/task_id.\\n")
                }} else {{
                    cat('\\n(Mixed linear model: {m} ~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item))\\n')
                    suppressWarnings(
                        tryCatch(
                            {{
                                model_cond <- lmerTest::lmer({m} ~ experimental_condition +
                                                                     (1|participant_identification) +
                                                                     (1|task_id) +
                                                                     (1|condition_item),
                                                            data=df_sub)
                                cat('ANOVA table:\\n')
                                print(anova(model_cond))
                            }},
                            error=function(e) {{
                                cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                            }}
                        )
                    )
                    cat('\\n----------------------------------------\\n')
                }}
            """)

        # === CONDITION ANALYSIS for BI_score (using only (1|task_id)) ===
        ro.r("""
            cat('\\nMeasure: BI_score - Means & SD by Condition\\n')
            df_summary_bi <- aggregate(df_sub$BI_score,
                                        by=list(df_sub$experimental_condition),
                                        FUN=function(x) c(mean=mean(x), sd=sd(x)))
            colnames(df_summary_bi) <- c('Condition', 'BI_score')
            print(df_summary_bi)

            if (n_items < 2) {
                cat("Skipping model for BI_score due to insufficient levels in task_id.\\n")
            } else {
                cat('\\n(Mixed linear model: BI_score ~ experimental_condition + (1|task_id) + (1|condition_item))\\n')
                suppressWarnings(
                    tryCatch(
                        {
                            model_cond_bi <- lmerTest::lmer(BI_score ~ experimental_condition + (1|task_id) + (1|condition_item),
                                                            data=df_sub)
                            cat('ANOVA table:\\n')
                            print(anova(model_cond_bi))
                        },
                        error=function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                        }
                    )
                )
                cat('\\n----------------------------------------\\n')
            }
        """)

        # === BLOCK NUMBER ANALYSIS for standard measures ===
        print("\n=== BLOCK NUMBER ANALYSIS for standard measures ===")
        for m in all_measures:
            ro.r(f"""
                df_summary_block <- aggregate(df_sub${m},
                                              by=list(df_sub$block_number),
                                              FUN=function(x) c(mean=mean(x), sd=sd(x)))
                colnames(df_summary_block) <- c('Block', '{m}')
                cat('\\nMeasure: {m} - Means & SD by Block\\n')
                print(df_summary_block)

                if (n_participants < 2 || n_items < 2) {{
                    cat("Skipping model for {m} due to only one level in participant/task_id.\\n")
                }} else {{
                    cat('\\n(Mixed linear model: {m} ~ block_number + (1|participant_identification) + (1|task_id) + (1|condition_item))\\n')
                    suppressWarnings(
                        tryCatch(
                            {{
                                model_block <- lmerTest::lmer({m} ~ block_number +
                                                                     (1|participant_identification) +
                                                                     (1|task_id) +
                                                                     (1|condition_item),
                                                            data=df_sub)
                                cat('ANOVA table:\\n')
                                print(anova(model_block))
                            }},
                            error=function(e) {{
                                cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                            }}
                        )
                    )
                    cat('\\n----------------------------------------\\n')
                }}
            """)

        # === BLOCK NUMBER ANALYSIS for BI_score (using only (1|task_id)) ===
        ro.r("""
            df_summary_block_bi <- aggregate(df_sub$BI_score,
                                             by=list(df_sub$block_number),
                                             FUN=function(x) c(mean=mean(x), sd=sd(x)))
            colnames(df_summary_block_bi) <- c('Block', 'BI_score')
            cat('\\nMeasure: BI_score - Means & SD by Block\\n')
            print(df_summary_block_bi)

            if (n_items < 2) {
                cat("Skipping model for BI_score due to insufficient levels in task_id.\\n")
            } else {
                cat('\\n(Mixed linear model: BI_score ~ block_number + (1|task_id) + (1|condition_item))\\n')
                suppressWarnings(
                    tryCatch(
                        {
                            model_block_bi <- lmerTest::lmer(BI_score ~ block_number + (1|task_id) + (1|condition_item),
                                                             data=df_sub)
                            cat('ANOVA table:\\n')
                            print(anova(model_block_bi))
                        },
                        error=function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                        }
                    )
                )
                cat('\\n----------------------------------------\\n')
            }
        """)

        # === Create boxplots for the 4 main measures of interest (standard measures) ===
        plot_measures = ["score_MCQ", "reaction_time_MCQ", "score_TEST", "reaction_time_TEST"]
        for pm in plot_measures:
            ro.r(f"""
                if (!("{pm}" %in% colnames(df_sub))) {{
                    cat("Column {pm} not found. Skipping plot.\\n")
                }} else {{
                    if (n_participants < 2 || n_items < 2) {{
                        cat("Skipping plot model for {pm} due to only one level in participant/task_id.\\n")
                    }} else {{
                        suppressWarnings(
                            tryCatch(
                                {{
                                    model_plot <- lmerTest::lmer({pm} ~ experimental_condition +
                                                                       (1|participant_identification) +
                                                                       (1|task_id) +
                                                                       (1|condition_item),
                                                               data=df_sub)

                                    posthoc_cond <- emmeans::emmeans(model_plot, specs = ~ experimental_condition)
                                    contr_cond   <- emmeans::contrast(posthoc_cond, method='revpairwise') 
                                    eff_cond <- emmeans::eff_size(posthoc_cond, sigma = sigma(model_plot), edf = df.residual(model_plot))
                                    contr_labels <- rownames(as.data.frame(eff_cond))
                                    d_value <- NA
                                    if ('DFU - F' %in% contr_labels) {{
                                        d_value <- as.data.frame(eff_cond)['DFU - F','effect.size']
                                    }} else if ('F - DFU' %in% contr_labels) {{
                                        d_value <- as.data.frame(eff_cond)['F - DFU','effect.size']
                                    }} else {{
                                        d_value <- as.data.frame(eff_cond)[1,'effect.size']
                                    }}

                                    contr_data <- as.data.frame(contr_cond)
                                    p_value <- contr_data['revpairwise','p.value']

                                    p_label <- paste0("p = ", format(p_value, digits=3))
                                    d_label <- paste0("d = ", format(d_value, digits=3))
                                    label_str <- paste(p_label, d_label, sep=", ")

                                    p <- ggplot2::ggplot(df_sub, ggplot2::aes(x=experimental_condition, y={pm})) +
                                            ggplot2::geom_boxplot(outlier.shape=NA, fill='skyblue', alpha=0.5) +
                                            ggplot2::geom_jitter(width=0.2, alpha=0.5) +
                                            ggplot2::theme_bw(base_size=14) +
                                            ggplot2::labs(title=paste("{current_task_type}", "{pm}", "(F vs. DFU)"),
                                                          x="Experimental Condition", y="{pm}") +
                                            ggplot2::theme(panel.background = ggplot2::element_rect(fill = "white"),
                                                           plot.background  = ggplot2::element_rect(fill = "white")) +
                                            ggplot2::annotate("text", x=1.5, y=max(df_sub${pm}, na.rm=TRUE),
                                                              label=label_str, vjust=-0.5, size=5)

                                    outfile <- paste0("{current_task_type}_{pm}_boxplot.png")
                                    ggplot2::ggsave(filename=outfile, plot=p,
                                                    path=plot_dir, dpi=300, width=5, height=4, bg="white")
                                    cat("Saved plot:", file.path(plot_dir, outfile), "\\n\\n")

                                }},
                                error=function(e) {{
                                    cat("Plot model for {pm} failed (singular or other error):", e$message, "\\n")
                                }}
                            )
                        )
                    }}
                }}
            """)

        # === Create boxplot for BI_score (using only (1|task_id)) ===
        ro.r(f"""
            if (!("BI_score" %in% colnames(df_sub))) {{
                cat("Column BI_score not found. Skipping plot.\\n")
            }} else {{
                if (n_items < 2) {{
                    cat("Skipping plot model for BI_score due to insufficient levels in task_id.\\n")
                }} else {{
                    suppressWarnings(
                        tryCatch(
                            {{
                                model_plot_bi <- lmerTest::lmer(BI_score ~ experimental_condition + (1|task_id) + (1|condition_item),
                                                                data=df_sub)

                                posthoc_bi <- emmeans::emmeans(model_plot_bi, specs = ~ experimental_condition)
                                contr_bi   <- emmeans::contrast(posthoc_bi, method='revpairwise') 
                                eff_bi <- emmeans::eff_size(posthoc_bi, sigma = sigma(model_plot_bi), edf = df.residual(model_plot_bi))
                                contr_labels <- rownames(as.data.frame(eff_bi))
                                d_value <- NA
                                if ('DFU - F' %in% contr_labels) {{
                                    d_value <- as.data.frame(eff_bi)['DFU - F','effect.size']
                                }} else if ('F - DFU' %in% contr_labels) {{
                                    d_value <- as.data.frame(eff_bi)['F - DFU','effect.size']
                                }} else {{
                                    d_value <- as.data.frame(eff_bi)[1,'effect.size']
                                }}

                                contr_data <- as.data.frame(contr_bi)
                                p_value <- contr_data['revpairwise','p.value']

                                p_label <- paste0("p = ", format(p_value, digits=3))
                                d_label <- paste0("d = ", format(d_value, digits=3))
                                label_str <- paste(p_label, d_label, sep=", ")

                                p <- ggplot2::ggplot(df_sub, ggplot2::aes(x=experimental_condition, y=BI_score)) +
                                        ggplot2::geom_boxplot(outlier.shape=NA, fill='skyblue', alpha=0.5) +
                                        ggplot2::geom_jitter(width=0.2, alpha=0.5) +
                                        ggplot2::theme_bw(base_size=14) +
                                        ggplot2::labs(title=paste("{current_task_type}", "BI_score", "(F vs. DFU)"),
                                                      x="Experimental Condition", y="BI_score") +
                                        ggplot2::theme(panel.background = ggplot2::element_rect(fill = "white"),
                                                       plot.background  = ggplot2::element_rect(fill = "white")) +
                                        ggplot2::annotate("text", x=1.5, y=max(df_sub$BI_score, na.rm=TRUE),
                                                          label=label_str, vjust=-0.5, size=5)

                                outfile <- paste0("{current_task_type}_BI_score_boxplot.png")
                                ggplot2::ggsave(filename=outfile, plot=p,
                                                path=plot_dir, dpi=300, width=5, height=4, bg="white")
                                cat("Saved plot:", file.path(plot_dir, outfile), "\\n\\n")

                            }},
                            error=function(e) {{
                                cat("Plot model for BI_score failed (singular or other error):", e$message, "\\n")
                            }}
                        )
                    )
                }}
            }}
        """)

    # 3) Correlate RAT vs. Insight average score_TEST across participants
    print("\n=== CORRELATION: RAT vs. INSIGHT (score_TEST) ===\n")
    ro.r('df_rat <- subset(df, task_type=="RAT")')
    ro.r('df_ins <- subset(df, task_type=="Insight")')

    # Show overall means & SD for RAT/Insight score_TEST
    ro.r("""
        cat('Overall RAT score_TEST stats:\\n')
        if(nrow(df_rat) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n',
                        mean(df_rat$score_TEST), sd(df_rat$score_TEST)))
        } else {
            cat('No RAT rows found!\\n')
        }

        cat('\\nOverall Insight score_TEST stats:\\n')
        if(nrow(df_ins) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n',
                        mean(df_ins$score_TEST), sd(df_ins$score_TEST)))
        } else {
            cat('No Insight rows found!\\n')
        }
    """)

    # Merge by participant
    ro.r('df_rat_agg <- aggregate(score_TEST ~ participant_identification, data=df_rat, FUN=mean)')
    ro.r('colnames(df_rat_agg)[2] <- "RAT_performance"')
    ro.r('df_ins_agg <- aggregate(score_TEST ~ participant_identification, data=df_ins, FUN=mean)')
    ro.r('colnames(df_ins_agg)[2] <- "Insight_performance"')
    ro.r('df_perf <- merge(df_rat_agg, df_ins_agg, by="participant_identification")')

    ro.r('cat("\\nCorrelation between average RAT and Insight performance (score_TEST):\\n")')
    ro.r("""
        if(nrow(df_perf) > 0) {
            print(cor.test(df_perf$RAT_performance, df_perf$Insight_performance))
        } else {
            cat("No participants have both RAT and Insight data.\\n")
        }
    """)

    # Also correlate RAT vs. Insight average reaction_time_TEST
    print("\n=== CORRELATION: RAT vs. INSIGHT (reaction_time_TEST) ===\n")
    ro.r("""
            if(!("reaction_time_TEST" %in% colnames(df_rat)) || !("reaction_time_TEST" %in% colnames(df_ins))) {
                cat("No reaction_time_TEST column found in RAT or Insight data.\\n")
            } else {
                df_rat_rt <- aggregate(reaction_time_TEST ~ participant_identification, data=df_rat, FUN=mean)
                colnames(df_rat_rt)[2] <- "RAT_RT"

                df_ins_rt <- aggregate(reaction_time_TEST ~ participant_identification, data=df_ins, FUN=mean)
                colnames(df_ins_rt)[2] <- "Insight_RT"

                df_perf_rt <- merge(df_rat_rt, df_ins_rt, by="participant_identification")

                cat("Correlation between average RAT and Insight reaction_time_TEST:\\n")
                if(nrow(df_perf_rt) > 0) {
                    print(cor.test(df_perf_rt$RAT_RT, df_perf_rt$Insight_RT))
                } else {
                    cat("No participants have both RAT and Insight RT data.\\n")
                }
            }
        """)

    # --- NEW ANALYSES ADDED BELOW ---

    # Correlation within each task: reaction time vs. score (using MCQ measures)
    print("\n=== CORRELATION WITHIN TASK: RAT (score_MCQ vs. reaction_time_MCQ) ===\n")
    ro.r("""
        df_rat <- subset(df, task_type=="RAT")
        if(nrow(df_rat) > 0) {
            print(cor.test(df_rat$score_MCQ, df_rat$reaction_time_MCQ))
        } else {
            cat("No RAT rows found for within-task correlation.\\n")
        }
    """)

    print("\n=== CORRELATION WITHIN TASK: INSIGHT (score_MCQ vs. reaction_time_MCQ) ===\n")
    ro.r("""
        df_ins <- subset(df, task_type=="Insight")
        if(nrow(df_ins) > 0) {
            print(cor.test(df_ins$score_MCQ, df_ins$reaction_time_MCQ))
        } else {
            cat("No Insight rows found for within-task correlation.\\n")
        }
    """)

    print("\n=== CORRELATION WITHIN TASK: RAT (score_TEST vs. reaction_time_TEST) ===\n")
    ro.r("""
        df_rat <- subset(df, task_type=="RAT")
        if(nrow(df_rat) > 0) {
            print(cor.test(df_rat$score_TEST, df_rat$reaction_time_TEST))
        } else {
            cat("No RAT rows found for within-task correlation.\\n")
        }
    """)

    print("\n=== CORRELATION WITHIN TASK: INSIGHT (score_TEST vs. reaction_time_TEST) ===\n")
    ro.r("""
        df_ins <- subset(df, task_type=="Insight")
        if(nrow(df_ins) > 0) {
            print(cor.test(df_ins$score_TEST, df_ins$reaction_time_TEST))
        } else {
            cat("No Insight rows found for within-task correlation.\\n")
        }
    """)

    # Exclude trials: use only rows where MCQ_response_unprocessed is 'correct'
    # Then perform a simple unpaired t-test for DFU vs. F experimental manipulation (using score_MCQ)
    print("\n=== UNPAIRED T-TEST FOR DFU vs. F (Only Correct Trials) ===\n")
    ro.r("""
        df_correct <- subset(df, MCQ_response_unprocessed == 'correct')
        if(nrow(df_correct) > 0) {
            ttest_result <- t.test(score_MCQ ~ experimental_condition, data=df_correct)
            print(ttest_result)
        } else {
            cat("No rows with MCQ_response_unprocessed == 'correct' found.\\n")
        }
    """)

    # --- NEW ANALYSES: LME analyses on correct trials for score_TEST and reaction_time_TEST (with descriptive stats and exclusion counts) ---
    print(
        "\n=== LME ANALYSES ON CORRECT TRIALS: score_TEST, reaction_time_TEST, and BI_score (Descriptive Stats & Exclusion Counts) ===\n")

    # Report exclusion counts: total trials vs. correct trials
    ro.r("""
        total_trials <- nrow(df)
        correct_trials <- nrow(df_correct)
        cat("Total trials in original dataset:", total_trials, "\n")
        cat("Trials with MCQ_response_unprocessed == 'correct':", correct_trials, "\n")
        cat("Trials excluded:", total_trials - correct_trials, "\n")
    """)

    # Loop through task types and measures for LME analysis on correct trials with descriptive stats
    for current_task_type in ["RAT", "Insight"]:
        ro.r(f"df_sub_correct <- subset(df_correct, task_type == '{current_task_type}')")
        ro.r(f"""
             n_participants <- length(unique(df_sub_correct$participant_identification))
             n_items <- length(unique(df_sub_correct$task_id))
             cat(sprintf("\\n%s Task - Correct Trials: %d trials, %d participants, %d items\\n",
                         "{current_task_type}", nrow(df_sub_correct), n_participants, n_items))
             """)
        # Standard measures: score_TEST and reaction_time_TEST
        for m in ["score_TEST", "reaction_time_TEST"]:
            ro.r(f"""
                  cat(sprintf("\\n{current_task_type} - {m} Descriptive Statistics (Correct Trials):\\n"))
                  df_desc <- aggregate(df_sub_correct${m},
                                       by=list(df_sub_correct$experimental_condition),
                                       FUN=function(x) c(mean=mean(x), sd=sd(x), n=length(x)))
                  colnames(df_desc) <- c('Condition', '{m}_stats')
                  print(df_desc)

                  cat(sprintf("\\n{current_task_type} - {m} LME Analysis (Correct Trials):\\n"))
                  if (n_participants < 2 || n_items < 2) {{
                      cat("Skipping model for {m} due to insufficient levels in participant/task_id.\\n")
                  }} else {{
                      cat('(Mixed linear model: {m} ~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item))\\n')
                      suppressWarnings(
                          tryCatch(
                              {{
                                  model_correct <- lmerTest::lmer({m} ~ experimental_condition +
                                                                           (1|participant_identification) +
                                                                           (1|task_id) +
                                                                           (1|condition_item),
                                                                    data=df_sub_correct)
                                  print(anova(model_correct))
                              }},
                              error=function(e) {{
                                  cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                              }}
                          )
                      )
                      cat('\\n----------------------------------------\\n')
                  }}
             """)
        # BI_score analysis on correct trials (using only (1|task_id))
        ro.r(f"""
              cat(sprintf("\\n{current_task_type} - BI_score Descriptive Statistics (Correct Trials):\\n"))
              df_desc_bi <- aggregate(df_sub_correct$BI_score,
                                      by=list(df_sub_correct$experimental_condition),
                                      FUN=function(x) c(mean=mean(x), sd=sd(x), n=length(x)))
              colnames(df_desc_bi) <- c('Condition', 'BI_score_stats')
              print(df_desc_bi)

              cat(sprintf("\\n{current_task_type} - BI_score LME Analysis (Correct Trials):\\n"))
              if (n_items < 2) {{
                  cat("Skipping model for BI_score due to insufficient levels in task_id.\\n")
              }} else {{
                  cat('(Mixed linear model: BI_score ~ experimental_condition + (1|task_id) + (1|condition_item))\\n')
                  suppressWarnings(
                      tryCatch(
                          {{
                              model_correct_bi <- lmerTest::lmer(BI_score ~ experimental_condition +
                                                                           (1|task_id) +
                                                                           (1|condition_item),
                                                                    data=df_sub_correct)
                              print(anova(model_correct_bi))
                          }},
                          error=function(e) {{
                              cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                          }}
                      )
                  )
                  cat('\\n----------------------------------------\\n')
              }}
         """)

    # --- NEW CODE PART: Exclude full blocks with no correct answers and analyze filtered data for RAT and Insight (TEST scores, reaction times and BI_score) ---
    print("\n=== EXCLUSION OF FULL BLOCKS WITH NO CORRECT ANSWER AND ANALYSIS ON FILTERED DATA ===\n")
    ro.r("""
        # Create a flag indicating whether the response was correct
        df$correct_flag <- ifelse(df$MCQ_response_unprocessed == 'correct', 1, 0)

        # Summarize by participant and block: count correct responses in each block
        block_summary <- aggregate(correct_flag ~ participant_identification + block_number, 
                                   data=df, FUN=sum)

        # Identify valid blocks: blocks where at least one trial was correct
        valid_blocks <- subset(block_summary, correct_flag >= 1)

        # Merge to keep only trials from valid blocks
        df_valid_blocks <- merge(df, valid_blocks[, c('participant_identification', 'block_number')], 
                                 by=c('participant_identification', 'block_number'))

        # Report the exclusion counts
        total_trials <- nrow(df)
        valid_trials <- nrow(df_valid_blocks)
        cat("Total trials in original dataset:", total_trials, "\n")
        cat("Trials kept after excluding blocks with no correct answer:", valid_trials, "\n")
        cat("Trials excluded:", total_trials - valid_trials, "\n")

        # --- ANALYSIS ON FILTERED DATA (Only blocks with at least one correct answer) ---
        for(current_task_type in c("RAT", "Insight")) {
            cat(sprintf("\\n--- %s TASK ---\\n", current_task_type))
            df_sub_valid <- subset(df_valid_blocks, task_type == current_task_type)
            n_participants <- length(unique(df_sub_valid$participant_identification))
            n_items <- length(unique(df_sub_valid$task_id))
            cat(sprintf("Total trials: %d, Participants: %d, Items: %d\\n", nrow(df_sub_valid), n_participants, n_items))

            # For standard measures: score_TEST and reaction_time_TEST
            for(m in c("score_TEST", "reaction_time_TEST")) {
                cat(sprintf("\\n%s - %s Descriptive Statistics (Filtered Blocks):\\n", current_task_type, m))
                df_desc <- aggregate(df_sub_valid[[m]], 
                                     by=list(df_sub_valid$experimental_condition),
                                     FUN=function(x) c(mean=mean(x), sd=sd(x), n=length(x)))
                colnames(df_desc) <- c("Condition", paste0(m, "_stats"))
                print(df_desc)

                if(n_participants < 2 || n_items < 2) {
                    cat("Skipping model for", m, "due to insufficient levels in participant/task_id.\\n")
                } else {
                    cat(sprintf("\\n%s - %s Mixed Linear Model (Filtered Blocks):\\n", current_task_type, m))
                    cat(sprintf("(Model: %s ~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item))\\n", m))
                    suppressWarnings(
                        tryCatch(
                            {
                                model_valid <- lmerTest::lmer(as.formula(paste(m, "~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item)")), data=df_sub_valid)
                                print(anova(model_valid))
                            },
                            error=function(e) {
                                cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                            }
                        )
                    )
                    cat("-----------------------------------------------------\\n")
                }
            }
            # BI_score analysis on filtered blocks (using only (1|task_id))
            cat(sprintf("\\n%s - BI_score Descriptive Statistics (Filtered Blocks):\\n", current_task_type))
            df_desc_bi <- aggregate(df_sub_valid$BI_score, 
                                    by=list(df_sub_valid$experimental_condition),
                                    FUN=function(x) c(mean=mean(x), sd=sd(x), n=length(x)))
            colnames(df_desc_bi) <- c("Condition", "BI_score_stats")
            print(df_desc_bi)

            if(n_items < 2) {
                cat("Skipping model for BI_score due to insufficient levels in task_id.\\n")
            } else {
                cat(sprintf("\\n%s - BI_score Mixed Linear Model (Filtered Blocks):\\n", current_task_type))
                cat("(Model: BI_score ~ experimental_condition + (1|task_id) + (1|condition_item))\\n")
                suppressWarnings(
                    tryCatch(
                        {
                            model_valid_bi <- lmerTest::lmer(BI_score ~ experimental_condition + (1|task_id) + (1|condition_item), data=df_sub_valid)
                            print(anova(model_valid_bi))
                        },
                        error=function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                        }
                    )
                )
                cat("-----------------------------------------------------\\n")
            }
        }
    """)

    print("\n=== EXCLUSION OF TRIALS UNTIL FIRST CORRECT ANSWER IN EACH BLOCK AND ANALYSIS ON FILTERED DATA ===\n")
    ro.r("""
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

        # --- ANALYSIS ON FILTERED DATA (Using new exclusion logic: trials until first correct are removed) ---
        for(current_task_type in c("RAT", "Insight")) {
            cat(sprintf("\\n--- %s TASK ---\\n", current_task_type))
            df_sub_valid <- subset(df_filtered, task_type == current_task_type)
            n_participants <- length(unique(df_sub_valid$participant_identification))
            n_items <- length(unique(df_sub_valid$task_id))
            cat(sprintf("Total trials: %d, Participants: %d, Items: %d\\n", nrow(df_sub_valid), n_participants, n_items))

            # For standard measures: score_TEST and reaction_time_TEST
            for(m in c("score_TEST", "reaction_time_TEST")) {
                cat(sprintf("\\n%s - %s Descriptive Statistics (Filtered Trials):\\n", current_task_type, m))
                df_desc <- aggregate(df_sub_valid[[m]],
                                     by = list(df_sub_valid$experimental_condition),
                                     FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x)))
                colnames(df_desc) <- c("Condition", paste0(m, "_stats"))
                print(df_desc)

                if(n_participants < 2 || n_items < 2) {
                    cat("Skipping model for", m, "due to insufficient levels in participant/task_id.\\n")
                } else {
                    cat(sprintf("\\n%s - %s Mixed Linear Model (Filtered Trials):\\n", current_task_type, m))
                    cat(sprintf("(Model: %s ~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item))\\n", m))
                    suppressWarnings(
                        tryCatch(
                            {
                                model_valid <- lmerTest::lmer(as.formula(paste(m, "~ experimental_condition + (1|participant_identification) + (1|task_id) + (1|condition_item)")), data = df_sub_valid)
                                print(anova(model_valid))
                            },
                            error = function(e) {
                                cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                            }
                        )
                    )
                    cat("-----------------------------------------------------\\n")
                }
            }
            # BI_score analysis on filtered trials (using only (1|task_id))
            cat(sprintf("\\n%s - BI_score Descriptive Statistics (Filtered Trials):\\n", current_task_type))
            df_desc_bi <- aggregate(df_sub_valid$BI_score,
                                    by = list(df_sub_valid$experimental_condition),
                                    FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x)))
            colnames(df_desc_bi) <- c("Condition", "BI_score_stats")
            print(df_desc_bi)

            if(n_items < 2) {
                cat("Skipping model for BI_score due to insufficient levels in task_id.\\n")
            } else {
                cat(sprintf("\\n%s - BI_score Mixed Linear Model (Filtered Trials):\\n", current_task_type))
                cat("(Model: BI_score ~ experimental_condition + (1|task_id) + (1|condition_item))\\n")
                suppressWarnings(
                    tryCatch(
                        {
                            model_valid_bi <- lmerTest::lmer(BI_score ~ experimental_condition + (1|task_id) + (1|condition_item), data = df_sub_valid)
                            print(anova(model_valid_bi))
                        },
                        error = function(e) {
                            cat("Model failed (singular or unidentifiable):", e$message, "\\n")
                        }
                    )
                )
                cat("-----------------------------------------------------\\n")
            }
        }
    """)

    print(
        "\n=== FULL DATASET LME ANALYSIS WITH EXCLUSION LOGIC (Exclude Trials Until First Correct in Each Block) ===\n")
    ro.r("""
        library(dplyr)
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

        # --- FULL DATASET score_TEST Analysis (Descriptive Stats & LME) ---
        cat("\\nscore_TEST Descriptive Statistics (Full Filtered Dataset):\\n")
        df_desc_score <- aggregate(df_filtered$score_TEST, 
                                   by=list(df_filtered$experimental_condition),
                                   FUN=function(x) c(mean=mean(x), sd=sd(x)))
        colnames(df_desc_score) <- c("Condition", "score_TEST_stats")
        print(df_desc_score)

        # --- FULL DATASET reaction_time_TEST Analysis (Descriptive Stats & LME) ---
        cat("\\nreaction_time_TEST Descriptive Statistics (Full Filtered Dataset):\\n")
        df_desc_rt <- aggregate(df_filtered$reaction_time_TEST, 
                                by=list(df_filtered$experimental_condition),
                                FUN=function(x) c(mean=mean(x), sd=sd(x)))
        colnames(df_desc_rt) <- c("Condition", "reaction_time_TEST_stats")
        print(df_desc_rt)

        # --- FULL DATASET BI_score Analysis (Descriptive Stats & LME) ---
        cat("\\nBI_score Descriptive Statistics (Full Filtered Dataset):\\n")
        df_desc_bi <- aggregate(df_filtered$BI_score, 
                                by=list(df_filtered$experimental_condition),
                                FUN=function(x) c(mean=mean(x), sd=sd(x)))
        colnames(df_desc_bi) <- c("Condition", "BI_score_stats")
        print(df_desc_bi)

        # Mixed Linear Model for score_TEST on the full filtered dataset (both RAT and Insight)
        cat("\\nMixed Linear Model for score_TEST (Full Filtered Dataset):\\n")
        model_score <- lmerTest::lmer(score_TEST ~ experimental_condition 
        + (experimental_condition|participant_identification) + (1|task_id) + (1|condition_item), data=df_filtered)
        
        print(anova(model_score))

        # Mixed Linear Model for reaction_time_TEST on the full filtered dataset (both RAT and Insight)
        cat("\\nMixed Linear Model for reaction_time_TEST (Full Filtered Dataset):\\n")
        model_rt <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * task_type * affinity_score
        + (experimental_condition|participant_identification) + (1|task_id) + (1|condition_item), data=df_filtered)
        print(summary(model_rt))
        print(anova(model_rt))

        # Mixed Linear Model for BI_score on the full filtered dataset (using only (1|task_id))
        cat("\\nMixed Linear Model for BI_score (Full Filtered Dataset):\\n")
        model_bi_full <- lmerTest::lmer(BI_score ~ experimental_condition 
        + (1|task_id) + (1|condition_item), data=df_filtered)
        
        print(anova(model_bi_full))

        # Unpaired t-test between the two conditions for score_TEST
        cat("\\nUnpaired t-test for score_TEST (DFU vs F):\\n")
        ttest_score <- t.test(score_TEST ~ experimental_condition, data=df_filtered)
        print(ttest_score)

        # Unpaired t-test between the two conditions for reaction_time_TEST
        cat("\\nUnpaired t-test for reaction_time_TEST (DFU vs F):\\n")
        ttest_rt <- t.test(reaction_time_TEST ~ experimental_condition, data=df_filtered)
        print(ttest_rt)

        # Unpaired t-test for BI_score
        cat("\\nUnpaired t-test for BI_score (DFU vs F):\\n")
        ttest_bi <- t.test(BI_score ~ experimental_condition, data=df_filtered)
        print(ttest_bi)

        # General Linear Model for score_TEST with experimental_condition (no random effects)
        cat("\\nGeneral Linear Model for score_TEST (no random effects):\\n")
        glm_score <- lm(score_TEST ~ experimental_condition, data=df_filtered)
        print(summary(glm_score))

        # General Linear Model for reaction_time_TEST with experimental_condition (no random effects)
        cat("\\nGeneral Linear Model for reaction_time_TEST (no random effects):\\n")
        glm_rt <- lm(reaction_time_TEST ~ experimental_condition, data=df_filtered)
        print(summary(glm_rt))

        # General Linear Model for BI_score with experimental_condition (no random effects)
        cat("\\nGeneral Linear Model for BI_score (no random effects):\\n")
        glm_bi <- lm(BI_score ~ experimental_condition, data=df_filtered)
        print(summary(glm_bi))
    """)

    ro.r('''
        library(dplyr)
        library(car)  # For Type III ANOVA

        #################################################################
        # ANALYSES ON FULL FILTERED DATASET
        #################################################################

        # Exclusion logic: For each participant and block, number the trials 
        # and compute the cumulative count of correct MCQ responses.
        # Exclude trials until the first correct response is given.
        df_with_seq <- df %>%
          group_by(participant_identification, block_number) %>%
          mutate(trial_index = row_number(),
                 cum_correct = cumsum(MCQ_response_unprocessed == "correct")) %>%
          ungroup()
        # Filter to keep only trials after (and including) the first correct response,
        # and restrict trial_index to 1 to 4.
        df_filtered <- df_with_seq %>% filter(cum_correct > 0, trial_index <= 4)

        cat("\n\n##########################\nFULL DATASET ANALYSES\n##########################\n")

        ##############################
        # score_TEST Models
        ##############################
        cat("\n--- score_TEST Models ---\n")
        cat("\nInteraction Model (score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item)):\n")
        model_score_inter <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                              (1|participant_identification) + (1|task_id) + (1|condition_item),
                                            data = df_filtered)
        print(summary(model_score_inter))
        cat("\nType III ANOVA for interaction model (score_TEST):\n")
        print(car::Anova(model_score_inter, type="III"))

        cat("\nNo Interaction Model (score_TEST ~ experimental_condition + trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item)):\n")
        model_score_nointer <- lmerTest::lmer(score_TEST ~ experimental_condition + trial_index +
                                                (1|participant_identification) + (1|task_id) + (1|condition_item),
                                              data = df_filtered)
        print(summary(model_score_nointer))

        cat("\nModel Comparison (score_TEST):\n")
        print(anova(model_score_nointer, model_score_inter))

        # Compute the difference in means (DFU minus F) for score_TEST by trial_index
        score_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for score_TEST (DFU minus F) by trial_index:\n")
        print(score_diff)

        ##############################
        # reaction_time_TEST Models
        ##############################
        cat("\n--- reaction_time_TEST Models ---\n")
        cat("\nInteraction Model (reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item)):\n")
        model_rt_inter <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                           (1|participant_identification) + (1|task_id) + (1|condition_item),
                                         data = df_filtered)
        print(summary(model_rt_inter))
        cat("\nType III ANOVA for interaction model (reaction_time_TEST):\n")
        print(car::Anova(model_rt_inter, type="III"))

        cat("\nNo Interaction Model (reaction_time_TEST ~ experimental_condition + trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item)):\n")
        model_rt_nointer <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition + trial_index +
                                             (1|participant_identification) + (1|task_id) + (1|condition_item),
                                           data = df_filtered)
        print(summary(model_rt_nointer))

        cat("\nModel Comparison (reaction_time_TEST):\n")
        print(anova(model_rt_nointer, model_rt_inter))

        # Compute the difference in means (DFU minus F) for reaction_time_TEST by trial_index
        rt_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for reaction_time_TEST (DFU minus F) by trial_index:\n")
        print(rt_diff)

        ##############################
        # BI_score Models
        ##############################
        cat("\n--- BI_score Models ---\n")
        cat("\nInteraction Model (BI_score ~ experimental_condition * trial_index + (1|task_id) + (1|condition_item)):\n")
        model_bi_inter <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                           (1|task_id) + (1|condition_item),
                                         data = df_filtered)
        print(summary(model_bi_inter))
        cat("\nType III ANOVA for interaction model (BI_score):\n")
        print(car::Anova(model_bi_inter, type="III"))

        cat("\nNo Interaction Model (BI_score ~ experimental_condition + trial_index + (1|task_id) + (1|condition_item)):\n")
        model_bi_nointer <- lmerTest::lmer(BI_score ~ experimental_condition + trial_index +
                                             (1|task_id) + (1|condition_item),
                                           data = df_filtered)
        print(summary(model_bi_nointer))

        cat("\nModel Comparison (BI_score):\n")
        print(anova(model_bi_nointer, model_bi_inter))

        # Compute the difference in means (DFU minus F) for BI_score by trial_index
        bi_diff <- df_filtered %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for BI_score (DFU minus F) by trial_index:\n")
        print(bi_diff)

        #################################################################
        # ANALYSES ON RAT SUBSET
        #################################################################

        # Subset data for RAT task_type
        df_RAT <- df_filtered %>% filter(task_type == "RAT")
        cat("\n\n##########################\nRAT SUBSET ANALYSES\n##########################\n")

        # --- MODEL FOR score_TEST (RAT) ---
        cat("\nMixed Linear Model for score_TEST with trial_index interaction (RAT Subset):\n")
        cat("(Model formula: score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item))\n")
        model_score_trial_RAT <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                                   (1|participant_identification) + (1|task_id) + (1|condition_item),
                                                 data = df_RAT)
        print(summary(model_score_trial_RAT))
        cat("\nType III ANOVA for score_TEST model (RAT):\n")
        print(car::Anova(model_score_trial_RAT, type="III"))

        # Compute the difference in means for score_TEST in RAT subset
        score_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for score_TEST (DFU minus F) by trial_index (RAT):\n")
        print(score_diff_RAT)

        # --- MODEL FOR reaction_time_TEST (RAT) ---
        cat("\nMixed Linear Model for reaction_time_TEST with trial_index interaction (RAT Subset):\n")
        cat("(Model formula: reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item))\n")
        model_rt_trial_RAT <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                                (1|participant_identification) + (1|task_id) + (1|condition_item),
                                              data = df_RAT)
        print(summary(model_rt_trial_RAT))
        cat("\nType III ANOVA for reaction_time_TEST model (RAT):\n")
        print(car::Anova(model_rt_trial_RAT, type="III"))

        # Compute the difference in means for reaction_time_TEST in RAT subset
        rt_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for reaction_time_TEST (DFU minus F) by trial_index (RAT):\n")
        print(rt_diff_RAT)

        # --- MODEL FOR BI_score (RAT) ---
        cat("\nMixed Linear Model for BI_score with trial_index interaction (RAT Subset):\n")
        cat("(Model formula: BI_score ~ experimental_condition * trial_index + (1|task_id) + (1|condition_item))\n")
        model_bi_trial_RAT <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                                (1|task_id) + (1|condition_item),
                                              data = df_RAT)
        print(summary(model_bi_trial_RAT))
        cat("\nType III ANOVA for BI_score model (RAT):\n")
        print(car::Anova(model_bi_trial_RAT, type="III"))

        # Compute the difference in means for BI_score in RAT subset
        bi_diff_RAT <- df_RAT %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for BI_score (DFU minus F) by trial_index (RAT):\n")
        print(bi_diff_RAT)

        #################################################################
        # ANALYSES ON INSIGHT SUBSET
        #################################################################

        # Subset data for Insight task_type
        df_Insight <- df_filtered %>% filter(task_type == "Insight")
        cat("\n\n##########################\nINSIGHT SUBSET ANALYSES\n##########################\n")

        # --- MODEL FOR score_TEST (Insight) ---
        cat("\nMixed Linear Model for score_TEST with trial_index interaction (Insight Subset):\n")
        cat("(Model formula: score_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item))\n")
        model_score_trial_Insight <- lmerTest::lmer(score_TEST ~ experimental_condition * trial_index +
                                                       (1|participant_identification) + (1|task_id) + (1|condition_item),
                                                     data = df_Insight)
        print(summary(model_score_trial_Insight))
        cat("\nType III ANOVA for score_TEST model (Insight):\n")
        print(car::Anova(model_score_trial_Insight, type="III"))

        # Compute the difference in means for score_TEST in Insight subset
        score_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(score_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(score_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for score_TEST (DFU minus F) by trial_index (Insight):\n")
        print(score_diff_Insight)

        # --- MODEL FOR reaction_time_TEST (Insight) ---
        cat("\nMixed Linear Model for reaction_time_TEST with trial_index interaction (Insight Subset):\n")
        cat("(Model formula: reaction_time_TEST ~ experimental_condition * trial_index + (1|participant_identification) + (1|task_id) + (1|condition_item))\n")
        model_rt_trial_Insight <- lmerTest::lmer(reaction_time_TEST ~ experimental_condition * trial_index +
                                                    (1|participant_identification) + (1|task_id) + (1|condition_item),
                                                  data = df_Insight)
        print(summary(model_rt_trial_Insight))
        cat("\nType III ANOVA for reaction_time_TEST model (Insight):\n")
        print(car::Anova(model_rt_trial_Insight, type="III"))

        # Compute the difference in means for reaction_time_TEST in Insight subset
        rt_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(reaction_time_TEST[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(reaction_time_TEST[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for reaction_time_TEST (DFU minus F) by trial_index (Insight):\n")
        print(rt_diff_Insight)

        # --- MODEL FOR BI_score (Insight) ---
        cat("\nMixed Linear Model for BI_score with trial_index interaction (Insight Subset):\n")
        cat("(Model formula: BI_score ~ experimental_condition * trial_index + (1|task_id) + (1|condition_item))\n")
        model_bi_trial_Insight <- lmerTest::lmer(BI_score ~ experimental_condition * trial_index +
                                                    (1|task_id) + (1|condition_item),
                                                  data = df_Insight)
        print(summary(model_bi_trial_Insight))
        cat("\nType III ANOVA for BI_score model (Insight):\n")
        print(car::Anova(model_bi_trial_Insight, type="III"))

        # Compute the difference in means for BI_score in Insight subset
        bi_diff_Insight <- df_Insight %>%
          group_by(trial_index) %>%
          summarize(mean_DFU = mean(BI_score[experimental_condition == "DFU"], na.rm = TRUE),
                    mean_F = mean(BI_score[experimental_condition == "F"], na.rm = TRUE),
                    diff = mean_DFU - mean_F,
                    .groups = "drop")
        cat("\nDifference in means for BI_score (DFU minus F) by trial_index (Insight):\n")
        print(bi_diff_Insight)
    ''')


if __name__ == "__main__":
    plot_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/experimental_manipulation"
    csv_path = ("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
    )
    analyze_experiment(csv_path)

# TODO: apply clipping (RAT: 0.5-19.8, Insight: 2-118 ; enkel toepassen voor below, niet above) & remove trial-based outliers 3SD
# TODO: gebruik multivariate methode
# NOTE: BI-score betekent hier 'Balanced Integration score' ; seperately computed within 'task_type' & 'participant_identificatinon' groups
# Change columns in csv!
