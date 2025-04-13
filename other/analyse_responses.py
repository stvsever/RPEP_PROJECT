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
stats = importr('stats')        # For cor.test, anova, etc.
utils = importr('utils')        # For read.csv, etc.

# Load additional R libraries
ro.r('library(ggplot2)')
ro.r('library(emmeans)')
ro.r('library(effectsize)')


def analyze_experiment(file_path):
    """
    1) Read CSV into R and remove missing rows.
    2) For RAT vs. Insight tasks:
       - Print mean/SD by experimental_condition and run mixed models for various measures.
       - Print the model formula (in parentheses) under each result.
       - For score_TEST and reaction_time_TEST only: produce boxplots with p-value & Cohen's d annotation (4 plots total).
    3) Run eight correlation tests:
       - Aggregated (per participant):
         A1: Aggregated RAT score vs. Aggregated Insight score.
         A2: Aggregated RAT reaction_time_TEST vs. Aggregated Insight reaction_time_TEST.
         A3: Within RAT aggregated: reaction_time_TEST vs. score_TEST.
         A4: Within Insight aggregated: reaction_time_TEST vs. score_TEST.
       - Non-Aggregated (raw row-level; no merging):
         NA1: For RAT only: correlation between score_TEST and reaction_time_TEST.
         NA2: For Insight only: correlation between score_TEST and reaction_time_TEST.
         NA3: For full data (both tasks): correlation between score_TEST and reaction_time_TEST.
         NA4: For full data (both tasks): correlation between reaction_time_TEST and score_TEST.
    """

    # --- 1) Read data into R, remove missing rows ---
    ro.r(f"df <- read.csv('{file_path}', header=TRUE, stringsAsFactors=FALSE)")
    ro.r("df <- na.omit(df)")  # remove rows with any NA

    # Print columns to confirm names
    ro.r("cat('Column names in df:\\n')")
    ro.r("print(colnames(df))")

    # Convert key columns to factors
    ro.r('df$participant_identification <- as.factor(df$participant_identification)')
    ro.r('df$experimental_condition <- as.factor(df$experimental_condition)')
    ro.r('df$block_number <- as.factor(df$block_number)')
    ro.r('df$task_type <- as.factor(df$task_type)')
    ro.r('df$task_id <- as.factor(df$task_id)')

    # All measures of interest
    all_measures = ["score_MCQ", "reaction_time_MCQ", "score_TEST", "reaction_time_TEST", "affinity_score"]

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

        # For each measure, print mean/SD by condition, run the model, and show ANOVA
        print("=== CONDITION ANALYSIS (F vs. DFU) ===")
        for m in all_measures:
            ro.r(f"""
                # Print group means & SD
                df_summary_cond <- aggregate(df_sub${m},
                                             by=list(df_sub$experimental_condition),
                                             FUN=function(x) c(mean=mean(x), sd=sd(x)))
                colnames(df_summary_cond) <- c('Condition', '{m}')
                cat('\\nMeasure: {m} - Means & SD by Condition\\n')
                print(df_summary_cond)

                # Fit mixed model with random intercepts for participant & task_id
                model_cond <- lmerTest::lmer({m} ~ experimental_condition +
                                                     (1|participant_identification) +
                                                     (1|task_id),
                                             data=df_sub)
                cat('\\n(Mixed linear model: {m} ~ experimental_condition + (1|participant_identification) + (1|task_id))\\n')
                cat('ANOVA table:\\n')
                print(anova(model_cond))
                cat('\\n----------------------------------------\\n')
            """)

        # For block_number analysis (no plots)
        print("\n=== BLOCK NUMBER ANALYSIS ===")
        for m in all_measures:
            ro.r(f"""
                df_summary_block <- aggregate(df_sub${m},
                                              by=list(df_sub$block_number),
                                              FUN=function(x) c(mean=mean(x), sd=sd(x)))
                colnames(df_summary_block) <- c('Block', '{m}')
                cat('\\nMeasure: {m} - Means & SD by Block\\n')
                print(df_summary_block)

                model_block <- lmerTest::lmer({m} ~ block_number + 
                                                      (1|participant_identification) +
                                                      (1|task_id),
                                              data=df_sub)
                cat('\\n(Mixed linear model: {m} ~ block_number + (1|participant_identification) + (1|task_id))\\n')
                cat('ANOVA table:\\n')
                print(anova(model_block))
                cat('\\n----------------------------------------\\n')
            """)

        # --- Create boxplots with significance markers only for score_TEST / reaction_time_TEST ---
        plot_measures = ["score_TEST", "reaction_time_TEST"]
        for pm in plot_measures:
            ro.r(f"""
                if (!("{pm}" %in% colnames(df_sub))) {{
                    cat("Column {pm} not found. Skipping plot.\\n")
                }} else {{
                    model_plot <- lmerTest::lmer({pm} ~ experimental_condition +
                                                           (1|participant_identification) +
                                                           (1|task_id),
                                                 data=df_sub)
                    posthoc_cond <- emmeans::emmeans(model_plot, specs = ~ experimental_condition)
                    contr_cond   <- emmeans::contrast(posthoc_cond, method='revpairwise')  # DFU - F
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
                            ggplot2::theme_minimal(base_size=14) +
                            ggplot2::labs(title=paste("{current_task_type}", "{pm}", "(F vs. DFU)"),
                                          x="Experimental Condition", y="{pm}") +
                            ggplot2::annotate("text", x=1.5, y=max(df_sub${pm}, na.rm=TRUE),
                                              label=label_str, vjust=-0.5, size=5)
                    outfile <- paste0("{current_task_type}_{pm}_boxplot.png")
                    ggplot2::ggsave(outfile, plot=p, width=5, height=4, dpi=300)
                    cat("Saved plot:", outfile, "\\n\\n")
                }}
            """)

    # --- 3) CORRELATION ANALYSES ---
    print("\n=== CORRELATION ANALYSES ===\n")
    # Subset data for RAT and Insight tasks
    ro.r('df_rat <- subset(df, task_type=="RAT")')
    ro.r('df_ins <- subset(df, task_type=="Insight")')

    # Show overall means & SD for score_TEST and reaction_time_TEST in each group
    ro.r("""
        cat('Overall RAT score_TEST stats:\\n')
        if(nrow(df_rat) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n', mean(df_rat$score_TEST), sd(df_rat$score_TEST)))
        } else {
            cat('No RAT rows found!\\n')
        }
        cat('\\nOverall RAT reaction_time_TEST stats:\\n')
        if(nrow(df_rat) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n', mean(df_rat$reaction_time_TEST), sd(df_rat$reaction_time_TEST)))
        } else {
            cat('No RAT rows found!\\n')
        }
        cat('\\nOverall Insight score_TEST stats:\\n')
        if(nrow(df_ins) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n', mean(df_ins$score_TEST), sd(df_ins$score_TEST)))
        } else {
            cat('No Insight rows found!\\n')
        }
        cat('\\nOverall Insight reaction_time_TEST stats:\\n')
        if(nrow(df_ins) > 0) {
            cat(sprintf('Mean = %.2f, SD = %.2f\\n', mean(df_ins$reaction_time_TEST), sd(df_ins$reaction_time_TEST)))
        } else {
            cat('No Insight rows found!\\n')
        }
    """)

    # ==== Aggregated (per participant) Correlations ====
    ro.r("cat('\\n--- AGGREGATED CORRELATION ANALYSES ---\\n')")
    # Aggregate RAT data by participant for score and reaction time
    ro.r('df_rat_agg <- aggregate(cbind(score_TEST, reaction_time_TEST) ~ participant_identification, data=df_rat, FUN=mean)')
    # Aggregate Insight data by participant
    ro.r('df_ins_agg <- aggregate(cbind(score_TEST, reaction_time_TEST) ~ participant_identification, data=df_ins, FUN=mean)')
    # Merge aggregated RAT and Insight data on participant_identification
    ro.r('df_merge_agg <- merge(df_rat_agg, df_ins_agg, by="participant_identification", suffixes=c(".RAT", ".Insight"))')

    # A1: Aggregated: RAT score vs. Insight score
    ro.r('cat("\\n[Aggregated A1] Correlation: RAT score vs. Insight score\\n")')
    ro.r("""
        if(nrow(df_merge_agg) > 0) {
            print(cor.test(df_merge_agg$score_TEST.RAT, df_merge_agg$score_TEST.Insight))
        } else {
            cat("No aggregated matched participants for A1.\\n")
        }
    """)
    # A2: Aggregated: RAT reaction_time_TEST vs. Insight reaction_time_TEST
    ro.r('cat("\\n[Aggregated A2] Correlation: RAT reaction_time_TEST vs. Insight reaction_time_TEST\\n")')
    ro.r("""
        if(nrow(df_merge_agg) > 0) {
            print(cor.test(df_merge_agg$reaction_time_TEST.RAT, df_merge_agg$reaction_time_TEST.Insight))
        } else {
            cat("No aggregated matched participants for A2.\\n")
        }
    """)
    # A3: Aggregated within RAT: reaction_time_TEST vs. score_TEST
    ro.r('cat("\\n[Aggregated A3] Correlation: Within RAT: reaction_time_TEST vs. score_TEST\\n")')
    ro.r("""
        if(nrow(df_rat_agg) > 0) {
            print(cor.test(df_rat_agg$reaction_time_TEST, df_rat_agg$score_TEST))
        } else {
            cat("No aggregated RAT data for A3.\\n")
        }
    """)
    # A4: Aggregated within Insight: reaction_time_TEST vs. score_TEST
    ro.r('cat("\\n[Aggregated A4] Correlation: Within Insight: reaction_time_TEST vs. score_TEST\\n")')
    ro.r("""
        if(nrow(df_ins_agg) > 0) {
            print(cor.test(df_ins_agg$reaction_time_TEST, df_ins_agg$score_TEST))
        } else {
            cat("No aggregated Insight data for A4.\\n")
        }
    """)

    # ==== Non-Aggregated (Raw Row-Level) Correlations ====
    ro.r("cat('\\n--- NON-AGGREGATED (RAW) CORRELATION ANALYSES ---\\n')")
    # NA1: For RAT raw data: correlation between score_TEST and reaction_time_TEST
    ro.r('cat("\\n[Raw NA1] (RAT only) Correlation: score_TEST vs. reaction_time_TEST\\n")')
    ro.r("""
        if(nrow(df_rat) > 0) {
            print(cor.test(df_rat$score_TEST, df_rat$reaction_time_TEST))
        } else {
            cat("No raw RAT data for NA1.\\n")
        }
    """)
    # NA2: For Insight raw data: correlation between score_TEST and reaction_time_TEST
    ro.r('cat("\\n[Raw NA2] (Insight only) Correlation: score_TEST vs. reaction_time_TEST\\n")')
    ro.r("""
        if(nrow(df_ins) > 0) {
            print(cor.test(df_ins$score_TEST, df_ins$reaction_time_TEST))
        } else {
            cat("No raw Insight data for NA2.\\n")
        }
    """)
    # NA3: For full data (both tasks) raw: correlation between score_TEST and reaction_time_TEST
    ro.r('cat("\\n[Raw NA3] (Full Data) Correlation: score_TEST vs. reaction_time_TEST\\n")')
    ro.r("""
        if(nrow(df) > 0) {
            print(cor.test(df$score_TEST, df$reaction_time_TEST))
        } else {
            cat("No raw full data available for NA3.\\n")
        }
    """)


if __name__ == "__main__":
    csv_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/"
        "School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/"
        "RPEP_experiment/data/preprocessed/concatenated_data.csv"
    )
    analyze_experiment(csv_path)
