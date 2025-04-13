import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri as pandas2ri

pandas2ri.activate()

# Import required R packages
lmerTest = importr('lmerTest')
base = importr('base')
utils = importr('utils')

# Set the CSV file path (update if necessary)
csv_path = (
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
)

# Read CSV into R and remove rows with missing values in key columns
ro.r(f"""
df <- read.csv('{csv_path}', header=TRUE, stringsAsFactors=FALSE)
df <- df[!is.na(df$affinity_score) & !is.na(df$score_TEST) & !is.na(df$reaction_time_TEST), ]
""")

# Dictionary: model results
lme_results_affinity_index = {
    ("DFU", "RAT", "score_TEST"): {"estimate": -0.029345, "std_error": 0.007779, "pvalue": 0.00019},
    ("DFU", "RAT", "reaction_time_TEST"): {"estimate": 0.3789, "std_error": 0.1316, "pvalue": 0.00424},
    ("DFU", "RAT", "BI-score"): {"estimate": -0.13885, "std_error": 0.03029, "pvalue": 6.3e-06},
    ("DFU", "RAT", "BI_score_TaskType"): {"estimate": -0.12720, "std_error": 0.03015, "pvalue": 3.14e-05},

    ("DFU", "Insight", "score_TEST"): {"estimate": -0.06006, "std_error": 0.01008, "pvalue": 6.26e-09},
    ("DFU", "Insight", "reaction_time_TEST"): {"estimate": 7.8606, "std_error": 0.6618, "pvalue": 2e-16},
    ("DFU", "Insight", "BI-score"): {"estimate": -0.23597, "std_error": 0.02289, "pvalue": 2e-16},
    ("DFU", "Insight", "BI_score_TaskType"): {"estimate": -0.25900, "std_error": 0.02361, "pvalue": 2e-16},

    ("DFU", "Both", "score_TEST"): {"estimate": -0.043113, "std_error": 0.006498, "pvalue": 6.54e-11},
    ("DFU", "Both", "reaction_time_TEST"): {"estimate": 4.3030, "std_error": 0.3849, "pvalue": 2e-16},
    ("DFU", "Both", "BI-score"): {"estimate": -0.18727, "std_error": 0.01911, "pvalue": 2e-16},
    ("DFU", "Both", "BI_score_TaskType"): {"estimate": -0.19392, "std_error": 0.01937, "pvalue": 2e-16},

    ("F", "RAT", "score_TEST"): {"estimate": -0.002477, "std_error": 0.008559, "pvalue": 0.772406},
    ("F", "RAT", "reaction_time_TEST"): {"estimate": 0.1542, "std_error": 0.1424, "pvalue": 0.279},
    ("F", "RAT", "BI-score"): {"estimate": 0.01317, "std_error": 0.03277, "pvalue": 0.688},
    ("F", "RAT", "BI_score_TaskType"): {"estimate": -0.02011, "std_error": 0.03526, "pvalue": 0.569},

    ("F", "Insight", "score_TEST"): {"estimate": -0.02080, "std_error": 0.01091, "pvalue": 0.0573},
    ("F", "Insight", "reaction_time_TEST"): {"estimate": 0.9020, "std_error": 0.7635, "pvalue": 0.238},
    ("F", "Insight", "BI-score"): {"estimate": -0.03531, "std_error": 0.02708, "pvalue": 0.193},
    ("F", "Insight", "BI_score_TaskType"): {"estimate": -0.05317, "std_error": 0.02689, "pvalue": 0.0488},

    ("F", "Both", "score_TEST"): {"estimate": -0.009113, "std_error": 0.007067, "pvalue": 0.198},
    ("F", "Both", "reaction_time_TEST"): {"estimate": 0.5109, "std_error": 0.3991, "pvalue": 0.201},
    ("F", "Both", "BI-score"): {"estimate": -0.01330, "std_error": 0.02124, "pvalue": 0.531},
    ("F", "Both", "BI_score_TaskType"): {"estimate": -0.03580, "std_error": 0.02218, "pvalue": 0.107},

    ("Both", "RAT", "score_TEST"): {"estimate": -0.016194, "std_error": 0.005654, "pvalue": 0.00431},
    ("Both", "RAT", "reaction_time_TEST"): {"estimate": 0.24234, "std_error": 0.09566, "pvalue": 0.0115},
    ("Both", "RAT", "BI-score"): {"estimate": --0.06252, "std_error": 0.02231, "pvalue": 0.00522},
    ("Both", "RAT", "BI_score_TaskType"): {"estimate": -0.07353, "std_error": 0.02267, "pvalue": 0.00124},

    ("Both", "Insight", "score_TEST"): {"estimate": -0.038994, "std_error": 0.007343, "pvalue": 1.47e-07},
    ("Both", "Insight", "reaction_time_TEST"): {"estimate": 4.5433, "std_error": 0.5142, "pvalue": 2e-16},
    ("Both", "Insight", "BI-score"): {"estimate": -0.13667, "std_error": 0.01783, "pvalue": 5.89e-14},
    ("Both", "Insight", "BI_score_TaskType"): {"estimate": -0.15682, "std_error": 0.01804, "pvalue": 2e-16},

    ("Both", "Both", "score_TEST"): {"estimate": -0.02624, "std_error": 0.00469, "pvalue": 2.69e-08},
    ("Both", "Both", "reaction_time_TEST"): {"estimate": 2.4430, "std_error": 0.2759, "pvalue": 2e-16},
    ("Both", "Both", "BI-score"): {"estimate": -0.10092, "std_error": 0.01426, "pvalue": 2.32e-12},
    ("Both", "Both", "BI_score_TaskType"): {"estimate": -0.11460, "std_error": 0.01451, "pvalue": 5.77e-15}
}

for key, stats in lme_results_affinity_index.items():
    print(f"{key}: {stats}")

def get_signif_code(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    elif pval < 0.1:
        return '.'
    else:
        return ''

# Define filtering options for experimental condition and task type.
exp_conditions = ["DFU", "F", "Both"]
task_types = ["RAT", "Insight", "Both"]
# Added new DV spec for BI_score_TaskType.
dv_specs = [
    ("score_TEST", "score_TEST"),
    ("reaction_time_TEST", "reaction_time_TEST"),
    ("BI-score", "BI_score"),
    ("BI_score_TaskType", "BI_score_TaskType")
]

# Function in R to produce mean+SD text
agg_fun = "function(x) { paste0(round(mean(x),2), ' (SD: ', round(sd(x),2), ')') }"

# Colors for data points
dv_colors = {
    "score_TEST": "cornflowerblue",
    "reaction_time_TEST": "darkorange",
    "BI_score": "purple",
    "BI_score_TaskType": "darkgreen"
}
# Regression line color
reg_line_col = "firebrick"

def run_analyses():
    """
    Runs LME for each condition, task, and DV.
    Prints aggregated data, model summary, and ANOVA in the R console.
    For BI-score we now run two analyses:
      - Original: BI_score ~ affinity_score + (1|task_id) + (1|condition_item)
      - Additional (for BI_score_TaskType): BI_score_TaskType ~ affinity_score + (1|participant_identification) + (1|condition_item)
    """
    model_count = 0
    for exp in exp_conditions:
        if exp == "Both":
            exp_filter = "df_sub <- df"
            exp_label = "Both experimental conditions"
        else:
            exp_filter = f'df_sub <- subset(df, experimental_condition == "{exp}")'
            exp_label = exp
        for task in task_types:
            if task == "Both":
                task_filter = ""
                task_label = "Both task types"
            else:
                task_filter = f'df_sub <- subset(df_sub, task_type == "{task}")'
                task_label = task
            for dv_label, dv_col in dv_specs:
                if dv_col == "BI_score":
                    template = r'''
{exp_filter}
{task_filter}
cat("-----------------------------------------------------\n")
cat("Running model:\n  Experimental Condition: {exp_label}\n  Task Type: {task_label}\n  DV: {dv_label} (Balanced Integration Score with task random effect)\n")
cat("  N =", nrow(df_sub), "\n\n")
agg_data <- aggregate(BI_score ~ affinity_score, data=df_sub, FUN={agg_fun})
cat("Aggregated data for affinity levels:\n")
print(agg_data)
# Create base aggregated plot
plot(agg_data$affinity_score, as.numeric(sapply(agg_data$BI_score, function(x) strsplit(x, " ")[[1]][1])),
     type="b", pch=16, cex=1.5, lwd=2, col="{color_data}",
     main=paste("{dv_label}", "\n", "{exp_label}", "&", "{task_label}"),
     xlab="Affinity Score", ylab=paste("Mean", "{dv_label}"), cex.lab=1.2, cex.main=0.95)
grid(col="gray70"); box()
mtext(paste0("N = ", nrow(df_sub)), side=3, line=0.5, cex=0.8)
points(df_sub$affinity_score, df_sub$BI_score, pch=1, col="gray50", cex=0.7)
model <- lmerTest::lmer(BI_score ~ affinity_score + verbal_intelligence + (1|task_id) + (1|condition_item) , data=df_sub)
cat("\n")
print(summary(model))
print(anova(model))
cat("-----------------------------------------------------\n\n")
'''
                    script = template.format(
                        exp_filter=exp_filter,
                        task_filter=task_filter,
                        exp_label=exp_label,
                        task_label=task_label,
                        dv_label=dv_label,
                        color_data=dv_colors[dv_col],
                        agg_fun=agg_fun
                    )
                elif dv_col == "BI_score_TaskType":
                    template = r'''
{exp_filter}
{task_filter}
cat("-----------------------------------------------------\n")
cat("Running model:\n  Experimental Condition: {exp_label}\n  Task Type: {task_label}\n  DV: {dv_label} \n")
cat("  N =", nrow(df_sub), "\n\n")
agg_data <- aggregate(BI_score_TaskType ~ affinity_score, data=df_sub, FUN={agg_fun})
cat("Aggregated data for affinity levels:\n")
print(agg_data)
plot(agg_data$affinity_score, as.numeric(sapply(agg_data$BI_score_TaskType, function(x) strsplit(x, " ")[[1]][1])),
     type="b", pch=16, cex=1.5, lwd=2, col="{color_data}",
     main=paste("{dv_label}", "\n", "{exp_label}", "&", "{task_label}"),
     xlab="Affinity Score", ylab=paste("Mean", "{dv_label}"), cex.lab=1.2, cex.main=0.95)
grid(col="gray70"); box()
mtext(paste0("N = ", nrow(df_sub)), side=3, line=0.5, cex=0.8)
points(df_sub$affinity_score, df_sub$BI_score_TaskType, pch=1, col="gray50", cex=0.7)
model <- lmerTest::lmer(BI_score_TaskType ~ affinity_score + verbal_intelligence + (1|participant_identification) + (1|task_id) + (1|condition_item), data=df_sub)
cat("\n")
print(summary(model))
print(anova(model))
cat("-----------------------------------------------------\n\n")
'''
                    script = template.format(
                        exp_filter=exp_filter,
                        task_filter=task_filter,
                        exp_label=exp_label,
                        task_label=task_label,
                        dv_label=dv_label,
                        color_data=dv_colors[dv_col],
                        agg_fun=agg_fun
                    )
                else:
                    template = r'''
{exp_filter}
{task_filter}
cat("-----------------------------------------------------\n")
cat("Running model:\n  Experimental Condition: {exp_label}\n  Task Type: {task_label}\n  DV: {dv_label}\n")
cat("  N =", nrow(df_sub), "\n\n")
df_sub$dv_temp <- df_sub${dv_col}
agg_data <- aggregate(dv_temp ~ affinity_score, data=df_sub, FUN={agg_fun})
cat("Aggregated data for affinity levels:\n")
print(agg_data)
plot(agg_data$affinity_score, as.numeric(sapply(agg_data$dv_temp, function(x) strsplit(x, " ")[[1]][1])),
     type="b", pch=16, cex=1.5, lwd=2, col="{color_data}",
     main=paste("{dv_label}", "\n", "{exp_label}", "&", "{task_label}"),
     xlab="Affinity Score", ylab=paste("Mean", "{dv_label}"), cex.lab=1.2, cex.main=0.95)
grid(col="gray70"); box()
mtext(paste0("N = ", nrow(df_sub)), side=3, line=0.5, cex=0.8)
points(df_sub$affinity_score, df_sub${dv_col}, pch=1, col="gray50", cex=0.7)
model <- lmerTest::lmer(dv_temp ~ affinity_score + verbal_intelligence + (1|participant_identification) + (1|task_id) + (1|condition_item), data=df_sub)
cat("\n")
print(summary(model))
print(anova(model))
cat("-----------------------------------------------------\n\n")
'''
                    script = template.format(
                        exp_filter=exp_filter,
                        task_filter=task_filter,
                        exp_label=exp_label,
                        task_label=task_label,
                        dv_label=dv_label,
                        dv_col=dv_col,
                        color_data=dv_colors[dv_col],
                        agg_fun=agg_fun
                    )
                ro.r(script)
                model_count += 1
    print("\nTotal number of models tested:", model_count)
    print("Explanation: 3 experimental conditions x 3 task types x 4 DVs = 36 models.")

def create_affinity_plots():
    """
    Creates 3x3 grid plots for each DV across (exp_condition x task_type).
    Each panel:
      - Plots aggregated data (mean+SD by affinity score) with larger points.
      - Overlays raw individual data points.
      - Plots a regression line in firebrick.
      - Places a legend in the top-right.
      - Places 'N = ...' in the top margin.
      - Places the estimate + p-value in the lower-right corner in the same color as the regression line.
      - Prints the aggregated data (mean and SD) to the R console.
    """
    plot_dir = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
        "Master 1/SEMESTER 2/Research Project Experimental Psychology/"
        "RPEP_experiment/analysis/plots/affinity_levels"
    )
    ro.r(f'dir.create("{plot_dir}", recursive=TRUE, showWarnings=FALSE)')

    for dv_label, dv_col in dv_specs:
        png_file = f'{plot_dir}/{dv_col}.png'
        ro.r(f'png("{png_file}", width=1200, height=1200)')
        ro.r('par(mfrow=c(3,3), mar=c(5,5,8,2))')

        for exp in exp_conditions:
            for task in task_types:
                if exp == "Both":
                    exp_filter = "df_sub <- df"
                    exp_label = "Both experimental conditions"
                    exp_key = "Both"
                else:
                    exp_filter = f'df_sub <- subset(df, experimental_condition == "{exp}")'
                    exp_label = exp
                    exp_key = exp
                if task == "Both":
                    task_filter = ""
                    task_label = "Both task types"
                    task_key = "Both"
                else:
                    task_filter = f'df_sub <- subset(df_sub, task_type == "{task}")'
                    task_label = task
                    task_key = task

                # Adjust key label for BI_score_TaskType so that the dictionary key matches.
                key_label = dv_label
                if dv_col == "BI_score_TaskType":
                    key_label = "BI_score_TaskType"
                key = (exp_key, task_key, key_label)
                if key in lme_results_affinity_index:
                    stats = lme_results_affinity_index[key]
                    scode = get_signif_code(stats["pvalue"])
                    annotation_text = f"Est={stats['estimate']:.3g}, p={stats['pvalue']:.2g}{scode}"
                else:
                    annotation_text = "No model info"

                color_data = dv_colors[dv_col]

                r_code = f'''
                {exp_filter}
                {task_filter}
                if(nrow(df_sub) > 0) {{
                  if("{dv_col}"=="BI_score" || "{dv_col}"=="BI_score_TaskType") {{
                    agg_data <- aggregate({dv_col} ~ affinity_score, data=df_sub, FUN={agg_fun})
                    x_vals <- agg_data$affinity_score
                    y_means <- as.numeric(sapply(agg_data[[ "{dv_col}" ]], function(x) strsplit(x, " ")[[1]][1]))
                  }} else {{
                    agg_data <- aggregate({dv_col} ~ affinity_score, data=df_sub, FUN={agg_fun})
                    x_vals <- agg_data$affinity_score
                    y_means <- as.numeric(sapply(agg_data${dv_col}, function(x) strsplit(x, " ")[[1]][1]))
                  }}
                  plot(x_vals, y_means, type="b", pch=16, cex=1.5, lwd=2, col="{color_data}",
                       main=paste("{dv_label}", "\\n", "{exp_label}", "&", "{task_label}"),
                       xlab="Affinity Score", ylab=paste("Mean", "{dv_label}"), cex.lab=1.2, cex.main=0.95)
                  grid(col="gray70"); box()
                  mtext(paste0("N = ", nrow(df_sub)), side=3, line=0.5, cex=0.8)
                  points(df_sub$affinity_score, if("{dv_col}"=="BI_score"){{ df_sub$BI_score }} else if("{dv_col}"=="BI_score_TaskType"){{ df_sub$BI_score_TaskType }} else{{ df_sub${dv_col} }}, pch=1, col="gray50", cex=0.7)
                  df_fit <- data.frame(x=x_vals, y=y_means)
                  fit <- lm(y ~ x, data=df_fit)
                  lines(x_vals, predict(fit), col="{reg_line_col}", lwd=2, lty=2)
                  legend("topright", legend=c("Data", "Regression"),
                         col=c("{color_data}", "{reg_line_col}"), lwd=c(2,2), lty=c(1,2), bty="n", cex=0.8)
                  usr <- par("usr")
                  corner_x <- usr[2] - 0.02*(usr[2]-usr[1])
                  corner_y <- usr[3] + 0.05*(usr[4]-usr[3])
                  text(corner_x, corner_y, "{annotation_text}",
                       col="{reg_line_col}", cex=0.9, adj=c(1,0))
                }} else {{
                  plot(NA, xlim=c(0,1), ylim=c(0,1),
                       main=paste("No data", "\\n", "{exp_label}", "&", "{task_label}"),
                       xlab="Affinity Score", ylab="{dv_label}", cex.lab=1.2, cex.main=0.95)
                  box()
                }}
                '''
                ro.r(r_code)
        ro.r('dev.off()')
        print(f"Saved enhanced affinity-level plot for {dv_col} to {png_file}")

def create_taskid_plots():
    """
    For each DV and for each task type ("Insight" and "RAT"), create a 3x4 grid PNG plot
    (one panel per task_id) showing aggregated mean DV by affinity score with a regression line.
    Raw data points are overlaid.
    """
    plot_dir = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
        "Master 1/SEMESTER 2/Research Project Experimental Psychology/"
        "RPEP_experiment/analysis/plots/affinity_levels/task_id_levels"
    )
    ro.r(f'dir.create("{plot_dir}", recursive=TRUE, showWarnings=FALSE)')
    for dv_label, dv_col in dv_specs:
        for task_type in ["Insight", "RAT"]:
            png_file = f'{plot_dir}/{dv_col}_{task_type}.png'
            ro.r(f'png("{png_file}", width=1200, height=1600)')
            ro.r('par(mfrow=c(4,3), mar=c(5,5,8,2))')
            for task_id in range(1, 13):
                r_code = f'''
                df_sub <- subset(df, task_type=="{task_type}" & task_id=={task_id})
                if(nrow(df_sub) > 0) {{
                  if("{dv_col}"=="BI_score" || "{dv_col}"=="BI_score_TaskType") {{
                    agg_data <- aggregate({dv_col} ~ affinity_score, data=df_sub, FUN={agg_fun})
                    x_vals <- agg_data$affinity_score
                    y_means <- as.numeric(sapply(agg_data[[ "{dv_col}" ]], function(x) strsplit(x, " ")[[1]][1]))
                  }} else {{
                    agg_data <- aggregate({dv_col} ~ affinity_score, data=df_sub, FUN={agg_fun})
                    x_vals <- agg_data$affinity_score
                    y_means <- as.numeric(sapply(agg_data${dv_col}, function(x) strsplit(x, " ")[[1]][1]))
                  }}
                  cat("Aggregated data for TaskID", {task_id}, ":\n")
                  print(agg_data)
                  plot(x_vals, y_means, type="b", pch=16, cex=1.5, lwd=2, col="{dv_colors[dv_col]}",
                       main=paste("{dv_label}", "\\nTaskID:", {task_id}),
                       xlab="Affinity Score", ylab=paste("Mean", "{dv_label}"), cex.lab=1.2, cex.main=0.95)
                  grid(col="gray70"); box()
                  mtext(paste("N =", nrow(df_sub)), side=3, line=0.5, cex=0.8)
                  points(df_sub$affinity_score, if("{dv_col}"=="BI_score"){{ df_sub$BI_score }} else if("{dv_col}"=="BI_score_TaskType"){{ df_sub$BI_score_TaskType }} else{{ df_sub${dv_col} }}, pch=1, col="gray50", cex=0.7)
                  df_fit <- data.frame(x=x_vals, y=y_means)
                  fit <- lm(y ~ x, data=df_fit)
                  lines(x_vals, predict(fit), col="{reg_line_col}", lwd=2, lty=2)
                }} else {{
                  plot(NA, xlim=c(0,1), ylim=c(0,1),
                       main=paste("No data", "\\nTaskID:", {task_id}),
                       xlab="Affinity Score", ylab="{dv_label}", cex.lab=1.2, cex.main=0.95)
                  box()
                }}
                '''
                ro.r(r_code)
            ro.r('dev.off()')
            print(f"Saved task_id-level plot for {dv_col} ({task_type}) to {png_file}")

# Run analyses and create plots.
run_analyses()
create_affinity_plots()
# Uncomment the following line to create task_id-level plots.
# create_taskid_plots()
