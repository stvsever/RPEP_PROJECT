import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel

# ----- R Integration Imports -----
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

# Activate R <-> Python data conversions
pandas2ri.activate()
numpy2ri.activate()

def get_signif_code(p_val):
    """Return significance code based on p-value thresholds."""
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    elif p_val < 0.1:
        return '.'
    else:
        return ' '

# ----- Function to run linear model using R -----
def run_linear_model_r(df, predictor, response, label):
    """
    Runs a linear model using R's lm function on the given dataframe.
    Prints a concise summary (only the coefficients table) with a clear label.
    """
    # Convert the DataFrame to an R dataframe
    r_df = pandas2ri.py2rpy(df)
    # Create the formula string (e.g., "response ~ predictor")
    formula = f"{response} ~ {predictor}"
    # Fit the linear model using R's lm function
    lm_model = ro.r.lm(formula, data=r_df)
    # Get the summary of the model and extract only the coefficients table
    summary_lm = ro.r.summary(lm_model)
    coeff_table = summary_lm.rx2('coefficients')
    print(f"\n[Linear Model: {label}]")
    print(coeff_table)
    print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

def remove_outliers(
        sub_df,
        x_col,
        y_col,
        subject_col='subject_id',
        threshold=3.0,
        correlation_label=None
):
    """
    Removes outliers in 'sub_df' where either x_col or y_col is beyond
    'threshold' standard deviations from their respective means.
    Returns (cleaned_df, outliers_df).

    Also prints which subject IDs were excluded as outliers.
    """
    # Compute mean and std
    x_mean, x_std = sub_df[x_col].mean(), sub_df[x_col].std()
    y_mean, y_std = sub_df[y_col].mean(), sub_df[y_col].std()

    # Identify outliers
    sub_df['x_outlier'] = ((sub_df[x_col] < x_mean - threshold * x_std) |
                           (sub_df[x_col] > x_mean + threshold * x_std))
    sub_df['y_outlier'] = ((sub_df[y_col] < y_mean - threshold * y_std) |
                           (sub_df[y_col] > y_mean + threshold * y_std))
    sub_df['is_outlier'] = sub_df['x_outlier'] | sub_df['y_outlier']

    # Split
    outliers = sub_df[sub_df['is_outlier']]
    clean_df = sub_df[~sub_df['is_outlier']].copy()

    # Print outliers
    if not outliers.empty:
        if correlation_label:
            print(f"Outliers detected (threshold={threshold} SD) for correlation: {correlation_label}")
        else:
            print("Outliers detected (threshold={threshold} SD away from mean) for correlation:")
        print(" - Outlier subject IDs:", outliers[subject_col].unique().tolist())
    else:
        if correlation_label:
            print(f"No outliers detected for correlation: {correlation_label}")
        else:
            print("No outliers detected.")

    # Drop the helper columns
    outliers.drop(columns=['x_outlier', 'y_outlier', 'is_outlier'], inplace=True, errors='ignore')
    clean_df.drop(columns=['x_outlier', 'y_outlier', 'is_outlier'], inplace=True, errors='ignore')

    return clean_df, outliers


def run_correlation_analysis(classical_excel_path: str, modern_excel_path: str, output_plot_dir: str):
    """
    Reads two Excel files (CLASSICAL and MODERN data), computes average performance
    metrics and affinity scores, produces plots for DFU and F affinity indices vs. performance
    metrics, calculates Pearson correlations and runs R linear models for each test.
    """
    os.makedirs(output_plot_dir, exist_ok=True)

    # ----- Load CLASSICAL data -----
    classical_df = pd.read_excel(classical_excel_path)
    req_cols = ['subject_id', 'experimental_condition', 'affinity_score', 'score_TEST', 'reaction_time_TEST']
    missing_c = [col for col in req_cols if col not in classical_df.columns]
    if missing_c:
        raise KeyError(f"Missing columns in CLASSICAL Excel: {missing_c}")

    # Average metrics per subject (over DFU and F)
    classical_metrics = (classical_df
                         .groupby('subject_id', as_index=False)[['score_TEST', 'reaction_time_TEST']]
                         .mean()
                         .rename(columns={
                             'score_TEST': 'score_TEST_classical',
                             'reaction_time_TEST': 'reaction_time_TEST_classical'
                         }))

    # Pivot affinity_score to conditions
    classical_affinity_pivot = (classical_df
                                .groupby(['subject_id', 'experimental_condition'])['affinity_score']
                                .mean()
                                .unstack()
                                .reset_index()
                                .rename_axis(None, axis=1))
    classical_affinity_pivot.rename(columns={
        'DFU': 'classical_dfu_affinity',
        'F': 'classical_f_affinity'
    }, inplace=True)

    # ----- Load MODERN data -----
    modern_df = pd.read_excel(modern_excel_path)
    missing_m = [col for col in req_cols if col not in modern_df.columns]
    if missing_m:
        raise KeyError(f"Missing columns in MODERN Excel: {missing_m}")

    # Average metrics per subject (over DFU and F)
    modern_metrics = (modern_df
                      .groupby('subject_id', as_index=False)[['score_TEST', 'reaction_time_TEST']]
                      .mean()
                      .rename(columns={
                          'score_TEST': 'score_TEST_modern',
                          'reaction_time_TEST': 'reaction_time_TEST_modern'
                      }))

    # Pivot affinity_score to conditions
    modern_affinity_pivot = (modern_df
                             .groupby(['subject_id', 'experimental_condition'])['affinity_score']
                             .mean()
                             .unstack()
                             .reset_index()
                             .rename_axis(None, axis=1))
    modern_affinity_pivot.rename(columns={
        'DFU': 'modern_dfu_affinity',
        'F': 'modern_f_affinity'
    }, inplace=True)

    # ----- Combine CLASSICAL & MODERN affinity data -----
    combined_affinity = pd.merge(classical_affinity_pivot, modern_affinity_pivot, on='subject_id', how='inner')
    combined_affinity['dfu_combined'] = combined_affinity[['classical_dfu_affinity', 'modern_dfu_affinity']].mean(axis=1)
    combined_affinity['f_combined'] = combined_affinity[['classical_f_affinity', 'modern_f_affinity']].mean(axis=1)
    combined_affinity['overall_combined'] = combined_affinity[['dfu_combined', 'f_combined']].mean(axis=1)

    # Add performance metrics from both datasets
    final_df = pd.merge(combined_affinity, classical_metrics, on='subject_id', how='inner')
    final_df = pd.merge(final_df, modern_metrics, on='subject_id', how='inner')

    # ----- Correlation and Plotting for DFU and F affinity indices -----
    affinity_measures = [
        ('dfu_combined', "DFU Affinity Index"),
        ('f_combined', "F Affinity Index"),
        ('overall_combined', "Overall Affinity Index")
    ]
    metrics = [
        ('score_TEST_classical', "Score (Classical Task)"),
        ('reaction_time_TEST_classical', "RT (Classical Task)"),
        ('score_TEST_modern', "Score (Modern Task)"),
        ('reaction_time_TEST_modern', "RT (Modern Task)")
    ]

    sns.set_style("whitegrid")
    colors = ['blue', 'green', 'red', 'purple']

    print("=== Correlation Summary (with outlier exclusion) ===")
    # For DFU and F visualizations
    for (aff_col, aff_label) in affinity_measures[:2]:
        fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=False)
        #fig.suptitle(f"{aff_label} vs. Insight Performance", fontsize=16)

        for i, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[i]
            plot_data = final_df[['subject_id', aff_col, metric_col]].dropna()

            if plot_data.empty:
                ax.set_title(f"No data for {metric_label}")
                ax.set_xlabel(aff_label)
                ax.set_ylabel(metric_label)
                continue

            correlation_desc = f"{aff_label} vs {metric_label}"
            clean_df, outliers_df = remove_outliers(
                plot_data.copy(),
                x_col=aff_col,
                y_col=metric_col,
                subject_col='subject_id',
                threshold=3.0,
                correlation_label=correlation_desc
            )

            if clean_df.empty:
                ax.set_title(f"All data outliers? {metric_label}")
                ax.set_xlabel(aff_label)
                ax.set_ylabel(metric_label)
                continue

            x_data = clean_df[aff_col]
            y_data = clean_df[metric_col]
            x_min, x_max = x_data.min(), x_data.max()
            ax.set_xlim(x_min, x_max)

            sns.regplot(
                x=x_data,
                y=y_data,
                ax=ax,
                scatter_kws={'alpha': 0.7, 'color': colors[i]},
                line_kws={'color': colors[i]}
            )
            r_val, p_val = pearsonr(x_data, y_data)
            signif = get_signif_code(p_val)
            ax.set_xlabel(aff_label)
            ax.set_ylabel(metric_label)
            # Append significance only to the p-value value
            ax.set_title(f"{metric_label}\nr={r_val:.3f}, p={p_val:.3f}{signif}")

            # ----- Run R linear model for this correlation -----
            run_linear_model_r(clean_df, predictor=aff_col, response=metric_col, label=correlation_desc)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_name = f"{aff_col}_Correlations.png"
        plt.savefig(os.path.join(output_plot_dir, save_name))
        plt.close()

        # Print numeric correlations
        print(f"\nAffinity: {aff_label}")
        for i, (metric_col, metric_label) in enumerate(metrics):
            data_for_corr = final_df[['subject_id', aff_col, metric_col]].dropna().copy()
            if data_for_corr.empty:
                print(f" - {metric_label}: No data available.")
                continue
            correlation_desc = f"{aff_label} vs {metric_label}"
            clean_df, outliers_df = remove_outliers(
                data_for_corr.copy(),
                x_col=aff_col,
                y_col=metric_col,
                subject_col='subject_id',
                threshold=3.0,
                correlation_label=correlation_desc
            )
            if clean_df.empty:
                print(f" - {metric_label}: All data flagged as outliers.")
                continue
            r_val, p_val = pearsonr(clean_df[aff_col], clean_df[metric_col])
            print(f" - {metric_label}: r={r_val:.3f}, p={p_val:.3f}")

    # ----- Correlation and Plot: DFU vs F Affinity Index -----
    print("\nCorrelation between DFU and F Affinity Index:")
    dfu_f_df = final_df[['subject_id', 'dfu_combined', 'f_combined']].dropna().copy()
    clean_df, _ = remove_outliers(
        dfu_f_df.copy(),
        x_col='dfu_combined',
        y_col='f_combined',
        subject_col='subject_id',
        threshold=3.0,
        correlation_label="DFU vs F Affinity"
    )
    if not clean_df.empty:
        r_val, p_val = pearsonr(clean_df['dfu_combined'], clean_df['f_combined'])
        print(f" - Pearson correlation: r={r_val:.3f}, p={p_val:.3f}")

        # Run R linear model for DFU vs F Affinity
        run_linear_model_r(clean_df, predictor='dfu_combined', response='f_combined', label="DFU vs F Affinity")

        # Plot DFU vs F Affinity Index
        plt.figure(figsize=(6, 6))
        sns.regplot(
            x=clean_df['dfu_combined'],
            y=clean_df['f_combined'],
            scatter_kws={'alpha': 0.7, 'color': 'grey'},
            line_kws={'color': 'grey'}
        )
        plt.xlabel("DFU Combined Affinity")
        plt.ylabel("F Combined Affinity")
        plt.title(f"DFU vs F Affinity\nr={r_val:.3f}, p={p_val:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "DFU_vs_F_Affinity.png"))
        plt.close()
    else:
        print(" - All data flagged as outliers.")

    # ----- Paired t-test: DFU vs F Affinity Scores -----
    paired_data = final_df[['subject_id', 'dfu_combined', 'f_combined']].dropna()
    if not paired_data.empty:
        t_stat, t_p = ttest_rel(paired_data['dfu_combined'], paired_data['f_combined'])
        print("\nPaired t-test for DFU vs F Affinity Scores:")
        print(f" - t-statistic: {t_stat:.3f}, p-value: {t_p:.3f}")

        dfu_mean = paired_data['dfu_combined'].mean()
        dfu_sd = paired_data['dfu_combined'].std()
        f_mean = paired_data['f_combined'].mean()
        f_sd = paired_data['f_combined'].std()
        delta = dfu_mean - f_mean
        print(f"\nDFU Affinity: mean = {dfu_mean:.3f}, SD = {dfu_sd:.3f}")
        print(f"F Affinity: mean = {f_mean:.3f}, SD = {f_sd:.3f}")
        print(f"Delta (DFU - F) = {delta:.3f}")

        plot_data = pd.melt(paired_data, id_vars=['subject_id'],
                            value_vars=['dfu_combined', 'f_combined'],
                            var_name='Condition', value_name='Affinity')
        # Map condition names
        plot_data['Condition'] = plot_data['Condition'].map({
            'dfu_combined': 'De Facto Unfalsifiability',
            'f_combined': 'Falsifiability'
        })

        # Modify violin plot: assign hue and remove legend if it exists
        plt.figure(figsize=(6, 6))
        ax = sns.violinplot(x='Condition', y='Affinity', data=plot_data, inner='box',
                            hue='Condition', palette=['skyblue', 'lightsalmon'])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.xlabel("Condition")
        plt.ylabel("Affinity Score")
        plt.title("Index comparison")

        y_min, y_max = ax.get_ylim()
        y_offset = 0.05 * (y_max - y_min)
        line_y = y_max - y_offset

        ax.plot([0, 1], [line_y, line_y], color='black', lw=1.5)
        ax.plot([0, 0], [line_y, line_y - y_offset / 2], color='black', lw=1.5)
        ax.plot([1, 1], [line_y, line_y - y_offset / 2], color='black', lw=1.5)

        ax.text(0.5, (line_y + y_offset / 4) - 0.05, f"Δ = {delta:.3f}, p = {t_p:.3f} ; n.s.",
                ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "Index_comparison.png"))
        plt.close()
    else:
        print("No data available for paired t-test between DFU and F Affinity Scores.")


def run_bi_tasktype_correlation_excel(aggregated_excel_path: str, output_plot_dir: str):
    """
    Reads the aggregated Excel file (which now includes BI_score_TaskType averages per subject and condition)
    and computes the Pearson correlation between the aggregated BI_score_TaskType and affinity_score for
    DFU and F conditions. It then visualizes the results with regression plots and runs R linear models.
    For this analysis, the regression plot has the affinity score on the x-axis and BI_score_TaskType on the y-axis.
    """
    os.makedirs(output_plot_dir, exist_ok=True)
    df = pd.read_excel(aggregated_excel_path)

    conditions = ['DFU', 'F']
    print("\n=== BI_score_TaskType vs. Affinity_score Correlation (Excel) ===")
    for cond in conditions:
        cond_df = df[df['experimental_condition'] == cond].dropna(subset=['BI_score_TaskType', 'affinity_score'])
        if cond_df.empty:
            print(f"No aggregated data available for condition {cond}")
            continue

        # Remove outliers
        clean_df, _ = remove_outliers(
            cond_df.copy(),
            x_col='BI_score_TaskType',
            y_col='affinity_score',
            subject_col='subject_id',
            threshold=3.0,
            correlation_label=f"Aggregated BI_taskID vs Affinity in {cond}"
        )

        if clean_df.empty:
            print(f"All aggregated data flagged as outliers for condition {cond}")
            continue

        # For BI_score_TaskType analysis, use affinity_score as the independent variable (predictor)
        r_val, p_val = pearsonr(clean_df['affinity_score'], clean_df['BI_score_TaskType'])
        signif = get_signif_code(p_val)
        print(f"Condition {cond} (Aggregated): Pearson r = {r_val:.3f}, p = {p_val:.3f}{signif}")

        # Run R linear model for aggregated BI_score_TaskType vs. Affinity
        # Here affinity_score is the predictor (IV) and BI_score_TaskType is the response (DV)
        run_linear_model_r(clean_df, predictor='affinity_score', response='BI_score_TaskType',
                           label=f"Aggregated BI_taskID vs Affinity in {cond}")

        # Plot correlation: x-axis = affinity_score, y-axis = BI_score_TaskType, using teal color
        plt.figure(figsize=(6, 6))
        sns.regplot(
            x=clean_df['affinity_score'],
            y=clean_df['BI_score_TaskType'],
            scatter_kws={'alpha': 0.7, 'color': 'teal'},
            line_kws={'color': 'teal'}
        )
        plt.xlabel("Aggregated Affinity Score")
        plt.ylabel("Aggregated BI-score")
        plt.title(f"{cond} affinity index\nr={r_val:.3f}, p={p_val:.4f}{signif}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, f"BI_TaskID_vs_Affinity_{cond}.png"))
        plt.close()


if __name__ == "__main__":
    # Paths for classical and modern aggregated Excel files
    classical_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/aggregated_by_subjectID_INSIGHT.xlsx"
    )
    modern_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/aggregated_by_subjectID_RAT.xlsx"
    )
    # Path for the BOTH aggregated Excel file (which includes BI_score_TaskType)
    both_path = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/aggregated_by_subjectID_BOTH.xlsx"
    )
    plot_dir = (
        "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/affinity_correlations"
    )

    run_correlation_analysis(classical_path, modern_path, plot_dir)
    # New part: BI_score_TaskType correlation from aggregated Excel file
    run_bi_tasktype_correlation_excel(both_path, plot_dir)
