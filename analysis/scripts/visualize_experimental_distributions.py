import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.lines import Line2D

###############################################################################
# 0) Directory Management & Safe Saving
###############################################################################
BASE_DIR = (
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
    "Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/"
    "analysis/plots/RT_distributions"
)


def save_and_close(fig_path: str):
    """
    Save the current matplotlib figure to fig_path and close it.
    Ensures subdirectories exist before saving.
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()


###############################################################################
# 1) Load Full Dataset
###############################################################################
df = pd.read_csv(
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
    "Master 1/SEMESTER 2/Research Project Experimental Psychology/"
    "RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
)


###############################################################################
# 2) Clipping Logic
#    - RAT tasks: only RT values between 0.2 and 19.9 are used
#    - Insight tasks: only RT values between 1 and 119 are used
###############################################################################
def clip_data(df_in):
    """
    Return a new DataFrame with:
      - RAT RT clipped between 0.2 and 19.9 seconds
      - Insight RT clipped between 1 and 119 seconds
    """
    df_rat = df_in[(df_in["task_type"] == "RAT") &
                   (df_in["reaction_time_TEST"] >= 0.2) &
                   (df_in["reaction_time_TEST"] <= 19.9)]
    df_insight = df_in[(df_in["task_type"] == "Insight") &
                       (df_in["reaction_time_TEST"] >= 1) &
                       (df_in["reaction_time_TEST"] <= 119)]
    return pd.concat([df_rat, df_insight], ignore_index=True)


df_clipped = clip_data(df)  # We'll use this for Part B analyses


###############################################################################
# 3) Plot Helper Functions
###############################################################################
def plot_hist_with_four_fits(
        data_dfu, data_f, bin_width, title, fig_path,
        poly_order=3
):
    """
    Plots:
      - DFU histogram (blue) and F histogram (orange)
      - Polynomial fits for DFU (blue dashed) and F (orange dashed)
      - Chi2 fits for DFU (blue dotted, thick) and F (orange dotted, thick)
      - ex-Gaussian fits for DFU (blue dash-dot) and F (orange dash-dot)
    A custom legend is added with 4 items: "DFU", "F", "Polynomial", "Chi2"
    (the fits in the legend are shown in grey).
    """
    dfu_vals = np.array(data_dfu.dropna())
    f_vals = np.array(data_f.dropna())

    if dfu_vals.size == 0 and f_vals.size == 0:
        print(f"[WARNING] No data found for {title}")
        return

    combined = np.concatenate([dfu_vals, f_vals])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        max_val += bin_width

    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([min_val, max_val])

    plt.figure(figsize=(8, 5))

    # Histograms
    plt.hist(dfu_vals, bins=bins, density=True, alpha=0.5,
             color='tab:blue', label="DFU")
    plt.hist(f_vals, bins=bins, density=True, alpha=0.5,
             color='tab:orange', label="F")

    # Polynomial fits (ignoring first histogram bin)
    # DFU polynomial
    hist_dfu, edges_dfu = np.histogram(dfu_vals, bins=bins, density=True)
    centers_dfu = 0.5 * (edges_dfu[:-1] + edges_dfu[1:])
    if len(hist_dfu) > 1:
        hist_dfu = hist_dfu[1:]
        centers_dfu = centers_dfu[1:]
    p_dfu = None
    if len(hist_dfu) > poly_order:
        p_dfu = np.polyfit(centers_dfu, hist_dfu, poly_order)
    # F polynomial
    hist_f, edges_f = np.histogram(f_vals, bins=bins, density=True)
    centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])
    if len(hist_f) > 1:
        hist_f = hist_f[1:]
        centers_f = centers_f[1:]
    p_f = None
    if len(hist_f) > poly_order:
        p_f = np.polyfit(centers_f, hist_f, poly_order)
    x_fit = np.linspace(min_val, max_val, 300)
    if p_dfu is not None:
        y_dfu = np.polyval(p_dfu, x_fit)
        plt.plot(x_fit, y_dfu, '--', color='tab:blue', linewidth=1,
                 label='_nolegend_')
    if p_f is not None:
        y_f = np.polyval(p_f, x_fit)
        plt.plot(x_fit, y_f, '--', color='tab:orange', linewidth=1,
                 label='_nolegend_')

    # Chi-square Fits
    if dfu_vals.size > 1:
        try:
            shape_dfu, loc_dfu, scale_dfu = st.chi2.fit(dfu_vals, floc=0)
            y_pdf = st.chi2.pdf(x_fit, shape_dfu, loc=loc_dfu, scale=scale_dfu)
            plt.plot(x_fit, y_pdf, ':', color='tab:blue', linewidth=5,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] DFU chi2 fit failed for {title}: {e}")
    if f_vals.size > 1:
        try:
            shape_f, loc_f, scale_f = st.chi2.fit(f_vals, floc=0)
            y_pdf = st.chi2.pdf(x_fit, shape_f, loc=loc_f, scale=scale_f)
            plt.plot(x_fit, y_pdf, ':', color='tab:orange', linewidth=5,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] F chi2 fit failed for {title}: {e}")

    # ex-Gaussian Fits
    if dfu_vals.size > 1:
        try:
            K_dfu, loc_ex_dfu, scale_ex_dfu = st.exponnorm.fit(dfu_vals, floc=0)
            y_pdf_ex = st.exponnorm.pdf(x_fit, K_dfu, loc_ex_dfu, scale_ex_dfu)
            plt.plot(x_fit, y_pdf_ex, '-.', color='tab:blue', linewidth=1,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] DFU ex-Gaussian fit failed for {title}: {e}")
    if f_vals.size > 1:
        try:
            K_f, loc_ex_f, scale_ex_f = st.exponnorm.fit(f_vals, floc=0)
            y_pdf_ex = st.exponnorm.pdf(x_fit, K_f, loc_ex_f, scale_ex_f)
            plt.plot(x_fit, y_pdf_ex, '-.', color='tab:orange', linewidth=1,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] F ex-Gaussian fit failed for {title}: {e}")

    # Custom Legend: uses grey for the fit lines
    custom_handles = [
        Line2D([0], [0], color='tab:blue', lw=4, label="DFU"),
        Line2D([0], [0], color='tab:orange', lw=4, label="F"),
        Line2D([0], [0], color='grey', lw=1, linestyle='--', label="Polynomial"),
        Line2D([0], [0], color='grey', lw=5, linestyle=':', label="Chi2")
    ]
    plt.title(title, fontsize=14)
    plt.xlabel("Reaction Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(handles=custom_handles, fontsize=12)
    plt.tight_layout()
    save_and_close(fig_path)


def plot_hist_with_poly_chi(
        data_dfu, data_f, bin_width, title, fig_path,
        poly_order=3
):
    """
    Plots histogram, polynomial, and chi2 fits.
    Legend shows: "DFU", "F", "Polynomial", "Chi2".
    """
    dfu_vals = np.array(data_dfu.dropna())
    f_vals = np.array(data_f.dropna())

    if dfu_vals.size == 0 and f_vals.size == 0:
        print(f"[WARNING] No data found for {title}")
        return

    combined = np.concatenate([dfu_vals, f_vals])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        max_val += bin_width

    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([min_val, max_val])

    plt.figure(figsize=(8, 5))
    # Histograms
    plt.hist(dfu_vals, bins=bins, density=True, alpha=0.5,
             color='tab:blue', label="DFU")
    plt.hist(f_vals, bins=bins, density=True, alpha=0.5,
             color='tab:orange', label="F")

    # Polynomial fits (ignoring the first bin)
    hist_dfu, edges_dfu = np.histogram(dfu_vals, bins=bins, density=True)
    centers_dfu = 0.5 * (edges_dfu[:-1] + edges_dfu[1:])
    if len(hist_dfu) > 1:
        hist_dfu = hist_dfu[1:]
        centers_dfu = centers_dfu[1:]
    p_dfu = None
    if len(hist_dfu) > poly_order:
        p_dfu = np.polyfit(centers_dfu, hist_dfu, poly_order)
    hist_f, edges_f = np.histogram(f_vals, bins=bins, density=True)
    centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])
    if len(hist_f) > 1:
        hist_f = hist_f[1:]
        centers_f = centers_f[1:]
    p_f = None
    if len(hist_f) > poly_order:
        p_f = np.polyfit(centers_f, hist_f, poly_order)
    x_fit = np.linspace(min_val, max_val, 300)
    if p_dfu is not None:
        y_dfu = np.polyval(p_dfu, x_fit)
        plt.plot(x_fit, y_dfu, '--', color='tab:blue', linewidth=1,
                 label='_nolegend_')
    if p_f is not None:
        y_f = np.polyval(p_f, x_fit)
        plt.plot(x_fit, y_f, '--', color='tab:orange', linewidth=1,
                 label='_nolegend_')

    # Chi-square Fits
    if dfu_vals.size > 1:
        try:
            shape_dfu, loc_dfu, scale_dfu = st.chi2.fit(dfu_vals, floc=0)
            y_pdf = st.chi2.pdf(x_fit, shape_dfu, loc=loc_dfu, scale=scale_dfu)
            plt.plot(x_fit, y_pdf, ':', color='tab:blue', linewidth=5,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] DFU chi2 fit failed for {title}: {e}")
    if f_vals.size > 1:
        try:
            shape_f, loc_f, scale_f = st.chi2.fit(f_vals, floc=0)
            y_pdf = st.chi2.pdf(x_fit, shape_f, loc=loc_f, scale=scale_f)
            plt.plot(x_fit, y_pdf, ':', color='tab:orange', linewidth=5,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] F chi2 fit failed for {title}: {e}")

    # Custom Legend
    custom_handles = [
        Line2D([0], [0], color='tab:blue', lw=4, label="DFU"),
        Line2D([0], [0], color='tab:orange', lw=4, label="F"),
        Line2D([0], [0], color='grey', lw=1, linestyle='--', label="Polynomial"),
        Line2D([0], [0], color='grey', lw=5, linestyle=':', label="Chi2")
    ]
    plt.title(title, fontsize=14)
    plt.xlabel("Reaction Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(handles=custom_handles, fontsize=12)
    plt.tight_layout()
    save_and_close(fig_path)


def plot_hist_with_poly_ex(
        data_dfu, data_f, bin_width, title, fig_path,
        poly_order=3
):
    """
    Plots histogram, polynomial, and ex-Gaussian fits.
    Legend shows: "DFU", "F", "Polynomial", "ex-Gaussian".
    """
    dfu_vals = np.array(data_dfu.dropna())
    f_vals = np.array(data_f.dropna())

    if dfu_vals.size == 0 and f_vals.size == 0:
        print(f"[WARNING] No data found for {title}")
        return

    combined = np.concatenate([dfu_vals, f_vals])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        max_val += bin_width

    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([min_val, max_val])

    plt.figure(figsize=(8, 5))
    # Histograms
    plt.hist(dfu_vals, bins=bins, density=True, alpha=0.5,
             color='tab:blue', label="DFU")
    plt.hist(f_vals, bins=bins, density=True, alpha=0.5,
             color='tab:orange', label="F")

    # Polynomial fits (ignoring the first bin)
    hist_dfu, edges_dfu = np.histogram(dfu_vals, bins=bins, density=True)
    centers_dfu = 0.5 * (edges_dfu[:-1] + edges_dfu[1:])
    if len(hist_dfu) > 1:
        hist_dfu = hist_dfu[1:]
        centers_dfu = centers_dfu[1:]
    p_dfu = None
    if len(hist_dfu) > poly_order:
        p_dfu = np.polyfit(centers_dfu, hist_dfu, poly_order)
    hist_f, edges_f = np.histogram(f_vals, bins=bins, density=True)
    centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])
    if len(hist_f) > 1:
        hist_f = hist_f[1:]
        centers_f = centers_f[1:]
    p_f = None
    if len(hist_f) > poly_order:
        p_f = np.polyfit(centers_f, hist_f, poly_order)
    x_fit = np.linspace(min_val, max_val, 300)
    if p_dfu is not None:
        y_dfu = np.polyval(p_dfu, x_fit)
        plt.plot(x_fit, y_dfu, '--', color='tab:blue', linewidth=2,
                 label='_nolegend_')
    if p_f is not None:
        y_f = np.polyval(p_f, x_fit)
        plt.plot(x_fit, y_f, '--', color='tab:orange', linewidth=2,
                 label='_nolegend_')

    # ex-Gaussian Fits
    if dfu_vals.size > 1:
        try:
            K_dfu, loc_ex_dfu, scale_ex_dfu = st.exponnorm.fit(dfu_vals, floc=0)
            y_pdf_ex = st.exponnorm.pdf(x_fit, K_dfu, loc_ex_dfu, scale_ex_dfu)
            plt.plot(x_fit, y_pdf_ex, '-.', color='tab:blue', linewidth=1,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] DFU ex-Gaussian fit failed for {title}: {e}")
    if f_vals.size > 1:
        try:
            K_f, loc_ex_f, scale_ex_f = st.exponnorm.fit(f_vals, floc=0)
            y_pdf_ex = st.exponnorm.pdf(x_fit, K_f, loc_ex_f, scale_ex_f)
            plt.plot(x_fit, y_pdf_ex, '-.', color='tab:orange', linewidth=1,
                     label='_nolegend_')
        except Exception as e:
            print(f"[WARNING] F ex-Gaussian fit failed for {title}: {e}")

    # Custom Legend
    custom_handles = [
        Line2D([0], [0], color='tab:blue', lw=4, label="DFU"),
        Line2D([0], [0], color='tab:orange', lw=4, label="F"),
        Line2D([0], [0], color='grey', lw=2, linestyle='--', label="Polynomial"),
        Line2D([0], [0], color='grey', lw=1, linestyle='-.', label="ex-Gaussian")
    ]
    plt.title(title, fontsize=14)
    plt.xlabel("Reaction Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(handles=custom_handles, fontsize=12)
    plt.tight_layout()
    save_and_close(fig_path)


def plot_hist_poly_only(
        data_dfu, data_f, bin_width, title, fig_path,
        poly_order=3
):
    """
    Plots histogram and a polynomial fit ONLY (using very thick lines).
    Legend shows: "DFU", "F", "Polynomial".
    """
    dfu_vals = np.array(data_dfu.dropna())
    f_vals = np.array(data_f.dropna())

    if dfu_vals.size == 0 and f_vals.size == 0:
        print(f"[WARNING] No data found for {title}")
        return

    combined = np.concatenate([dfu_vals, f_vals])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        max_val += bin_width

    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([min_val, max_val])

    plt.figure(figsize=(8, 5))
    # Histograms
    plt.hist(dfu_vals, bins=bins, density=True, alpha=0.5,
             color='tab:blue', label="DFU")
    plt.hist(f_vals, bins=bins, density=True, alpha=0.5,
             color='tab:orange', label="F")

    # Polynomial fits (ignoring the first bin)
    # DFU polynomial
    hist_dfu, edges_dfu = np.histogram(dfu_vals, bins=bins, density=True)
    centers_dfu = 0.5 * (edges_dfu[:-1] + edges_dfu[1:])
    if len(hist_dfu) > 1:
        hist_dfu = hist_dfu[1:]
        centers_dfu = centers_dfu[1:]
    p_dfu = None
    if len(hist_dfu) > poly_order:
        p_dfu = np.polyfit(centers_dfu, hist_dfu, poly_order)

    # F polynomial
    hist_f, edges_f = np.histogram(f_vals, bins=bins, density=True)
    centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])
    if len(hist_f) > 1:
        hist_f = hist_f[1:]
        centers_f = centers_f[1:]
    p_f = None
    if len(hist_f) > poly_order:
        p_f = np.polyfit(centers_f, hist_f, poly_order)

    x_fit = np.linspace(min_val, max_val, 300)
    if p_dfu is not None:
        y_dfu = np.polyval(p_dfu, x_fit)
        plt.plot(x_fit, y_dfu, '--', color='tab:blue', linewidth=5,
                 label='_nolegend_')
    if p_f is not None:
        y_f = np.polyval(p_f, x_fit)
        plt.plot(x_fit, y_f, '--', color='tab:orange', linewidth=5,
                 label='_nolegend_')

    # Custom Legend: polynomial fit shown in grey with a thick line
    custom_handles = [
        Line2D([0], [0], color='tab:blue', lw=4, label="DFU"),
        Line2D([0], [0], color='tab:orange', lw=4, label="F"),
        Line2D([0], [0], color='grey', lw=3, linestyle='--', label="Polynomial")
    ]
    plt.title(title, fontsize=14)
    plt.xlabel("Reaction Time (s)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(handles=custom_handles, fontsize=12)
    plt.tight_layout()
    save_and_close(fig_path)


###############################################################################
# 4) Helper Functions: RAT vs. Insight subsets
###############################################################################
def get_rat_data_all(df_in):
    """Return RAT subset from the full DataFrame (no clip)."""
    return df_in[df_in["task_type"] == "RAT"]


def get_insight_data_all(df_in):
    """Return Insight subset from the full DataFrame (no clip)."""
    return df_in[df_in["task_type"] == "Insight"]


def get_rat_data_clipped(df_in):
    """Return RAT subset, clipped between 0.2 and 19.9s."""
    return df_in[(df_in["task_type"] == "RAT") &
                 (df_in["reaction_time_TEST"] >= 0.2) &
                 (df_in["reaction_time_TEST"] <= 20)]


def get_insight_data_clipped(df_in):
    """Return Insight subset, clipped between 1 and 119s."""
    return df_in[(df_in["task_type"] == "Insight") &
                 (df_in["reaction_time_TEST"] >= 1) &
                 (df_in["reaction_time_TEST"] <= 120)]


###############################################################################
# 5) PART A: "ALL DATA" (no clipping)
###############################################################################
print("=== PART A: Using ALL data (no clipping) ===")
all_data_dir = os.path.join(BASE_DIR, "AllData")

# 5.1: RAT Trial-level
df_rat_all = get_rat_data_all(df)
df_rat_all_dfu = df_rat_all[df_rat_all["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_rat_all_f = df_rat_all[df_rat_all["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_rat_all_dfu,
    data_f=df_rat_all_f,
    bin_width=0.1,
    title="RAT - Trial Level (All Data)",
    fig_path=os.path.join(all_data_dir, "trial_level_RAT.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_rat_all_dfu,
    data_f=df_rat_all_f,
    bin_width=0.1,
    title="RAT - Trial Level (All Data, Poly+Chi2)",
    fig_path=os.path.join(all_data_dir, "trial_level_RAT_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_rat_all_dfu,
    data_f=df_rat_all_f,
    bin_width=0.1,
    title="RAT - Trial Level (All Data, Poly+ExGauss)",
    fig_path=os.path.join(all_data_dir, "trial_level_RAT_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_rat_all_dfu,
    data_f=df_rat_all_f,
    bin_width=0.1,
    title="RAT - Trial Level (All Data, Poly Only)",
    fig_path=os.path.join(all_data_dir, "trial_level_RAT_polyonly.png")
)

# 5.2: Insight Trial-level
df_in_all = get_insight_data_all(df)
df_in_all_dfu = df_in_all[df_in_all["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_in_all_f = df_in_all[df_in_all["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_in_all_dfu,
    data_f=df_in_all_f,
    bin_width=1.0,
    title="Insight - Trial Level (All Data)",
    fig_path=os.path.join(all_data_dir, "trial_level_Insight.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_in_all_dfu,
    data_f=df_in_all_f,
    bin_width=1.0,
    title="Insight - Trial Level (All Data, Poly+Chi2)",
    fig_path=os.path.join(all_data_dir, "trial_level_Insight_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_in_all_dfu,
    data_f=df_in_all_f,
    bin_width=1.0,
    title="Insight - Trial Level (All Data, Poly+ExGauss)",
    fig_path=os.path.join(all_data_dir, "trial_level_Insight_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_in_all_dfu,
    data_f=df_in_all_f,
    bin_width=1.0,
    title="Insight - Trial Level (All Data, Poly Only)",
    fig_path=os.path.join(all_data_dir, "trial_level_Insight_polyonly.png")
)

# 5.3: RAT Aggregated (per participant)
df_rat_agg_all = (
    df_rat_all.groupby(["participant_identification", "experimental_condition"])["reaction_time_TEST"]
    .mean().reset_index()
)
df_rat_agg_all_dfu = df_rat_agg_all[df_rat_agg_all["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_rat_agg_all_f = df_rat_agg_all[df_rat_agg_all["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_rat_agg_all_dfu,
    data_f=df_rat_agg_all_f,
    bin_width=0.1,
    title="RAT - Aggregated by Participant (All Data)",
    fig_path=os.path.join(all_data_dir, "aggregated_RAT.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_rat_agg_all_dfu,
    data_f=df_rat_agg_all_f,
    bin_width=0.1,
    title="RAT - Aggregated (All Data, Poly+Chi2)",
    fig_path=os.path.join(all_data_dir, "aggregated_RAT_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_rat_agg_all_dfu,
    data_f=df_rat_agg_all_f,
    bin_width=0.1,
    title="RAT - Aggregated (All Data, Poly+ExGauss)",
    fig_path=os.path.join(all_data_dir, "aggregated_RAT_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_rat_agg_all_dfu,
    data_f=df_rat_agg_all_f,
    bin_width=0.1,
    title="RAT - Aggregated (All Data, Poly Only)",
    fig_path=os.path.join(all_data_dir, "aggregated_RAT_polyonly.png")
)

# 5.4: Insight Aggregated (per participant)
df_in_agg_all = (
    df_in_all.groupby(["participant_identification", "experimental_condition"])["reaction_time_TEST"]
    .mean().reset_index()
)
df_in_agg_all_dfu = df_in_agg_all[df_in_agg_all["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_in_agg_all_f = df_in_agg_all[df_in_agg_all["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_in_agg_all_dfu,
    data_f=df_in_agg_all_f,
    bin_width=1.0,
    title="Insight - Aggregated by Participant (All Data)",
    fig_path=os.path.join(all_data_dir, "aggregated_Insight.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_in_agg_all_dfu,
    data_f=df_in_agg_all_f,
    bin_width=1.0,
    title="Insight - Aggregated (All Data, Poly+Chi2)",
    fig_path=os.path.join(all_data_dir, "aggregated_Insight_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_in_agg_all_dfu,
    data_f=df_in_agg_all_f,
    bin_width=1.0,
    title="Insight - Aggregated (All Data, Poly+ExGauss)",
    fig_path=os.path.join(all_data_dir, "aggregated_Insight_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_in_agg_all_dfu,
    data_f=df_in_agg_all_f,
    bin_width=1.0,
    title="Insight - Aggregated (All Data, Poly Only)",
    fig_path=os.path.join(all_data_dir, "aggregated_Insight_polyonly.png")
)

# 5.5: By Task ID (all data)
unique_task_ids = sorted(df["task_id"].unique())
taskid_all_dir = os.path.join(all_data_dir, "by_task_id")
for t_id in unique_task_ids:
    df_task = df[df["task_id"] == t_id]
    data_dfu = df_task[df_task["experimental_condition"] == "DFU"]["reaction_time_TEST"]
    data_f = df_task[df_task["experimental_condition"] == "F"]["reaction_time_TEST"]
    this_bin = 0.1 if t_id.startswith("R") else 1.0
    base_title = f"Task ID {t_id} (All Data)"
    base_path = os.path.join(taskid_all_dir, t_id)
    plot_hist_with_four_fits(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title,
        fig_path=base_path + ".png"
    )
    plot_hist_with_poly_chi(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly+Chi2)",
        fig_path=base_path + "_polychi.png"
    )
    plot_hist_with_poly_ex(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly+ExGauss)",
        fig_path=base_path + "_polyex.png"
    )
    plot_hist_poly_only(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly Only)",
        fig_path=base_path + "_polyonly.png"
    )

###############################################################################
# 6) PART B: "CLIPPED DATA" (RAT: 0.2-19.9, Insight: 1-119)
###############################################################################
print("\n=== PART B: Using CLIPPED data (RAT: 0.2-19.9, Insight: 1-119) ===")
clipped_dir = os.path.join(BASE_DIR, "Clipped")

# 6.1: RAT Trial-level (clipped)
df_rat_clip = get_rat_data_clipped(df)
df_rat_clip_dfu = df_rat_clip[df_rat_clip["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_rat_clip_f = df_rat_clip[df_rat_clip["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_rat_clip_dfu,
    data_f=df_rat_clip_f,
    bin_width=0.1,
    title="RAT - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_RAT.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_rat_clip_dfu,
    data_f=df_rat_clip_f,
    bin_width=0.1,
    title="RAT - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_RAT_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_rat_clip_dfu,
    data_f=df_rat_clip_f,
    bin_width=0.1,
    title="Remote Associates' Tasks - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_RAT_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_rat_clip_dfu,
    data_f=df_rat_clip_f,
    bin_width=0.1,
    title="RAT - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_RAT_polyonly.png")
)

# 6.2: Insight Trial-level (clipped)
df_in_clip = get_insight_data_clipped(df)
df_in_clip_dfu = df_in_clip[df_in_clip["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_in_clip_f = df_in_clip[df_in_clip["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_in_clip_dfu,
    data_f=df_in_clip_f,
    bin_width=1.0,
    title="Insight - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_Insight.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_in_clip_dfu,
    data_f=df_in_clip_f,
    bin_width=1.0,
    title="Insight - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_Insight_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_in_clip_dfu,
    data_f=df_in_clip_f,
    bin_width=1.0,
    title="Classical Insight Tasks - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_Insight_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_in_clip_dfu,
    data_f=df_in_clip_f,
    bin_width=1.0,
    title="Insight - Trial Level",
    fig_path=os.path.join(clipped_dir, "trial_level_Insight_polyonly.png")
)

# 6.3: RAT Aggregated (clipped)
df_rat_clip_agg = (
    df_rat_clip.groupby(["participant_identification", "experimental_condition"])["reaction_time_TEST"]
    .mean().reset_index()
)
df_rat_clip_agg_dfu = df_rat_clip_agg[df_rat_clip_agg["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_rat_clip_agg_f = df_rat_clip_agg[df_rat_clip_agg["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_rat_clip_agg_dfu,
    data_f=df_rat_clip_agg_f,
    bin_width=0.1,
    title="RAT - Aggregated by Participant",
    fig_path=os.path.join(clipped_dir, "aggregated_RAT.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_rat_clip_agg_dfu,
    data_f=df_rat_clip_agg_f,
    bin_width=0.1,
    title="RAT - Aggregated",
    fig_path=os.path.join(clipped_dir, "aggregated_RAT_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_rat_clip_agg_dfu,
    data_f=df_rat_clip_agg_f,
    bin_width=0.1,
    title="RAT - Aggregated",
    fig_path=os.path.join(clipped_dir, "aggregated_RAT_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_rat_clip_agg_dfu,
    data_f=df_rat_clip_agg_f,
    bin_width=0.1,
    title="RAT - Aggregated",
    fig_path=os.path.join(clipped_dir, "aggregated_RAT_polyonly.png")
)

# 6.4: Insight Aggregated (clipped)
df_in_clip_agg = (
    df_in_clip.groupby(["participant_identification", "experimental_condition"])["reaction_time_TEST"]
    .mean().reset_index()
)
df_in_clip_agg_dfu = df_in_clip_agg[df_in_clip_agg["experimental_condition"] == "DFU"]["reaction_time_TEST"]
df_in_clip_agg_f = df_in_clip_agg[df_in_clip_agg["experimental_condition"] == "F"]["reaction_time_TEST"]
plot_hist_with_four_fits(
    data_dfu=df_in_clip_agg_dfu,
    data_f=df_in_clip_agg_f,
    bin_width=1.0,
    title="Insight - Aggregated by Participant",
    fig_path=os.path.join(clipped_dir, "aggregated_Insight.png")
)
plot_hist_with_poly_chi(
    data_dfu=df_in_clip_agg_dfu,
    data_f=df_in_clip_agg_f,
    bin_width=1.0,
    title="Insight - Aggregated",
    fig_path=os.path.join(clipped_dir, "aggregated_Insight_polychi.png")
)
plot_hist_with_poly_ex(
    data_dfu=df_in_clip_agg_dfu,
    data_f=df_in_clip_agg_f,
    bin_width=1.0,
    title="Insight - Aggregated",
    fig_path=os.path.join(clipped_dir, "aggregated_Insight_polyex.png")
)
plot_hist_poly_only(
    data_dfu=df_in_clip_agg_dfu,
    data_f=df_in_clip_agg_f,
    bin_width=1.0,
    title="Insight - Aggregated)",
    fig_path=os.path.join(clipped_dir, "aggregated_Insight_polyonly.png")
)

# 6.5: By Task ID (clipped)
taskid_clip_dir = os.path.join(clipped_dir, "by_task_id")
for t_id in unique_task_ids:
    if t_id.startswith("R"):
        df_task_clip = df[
            (df["task_id"] == t_id) & (df["reaction_time_TEST"] >= 0.2) & (df["reaction_time_TEST"] <= 20)]
        this_bin = 0.1
    else:
        df_task_clip = df[(df["task_id"] == t_id) & (df["reaction_time_TEST"] >= 1) & (df["reaction_time_TEST"] <= 120)]
        this_bin = 1.0
    data_dfu = df_task_clip[df_task_clip["experimental_condition"] == "DFU"]["reaction_time_TEST"]
    data_f = df_task_clip[df_task_clip["experimental_condition"] == "F"]["reaction_time_TEST"]
    base_title = f"Task ID {t_id} (Clipped)"
    base_path = os.path.join(taskid_clip_dir, t_id)
    plot_hist_with_four_fits(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title,
        fig_path=base_path + ".png"
    )
    plot_hist_with_poly_chi(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly+Chi2)",
        fig_path=base_path + "_polychi.png"
    )
    plot_hist_with_poly_ex(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly+ExGauss)",
        fig_path=base_path + "_polyex.png"
    )
    plot_hist_poly_only(
        data_dfu=data_dfu,
        data_f=data_f,
        bin_width=this_bin,
        title=base_title + " (Poly Only)",
        fig_path=base_path + "_polyonly.png"
    )

# Note ; increased upper clipping to max 20s for modern ITs and 120s for classical ITs