import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

# --------------------------------------------------------------------
# Global output directory for saving plots
# --------------------------------------------------------------------
output_base = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/analysis/plots/RT_distributions/task_ids"

# Helper to create and return a directory path
def get_output_dir(plot_type, task_type, title_suffix):
    # If title_suffix contains "Clipped", use a "clipped" folder, otherwise "full"
    condition_folder = "clipped" if "Clipped" in title_suffix else "full"
    # Remove spaces and hyphens from title_suffix for filename clarity
    suffix_clean = title_suffix.strip().replace(" ", "_").replace("-", "")
    # Construct the directory path: output_base/plot_type/task_type/condition_folder
    dir_path = os.path.join(output_base, plot_type, task_type, condition_folder)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# New helper: Append subdirectory for histogram plots.
def get_histogram_output_dir(plot_type, task_type, title_suffix, with_histogram=True):
    base_dir = get_output_dir(plot_type, task_type, title_suffix)
    subfolder = "with_histogram" if with_histogram else "without_histogram"
    hist_dir = os.path.join(base_dir, subfolder)
    os.makedirs(hist_dir, exist_ok=True)
    return hist_dir

# --------------------------------------------------------------------
# 1) Load Data
# --------------------------------------------------------------------
df = pd.read_csv(
    "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv"
)
df = df.dropna()

# The Welch t-test results dictionary (for reference in some plots)
Unpaired_ttest_Welch_taskID_results = {
    "RAT": {
        "R8": {
            "Score_TEST": {
                "DFU": {"mean": 0.1096091, "sd": 0.1596912},
                "F": {"mean": 0.1066126, "sd": 0.1911686},
                "p_value": 0.9492
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 15.23753, "sd": 4.219967},
                "F": {"mean": 12.591244, "sd": 5.705246},
                "p_value": 0.05588
            }
        },
        "R11": {
            "Score_TEST": {
                "DFU": {"mean": 0.07252699, "sd": 0.05049347},
                "F": {"mean": 0.08132056, "sd": 0.05043895},
                "p_value": 0.5088
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 15.93832, "sd": 3.719966},
                "F": {"mean": 13.392687, "sd": 5.490167},
                "p_value": 0.03656
            }
        },
        "R9": {
            "Score_TEST": {
                "DFU": {"mean": 0.2095967, "sd": 0.3452606},
                "F": {"mean": 0.2885227, "sd": 0.4015483},
                "p_value": 0.4172
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 13.68458, "sd": 5.075225},
                "F": {"mean": 12.670861, "sd": 4.2811},
                "p_value": 0.4164
            }
        },
        "R4": {
            "Score_TEST": {
                "DFU": {"mean": 0.3268194, "sd": 0.4243347},
                "F": {"mean": 0.2071422, "sd": 0.3556203},
                "p_value": 0.2431
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 13.03051, "sd": 4.67223},
                "F": {"mean": 14.21272, "sd": 4.847901},
                "p_value": 0.3402
            }
        },
        "R7": {
            "Score_TEST": {
                "DFU": {"mean": 0.1654493, "sd": 0.3161429},
                "F": {"mean": 0.1242085, "sd": 0.2540656},
                "p_value": 0.5775
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 15.37476, "sd": 5.285207},
                "F": {"mean": 14.59264, "sd": 4.627146},
                "p_value": 0.5437
            }
        },
        "R5": {
            "Score_TEST": {
                "DFU": {"mean": 0.1901988, "sd": 0.2940834},
                "F": {"mean": 0.2575560, "sd": 0.3740369},
                "p_value": 0.4544
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 15.79809, "sd": 4.817521},
                "F": {"mean": 14.86866, "sd": 4.365361},
                "p_value": 0.4793
            }
        },
        "R2": {
            "Score_TEST": {
                "DFU": {"mean": 0.1624128, "sd": 0.2856364},
                "F": {"mean": 0.1089617, "sd": 0.1723013},
                "p_value": 0.3845
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 13.99083, "sd": 4.392193},
                "F": {"mean": 15.03112, "sd": 3.986541},
                "p_value": 0.3408
            }
        },
        "R10": {
            "Score_TEST": {
                "DFU": {"mean": 0.07831987, "sd": 0.04655139},
                "F": {"mean": 0.07350350, "sd": 0.03919808},
                "p_value": 0.6655
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 16.12642, "sd": 4.439874},
                "F": {"mean": 15.50170, "sd": 4.437834},
                "p_value": 0.5913
            }
        },
        "R12": {
            "Score_TEST": {
                "DFU": {"mean": 0.08819911, "sd": 0.16960557},
                "F": {"mean": 0.13403680, "sd": 0.25178437},
                "p_value": 0.4235
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 16.51343, "sd": 3.65281},
                "F": {"mean": 14.95652, "sd": 4.557966},
                "p_value": 0.157
            }
        },
        "R6": {
            "Score_TEST": {
                "DFU": {"mean": 0.1587220, "sd": 0.2903034},
                "F": {"mean": 0.2045691, "sd": 0.3356756},
                "p_value": 0.6004
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 15.24422, "sd": 5.044459},
                "F": {"mean": 15.70549, "sd": 4.206785},
                "p_value": 0.7077
            }
        },
        "R1": {
            "Score_TEST": {
                "DFU": {"mean": 0.05902406, "sd": 0.03548099},
                "F": {"mean": 0.11500121, "sd": 0.22391754},
                "p_value": 0.1546
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 14.55674, "sd": 5.133734},
                "F": {"mean": 13.77461, "sd": 5.613312},
                "p_value": 0.5781
            }
        },
        "R3": {
            "Score_TEST": {
                "DFU": {"mean": 0.28584, "sd": 0.4203291},
                "F": {"mean": 0.2882836, "sd": 0.4196716},
                "p_value": 0.9821
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 16.44270, "sd": 4.554713},
                "F": {"mean": 15.51192, "sd": 4.036964},
                "p_value": 0.4048
            }
        }
    },
    "Insight": {
        "I2": {
            "Score_TEST": {
                "DFU": {"mean": 0.3571429, "sd": 0.4035036},
                "F": {"mean": 0.23, "sd": 0.3674235},
                "p_value": 0.2101
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 59.12789, "sd": 30.63056},
                "F": {"mean": 50.26118, "sd": 35.27213},
                "p_value": 0.3161
            }
        },
        "I7": {
            "Score_TEST": {
                "DFU": {"mean": 0.6538462, "sd": 0.4187895},
                "F": {"mean": 0.5588235, "sd": 0.4038584},
                "p_value": 0.3805
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 52.52251, "sd": 27.85387},
                "F": {"mean": 46.32774, "sd": 26.89404},
                "p_value": 0.3901
            }
        },
        "I10": {
            "Score_TEST": {
                "DFU": {"mean": 0.1413043, "sd": 0.2239380},
                "F": {"mean": 0.1486486, "sd": 0.1905717},
                "p_value": 0.8967
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 92.40192, "sd": 28.17144},
                "F": {"mean": 91.28652, "sd": 31.94648},
                "p_value": 0.888
            }
        },
        "I11": {
            "Score_TEST": {
                "DFU": {"mean": 0.5138889, "sd": 0.4223423},
                "F": {"mean": 0.4479167, "sd": 0.3757541},
                "p_value": 0.529
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 49.74219, "sd": 34.77374},
                "F": {"mean": 36.45396, "sd": 22.64979},
                "p_value": 0.07829
            }
        },
        "I6": {
            "Score_TEST": {
                "DFU": {"mean": 0.7682927, "sd": 0.3325575},
                "F": {"mean": 0.7105263, "sd": 0.4019325},
                "p_value": 0.5892
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 36.02911, "sd": 24.31739},
                "F": {"mean": 32.99954, "sd": 15.86058},
                "p_value": 0.5671
            }
        },
        "I1": {
            "Score_TEST": {
                "DFU": {"mean": 0.7314815, "sd": 0.3391585},
                "F": {"mean": 0.8257576, "sd": 0.3028823},
                "p_value": 0.2663
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 52.27032, "sd": 28.69239},
                "F": {"mean": 45.43994, "sd": 23.78043},
                "p_value": 0.327
            }
        },
        "I5": {
            "Score_TEST": {
                "DFU": {"mean": 0.2788462, "sd": 0.3487615},
                "F": {"mean": 0.1323529, "sd": 0.2062957},
                "p_value": 0.0647
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 51.44896, "sd": 33.25445},
                "F": {"mean": 50.96584, "sd": 30.49138},
                "p_value": 0.9541
            }
        },
        "I9": {
            "Score_TEST": {
                "DFU": {"mean": 0.5367647, "sd": 0.2761502},
                "F": {"mean": 0.5961538, "sd": 0.2653010},
                "p_value": 0.4023
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 69.46299, "sd": 27.49407},
                "F": {"mean": 65.64784, "sd": 21.48637},
                "p_value": 0.5487
            }
        },
        "I8": {
            "Score_TEST": {
                "DFU": {"mean": 0.4537037, "sd": 0.4548723},
                "F": {"mean": 0.5227273, "sd": 0.4391204},
                "p_value": 0.555
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 98.18433, "sd": 22.39830},
                "F": {"mean": 99.74295, "sd": 17.38800},
                "p_value": 0.7686
            }
        },
        "I12": {
            "Score_TEST": {
                "DFU": {"mean": 0.5357143, "sd": 0.5078745},
                "F": {"mean": 0.5468750, "sd": 0.4936594},
                "p_value": 0.9317
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 38.13350, "sd": 29.47709},
                "F": {"mean": 36.02737, "sd": 20.57308},
                "p_value": 0.753
            }
        },
        "I4": {
            "Score_TEST": {
                "DFU": {"mean": 0.5185185, "sd": 0.4849957},
                "F": {"mean": 0.6363636, "sd": 0.4885042},
                "p_value": 0.3547
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 47.45523, "sd": 37.68471},
                "F": {"mean": 37.00417, "sd": 31.23149},
                "p_value": 0.2543
            }
        },
        "I3": {
            "Score_TEST": {
                "DFU": {"mean": 0.8333333, "sd": 0.3432435},
                "F": {"mean": 0.8, "sd": 0.3559833},
                "p_value": 0.7133
            },
            "reaction_time_TEST": {
                "DFU": {"mean": 25.60816, "sd": 25.97161},
                "F": {"mean": 21.43302, "sd": 21.61555},
                "p_value": 0.5013
            }
        }
    }
}

# --------------------------------------------------------------------
# Helper: Get tasks with p < alpha from dictionary
# --------------------------------------------------------------------
def get_significant_tasks(dict_results, measure="reaction_time_TEST", alpha=0.2):
    """
    Returns a dict:
      {
        'RAT': [list_of_task_ids],
        'Insight': [list_of_task_ids]
      }
    where each task_id has p_value < alpha for the specified measure.
    """
    significant = {"RAT": [], "Insight": []}
    for task_type, tasks in dict_results.items():
        for task_id, measures in tasks.items():
            if measure in measures:
                try:
                    p_val = float(measures[measure]["p_value"])
                    if p_val < alpha:
                        significant[task_type].append(task_id)
                except Exception:
                    continue
    return significant

# --------------------------------------------------------------------
# 2) Plot Functions (with optional data override)
# --------------------------------------------------------------------
def plot_exgaussian(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f   = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        vals_dfu = df_dfu[measure].values
        vals_f   = df_f[measure].values
        min_val = min(vals_dfu.min(), vals_f.min())
        max_val = max(vals_dfu.max(), vals_f.max())
        x_pdf = np.linspace(min_val, max_val, 300)
        try:
            params_dfu = st.exponnorm.fit(vals_dfu, floc=0)
            params_f = st.exponnorm.fit(vals_f, floc=0)
            y_pdf_dfu = st.exponnorm.pdf(x_pdf, *params_dfu)
            y_pdf_f = st.exponnorm.pdf(x_pdf, *params_f)
            plt.plot(x_pdf, y_pdf_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU")
            plt.plot(x_pdf, y_pdf_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F")
        except Exception:
            continue
    plt.title(f"Ex-Gaussian Fits - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    # Save the figure
    dir_path = get_output_dir("exgaussian", task_type, title_suffix)
    file_name = f"{task_type}_exgaussian_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_chi2(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f   = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        vals_dfu = df_dfu[measure].values
        vals_f   = df_f[measure].values
        min_val = min(vals_dfu.min(), vals_f.min())
        max_val = max(vals_dfu.max(), vals_f.max())
        x_pdf = np.linspace(min_val, max_val, 300)
        try:
            vals_dfu_shift = vals_dfu - min_val + 1e-6
            vals_f_shift = vals_f - min_val + 1e-6
            df_chi_dfu, loc_dfu, scale_dfu = st.chi2.fit(vals_dfu_shift, floc=0)
            df_chi_f, loc_f, scale_f = st.chi2.fit(vals_f_shift, floc=0)
            x_shift = x_pdf - min_val + 1e-6
            y_pdf_dfu = st.chi2.pdf(x_shift, df_chi_dfu, loc=0, scale=scale_dfu)
            y_pdf_f = st.chi2.pdf(x_shift, df_chi_f, loc=0, scale=scale_f)
            plt.plot(x_pdf, y_pdf_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU")
            plt.plot(x_pdf, y_pdf_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F")
        except Exception:
            continue
    plt.title(f"Chi-square Fits - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    # Save the figure
    dir_path = get_output_dir("chi2", task_type, title_suffix)
    file_name = f"{task_type}_chi2_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_polynomial(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        vals_dfu = df_dfu[measure].values
        vals_f = df_f[measure].values
        bins = 20
        counts_dfu, bin_edges = np.histogram(vals_dfu, bins=bins, density=True)
        counts_f, _ = np.histogram(vals_f, bins=bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        try:
            coeffs_dfu = np.polyfit(bin_centers, counts_dfu, deg=3)
            coeffs_f = np.polyfit(bin_centers, counts_f, deg=3)
            poly_dfu = np.poly1d(coeffs_dfu)
            poly_f = np.poly1d(coeffs_f)
            x_line = np.linspace(bin_centers[0], bin_centers[-1], 300)
            y_dfu = poly_dfu(x_line)
            y_f = poly_f(x_line)
            plt.plot(x_line, y_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU")
            plt.plot(x_line, y_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F")
        except Exception:
            continue
    plt.title(f"Polynomial (Degree=3) Fits - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Estimated Density")
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    # Save the figure
    dir_path = get_output_dir("polynomial", task_type, title_suffix)
    file_name = f"{task_type}_polynomial_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_normal_from_dict(task_type, measure, dict_data, title_suffix="", alpha=None):
    # If an alpha is provided and the title indicates significance-based selection, filter tasks.
    if alpha is not None and "p<" in title_suffix:
        task_ids = get_significant_tasks(dict_data, measure=measure, alpha=alpha)[task_type]
    else:
        task_ids = sorted(dict_data[task_type].keys())
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("tab20", n_colors=len(task_ids))
    for i, t_id in enumerate(task_ids):
        if measure not in dict_data[task_type][t_id]:
            continue
        info = dict_data[task_type][t_id][measure]
        mu_dfu = info["DFU"]["mean"]
        sd_dfu = info["DFU"]["sd"]
        mu_f = info["F"]["mean"]
        sd_f = info["F"]["sd"]
        min_val = min(mu_dfu - 3*sd_dfu, mu_f - 3*sd_f)
        max_val = max(mu_dfu + 3*sd_dfu, mu_f + 3*sd_f)
        x_pdf = np.linspace(min_val, max_val, 300)
        y_dfu = st.norm.pdf(x_pdf, mu_dfu, sd_dfu) if sd_dfu > 0 else np.zeros_like(x_pdf)
        y_f = st.norm.pdf(x_pdf, mu_f, sd_f) if sd_f > 0 else np.zeros_like(x_pdf)
        plt.plot(x_pdf, y_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU")
        plt.plot(x_pdf, y_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F")
    plt.title(f"Normal Dist - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    # Save the figure
    dir_path = get_output_dir("normal", task_type, title_suffix)
    file_name = f"{task_type}_normal_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

# --------------------------------------------------------------------
# NEW: Plot Functions with Histogram Overlays
# For each fitted model type we add histograms of the raw data.
# Histogram colors: DFU = "#636363" (dark gray), F = "#bdbdbd" (light gray),
# and overlap = "#969696" (mid gray).
# X-axis limits: Insight: 0-120, RAT: 0-20.
# Bin sizes: RAT = 0.2 sec, Insight = 1 sec.
# --------------------------------------------------------------------
def plot_exgaussian_with_histogram(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    # Set x limits and bin size based on task type
    if task_type == "Insight":
        x_min, x_max, bin_size = 0, 120, 1
    else:
        x_min, x_max, bin_size = 0, 20, 0.2
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        # Clip values to [x_min, x_max) (used only for histogram binning)
        vals_dfu = np.clip(df_dfu[measure].values, x_min, x_max)
        vals_f = np.clip(df_f[measure].values, x_min, x_max)
        # Define bins including the last bin
        bins = np.arange(x_min, x_max + bin_size, bin_size)
        counts_dfu, _ = np.histogram(vals_dfu, bins=bins, density=True)
        counts_f, _ = np.histogram(vals_f, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # Plot histograms with neutral colors
        plt.bar(bin_centers, counts_dfu, width=bin_width, color="#636363", alpha=0.5,
                label=f"{t_id} DFU Hist" if i==0 else "")
        plt.bar(bin_centers, counts_f, width=bin_width, color="#bdbdbd", alpha=0.5,
                label=f"{t_id} F Hist" if i==0 else "")
        overlap = np.minimum(counts_dfu, counts_f)
        plt.bar(bin_centers, overlap, width=bin_width, color="#969696", alpha=1.0,
                label=f"{t_id} Overlap" if i==0 else "")
        # Overlay fitted curves (using full un-clipped values)
        x_pdf = np.linspace(x_min, x_max, 300)
        try:
            params_dfu = st.exponnorm.fit(df_dfu[measure].values, floc=0)
            params_f = st.exponnorm.fit(df_f[measure].values, floc=0)
            y_pdf_dfu = st.exponnorm.pdf(x_pdf, *params_dfu)
            y_pdf_f = st.exponnorm.pdf(x_pdf, *params_f)
            plt.plot(x_pdf, y_pdf_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU Fit")
            plt.plot(x_pdf, y_pdf_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F Fit")
        except Exception:
            continue
    plt.title(f"Ex-Gaussian Fits with Histogram - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.xlim(x_min, x_max)
    # Set y-axis limit based on task type
    if task_type == "Insight":
        plt.ylim(0, 0.14)
    else:
        plt.ylim(0, 0.7)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    dir_path = get_histogram_output_dir("exgaussian", task_type, title_suffix, with_histogram=True)
    file_name = f"{task_type}_exgaussian_hist_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_chi2_with_histogram(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    if task_type == "Insight":
        x_min, x_max, bin_size = 0, 120, 1
    else:
        x_min, x_max, bin_size = 0, 20, 0.2
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        vals_dfu = np.clip(df_dfu[measure].values, x_min, x_max)
        vals_f = np.clip(df_f[measure].values, x_min, x_max)
        bins = np.arange(x_min, x_max + bin_size, bin_size)
        counts_dfu, _ = np.histogram(vals_dfu, bins=bins, density=True)
        counts_f, _ = np.histogram(vals_f, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.bar(bin_centers, counts_dfu, width=bin_width, color="#636363", alpha=0.5,
                label=f"{t_id} DFU Hist" if i==0 else "")
        plt.bar(bin_centers, counts_f, width=bin_width, color="#bdbdbd", alpha=0.5,
                label=f"{t_id} F Hist" if i==0 else "")
        overlap = np.minimum(counts_dfu, counts_f)
        plt.bar(bin_centers, overlap, width=bin_width, color="#969696", alpha=1.0,
                label=f"{t_id} Overlap" if i==0 else "")
        x_pdf = np.linspace(x_min, x_max, 300)
        try:
            vals_dfu_shift = df_dfu[measure].values - np.min(df_dfu[measure].values) + 1e-6
            vals_f_shift = df_f[measure].values - np.min(df_f[measure].values) + 1e-6
            df_chi_dfu, loc_dfu, scale_dfu = st.chi2.fit(vals_dfu_shift, floc=0)
            df_chi_f, loc_f, scale_f = st.chi2.fit(vals_f_shift, floc=0)
            x_shift = x_pdf - np.min(df_dfu[measure].values) + 1e-6
            y_pdf_dfu = st.chi2.pdf(x_shift, df_chi_dfu, loc=0, scale=scale_dfu)
            y_pdf_f = st.chi2.pdf(x_shift, df_chi_f, loc=0, scale=scale_f)
            plt.plot(x_pdf, y_pdf_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU Fit")
            plt.plot(x_pdf, y_pdf_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F Fit")
        except Exception:
            continue
    plt.title(f"Chi-square Fits with Histogram - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.xlim(x_min, x_max)
    if task_type == "Insight":
        plt.ylim(0, 0.14)
    else:
        plt.ylim(0, 0.7)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    dir_path = get_histogram_output_dir("chi2", task_type, title_suffix, with_histogram=True)
    file_name = f"{task_type}_chi2_hist_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_polynomial_with_histogram(task_type, measure, tasks, title_suffix="", data=None):
    local_data = data if data is not None else df
    local_df = local_data[local_data["task_type"] == task_type]
    if task_type == "Insight":
        x_min, x_max, bin_size = 0, 120, 1
    else:
        x_min, x_max, bin_size = 0, 20, 0.2
    colors = sns.color_palette("tab20", n_colors=len(tasks))
    plt.figure(figsize=(10, 6))
    for i, t_id in enumerate(tasks):
        df_taskid = local_df[local_df["task_id"] == t_id]
        df_dfu = df_taskid[df_taskid["experimental_condition"] == "DFU"]
        df_f = df_taskid[df_taskid["experimental_condition"] == "F"]
        if len(df_dfu) < 2 or len(df_f) < 2:
            continue
        vals_dfu = np.clip(df_dfu[measure].values, x_min, x_max)
        vals_f = np.clip(df_f[measure].values, x_min, x_max)
        bins = np.arange(x_min, x_max + bin_size, bin_size)
        counts_dfu, bin_edges = np.histogram(vals_dfu, bins=bins, density=True)
        counts_f, _ = np.histogram(vals_f, bins=bin_edges, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, counts_dfu, width=bin_width, color="#636363", alpha=0.5,
                label=f"{t_id} DFU Hist" if i==0 else "")
        plt.bar(bin_centers, counts_f, width=bin_width, color="#bdbdbd", alpha=0.5,
                label=f"{t_id} F Hist" if i==0 else "")
        overlap = np.minimum(counts_dfu, counts_f)
        plt.bar(bin_centers, overlap, width=bin_width, color="#969696", alpha=1.0,
                label=f"{t_id} Overlap" if i==0 else "")
        # Polynomial fit from histogram counts
        try:
            coeffs_dfu = np.polyfit(bin_centers, counts_dfu, deg=3)
            coeffs_f = np.polyfit(bin_centers, counts_f, deg=3)
            poly_dfu = np.poly1d(coeffs_dfu)
            poly_f = np.poly1d(coeffs_f)
            x_line = np.linspace(x_min, x_max, 300)
            plt.plot(x_line, poly_dfu(x_line), '-', color=colors[i], linewidth=2, label=f"{t_id} DFU Fit")
            plt.plot(x_line, poly_f(x_line), '--', color=colors[i], linewidth=2, label=f"{t_id} F Fit")
        except Exception:
            continue
    plt.title(f"Polynomial (Degree=3) Fits with Histogram - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Estimated Density")
    plt.xlim(x_min, x_max)
    if task_type == "Insight":
        plt.ylim(0, 0.14)
    else:
        plt.ylim(0, 0.7)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    dir_path = get_histogram_output_dir("polynomial", task_type, title_suffix, with_histogram=True)
    file_name = f"{task_type}_polynomial_hist_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

def plot_normal_from_dict_with_histogram(task_type, measure, dict_data, title_suffix="", alpha=None):
    # For histogram we use the raw data from the main dataframe.
    if task_type == "Insight":
        x_min, x_max, bin_size = 0, 120, 1
    else:
        x_min, x_max, bin_size = 0, 20, 0.2
    if alpha is not None and "p<" in title_suffix:
        task_ids = get_significant_tasks(dict_data, measure=measure, alpha=alpha)[task_type]
    else:
        task_ids = sorted(dict_data[task_type].keys())
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("tab20", n_colors=len(task_ids))
    for i, t_id in enumerate(task_ids):
        local_df = df[df["task_type"] == task_type]
        df_taskid = local_df[local_df["task_id"] == t_id]
        vals_dfu = np.clip(df_taskid[df_taskid["experimental_condition"] == "DFU"][measure].values, x_min, x_max)
        vals_f = np.clip(df_taskid[df_taskid["experimental_condition"] == "F"][measure].values, x_min, x_max)
        bins = np.arange(x_min, x_max + bin_size, bin_size)
        counts_dfu, _ = np.histogram(vals_dfu, bins=bins, density=True)
        counts_f, _ = np.histogram(vals_f, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.bar(bin_centers, counts_dfu, width=bin_width, color="#636363", alpha=0.5,
                label=f"{t_id} DFU Hist" if i==0 else "")
        plt.bar(bin_centers, counts_f, width=bin_width, color="#bdbdbd", alpha=0.5,
                label=f"{t_id} F Hist" if i==0 else "")
        overlap = np.minimum(counts_dfu, counts_f)
        plt.bar(bin_centers, overlap, width=bin_width, color="#969696", alpha=1.0,
                label=f"{t_id} Overlap" if i==0 else "")
        if measure not in dict_data[task_type][t_id]:
            continue
        info = dict_data[task_type][t_id][measure]
        mu_dfu = info["DFU"]["mean"]
        sd_dfu = info["DFU"]["sd"]
        mu_f = info["F"]["mean"]
        sd_f = info["F"]["sd"]
        x_pdf = np.linspace(x_min, x_max, 300)
        y_dfu = st.norm.pdf(x_pdf, mu_dfu, sd_dfu) if sd_dfu > 0 else np.zeros_like(x_pdf)
        y_f = st.norm.pdf(x_pdf, mu_f, sd_f) if sd_f > 0 else np.zeros_like(x_pdf)
        plt.plot(x_pdf, y_dfu, '-', color=colors[i], linewidth=2, label=f"{t_id} DFU Fit")
        plt.plot(x_pdf, y_f, '--', color=colors[i], linewidth=2, label=f"{t_id} F Fit")
    plt.title(f"Normal Dist with Histogram - {task_type} ({measure}){title_suffix}")
    plt.xlabel(measure)
    plt.ylabel("Density")
    plt.xlim(x_min, x_max)
    if task_type == "Insight":
        plt.ylim(0, 0.14)
    else:
        plt.ylim(0, 0.7)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    dir_path = get_histogram_output_dir("normal", task_type, title_suffix, with_histogram=True)
    file_name = f"{task_type}_normal_hist_{measure}_{title_suffix.strip().replace(' ','_').replace('-','')}.png"
    plt.savefig(os.path.join(dir_path, file_name))
    plt.close()

# --------------------------------------------------------------------
# 3) Produce the 32 Plots (16 for full data and 16 for clipped data)
# Original plots (without histograms)
# --------------------------------------------------------------------
def run_all_plots(alpha=0.2):
    measure = "reaction_time_TEST"
    # 1) Ex-Gaussian
    plot_exgaussian("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_exgaussian("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 2) Chi-square
    plot_chi2("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_chi2("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 3) Polynomial
    plot_polynomial("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_polynomial("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 4) Normal from dictionary
    plot_normal_from_dict("RAT", measure, Unpaired_ttest_Welch_taskID_results, title_suffix=" - All tasks")
    plot_normal_from_dict("Insight", measure, Unpaired_ttest_Welch_taskID_results, title_suffix=" - All tasks")
    sig_tasks = get_significant_tasks(Unpaired_ttest_Welch_taskID_results, measure=measure, alpha=alpha)
    # 5) Ex-Gaussian for tasks with p < alpha
    plot_exgaussian("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_exgaussian("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 6) Chi-square for tasks with p < alpha
    plot_chi2("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_chi2("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 7) Polynomial for tasks with p < alpha
    plot_polynomial("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_polynomial("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 8) Normal from dictionary for tasks with p < alpha
    plot_normal_from_dict("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - p<{alpha} tasks only", alpha=alpha)
    plot_normal_from_dict("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - p<{alpha} tasks only", alpha=alpha)

def run_all_plots_clipped(alpha=0.2):
    measure = "reaction_time_TEST"
    df_rat_clipped = df[df["task_type"]=="RAT"].copy()
    df_insight_clipped = df[df["task_type"]=="Insight"].copy()
    # Clipping: for RAT clip to [0.5, 19.8), for Insight clip to [2, 118)
    df_rat_clipped[measure] = np.clip(df_rat_clipped[measure].values, 0.5, 19.8 - 1e-6)
    df_insight_clipped[measure] = np.clip(df_insight_clipped[measure].values, 2, 118 - 1e-6)
    # 1) Ex-Gaussian
    plot_exgaussian("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_exgaussian("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 2) Chi-square
    plot_chi2("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
              title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_chi2("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
              title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 3) Polynomial
    plot_polynomial("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_polynomial("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 4) Normal from dictionary (using full dictionary; clipping not applied to dict)
    plot_normal_from_dict("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=" - Clipped All tasks")
    plot_normal_from_dict("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=" - Clipped All tasks")
    sig_tasks = get_significant_tasks(Unpaired_ttest_Welch_taskID_results, measure=measure, alpha=alpha)
    # 5) Ex-Gaussian for tasks with p < alpha
    plot_exgaussian("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_exgaussian("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 6) Chi-square for tasks with p < alpha
    plot_chi2("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_chi2("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 7) Polynomial for tasks with p < alpha
    plot_polynomial("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_polynomial("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 8) Normal from dictionary for tasks with p < alpha
    plot_normal_from_dict("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - Clipped & p<{alpha} tasks only", alpha=alpha)
    plot_normal_from_dict("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - Clipped & p<{alpha} tasks only", alpha=alpha)

# --------------------------------------------------------------------
# NEW: Produce the Histogram Overlaid Plots
# Same logic as above but using the new functions with histogram.
# --------------------------------------------------------------------
def run_all_plots_histogram(alpha=0.2):
    measure = "reaction_time_TEST"
    # 1) Ex-Gaussian with histogram
    plot_exgaussian_with_histogram("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_exgaussian_with_histogram("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 2) Chi-square with histogram
    plot_chi2_with_histogram("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_chi2_with_histogram("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 3) Polynomial with histogram
    plot_polynomial_with_histogram("RAT", measure, df[df["task_type"]=="RAT"]["task_id"].unique(), title_suffix=" - All tasks")
    plot_polynomial_with_histogram("Insight", measure, df[df["task_type"]=="Insight"]["task_id"].unique(), title_suffix=" - All tasks")
    # 4) Normal from dict with histogram
    plot_normal_from_dict_with_histogram("RAT", measure, Unpaired_ttest_Welch_taskID_results, title_suffix=" - All tasks")
    plot_normal_from_dict_with_histogram("Insight", measure, Unpaired_ttest_Welch_taskID_results, title_suffix=" - All tasks")
    sig_tasks = get_significant_tasks(Unpaired_ttest_Welch_taskID_results, measure=measure, alpha=alpha)
    # 5) Ex-Gaussian for tasks with p < alpha with histogram
    plot_exgaussian_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_exgaussian_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 6) Chi-square for tasks with p < alpha with histogram
    plot_chi2_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_chi2_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 7) Polynomial for tasks with p < alpha with histogram
    plot_polynomial_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - p<{alpha}")
    plot_polynomial_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - p<{alpha}")
    # 8) Normal from dict for tasks with p < alpha with histogram
    plot_normal_from_dict_with_histogram("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - p<{alpha} tasks only", alpha=alpha)
    plot_normal_from_dict_with_histogram("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - p<{alpha} tasks only", alpha=alpha)

def run_all_plots_clipped_histogram(alpha=0.2):
    measure = "reaction_time_TEST"
    df_rat_clipped = df[df["task_type"]=="RAT"].copy()
    df_insight_clipped = df[df["task_type"]=="Insight"].copy()
    df_rat_clipped[measure] = np.clip(df_rat_clipped[measure].values, 0.5, 19.8 - 1e-6)
    df_insight_clipped[measure] = np.clip(df_insight_clipped[measure].values, 2, 118 - 1e-6)
    # 1) Ex-Gaussian with histogram
    plot_exgaussian_with_histogram("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_exgaussian_with_histogram("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 2) Chi-square with histogram
    plot_chi2_with_histogram("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
              title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_chi2_with_histogram("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
              title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 3) Polynomial with histogram
    plot_polynomial_with_histogram("RAT", measure, df_rat_clipped[df_rat_clipped["task_type"]=="RAT"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_rat_clipped)
    plot_polynomial_with_histogram("Insight", measure, df_insight_clipped[df_insight_clipped["task_type"]=="Insight"]["task_id"].unique(),
                    title_suffix=" - Clipped All tasks", data=df_insight_clipped)
    # 4) Normal from dict with histogram
    plot_normal_from_dict_with_histogram("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=" - Clipped All tasks")
    plot_normal_from_dict_with_histogram("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=" - Clipped All tasks")
    sig_tasks = get_significant_tasks(Unpaired_ttest_Welch_taskID_results, measure=measure, alpha=alpha)
    # 5) Ex-Gaussian for tasks with p < alpha with histogram
    plot_exgaussian_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_exgaussian_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 6) Chi-square for tasks with p < alpha with histogram
    plot_chi2_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_chi2_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 7) Polynomial for tasks with p < alpha with histogram
    plot_polynomial_with_histogram("RAT", measure, sig_tasks["RAT"], title_suffix=f" - Clipped & p<{alpha}", data=df_rat_clipped)
    plot_polynomial_with_histogram("Insight", measure, sig_tasks["Insight"], title_suffix=f" - Clipped & p<{alpha}", data=df_insight_clipped)
    # 8) Normal from dict for tasks with p < alpha with histogram
    plot_normal_from_dict_with_histogram("RAT", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - Clipped & p<{alpha} tasks only", alpha=alpha)
    plot_normal_from_dict_with_histogram("Insight", measure, Unpaired_ttest_Welch_taskID_results,
                          title_suffix=f" - Clipped & p<{alpha} tasks only", alpha=alpha)

if __name__ == "__main__":
    run_all_plots(alpha=0.35)
    run_all_plots_clipped(alpha=0.35)
    run_all_plots_histogram(alpha=0.35)
    run_all_plots_clipped_histogram(alpha=0.35)
