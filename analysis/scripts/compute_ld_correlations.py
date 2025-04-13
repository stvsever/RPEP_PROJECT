import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the Excel file (latent disposition data)
excel_path = ("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
              "Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/"
              "data/preprocessed/latent_disposition.xlsx")

# Load the data into a pandas DataFrame
df = pd.read_excel(excel_path)

# Select only the numeric columns for correlation (exclude participant_identification)
cols = ['mean_all', 'mean_F', 'mean_DFU', 'fa_all', 'fa_F', 'fa_DFU']
corr_matrix = df[cols].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1, style="white")
heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", square=True, cbar_kws={'shrink': 0.8})
plt.title("Correlation Matrix: Mean and Factor Scores", fontsize=14)
plt.tight_layout()

# Define the output directory for the plot and ensure it exists
output_dir = ("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
              "Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/"
              "analysis/plots/affinity_correlations/plots_ld_estimation")
os.makedirs(output_dir, exist_ok=True)

# Save the plot as a PNG file in the specified directory
plot_path = os.path.join(output_dir, "correlation_matrix.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Correlation matrix plot saved at:", plot_path)
