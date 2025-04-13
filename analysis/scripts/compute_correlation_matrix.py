import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_matrix(df):
    # Select only numeric columns to avoid conversion issues
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    return corr_matrix


def visualize_correlation_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))
    # Create a heatmap with annotations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # Rotate x-axis labels 45 degrees for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    # Adjust layout to ensure nothing is cut off
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File path to the CSV file
    file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Optionally, print the names of the numeric variables available
    num_vars = df.select_dtypes(include=['float64', 'int64']).columns
    print("Numeric variables used for correlation:", list(num_vars))

    # Compute the correlation matrix and visualize it
    corr_matrix = correlation_matrix(df)
    visualize_correlation_matrix(corr_matrix)

    # Read xls file
    file_path_xls = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/aggregated_by_subjectID_BOTH.xlsx"
    df = pd.read_excel(file_path_xls) # (both classical & insight))

    # Compute the correlation matrix and visualize it
    corr_matrix = correlation_matrix(df)
    visualize_correlation_matrix(corr_matrix)
