import pandas as pd

# Mappings from Dutch to English for gender values
gender_mapping = {
    "Man": "Male",
    "Vrouw": "Female",
    "X": "Other"
}


def analyze_demographics(file_path):
    """
    Loads the CSV file, aggregates rows by participant_identification,
    and computes overall statistics for gender, age, and social media hours (daily).
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Aggregate by participant_identification; assume demographic info remains constant per participant
    aggregated = df.groupby('participant_identification')[
        ['gender', 'age', 'social_media_hours (daily)']].first().reset_index()

    # Print number of unique participants
    print(f"Number of unique participants: {len(aggregated)}")

    # Map gender values to English
    aggregated['gender'] = aggregated['gender'].map(gender_mapping)

    # Compute overall gender proportions
    total_participants = len(aggregated)
    gender_counts = aggregated['gender'].value_counts()
    print("Overall Gender Proportions:")
    for gender, count in gender_counts.items():
        proportion = count / total_participants
        print(f"Gender: {gender}, Count: {count}, Proportion: {proportion:.2f}")

    # Compute age statistics
    age_mean = aggregated['age'].mean()
    age_std = aggregated['age'].std()
    age_min = aggregated['age'].min()
    age_max = aggregated['age'].max()
    print("\nAge Statistics:")
    print(f"Mean: {age_mean:.2f}, SD: {age_std:.2f}, Range: {age_min} - {age_max}")

    # Compute social media hours (daily) statistics
    sm_mean = aggregated['social_media_hours (daily)'].mean()
    sm_std = aggregated['social_media_hours (daily)'].std()
    sm_min = aggregated['social_media_hours (daily)'].min()
    sm_max = aggregated['social_media_hours (daily)'].max()
    print("\nSocial Media Hours (Daily) Statistics:")
    print(f"Mean: {sm_mean:.2f}, SD: {sm_std:.2f}, Range: {sm_min} - {sm_max}")


if __name__ == "__main__":
    file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
    analyze_demographics(file_path)
