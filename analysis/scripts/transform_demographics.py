import pandas as pd

# File path
file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"

# Read the data
df = pd.read_csv(file_path)

# Define mapping dictionaries
social_media_map = {
    '0-1 uur': 0.5,
    '1-2 uren': 1.5,
    '2-3 uren': 2.5,
    '3-4 uren': 3.5,
    '4-5 uren': 4.5,
    '5-6 uren': 5.5,
    '6-7 uren': 6.5,
    '7-8 uren': 7.5,
    '8-9 uren': 8.5,
    '9-10 uren': 9.5,
    '10+ uren': 11
}

# Invert dictionary (keys become values and vice versa)
verbal_intel_map = {
    'Extreem laag': 1,
    'Zeer laag': 2,
    'Laag': 3,
    'Onder gemiddeld': 4,
    'Gemiddeld': 5,
    'Boven gemiddeld': 6,
    'Hoog': 7,
    'Zeer hoog': 8,
    'Extreem hoog': 9
}

# Apply mappings
df['social_media_hours (daily)'] = df['social_media_hours (daily)'].map(social_media_map)
#df['verbal_intelligence (self-perceived)'] = df['verbal_intelligence (self-perceived)'].map(verbal_intel_map)

# Overwrite the CSV file
df.to_csv(file_path, index=False)
