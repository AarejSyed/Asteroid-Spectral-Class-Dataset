import pandas as pd
import numpy as np


# Load data
file_name = 'https://raw.githubusercontent.com/AarejSyed/Asteroid-Spectral-Class-Dataset/refs/heads/main/sbdb_query_results.csv'
df = pd.read_csv(file_name, dtype={"spec_B": str})

# Old row count before cleaning
old_row_count = len(df)

# Define required columns
# Some columns have many missing values, so we are excluding to get a larger dataset
required_columns = [
    "H",
    "albedo",
    # "BV",     # Too many missing values 
    # "UB",     # Too many missing values
    # "IR",     # Too many missing values
    "rot_per",
    "spec_B"
]

# Keep only required columns
df = df[required_columns].copy()

# Debugging: For each column, print proportion of missing values
print("\nProportion of missing values in each column:")
print(df.isnull().mean().sort_values(ascending=False))
print("")

# Drop rows where any column is missing
df_cleaned = df.dropna(subset=required_columns).copy()

# Drop duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Remove uncertain classifications (those containing ':')
df_cleaned = df_cleaned[~df_cleaned["spec_B"].str.contains(":", na=False)]

# Strip whitespace from spectral class just in case
df_cleaned["spec_B"] = df_cleaned["spec_B"].str.strip()

# Group spectral subclasses together into classes
# First letter in a subclass name indicates class, so we keep only that
df_cleaned["spec_B"] = df_cleaned["spec_B"].str[0]

# Apply log transform to reduce skew in rotation period
df_cleaned["rot_per"] = np.log1p(df_cleaned["rot_per"])

# New row count after cleaning
new_row_count = len(df_cleaned)

# Save cleaned dataset
df_cleaned.to_csv('cleaned_data.csv', index=False)

# Print stats
print(f"Original rows: {old_row_count}")
print(f"Cleaned rows: {new_row_count}")
print(f"Rows removed: {old_row_count - new_row_count}\n")

print("Class distribution:")
print(df_cleaned["spec_B"].value_counts())

print("\nFeature summary:")
print(df_cleaned.describe())
print("")
