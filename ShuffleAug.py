import pandas as pd
import numpy as np

# Load your original dataset
# Assuming df_original is your DataFrame containing the original dataset
# Replace "DATA_FROM_SENSORS.xlsx" with the path to your Excel file
df_original = pd.read_excel("COLLECTED DATASET.xlsx")

# Create a copy of the original dataset to apply changes
df = df_original.copy()

# Mapping 'Yes' to 1 and 'No' to 0 in the 'FLAME DETECTED' column
df['FLAME DETECTED'] = df['FLAME DETECTED'].map({'Yes': 1, 'No': 0})

# Define function for shuffling data
def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)

# Augment the dataset by shuffling
shuffled_df = shuffle_data(df)

# Calculate the number of additional samples needed
desired_size = 10000  # or 4000
additional_samples_needed = desired_size - len(shuffled_df)

# Sample additional data points from the shuffled dataset with replacement
augmented_df = shuffled_df.sample(n=additional_samples_needed, replace=True)

# Concatenate original and augmented datasets
concatenated_df = pd.concat([shuffled_df, augmented_df], ignore_index=True)

# Save concatenated data to a new CSV file
# Replace "augmented_dataset.csv" with the desired file name for your concatenated dataset
concatenated_df.to_csv("augmented_dataset.csv", index=False)
