import numpy as np
import pandas as pd

# Load your dataset from Excel
xlsx_file_path = "COLLECTED DATASET.xlsx"  # Replace with the path to your Excel file
df = pd.read_excel(xlsx_file_path)

# Convert 'FLAME DETECTED' column to binary (1 for 'Yes', 0 for 'No')
df['FLAME DETECTED'] = df['FLAME DETECTED'].map({'Yes': 1, 'No': 0})

# Define the desired dataset size
desired_size = 10000

# Calculate the number of additional samples needed
num_additional_samples = desired_size - len(df)

# Define noise parameters
mean = 0
std_dev = 0.05  # Adjust as needed based on the scale of your features

# Generate additional samples with noise
additional_data = pd.DataFrame({
    'FLAME DETECTED': np.random.choice([1, 0], num_additional_samples),  # Encoding 'Yes' as 1 and 'No' as 0
    'HUMIDITY': np.random.normal(0, std_dev, num_additional_samples) + 55,  # Mean humidity value
    'TEMPERATURE (Centigrade)': np.random.normal(0, std_dev, num_additional_samples) + 32.5,  # Mean temperature value
    'GAS LEVEL (PPM)': np.random.normal(0, std_dev, num_additional_samples) + 800  # Mean gas level value
})

# Clip the values to ensure they remain within the original range
additional_data['HUMIDITY'] = np.clip(additional_data['HUMIDITY'], 30, 80)
additional_data['TEMPERATURE (Centigrade)'] = np.clip(additional_data['TEMPERATURE (Centigrade)'], 25, 40)
additional_data['GAS LEVEL (PPM)'] = np.clip(additional_data['GAS LEVEL (PPM)'], 500, 1100)

# Round the values of temperature and humidity to one decimal place
additional_data['HUMIDITY'] = additional_data['HUMIDITY'].round(1)
additional_data['TEMPERATURE (Centigrade)'] = additional_data['TEMPERATURE (Centigrade)'].round(1)

# Convert 'GAS LEVEL (PPM)' column to integers
additional_data['GAS LEVEL (PPM)'] = additional_data['GAS LEVEL (PPM)'].astype(int)

# Recalculate SAFETY CONDITION based on the augmented data
conditions = (
    (additional_data['FLAME DETECTED'] == 1) |  # Checking if 'FLAME DETECTED' is 1
    (additional_data['HUMIDITY'] > 75) |
    (additional_data['TEMPERATURE (Centigrade)'] > 50) |
    (additional_data['GAS LEVEL (PPM)'] > 650)  # Modified condition for gas level
)
additional_data['SAFETY CONDITION'] = conditions.astype(int)

# Concatenate the additional data with the original dataset
augmented_df = pd.concat([df, additional_data], ignore_index=True)

# Save the augmented dataset
augmented_df.to_excel("augmented_dataset.xlsx", index=False)
