# data_merging.py

import pandas as pd
import os

def merge_patient_data(base_directory):
    """
    Merge steps and glucose data for all patients into a single dataframe and save as an Excel file.

    Parameters:
    - base_directory: The directory containing patient data folders.
    """
    # Initialize an empty list to hold the merged data for each patient
    all_patients_data = []

    # List all directories (assuming each directory corresponds to a patient)
    patient_directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    for patient_dir in patient_directories:
        # Define paths to the steps and glucose level Excel files
        steps_heart = os.path.join(base_directory, patient_dir, f"{patient_dir}-flattened_steps_heart.xlsx")
        glucose_file = os.path.join(base_directory, patient_dir, f"interpolated-{patient_dir}-glucose.xlsx")
        
        if os.path.exists(steps_heart) and os.path.exists(glucose_file):
            # Read the Excel files into dataframes
            df_steps = pd.read_excel(steps_heart)
            df_glucose = pd.read_excel(glucose_file)
            
            # Merge the dataframes on the datetime column
            df_merged = pd.merge(df_steps, df_glucose, on='datetime')

            df_merged['PatientID'] = patient_dir
            
            # Append the merged dataframe to the list
            all_patients_data.append(df_merged)

    # Combine all patient data into a single dataframe
    combined_data = pd.concat(all_patients_data, ignore_index=True)

    # Drop rows where both 'steps' and 'heart' columns are zero
    combined_data = combined_data.drop(combined_data[(combined_data['steps'] == 0) & (combined_data['heart'] == 0)].index)

    # Save the combined data to a new Excel file
    output_file = os.path.join(base_directory, "combined_data.xlsx")
    combined_data.to_excel(output_file, index=False)

    print(f"Combined dataset saved to {output_file}")
