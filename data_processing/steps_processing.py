import os
import pandas as pd

def process_all_patients(selected_root_dir):
    for patient_folder in os.listdir(selected_root_dir):
        patient_dir = os.path.join(selected_root_dir, patient_folder)
        if os.path.isdir(patient_dir):
            input_file_name = f"{patient_folder}-steps.xlsx"
            input_file_path = os.path.join(patient_dir, input_file_name)
            input_heart = f"{patient_folder}-heart.xlsx"
            input_heart_path = os.path.join(patient_dir, input_heart)
            
            if os.path.exists(input_file_path) and os.path.exists(input_heart_path):
                print(f"Processing data for patient: {patient_folder}")
                df = pd.read_excel(input_file_path)
                df_heart = pd.read_excel(input_heart_path)
                df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
                flattened_data = []
                for col in df.columns[1:]:
                    for (i, row), (j, j_row) in zip(df.iterrows(), df_heart.iterrows()):
                        datetime_combined = pd.to_datetime(f"{col} {row['time'].time()}")
                        steps = row[col]
                        heart = j_row[col]
                        flattened_data.append([datetime_combined, steps, heart])
                flattened_df = pd.DataFrame(flattened_data, columns=['datetime', 'steps', 'heart'])
                flattened_df.set_index('datetime', inplace=True)
                resampled_df = flattened_df.resample('15T', label='right', closed='right').sum()
                resampled_df.reset_index(inplace=True)
                output_file_path = os.path.join(patient_dir, f'{patient_folder}-flattened_steps_heart.xlsx')
                resampled_df.to_excel(output_file_path, index=False)
                print(f"Processed data saved to {output_file_path}")
            else:
                print(f"Input file {input_file_name} does not exist in {patient_dir}")
