import os
import pandas as pd

def process_glucose_data(selected_dir):
    for folder_name in os.listdir(selected_dir):
        print(f'Reading data for {folder_name}')
        folder_path = os.path.join(selected_dir, folder_name)
        glucose_file = os.path.join(folder_path, f'{folder_name}-glucose.txt')
        if os.path.isfile(glucose_file):
            glucose_df = pd.read_csv(glucose_file, sep='\t', decimal=',')
            if 'Tijd' in glucose_df.columns:
                glu_df = glucose_df[['Tijd', 'Historie glucose (mmol/L)']]
                glu_df.rename(columns={'Tijd': 'Datetime', 'Historie glucose (mmol/L)': 'Glucose level'}, inplace=True)
            else:
                continue
            glu_df.to_excel(f'{folder_path}/{folder_name}-glucose.xlsx')

    print('Reading process completed.')
