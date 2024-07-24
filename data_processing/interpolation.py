import os
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import timedelta

def categorize_glucose_level(glucose):
    if glucose < 3.9:
        return 'low'
    elif 3.9 <= glucose <= 6.9:
        return 'normal'
    else:
        return 'high'

def interpolate_glucose_levels(directory_path):
    for dir_path, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('glucose.xlsx'):
                file_path = os.path.join(dir_path, filename)
                try:
                    df = pd.read_excel(file_path)
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df.dropna(inplace=True)
                    df['datetime_numeric'] = df['Datetime'].apply(lambda x: x.timestamp())
                    X = df['datetime_numeric'].values
                    Y = df['Glucose level'].values
                    spline = make_interp_spline(X, Y, k=3)
                    start_time = df['Datetime'].min()
                    end_time = df['Datetime'].max()
                    start_time -= timedelta(minutes=start_time.minute % 15, seconds=start_time.second, microseconds=start_time.microsecond)
                    end_time += timedelta(minutes=(15 - end_time.minute % 15) % 15, seconds=-end_time.second, microseconds=-end_time.microsecond)
                    new_times = pd.date_range(start=start_time, end=end_time, freq='15T')
                    new_times_numeric = new_times.map(pd.Timestamp.timestamp).values
                    new_glucose_levels = spline(new_times_numeric)
                    new_df = pd.DataFrame({
                        'datetime': new_times,
                        'glucose_level': new_glucose_levels,
                        'level': [categorize_glucose_level(glucose) for glucose in new_glucose_levels]
                    })
                    for f in os.listdir(dir_path):
                        if f.startswith('interpolated') and f.endswith('.xlsx'):
                            try:
                                os.remove(os.path.join(dir_path, f))
                                print(f"Deleted existing file: {f}")
                            except Exception as e:
                                print(f"Failed to delete existing file {f}: {e}")
                    output_file = os.path.join(dir_path, f'interpolated-{os.path.splitext(filename)[0]}.xlsx')
                    new_df.to_excel(output_file, index=False)
                    print(f"Interpolated file saved: {output_file}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
