from data_processing.file_operations import process_directory
from data_processing.glucose_processing import process_glucose_data
from data_processing.interpolation import interpolate_glucose_levels
from data_processing.steps_processing import process_all_patients
from classification.classifiers import train_classifiers

def main():
    selected_dir = 'Selected_Patients'
    process_directory(selected_dir)
    process_glucose_data(selected_dir)
    interpolate_glucose_levels(selected_dir)
    process_all_patients(selected_dir)
    train_classifiers()

if __name__ == "__main__":
    main()
