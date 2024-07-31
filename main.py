from data_processing.file_operations import process_directory
from data_processing.glucose_processing import process_glucose_data
from data_processing.interpolation import interpolate_glucose_levels
from data_processing.steps_processing import process_all_patients
from classification.classifiers import steps_classifiers, sktime_classifiers, LSTM_classifier
from data_merging import merge_patient_data
from data_processing.time_series_processing import load_data_and_model, label_encoder, sliding_window_3d
from counterfactuals.attention_based_CF import generate_counterfactuals, plot_original_vs_counterfactuals
# from models.model_operations import load_model
import numpy as np



def main():
    selected_dir = 'Selected_Patients'
    # process_directory(selected_dir)
    # process_glucose_data(selected_dir)
    # interpolate_glucose_levels(selected_dir)
    # process_all_patients(selected_dir)
    # merge_patient_data(selected_dir)
    LSTM_classifier()

def run_counterfactual_experiment():

    # model = load_model("models/lstm.h5")
    X_train, X_test, y_train, y_test, model = load_data_and_model()
    # y_train_encoded, y_test_encoded = label_encoder(y_train, y_test)
    
    y_train=np.argmax(y_train, axis=1)
    y_test=np.argmax(y_test, axis=1)

    window_size = int(X_test[0].shape[1] * 0.1)
    stride = window_size

    cfs, times, target_probas, num_dim_changed, suc_indexes = generate_counterfactuals(
        X_test, y_test, model, X_train, y_train, window_size, stride
    )

    np.save("results/_cfs.npy", cfs)
    np.save("results/_times.npy", times)
    np.save("results/_probas.npy", target_probas)

    print("Average number of segments changed:", np.average(num_dim_changed))
    print("Finished processing")
    
    # Plot and save the results
    plot_original_vs_counterfactuals(X_test, y_test, cfs, model, suc_indexes)

if __name__ == "__main__":
    main()
    run_counterfactual_experiment()
