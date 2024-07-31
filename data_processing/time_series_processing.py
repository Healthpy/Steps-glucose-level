import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import load_model

def load_data_and_model():
    X_train = np.load("results/X_train.npy")
    X_test = np.load("results/X_test.npy")
    y_train = np.load("results/y_train.npy")
    y_test = np.load("results/y_test.npy")
    model = load_model("models/lstm.h5")
    return X_train, X_test, y_train, y_test, model

def label_encoder(training_labels, testing_labels):
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)
    return y_train, y_test

def sliding_window_3d(data, window_size, stride):
    num_features, num_timesteps = data.shape
    num_subsequences = ((num_timesteps - window_size) // stride) + 1
    subsequences = np.zeros((num_subsequences, num_features, window_size))
    for j in range(num_subsequences):
        start = j * stride
        end = start + window_size
        subsequences[j] = data[:, start:end]
    return subsequences
