import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.deep_learning import SimpleRNNClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.hybrid import HIVECOTEV2
from sktime.datatypes._panel._convert import from_3d_numpy_to_2d_array
import time
from sklearn import preprocessing
from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
import matplotlib.pyplot as plt


from data_processing.interpolation import categorize_glucose_level
df = pd.read_excel("Selected_Patients/combined_data.xlsx")

def steps_classifiers():
    # X = df[['steps', 'heart']].values
    X = df['steps'].values.reshape(-1, 1)
    y = df['level']
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
        # "Support Vector Machine": SVC(),
        # "K-Nearest Neighbors": KNeighborsClassifier(),
        # "Naive Bayes": GaussianNB(),

    }
    cv_scores = {}
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        cv_scores[name] = scores.mean()
    for name, score in cv_scores.items():
        print(f"{name}: {score:.4f}")

def LSTM_classifier():

    # # Normalize the data
    # scaler = StandardScaler()
    # df[['steps', 'heart', 'glucose_level']] = scaler.fit_transform(df[['steps', 'heart', 'glucose_level']])

    # Prepare data for LSTM model
    time_steps = 10  # Number of previous time steps to consider
    features = ['steps', 'heart', 'glucose_level']

    X, y = [], []
    for patient_id in df['PatientID'].unique():
        patient_data = df[df['PatientID'] == patient_id]
        for i in range(time_steps, len(patient_data)):
            X.append(patient_data[features].iloc[i-time_steps:i].values)
            y.append(patient_data['level'].iloc[i])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y, num_classes=3)

    X = np.array(X)
    X = X.transpose(0, 2, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save("results/X_train.npy", X_train)
    np.save("results/X_test.npy", X_test)
    np.save("results/y_train", y_train)
    np.save("results/y_test", y_test)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(len(features), time_steps)))
    model.add(LSTM(units=50))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=128)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # # Inverse transform the predicted values if scaling was applied
    # y_test_scaled = y_test * scaler.scale_[2] + scaler.mean_[2]
    # y_pred_scaled = y_pred.flatten() * scaler.scale_[2] + scaler.mean_[2]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared: {r2}')


    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Accuracy: {accuracy}')

    model.save('models/lstm.h5')


def sktime_classifiers():
    
    # Encode the 'level' column
    label_encoder = LabelEncoder()
    df['level'] = label_encoder.fit_transform(df['level'])
    timesteps = 8
    # Prepare data fpr sktime
    X, y = [], []
    for patient_id in df['PatientID'].unique():
        patient_data = df[df['PatientID'] == patient_id]
        for i in range(timesteps, len(patient_data)):
            X.append(patient_data[['steps', 'heart', 'glucose_level']].iloc[i-timesteps:i].values)
            y.append(patient_data['level'].iloc[i])

    X = np.array(X)
    X = X.transpose(0, 2, 1)
    y = np.array(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to sktime's data structure
    X_train_sktime = from_3d_numpy_to_2d_array(X_train)
    X_test_sktime = from_3d_numpy_to_2d_array(X_test)

    # Initialize classifiers
    classifiers = {
        "Time Series Forest": TimeSeriesForestClassifier()
        # "Shapelet Transform": ShapeletTransformClassifier(),
        # "Simple RNN": SimpleRNNClassifier()
        # "HIVE-COTE v2": HIVECOTEV2()
        # "BOSS": BOSSEnsemble(),
        # "KNeighbors Time Series": KNeighborsTimeSeriesClassifier()

    }

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train_sktime, y_train)
        y_pred = clf.predict(X_test_sktime)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy}')