import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.deep_learning import SimpleRNNClassifier

def train_classifiers():
    combined_data = pd.read_excel("Selected_Patients/combined_data.xlsx")
    X = combined_data[['steps', 'heart']].values
    y = combined_data['level']
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "TimeSeries Forest": TimeSeriesForestClassifier(),
        "Simple RNN": SimpleRNNClassifier()
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
