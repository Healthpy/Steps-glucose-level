import numpy as np

def load_model(model_path):
    model = np.load(model_path)
    return model