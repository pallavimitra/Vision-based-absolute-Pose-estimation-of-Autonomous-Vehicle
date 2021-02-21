"""
Predict car position in the world using 2D features
"""

import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def predict(coords, boxes_widths, model):
    return model.predict(np.concatenate((coords, boxes_widths), axis=1))