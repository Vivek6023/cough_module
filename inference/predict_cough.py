import numpy as np
import librosa
import json
from tensorflow.keras.models import load_model

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cough_classifier.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "features", "label_map.json")

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 200

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    return mfcc

# Load model and labels
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

def predict_cough(audio_path):
    features = extract_mfcc(audio_path)
    features = np.expand_dims(features, axis=0)

    preds = model.predict(features)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    return inv_label_map[class_id], float(confidence)

# ---- RUN ----
if __name__ == "__main__":
    test_audio = "test_cough.wav"  # put a cough wav here
    label, conf = predict_cough(test_audio)
    print(f"Predicted class: {label}")
    print(f"Confidence: {conf:.2f}")
