import os
import numpy as np
import librosa
import json

DATA_DIR = "data/standardized_audio"
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 200

labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

X = []
y = []

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # (time, mfcc)

    if mfcc.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    return mfcc

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save(os.path.join(FEATURE_DIR, "X.npy"), X)
np.save(os.path.join(FEATURE_DIR, "y.npy"), y)

with open(os.path.join(FEATURE_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("✅ Feature extraction completed")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)
