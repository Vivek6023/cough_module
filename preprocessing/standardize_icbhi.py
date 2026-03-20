import os
import librosa
import soundfile as sf
import pandas as pd

INPUT_DIR = os.path.join("data", "raw_datasets", "icbhi", "Respiratory_Sound_Database", "Respiratory_Sound_Database", "audio_and_txt_files")
OUTPUT_ROOT = "data/standardized_audio"
SR = 16000

# Load patient diagnosis
diagnosis_df = pd.read_csv(os.path.join(os.path.dirname(INPUT_DIR), "patient_diagnosis.csv"), header=None, names=["patient_id", "diagnosis"])
diagnosis_dict = {int(pid): diag for pid, diag in zip(diagnosis_df["patient_id"], diagnosis_df["diagnosis"])}

def get_label(filename):
    patient_id = int(filename.split("_")[0])
    diagnosis = diagnosis_dict.get(patient_id)
    if diagnosis == "Asthma":
        return "asthma"
    if diagnosis == "COPD":
        return "copd"
    if diagnosis == "Bronchiectasis":
        return "bronchitis"
    if diagnosis == "Pneumonia":
        return "pneumonia"
    return None

processed = 0

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".wav"):
        continue

    label = get_label(file)
    if label is None:
        continue

    y, _ = librosa.load(os.path.join(INPUT_DIR, file), sr=SR, mono=True)
    y = librosa.effects.trim(y)[0]

    out_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    sf.write(os.path.join(out_dir, file), y, SR)
    processed += 1

print("✅ ICBHI standardization completed")
print("Processed files:", processed)
