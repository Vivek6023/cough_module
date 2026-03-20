import os
import subprocess
import librosa
import soundfile as sf
import pandas as pd

INPUT_DIR = "data/raw_datasets/coughvid/coughvid_20211012"
CSV_PATH = os.path.join(INPUT_DIR, "metadata_compiled.csv")

OUTPUT_ROOT = "data/standardized_audio"
TEMP_WAV = "temp.wav"

SR = 16000

df = pd.read_csv(CSV_PATH)

def map_label(row):
    cough_types = [
        row.get("cough_type_1"),
        row.get("cough_type_2"),
        row.get("cough_type_3"),
        row.get("cough_type_4"),
    ]

    cough_types = [str(ct).lower() for ct in cough_types if pd.notna(ct)]

    dry_count = cough_types.count("dry")
    wet_count = cough_types.count("wet")

    # Majority vote
    if dry_count > wet_count and dry_count > 0:
        return "dry_cough"
    if wet_count > dry_count and wet_count > 0:
        return "wet_cough"

    # Fallback
    if row.get("status") == "healthy":
        return "healthy"
    if row.get("status") == "symptomatic":
        return "covid_like"

    return "other_resp"


processed = 0
skipped = 0

for _, row in df.iterrows():
    uuid = row["uuid"]
    label = map_label(row)

    webm_path = os.path.join(INPUT_DIR, f"{uuid}.webm")
    wav_path = os.path.join(INPUT_DIR, f"{uuid}.wav")

    # Case 1: WAV already exists
    if os.path.exists(wav_path):
        input_audio = wav_path

    # Case 2: Convert WEBM → WAV
    elif os.path.exists(webm_path):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, TEMP_WAV],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            input_audio = TEMP_WAV
        except FileNotFoundError:
            skipped += 1
            continue

    else:
        skipped += 1
        continue

    try:
        y, _ = librosa.load(input_audio, sr=SR, mono=True)
        y = librosa.effects.trim(y)[0]

        out_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)

        sf.write(os.path.join(out_dir, f"{uuid}.wav"), y, SR)
        processed += 1
    except:
        skipped += 1

if os.path.exists(TEMP_WAV):
    os.remove(TEMP_WAV)

print("✅ COUGHVID standardization completed")
print("Processed:", processed)
print("Skipped:", skipped)
