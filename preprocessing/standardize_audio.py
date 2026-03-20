import librosa
import soundfile as sf

def standardize_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    y = librosa.effects.trim(y)[0]
    sf.write(output_path, y, 16000)
