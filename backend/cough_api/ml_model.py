from inference.predict_cough import predict_cough

def run_prediction(audio_path):
    label, confidence = predict_cough(audio_path)
    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }
