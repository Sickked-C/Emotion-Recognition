"""
Emotion Recognition — Predict emotion from a single audio file.

Usage:
    python predict.py <path_to_audio_file>

Example:
    python predict.py sample.wav

Supported formats: .wav, .mp3, .ogg
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MODEL_PATH = "emotion_recognition_model_LSTM.h5"
SCALER_PATH = "scaler.pkl"  # optional — see note below

EMOTIONS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "surprise"
}

EMOTION_EMOJI = {
    "neutral":  "😐",
    "calm":     "😌",
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "disgust":  "🤢",
    "surprise": "😲"
}

# ─────────────────────────────────────────
# Feature extraction (same as training)
# ─────────────────────────────────────────
def extract_features(data, sample_rate):
    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(path):
    """Load audio and extract features — same preprocessing as training."""
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    return features, sample_rate


# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────
def predict_emotion(audio_path: str):
    # 1. Check file exists
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    # 2. Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Please train the model first by running APP.ipynb")
        print("   Then save it with: model_lstm.save('emotion_recognition_model_LSTM.h5')")
        sys.exit(1)

    print(f"🎵 Loading audio: {audio_path}")

    # 3. Extract features
    features, sample_rate = get_features(audio_path)
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Feature shape: {features.shape}")

    # 4. Scale features
    # Note: ideally use the same scaler fitted during training (saved via joblib)
    # If scaler.pkl exists, load it; otherwise fit a new one (less accurate)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        features_scaled = scaler.transform(features.reshape(1, -1))
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))

    # 5. Reshape for model input: (1, n_features, 1)
    features_input = np.expand_dims(features_scaled, axis=2)

    # 6. Load model & predict
    print(f"\n🤖 Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    predictions = model.predict(features_input, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_emotion = EMOTIONS[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    # 7. Display results
    print("\n" + "─" * 40)
    print(f"  {EMOTION_EMOJI[predicted_emotion]}  Predicted Emotion: {predicted_emotion.upper()}")
    print(f"     Confidence: {confidence:.1f}%")
    print("─" * 40)

    # Show top 3
    top3_idx = np.argsort(predictions)[::-1][:3]
    print("\n  Top 3 predictions:")
    for idx in top3_idx:
        bar_len = int(predictions[idx] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {EMOTION_EMOJI[EMOTIONS[idx]]} {EMOTIONS[idx]:<10} [{bar}] {predictions[idx]*100:.1f}%")

    print()
    return predicted_emotion, confidence


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        print("Example: python predict.py sample.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    predict_emotion(audio_path)
