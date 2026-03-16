# 🎧 Emotion Recognition from Speech

A deep learning project that classifies **8 emotions from audio recordings** using CNN and LSTM architectures, trained on the RAVDESS dataset.

---

## 📊 Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **LSTM** | **67.22%** | **0.67** |
| CNN | 57.96% | 0.56 |

**→ LSTM outperforms CNN by ~10%** — confirming sequential modeling is better suited for audio/temporal data.

### LSTM Detailed Report

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.68 | 0.63 | 0.65 |
| Calm | 0.78 | 0.70 | **0.74** |
| Disgust | 0.58 | 0.55 | 0.56 |
| Fear | 0.47 | 0.75 | 0.58 |
| Happy | 0.69 | 0.58 | 0.63 |
| Neutral | **0.78** | **0.79** | **0.79** |
| Sad | 0.73 | 0.65 | 0.69 |
| Surprise | 0.66 | 0.78 | 0.71 |

> Note: `class_weight='balanced'` was applied to handle class imbalance (Fear: 76 samples vs Angry: 160 samples), improving Fear F1 from 0.48 → 0.58.

---

## 🔍 How it works

```
Audio files (RAVDESS)
        ↓
Feature Extraction:
  - ZCR, Chroma STFT, MFCC, RMS, Mel Spectrogram
        ↓
Data Augmentation:
  - Noise injection, Time stretch, Pitch shift (3x samples)
        ↓
Train CNN  ──── accuracy: 57.96%
Train LSTM ──── accuracy: 67.22%  ✓ winner
        ↓
Inference via predict.py
```

---

## 🧠 Emotions Recognized

`angry` · `calm` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## 🚀 Quick Predict

```bash
# Install dependencies
pip install tensorflow librosa scikit-learn joblib numpy

# Predict emotion from audio file
python predict.py your_audio.wav
```

**Output:**
```
🎵 Loading audio: your_audio.wav
🤖 Loading model: emotion_recognition_model_LSTM.h5

────────────────────────────────────────
  😌  Predicted Emotion: CALM
     Confidence: 87.3%
────────────────────────────────────────

  Top 3 predictions:
  😌 calm       [████████████████████] 87.3%
  😐 neutral    [████░░░░░░░░░░░░░░░░] 19.1%
  😄 happy      [██░░░░░░░░░░░░░░░░░░] 8.2%
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Feature Extraction | librosa (MFCC, ZCR, Mel Spectrogram) |
| Deep Learning | TensorFlow / Keras |
| Models | CNN, LSTM |
| Dataset | RAVDESS (1,440 audio files) |
| Evaluation | scikit-learn (Macro F1) |

---

## 📦 Installation

```bash
# 1. Clone repo
git clone https://github.com/Sickked-C/Emotion-Recognition.git
cd Emotion-Recognition

# 2. Install dependencies
pip install tensorflow librosa scikit-learn joblib numpy pandas matplotlib seaborn

# 3. Download dataset (for training)
# RAVDESS: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en
# Place in: RAVDESS/audio_speech_actors_01-24/

# 4. Train model (optional — pretrained model included)
jupyter notebook APP.ipynb

# 5. Predict
python predict.py your_audio.wav
```

---

## 📁 Project Structure

```
.
├── APP.ipynb                           # Training notebook
├── predict.py                          # Inference script
├── emotion_recognition_model_LSTM.h5   # Pretrained LSTM model
├── scaler.pkl                          # Feature scaler
├── encoder.pkl                         # Label encoder
├── link data.txt                       # Dataset download link
└── README.md
```

---

## 💡 Key Takeaways

- **LSTM > CNN for audio** — sequential architecture captures temporal patterns better
- **Fear is hardest to classify** — fewest samples (76 vs avg 135)
- **class_weight helps** — balancing loss improved minority class performance
- **Macro F1 > Accuracy** for imbalanced datasets like this

---

## 📄 License

MIT License