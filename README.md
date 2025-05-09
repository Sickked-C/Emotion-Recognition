# 🎧 Emotion Recognition from Speech

This project demonstrates how to recognize **emotions from audio recordings** using deep learning techniques. The goal is to process audio data, extract relevant features, train two models (**CNN** and **LSTM**), and compare their performance to determine which is more accurate.

## 🎯 Project Objectives

- Load and preprocess speech data.
- Extract acoustic features like MFCCs using `librosa`.
- Build and train two models:
  - **CNN** (Convolutional Neural Network)
  - **LSTM** (Long Short-Term Memory)
- Evaluate the performance of both models.
- Save the better-performing model as an `.h5` file.

## 📁 Files in This Repository

- `APP.ipynb`: Main Jupyter notebook containing all steps:
  - Data loading
  - Feature extraction
  - Model training and comparison
  - Saving the final model
- `data.txt`: [Download training data here](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)  
  *(This text file contains a list of paths or references to emotion-labeled audio files.)*

## 🧠 Emotions Recognized

The number and type of emotions depend on the dataset referred to in `data.txt`, but typically include:

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise

## 🛠️ Requirements

Install required packages:

```bash
pip install numpy pandas librosa scikit-learn tensorflow matplotlib
```

---
## ▶️ How to Run
Download or clone this repository.

Open APP.ipynb in Jupyter Notebook or Google Colab.

Make sure data.txt is in the same directory or accessible.

Run the notebook cell by cell to train and compare the models.

The best model will be saved as best_model.h5.

---
## 📊 Model Evaluation
The notebook compares CNN and LSTM models using:

Accuracy

Precision

Recall

F1-score

---
## 📚 Technologies Used
Python

Jupyter Notebook

Librosa

TensorFlow/Keras

Scikit-learn

---
## 👨‍💻 Author
Sickked-C

📝 License
This project is licensed under the MIT License.
