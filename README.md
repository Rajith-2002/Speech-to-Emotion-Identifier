# Speech Emotion Recognition using Deep Learning

## Overview

Speech Emotion Recognition is a Deep Learning application that predicts human emotions from speech audio. Understanding emotions from speech has practical applications in customer support, virtual assistants, healthcare, education, and human-computer interaction. This project allows users to upload an audio file and predicts the speaker's emotion along with a confidence score through a FastAPI-based web interface.

---

## Features

- Predicts emotions from speech audio
- Supports WAV and MP3 audio files
- Automatically converts MP3 files to WAV
- Extracts MFCC features from speech audio
- Uses a CNN-LSTM Deep Learning model
- Displays predicted emotion with confidence score
- FastAPI web interface for real-time prediction
- Automatically downloads trained model files from Google Drive on first run

---

## Dataset

Dataset Used: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

The model classifies the following emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise

---

## Technologies Used

- Python
- TensorFlow
- FastAPI
- Librosa
- NumPy
- Scikit-learn
- Joblib
- Pydub
- Gdown
- HTML
- CSS
- JavaScript

---

## Why These Technologies

**TensorFlow**

Used to build and train the CNN-LSTM Deep Learning model because it provides powerful tools for neural network development.

**Librosa**

Used for audio preprocessing and MFCC feature extraction since it is widely used in speech and audio processing.

**FastAPI**

Chosen because it provides a fast, lightweight, and high-performance backend for deploying machine learning models.

**Scikit-learn**

Used for feature scaling, label encoding, train-test splitting, and computing class weights.

**Gdown**

Used to automatically download trained model files from Google Drive, allowing users to run the project without manually copying model files.

---

# Project Workflow

## 1 Data Collection

The project uses the RAVDESS speech emotion dataset containing 1440 original audio files.

---

## 2 Audio Preprocessing

Each audio file is

- Converted to Mono
- Resampled to 16000 Hz

This standardizes every audio sample before feature extraction.

---

## 3 Data Augmentation

To improve model performance, four versions of every audio sample were created.

- Original Audio
- Noise Addition
- Pitch Shift
- Time Stretch

This increased the dataset size from 1440 audio files to 5760 training samples and helped reduce overfitting.

---

## 4 Feature Extraction

MFCC (Mel Frequency Cepstral Coefficients) features were extracted.

Configuration

- Number of MFCC Features = 13
- Maximum Length = 173 Frames

MFCC captures the important speech characteristics required for emotion recognition.

---

## 5 Feature Scaling

StandardScaler was used to normalize MFCC features before training.

---

## 6 Label Encoding

Emotion labels were converted into numerical values using LabelEncoder.

---

## 7 Model Architecture

The model consists of

Input

↓

Conv1D (32 Filters)

↓

MaxPooling1D

↓

Conv1D (64 Filters)

↓

MaxPooling1D

↓

LSTM (128 Units)

↓

Dense (64)

↓

Dropout (0.5)

↓

Softmax Output Layer (8 Emotions)

---

## Why CNN-LSTM

CNN extracts important local features from MFCC representations.

LSTM captures temporal dependencies in speech.

Combining CNN and LSTM enables the model to learn both spatial and sequential information from speech audio.

---

## Model Training

Optimizer

Adam

Learning Rate

0.0002

Loss Function

Sparse Categorical Crossentropy

Early Stopping

Validation Loss Monitoring

Patience = 5

Class Weights were used to reduce bias caused by class imbalance.

---

## Deployment

The trained model was deployed using FastAPI.

Workflow

User uploads an audio file

↓

MP3 is automatically converted into WAV if necessary

↓

MFCC features are extracted

↓

Features are normalized using the saved StandardScaler

↓

CNN-LSTM predicts the emotion

↓

Predicted emotion and confidence score are displayed on the webpage

---

## Project Structure

```
Speech-to-Emotion-Identifier/

│── models/
│      emotion_cnn_lstm.h5
│      scaler.pkl
│      label_encoder.pkl
│
│── templates/
│      index.html
│
│── uploads/
│
│── app.py
│── requirements.txt
│── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/Rajith-2002/Speech-to-Emotion-Identifier.git
```

## Navigate to the Project

```bash
cd Speech-to-Emotion-Identifier
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
uvicorn app:app --reload
```

## Open in Browser

```
http://127.0.0.1:8000
```

---

## Sample Output

Input

Speech Audio File

Output

Predicted Emotion

Confidence Score

---

## Limitations

- Trained on the RAVDESS dataset only
- Performance may decrease on noisy real-world recordings
- Supports only English speech
- Emotion prediction may vary with recording quality

---

## Future Improvements

- Train using larger and more diverse datasets
- Improve robustness against background noise
- Support multilingual speech emotion recognition
- Deploy using Docker and cloud platforms
- Integrate Transformer-based speech models such as Wav2Vec2 or Whisper
- Add real-time microphone emotion detection

---

## Key Learnings

Through this project I gained practical experience in

- Audio preprocessing
- Feature engineering using MFCC
- Deep Learning model development
- CNN-LSTM architecture
- TensorFlow
- Model evaluation
- FastAPI deployment
- Machine Learning model integration
- REST API development

---

## Author

Rajith Bestus S

GitHub

https://github.com/Rajith-2002

LinkedIn

https://www.linkedin.com/in/rajith-bestus-s