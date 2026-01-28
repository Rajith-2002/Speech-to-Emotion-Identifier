from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import tensorflow as tf
import joblib
import librosa
import numpy as np
import os
import shutil
from pydub import AudioSegment
import uuid
import gdown

# =========================
# APP INIT
# =========================
app = FastAPI(title="Speech Emotion Recognition")
templates = Jinja2Templates(directory="templates")

# =========================
# PATH SETUP
# =========================
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "emotion_cnn_lstm.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# =========================
# GOOGLE DRIVE FILE URLs
# =========================
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"
SCALER_URL = "https://drive.google.com/uc?id=YOUR_SCALER_FILE_ID"
ENCODER_URL = "https://drive.google.com/uc?id=YOUR_ENCODER_FILE_ID"

# =========================
# AUTO DOWNLOAD MODELS
# =========================
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)

if not os.path.exists(ENCODER_PATH):
    gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)

# =========================
# LOAD MODEL & TOOLS
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

# =========================
# CONSTANTS
# =========================
N_MFCC = 13
MAX_LEN = 173

# =========================
# MP3 → WAV
# =========================
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# =========================
# MFCC EXTRACTION
# =========================
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = mfcc.T  # (173, 13)

    mfcc = scaler.transform(
        mfcc.reshape(-1, mfcc.shape[-1])
    ).reshape(mfcc.shape)

    return np.expand_dims(mfcc, axis=0)

# =========================
# ROUTES
# =========================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{file.filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if temp_path.lower().endswith(".mp3"):
        process_path = convert_mp3_to_wav(temp_path)
    else:
        process_path = temp_path

    X = extract_mfcc(process_path)
    preds = model.predict(X)[0]
    idx = np.argmax(preds)

    emotion = le.inverse_transform([idx])[0]
    confidence = round(float(preds[idx]) * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotion": emotion,
            "confidence": confidence,
            "audio_file": os.path.basename(process_path)
        }
    )

@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(file_path, media_type="audio/wav")
