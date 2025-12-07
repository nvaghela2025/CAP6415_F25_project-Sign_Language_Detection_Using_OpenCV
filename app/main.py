import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================

MODEL_PATH = os.path.join("models", "sign_model.pkl")

# FastAPI inference app: loads trained RandomForest pipeline and exposes
# /health and /predict-image endpoints for browser and demo clients.

# Your final 11 labels (for reference / docs only)
CLASS_NAMES = [
    "BEAUTIFUL",
    "CAREFUL",
    "FIGHT",
    "HELLO",
    "NO",
    "NOW",
    "SEE",
    "TALK",
    "THANK YOU",
    "US",
    "YES",
]

NUM_LANDMARKS = 21

# ==========================================

# ---- Load model ----
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train_classifier.py first.")

print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = joblib.load(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ---- Setup MediaPipe ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # one-off frames from HTTP uploads (no tracking state)
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# ---- FastAPI app ----
app = FastAPI(
    title="Sign Language Detection API",
    description="Real-time sign detection using MediaPipe + RandomForest",
    version="1.0.0",
)

# Optional: allow frontend (React, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_landmarks(results, image_width: int, image_height: int) -> Optional[np.ndarray]:
    """
    Same feature extraction as training & realtime_demo:
    - Get 21 landmarks
    - Convert to pixel coords
    - Make them relative to the wrist (landmark 0)
    - Flatten to 63-dim vector (21 * 3)
    """
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_width
        y = lm.y * image_height
        z = lm.z
        coords.append([x, y, z])

    coords = np.array(coords)  # (21, 3)

    # Make coordinates relative to wrist
    wrist = coords[0].copy()
    coords[:, 0] -= wrist[0]
    coords[:, 1] -= wrist[1]
    coords[:, 2] -= wrist[2]

    features = coords.flatten()  # (63,)
    return features


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file (JPG/PNG/etc).
    Returns the predicted sign label and optional confidence.
    """
    # 1) Read file bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    # 2) Decode as OpenCV image (BGR)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    h, w, _ = img.shape

    # 3) Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4) Run MediaPipe Hands
    results = hands.process(img_rgb)

    # 5) Extract features
    features = extract_landmarks(results, w, h)
    if features is None:
        raise HTTPException(status_code=400, detail="No hand detected in the image.")

    # 6) Predict using trained model
    features = features.reshape(1, -1)
    pred_label = model.predict(features)[0]

    # Optional confidence (if model supports predict_proba)
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        classes = list(model.classes_)
        if pred_label in classes:
            idx = classes.index(pred_label)
            confidence = float(probs[idx])

    response = {
        "label": pred_label,
        "confidence": confidence,
    }
    return response
