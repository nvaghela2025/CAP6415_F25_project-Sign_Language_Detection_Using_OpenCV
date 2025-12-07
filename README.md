# Real-Time Sign Language Detection (MediaPipe + OpenCV + Random Forest)

## Overview

A lightweight, real-time sign language recognizer that detects 3D hand landmarks with MediaPipe, converts them to a 63-D feature vector, and classifies 12 signs with a RandomForest model. A FastAPI backend serves predictions; a browser frontend streams your webcam to the API; an OpenCV desktop demo is also included.

### How it works
- Capture frame → MediaPipe Hands finds 21 landmarks.
- Landmarks become pixel coordinates, made relative to the wrist → 63 features.
- Features feed a `StandardScaler + RandomForestClassifier`.
- Output label (and proba) is shown in the browser or OpenCV overlay. Desktop demo optionally smooths predictions with a sliding window.

### Supported labels
`HELLO, YES, NO, THANK YOU, BEAUTIFUL, CAREFUL, FIGHT, GREAT, NOW, SEE, TALK, US`

## Code layout

- `scripts/collect_data.py` — capture webcam frames, label with hotkeys, append to `data/landmarks.csv`.
- `scripts/train_classifier.py` — train pipeline (scaler + RandomForest), save `models/sign_model.pkl`, and write reports to `reports/`.
- `scripts/real_demo.py` — OpenCV/MediaPipe desktop demo using `models/sign_model.pkl`.
- `app/main.py` — FastAPI app with `/predict-image` and `/health`.
- `frontend/index.html` — browser UI that captures webcam frames and posts to the API.
- `data/landmarks.csv` — collected landmark dataset (63 features + label).
- `models/sign_model.pkl` — trained model (created by the trainer).
- `requirement.txt` — pinned dependencies.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## Train (optional if `models/sign_model.pkl` exists)

```bash
python scripts/train_classifier.py
```
Uses `data/landmarks.csv`, drops any rows with NaNs, saves the model to `models/sign_model.pkl`, and writes metrics to `reports/`.

Current run (from `reports/classification_report.txt`): test accuracy ~0.988 with strong per-class F1 (see file for details).

## Run the backend

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Health check: `http://127.0.0.1:8000/health`
- Prediction endpoint: `POST /predict-image` with an image file (the frontend handles this).

## Run the frontend

```bash
python -m http.server 3000 --directory frontend
```
Open `http://127.0.0.1:3000` in your browser. It accesses your webcam and calls the API at port 8000. Allow camera permissions.

## Desktop demo (OpenCV)

```bash
python scripts/real_demo.py
```
Uses your webcam and `models/sign_model.pkl`; press ESC to quit. Adjust lighting and keep the hand centered for best results.

Config tip: `SMOOTHING_WINDOW` in `scripts/real_demo.py` controls how many recent predictions are averaged; set to 1 to see raw outputs.

## Collect more data

```bash
python scripts/collect_data.py
```
Hotkeys while recording:  
`1` HELLO, `2` YES, `3` NO, `4` THANK YOU, `5` BEAUTIFUL, `6` CAREFUL, `7` FIGHT, `8` GREAT, `9` NOW, `0` SEE, `q` TALK, `w` US. New samples append to `data/landmarks.csv`; retrain afterward.

## Requirements (pinned)

```
numpy==1.26.4
pandas==2.2.0
opencv-python==4.9.0.80
mediapipe==0.10.7
scikit-learn==1.3.2
fastapi==0.109.0
uvicorn==0.27.0.post1
joblib==1.3.2
python-multipart==0.0.6
matplotlib==3.8.2
seaborn==0.13.2
```

## Project workflow (end-to-end)
1. Collect labeled samples with `scripts/collect_data.py` (hotkeys above) → `data/landmarks.csv`.
2. Train with `scripts/train_classifier.py` → `models/sign_model.pkl` + reports in `reports/`.
3. Serve the model with `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
4. Consume it either via:
   - Browser UI (`python -m http.server 3000 --directory frontend` → open `http://127.0.0.1:3000`), or
   - Desktop demo (`python scripts/real_demo.py`).
5. If predictions drift, collect more balanced data and retrain.

## Frontend behavior (at a glance)
- Uses `getUserMedia` for webcam, captures frames to canvas, POSTs to `/predict-image`.
- Shows current label, confidence, sentence mode, and history.
- Calls API at `http://127.0.0.1:8000` by default; adjust URL in `frontend/index.html` if you change host/port.

## Troubleshooting
- Only one label appears: reduce smoothing in `scripts/real_demo.py`, improve lighting, center hand, or retrain with more data.
- Model not found: ensure `models/sign_model.pkl` exists (run trainer) and paths in scripts point to it.
- CORS or fetch errors: confirm API at `127.0.0.1:8000` is running before opening the frontend.
- Mediapipe import errors: ensure `pip install -r requirement.txt` succeeded inside your virtualenv.
