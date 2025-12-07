import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# ================ CONFIGURATION AND CONSTANTS ===============
# ============================================================
# Capture webcam frames, map hotkeys to labels, and append 63-D landmark
# features plus the label into data/landmarks.csv for training.

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

#: Human-readable list of the 12 sign classes we are going to store.
CLASS_NAMES: List[str] = [
    "HELLO",
    "YES",
    "NO",
    "THANK YOU",
    "BEAUTIFUL",
    "CAREFUL",
    "FIGHT",
    "GREAT",
    "NOW",
    "SEE",
    "TALK",
    "US",
]

#: Map keyboard keys to labels for fast data collection.
#: We use ASCII codes obtained via ord('character').
KEY_TO_LABEL: Dict[int, str] = {
    ord("1"): "HELLO",
    ord("2"): "YES",
    ord("3"): "NO",
    ord("4"): "THANK YOU",
    ord("5"): "BEAUTIFUL",
    ord("6"): "CAREFUL",
    ord("7"): "FIGHT",
    ord("8"): "GREAT",
    ord("9"): "NOW",
    ord("0"): "SEE",
    ord("q"): "TALK",
    ord("w"): "US",
}

#: Number of landmarks in MediaPipe Hands for a single hand.
NUM_LANDMARKS: int = 21  # each has x, y, z -> 63 features


# ============================================================
# =============== EXTRA DATA STRUCTURES (UNUSED) =============
# ============================================================

@dataclass
class AppConfig:
    """
    Configuration class for the application.

    NOTE: This class is **not used** directly in the current script.
    It is only here to demonstrate how configuration could be wrapped
    in a dataclass in a larger project.
    """

    data_dir: str = DATA_DIR
    csv_path: str = CSV_PATH
    num_landmarks: int = NUM_LANDMARKS
    class_names: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = CLASS_NAMES.copy()

    def summary(self) -> str:
        """
        Return a string summary of this configuration.
        """
        return (
            f"AppConfig(data_dir={self.data_dir}, "
            f"csv_path={self.csv_path}, "
            f"num_landmarks={self.num_landmarks}, "
            f"num_classes={len(self.class_names)})"
        )


@dataclass
class DebugTimer:
    """
    Simple timing helper to measure durations.

    This class is declared but never used in the main script.
    """

    start_time: float = time.time()

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000.0


@dataclass
class LandmarkSample:
    """
    A small data structure representing a single landmark sample.

    Attributes
    ----------
    features : np.ndarray
        A 1D array of length 63 (21 landmarks * 3 coordinates).
    label : str
        The class label associated with this sample.
    """

    features: np.ndarray
    label: str

    def to_csv_row(self) -> List[Any]:
        """
        Convert the sample to a list suitable for CSV writing.

        Returns
        -------
        List[Any]
            A flat list of features followed by the label.
        """
        return list(self.features) + [self.label]


class SimpleLogger:
    """
    A minimal logger-like class.

    This logger is not used in the main routine but included to illustrate
    how structured logging might be added to a real system.
    """

    def __init__(self, prefix: str = "[LOGGER]"):
        self.prefix = prefix

    def info(self, msg: str) -> None:
        print(f"{self.prefix} INFO: {msg}")

    def warn(self, msg: str) -> None:
        print(f"{self.prefix} WARN: {msg}")

    def error(self, msg: str) -> None:
        print(f"{self.prefix} ERROR: {msg}")




def debug_print_config() -> None:
    """
    Print configuration details to the console.

    This function is **not used** by the main loop, but it could be
    helpful during development or debugging to ensure that all
    configuration variables are set correctly.
    """
    print("===== DEBUG CONFIGURATION =====")
    print(f"DATA_DIR        : {DATA_DIR}")
    print(f"CSV_PATH        : {CSV_PATH}")
    print(f"NUM_LANDMARKS   : {NUM_LANDMARKS}")
    print(f"CLASS_NAMES     : {CLASS_NAMES}")
    print("KEY_TO_LABEL MAP:")
    for k, v in KEY_TO_LABEL.items():
        print(f"  key code {k} -> label '{v}'")
    print("===== END CONFIGURATION =====")


def unused_normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Return a normalized version of 'vec'.

    Not used in the main code.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def unused_center_points(points: np.ndarray) -> np.ndarray:
    """
    Center a point cloud by subtracting its centroid.

    Not used in the main code.
    """
    if points.size == 0:
        return points
    centroid = points.mean(axis=0)
    return points - centroid


def unused_dummy_feature_generator(num_points: int = 63) -> np.ndarray:
    """
    Generate a dummy feature vector filled with zeros.

    Not used in the main code.
    """
    return np.zeros(num_points, dtype=float)


def unused_statistics_on_samples(samples: List[LandmarkSample]) -> Dict[str, float]:
    """
    Compute simple statistics over a list of samples.

    Not used in the main code.
    """
    if not samples:
        return {"mean": 0.0, "std": 0.0}

    all_features = np.vstack([s.features for s in samples])
    return {
        "mean": float(all_features.mean()),
        "std": float(all_features.std()),
    }


def unused_dummy_menu() -> None:
    """
    Pretend to show a menu in the console.

    Not used in the main code.
    """
    print("===== UNUSED MENU =====")
    print("1) Collect data")
    print("2) Train model")
    print("3) Evaluate model")
    print("4) Exit")
    print("========================")


def unused_explain_key_mapping() -> str:
    """
    Return a human-readable string explaining the key mapping.

    Not used in the main code.
    """
    lines = ["Key mapping for data collection:"]
    for key_code, label in sorted(KEY_TO_LABEL.items(), key=lambda kv: kv[0]):
        lines.append(f"  {chr(key_code)} -> {label}")
    return "\n".join(lines)


def unused_get_class_index(label: str) -> int:
    """
    Get the index of a given label in CLASS_NAMES.

    Not used in the main code.
    """
    try:
        return CLASS_NAMES.index(label)
    except ValueError:
        return -1


def unused_pretty_frame_text() -> List[str]:
    """
    Return a list of text lines that could be rendered on the frame.

    Not used in the main code.
    """
    lines = [
        "Sign Language Data Collector",
        "----------------------------",
        "Position your hand inside the camera frame.",
        "Press the specified key to record a sample.",
        "Ensure consistent distance and angle.",
    ]
    return lines


def unused_color_palette() -> Dict[str, Tuple[int, int, int]]:
    """
    Return a dictionary of named BGR colors.

    Not used in the main code.
    """
    return {
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "white": (255, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
    }


# ---------- Extra unused ML-style helpers ----------

def unused_fake_train_model(data: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """
    Fake "training" function that pretends to train a model.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix.
    labels : List[str]
        List of labels.

    Returns
    -------
    Dict[str, Any]
        A fake model object (dictionary).
    """
    model = {
        "num_samples": data.shape[0],
        "num_features": data.shape[1] if data.size > 0 else 0,
        "num_classes": len(set(labels)),
        "trained": True,
    }
    return model


def unused_fake_evaluate_model(model: Dict[str, Any]) -> Dict[str, float]:
    """
    Fake evaluation for the dummy model.

    Not used in the code, but shows how evaluation could be structured.
    """
    if not model.get("trained", False):
        return {"accuracy": 0.0, "loss": 1.0}
    # Completely random values, for demo.
    return {"accuracy": 0.85, "loss": 0.3}


def unused_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> np.ndarray:
    """
    Compute a simple confusion matrix.

    This is not used in the script but demonstrates ML utilities.
    """
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            i = class_to_idx[t]
            j = class_to_idx[p]
            mat[i, j] += 1

    return mat


def unused_softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a 1D or 2D array.

    Not used in the script but illustrates typical ML math helpers.
    """
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def unused_one_hot_encode(label: str, classes: List[str]) -> np.ndarray:
    """
    One-hot encode a label.

    Not used in the main code.
    """
    vec = np.zeros(len(classes), dtype=float)
    if label in classes:
        idx = classes.index(label)
        vec[idx] = 1.0
    return vec


def unused_decode_one_hot(vec: np.ndarray, classes: List[str]) -> str:
    """
    Decode a one-hot vector back to a label.

    Not used in the main code.
    """
    idx = int(np.argmax(vec))
    if 0 <= idx < len(classes):
        return classes[idx]
    return "UNKNOWN"



def unused_cv_resize(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize image keeping aspect ratio."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def unused_cv_rotate(frame: np.ndarray, angle: float = 90) -> np.ndarray:
    """Rotate an image around its center."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(frame, M, (w, h))


def unused_cv_crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a region from the image."""
    return frame[y:y+h, x:x+w]


def unused_cv_gaussian_blur(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(frame, (k, k), 0)


def unused_cv_median_blur(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply median blur filter."""
    return cv2.medianBlur(frame, k)


def unused_cv_bilateral_filter(frame: np.ndarray) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(frame, 9, 75, 75)


def unused_cv_canny_edges(frame: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Detect edges using Canny."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def unused_cv_sobel(frame: np.ndarray) -> np.ndarray:
    """Compute Sobel edge gradients."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return cv2.magnitude(grad_x, grad_y)


def unused_cv_laplacian(frame: np.ndarray) -> np.ndarray:
    """Apply Laplacian edge detector."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)


def unused_cv_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def unused_cv_to_hsv(frame: np.ndarray) -> np.ndarray:
    """Convert image to HSV color space."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def unused_cv_to_lab(frame: np.ndarray) -> np.ndarray:
    """Convert image to LAB color space."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)


def unused_cv_equalize_hist(frame: np.ndarray) -> np.ndarray:
    """Apply histogram equalization on grayscale image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return eq


def unused_cv_erode(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Erode image."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(frame, kernel, iterations=1)


def unused_cv_dilate(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dilate image."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(frame, kernel, iterations=1)


def unused_cv_morph_open(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological opening (remove noise)."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)


def unused_cv_morph_close(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological closing (fill gaps)."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)


def unused_cv_draw_bbox(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Draw rectangle on frame."""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame


def unused_cv_draw_text(frame: np.ndarray, text: str, pos: tuple = (10, 30)) -> np.ndarray:
    """Draw text on frame."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame


def unused_cv_contours(frame: np.ndarray) -> list:
    """Find contours in image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def unused_cv_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
    """Return a background subtractor object."""
    return cv2.createBackgroundSubtractorMOG2()


def unused_cv_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """Compute dense optical flow (Farneback)."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def unused_apply_color_mask(frame: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:
    """Mask specific color range using HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(frame, frame, mask=mask)

# ============================================================
# ================== CSV / FILE UTILITIES ====================
# ============================================================


def ensure_data_dir_exists(path: str) -> None:
    """
    Ensure that the data directory exists.

    Parameters
    ----------
    path : str
        Path to the directory that should exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_csv_if_needed(csv_path: str, num_landmarks: int) -> None:
    """
    Create a CSV file with a header if it does not already exist.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    num_landmarks : int
        Number of landmarks (each contributes x, y, z).
    """
    if os.path.exists(csv_path):
        return

    header: List[str] = []
    for i in range(num_landmarks):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"[INFO] Created new CSV with header at {csv_path}")


def append_sample_to_csv(csv_path: str, sample: LandmarkSample) -> None:
    """
    Append a single LandmarkSample to the CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    sample : LandmarkSample
        The sample to append.
    """
    row = sample.to_csv_row()
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def unused_load_all_samples(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load all samples from the CSV as features and labels.

    This is unused in the main data collection script, but demonstrates
    how we might read back the data for training.
    """
    if not os.path.exists(csv_path):
        return np.zeros((0, NUM_LANDMARKS * 3)), []
    X = []
    y = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            *features, label = row
            feats = np.array(features, dtype=float)
            X.append(feats)
            y.append(label)
    if not X:
        return np.zeros((0, NUM_LANDMARKS * 3)), []
    return np.vstack(X), y


def unused_count_samples_per_class(labels: List[str]) -> Dict[str, int]:
    """
    Count label frequencies.

    Unused in the main script.
    """
    counts = defaultdict(int)
    for lbl in labels:
        counts[lbl] += 1
    return counts


# ============================================================
# ================= MEDIAPIPE / LANDMARK LOGIC ===============
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_mediapipe_hands() -> mp_hands.Hands:
    """
    Create and return a MediaPipe Hands object configured for video.

    Returns
    -------
    mp_hands.Hands
        The configured hands detector.
    """
    hands_instance = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands_instance


def extract_landmarks(
    results: mp_hands.Hands,
    image_width: int,
    image_height: int,
) -> Optional[np.ndarray]:
    """
    Extract a flattened 63-length feature vector from MediaPipe results.

    The pipeline is:
    1. Take the first detected hand from `results.multi_hand_landmarks`.
    2. Convert normalized (x, y, z) coordinates to pixel space for x and y.
    3. Subtract the wrist coordinates (landmark 0) from all points to
       make them relative.
    4. Flatten the resulting 21x3 array into a vector of length 63.

    Parameters
    ----------
    results : mp_hands.Hands
        The result object from MediaPipe Hands processing.
    image_width : int
        Width of the image (for converting normalized x).
    image_height : int
        Height of the image (for converting normalized y).

    Returns
    -------
    Optional[np.ndarray]
        A 1D numpy array of length 63 if a hand is detected, otherwise None.
    """
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    coords: List[List[float]] = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_width
        y = lm.y * image_height
        z = lm.z  # relative depth
        coords.append([x, y, z])

    coords_arr = np.array(coords, dtype=float)  # shape (21, 3)

    # Make coordinates relative to wrist (landmark 0)
    wrist = coords_arr[0].copy()
    coords_arr[:, 0] -= wrist[0]
    coords_arr[:, 1] -= wrist[1]
    coords_arr[:, 2] -= wrist[2]

    features: np.ndarray = coords_arr.flatten()  # 21 * 3 = 63
    return features


def unused_scale_features(features: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Multiply features by a scalar factor.

    Not used in current script.
    """
    return features * factor


def unused_clip_features(features: np.ndarray, limit: float = 1000.0) -> np.ndarray:
    """
    Clip features to a fixed absolute value.

    Not used in current script.
    """
    return np.clip(features, -limit, limit)


# ============================================================
# ================== DISPLAY / OVERLAY LOGIC =================
# ============================================================


def draw_landmarks_on_frame(
    frame: np.ndarray,
    results: mp_hands.Hands,
) -> None:
    """
    Draw hand landmarks on a frame using MediaPipe drawing utilities.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from OpenCV (modified in place).
    results : mp_hands.Hands
        The result object from MediaPipe Hands processing.
    """
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
            )


def draw_instructions(frame: np.ndarray) -> None:
    """
    Draws text instructions and key mappings onto the frame.

    Parameters
    ----------
    frame : np.ndarray
        The current video frame on which we will draw text.
    """
    h, w, _ = frame.shape
    y0 = 20
    dy = 20

    cv2.putText(
        frame,
        "Press key to SAVE sample:",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    lines = [
        "1:HELLO  2:YES   3:NO   4:THANK YOU",
        "5:BEAUTIFUL  6:CAREFUL  7:FIGHT  8:GREAT",
        "9:NOW  0:SEE  q:TALK  w:US",
        "ESC: Quit",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, y0 + (i + 1) * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
        )


def draw_sample_counts(
    frame: np.ndarray,
    sample_counts: Dict[str, int],
) -> None:
    """
    Draw the per-class sample counts at the bottom-left of the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    sample_counts : Dict[str, int]
        Mapping of label -> count.
    """
    h, w, _ = frame.shape
    y_offset = h - 20

    parts: List[str] = []
    for label in CLASS_NAMES:
        count = sample_counts[label]
        parts.append(f"{label}:{count}")
    text_counts = " | ".join(parts)

    truncated_text = text_counts[: min(len(text_counts), 120)]

    cv2.putText(
        frame,
        truncated_text,
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
    )


def unused_draw_center_crosshair(frame: np.ndarray) -> None:
    """
    Draw a small crosshair at the center of the frame.

    Not used in main loop.
    """
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)


def unused_put_transparent_text(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int] = (10, 10),
    alpha: float = 0.4,
) -> None:
    """
    Draw semi-transparent text on the frame.

    Not used in main loop. Demonstrates how overlays could be improved.
    """
    overlay = frame.copy()
    cv2.putText(
        overlay,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ============================================================
# ===================== MAIN APPLICATION =====================
# ============================================================


def main() -> None:
    """
    Main entry point for the sign language landmark collector.

    This sets up:
    - The data directory and CSV.
    - MediaPipe hands.
    - The OpenCV video capture loop.
    """
    # Ensure data directory and CSV file exist
    ensure_data_dir_exists(DATA_DIR)
    create_csv_if_needed(CSV_PATH, NUM_LANDMARKS)

    # For counting how many samples per label we have collected so far
    sample_counts: Dict[str, int] = defaultdict(int)

    # Initialize MediaPipe Hands
    hands = create_mediapipe_hands()

    # Initialize webcam (device index 0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting data collection.")
    print("[INFO] Controls:")
    print("  1: HELLO       2: YES        3: NO        4: THANK YOU")
    print("  5: BEAUTIFUL   6: CAREFUL    7: FIGHT     8:GREAT")
    print("  9: NOW         0: SEE        q: TALK      w: US")
    print("  ESC: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Mirror the image for a more natural experience
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR frame from OpenCV to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        results = hands.process(rgb)

        # Draw landmarks if present
        draw_landmarks_on_frame(frame, results)

        # Draw on-screen instructions
        draw_instructions(frame)

        # Draw current sample counts
        draw_sample_counts(frame, sample_counts)

        # Show the frame in a window
        cv2.imshow("Sign Data Collection", frame)

        # Wait for a key press for 1ms
        key = cv2.waitKey(1) & 0xFF

        # ESC key (27) -> exit program
        if key == 27:
            print("[INFO] ESC pressed. Exiting.")
            break

        # If a key corresponds to one of our labels, try to save sample
        if key in KEY_TO_LABEL:
            label = KEY_TO_LABEL[key]

            # Extract features only if a hand is detected
            features = extract_landmarks(results, w, h)
            if features is None:
                print(f"[WARN] No hand detected. Not saving for label: {label}")
                continue

            # Create a LandmarkSample and append to CSV
            sample = LandmarkSample(features=features, label=label)
            append_sample_to_csv(CSV_PATH, sample)

            sample_counts[label] += 1
            print(
                f"[INFO] Saved sample #{sample_counts[label]} for label: {label}"
            )

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ======================= SCRIPT ENTRY =======================
# ============================================================

if __name__ == "__main__":
    # We only run main() when this file is executed directly.
    main()
