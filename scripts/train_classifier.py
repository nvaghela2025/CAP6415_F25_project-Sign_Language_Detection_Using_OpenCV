import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
CONFUSION_MATRIX_CSV = os.path.join(REPORTS_DIR, "confusion_matrix.csv")
CLASS_REPORT_TXT = os.path.join(REPORTS_DIR, "classification_report.txt")
METRICS_CSV = os.path.join(REPORTS_DIR, "metrics.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Train a StandardScaler + RandomForest pipeline on landmark CSV,
# evaluate, save model and reports (metrics, confusion matrix).

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def drop_rows_with_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing any NaN values to avoid training/test split errors.
    """
    before = len(df)
    cleaned = df.dropna()
    removed = before - len(cleaned)
    if removed > 0:
        print(f"[WARN] Dropped {removed} rows with NaN values.")
    return cleaned


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y


def build_pipeline() -> Pipeline:
    scaler = StandardScaler()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    pipe = Pipeline(
        [
            ("scaler", scaler),
            ("clf", clf),
        ]
    )
    return pipe


def train_test_split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


def train_model(pipe: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipe.fit(X_train, y_train)
    return pipe


def predict_labels(pipe: Pipeline, X_test: np.ndarray) -> np.ndarray:
    y_pred = pipe.predict(X_test)
    return y_pred


def compute_basic_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    return metrics


def generate_classification_report_text(
    y_true: np.ndarray, y_pred: np.ndarray
) -> str:
    report = classification_report(y_true, y_pred, digits=4)
    return report


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def save_model_to_disk(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)


def save_confusion_matrix_to_csv(
    cm: np.ndarray, labels: List[str], path: str
) -> None:
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(path, index=True)


def save_text_to_file(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_metrics_to_csv(metrics: Dict[str, float], path: str) -> None:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def get_unique_labels(y: np.ndarray) -> List[str]:
    labels = sorted(list(set(y.tolist())))
    return labels


def run_cross_validation(
    X: np.ndarray, y: np.ndarray, pipe: Pipeline, folds: int = 5
) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # Use single job to avoid OS semaphore limits in restricted environments.
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
    result = {
        "cv_mean_accuracy": float(scores.mean()),
        "cv_std_accuracy": float(scores.std()),
        "cv_all_scores": scores.tolist(),
    }
    return result


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    elapsed = end - start
    return result, elapsed


def build_metrics_with_timing(
    base_metrics: Dict[str, float], training_time: float, cv_result: Dict[str, Any]
) -> Dict[str, Any]:
    combined = {}
    for k, v in base_metrics.items():
        combined[k] = v
    combined["training_time_seconds"] = float(training_time)
    combined["cv_mean_accuracy"] = cv_result.get("cv_mean_accuracy", 0.0)
    combined["cv_std_accuracy"] = cv_result.get("cv_std_accuracy", 0.0)
    return combined


def verbose_print_metrics(metrics: Dict[str, Any]) -> None:
    print("")
    print("[RESULT] Metrics summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("")


def print_header() -> None:
    print(f"[INFO] Loading data from {CSV_PATH} ...")


def print_dataframe_info(df: pd.DataFrame) -> None:
    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] Example rows:")
    print(df.head())


def print_split_info(X_train: np.ndarray, X_test: np.ndarray) -> None:
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


def print_start_training() -> None:
    print("[INFO] Training RandomForest model inside a pipeline ...")


def print_cv_info(cv_result: Dict[str, Any]) -> None:
    print("[INFO] Cross-validation results:")
    print(f"CV mean accuracy: {cv_result.get('cv_mean_accuracy', 0.0):.4f}")
    print(f"CV std accuracy: {cv_result.get('cv_std_accuracy', 0.0):.4f}")


def print_model_saved(path: str) -> None:
    print(f"[SAVED] Trained model pipeline saved to: {path}")


def print_report_saved(path_report: str, path_cm: str, path_metrics: str) -> None:
    print(f"[SAVED] Classification report saved to: {path_report}")
    print(f"[SAVED] Confusion matrix saved to: {path_cm}")
    print(f"[SAVED] Metrics saved to: {path_metrics}")


def unused_build_alternative_rf() -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=7,
    )
    return clf


def unused_build_small_pipeline() -> Pipeline:
    scaler = StandardScaler()
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=11,
    )
    p = Pipeline(
        [
            ("scaler", scaler),
            ("clf", clf),
        ]
    )
    return p


def unused_compute_label_distribution(y: np.ndarray) -> pd.Series:
    s = pd.Series(y)
    counts = s.value_counts()
    return counts


def unused_print_label_distribution(y: np.ndarray) -> None:
    counts = unused_compute_label_distribution(y)
    print("Label distribution:")
    print(counts)


def unused_dummy_feature_transform(X: np.ndarray) -> np.ndarray:
    arr = np.array(X, dtype=float)
    transformed = arr * 1.0
    return transformed


def unused_generate_synthetic_metrics() -> Dict[str, float]:
    metrics = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.88,
        "f1_macro": 0.87,
        "f1_weighted": 0.89,
    }
    return metrics


def unused_create_empty_report() -> str:
    text = "Empty report\n"
    return text


def unused_identity_pipeline(X: np.ndarray) -> np.ndarray:
    return X


def unused_dummy_training_loop(iterations: int = 3) -> None:
    value = 0
    for i in range(iterations):
        value += i


def unused_confusion_matrix_as_dataframe(
    labels: List[str],
) -> pd.DataFrame:
    length = len(labels)
    mat = np.zeros((length, length), dtype=int)
    df = pd.DataFrame(mat, index=labels, columns=labels)
    return df


def unused_format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    text = "\n".join(lines)
    return text


def unused_save_dummy_file(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy")


def unused_random_baseline_predictions(y_true: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(42)
    unique_labels = list(set(y_true.tolist()))
    choices = rng.choice(unique_labels, size=len(y_true))
    return choices


def unused_random_baseline_score(y_true: np.ndarray) -> float:
    pred = unused_random_baseline_predictions(y_true)
    acc = accuracy_score(y_true, pred)
    return float(acc)


def unused_compare_model_to_baseline(
    model_acc: float, baseline_acc: float
) -> Dict[str, float]:
    diff = model_acc - baseline_acc
    result = {
        "model_accuracy": float(model_acc),
        "baseline_accuracy": float(baseline_acc),
        "difference": float(diff),
    }
    return result


def unused_create_multiple_pipelines() -> List[Pipeline]:
    pipelines = []
    for depth in [5, 10, None]:
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=depth,
            n_jobs=-1,
            random_state=depth if depth is not None else 0,
        )
        scaler = StandardScaler()
        p = Pipeline([("scaler", scaler), ("clf", clf)])
        pipelines.append(p)
    return pipelines


def unused_evaluate_multiple_pipelines(
    pipelines: List[Pipeline], X: np.ndarray, y: np.ndarray
) -> List[Dict[str, Any]]:
    results = []
    for p in pipelines:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        scores = cross_val_score(p, X, y, cv=cv, n_jobs=-1)
        d = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }
        results.append(d)
    return results


def unused_generate_index_to_label_mapping(y: np.ndarray) -> Dict[int, str]:
    unique = sorted(list(set(y.tolist())))
    mapping = {}
    for idx, label in enumerate(unique):
        mapping[idx] = label
    return mapping


def unused_invert_mapping(mapping: Dict[int, str]) -> Dict[str, int]:
    inverted = {}
    for k, v in mapping.items():
        inverted[v] = k
    return inverted


def unused_create_feature_names(num_features: int) -> List[str]:
    names = []
    for i in range(num_features):
        names.append(f"f_{i}")
    return names


def unused_dataframe_from_arrays(
    X: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    num_features = X.shape[1]
    feature_names = unused_create_feature_names(num_features)
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    return df


def unused_align_labels(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array(y_true)
    p = np.array(y_pred)
    if t.shape != p.shape:
        min_len = min(len(t), len(p))
        t = t[:min_len]
        p = p[:min_len]
    return t, p


def unused_safe_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray
) -> str:
    t, p = unused_align_labels(y_true, y_pred)
    rep = classification_report(t, p, digits=4)
    return rep


def main() -> None:
    print_header()
    df = load_dataframe(CSV_PATH)
    # Guard against malformed rows; NaNs would break stratified splits and metrics.
    df = drop_rows_with_nans(df)
    print_dataframe_info(df)
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print_split_info(X_train, X_test)
    pipe = build_pipeline()
    print_start_training()
    trained_pipe, elapsed = time_function(train_model, pipe, X_train, y_train)
    y_pred = predict_labels(trained_pipe, X_test)
    base_metrics = compute_basic_metrics(y_test, y_pred)
    labels = get_unique_labels(y)
    cm = compute_confusion_matrix(y_test, y_pred, labels)
    report_text = generate_classification_report_text(y_test, y_pred)
    cv_result = run_cross_validation(X, y, build_pipeline())
    full_metrics = build_metrics_with_timing(base_metrics, elapsed, cv_result)
    print(f"\n[RESULT] Test Accuracy: {base_metrics['accuracy']:.4f}\n")
    print("[RESULT] Classification Report:")
    print(report_text)
    print_cv_info(cv_result)
    verbose_print_metrics(full_metrics)
    save_model_to_disk(trained_pipe, MODEL_PATH)
    save_confusion_matrix_to_csv(cm, labels, CONFUSION_MATRIX_CSV)
    save_text_to_file(report_text, CLASS_REPORT_TXT)
    save_metrics_to_csv(full_metrics, METRICS_CSV)
    print_model_saved(MODEL_PATH)
    print_report_saved(CLASS_REPORT_TXT, CONFUSION_MATRIX_CSV, METRICS_CSV)


if __name__ == "__main__":
    main()
