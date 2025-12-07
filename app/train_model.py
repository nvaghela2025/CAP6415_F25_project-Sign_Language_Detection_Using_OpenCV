import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import joblib

import matplotlib.pyplot as plt

# ========================
# PATHS
# ========================
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

CSV_PATH = os.path.join("E:/nilay sign/random_forest/data/landmarks.csv")
MODEL_PATH = os.path.join("E:/nilay sign/random_forest/models/sign_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================
# LOAD DATA
# ========================
print(f"[INFO] Loading data from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)

print(f"[INFO] Raw dataset shape: {df.shape}")
print("[INFO] Example rows:")
print(df.head())

# ------------------------
# CLEAN LABEL COLUMN
# ------------------------
VALID_LABELS = [
    "HELLO", "YES", "NO", "THANK YOU",
    "BEAUTIFUL", "CAREFUL", "FIGHT", "GREAT",
    "NOW", "SEE", "TALK", "US"
]

# convert to string
df["label"] = df["label"].astype(str)

# keep only valid labels
df = df[df["label"].isin(VALID_LABELS)].copy()

print(f"[INFO] Cleaned dataset shape: {df.shape}")
print("[INFO] Label value counts:")
print(df["label"].value_counts())

# ========================
# FEATURES / LABELS
# ========================
X = df.drop("label", axis=1).values
y = df["label"].values

# Sorted list of label names (now all strings)
label_names = sorted(df["label"].unique())

# ========================
# TRAIN / TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ========================
# RANDOM FOREST MODEL
# ========================
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

print("[INFO] Training RandomForest model ...")
clf.fit(X_train, y_train)

# ========================
# PREDICTIONS
# ========================
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

acc_test = accuracy_score(y_test, y_pred)
acc_train = accuracy_score(y_train, y_train_pred)

print(f"\n[RESULT] Test Accuracy: {acc_test:.4f}")
print(f"[RESULT] Train Accuracy: {acc_train:.4f}\n")

print("[RESULT] Classification Report:")
report = classification_report(y_test, y_pred, digits=4)
print(report)

# ========================
# SAVE MODEL
# ========================
joblib.dump(clf, MODEL_PATH)
print(f"\n[SAVED] Trained model saved to: {MODEL_PATH}")

# ========================
# CONFUSION MATRIX
# ========================
cm = confusion_matrix(y_test, y_pred, labels=label_names)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(label_names))
plt.xticks(tick_marks, label_names, rotation=45, ha="right")
plt.yticks(tick_marks, label_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"[SAVED] Confusion Matrix → {cm_path}")

# ========================
# SIMPLE METRICS SUMMARY (OVERALL ACCURACY)
# ========================
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy"], [acc_test])
plt.title("Model Accuracy Summary")
plt.ylim(0, 1)
for idx, value in enumerate([acc_test]):
    plt.text(idx, value + 0.01, f"{value:.3f}", ha="center")
metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.png")
plt.savefig(metrics_path)
plt.close()
print(f"[SAVED] Metrics Summary → {metrics_path}")

# ========================
# TRAIN VS TEST ACCURACY COMPARISON
# ========================
plt.figure(figsize=(6, 4))
bars = ["Train", "Test"]
vals = [acc_train, acc_test]
plt.bar(bars, vals, color=["orange", "green"])
plt.title("Train vs Test Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(vals):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
tt_path = os.path.join(RESULTS_DIR, "train_test_accuracy.png")
plt.savefig(tt_path)
plt.close()
print(f"[SAVED] Train vs Test Accuracy → {tt_path}")

# ========================
# PER-CLASS F1 SCORE BAR CHART
# ========================
f1_per_class = f1_score(
    y_test,
    y_pred,
    average=None,
    labels=label_names
)

plt.figure(figsize=(10, 5))
x_pos = np.arange(len(label_names))
plt.bar(x_pos, f1_per_class)
plt.xticks(x_pos, label_names, rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("F1 Score per Class")
for i, v in enumerate(f1_per_class):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
f1_path = os.path.join(RESULTS_DIR, "f1_per_class.png")
plt.tight_layout()
plt.savefig(f1_path)
plt.close()
print(f"[SAVED] F1 per Class → {f1_path}")

# ========================
# PER-CLASS ACCURACY / RECALL BAR CHART
# ========================
cm_sum = cm.sum(axis=1, keepdims=True)
per_class_acc = np.divide(
    cm.diagonal().reshape(-1, 1),
    cm_sum,
    out=np.zeros_like(cm_sum, dtype=float),
    where=cm_sum != 0
).flatten()

plt.figure(figsize=(10, 5))
x_pos = np.arange(len(label_names))
plt.bar(x_pos, per_class_acc)
plt.xticks(x_pos, label_names, rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("Per-Class Accuracy (Recall)")
for i, v in enumerate(per_class_acc):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
pc_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
plt.tight_layout()
plt.savefig(pc_path)
plt.close()
print(f"[SAVED] Per-Class Accuracy → {pc_path}")

# ========================
# ROC CURVE (MULTI-CLASS, MICRO + MACRO)
# ========================
# Only compute ROC if we have probabilities and at least 2 classes
if len(label_names) > 1:
    y_test_bin = label_binarize(y_test, classes=label_names)
    n_classes = y_test_bin.shape[1]
    y_score = clf.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC (AUC = {roc_auc['micro']:.3f})",
        linewidth=2,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC (AUC = {roc_auc['macro']:.3f})",
        linewidth=2,
    )

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1,
            alpha=0.6,
            label=f"{label_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(fontsize=7, loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"[SAVED] ROC Curve → {roc_path}")
else:
    print("[WARN] Only one class present after filtering; ROC curve not generated.")

# ========================
# PLACEHOLDER RESULT IMAGES FOR README / REPORT
# ========================
dummy = np.zeros((400, 400, 3), dtype=np.uint8)

hello_path = os.path.join(RESULTS_DIR, "sample_frame_hello.png")
thank_path = os.path.join(RESULTS_DIR, "sample_frame_thankyou.png")
ui_path = os.path.join(RESULTS_DIR, "ui_live_detection.png")

plt.imsave(hello_path, dummy)
plt.imsave(thank_path, dummy)
plt.imsave(ui_path, dummy)

print(f"[SAVED] Placeholder sample frames:")
print(f"  → {hello_path}")
print(f"  → {thank_path}")
print(f"  → {ui_path}")

print("\n[INFO] All metrics and result images saved successfully!")

