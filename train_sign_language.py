import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------------------------------------------------
# Configuration Dataclass
# -------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    val_split: float = 0.2
    seed: int = 42
    base_trainable: bool = False  # fine-tune base model or not
    learning_rate: float = 1e-4


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def set_global_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_output_dir(path: str):
    """Create output directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_class_names(data_dir: str) -> List[str]:
    """
    Get sorted list of class names from subfolders.
    Each subfolder in data_dir is treated as one class.
    """
    class_names = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    class_names = sorted(class_names)
    if not class_names:
        raise ValueError(f"No class folders found in {data_dir}")
    return class_names


def save_label_map(class_names: List[str], output_dir: str):
    """Save label map (index -> class name) as JSON."""
    label_map = {i: name for i, name in enumerate(class_names)}
    path = os.path.join(output_dir, "label_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    print(f"[INFO] Saved label map to {path}")


# -------------------------------------------------------------------
# Dataset Creation
# -------------------------------------------------------------------

def create_datasets(
    cfg: TrainConfig,
    class_names: List[str]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
    """
    Create training and validation datasets using image_dataset_from_directory
    and compute class weights for imbalanced datasets.
    """

    print("[INFO] Creating training and validation datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        validation_split=cfg.val_split,
        subset='training',
        seed=cfg.seed,
        image_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.data_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        validation_split=cfg.val_split,
        subset='validation',
        seed=cfg.seed,
        image_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size
    )

    # Cache and prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1024).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Compute class weights
    all_labels = []
    for _, labels in train_ds.unbatch():
        all_labels.append(int(labels.numpy()))
    all_labels = np.array(all_labels)
    num_classes = len(class_names)

    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

    print("[INFO] Class weights:", class_weights)

    return train_ds, val_ds, class_weights


# -------------------------------------------------------------------
# Data Augmentation & Preprocessing Layers
# -------------------------------------------------------------------

def build_augmentation_layer(img_size: int) -> tf.keras.Sequential:
    """Build data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_preprocessing_layer() -> layers.Layer:
    """Wrap EfficientNet's preprocess_input as a Keras Layer."""
    def preprocess_fn(images):
        return preprocess_input(images)

    return layers.Lambda(preprocess_fn, name="efficientnet_preprocessing")


# -------------------------------------------------------------------
# Model Building
# -------------------------------------------------------------------

def build_model(cfg: TrainConfig, num_classes: int) -> tf.keras.Model:
    """
    Build a transfer-learning model using EfficientNetB0 as base.
    """

    print("[INFO] Building EfficientNetB0-based model...")

    # Input layer
    inputs = layers.Input(
        shape=(cfg.img_size, cfg.img_size, 3),
        name="input_image"
    )

    # Data augmentation
    x = build_augmentation_layer(cfg.img_size)(inputs)

    # Preprocessing (scale + normalization for EfficientNet)
    x = build_preprocessing_layer()(x)

    # Base model - EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling="avg"
    )

    base_model.trainable = cfg.base_trainable  # freeze or fine-tune

    # Classification head
    x = base_model.output
    x = layers.Dropout(0.3, name="dropout_top")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="sign_language_model")

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())

    return model


# -------------------------------------------------------------------
# Training Utilities: Plots & Evaluation
# -------------------------------------------------------------------

def plot_training_history(history: tf.keras.callbacks.History, output_dir: str):
    """Plot training & validation loss/accuracy curves."""
    print("[INFO] Plotting training history...")

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved accuracy curve to {acc_path}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved loss curve to {loss_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """Plot and save confusion matrix."""
    print("[INFO] Plotting confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: List[str],
    output_dir: str
):
    """Evaluate model on validation set and save report + confusion matrix."""
    print("[INFO] Evaluating model on validation set...")

    # Collect predictions and true labels
    y_true = []
    y_pred = []

    for batch_images, batch_labels in val_ds:
        preds = model.predict(batch_images, verbose=0)
        batch_pred = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(batch_pred.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\n" + report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Saved classification report to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, output_dir)


# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------

def train(cfg: TrainConfig):
    """Main training function."""
    set_global_seed(cfg.seed)
    ensure_output_dir(cfg.output_dir)

    # Detect classes
    class_names = get_class_names(cfg.data_dir)
    num_classes = len(class_names)
    print(f"[INFO] Detected {num_classes} classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    save_label_map(class_names, cfg.output_dir)

    # Create datasets and class weights
    train_ds, val_ds, class_weights = create_datasets(cfg, class_names)

    # Build model
    model = build_model(cfg, num_classes)

    # Callbacks
    checkpoint_path = os.path.join(cfg.output_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Save final model (in case you also want the last epoch)
    final_model_path = os.path.join(cfg.output_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"[INFO] Saved final model to {final_model_path}")

    # Plot training curves
    plot_training_history(history, cfg.output_dir)

    # Evaluate with confusion matrix and classification report
    evaluate_model(model, val_ds, class_names, cfg.output_dir)

    print("[INFO] Training and evaluation complete.")


# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train Sign Language / Hand Gesture Detector with EfficientNetB0"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset root folder (each subfolder = one class)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sign_model_output",
        help="Directory to save model, plots, and reports."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size (height=width). EfficientNetB0 default is 224."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (0–1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--base_trainable",
        action="store_true",
        help="If set, unfreezes EfficientNet base (fine-tuning)."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate."
    )

    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        seed=args.seed,
        base_trainable=args.base_trainable,
        learning_rate=args.learning_rate,
    )

    return cfg


# -------------------------------------------------------------------
# Main Entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    config = parse_args()
    train(config)
