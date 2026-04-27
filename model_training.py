"""
model_training.py
Train a CNN to classify ASL hand-gesture images (A-Z).

Usage:
  python model_training.py [--data-dir ./Data] [--epochs 50] [--model-out sign_language_model.keras]
"""

import os
import pickle
import argparse

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress verbose TF logs

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 64  # pixels (must match data_collection.py pipeline)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ASL sign-language CNN.")
    p.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "Data"),
        help="Root dataset directory (default: ./Data).",
    )
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50).")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    p.add_argument(
        "--model-out",
        default=os.path.join(os.path.dirname(__file__), "sign_language_model.keras"),
        help="Output model path (default: sign_language_model.keras).",
    )
    p.add_argument(
        "--label-dict-out",
        default=os.path.join(os.path.dirname(__file__), "label_dict.pkl"),
        help="Output label-dict path (default: label_dict.pkl).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(data_dir: str):
    images, labels, label_dict = [], [], {}
    folders = sorted([f for f in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, f))])
    for idx, folder_name in enumerate(folders):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label_dict[idx] = folder_name
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)

    if not images:
        raise RuntimeError(f"No images found in '{data_dir}'. Check the path.")

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    num_classes = len(label_dict)
    labels_oh = to_categorical(labels, num_classes=num_classes)
    return images, labels_oh, label_dict, num_classes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"[INFO] Loading dataset from: {args.data_dir}")
    images, labels_oh, label_dict, num_classes = load_dataset(args.data_dir)
    print(f"[INFO] Loaded {len(images)} images across {num_classes} classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels_oh, test_size=0.2, random_state=42, stratify=labels_oh
    )

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    print("[INFO] Training...")
    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[RESULT] Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    target_names = [label_dict[i] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Save
    model.save(args.model_out)
    print(f"[INFO] Model saved to: {args.model_out}")

    with open(args.label_dict_out, "wb") as f:
        pickle.dump(label_dict, f)
    print(f"[INFO] Label dict saved to: {args.label_dict_out}")


if __name__ == "__main__":
    main()
