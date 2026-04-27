"""
real_time_prediction.py
Real-time ASL sign-language interpreter using webcam.

Controls:
  s  - save the current predicted letter to the word buffer
  c  - clear the word buffer
  q  - quit
"""

import os
import math
import pickle
import argparse
import pathlib

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# Try cvzone; fall back to MediaPipe if unavailable.
try:
    from cvzone.HandTrackingModule import HandDetector
    _USE_CVZONE = True
except ImportError:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _USE_CVZONE = False


# ---------------------------------------------------------------------------
# Fallback hand detector (MediaPipe Tasks API - works with 0.10+)
# ---------------------------------------------------------------------------
class _MPHandDetector:
    def __init__(self, max_hands: int = 1):
        import urllib.request, tempfile, os
        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("[INFO] Downloading MediaPipe hand landmarker model (~9MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path,
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw: bool = True):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            bbox_w, bbox_h = x2 - x1, y2 - y1
            hand_type = result.handedness[i][0].display_name if result.handedness else ""
            hands.append({
                "bbox": (x1, y1, bbox_w, bbox_h),
                "type": hand_type,
                "landmarks": hand_landmarks,
            })
            if draw:
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
        return hands, img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMG_SIZE = 64  # must match training


def _resolve_model_path(candidate: str) -> str:
    """Return first existing path from legacy .h5 or modern .keras variants."""
    script_dir = pathlib.Path(__file__).parent
    candidates = [
        candidate,
        str(script_dir / candidate),
        str(script_dir / "sign_language_model.keras"),
        str(script_dir / "sign_language_model.h5"),
        str(script_dir / "model.h5"),
        str(script_dir / "test_model.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No model file found. Train the model first with model_training.py."
    )


def _preprocess(img_white: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_white, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time ASL sign-language interpreter.")
    p.add_argument(
        "--model",
        default="sign_language_model.keras",
        help="Path to the trained model file (default: sign_language_model.keras).",
    )
    p.add_argument(
        "--label-dict",
        default=os.path.join(os.path.dirname(__file__), "label_dict.pkl"),
        help="Path to label_dict.pkl (default: ./label_dict.pkl).",
    )
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    p.add_argument(
        "--img-size", type=int, default=300, help="Hand canvas size (default: 300)."
    )
    p.add_argument(
        "--offset", type=int, default=20, help="Bounding-box padding (default: 20)."
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum prediction confidence to display (default: 0.6).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    model_path = _resolve_model_path(args.model)
    print(f"[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    label_dict_path = args.label_dict
    if not os.path.exists(label_dict_path):
        label_dict_path = os.path.join(os.path.dirname(__file__), "label_dict.pkl")
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError(f"label_dict.pkl not found at '{label_dict_path}'.")
    with open(label_dict_path, "rb") as f:
        label_dict = pickle.load(f)
    print(f"[INFO] Loaded {len(label_dict)} labels.")

    if _USE_CVZONE:
        detector = HandDetector(maxHands=1)
    else:
        print("[INFO] cvzone not found - using MediaPipe fallback.")
        detector = _MPHandDetector(max_hands=1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    offset = args.offset
    img_size = args.img_size
    predicted_word = ""

    print("Controls: 's' save letter | 'c' clear word | 'q' quit")

    while True:
        success, img = cap.read()
        if not success:
            print("[ERROR] Failed to read frame.")
            break

        key = cv2.waitKey(1) & 0xFF
        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            hand_type = hand.get("type", "")

            # Draw bounding box
            cv2.rectangle(
                img,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (255, 0, 255),
                2,
            )

            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            img_crop = img[y1:y2, x1:x2]
            if img_crop.size == 0:
                cv2.imshow("Sign Language Interpreter", img)
                continue

            aspect_ratio = h / w if w else 1
            if aspect_ratio > 1:
                k = img_size / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, img_size))
                w_gap = math.ceil((img_size - w_cal) / 2)
                img_white[:, w_gap : w_cal + w_gap] = img_resize
            else:
                k = img_size / w if w else 1
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (img_size, h_cal))
                h_gap = math.ceil((img_size - h_cal) / 2)
                img_white[h_gap : h_cal + h_gap, :] = img_resize

            # Prediction
            prediction = model.predict(_preprocess(img_white), verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            confidence = float(prediction[predicted_index])
            predicted_label = label_dict[predicted_index]

            # Overlay text - hand type above box, predicted letter near box
            text_x = max(x - offset, 5)
            hand_y = max(y - offset - 50, 30)
            label_y = max(y - offset - 10, 60)

            cv2.putText(
                img, f"Hand: {hand_type}",
                (text_x, hand_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            if confidence >= args.confidence:
                cv2.putText(
                    img, f"{predicted_label}  ({confidence:.0%})",
                    (text_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 3,
                )

            # Save letter on 's'
            if key == ord("s") and confidence >= args.confidence:
                predicted_word += predicted_label
                print(f"[SAVED] '{predicted_label}' -> word so far: '{predicted_word}'")

        # Clear on 'c'
        if key == ord("c"):
            predicted_word = ""
            print("[INFO] Word cleared.")

        # Display word at bottom
        cv2.putText(
            img, f"Word: {predicted_word}",
            (30, img.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3,
        )

        cv2.imshow("Sign Language Interpreter", img)

        if key == ord("q"):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
