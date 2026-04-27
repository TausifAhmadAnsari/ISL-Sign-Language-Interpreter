"""
data_collection.py
Collect hand-gesture images for each ASL letter (A-Z).

Controls:
  s  - save the current cropped frame
  q  - quit
"""

import os
import math
import time
import argparse

import cv2
import numpy as np

# Try importing cvzone; fall back to a pure-MediaPipe detector if unavailable.
try:
    from cvzone.HandTrackingModule import HandDetector
    _USE_CVZONE = True
except ImportError:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _USE_CVZONE = False


# ---------------------------------------------------------------------------
# Fallback hand detector using MediaPipe Tasks API (0.10+)
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
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect hand-gesture images for a given ASL letter."
    )
    parser.add_argument(
        "letter",
        type=str,
        help="The letter to collect images for (A-Z).",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "Data"),
        help="Root directory for the dataset (default: ./Data).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=300,
        help="Output image canvas size in pixels (default: 300).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=20,
        help="Padding around detected hand bounding box (default: 20).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    letter = args.letter.upper()
    if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(letter) != 1:
        raise ValueError(f"Invalid letter: '{letter}'. Must be A-Z.")

    folder = os.path.join(args.data_dir, letter)
    os.makedirs(folder, exist_ok=True)

    if _USE_CVZONE:
        detector = HandDetector(maxHands=1)
    else:
        print("[INFO] cvzone not found - using MediaPipe fallback.")
        detector = _MPHandDetector(max_hands=1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    counter = 0
    offset = args.offset
    img_size = args.img_size

    print(f"Collecting images for '{letter}' -> {folder}")
    print("Press 's' to save | 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from camera.")
            break

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            img_crop = img[y1:y2, x1:x2]
            if img_crop.size == 0:
                cv2.imshow("Webcam", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
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

            cv2.imshow("Cropped", img_crop)
            cv2.imshow("White Canvas", img_white)

        cv2.imshow("Webcam", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and hands:
            filename = os.path.join(folder, f"Image_{time.time():.6f}.jpg")
            cv2.imwrite(filename, img_white)
            counter += 1
            print(f"  Saved [{counter}]: {filename}")

        elif key == ord("q"):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total images saved: {counter}")


if __name__ == "__main__":
    main()
