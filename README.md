# Sign Language Interpreter

An AI-powered real-time American Sign Language (ASL) interpreter that recognises hand gestures A-Z via webcam and builds words letter by letter.

---

## Overview

The system uses computer vision and a convolutional neural network (CNN) to detect hand gestures and predict the corresponding alphabet letter. Each detected letter can be appended to an on-screen word buffer, enabling basic word formation for the hearing and speech-impaired community.

---

## Features

- Real-time hand detection via webcam
- Alphabet recognition (A-Z)
- CNN-based classification with confidence threshold
- Step-by-step word formation with keyboard control
- Left/Right hand type detection
- Works on Windows, macOS, and Linux
- Optional dependency on `cvzone` - falls back to pure MediaPipe if unavailable

---

## Technologies

- Python 3.9+
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- scikit-learn
- cvzone (optional)

---

## Project Structure

```
Sign-Language-Interpreter/
├── Data/                       # Dataset folders (one sub-folder per letter A-Z)
├── image/                      # Screenshots and demo outputs
├── data_collection.py          # Collect gesture images per letter
├── model_training.py           # Train the CNN
├── real_time_prediction.py     # Live sign detection and word formation
├── label_dict.pkl              # Label mapping (index -> letter)
├── sign_language_model.keras   # Trained model (generated after training)
├── requirements.txt
└── README.md
```

---

## Setup

### 0. Python Version Requirement

> **Important:** This project requires **Python 3.11**. Python 3.12 and above are not supported because TensorFlow does not yet have compatible builds for those versions on macOS.

Check your Python version:
```bash
python3 --version
```

If you don't have Python 3.11:
```bash
brew install python@3.11
```

### 1. Create a virtual environment

**Windows / Linux**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux
```

**macOS**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

**Windows / Linux**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS (Apple Silicon or Intel)**
```bash
pip install --upgrade pip
pip install -r requirements-macos.txt
```

> `tensorflow-macos` and `tensorflow-metal` are required on macOS — the standard `tensorflow` package does not install correctly on Apple hardware.

> `cvzone` is optional. If it is not installed the scripts fall back to MediaPipe automatically.

**macOS camera access:** Go to System Settings → Privacy & Security → Camera and enable access for Terminal (or your IDE).

### 2. Collect training images (skip if using the provided dataset)

Run `data_collection.py` once per letter, pressing `s` to save frames and `q` to quit:

```bash
python data_collection.py A
python data_collection.py B
# ... repeat for each letter C-Z
```

Each invocation saves white-background hand images to `Data/<LETTER>/`.

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./Data` | Root dataset directory |
| `--img-size` | `300` | Canvas size (pixels) |
| `--offset` | `20` | Bounding-box padding |
| `--camera` | `0` | Camera index |

### 3. Train the model

```bash
python model_training.py
```

This produces `sign_language_model.keras` and `label_dict.pkl`.

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./Data` | Dataset root |
| `--epochs` | `50` | Maximum training epochs |
| `--batch-size` | `32` | Batch size |
| `--model-out` | `sign_language_model.keras` | Output model path |

### 4. Run the interpreter

```bash
python real_time_prediction.py
```

| Key | Action |
|-----|--------|
| `s` | Append the current predicted letter to the word |
| `c` | Clear the word buffer |
| `q` | Quit |

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `sign_language_model.keras` | Model path |
| `--camera` | `0` | Camera index |
| `--confidence` | `0.6` | Minimum confidence to show/save a prediction |

---

## Notes on model format

The training script saves models in the modern `.keras` format (TensorFlow 2.x+). If you have a legacy `.h5` file, it will be loaded automatically - the scripts search for both formats in the project directory.

---

## Future improvements

- Sentence-level translation
- Text-to-speech output
- GUI or web-based interface
- Support for dynamic / motion gestures

---

## Author

Tausif Ahmad Ansari
