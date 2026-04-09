# GestureIQ


**[Live DEMO!](https://aaryandiwan.github.io/GestureIQ/web/)** 

A complete, end-to-end hand gesture recognition pipeline in Python
| Component | Tech |
|---|---|
| Hand landmark detection | MediaPipe (21 landmarks, x/y/z) |
| Webcam capture & display | OpenCV |
| Classification | scikit-learn `RandomForestClassifier` |
| Feature engineering | NumPy |
| Model persistence | joblib |

---

## Project Structure

```
hand_gesture_project/
├── data_collection.py   # Step 1 — collect labelled gesture samples
├── train_model.py       # Step 2 — train the classifier
├── inference.py         # Step 3 — real-time webcam predictions
├── utils.py             # Shared feature-extraction utilities
├── model/               # Saved model & label encoder (.pkl)
├── dataset/             # Collected gesture data (.csv)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
 Clone / copy this folder
cd hand_gesture_project

Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Collect Data

```bash
python data_collection.py
```

- A webcam window opens with hand-landmark overlay.
- **Press `0`–`9`** to label the current hand pose as that class.
  - For example, press `0` for an "open palm", `1` for a "fist", etc.
- The on-screen HUD shows how many samples have been saved per class.
- **Press `q`** to quit.
- Samples are appended to `dataset/gesture_data.csv`.

---

## Step 2 — Train the Model

```bash
python train_model.py
```

- Loads `dataset/gesture_data.csv`.
- Splits data 80/20 (stratified).
- Trains a `RandomForestClassifier` with 200 estimators.
- Prints a classification report with per-class precision, recall, and F1.
- Saves the model to `model/gesture_model.pkl` and the label encoder to `model/label_encoder.pkl`.

---

## Step 3 — Real-Time Inference

```bash
python inference.py
```

- Opens a webcam feed with live hand-landmark drawing.
- Displays the **predicted gesture class** and **confidence %** in large on-screen text.
- Shows an **FPS counter** in the top-right corner.
- **Press `q`** to quit.

---

## Adding More Gesture Classes

1. **Define your new gesture** (e.g., "thumbs up" = class `5`).
2. Run `data_collection.py` and press the corresponding digit key while performing the gesture.
3. Collect enough samples (50–100 recommended).
4. Re-run `train_model.py` to retrain with the new class included.
5. Run `inference.py` — the new gesture will be recognised automatically.

You can use classes `0` through `9` for up to **10 gesture types**.

---
## Edge-Case Handling

| Scenario | Behaviour |
|---|---|
| No hand detected | On-screen message; no prediction made |
| Model files missing | Clear error message directing you to run `train_model.py` |
| Empty / missing dataset | Error message directing you to run `data_collection.py` |
| Fewer than 2 classes | Training aborts with a helpful message |

---

##  Feature Vector (73 elements)

| Slice | Count | Description |
|---|---|---|
| `[0:63]` | 63 | Normalised (x, y, z) for each of the 21 landmarks |
| `[63:68]` | 5 | Euclidean distance from each fingertip to the wrist |
| `[68:73]` | 5 | Joint angle (base → mid → tip) for each finger |

All coordinates are translated so the wrist is at the origin and scaled to [−1, 1].
