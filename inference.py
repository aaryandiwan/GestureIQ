"""
inference.py — GestureIQ: Real-time hand gesture recognition via webcam.

Usage
-----
    python inference.py

Prerequisites
-------------
    - A trained model at  model/gesture_model.pkl
    - A label encoder at  model/label_encoder.pkl
    (Run train_model.py first.)

Controls
--------
    q : Quit the application.
"""

import os
import sys
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
from utils import extract_features, GESTURE_NAMES

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

# ── MediaPipe setup ──────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def load_model():
    """Load the trained classifier and label encoder from disk."""
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("        Run train_model.py first.")
        sys.exit(1)
    if not os.path.isfile(ENCODER_PATH):
        print(f"[ERROR] Label encoder not found at {ENCODER_PATH}")
        print("        Run train_model.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("[INFO] Model and label encoder loaded successfully.")
    return model, label_encoder


def main() -> None:
    model, label_encoder = load_model()

    # ── Open webcam ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam opened. Press 'q' to quit.")

    # ── Set window to fullscreen ─────────────────────────────────────
    window_name = "GestureIQ"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            # Mirror the frame for a natural experience
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            prediction_text = "No hand detected"
            confidence_text = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # ── Draw landmarks ───────────────────────────────
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # ── Extract features and predict ─────────────────
                    features = extract_features(hand_landmarks.landmark)
                    features = features.reshape(1, -1)

                    proba = model.predict_proba(features)[0]
                    class_idx = np.argmax(proba)
                    confidence = proba[class_idx] * 100
                    label = label_encoder.inverse_transform([class_idx])[0]
                    friendly_name = GESTURE_NAMES.get(str(label), str(label))

                    prediction_text = f"Gesture: {friendly_name}"
                    confidence_text = f"Confidence: {confidence:.1f}%"

            # ── FPS calculation ──────────────────────────────────────
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time

            # ── Draw HUD ─────────────────────────────────────────────
            # Prediction label (large)
            cv2.putText(
                frame,
                prediction_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

            # Confidence percentage
            if confidence_text:
                cv2.putText(
                    frame,
                    confidence_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            # FPS counter (top-right corner)
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(window_name, frame)

            # ── Quit on 'q' or ESC ───────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC key
                print("[INFO] Quitting inference.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
