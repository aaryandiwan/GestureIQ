"""
utils.py — Feature extraction utilities for GestureIQ.

This module provides functions to convert raw MediaPipe hand landmarks
into a normalised, augmented feature vector suitable for classification.
"""

import numpy as np


# ── Indices of the five fingertip landmarks ──────────────────────────
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Triplets (base, mid, tip) used to compute per-finger joint angles
FINGER_JOINTS = [
    (1, 2, 4),    # Thumb
    (5, 6, 8),    # Index
    (9, 10, 12),  # Middle
    (13, 14, 16), # Ring
    (17, 18, 20), # Pinky
]

# ── Human-readable gesture names (key = class label string) ──────────
GESTURE_NAMES = {
    "0": "Fist",
    "1": "One",
    "2": "Peace",
    "3": "Three",
    "4": "Four",
    "5": "Open Palm",
    "6": "Thumbs Up",
    "7": "Thumbs Down",
    "8": "Hang Loose",
    "9": "OK Sign",
}


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle ∠ABC (in radians) formed by three 3-D points.

    Parameters
    ----------
    a, b, c : np.ndarray of shape (3,)
        The three points. 'b' is the vertex of the angle.

    Returns
    -------
    float
        Angle in radians in [0, π].
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    # Clamp to [-1, 1] to guard against floating-point errors
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def extract_features(landmarks) -> np.ndarray:
    """
    Convert 21 MediaPipe hand-landmark objects into a flat feature vector.

    Processing steps
    ----------------
    1. Collect (x, y, z) for each of the 21 landmarks → shape (21, 3).
    2. Subtract the wrist position (landmark 0) so features are
       position-invariant.
    3. Divide by the maximum absolute coordinate value so all
       normalised coordinates lie in [-1, 1].
    4. Flatten the 21×3 normalised coords into 63 values.
    5. Append 5 fingertip-to-wrist Euclidean distances.
    6. Append 5 finger joint angles (base → mid → tip).

    Parameters
    ----------
    landmarks : list-like of 21 objects with .x, .y, .z attributes
        Typically ``hand_landmarks.landmark`` from MediaPipe.

    Returns
    -------
    np.ndarray of shape (73,)
        [63 normalised coords | 5 distances | 5 angles]
    """
    # ── 1. Raw coordinates ───────────────────────────────────────────
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)

    # ── 2. Translate so wrist = origin ───────────────────────────────
    wrist = coords[0].copy()
    coords -= wrist

    # ── 3. Scale to [-1, 1] ──────────────────────────────────────────
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords /= max_abs

    # ── 4. Flatten normalised coordinates ────────────────────────────
    flat_coords = coords.flatten()  # length 63

    # ── 5. Fingertip-to-wrist distances ──────────────────────────────
    # After normalisation the wrist is at the origin, so distance
    # is simply the L2 norm of each fingertip's coordinate vector.
    distances = np.array([
        np.linalg.norm(coords[tip]) for tip in FINGERTIP_IDS
    ])  # length 5

    # ── 6. Per-finger joint angles ───────────────────────────────────
    angles = np.array([
        _angle_between(coords[base], coords[mid], coords[tip])
        for base, mid, tip in FINGER_JOINTS
    ])  # length 5

    # ── 7. Concatenate everything into one vector ────────────────────
    feature_vector = np.concatenate([flat_coords, distances, angles])
    return feature_vector  # shape (73,)


def get_feature_names() -> list[str]:
    """
    Return human-readable names for each element of the feature vector.

    Useful for inspecting feature importances or column headers in CSVs.
    """
    names = []
    for i in range(21):
        for axis in ("x", "y", "z"):
            names.append(f"lm{i}_{axis}")
    for tip in FINGERTIP_IDS:
        names.append(f"dist_tip{tip}_wrist")
    for base, mid, tip in FINGER_JOINTS:
        names.append(f"angle_{base}_{mid}_{tip}")
    return names
