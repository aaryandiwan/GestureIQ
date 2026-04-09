"""
server.py — Flask backend for community gesture data collection.

Serves the web app and provides API endpoints to persist gesture samples
in a SQLite database so the model improves from every user's contributions.

Usage:
    python server.py
    Then open http://localhost:5000
"""

import json
import time
import sqlite3
import os
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory

# ── Config ──────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "community_data.db")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
FEATURE_VECTOR_LENGTH = 73  # 63 coords + 5 distances + 5 angles
MAX_SAMPLES_PER_CLASS = 10000  # Safety cap per gesture class
RATE_LIMIT_WINDOW = 1  # seconds between submissions (per IP)

app = Flask(__name__, static_folder=WEB_DIR)

# Simple in-memory rate limiter
_rate_limits = {}


# ── Database setup ──────────────────────────────────────────────────

def get_db():
    """Get a thread-local database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
    return conn


def init_db():
    """Create the samples table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            gesture_name TEXT NOT NULL,
            features TEXT NOT NULL,
            ip_address TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_class_id ON samples(class_id)
    """)
    conn.commit()
    conn.close()


# ── Rate limiting ───────────────────────────────────────────────────

def rate_limit(f):
    """Simple per-IP rate limiter for sample submission."""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        last_request = _rate_limits.get(ip, 0)

        if now - last_request < RATE_LIMIT_WINDOW:
            return jsonify({"error": "Too many requests, slow down"}), 429

        _rate_limits[ip] = now
        return f(*args, **kwargs)
    return decorated


# ── Static file serving ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


# ── API: Submit samples ─────────────────────────────────────────────

@app.route("/api/samples", methods=["POST"])
@rate_limit
def submit_samples():
    """
    Save collected gesture samples.
    
    Body: { "classId": int, "gestureName": str, "features": [[float, ...], ...] }
    Accepts a batch of feature vectors for one gesture class.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    class_id = data.get("classId")
    gesture_name = data.get("gestureName", f"Class {class_id}")
    features_batch = data.get("features", [])

    # Validate
    if class_id is None or not isinstance(class_id, int) or class_id < 0 or class_id > 9:
        return jsonify({"error": "classId must be 0-9"}), 400

    if not isinstance(features_batch, list) or len(features_batch) == 0:
        return jsonify({"error": "features must be a non-empty array of feature vectors"}), 400

    # Validate each feature vector
    valid_features = []
    for fv in features_batch:
        if isinstance(fv, list) and len(fv) == FEATURE_VECTOR_LENGTH:
            if all(isinstance(v, (int, float)) for v in fv):
                valid_features.append(fv)

    if not valid_features:
        return jsonify({"error": "No valid feature vectors found"}), 400

    # Check class cap
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM samples WHERE class_id = ?", (class_id,)
    ).fetchone()[0]

    if count >= MAX_SAMPLES_PER_CLASS:
        conn.close()
        return jsonify({"error": f"Class {class_id} has reached the sample limit"}), 429

    # Cap to avoid exceeding limit
    remaining = MAX_SAMPLES_PER_CLASS - count
    valid_features = valid_features[:remaining]

    # Insert
    ip = request.remote_addr or "unknown"
    now = time.time()

    conn.executemany(
        "INSERT INTO samples (class_id, gesture_name, features, ip_address, created_at) VALUES (?, ?, ?, ?, ?)",
        [(class_id, gesture_name, json.dumps(fv), ip, now) for fv in valid_features]
    )
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "saved": len(valid_features),
        "message": f"Saved {len(valid_features)} samples for {gesture_name}"
    })


# ── API: Get all community samples ──────────────────────────────────

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return all community-collected samples grouped by class."""
    conn = get_db()
    rows = conn.execute(
        "SELECT class_id, features FROM samples ORDER BY class_id"
    ).fetchall()
    conn.close()

    # Group by class
    community_data = {}
    for row in rows:
        cid = str(row["class_id"])
        if cid not in community_data:
            community_data[cid] = []
        community_data[cid].append(json.loads(row["features"]))

    return jsonify({
        "data": community_data,
        "total": len(rows)
    })


# ── API: Get community stats ────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return per-class sample counts and total."""
    conn = get_db()
    rows = conn.execute(
        "SELECT class_id, gesture_name, COUNT(*) as count FROM samples GROUP BY class_id"
    ).fetchall()
    conn.close()

    stats = {}
    total = 0
    for row in rows:
        stats[str(row["class_id"])] = {
            "name": row["gesture_name"],
            "count": row["count"]
        }
        total += row["count"]

    return jsonify({"stats": stats, "total": total})


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"\n  GestureIQ Server")
    print(f"  Database: {DB_PATH}")
    print(f"  Web app:  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
