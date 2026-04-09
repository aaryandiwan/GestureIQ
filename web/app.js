/**
 * app.js — GestureIQ: Browser-based hand gesture recognition
 *
 * Uses MediaPipe Hands for landmark detection and a simple
 * K-Nearest Neighbours (KNN) classifier for gesture recognition.
 * Everything runs client-side — no server required.
 */

// ── Gesture definitions ────────────────────────────────────────────
const GESTURE_NAMES = {
  0: "Fist",
  1: "One",
  2: "Peace",
  3: "Three",
  4: "Four",
  5: "Open Palm",
  6: "Thumbs Up",
  7: "Thumbs Down",
  8: "Hang Loose",
  9: "OK Sign",
};

const GESTURE_EMOJIS = {
  0: "✊",  1: "☝️",  2: "✌️",  3: "🤟",  4: "🖐️",
  5: "✋",  6: "👍",  7: "👎",  8: "🤙",  9: "👌",
};

// ── Fingertip & joint indices (same as Python utils.py) ────────────
const FINGERTIP_IDS = [4, 8, 12, 16, 20];
const FINGER_JOINTS = [
  [1, 2, 4],   // Thumb
  [5, 6, 8],   // Index
  [9, 10, 12], // Middle
  [13, 14, 16],// Ring
  [17, 18, 20],// Pinky
];

// ── State ──────────────────────────────────────────────────────────
let currentMode = "collect";      // "collect" or "predict"
let activeClass = null;           // currently selected gesture class
let cameraRunning = false;
let modelTrained = false;

// Training data: { classId: [ [featureVector], ... ], ... }
const trainingData = {};
// Counts per class
const sampleCounts = {};

// Community data (loaded from server)
const communityData = {};
let communityTotal = 0;
let communityLoaded = false;

// Batch buffer for sending samples to server
const pendingUploads = {};  // { classId: [features, ...] }
let uploadTimer = null;
const BATCH_SIZE = 10;      // Send after collecting this many
const BATCH_INTERVAL = 3000; // Or send after this many ms

// FPS tracking
let prevTime = performance.now();
let fpsDisplay = 0;

// DOM elements
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("canvas");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const fpsEl = document.getElementById("fps");
const flashEl = document.getElementById("flash");
const ctx = canvasEl.getContext("2d");

// ── MediaPipe Hands setup ──────────────────────────────────────────
const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5,
});

hands.onResults(onResults);

let camera = null;

// ── Feature extraction (mirrors Python utils.py) ───────────────────

function angleBetween(a, b, c) {
  const ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const magBA = Math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2);
  const magBC = Math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2);
  let cosine = dot / (magBA * magBC + 1e-8);
  cosine = Math.max(-1, Math.min(1, cosine));
  return Math.acos(cosine);
}

function extractFeatures(landmarks) {
  // 1. Raw coords
  const coords = landmarks.map((lm) => [lm.x, lm.y, lm.z]);

  // 2. Subtract wrist
  const wrist = [...coords[0]];
  for (let i = 0; i < 21; i++) {
    coords[i][0] -= wrist[0];
    coords[i][1] -= wrist[1];
    coords[i][2] -= wrist[2];
  }

  // 3. Scale to [-1, 1]
  let maxAbs = 0;
  for (const c of coords) {
    for (const v of c) {
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }
  if (maxAbs > 0) {
    for (const c of coords) {
      c[0] /= maxAbs;
      c[1] /= maxAbs;
      c[2] /= maxAbs;
    }
  }

  // 4. Flatten
  const flat = coords.flat(); // 63 values

  // 5. Fingertip-to-wrist distances
  const distances = FINGERTIP_IDS.map((tip) =>
    Math.sqrt(coords[tip][0] ** 2 + coords[tip][1] ** 2 + coords[tip][2] ** 2)
  );

  // 6. Joint angles
  const angles = FINGER_JOINTS.map(([base, mid, tip]) =>
    angleBetween(coords[base], coords[mid], coords[tip])
  );

  return [...flat, ...distances, ...angles]; // 73 values
}

// ── KNN Classifier ─────────────────────────────────────────────────

const K = 5; // number of neighbours

function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

function knnPredict(features) {
  const allSamples = [];

  for (const [classId, samples] of Object.entries(trainingData)) {
    for (const sample of samples) {
      allSamples.push({ classId: parseInt(classId), features: sample });
    }
  }

  if (allSamples.length === 0) return { label: null, confidence: 0 };

  // Calculate distances
  const distances = allSamples.map((s) => ({
    classId: s.classId,
    dist: euclideanDist(features, s.features),
  }));

  // Sort by distance
  distances.sort((a, b) => a.dist - b.dist);

  // Take K nearest
  const k = Math.min(K, distances.length);
  const nearest = distances.slice(0, k);

  // Vote
  const votes = {};
  for (const n of nearest) {
    votes[n.classId] = (votes[n.classId] || 0) + 1;
  }

  // Find winner
  let bestClass = null;
  let bestVotes = 0;
  for (const [cls, count] of Object.entries(votes)) {
    if (count > bestVotes) {
      bestVotes = count;
      bestClass = parseInt(cls);
    }
  }

  return { label: bestClass, confidence: bestVotes / k };
}

// ── MediaPipe results callback ─────────────────────────────────────

function onResults(results) {
  // FPS
  const now = performance.now();
  fpsDisplay = 1000 / (now - prevTime + 0.001);
  prevTime = now;
  fpsEl.textContent = `FPS: ${fpsDisplay.toFixed(0)}`;

  // Draw
  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // Mirror canvas to match mirrored video
  ctx.translate(canvasEl.width, 0);
  ctx.scale(-1, 1);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Draw connections
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
      color: "rgba(99, 102, 241, 0.6)",
      lineWidth: 3,
    });
    // Draw landmarks
    drawLandmarks(ctx, landmarks, {
      color: "#22c55e",
      lineWidth: 1,
      radius: 4,
    });

    const features = extractFeatures(landmarks);

    // ── Collect mode: save sample ──────────────────────────────
    if (currentMode === "collect" && activeClass !== null) {
      if (!trainingData[activeClass]) trainingData[activeClass] = [];
      trainingData[activeClass].push(features);
      sampleCounts[activeClass] = trainingData[activeClass].length;
      updateSampleCountsUI();
      updateGestureButtons();

      // Queue for server upload
      queueSampleForUpload(activeClass, features);

      // Flash effect
      flashEl.classList.add("active");
      setTimeout(() => flashEl.classList.remove("active"), 80);
    }

    // ── Predict mode: classify ─────────────────────────────────
    if (currentMode === "predict" && modelTrained) {
      const result = knnPredict(features);
      if (result.label !== null) {
        const name = GESTURE_NAMES[result.label] || `Class ${result.label}`;
        const emoji = GESTURE_EMOJIS[result.label] || "";
        const conf = (result.confidence * 100).toFixed(0);
        predictionEl.textContent = `${emoji} ${name}`;
        confidenceEl.textContent = `Confidence: ${conf}%`;
      }
    }
  } else {
    if (currentMode === "predict") {
      predictionEl.textContent = "No hand detected";
      confidenceEl.textContent = "";
    }
  }

  ctx.restore();
}

// ── Camera control ─────────────────────────────────────────────────

async function toggleCamera() {
  const btn = document.getElementById("btn-camera");

  if (cameraRunning) {
    // Stop
    if (camera) {
      camera.stop();
      camera = null;
    }
    cameraRunning = false;
    btn.innerHTML = '<span>▶</span> Start Camera';
    predictionEl.textContent = "Camera stopped";
    confidenceEl.textContent = "";
    fpsEl.textContent = "";
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    return;
  }

  // Start
  btn.innerHTML = '<span>⏸</span> Stop Camera';
  predictionEl.textContent = "Starting camera...";

  try {
    const isMobile = window.innerWidth <= 768;
    camera = new Camera(videoEl, {
      onFrame: async () => {
        await hands.send({ image: videoEl });
      },
      width: isMobile ? 640 : 1280,
      height: isMobile ? 480 : 960,
    });
    await camera.start();
    cameraRunning = true;
    predictionEl.textContent = currentMode === "collect"
      ? "Select a gesture class below"
      : (modelTrained ? "Show your hand!" : "Train a model first");
  } catch (err) {
    predictionEl.textContent = "Camera access denied";
    btn.innerHTML = '<span>▶</span> Start Camera';
    console.error(err);
  }
}

// ── Mode switching ─────────────────────────────────────────────────

function switchMode(mode) {
  currentMode = mode;
  activeClass = null;

  document.getElementById("btn-collect").classList.toggle("active", mode === "collect");
  document.getElementById("btn-predict").classList.toggle("active", mode === "predict");
  document.getElementById("collect-panel").classList.toggle("hidden", mode !== "collect");
  document.getElementById("predict-panel").classList.toggle("hidden", mode !== "predict");

  // Update gesture button states
  updateGestureButtons();

  if (mode === "predict") {
    predictionEl.textContent = modelTrained ? "Show your hand!" : "Train a model first";
    confidenceEl.textContent = "";
  } else {
    predictionEl.textContent = "Select a gesture class below";
    confidenceEl.textContent = "";
  }
}

// ── Gesture button handling ────────────────────────────────────────

function initGestureButtons() {
  const container = document.getElementById("gesture-buttons");
  container.innerHTML = "";

  for (let i = 0; i <= 9; i++) {
    const btn = document.createElement("button");
    btn.className = "gesture-btn";
    btn.id = `gesture-btn-${i}`;
    btn.onclick = () => toggleGestureClass(i);

    const label = document.createElement("span");
    label.innerHTML = `<strong>[${i}]</strong> ${GESTURE_EMOJIS[i]} ${GESTURE_NAMES[i]}`;

    const count = document.createElement("span");
    count.className = "count";
    count.id = `count-${i}`;
    count.textContent = sampleCounts[i] || 0;

    btn.appendChild(label);
    btn.appendChild(count);
    container.appendChild(btn);
  }
}

function toggleGestureClass(classId) {
  if (activeClass === classId) {
    activeClass = null;
  } else {
    activeClass = classId;
  }
  updateGestureButtons();

  if (activeClass !== null) {
    predictionEl.innerHTML = `Recording: <strong>[${classId}]</strong> ${GESTURE_EMOJIS[classId]} ${GESTURE_NAMES[classId]}`;
    confidenceEl.textContent = "Show your hand & keep pressing the key to collect samples";
  } else {
    predictionEl.textContent = "Select a gesture class below";
    confidenceEl.textContent = "";
  }
}

function updateGestureButtons() {
  for (let i = 0; i <= 9; i++) {
    const btn = document.getElementById(`gesture-btn-${i}`);
    if (btn) {
      btn.classList.toggle("active", activeClass === i);
      const countEl = document.getElementById(`count-${i}`);
      if (countEl) countEl.textContent = sampleCounts[i] || 0;
    }
  }
}

// ── Sample counts display ──────────────────────────────────────────

function updateSampleCountsUI() {
  const container = document.getElementById("sample-counts");
  const total = Object.values(sampleCounts).reduce((a, b) => a + b, 0);
  const classes = Object.keys(sampleCounts).length;

  let html = `<span class="count-label">Your samples:</span> ${total} across ${classes} classes`;
  if (communityLoaded) {
    html += `<br><span class="count-label">🌐 Community:</span> ${communityTotal} samples shared by all users`;
  }
  container.innerHTML = html;
}

// ── Training ───────────────────────────────────────────────────────

function trainModel() {
  const classCount = Object.keys(trainingData).length;
  const totalSamples = Object.values(trainingData).reduce(
    (sum, arr) => sum + arr.length, 0
  );

  if (classCount < 2) {
    alert("Need at least 2 gesture classes to train. Collect more data!");
    return;
  }

  if (totalSamples < 10) {
    alert("Need more samples. Collect at least 5 per class.");
    return;
  }

  // The KNN doesn't need explicit training — it's instance-based.
  // We just mark it as "trained" so prediction mode works.
  modelTrained = true;

  // Update UI
  const statusEl = document.getElementById("model-status");
  statusEl.innerHTML = `
    <span class="status-dot green"></span>
    Model ready — ${totalSamples} samples, ${classCount} classes
  `;

  document.getElementById("btn-train").textContent = "✅ Model Trained!";
  setTimeout(() => {
    document.getElementById("btn-train").textContent = "🧠 Train Model";
  }, 2000);

  // Auto-switch to predict
  switchMode("predict");
}

// ── Clear data ─────────────────────────────────────────────────────

function clearData() {
  if (!confirm("Clear all collected data?")) return;

  for (const key in trainingData) delete trainingData[key];
  for (const key in sampleCounts) delete sampleCounts[key];
  modelTrained = false;
  activeClass = null;

  updateGestureButtons();
  updateSampleCountsUI();

  const statusEl = document.getElementById("model-status");
  statusEl.innerHTML = `<span class="status-dot red"></span> No model trained yet`;

  predictionEl.textContent = "Data cleared";
  confidenceEl.textContent = "";
}

// ── Keyboard shortcuts (same as Python version) ────────────────────

document.addEventListener("keydown", (e) => {
  const key = parseInt(e.key);
  if (!isNaN(key) && key >= 0 && key <= 9 && currentMode === "collect") {
    toggleGestureClass(key);
  }
  if (e.key === "q" || e.key === "Escape") {
    if (cameraRunning) toggleCamera();
  }
});

// ── Community data: server communication ────────────────────────────

/**
 * Queue a sample for batched upload to the server.
 */
function queueSampleForUpload(classId, features) {
  if (!pendingUploads[classId]) pendingUploads[classId] = [];
  pendingUploads[classId].push(features);

  // Send immediately if batch is full
  if (pendingUploads[classId].length >= BATCH_SIZE) {
    flushUploads(classId);
  } else {
    // Otherwise, set a timer to send soon
    clearTimeout(uploadTimer);
    uploadTimer = setTimeout(() => flushAllUploads(), BATCH_INTERVAL);
  }
}

/**
 * Flush pending uploads for a specific class to the server.
 */
async function flushUploads(classId) {
  const batch = pendingUploads[classId];
  if (!batch || batch.length === 0) return;

  // Clear the buffer immediately
  pendingUploads[classId] = [];

  try {
    await fetch("/api/samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        classId: parseInt(classId),
        gestureName: GESTURE_NAMES[classId] || `Class ${classId}`,
        features: batch,
      }),
    });
  } catch (err) {
    // Silently fail — offline usage should still work
    console.warn("Could not upload samples to server:", err.message);
  }
}

/**
 * Flush all pending uploads across all classes.
 */
function flushAllUploads() {
  for (const classId in pendingUploads) {
    flushUploads(classId);
  }
}

/**
 * Load community-collected data from the server and merge into trainingData.
 */
async function loadCommunityData() {
  try {
    const response = await fetch("/api/samples");
    if (!response.ok) return;

    const result = await response.json();
    const data = result.data || {};
    communityTotal = result.total || 0;

    // Merge community data into training data
    for (const [classId, samples] of Object.entries(data)) {
      const cid = parseInt(classId);
      if (!trainingData[cid]) trainingData[cid] = [];
      if (!communityData[cid]) communityData[cid] = [];

      communityData[cid] = samples;
      trainingData[cid].push(...samples);
      sampleCounts[cid] = trainingData[cid].length;
    }

    communityLoaded = true;

    // Auto-mark as trained if we have enough community data
    const classCount = Object.keys(trainingData).length;
    const totalSamples = Object.values(trainingData).reduce(
      (sum, arr) => sum + arr.length, 0
    );
    if (classCount >= 2 && totalSamples >= 10) {
      modelTrained = true;
      const statusEl = document.getElementById("model-status");
      statusEl.innerHTML = `
        <span class="status-dot green"></span>
        Community model ready — ${totalSamples} samples, ${classCount} classes
      `;
    }

    updateGestureButtons();
    updateSampleCountsUI();
    console.log(`Loaded ${communityTotal} community samples`);
  } catch (err) {
    console.warn("Could not load community data:", err.message);
  }
}

// Flush remaining samples before user leaves
window.addEventListener("beforeunload", () => {
  flushAllUploads();
});

// ── Init ───────────────────────────────────────────────────────────

initGestureButtons();
updateSampleCountsUI();
loadCommunityData();
