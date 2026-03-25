import cv2
import numpy as np
import tensorflow as tf
import threading
import sys
from collections import deque
from config import *

# ── Optional Windows TTS (graceful fallback if not on Windows) ────────────────
try:
    import pyttsx3
    import pythoncom
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("pyttsx3 / pythoncom not found — speech disabled.")


def speak_text(text):
    if not SPEECH_AVAILABLE:
        return
    # pythoncom.CoInitialize() is Windows-only; guard it
    if sys.platform == 'win32':
        pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        del engine
    except Exception as e:
        print(f"Speech error: {e}")


# ── Skin-based hand extractor (must match collect_data.py exactly) ────────────
def extract_hand(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,  20,  70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3000:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 20
    x, y = max(0, x - pad), max(0, y - pad)
    w    = min(frame.shape[1] - x, w + 2 * pad)
    h    = min(frame.shape[0] - y, h + 2 * pad)

    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half   = side // 2
    x1 = max(0, cx - half);  y1 = max(0, cy - half)
    x2 = min(frame.shape[1], cx + half)
    y2 = min(frame.shape[0], cy + half)

    crop    = frame[y1:y2, x1:x2]
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Fix: cv2.resize expects (W, H); IMG_SIZE is (H, W)
    resized = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]))
    return resized, (x1, y1, x2, y2)


# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('scratch_sign_model.h5')
print("Model loaded.")

# ── Inference state ───────────────────────────────────────────────────────────
sequence           = deque(maxlen=SEQUENCE_LENGTH)
THRESHOLD          = 0.85
last_prediction    = ""
frames_since_speech = 0

# Smoothing: keep last N predictions and only confirm if majority agrees
SMOOTH_WINDOW = 5
recent_preds  = deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(0)
print("Starting webcam... Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ── Preprocess exactly as training ────────────────────────────────────────
    processed, bbox = extract_hand(frame)

    if processed is not None:
        normalized = processed.astype(np.float32) / 255.0
        sequence.append(np.expand_dims(normalized, -1))  # (H, W, 1)
    else:
        # No hand detected — append blank frame to keep sequence flowing
        sequence.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32))

    # Draw bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ── Predict ───────────────────────────────────────────────────────────────
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(list(sequence), axis=0)   # (1, T, H, W, 1)
        res        = model.predict(input_data, verbose=0)[0]
        confidence = float(np.max(res))

        if confidence > THRESHOLD:
            current_pred = ACTIONS[np.argmax(res)]
            recent_preds.append(current_pred)

            # Only fire if the majority of recent predictions agree
            if recent_preds.count(current_pred) >= (SMOOTH_WINDOW // 2 + 1):
                cv2.putText(frame, f"{current_pred}  ({confidence:.2f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Speak only on new prediction after cooldown
                if (current_pred != last_prediction
                        and frames_since_speech > SEQUENCE_LENGTH * 2):
                    print(f"Recognized: {current_pred}")
                    threading.Thread(target=speak_text,
                                     args=(current_pred,),
                                     daemon=True).start()
                    last_prediction     = current_pred
                    frames_since_speech = 0
        else:
            recent_preds.clear()   # Reset smoothing on low confidence
            cv2.putText(frame, "...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frames_since_speech += 1

    # HUD
    cv2.putText(frame, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Real-Time Sign Translation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()