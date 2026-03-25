import cv2
import numpy as np
import os
from config import *

# ── Skin-based hand extractor ──────────────────────────────────────────────────
def extract_hand(frame):
    """Crop and isolate hand using skin-colour HSV thresholding."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,  20,  70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3000:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 20
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(frame.shape[1] - x, w + 2 * pad)
    h = min(frame.shape[0] - y, h + 2 * pad)

    # Square crop for consistent CNN input
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half   = side // 2
    x1 = max(0, cx - half);  y1 = max(0, cy - half)
    x2 = min(frame.shape[1], cx + half)
    y2 = min(frame.shape[0], cy + half)

    crop  = frame[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Fix: IMG_SIZE is (H, W) but cv2.resize expects (W, H)
    resized = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]))
    return resized, (x1, y1, x2, y2)


# ── Create directories up front ────────────────────────────────────────────────
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)
print("Starting Data Collection...")
print("Controls: SPACE = start sequence | Q = skip to next action | ESC = quit\n")

for action in ACTIONS:
    # Count already-collected sequences so you can safely resume
    existing = sum(
        1 for s in range(NO_SEQUENCES)
        if os.path.exists(os.path.join(DATA_PATH, action, str(s), f"{SEQUENCE_LENGTH-1}.jpg"))
    )
    needed = NO_SEQUENCES - existing
    if needed <= 0:
        print(f"[SKIP] {action} — already complete ({existing}/{NO_SEQUENCES})")
        continue

    print(f"\n{'='*45}")
    print(f"  Action : {action}  ({existing} done, {needed} to collect)")
    print(f"  Tips   : vary hand speed & angle slightly each time")
    print(f"{'='*45}")

    seq_num = existing
    while seq_num < NO_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            break

        _, bbox = extract_hand(frame)
        display = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display,
                    f"{action}  [{seq_num}/{NO_SEQUENCES}]  SPACE=start  Q=skip action  ESC=quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imshow('Data Collection', display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                   # ESC — quit everything
            cap.release()
            cv2.destroyAllWindows()
            print("Collection aborted.")
            exit()
        if key == ord('q'):             # skip to next action
            break
        if key != ord(' '):             # wait for SPACE to start
            continue

        # ── Countdown ─────────────────────────────────────────────────────────
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f"Starting in {countdown}...",
                        (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000)

        # ── Record one sequence ────────────────────────────────────────────────
        saved_ok = True
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                saved_ok = False
                break

            processed, bbox = extract_hand(frame)

            if processed is None:
                # Fallback: plain grayscale so the slot is never empty
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]))

            img_path = os.path.join(DATA_PATH, action, str(seq_num), f"{frame_num}.jpg")
            cv2.imwrite(img_path, processed)

            # Live feedback
            display = frame.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Recording {frame_num+1}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collection', display)
            cv2.waitKey(1)

        if saved_ok:
            print(f"  Saved sequence {seq_num}")
            seq_num += 1

cap.release()
cv2.destroyAllWindows()
print("\nCollection Complete.")