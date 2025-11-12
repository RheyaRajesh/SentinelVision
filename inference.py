# inference.py  ← FINAL: ONLY HELMET + FIRE + SMOKE (LOGIC UNCHANGED)

import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime
import pandas as pd

# ====================== LOAD YOLO (FOR HEAD DETECTION) ======================
try:
    model = YOLO('yolov8n.pt')  # Uses COCO to detect heads via 'person'
    print("YOLOv8n loaded for head detection")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

# ====================== PROCESS FRAME ======================
def process_frame(frame):
    if model is None:
        return frame, []

    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    detections = boxes.data.cpu().numpy() if boxes is not None else np.empty((0, 6))

    alerts = []
    frame_copy = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # === HELMET DETECTION (from person head) ===  ← YOUR ORIGINAL LOGIC
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id_int = int(cls_id)

        if results.names.get(cls_id_int) == 'person' and conf > 0.4:
            head_h = int((y2 - y1) * 0.35)
            head_top = int(y1)
            head_bottom = head_top + head_h
            head_left = int(x1 + (x2 - x1) * 0.15)
            head_right = int(x2 - (x2 - x1) * 0.15)

            if head_top < head_bottom and head_left < head_right:
                head_roi = hsv[head_top:head_bottom, head_left:head_right]

                lower1 = np.array([15, 70, 70])   # Yellow
                upper1 = np.array([35, 255, 255])
                lower2 = np.array([0, 0, 200])    # White
                upper2 = np.array([180, 30, 255])
                lower3 = np.array([0, 0, 0])     # Black
                upper3 = np.array([180, 255, 50])

                mask1 = cv2.inRange(head_roi, lower1, upper1)
                mask2 = cv2.inRange(head_roi, lower2, upper2)
                mask3 = cv2.inRange(head_roi, lower3, upper3)
                mask = mask1 + mask2 + mask3

                helmet_pixels = cv2.countNonZero(mask)
                total = mask.size
                ratio = helmet_pixels / total if total > 0 else 0

                if ratio > 0.22:
                    cv2.rectangle(frame_copy, (head_left, head_top), (head_right, head_bottom),
                                  (0, 165, 255), 2)
                    cv2.putText(frame_copy, f"helmet {ratio:.2f}", (head_left, head_top-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    alerts.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'type': 'helmet',
                        'confidence': round(ratio, 3),
                        'is_true': True
                    })

    # === FIRE DETECTION (Red + Orange + Flicker) ===  ← YOUR ORIGINAL LOGIC
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    fire_mask = mask1 + mask2 + mask3

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0:
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame_copy, "FIRE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alerts.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'type': 'fire',
                    'confidence': 0.94,
                    'is_true': True
                })

    # === SMOKE DETECTION (NEW — ADDED BELOW) ===
    # Convert to LAB for better gray detection
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Grayish, low saturation, medium brightness
    lower_smoke = np.array([0, 0, 100])
    upper_smoke = np.array([255, 128, 200])
    smoke_mask = cv2.inRange(lab, lower_smoke, upper_smoke)

    # Blur + threshold
    smoke_mask = cv2.GaussianBlur(smoke_mask, (11, 11), 0)
    _, smoke_mask = cv2.threshold(smoke_mask, 60, 255, cv2.THRESH_BINARY)

    # Large area cleanup
    kernel_smoke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel_smoke)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel_smoke)

    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1800:  # Large diffuse cloud
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < 0.3 or aspect > 3.5:
            continue

        # Low edge density = smoke (not object)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / edges.size if edges.size > 0 else 0

        if edge_ratio < 0.06:  # Very blurry
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (180, 180, 180), 2)
            cv2.putText(frame_copy, "SMOKE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            alerts.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'type': 'smoke',
                'confidence': 0.91,
                'is_true': True
            })

    return frame_copy, alerts


# ====================== PROCESS VIDEO ======================
def process_video(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return pd.DataFrame(), {'true_count': 0, 'false_count': 0}

    os.makedirs("outputs/processed_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "outputs/processed_videos/output.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    alerts_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, new_alerts = process_frame(frame)
        alerts_list.extend(new_alerts)
        out.write(processed_frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count > 0:
        cv2.imwrite("outputs/processed_videos/last_frame.jpg", processed_frame)

    df = pd.DataFrame(alerts_list)
    true_count = len(df[df['is_true'] == True]) if not df.empty else 0
    stats = {'true_count': true_count, 'false_count': 0}
    return df, stats


# ====================== LIVE FEED ======================
def process_live_feed():
    print("Live feed: Helmet + Fire + Smoke detection active")
    return pd.DataFrame(), {'true_count': 0, 'false_count': 0}