# import cv2
# import time
# from ultralytics import YOLO
# from database import save_count
# from logger import log_event

# model = YOLO("yolov8n.pt")
# cap = cv2.VideoCapture(0)

# log_event("INFO", "People counter started")

# last_saved = 0  # for 5 sec saving control

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         log_event("ERROR", "Camera disconnected")
#         break

#     # Run YOLO
#     results = model(frame, classes=[0])

#     # ---- ZONE SETUP ----
#     h, w, _ = frame.shape
#     zone1 = w // 3
#     zone2 = 2 * w // 3

#     zone_a = 0
#     zone_b = 0
#     zone_c = 0

#     # ---- COUNT PER ZONE ----
#     if results[0].boxes is not None:
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cx = (x1 + x2) // 2

#             if cx < zone1:
#                 zone_a += 1
#             elif cx < zone2:
#                 zone_b += 1
#             else:
#                 zone_c += 1

#     total_count = zone_a + zone_b + zone_c

#     # ---- SAVE TO DB EVERY 5 SECONDS ----
#     if time.time() - last_saved >= 5:
#         save_count(
#             total_count,
#             zone_a,
#             zone_b,
#             zone_c
#         )
#         last_saved = time.time()

#     # ---- OPTIONAL: DRAW ZONE LINES ----
#     cv2.line(frame, (zone1, 0), (zone1, h), (0, 255, 0), 2)
#     cv2.line(frame, (zone2, 0), (zone2, h), (0, 255, 0), 2)

#     cv2.imshow("People Counter", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("🔥 DATABASE.PY LOADED 🔥")
# print("FILE PATH:", __file__)

# from database import init_db, save_count

# print("🚀 people_counter.py STARTED")

# # 1️⃣ Initialize the database (creates table if not exists)
# init_db()

# # 2️⃣ Save test data (replace with your YOLO counts later)
# save_count(total=10, zone_a=3, zone_b=2, zone_c=4, zone_d=1, source="Webcam")

# print("✅ people_counter.py finished")




# from database import save_count, init_db

# init_db()  # 🔥 MUST BE CALLED

# import sqlite3
# import os
# from datetime import datetime

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "people_count.db")

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS people_count (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT,
#             total INTEGER,
#             zone_a INTEGER,
#             zone_b INTEGER,
#             zone_c INTEGER
#         )
#     """)
#     conn.commit()
#     conn.close()
#     print("✅ Database initialized at:", DB_PATH)

# def save_count(total, zone_a, zone_b, zone_c):
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute("""
#         INSERT INTO people_count (timestamp, total, zone_a, zone_b, zone_c)
#         VALUES (?, ?, ?, ?, ?)
#     """, (
#         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         total, zone_a, zone_b, zone_c
#     ))
#     conn.commit()
#     conn.close()

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from database import init_db, save_count, save_log

print("🚀 people_counter.py STARTED")

# ---------- 1️⃣ Initialize Database ----------
init_db()

# ---------- 2️⃣ Initialize YOLOv8 ----------
model = YOLO("yolov8n.pt")  # YOLOv8 Nano for speed

# ---------- 3️⃣ Initialize DeepSORT ----------
tracker = DeepSort(max_age=30)

# ---------- 4️⃣ Video Source ----------
cap = cv2.VideoCapture(0)  # Webcam
if not cap.isOpened():
    save_log("ERROR", "Cannot open webcam")
    raise RuntimeError("Cannot open webcam")

# ---------- 5️⃣ Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        save_log("ERROR", "Failed to read frame from camera")
        break

    height, width, _ = frame.shape

    # ---------- Define zones dynamically ----------
    zone_coords = {
        "A": (0, 0, width // 2, height // 2),
        "B": (width // 2, 0, width, height // 2),
        "C": (0, height // 2, width // 2, height),
        "D": (width // 2, height // 2, width, height)
    }

    # ---------- YOLO Detection ----------
    results = model.predict(frame, classes=[0], stream=False)

    detections = []

for r in results:
    # Convert tensor to list of lists
    boxes = r.boxes.xyxy.tolist()  # [[x1, y1, x2, y2], ...]
    
    for box in boxes:
        if len(box) != 4:
            continue  # skip invalid boxes
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        detections.append([x1, y1, w, h])

    # ---------- DeepSORT Tracking ----------
    tracks = tracker.update_tracks(detections, frame=frame)

    # ---------- Real Zone Counting ----------
    zone_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_count = 0

    for t in tracks:
        if not t.is_confirmed():
            continue

        x1, y1, x2, y2 = t.to_ltrb()
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Check which zone the person is in
        for zone, (zx1, zy1, zx2, zy2) in zone_coords.items():
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                zone_counts[zone] += 1
                break

        total_count += 1

    # ---------- Save count and log ----------
    save_count(
        total=total_count,
        zone_a=zone_counts["A"],
        zone_b=zone_counts["B"],
        zone_c=zone_counts["C"],
        zone_d=zone_counts["D"],
        source="Webcam"
    )

    # ---------- Display ----------
    for zone, (zx1, zy1, zx2, zy2) in zone_coords.items():
        color = (0, 255, 0)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
        cv2.putText(frame, f"{zone}: {zone_counts[zone]}", (zx1 + 10, zy1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Total: {total_count}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_log("INFO", "Webcam counting stopped by user")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ people_counter.py finished")
