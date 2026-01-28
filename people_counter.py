
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

