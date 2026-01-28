import cv2
import sqlite3
from datetime import datetime
from flask import render_template
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Database
conn = sqlite3.connect("analytics.db", check_same_thread=False)
cur = conn.cursor()



def index():
    conn = sqlite3.connect("people_count.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT timestamp, source,
               total_count, zone_a, zone_b, zone_c
        FROM people_count
        ORDER BY id DESC
        LIMIT 50
    """)

    data = cur.fetchall()
    conn.close()

    return render_template("table.html", data=data)





cur.execute("""
            
            
            
            
            
            
            
CREATE TABLE IF NOT EXISTS video_analytics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    people_count INTEGER,
    zone_a INTEGER,
    zone_b INTEGER,
    zone_c INTEGER
)
""")
conn.commit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    people_count = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person
                people_count += 1

    # Dummy zone values (replace later)
    zone_a = people_count // 3
    zone_b = people_count // 3
    zone_c = people_count - zone_a - zone_b

    # Insert into DB every few seconds
    cur.execute("""
        INSERT INTO video_analytics
        (timestamp, source, people_count, zone_a, zone_b, zone_c)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "Webcam", people_count, zone_a, zone_b, zone_c))

    conn.commit()

    cv2.imshow("Crowd Analytics", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
