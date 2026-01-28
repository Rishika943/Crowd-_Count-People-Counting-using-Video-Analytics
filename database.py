

import sqlite3
import os
from datetime import datetime

print("📦 database.py LOADED")

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.path.join(INSTANCE_DIR, "people_count.db")

# ---------- INIT DATABASE ----------
def init_db():
    """Create people_count and logs tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # People count table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS people_count (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source TEXT,
        total_count INTEGER,
        zone_a INTEGER,
        zone_b INTEGER,
        zone_c INTEGER,
        zone_d INTEGER
    )
    """)

    # Logs table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        level TEXT,
        message TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized with tables: people_count + logs")

# ---------- SAVE PEOPLE COUNT ----------
def save_count(total, zone_a, zone_b, zone_c, source="Webcam", zone_d=0):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO people_count
    (timestamp, source, total_count, zone_a, zone_b, zone_c, zone_d)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source, total, zone_a, zone_b, zone_c, zone_d
    ))

    conn.commit()
    conn.close()
    print(f"💾 People count saved: total={total}, zones=({zone_a},{zone_b},{zone_c},{zone_d})")

    # Also add a log automatically
    save_log("INFO", f"People count saved: total={total}, zones=({zone_a},{zone_b},{zone_c},{zone_d})")

# ---------- SAVE LOG ----------
def save_log(level, message):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO logs (level, message)
    VALUES (?, ?)
    """, (level, message))
    conn.commit()
    conn.close()
    print(f"📝 Log saved: [{level}] {message}")

# ---------- TEST RUN ----------
if __name__ == "__main__":
    init_db()
    save_count(total=10, zone_a=3, zone_b=2, zone_c=4, zone_d=1, source="Webcam")
    save_count(total=5, zone_a=1, zone_b=1, zone_c=2, zone_d=1, source="Camera2")

