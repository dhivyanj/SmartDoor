# doorlock_single.py
# One-file desktop door lock: Face (LBPH) + PIN + SQLite logs

import argparse
import getpass
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import bcrypt
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# ---- Paths ----
BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
TRAINER_PATH = BASE / "trainer.yml"
LABELS_JSON = BASE / "labels.json"
DB_PATH = BASE / "access.db"

CONFIDENCE_THRESHOLD = 50.0  # lower = stricter

# ---- DB ----
def db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                pin_hash BLOB NOT NULL,
                label INTEGER UNIQUE
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ts TEXT NOT NULL,
                method TEXT NOT NULL,
                success INTEGER NOT NULL,
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        conn.commit()

def add_user(name: str, pin_hash: bytes, label: int):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (name, pin_hash, label) VALUES (?, ?, ?);",
                  (name, pin_hash, label))
        conn.commit()
        return c.lastrowid

def next_label():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT MAX(label) FROM users;")
        row = c.fetchone()
        return 0 if row is None or row[0] is None else (row[0] + 1)

def get_user_by_label(label: int):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, pin_hash, label FROM users WHERE label = ?;", (label,))
        return c.fetchone()

def log_event(user_id: int, method: str, success: bool, confidence: float | None):
    with db_conn() as conn:
        c = conn.cursor()
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        c.execute("INSERT INTO logs (user_id, ts, method, success, confidence) VALUES (?, ?, ?, ?, ?);",
                  (user_id, ts, method, 1 if success else 0, confidence))
        conn.commit()

# ---- PIN utils ----
def hash_pin(pin: str) -> bytes:
    if not pin or not pin.isdigit():
        raise ValueError("PIN must be numeric")
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt())

def verify_pin(pin: str, pin_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(pin.encode("utf-8"), pin_hash)
    except Exception:
        return False

# ---- Face rec ----
def ensure_dirs():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(cascade_path)
    if det.empty():
        raise RuntimeError("Failed to load Haar cascade")
    return det

def get_camera(index: int = 0):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        raise RuntimeError("Could not open camera")
    return cam

def load_labels():
    if LABELS_JSON.exists():
        return json.loads(LABELS_JSON.read_text())
    return {}

def save_labels(mapping: dict):
    LABELS_JSON.write_text(json.dumps(mapping, indent=2))

class LBPH:
    def __init__(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face not found. Install opencv-contrib-python.")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = get_face_detector()

    def train(self):
        ensure_dirs()
        faces, labels = [], []
        for label_dir in DATASET_DIR.iterdir():
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                continue
            for img_path in label_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(label)
        if not faces:
            raise RuntimeError("No training data found. Enroll a user first.")
        self.recognizer.train(faces, np.array(labels, dtype=np.int32))
        self.recognizer.save(str(TRAINER_PATH))

    def load(self):
        if not TRAINER_PATH.exists():
            raise RuntimeError("No trained model found. Run enrollment first.")
        self.recognizer.read(str(TRAINER_PATH))

    def detect_faces(self, gray):
        return self.detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    def predict_on_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        if len(faces) == 0:
            return None, None, None
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = self.recognizer.predict(roi)
        return label, confidence, (x, y, w, h)

def capture_samples_for_label(label: int, samples: int = 25, cam_index: int = 0):
    ensure_dirs()
    out_dir = DATASET_DIR / str(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = get_face_detector()
    cam = get_camera(cam_index)
    saved = 0
    try:
        while saved < samples:
            ret, frame = cam.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.imwrite(str(out_dir / f"{label}_{saved+1:03d}.jpg"), roi)
                saved += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{saved}/{samples}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break
            cv2.imshow("Capture - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return saved

# ---- GUI App ----
class DoorLockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Door Lock - Face + PIN")
        self.root.geometry("900x600")

        self.rec = LBPH()
        self.rec.load()
        self.labels_map = load_labels()
        self.current_label = None
        self.current_confidence = None
        self.cam = get_camera(0)
        self.stop_event = threading.Event()

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_var = tk.StringVar(value="Waiting for face...")
        ttk.Label(right, textvariable=self.status_var, font=("Segoe UI", 14)).pack(pady=10)

        self.pin_var = tk.StringVar(value="")
        ttk.Entry(right, textvariable=self.pin_var, font=("Segoe UI", 18),
                  show="•", width=12, justify="center").pack(pady=5)

        pad = ttk.Frame(right); pad.pack(pady=10)
        for (t, r, c) in [('1',1,0),('2',1,1),('3',1,2),
                          ('4',2,0),('5',2,1),('6',2,2),
                          ('7',3,0),('8',3,1),('9',3,2),
                          ('←',4,0),('0',4,1),('C',4,2)]:
            ttk.Button(pad, text=t, command=lambda t=t:self.key(t),
                       width=5).grid(row=r, column=c, padx=6, pady=6)

        self.unlock_btn = ttk.Button(right, text="Unlock", command=self.unlock)
        self.unlock_btn.pack(pady=10)
        self.unlock_btn.state(["disabled"])
 
        self.msg = ttk.Label(right, text="", font=("Segoe UI", 16))
        self.msg.pack(pady=10)

        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def key(self, k):
        if k == 'C': self.pin_var.set("")
        elif k == '←': self.pin_var.set(self.pin_var.get()[:-1])
        else: self.pin_var.set(self.pin_var.get()+k)

    def camera_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cam.read()
            if not ret: continue
            label, conf, rect = self.rec.predict_on_frame(frame)
            if rect is not None and conf is not None and conf < CONFIDENCE_THRESHOLD:
                self.unlock_btn.state(["!disabled"])
            else:
                self.unlock_btn.state(["disabled"])

            disp = frame.copy()
            if rect:
                x,y,w,h = rect
                color = (0,255,0) if conf < CONFIDENCE_THRESHOLD else (0,0,255)
                cv2.rectangle(disp, (x,y), (x+w,y+h), color, 2)
                self.current_label, self.current_confidence = label, conf
                self.status_var.set(f"Label={label}, conf={conf:.1f}")
            else:
                self.current_label, self.current_confidence = None, None
                self.status_var.set("Waiting for face...")
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb).resize((640,480)))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def unlock(self):
        if self.current_label is None or self.current_confidence is None:
            messagebox.showwarning("Warning", "No face detected.")
            return
        if self.current_confidence >= CONFIDENCE_THRESHOLD:
            messagebox.showwarning("Warning", "Recognition not confident enough.")
            return
        uid = self.labels_map.get(str(self.current_label))
        if not uid:
            messagebox.showerror("Error", "Unknown label. Re-enroll user.")
            return
        row = get_user_by_label(self.current_label)
        if not row:
            messagebox.showerror("Error", "User not found in DB.")
            return
        user_id, name, pin_hash, _ = row
        ok = verify_pin(self.pin_var.get(), pin_hash)
        if ok:
            self.msg.configure(text=f"UNLOCKED ✅ Welcome, {name}")
            log_event(user_id, "face+pin", True, float(self.current_confidence))
        else:
            self.msg.configure(text="DENIED ❌ Wrong PIN")
            log_event(user_id, "face+pin", False, float(self.current_confidence))
        self.pin_var.set("")

    def close(self):
        self.stop_event.set()
        try:
            self.cam.release()
        except Exception:
            pass
        self.root.destroy()

# ---- CLI ----
def cli():
    p = argparse.ArgumentParser(description="Door Lock (single file)")
    p.add_argument("--init-db", action="store_true", help="Initialize DB and exit")
    p.add_argument("--enroll", action="store_true", help="Enroll a user")
    p.add_argument("--name", type=str, help="Name for enrollment")
    p.add_argument("--samples", type=int, default=25, help="Face samples to capture")
    p.add_argument("--run", action="store_true", help="Run the GUI app")
    args = p.parse_args()

    init_db()

    if args.init_db and not (args.enroll or args.run):
        print("Database initialized.")
        return

    if args.enroll:
        if not args.name:
            print("Provide --name for enrollment.")
            return
        pin1 = getpass.getpass("Enter numeric PIN: ")
        pin2 = getpass.getpass("Confirm PIN: ")
        if pin1 != pin2:
            print("PIN mismatch.")
            return
        try:
            pin_hash = hash_pin(pin1)
        except ValueError as e:
            print(f"Error: {e}")
            return
        label = next_label()
        user_id = add_user(args.name, pin_hash, label)
        print(f"Added user '{args.name}' with label {label} (user_id={user_id})")
        saved = capture_samples_for_label(label, samples=args.samples, cam_index=0)
        print(f"Captured {saved} samples.")
        rec = LBPH()
        rec.train()
        print("Trained LBPH model.")
        mapping = load_labels()
        mapping[str(label)] = user_id
        save_labels(mapping)
        print("Updated labels.json")

    if args.run or (not args.init_db and not args.enroll):
        root = tk.Tk()
        DoorLockApp(root)
        root.mainloop()

if __name__ == "__main__":
    cli()
