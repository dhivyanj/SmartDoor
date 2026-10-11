# doorlock_esp32.py  — v2.0
# Door Lock: Face (LBPH) + PIN + SQLite logs
# Supports: local webcam OR ESP32-CAM MJPEG stream, Kaggle dataset import, live enrollment
# Kaggle dataset: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
#
# pip install opencv-contrib-python pillow bcrypt numpy

import argparse, getpass, json, sqlite3, threading, urllib.request, shutil
from datetime import datetime
from pathlib import Path
import bcrypt, cv2, numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

BASE          = Path(__file__).parent
DATASET_DIR   = BASE / "dataset"
TRAINER_PATH  = BASE / "trainer.yml"
LABELS_JSON   = BASE / "labels.json"
DB_PATH       = BASE / "access.db"
CONFIDENCE_THRESHOLD = 45.0   # lower = stricter match

DARK_BG = "#1e1e2e"; PANEL_BG = "#2a2a3e"; ACCENT = "#7c3aed"
SUCCESS = "#22c55e"; DANGER = "#ef4444";    FG = "#e2e8f0"; FG2 = "#94a3b8"

# ─────────────── ESP32-CAM ───────────────────
class ESP32Camera:
    def __init__(self, url):
        self.url = url; self._buf = b""
        self.stream = urllib.request.urlopen(url, timeout=5)

    def read(self):
        try:
            while True:
                self._buf += self.stream.read(4096)
                a = self._buf.find(b"\xff\xd8"); b = self._buf.find(b"\xff\xd9")
                if a != -1 and b > a:
                    jpg = self._buf[a:b+2]; self._buf = self._buf[b+2:]
                    arr = np.frombuffer(jpg, np.uint8)
                    f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if f is not None: return True, f
        except: return False, None

    def isOpened(self): return self.stream is not None
    def release(self):
        try: self.stream.close()
        except: pass

def get_camera(source):
    if isinstance(source, str): return ESP32Camera(source)
    cam = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cam.isOpened(): raise RuntimeError("Cannot open webcam")
    return cam

# ─────────────── Database ────────────────────
def db_conn(): return sqlite3.connect(DB_PATH)

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE, pin_hash BLOB NOT NULL, label INTEGER UNIQUE)""")
        c.execute("""CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
            ts TEXT NOT NULL, method TEXT NOT NULL, success INTEGER NOT NULL, confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users(id))""")
        conn.commit()

def add_user(name, pin_hash, label):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (name,pin_hash,label) VALUES (?,?,?)",(name,pin_hash,label))
        conn.commit(); return c.lastrowid

def get_or_add_user(name, pin_hash, label):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE name=?",(name,)); row=c.fetchone()
        if row: return row[0]
        c.execute("INSERT INTO users (name,pin_hash,label) VALUES (?,?,?)",(name,pin_hash,label))
        conn.commit(); return c.lastrowid

def next_label():
    with db_conn() as conn:
        c=conn.cursor(); c.execute("SELECT MAX(label) FROM users"); row=c.fetchone()
        return 0 if (not row or row[0] is None) else row[0]+1

def get_user_by_label(label):
    with db_conn() as conn:
        c=conn.cursor()
        c.execute("SELECT id,name,pin_hash,label FROM users WHERE label=?",(label,))
        return c.fetchone()

def get_all_users():
    with db_conn() as conn:
        c=conn.cursor(); c.execute("SELECT id,name,label FROM users ORDER BY name")
        return c.fetchall()

def update_pin(uid, new_hash):
    with db_conn() as conn:
        c=conn.cursor(); c.execute("UPDATE users SET pin_hash=? WHERE id=?",(new_hash,uid))
        conn.commit()

def delete_user(uid):
    with db_conn() as conn:
        c=conn.cursor(); c.execute("SELECT label FROM users WHERE id=?",(uid,))
        row=c.fetchone(); label=row[0] if row else None
        c.execute("DELETE FROM users WHERE id=?",(uid,)); conn.commit()
    return label

def get_recent_logs(limit=100):
    with db_conn() as conn:
        c=conn.cursor()
        c.execute("""SELECT l.ts,u.name,l.method,l.success,l.confidence
                     FROM logs l LEFT JOIN users u ON l.user_id=u.id
                     ORDER BY l.id DESC LIMIT ?""",(limit,))
        return c.fetchall()

def log_event(uid, method, success, conf):
    with db_conn() as conn:
        ts = datetime.utcnow().isoformat(timespec="seconds")+"Z"
        conn.cursor().execute(
            "INSERT INTO logs (user_id,ts,method,success,confidence) VALUES (?,?,?,?,?)",
            (uid,ts,method,1 if success else 0,conf)); conn.commit()

# ─────────────── PIN ─────────────────────────
def hash_pin(pin):
    if not pin or not pin.isdigit(): raise ValueError("PIN must be numeric")
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt())

def verify_pin(pin, pin_hash):
    try: return bcrypt.checkpw(pin.encode(), pin_hash)
    except: return False

# ─────────────── Face utils ──────────────────
def ensure_dirs(): DATASET_DIR.mkdir(parents=True, exist_ok=True)

def get_face_detector():
    d = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    if d.empty(): raise RuntimeError("Haar cascade not found")
    return d

def load_labels(): return json.loads(LABELS_JSON.read_text()) if LABELS_JSON.exists() else {}
def save_labels(m): LABELS_JSON.write_text(json.dumps(m, indent=2))

# ─────────────── Kaggle import ───────────────
def import_kaggle_dataset(kaggle_root, default_pin="0000"):
    """
    Import the Vasuki Patel Face Recognition Dataset.
    Structure: <root>/<Person_Name>/*.jpg
    Each subfolder → one registered user with default_pin.
    """
    root = Path(kaggle_root)
    if not root.is_dir(): print(f"[ERROR] Not found: {root}"); return
    detector = get_face_detector(); ensure_dirs(); mapping = load_labels()
    try:    pin_hash = hash_pin(default_pin)
    except ValueError as e: print(f"[ERROR] {e}"); return

    person_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not person_dirs: print("[ERROR] No sub-folders found."); return
    print(f"Found {len(person_dirs)} person folder(s).")
    total = 0

    for pd in person_dirs:
        name = pd.name.replace("_"," ").strip()
        label = next_label(); out = DATASET_DIR/str(label); out.mkdir(parents=True,exist_ok=True)
        imgs = list(pd.glob("*.jpg"))+list(pd.glob("*.jpeg"))+list(pd.glob("*.png"))
        saved = 0
        for ip in imgs:
            img = cv2.imread(str(ip))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
            roi = cv2.resize(gray,(200,200)) if len(faces)==0 else \
                  cv2.resize(gray[faces[0][1]:faces[0][1]+faces[0][3],
                                  faces[0][0]:faces[0][0]+faces[0][2]],(200,200))
            cv2.imwrite(str(out/f"{label}_{saved+1:04d}.jpg"), roi); saved+=1
        if saved==0:
            print(f"  [SKIP] {name}"); out.rmdir(); continue
        uid = get_or_add_user(name, pin_hash, label)
        mapping[str(label)] = uid; total+=saved
        print(f"  [OK]   {name} → label={label}, images={saved}")

    save_labels(mapping)
    print(f"\nImported {len(mapping)} users, {total} images total.")
    print("Training LBPH model…")
    LBPH().train(); print("Done → trainer.yml")
    print(f"All users share default PIN: {default_pin}")
    print("⚠  Change PINs: python doorlock_esp32.py --change-pin --name <Name>")

# ─────────────── LBPH recognizer ─────────────
class LBPH:
    def __init__(self):
        if not hasattr(cv2,"face"): raise RuntimeError("Install opencv-contrib-python")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector   = get_face_detector()

    def train(self):
        ensure_dirs(); faces,labels=[],[]
        for ld in DATASET_DIR.iterdir():
            if not ld.is_dir(): continue
            try: lbl=int(ld.name)
            except: continue
            for p in list(ld.glob("*.jpg"))+list(ld.glob("*.png")):
                img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
                if img is not None: faces.append(cv2.resize(img,(200,200))); labels.append(lbl)
        if not faces: raise RuntimeError("No training data found.")
        self.recognizer.train(faces,np.array(labels,dtype=np.int32))
        self.recognizer.save(str(TRAINER_PATH))

    def load(self):
        if not TRAINER_PATH.exists(): raise RuntimeError("No trained model. Enroll or import first.")
        self.recognizer.read(str(TRAINER_PATH))

    def predict_on_frame(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray,1.2,5,minSize=(80,80))
        if len(faces)==0: return None,None,None
        x,y,w,h = max(faces,key=lambda r:r[2]*r[3])
        roi = cv2.resize(gray[y:y+h,x:x+w],(200,200))
        lbl,conf = self.recognizer.predict(roi)
        return lbl,conf,(x,y,w,h)

# ─────────────── Live enrollment ─────────────
def capture_samples_for_label(label, samples=30, cam_source=0):
    ensure_dirs(); out=DATASET_DIR/str(label); out.mkdir(parents=True,exist_ok=True)
    det=get_face_detector(); cam=get_camera(cam_source); saved=0
    print("Look at the camera. Press Q to stop early.")
    try:
        while saved<samples:
            ret,frame=cam.read()
            if not ret: continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            for(x,y,w,h) in det.detectMultiScale(gray,1.2,5,minSize=(80,80)):
                roi=cv2.resize(gray[y:y+h,x:x+w],(200,200))
                cv2.imwrite(str(out/f"{label}_{saved+1:03d}.jpg"),roi); saved+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"{saved}/{samples}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                break
            cv2.imshow("Enrollment — Q to quit",frame)
            if cv2.waitKey(1)&0xFF in(ord("q"),ord("Q")): break
    finally: cam.release(); cv2.destroyAllWindows()
    return saved

# ─────────────── PIN-change dialog ───────────
class _PINDialog(tk.Toplevel):
    def __init__(self,parent,uid,on_done):
        super().__init__(parent); self.title("Change PIN")
        self.configure(bg=DARK_BG); self.resizable(False,False); self.grab_set()
        self._uid=uid; self._done=on_done
        for lbl,attr in[("New PIN:","_p1"),("Confirm:","_p2")]:
            tk.Label(self,text=lbl,bg=DARK_BG,fg=FG,font=("Segoe UI",11)).pack(padx=20,pady=(10,2),anchor="w")
            v=tk.StringVar(); setattr(self,attr,v)
            tk.Entry(self,textvariable=v,show="●",font=("Courier",14),width=14,
                     bg="#0f172a",fg=FG,insertbackground=FG,relief=tk.FLAT).pack(padx=20)
        tk.Button(self,text="Save",font=("Segoe UI",11,"bold"),bg=ACCENT,fg="white",
                  relief=tk.FLAT,padx=12,pady=6,command=self._save).pack(pady=14)

    def _save(self):
        p1,p2=self._p1.get(),self._p2.get()
        if p1!=p2: messagebox.showerror("Mismatch","PINs do not match.",parent=self); return
        try: update_pin(self._uid,hash_pin(p1))
        except ValueError as e: messagebox.showerror("Invalid",str(e),parent=self); return
        self._done(); messagebox.showinfo("Done","PIN updated.",parent=self); self.destroy()

# ─────────────── GUI ─────────────────────────
class DoorLockApp:
    def __init__(self,root,cam_source=0):
        self.root=root; root.title("ESP32 Door Lock — Face + PIN")
        root.configure(bg=DARK_BG); root.geometry("1080x660")
        self.rec=LBPH(); self.rec.load()
        self.labels_map=load_labels()
        self.current_label=self.current_conf=None
        self.stop_event=threading.Event()
        self.cam=get_camera(cam_source)
        self._build_ui()
        threading.Thread(target=self._cam_loop,daemon=True).start()
        root.protocol("WM_DELETE_WINDOW",self._close)

    def _build_ui(self):
        s=ttk.Style(); s.theme_use("clam")
        s.configure("TNotebook",background=DARK_BG,borderwidth=0)
        s.configure("TNotebook.Tab",background=PANEL_BG,foreground=FG,padding=[14,6],font=("Segoe UI",10,"bold"))
        s.map("TNotebook.Tab",background=[("selected",ACCENT)])
        nb=ttk.Notebook(self.root); nb.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        self._lock_tab(nb); self._admin_tab(nb); self._logs_tab(nb)

    # — Lock tab —
    def _lock_tab(self,nb):
        frm=tk.Frame(nb,bg=DARK_BG); nb.add(frm,text="  🔒 Door Lock  ")
        left=tk.Frame(frm,bg=DARK_BG); left.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=10,pady=10)
        self.vid=tk.Label(left,bg="#000",width=640,height=480); self.vid.pack()
        right=tk.Frame(frm,bg=PANEL_BG,width=300); right.pack(side=tk.RIGHT,fill=tk.Y,padx=(0,10),pady=10)
        right.pack_propagate(False)
        tk.Label(right,text="🔒  Door Lock",font=("Segoe UI",20,"bold"),bg=PANEL_BG,fg=FG).pack(pady=(20,5))
        self.status_var=tk.StringVar(value="Waiting for face…")
        tk.Label(right,textvariable=self.status_var,wraplength=270,font=("Segoe UI",12),bg=PANEL_BG,fg=FG2).pack(pady=6)
        tk.Label(right,text="Enter PIN:",font=("Segoe UI",11),bg=PANEL_BG,fg=FG).pack(pady=(10,2))
        self.pin_var=tk.StringVar()
        tk.Entry(right,textvariable=self.pin_var,font=("Courier",22),show="●",width=10,
                 justify="center",bg="#0f172a",fg=FG,insertbackground=FG,
                 relief=tk.FLAT,highlightthickness=1,highlightbackground=ACCENT).pack(pady=4)
        pad=tk.Frame(right,bg=PANEL_BG); pad.pack(pady=8)
        for t,r,c in[("1",0,0),("2",0,1),("3",0,2),("4",1,0),("5",1,1),("6",1,2),
                      ("7",2,0),("8",2,1),("9",2,2),("←",3,0),("0",3,1),("C",3,2)]:
            tk.Button(pad,text=t,font=("Segoe UI",13,"bold"),width=4,
                      bg="#334155",fg=FG,activebackground=ACCENT,relief=tk.FLAT,
                      command=lambda k=t:self._key(k)).grid(row=r,column=c,padx=4,pady=4)
        self.unlock_btn=tk.Button(right,text="Unlock 🔓",font=("Segoe UI",13,"bold"),
                                  bg=ACCENT,fg="white",relief=tk.FLAT,state=tk.DISABLED,
                                  width=18,command=self._unlock)
        self.unlock_btn.pack(pady=10)
        self.msg_lbl=tk.Label(right,text="",font=("Segoe UI",16,"bold"),bg=PANEL_BG,fg=FG)
        self.msg_lbl.pack(pady=6)

    # — Admin tab —
    def _admin_tab(self,nb):
        frm=tk.Frame(nb,bg=DARK_BG); nb.add(frm,text="  ⚙ Admin  ")
        tk.Label(frm,text="Registered Users",font=("Segoe UI",14,"bold"),bg=DARK_BG,fg=FG).pack(pady=(15,5))
        cols=("ID","Name","Label")
        self.utree=ttk.Treeview(frm,columns=cols,show="headings",height=14)
        for col in cols: self.utree.heading(col,text=col); self.utree.column(col,width=160,anchor="center")
        self.utree.pack(padx=20,fill=tk.X)
        row=tk.Frame(frm,bg=DARK_BG); row.pack(pady=8)
        def btn(t,cmd,col=ACCENT):
            return tk.Button(row,text=t,font=("Segoe UI",10,"bold"),bg=col,fg="white",
                             relief=tk.FLAT,padx=12,pady=5,command=cmd)
        btn("🔄 Refresh",self._refresh_users).pack(side=tk.LEFT,padx=6)
        btn("🔑 Change PIN",self._gui_change_pin).pack(side=tk.LEFT,padx=6)
        btn("🗑 Remove User",self._gui_remove_user,DANGER).pack(side=tk.LEFT,padx=6)
        self._refresh_users()

    # — Logs tab —
    def _logs_tab(self,nb):
        frm=tk.Frame(nb,bg=DARK_BG); nb.add(frm,text="  📋 Access Logs  ")
        tk.Label(frm,text="Recent Access Logs (last 100)",font=("Segoe UI",14,"bold"),bg=DARK_BG,fg=FG).pack(pady=(15,5))
        cols=("Timestamp","User","Method","Result","Confidence")
        self.ltree=ttk.Treeview(frm,columns=cols,show="headings",height=20)
        for col,w in zip(cols,[180,160,100,90,110]):
            self.ltree.heading(col,text=col); self.ltree.column(col,width=w,anchor="center")
        sb=ttk.Scrollbar(frm,orient=tk.VERTICAL,command=self.ltree.yview)
        self.ltree.configure(yscrollcommand=sb.set)
        self.ltree.pack(side=tk.LEFT,padx=(20,0),fill=tk.BOTH,expand=True)
        sb.pack(side=tk.LEFT,fill=tk.Y,pady=5)
        tk.Button(frm,text="🔄 Refresh Logs",font=("Segoe UI",10,"bold"),bg=ACCENT,fg="white",
                  relief=tk.FLAT,padx=12,pady=5,command=self._refresh_logs).pack(pady=8)
        self._refresh_logs()

    # — Helpers —
    def _key(self,k):
        if k=="C": self.pin_var.set("")
        elif k=="←": self.pin_var.set(self.pin_var.get()[:-1])
        else: self.pin_var.set(self.pin_var.get()+k)

    def _refresh_users(self):
        [self.utree.delete(r) for r in self.utree.get_children()]
        [self.utree.insert("",tk.END,values=(u,n,l)) for u,n,l in get_all_users()]

    def _refresh_logs(self):
        [self.ltree.delete(r) for r in self.ltree.get_children()]
        for ts,name,method,success,conf in get_recent_logs():
            self.ltree.insert("",tk.END,values=(ts,name or"unknown",method,
                "✅ GRANTED"if success else"❌ DENIED",f"{conf:.1f}"if conf else"—"))

    def _gui_change_pin(self):
        sel=self.utree.selection()
        if not sel: messagebox.showwarning("Select","Select a user first."); return
        uid=self.utree.item(sel[0])["values"][0]
        _PINDialog(self.root,uid,self._refresh_users)

    def _gui_remove_user(self):
        sel=self.utree.selection()
        if not sel: messagebox.showwarning("Select","Select a user first."); return
        uid,name,label=self.utree.item(sel[0])["values"]
        if not messagebox.askyesno("Confirm",f"Remove '{name}'?"): return
        lbl=delete_user(uid)
        folder=DATASET_DIR/str(lbl)
        if folder.exists(): shutil.rmtree(folder,ignore_errors=True)
        try: LBPH().train()
        except RuntimeError: pass
        m={k:v for k,v in load_labels().items() if v!=uid}; save_labels(m); self.labels_map=m
        self._refresh_users(); messagebox.showinfo("Done",f"'{name}' removed.")

    def _cam_loop(self):
        while not self.stop_event.is_set():
            ret,frame=self.cam.read()
            if not ret: continue
            label,conf,rect=self.rec.predict_on_frame(frame)
            recog=rect is not None and conf is not None and conf<CONFIDENCE_THRESHOLD
            self.root.after(0,lambda r=recog:self.unlock_btn.configure(state=tk.NORMAL if r else tk.DISABLED))
            disp=frame.copy()
            if rect:
                x,y,w,h=rect; self.current_label=label; self.current_conf=conf
                row=get_user_by_label(label) if recog else None
                uname=row[1] if row else "Unknown"
                color=(0,200,0) if recog else(0,60,220)
                cv2.rectangle(disp,(x,y),(x+w,y+h),color,2)
                cv2.putText(disp,f"{uname} ({conf:.1f})",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                s=f"✅ {uname} — enter PIN to unlock"if recog else f"❌ Unknown (score={conf:.1f})"
                self.root.after(0,lambda v=s:self.status_var.set(v))
            else:
                self.current_label=self.current_conf=None
                self.root.after(0,lambda:self.status_var.set("Waiting for face…"))
            rgb=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
            imgtk=ImageTk.PhotoImage(Image.fromarray(rgb).resize((640,480)))
            def up(img=imgtk): self.vid.imgtk=img; self.vid.configure(image=img)
            self.root.after(0,up)

    def _unlock(self):
        if self.current_label is None:
            messagebox.showwarning("No face","Look at the camera first."); return
        if self.current_conf>=CONFIDENCE_THRESHOLD:
            messagebox.showwarning("Unknown","Face not recognized."); return
        row=get_user_by_label(self.current_label)
        if not row: messagebox.showerror("Error","User not in DB."); return
        uid,name,pin_hash,_=row
        ok=verify_pin(self.pin_var.get(),pin_hash)
        if ok:
            self.root.after(0,lambda:self.msg_lbl.configure(text=f"UNLOCKED ✅  Welcome, {name}!",fg=SUCCESS))
            log_event(uid,"face+pin",True,float(self.current_conf))
        else:
            self.root.after(0,lambda:self.msg_lbl.configure(text="DENIED ❌  Wrong PIN",fg=DANGER))
            log_event(uid,"face+pin",False,float(self.current_conf))
        self.pin_var.set(""); self._refresh_logs()

    def _close(self):
        self.stop_event.set()
        try: self.cam.release()
        except: pass
        self.root.destroy()

# ─────────────── CLI ─────────────────────────
def cli():
    p=argparse.ArgumentParser(description="ESP32 Door Lock — Face (LBPH) + PIN",
        formatter_class=argparse.RawTextHelpFormatter,epilog="""
Examples:
  python doorlock_esp32.py --init-db
  python doorlock_esp32.py --import-dataset /path/to/Face_Recognition_Dataset --default-pin 1234
  python doorlock_esp32.py --enroll --name "Alice"
  python doorlock_esp32.py --enroll --name "Bob" --cam http://192.168.1.105:81/stream
  python doorlock_esp32.py --change-pin --name "Alice"
  python doorlock_esp32.py --run
  python doorlock_esp32.py --run --cam http://192.168.1.105:81/stream
""")
    p.add_argument("--init-db",action="store_true"); p.add_argument("--enroll",action="store_true")
    p.add_argument("--change-pin",action="store_true"); p.add_argument("--run",action="store_true")
    p.add_argument("--name",type=str); p.add_argument("--samples",type=int,default=30)
    p.add_argument("--cam",type=str,default=None)
    p.add_argument("--import-dataset",type=str,metavar="PATH")
    p.add_argument("--default-pin",type=str,default="0000")
    args=p.parse_args()

    cam_source=0
    if args.cam:
        cam_source=args.cam if args.cam.startswith("http") else (int(args.cam) if args.cam.isdigit() else args.cam)

    init_db()

    if args.init_db and not any([args.enroll,args.run,args.import_dataset,args.change_pin]):
        print("Database initialized."); return

    if args.import_dataset:
        import_kaggle_dataset(args.import_dataset,default_pin=args.default_pin); return

    if args.enroll:
        if not args.name: print("Provide --name"); return
        p1=getpass.getpass("Enter PIN: "); p2=getpass.getpass("Confirm:   ")
        if p1!=p2: print("PIN mismatch."); return
        try: ph=hash_pin(p1)
        except ValueError as e: print(f"Error: {e}"); return
        lbl=next_label(); uid=add_user(args.name,ph,lbl)
        print(f"Registered '{args.name}' label={lbl} uid={uid}")
        saved=capture_samples_for_label(lbl,args.samples,cam_source)
        print(f"Captured {saved} sample(s).")
        if saved==0: print("No samples — enrollment failed."); return
        LBPH().train(); print("Model trained.")
        m=load_labels(); m[str(lbl)]=uid; save_labels(m); return

    if args.change_pin:
        if not args.name: print("Provide --name"); return
        users={n:(u,l) for u,n,l in get_all_users()}
        if args.name not in users: print(f"User '{args.name}' not found."); return
        uid,_=users[args.name]
        p1=getpass.getpass("New PIN:     "); p2=getpass.getpass("Confirm PIN: ")
        if p1!=p2: print("Mismatch."); return
        try: update_pin(uid,hash_pin(p1)); print(f"PIN updated for '{args.name}'.")
        except ValueError as e: print(f"Error: {e}")
        return

    root=tk.Tk(); DoorLockApp(root,cam_source=cam_source); root.mainloop()

if __name__=="__main__":
    cli()