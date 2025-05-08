from flask import Flask, render_template, request, redirect, session, flash, url_for
import os
from PIL import Image
import torch
import clip
import faiss
import numpy as np
import uuid
import sqlite3

app = Flask(__name__)
app.secret_key = 'supersecretkey'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

ADMIN_DIR = "admin_images"
USER_DIR = "user_uploads"
FAISS_INDEX_PATH = "faiss_index/index.faiss"
IMAGE_IDS_PATH = "faiss_index/image_ids.npy"

os.makedirs(ADMIN_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

# ---- DB Setup ----
def init_db():
    with sqlite3.connect("users.db") as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            credits INTEGER DEFAULT 0
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            embedding BLOB
        )""")
init_db()

# ---- FAISS Setup ----
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(IMAGE_IDS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=True)
    else:
        index = faiss.IndexFlatIP(512)
        image_ids = np.array([], dtype=object)
    return index, image_ids

faiss_index, image_ids = load_faiss_index()

def save_faiss_index():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    np.save(IMAGE_IDS_PATH, image_ids)

def get_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def is_duplicate_embedding(username, new_embedding):
    with sqlite3.connect("users.db") as con:
        rows = con.execute("SELECT embedding FROM uploads WHERE username=?", (username,)).fetchall()
        for (blob,) in rows:
            prev_embedding = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
            similarity = float(np.dot(prev_embedding, new_embedding.T))
            if similarity >= 0.99:
                return True
    return False

# ---- Routes ----

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            with sqlite3.connect("users.db") as con:
                con.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            flash("Registration successful! Please log in.")
            return redirect("/login")
        except:
            flash("Username already exists.")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":
            session["user"] = "admin"
            return redirect("/dashboard_admin")

        with sqlite3.connect("users.db") as con:
            result = con.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password)).fetchone()
            if result:
                session["user"] = username
                return redirect("/dashboard_user")
            else:
                flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/dashboard_admin", methods=["GET", "POST"])
def dashboard_admin():
    if session.get("user") != "admin":
        return redirect("/login")

    users = []
    with sqlite3.connect("users.db") as con:
        users = con.execute("SELECT username, credits FROM users").fetchall()

    if request.method == "POST":
        file = request.files["image"]
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(ADMIN_DIR, filename)
        file.save(filepath)

        embedding = get_embedding(filepath)
        faiss_index.add(embedding)
        global image_ids
        image_ids = np.append(image_ids, filename)
        save_faiss_index()

        flash("Admin image uploaded and indexed.")
        return redirect("/dashboard_admin")

    return render_template("dashboard_admin.html", users=users)

@app.route("/dashboard_user", methods=["GET", "POST"])
def dashboard_user():
    username = session.get("user")
    if not username or username == "admin":
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(USER_DIR, filename)
        file.save(filepath)

        embedding = get_embedding(filepath)

        # Check for duplicate
        if is_duplicate_embedding(username, embedding):
            flash("⚠️ Duplicate image detected. No credits awarded.")
            return redirect("/dashboard_user")

        # Search for match with admin images
        D, I = faiss_index.search(embedding, k=1)
        similarity = D[0][0]
        matched_id = image_ids[int(I[0][0])] if I[0][0] < len(image_ids) else None

        threshold = 0.9
        if similarity >= threshold:
            with sqlite3.connect("users.db") as con:
                con.execute("UPDATE users SET credits = credits + 10 WHERE username=?", (username,))
                con.execute("INSERT INTO uploads (username, embedding) VALUES (?, ?)", (username, embedding.astype(np.float32).tobytes()))
            flash(f"✅ Match found! 10 points awarded.")
        else:
            flash("❌ No match found.")

        return redirect("/dashboard_user")

    return render_template("dashboard_user.html", username=username)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
