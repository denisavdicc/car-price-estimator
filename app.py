import os
from pathlib import Path
from flask import Flask, render_template, request, session, redirect
import joblib
import pandas as pd
import numpy as np
from werkzeug.security import check_password_hash, generate_password_hash
from cs50 import SQL
from flask_session import Session
from functools import wraps

Path("cars.db").touch(exist_ok=True)

db = SQL("sqlite:///cars.db")

db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        hash TEXT NOT NULL
    )
""")

db.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        brand TEXT NOT NULL,
        model TEXT NOT NULL,
        model_year INTEGER NOT NULL,
        mileage INTEGER NOT NULL,
        fuel_type TEXT NOT NULL,
        transmission TEXT NOT NULL,
        prediction REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
""")

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

model = joblib.load("model.pkl")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST","GET"])
@login_required
def predict():
    prediction = None
    if request.method == "POST":
        brand = request.form.get("brand")
        model_name = request.form.get("model")
        model_year = int(request.form.get("model_year") or 0)
        mileage = int(request.form.get("mileage") or 0)
        fuel_type = request.form.get("fuel_type")
        transmission = request.form.get("transmission")

        if not request.form.get("brand") or not request.form.get("model") or not request.form.get("model_year") or not request.form.get("mileage") or not request.form.get("fuel_type") or not request.form.get("transmission"):
            return render_template("predict.html",prediction=None,error="Please fill out all fields.")
        if model_year < 1900 or model_year > 2026:
            return render_template("predict.html", prediction=None, error="Model year must be between 1900 and 2026.")

        data = pd.DataFrame([{"brand": brand,"model": model_name,"model_year": model_year,"milage": mileage,"fuel_type": fuel_type,"transmission": transmission}])

        log_prediction = model.predict(data)[0]
        prediction = np.expm1(log_prediction)

        db.execute("INSERT INTO history (user_id, brand, model, model_year, mileage, fuel_type, transmission, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", session["user_id"], brand, model_name, model_year, mileage, fuel_type, transmission, prediction)

        return render_template("predict.html",prediction=prediction,brand=brand,model=model_name,model_year=model_year,mileage=mileage,fuel_type=fuel_type,transmission=transmission)
    else:
        return render_template("predict.html", prediction=None)
    

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        if not request.form.get("username"):
            return render_template("login.html",error="Please fill out all fields.")
        elif not request.form.get("password"):
            return render_template("login.html",error="Please fill out all fields.")
        
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return render_template("login.html",error="Username or password is not correct")
        session["user_id"] = rows[0]["id"]
        return redirect("/predict")
    else:
        session.clear()
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/register", methods=["POST", "GET"])
def register():
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        if not request.form.get("username") or not request.form.get("password") or not request.form.get("confirmation"):
            return render_template("register.html", error="Please fill out all fields.")
        if len(password) < 8:
            return render_template("register.html", error="Password must be at least 8 characters long.")
        if password != confirmation:
            return render_template("register.html", error="Password confirmation does not match.")
        
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)
        if len(rows) > 0:
            return render_template("register.html",error="Username already exists.")

        user_id = db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, generate_password_hash(password, method="pbkdf2:sha256"))

        session["user_id"] = user_id

        return redirect("/predict")
    else:
        return render_template("register.html")
    

@app.route("/change_password", methods=["GET","POST"])
@login_required
def change_password():
    if request.method == "POST":
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")
        confirmation = request.form.get("confirmation")
        if not old_password or not new_password or not confirmation:
            return render_template("change_password.html", error="Please input all information.")
        elif len(new_password) < 8:
            return render_template("change_password.html", error="Password must be at least 8 characters long.")
        elif new_password != confirmation:
            return render_template("change_password.html", error="The passwords do not match.")

        rows = db.execute("SELECT hash FROM users WHERE id = ?", session["user_id"])
        if not check_password_hash(rows[0]["hash"], old_password):
            return render_template("change_password.html", error="The old password is not correct")

        new_hash = generate_password_hash(new_password, method="pbkdf2:sha256")
        db.execute("UPDATE users SET hash = ? WHERE id = ?", new_hash, session["user_id"])

        return redirect("/predict")
    else:
        return render_template("change_password.html")


@app.route("/history", methods=["GET"])
@login_required
def history():
    searches = db.execute("SELECT id, brand, model, model_year, mileage, fuel_type, transmission, prediction, created_at FROM history WHERE user_id = ? ORDER BY created_at DESC", session["user_id"])
    return render_template("history.html", searches=searches)


@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    db.execute("DELETE FROM history WHERE user_id = ?", session["user_id"])
    return redirect("/history")


@app.route("/delete_prediction", methods=["POST"])
@login_required
def delete_prediction():
    prediction_id = request.form.get("id")
    db.execute("DELETE FROM history WHERE id = ? AND user_id = ?", prediction_id, session["user_id"])
    return redirect("/history")


if __name__ == "__main__":
    app.run(debug=True)



