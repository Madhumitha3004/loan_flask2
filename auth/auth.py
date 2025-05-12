from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify 
import sqlite3
import os

auth_bp = Blueprint('auth', __name__)
DB_PATH = os.path.join("database", "users.db")

# Initialize the database
def init_db():
    conn = sqlite3.connect('database/users.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT
        );
    ''')
    conn.commit()
    conn.close()

@auth_bp.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()  # This will now parse the incoming JSON request
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        phone = data.get('phone')

        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO users (username, password, email, phone) VALUES (?, ?, ?, ?)",
                (username, password, email, phone)
            )
            conn.commit()
            return jsonify({"message": "User created successfully"}), 200
        except sqlite3.IntegrityError:
            return jsonify({"message": "Username already exists"}), 409  # Conflict Error
        except Exception as e:
            return jsonify({"message": f"Error: {str(e)}"}), 500  # General error
        finally:
            conn.close()
    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('auth.home'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@auth_bp.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('auth.login'))

