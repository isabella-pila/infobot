import sqlite3
import bcrypt
import streamlit_authenticator as stauth

def conectar_db():
    return sqlite3.connect("users.db")

def inicializar_db():
    conn = conectar_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def carregar_usuarios():
    conn = conectar_db()
    c = conn.cursor()
    c.execute("SELECT username, name, password FROM users")
    rows = c.fetchall()
    conn.close()

    users = {"usernames": {}}
    for username, name, password in rows:
        users["usernames"][username] = {"name": name, "password": password}
    return users if users["usernames"] else None

def verificar_credenciais(username, password):
    users = carregar_usuarios()
    if not users:
        return None, None, None

    user_data = users["usernames"].get(username)
    if user_data and bcrypt.checkpw(password.encode(), user_data["password"].encode()):
        return user_data["name"], True, username
    elif user_data:
        return None, False, username
    else:
        return None, False, None

