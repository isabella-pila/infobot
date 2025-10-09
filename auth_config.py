import sqlite3
import bcrypt
import streamlit_authenticator as stauth
from dotenv import load_dotenv
import os
import bcrypt
import streamlit_authenticator as stauth
from psycopg2.errors import DuplicateTable
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()


def conectar_db():
    return psycopg2.connect(
        os.getenv("DATABASE_URL"),
        cursor_factory=RealDictCursor
    )




def inicializar_db():
    conn = conectar_db()
    c = conn.cursor()
    try:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
    except Exception as e:
        print("Erro ao criar tabela:", e)
    finally:
        conn.close()

def carregar_usuarios():
    conn = conectar_db()
    c = conn.cursor()
    c.execute("SELECT username, name, password FROM users")
    rows = c.fetchall()
    conn.close()

    users = {"usernames": {}}
    for row in rows:
        users["usernames"][row["username"]] = {
            "name": row["name"],
            "password": row["password"]
        }
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
