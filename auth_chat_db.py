#auth_chat_db.py
import sqlite3
import json
import os

def conectar_db():
    return sqlite3.connect("users.db")

def inicializar_tabelas():
    conn = conectar_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            chat_name TEXT NOT NULL,
            messages TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def salvar_chat(username, chat_name, messages):
    conn = conectar_db()
    c = conn.cursor()
    messages_json = json.dumps(messages)
    c.execute("INSERT INTO chats (username, chat_name, messages) VALUES (?, ?, ?)",
              (username, chat_name, messages_json))
    conn.commit()
    conn.close()

def listar_chats(username):
    conn = conectar_db()
    c = conn.cursor()
    c.execute("SELECT id, chat_name FROM chats WHERE username = ? ORDER BY created_at DESC", (username,))
    result = c.fetchall()
    conn.close()
    return result

def carregar_chat(chat_id):
    conn = conectar_db()
    c = conn.cursor()
    c.execute("SELECT messages FROM chats WHERE id = ?", (chat_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return []

def delete_chat(chat_id):
    conn = conectar_db()
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()



