# chat_db.py
import os
from typing import List, Optional, Tuple
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

# Conexão com MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chat_database"]  
chats_collection = db["chats"] 

def salvar_chat(
    username: str,
    chat_name: str,
    messages: List[dict],
    pdf_paths: Optional[List[str]] = None,
) -> None:
    """Insere ou atualiza um chat com suas mensagens e caminhos de PDFs."""
    dados = {"messages": messages}
    if pdf_paths is not None:
        dados["pdf_paths"] = pdf_paths

    existing = chats_collection.find_one({"username": username, "chat_name": chat_name})

    if existing:
        chats_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": dados}
        )
    else:
        dados.update({"username": username, "chat_name": chat_name})
        chats_collection.insert_one(dados)

def listar_chats(username: str) -> List[Tuple[str, str]]:
    """Lista todos os chats de um usuário (retorna id e nome)."""
    resultados = chats_collection.find({"username": username})
    return [(str(r["_id"]), r["chat_name"]) for r in resultados]

def carregar_chat(chat_id: str) -> Tuple[List[dict], Optional[List[str]]]:
    """Carrega mensagens e PDFs de um chat pelo ID."""
    chat = chats_collection.find_one({"_id": ObjectId(chat_id)})
    if chat:
        return chat.get("messages", []), chat.get("pdf_paths")
    return [], None

def delete_chat(chat_id: str) -> None:
    """Remove um chat pelo ID."""
    chats_collection.delete_one({"_id": ObjectId(chat_id)})
