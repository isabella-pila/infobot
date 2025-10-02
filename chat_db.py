# chat_db.py
from tinydb import TinyDB, Query
import os
from typing import List, Optional, Tuple

os.makedirs("data", exist_ok=True)
db = TinyDB("data/chat_db.json")
Chat = Query()

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

    existing = db.search((Chat.username == username) & (Chat.chat_name == chat_name))
    if existing:
        db.update(dados, doc_ids=[existing[0].doc_id])
    else:
        dados.update({"username": username, "chat_name": chat_name})
        db.insert(dados)

def listar_chats(username: str) -> List[Tuple[int, str]]:
    resultados = db.search(Chat.username == username)
    return [(r.doc_id, r["chat_name"]) for r in resultados]

def carregar_chat(chat_id: int) -> Tuple[List[dict], Optional[List[str]]]:
    chat = db.get(doc_id=chat_id)
    if chat:
        return chat["messages"], chat.get("pdf_paths")
    return [], None


def delete_chat(chat_id: int) -> None:
    db.remove(doc_ids=[chat_id])
