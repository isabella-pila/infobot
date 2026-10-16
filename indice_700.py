"""
indice_700.py
=============
Monta (ou carrega do disco) o índice FAISS com trechos de 700 caracteres e sobreposição de 200,
usando a MESMA extração de texto do seu rag_core.py (PyMuPDF, texto das páginas concatenado,
FAISS.from_texts, embedding gemini-embedding-2-preview).

Antes de gastar cota de embeddings, ele SEGMENTA o texto, MEDE os trechos e PARA se não forem
de ~700 caracteres. Assim você não roda tudo por engano com o índice de 1.500.

ATENÇÃO - qual segmentador? Há três opções (todas 700/200 FIXOS, sem depender do rag_core):
  * "recursive" (PADRÃO)  : RecursiveCharacterTextSplitter com os separadores do seu rag_core.py
                            ["\\n\\n", "\\n", ". ", " ", ""]
  * "character"           : CharacterTextSplitter, separador "\\n" (é o que o texto do Artigo 1 descreve)
  * "rag_core"            : chama rag_core.get_text_chunks (só use se o rag_core da SUA pasta estiver em 700/200)
Use o que realmente gerou os resultados do Artigo 1 (veja o app.py/histórico daquela época) e
descreva o mesmo na Tabela I. Para comparar os dois sem gastar embeddings:
    python indice_700.py --comparar

Uso:
    python indice_700.py --comparar             # só mede os trechos de cada segmentador
    python indice_700.py                        # monta/carrega o índice e confere
    python indice_700.py --reconstruir          # força refazer o índice
"""

import hashlib
import json
import os
import statistics
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------- AJUSTE ----------------
CAMINHO_DOCUMENTO = os.getenv("CAMINHO_DOCUMENTO", "perguntas2.pdf")   # o mesmo arquivo do Artigo 1
SPLITTER = os.getenv("SPLITTER", "recursive")                          # "recursive", "character" ou "rag_core"
# ----------------------------------------
MODELO_EMBEDDING = "models/gemini-embedding-2-preview"   # igual ao rag_core.get_vectorstore
PASTA_INDICE = "indice_faiss_700"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 200


def _texto_do_pdf(caminho: str) -> str:
    import fitz  # PyMuPDF, como no rag_core

    if not os.path.exists(caminho):
        sys.exit(f"Arquivo '{caminho}' não encontrado. Ajuste CAMINHO_DOCUMENTO (ou a variável de ambiente).")
    with fitz.open(caminho) as doc:
        return "".join(page.get_text("text") for page in doc)


def segmentar(texto: str, splitter: str = SPLITTER) -> list:
    """Segmenta em 700/200 com o segmentador escolhido."""
    if splitter == "recursive":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:  # LangChain antigo
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
        ).split_text(texto)
    if splitter == "rag_core":
        # Usa a função do seu projeto; ela precisa estar em 700/200 (conferido abaixo).
        import rag_core

        print(f"[Índice] rag_core carregado de: {rag_core.__file__}")
        return rag_core.get_text_chunks(texto)
    if splitter == "character":
        try:
            from langchain_text_splitters import CharacterTextSplitter
        except ImportError:  # LangChain antigo
            from langchain.text_splitter import CharacterTextSplitter
        return CharacterTextSplitter(
            separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
        ).split_text(texto)
    sys.exit(f"SPLITTER inválido: {splitter!r} (use 'recursive', 'character' ou 'rag_core').")


def estatisticas(trechos: list) -> dict:
    tam = [len(t) for t in trechos]
    return {
        "n_trechos": len(tam),
        "tamanho_medio": round(statistics.mean(tam)),
        "tamanho_min": min(tam),
        "tamanho_max": max(tam),
    }


def info_indice(vectorstore) -> dict:
    """Mede os trechos de um FAISS já carregado (usado no .config.json dos resultados)."""
    try:
        return estatisticas([d.page_content for d in vectorstore.docstore._dict.values()])
    except Exception:
        return {}


def verificar(info: dict, esperado: int = CHUNK_SIZE) -> None:
    """Para a execução se os trechos não forem de ~700 caracteres."""
    print("[Índice] tamanho dos trechos:", info)
    if not info:
        print("[Índice] AVISO: não consegui medir os trechos; confira o índice manualmente.")
        return
    # Com 700/200 nenhum trecho deve passar muito de 700 (o CharacterTextSplitter pode passar um
    # pouco se uma linha for longa). Perto de 1.500 indica o índice errado.
    if info["tamanho_medio"] > esperado * 1.25 or info["tamanho_max"] > 1000:
        sys.exit(
            f"\nOs trechos NÃO parecem ser de {esperado} caracteres (média {info['tamanho_medio']}, "
            f"máximo {info['tamanho_max']}). Se o segmentador for 'rag_core', o rag_core.py da sua "
            f"pasta está com chunk_size maior; use SPLITTER=recursive (padrão) ou corrija o rag_core.py. "
            f"Depois apague a pasta '{PASTA_INDICE}' e rode de novo."
        )


def _embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING, google_api_key=os.getenv("GOOGLE_API_KEY"))


def _assinatura(trechos: list) -> str:
    h = hashlib.sha256()
    for t in trechos:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def carregar_vectorstore_700(reconstruir: bool = False, splitter: str = SPLITTER,
                             caminho: str = CAMINHO_DOCUMENTO):
    """Segmenta, confere e (se preciso) gera os embeddings. Guarda o índice em disco e só o
    reaproveita se os trechos forem exatamente os mesmos (mesma assinatura)."""
    from langchain_community.vectorstores import FAISS

    texto = _texto_do_pdf(caminho)
    trechos = segmentar(texto, splitter)
    info = estatisticas(trechos)
    print(f"[Índice] segmentador: {splitter} | arquivo: {caminho}")
    verificar(info)                       # antes de gastar cota de embeddings
    assin = _assinatura(trechos)

    meta = os.path.join(PASTA_INDICE, "meta.json")
    if os.path.isdir(PASTA_INDICE) and os.path.exists(meta) and not reconstruir:
        with open(meta, encoding="utf-8") as f:
            if json.load(f).get("assinatura") == assin:
                print(f"[Índice] carregando de '{PASTA_INDICE}' (trechos idênticos)")
                return FAISS.load_local(PASTA_INDICE, _embeddings(), allow_dangerous_deserialization=True)
        print("[Índice] o índice salvo é de outra segmentação; refazendo.")

    print(f"[Índice] gerando embeddings de {len(trechos)} trechos (consome cota do Gemini)...")
    vs = FAISS.from_texts(trechos, embedding=_embeddings())     # como no rag_core
    vs.save_local(PASTA_INDICE)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"assinatura": assin, "splitter": splitter, "chunk_size": CHUNK_SIZE,
                   "chunk_overlap": CHUNK_OVERLAP, **info}, f, ensure_ascii=False, indent=2)
    print(f"[Índice] salvo em '{PASTA_INDICE}'")
    return vs


if __name__ == "__main__":
    if "--comparar" in sys.argv:
        txt = _texto_do_pdf(CAMINHO_DOCUMENTO)
        for s in ("recursive", "character", "rag_core"):
            try:
                print(f"{s:9s}", estatisticas(segmentar(txt, s)))
            except SystemExit:
                raise
            except Exception as e:
                print(f"{s:9s} erro: {e}")
    else:
        carregar_vectorstore_700(reconstruir="--reconstruir" in sys.argv)