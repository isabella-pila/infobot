"""
rag_core.py
============
Funções puras (sem Streamlit) extraídas do app.py do Artigo 1: carregamento/vetorização
de PDFs, prompt de busca web e pipeline de fallback via Serper. Tanto o app.py (interface)
quanto o agentic_rag_langgraph.py (Artigo 2) importam a partir daqui.

Copie este arquivo para a mesma pasta do seu app.py e ajuste o `import` no app.py para
puxar estas funções de cá, em vez de defini-las novamente lá (evita duplicação de código).
"""

import os
import re
import time
import httpx
import fitz
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY) if SERPER_API_KEY else None


# ------------------------------------------------------------
# PDFs / vetorização
# ------------------------------------------------------------
def get_text_chunks(text: str):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=700, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


def carregar_vectorstore_default(path: str = "perguntas2.pdf"):
    """Carrega e vetoriza a base institucional padrão (mesmo arquivo do Artigo 1)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo padrão '{path}' não encontrado! Coloque-o na mesma pasta do script.")

    with fitz.open(path) as doc:
        text = "".join(page.get_text("text") for page in doc)

    chunks = get_text_chunks(text)
    return get_vectorstore(chunks)


# ------------------------------------------------------------
# Busca web (fallback) — mesma lógica do Artigo 1, sem chamadas st.*
# ------------------------------------------------------------
QUERY_TEMPLATE = """
Você é um especialista em pesquisa na internet. Transforme a pergunta em uma query de busca
eficaz para o Google (3 a 7 palavras-chave, sem aspas/booleanos). Inclua "CEFET-MG" apenas se
a pergunta for claramente institucional.

Pergunta do usuário: "{pergunta}"
Query de busca (apenas o texto):
"""

DOMINIOS_BLOQUEADOS = {
    "pinterest.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com", "slideshare.net",
}
TAGS_UTEIS = ["p", "h1", "h2", "h3", "h4", "li", "td", "th", "article", "section"]
MAX_CHARS_POR_SITE = 4000
TIMEOUT_SEGUNDOS = 10
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]
HEADERS_BROWSER = {
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
}


def criar_query_de_busca(llm, pergunta: str) -> str:
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(QUERY_TEMPLATE))
    try:
        resultado = chain.invoke({"pergunta": pergunta})
        query = resultado.get("text", pergunta).strip().strip('"').strip("'")
        return re.sub(r"[`*_]", "", query).strip() or pergunta
    except Exception as e:
        print(f"[criar_query_de_busca] erro: {e}")
        palavras = [p for p in pergunta.split() if len(p) > 3]
        return " ".join(palavras[:6])


def buscar_serper(query: str, max_results: int = 5) -> list[dict]:
    if serper is None:
        return []
    try:
        res = serper.results(query)
    except Exception as e:
        print(f"[buscar_serper] erro: {e}")
        return []

    resultados = []
    for r in res.get("organic", [])[:max_results + 3]:
        link, title, snippet = r.get("link", ""), r.get("title", ""), r.get("snippet", "")
        if not link or not title:
            continue
        dominio = re.sub(r"https?://(www\.)?", "", link).split("/")[0]
        if any(b in dominio for b in DOMINIOS_BLOQUEADOS):
            continue
        resultados.append({"title": title, "link": link, "snippet": snippet, "dominio": dominio})
        if len(resultados) >= max_results:
            break
    return resultados


def _extrair_texto_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    partes = [t.get_text(separator=" ", strip=True) for t in soup.find_all(TAGS_UTEIS)]
    partes = [p for p in partes if len(p) >= 30]
    return "\n".join(partes)[:MAX_CHARS_POR_SITE]


def extrair_conteudo_urls(urls: list[str], resultados_serper: list[dict]) -> list[Document]:
    snippets_por_url = {r["link"]: r for r in resultados_serper}
    documentos = []
    with httpx.Client(timeout=TIMEOUT_SEGUNDOS, follow_redirects=True) as client:
        for url in urls:
            texto = ""
            try:
                resp = client.get(url, headers={**HEADERS_BROWSER, "User-Agent": USER_AGENTS[0]})
                if resp.status_code == 200 and "html" in resp.headers.get("content-type", ""):
                    texto = _extrair_texto_html(resp.text)
            except Exception:
                pass

            if len(texto) < 100:
                info = snippets_por_url.get(url, {})
                if info.get("snippet"):
                    texto = f"{info.get('title', '')}\n{info['snippet']}"

            if len(texto) >= 50:
                documentos.append(Document(page_content=texto, metadata={"source": url}))
            time.sleep(0.3)
    return documentos


SINTESE_TEMPLATE = """
Com base EXCLUSIVAMENTE no conteúdo abaixo, responda à pergunta em português, de forma clara
e organizada. Cite a fonte (URL) quando possível. Se as fontes forem insuficientes, diga isso.

Conteúdo coletado da web:
{context}

Pergunta: {question}

Resposta:
"""


def sintetizar_resposta_web(llm, documentos: list[Document], pergunta: str) -> str:
    if not documentos:
        return ""
    chain = load_qa_chain(llm, chain_type="stuff", prompt=ChatPromptTemplate.from_template(SINTESE_TEMPLATE))
    try:
        resultado = chain.invoke({
            "input_documents": documentos,
            "question": pergunta,
            "context": "\n\n---\n\n".join(f"Fonte: {d.metadata['source']}\n{d.page_content}" for d in documentos),
        })
        texto = resultado.get("output_text", "").strip()
    except Exception as e:
        print(f"[sintetizar_resposta_web] erro: {e}")
        return ""
    if not texto:
        return ""
    fontes_md = "\n".join(f"- {f}" for f in dict.fromkeys(d.metadata["source"] for d in documentos))
    return f"{texto}\n\n---\n**Fontes consultadas:**\n{fontes_md}"


def pesquisar_na_web(llm, pergunta: str) -> str:
    """Pipeline completo de fallback web: query -> Serper -> extração -> síntese."""
    query = criar_query_de_busca(llm, pergunta)
    print(f"[pesquisar_na_web] buscando por: {query}")

    resultados = buscar_serper(query, max_results=5)
    if not resultados:
        return "Tentei buscar na web, mas não encontrei resultados relevantes."

    urls = [r["link"] for r in resultados]
    documentos = extrair_conteudo_urls(urls, resultados)

    if not documentos:
        snippets = "\n\n".join(f"Título: {r['title']}\nFonte: {r['link']}\nResumo: {r['snippet']}" for r in resultados)
        documentos = [Document(page_content=snippets, metadata={"source": "Google Search Snippets"})]

    resposta = sintetizar_resposta_web(llm, documentos, pergunta)
    return resposta or "Encontrei resultados, mas não consegui extrair uma resposta clara."