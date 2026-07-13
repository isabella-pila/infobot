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
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    # RecursiveCharacterTextSplitter respeita a hierarquia do texto (parágrafos ->
    # linhas -> palavras), então evita partir uma lista/tabela no meio. chunk_size
    # maior mantém a lista de disciplinas de cada período inteira dentro de um chunk.
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
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
eficaz para o Google (4 a 8 palavras-chave, sem aspas e sem operadores booleanos).

Regras:
- PRESERVE as entidades específicas da pergunta: nome do curso, campus, disciplina, evento.
- EXPANDA siglas para o nome por extenso (ex.: "SI" -> "Sistemas de Informação";
  "TCC" -> "Trabalho de Conclusão de Curso").
- Inclua "CEFET-MG" quando a pergunta for institucional (sobre o CEFET, seus cursos,
  eventos, campi ou disciplinas).

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


def buscar_serper_multi(queries: list[str], max_results: int = 5) -> list[dict]:
    """Executa várias variantes de query e mescla os resultados (dedup por link,
    preservando a ordem — variantes mais específicas primeiro)."""
    vistos, mesclados = set(), []
    for q in queries:
        for r in buscar_serper(q, max_results=max_results):
            if r["link"] in vistos:
                continue
            vistos.add(r["link"])
            mesclados.append(r)
    return mesclados[: max_results + 3]


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
Responda à pergunta em português, de forma clara e organizada (use títulos/bullet points
quando fizer sentido), usando as informações abaixo. Elas podem vir da BASE INSTITUCIONAL
do CEFET-MG e/ou de PÁGINAS DA WEB.

Instruções:
- Extraia e combine TODAS as informações relevantes disponíveis, mesmo que respondam apenas
  PARCIALMENTE à pergunta.
- Cite a fonte (URL, ou "base institucional") quando possível.
- Só afirme que não há informação suficiente se REALMENTE nenhum trecho for relevante.

Conteúdo disponível:
{context}

Pergunta: {question}

Resposta:
"""


def sintetizar_resposta_web(llm, documentos: list[Document], pergunta: str,
                            documentos_locais: list[Document] | None = None) -> str:
    """Sintetiza a resposta a partir dos documentos da web e, opcionalmente, do
    contexto local já recuperado do PDF (síntese híbrida). Retorna apenas o texto
    da resposta (sem rodapé de fontes — quem chama decide como exibir as fontes)."""
    docs_locais = list(documentos_locais or [])
    todos = docs_locais + list(documentos)
    if not todos:
        return ""

    def _rotular(d: Document) -> str:
        fonte = d.metadata.get("source", "base institucional")
        origem = "BASE INSTITUCIONAL" if d in docs_locais else "WEB"
        return f"[{origem}] Fonte: {fonte}\n{d.page_content}"

    chain = load_qa_chain(llm, chain_type="stuff", prompt=ChatPromptTemplate.from_template(SINTESE_TEMPLATE))
    try:
        resultado = chain.invoke({
            "input_documents": todos,
            "question": pergunta,
            "context": "\n\n---\n\n".join(_rotular(d) for d in todos),
        })
        return resultado.get("output_text", "").strip()
    except Exception as e:
        print(f"[sintetizar_resposta_web] erro: {e}")
        return ""


def pesquisar_na_web_estruturado(llm, pergunta: str, contexto_local: list[Document] | None = None) -> dict:
    """Pipeline de busca web que retorna resposta E fontes de forma estruturada.

    query -> (variantes, incl. site:cefetmg.br p/ perguntas institucionais) -> Serper
    -> extração httpx -> síntese híbrida (contexto local do PDF + web).

    Retorna: {"resposta": str, "fontes": list[str]}.
    """
    query = criar_query_de_busca(llm, pergunta)

    # (E) Perguntas institucionais ganham uma variante restrita ao domínio oficial,
    # colocada PRIMEIRO para que os resultados do cefetmg.br tenham prioridade no merge.
    institucional = "cefet" in f"{pergunta} {query}".lower()
    variantes = [f"{query} site:cefetmg.br", query] if institucional else [query]
    print(f"[pesquisar_na_web] queries: {variantes}")

    docs_locais = list(contexto_local or [])
    fontes_locais = [d.metadata.get("source", "base institucional") for d in docs_locais]

    resultados = buscar_serper_multi(variantes, max_results=5)
    if not resultados:
        # Sem web: ainda assim tenta responder com o contexto local, se houver.
        if docs_locais:
            texto = sintetizar_resposta_web(llm, [], pergunta, documentos_locais=docs_locais)
            if texto:
                return {"resposta": texto, "fontes": list(dict.fromkeys(fontes_locais))}
        return {"resposta": "Tentei buscar na web, mas não encontrei resultados relevantes.", "fontes": []}

    urls = [r["link"] for r in resultados]
    documentos = extrair_conteudo_urls(urls, resultados)

    if not documentos:
        snippets = "\n\n".join(f"Título: {r['title']}\nFonte: {r['link']}\nResumo: {r['snippet']}" for r in resultados)
        documentos = [Document(page_content=snippets, metadata={"source": "Google Search Snippets"})]

    texto = sintetizar_resposta_web(llm, documentos, pergunta, documentos_locais=docs_locais)
    fontes = list(dict.fromkeys(fontes_locais + [d.metadata.get("source", "") for d in documentos]))
    fontes = [f for f in fontes if f]

    if not texto:
        return {"resposta": "Encontrei resultados, mas não consegui extrair uma resposta clara.", "fontes": fontes}
    return {"resposta": texto, "fontes": fontes}


def pesquisar_na_web(llm, pergunta: str) -> str:
    """Wrapper retrocompatível (retorna string com rodapé de fontes em markdown).

    Mantido para chamadas antigas que esperam uma string. O pipeline agêntico usa
    pesquisar_na_web_estruturado, que devolve resposta e fontes separadamente."""
    res = pesquisar_na_web_estruturado(llm, pergunta)
    resposta, fontes = res["resposta"], res.get("fontes", [])
    if fontes:
        fontes_md = "\n".join(f"- {f}" for f in fontes)
        return f"{resposta}\n\n---\n**Fontes consultadas:**\n{fontes_md}"
    return resposta