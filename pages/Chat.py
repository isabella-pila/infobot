import os
import re
import time
import uuid
import httpx
import asyncio
import sys
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import fitz
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from chat_db import salvar_chat, listar_chats, carregar_chat, delete_chat
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from utils import tratar_erro_api
from langchain_openai import ChatOpenAI


# -----------------------------
# Inicialização
# -----------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Assistente CEFET-MG", page_icon="✨")

if not GOOGLE_API_KEY or not SERPER_API_KEY:
    st.error("Chaves de API não encontradas. Verifique seu arquivo .env.")
    st.stop()

serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

# -----------------------------
# Funções auxiliares — PDFs
# -----------------------------
def get_pdf_text(pdf_paths):
    text = ""
    for pdf in pdf_paths:
        try:
            reader = PdfReader(pdf) if not isinstance(pdf, str) else PdfReader(open(pdf, "rb"))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            st.warning(f"⚠️ Não foi possível ler o PDF: {e}")
    return text


def get_text_chunks(text):
    # RecursiveCharacterTextSplitter + chunk maior evita partir listas/tabelas
    # (ex: disciplinas de um período) no meio, o que cortava informação na resposta.
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
    )
    return splitter.split_text(text)


def get_vectorstore(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview", google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        tratar_erro_api("Google Embeddings", e)
        st.stop()
    return FAISS.from_texts(chunks, embedding=embeddings)


def carregar_vectorstore_default():
    path = "perguntas2.pdf"
    if not os.path.exists(path):
        st.error("Arquivo padrão 'perguntas2.pdf' não encontrado!")
        return None
    with fitz.open(path) as doc:
        text = "".join(page.get_text("text") for page in doc)
    chunks = get_text_chunks(text)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview", google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        tratar_erro_api("Google Embeddings", e)
        st.stop()
    return FAISS.from_texts(chunks, embedding=embeddings)


# -----------------------------
# RAG Prompts
# -----------------------------
RAG_PROMPT = """
# DIRETIVA PRINCIPAL
Você é o "Infobot", o assistente virtual do CEFET-MG, campus Varginha. Sua identidade é a de um colega experiente de Sistemas de Informação, que domina os assuntos do curso e está sempre disposto a auxiliar os estudantes com clareza e objetividade.

Sua missão é traduzir materiais de estudo complexos em resumos e explicações didáticas e fáceis de compreender.
sempre traga as informações de forma organizada.
---
# PERSONA E TOM DE VOZ

    quando for sobre matriz curricular ou materias sempre mande todas sem exeção 
1.  **Linguagem:** Comunique-se de forma clara, objetiva e encorajadora. seja legal pode usar algumas girias mas não exagere 
2.  **Atitude:** Mantenha sempre uma postura positiva, paciente e motivadora. O objetivo é fazer com que o estudante se sinta confiante em sua capacidade de aprender qualquer tópico.

---
# DIRETRIZES DE EXECUÇÃO

1.  **Fonte de Verdade Primária:** Sua primeira e principal fonte de informação é sempre o conteúdo fornecido no `{{context}}` (os arquivos que o usuário enviou).

2.  **Análise Estruturada (O Ponto-Chave):** Ao receber um material no `{{context}}` para resumir ou explicar, sua primeira ação é analisá-lo para identificar os seguintes pontos:
    * **Conceitos-chave e Definições:** Quais são os termos técnicos centrais?
    * **Tópicos Principais:** Quais são os grandes blocos de assunto?
    * **Exemplos e Analogias:** O texto utiliza casos práticos para ilustrar a teoria?
    * **Argumento Central:** Qual é a ideia principal que o texto defende ou explica?
    * **Processos ou Fórmulas:** Existe um passo a passo, algoritmo, código ou fórmula?
    * **Conclusões:** Qual é o fechamento ou o resultado principal do assunto?

3.  **Como Criar RESUMOS Eficazes:**
    * **Completo e Detalhado:** Utilize a análise do passo anterior como um roteiro. Seu resumo **deve obrigatoriamente** abordar todos os tópicos e conceitos-chave identificados no material. Não deixe nada de fora.
    * **Organização é Tudo:** Empregue títulos, subtítulos e listas (`bullet points`) para segmentar o conteúdo, melhorando a clareza e a leitura. Evite parágrafos muito longos.
    * **O objetivo final:** Ao final da leitura, o estudante deve ter uma compreensão completa e bem estruturada do material original.

4.  **Quando o Contexto for Insuficiente (Busca na Web):**
    * Se a resposta não for encontrada no material fornecido, você está autorizado a realizar uma busca externa na internet.
    * **Sinalize a Busca:** Sempre informe ao usuário que a informação veio de fora. Por exemplo: *"No material que você enviou não encontrei detalhes sobre isso. Fazendo uma pesquisa adicional, descobri que..."*.
    * **Mantenha o Foco:** A busca externa deve ser usada apenas para complementar o assunto da pergunta, sem fugir do contexto acadêmico.

---
# REGRAS DE INTERAÇÃO

* **Comunicação Direta:** Não há necessidade de saudações ("Olá", "Bom dia") a cada nova pergunta. Mantenha a conversa fluida.
* **Agradecimentos:** Se o usuário agradecer com "obrigado(a)" ou similar, responda de forma educada e prestativa, com algo como "De nada! Se precisar de mais alguma coisa, é só chamar." ou "Disponha!".
* **Proatividade:** Se a pergunta do estudante for muito ampla ("me explica sobre Redes de Computadores"), ajude-o a especificar. Exemplo: *"Claro! Redes é um assunto bem grande. Para te ajudar melhor, você quer saber sobre os modelos de camadas como OSI/TCP, tipos de topologia, ou talvez exemplos de protocolos como HTTP e DNS?"*.
**Quando a pessoa so cumprimentar ("ola", "bom dia") cumprimentar também e so isso

Histórico da conversa:
{chat_history}

Contexto:
{context}

Pergunta:
{question}
"""

CONDENSE_PROMPT = """
Dado o histórico da conversa e a nova pergunta do usuário,
reescreva a pergunta tornando-a COMPLETA e INDEPENDENTE,
incluindo todas as informações relevantes do histórico.

IMPORTANTE:
- Incorpore dados anteriores do usuário (ex: horas cursadas, período, situação)
- NÃO ignore o contexto
- NÃO resuma demais
- A pergunta final deve fazer sentido sozinha

Histórico:
{chat_history}

Pergunta atual:
{question}

Pergunta reescrita completa:
"""


def criar_chain(vectorstore, mensagens_anteriores=None):
    try:
    
        '''llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7, 
            google_api_key=GOOGLE_API_KEY
        )'''
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
        )

    except Exception as e:
        tratar_erro_api("Gemini", e)
        st.stop()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if mensagens_anteriores:
        for msg in mensagens_anteriores:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    question_generator = LLMChain(
        llm=llm, prompt=PromptTemplate.from_template(CONDENSE_PROMPT)
    )
    combine_docs_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=ChatPromptTemplate.from_template(RAG_PROMPT),
    )

    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        memory=memory,
    )


# ═══════════════════════════════════════════════════════════
# BUSCA WEB 
# ═══════════════════════════════════════════════════════════

QUERY_TEMPLATE = """
Você é um especialista em pesquisa na internet. Sua tarefa é transformar a pergunta do usuário em
uma query de busca eficaz para o Google.

Regras:
- Use apenas as palavras-chave essenciais (3 a 7 palavras)
- Inclua o termo "CEFET-MG" SOMENTE se a pergunta for claramente sobre a instituição
  (ex: matrícula, calendário, curso, câmpus, processo seletivo, etc.)
- Para perguntas técnicas/acadêmicas (ex: algoritmos, banco de dados, redes), NÃO inclua "CEFET-MG"
- Prefira termos em português, mas use inglês se o assunto for técnico
- Não adicione aspas, parênteses ou operadores booleanos
- Se for algo como iniciação cinetifica ou projetos de extensão procuro adicione CEFET-MG Varginha

Exemplos:
  Pergunta: "Como funciona o processo de matrícula no CEFET?"
  Query: matrícula CEFET-MG Varginha 2025

  Pergunta: "O que é normalização em banco de dados?"
  Query: normalização banco de dados formas normais explicação

  Pergunta: "Quais são as disciplinas do curso de SI?"
  Query: CEFET-MG Varginha Sistemas de Informação matriz curricular disciplinas

  Se perguntar por evento tente colocar as informação atualizada como o ano atual ex 2026

Pergunta do usuário: "{pergunta}"
Query de busca (apenas o texto, sem explicações):
"""


def criar_query_de_busca(llm, pergunta: str) -> str:
    """Gera uma query de busca inteligente e contextualizada."""
    prompt = PromptTemplate.from_template(QUERY_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        resultado = chain.invoke({"pergunta": pergunta})
        query = resultado.get("text", pergunta).strip().strip('"').strip("'")
        query = re.sub(r"[`*_]", "", query).strip()
        return query if query else pergunta
    except Exception as e:
        st.warning(f"Erro ao gerar query: {e}")
        palavras = [p for p in pergunta.split() if len(p) > 3]
        return " ".join(palavras[:6])


DOMINIOS_BLOQUEADOS = {
    "pinterest.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com",
    "slideshare.net",
}


def buscar_serper(query: str, max_results: int = 5) -> list[dict]:
    """Busca no Serper e retorna resultados filtrados e enriquecidos."""
    try:
        res = serper.results(query)
    except Exception as e:
        tratar_erro_api("Serper", e)
        return []

    resultados = []
    for r in res.get("organic", [])[:max_results + 3]:
        link = r.get("link", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")

        if not link or not title:
            continue

        dominio = re.sub(r"https?://(www\.)?", "", link).split("/")[0]
        if any(bloqueado in dominio for bloqueado in DOMINIOS_BLOQUEADOS):
            continue

        resultados.append({"title": title, "link": link, "snippet": snippet, "dominio": dominio})

        if len(resultados) >= max_results:
            break

    return resultados


TAGS_UTEIS = ["p", "h1", "h2", "h3", "h4", "li", "td", "th", "article", "section"]

RUIDO_REGEX = re.compile(
    r"(cookie|política de privacidade|todos os direitos|compartilhe|veja também"
    r"|publicidade|assine|newsletter|subscribe|menu|nav|sidebar|footer)",
    re.IGNORECASE,
)

MAX_CHARS_POR_SITE = 4000
TIMEOUT_SEGUNDOS = 10

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

HEADERS_BROWSER = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


def _extrair_texto_html(html: str) -> str:
    """Extrai texto limpo de HTML usando BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    partes = []
    for tag in soup.find_all(TAGS_UTEIS):
        texto = tag.get_text(separator=" ", strip=True)
        if len(texto) < 30:
            continue
        if RUIDO_REGEX.search(texto):
            continue
        partes.append(texto)

    return "\n".join(partes)[:MAX_CHARS_POR_SITE]


def _tentar_fetch(client: httpx.Client, url: str, user_agent: str) -> str | None:
    """Tenta buscar uma URL com um User-Agent específico. Retorna HTML ou None."""
    headers = {**HEADERS_BROWSER, "User-Agent": user_agent}
    try:
        resp = client.get(url, headers=headers)
        if resp.status_code == 200 and "html" in resp.headers.get("content-type", ""):
            return resp.text
    except Exception:
        pass
    return None


def _buscar_google_cache(url: str) -> str | None:
    """Tenta buscar a versão em cache do Google como último recurso."""
    cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
    try:
        with httpx.Client(timeout=TIMEOUT_SEGUNDOS, follow_redirects=True) as client:
            headers = {
                "User-Agent": USER_AGENTS[0],
                "Accept": "text/html",
                "Accept-Language": "pt-BR,pt;q=0.9",
            }
            resp = client.get(cache_url, headers=headers)
            if resp.status_code == 200:
                return resp.text
    except Exception:
        pass
    return None


def extrair_conteudo_urls(urls: list[str], resultados_serper: list[dict]) -> list[Document]:
    """
    Estratégia em 3 camadas para extrair conteúdo:
      1. httpx com rotação de User-Agent
      2. Cache do Google
      3. Snippet do Serper como fallback individual por URL
    """
    snippets_por_url = {r["link"]: r for r in resultados_serper}
    documentos = []

    with httpx.Client(timeout=TIMEOUT_SEGUNDOS, follow_redirects=True) as client:
        for i, url in enumerate(urls):
            html = None
            texto = ""

            for ua in USER_AGENTS:
                html = _tentar_fetch(client, url, ua)
                if html:
                    break
                time.sleep(0.2)

            if html:
                texto = _extrair_texto_html(html)

            if len(texto) < 100:
                html_cache = _buscar_google_cache(url)
                if html_cache:
                    texto = _extrair_texto_html(html_cache)

            if len(texto) < 100:
                info = snippets_por_url.get(url, {})
                snippet = info.get("snippet", "")
                title = info.get("title", "")
                if snippet:
                    texto = f"{title}\n{snippet}"

            if len(texto) >= 50:
                documentos.append(Document(page_content=texto, metadata={"source": url}))

            time.sleep(0.3)

    return documentos


SINTESE_TEMPLATE = """
Você é um assistente acadêmico especializado. Com base EXCLUSIVAMENTE nas informações abaixo
coletadas da web, responda à pergunta do usuário de forma clara, organizada e precisa.

Diretrizes:
- Responda em português
- Use bullet points ou subtítulos quando houver múltiplos pontos
- Cite a fonte (URL) ao final de cada informação relevante quando possível
- Se as fontes apresentarem dados conflitantes, mencione as duas versões
- Se as informações coletadas forem insuficientes para responder, diga claramente
- NÃO invente informações que não estejam nas fontes

─────────────────────────────────────
Conteúdo coletado da web:
{context}
─────────────────────────────────────

Pergunta: {question}

Resposta:
"""


def sintetizar_resposta_web(llm, documentos: list[Document], pergunta: str) -> str:
    """Sintetiza os documentos coletados em uma resposta coesa."""
    if not documentos:
        return ""

    prompt = ChatPromptTemplate.from_template(SINTESE_TEMPLATE)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    try:
        resultado = chain.invoke({
            "input_documents": documentos,
            "question": pergunta,
            "context": "\n\n---\n\n".join(
                f"Fonte: {doc.metadata['source']}\n{doc.page_content}"
                for doc in documentos
            ),
        })
        texto = resultado.get("output_text", "").strip()
    except Exception as e:
        st.warning(f"Erro ao sintetizar resposta: {e}")
        return ""

    if not texto:
        return ""

    fontes_unicas = list(dict.fromkeys(doc.metadata["source"] for doc in documentos))
    fontes_md = "\n".join(f"- {f}" for f in fontes_unicas)
    return f"{texto}\n\n---\n**🔗 Fontes consultadas:**\n{fontes_md}"


def pesquisar_na_web(llm, pergunta: str) -> str:
    """
    Pipeline completo:
      1. Gera query inteligente
      2. Busca no Serper
      3. Extrai conteúdo das páginas via httpx
      4. Sintetiza com LLM
    """
    query = criar_query_de_busca(llm, pergunta)
    st.info(f"🔍 **Buscando por:** `{query}`")

    resultados = buscar_serper(query, max_results=5)
    if not resultados:
        return "Tentei buscar na web, mas não encontrei resultados relevantes para sua pergunta."

    with st.expander("🔗 Fontes encontradas", expanded=False):
        for r in resultados:
            st.markdown(f"**[{r['title']}]({r['link']})**  \n_{r['snippet']}_")

    with st.spinner("📄 Lendo o conteúdo das páginas..."):
        urls = [r["link"] for r in resultados]
        documentos = extrair_conteudo_urls(urls, resultados)

    if not documentos:
        st.warning("Nenhuma página retornou conteúdo. Usando resumos da busca.")
        snippets = "\n\n".join(
            f"Título: {r['title']}\nFonte: {r['link']}\nResumo: {r['snippet']}"
            for r in resultados if r.get("snippet")
        )
        documentos = [Document(
            page_content=snippets,
            metadata={"source": "Google Search Snippets"}
        )]

    with st.spinner("🧠 Sintetizando informações..."):
        resposta = sintetizar_resposta_web(llm, documentos, pergunta)

    return resposta or "Encontrei resultados, mas não consegui extrair uma resposta clara. Veja as fontes acima."


# ═══════════════════════════════════════════════════════════
# Estado inicial e autenticação
# ═══════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = None
if "chat_name" not in st.session_state:
    st.session_state.chat_name = ""

if "auth_status" not in st.session_state or not st.session_state.auth_status:
    st.warning("🔐 Faça login para acessar.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔐 Login"):
            st.switch_page("pages/Login.py")
    st.stop()

username = st.session_state.username

st.title("🎓 CEFET-MG - Assistente Virtual")

# ── Sidebar ──────────────────────────────────────────────

with st.sidebar:
    st.subheader("📌 Envie seus PDFs")
    pdf_docs = st.file_uploader("Carregue PDFs", accept_multiple_files=True)

    if st.button("📄 Processar PDFs"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.pdf_paths = None
        st.session_state.chat_name = ""

        if not pdf_docs:
            vectorstore = carregar_vectorstore_default()
        else:
            texto = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(texto)
            vectorstore = get_vectorstore(chunks)
            os.makedirs("uploads", exist_ok=True)
            pdf_paths = []
            for pdf in pdf_docs:
                unique_name = f"{username}_{uuid.uuid4().hex}.pdf"
                path = os.path.join("uploads", unique_name)
                with open(path, "wb") as f:
                    f.write(pdf.getbuffer())
                pdf_paths.append(path)
            st.session_state.pdf_paths = pdf_paths

        if vectorstore:
            st.session_state.chain = criar_chain(vectorstore)
            st.success("Base carregada!")

    if st.button("🧹 Novo chat"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.pdf_paths = None
        st.session_state.chat_name = ""
        vectorstore_default = carregar_vectorstore_default()
        if vectorstore_default:
            st.session_state.chain = criar_chain(vectorstore_default)
        st.rerun()

    st.subheader("📅 Seus chats salvos")
    chats = listar_chats(username)
    if chats:
        chat_escolhido = st.selectbox("Escolha um chat", chats, format_func=lambda x: x[1])

        if st.button("🔄 Carregar chat"):
            st.session_state.messages, st.session_state.pdf_paths = carregar_chat(chat_escolhido[0])
            st.session_state.chat_name = chat_escolhido[1]
            if st.session_state.pdf_paths and all(
                os.path.exists(p) for p in st.session_state.pdf_paths
            ):
                texto = get_pdf_text(st.session_state.pdf_paths)
                chunks = get_text_chunks(texto)
                vectorstore = get_vectorstore(chunks)
            else:
                st.warning("📂 PDFs não encontrados. Carregando base padrão.")
                vectorstore = carregar_vectorstore_default()
            if vectorstore:
                st.session_state.chain = criar_chain(vectorstore, st.session_state.messages)
                st.success("Chat carregado!")

        if st.button("🗑️ Apagar chat"):
            delete_chat(chat_escolhido[0])
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.pdf_paths = None
            st.session_state.chat_name = ""
            st.success("Chat apagado com sucesso!")


# ── Inicialização da chain padrão ────────────────────────

if st.session_state.chain is None:
    with st.spinner("Carregando base padrão..."):
        vectorstore = carregar_vectorstore_default()
        if vectorstore:
            st.session_state.chain = criar_chain(vectorstore)


# ── Histórico de chat ────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ═══════════════════════════════════════════════════════════
# Detecção de resposta insuficiente
# ═══════════════════════════════════════════════════════════

def resposta_eh_insuficiente(texto: str) -> bool:
    """
    Detecta se o RAG não encontrou a informação no material,
    usando palavras-chave parciais para cobrir qualquer variação
    que o LLM possa gerar.
    """
    texto_lower = texto.lower()
    gatilhos = [
        # Negações diretas
        "não há informações",
        "não encontrei",
        "não há menção",
        "não tem menção",
        "não achei",
        "não consta",
        "não há registro",
        "não sei",
        "não tenho",
        "não vi informação",
        # Referências ao material
        "no material",          # cobre: "no material fornecido", "no material enviado", etc.
        "no pdf",
        # Redirecionamentos externos
        "sugiro verificar",
        "você pode verificar",
        "você pode conferir",
        "você pode consultar",
        "acesse o site",
        "visite o site",
        "confira no site",
        "no site oficial",
        "entrar em contato",
        "consulte a coordenação",
        "entre em contato",
    ]
    return any(g in texto_lower for g in gatilhos)


# ═══════════════════════════════════════════════════════════
# Interação principal
# ═══════════════════════════════════════════════════════════

if user_input := st.chat_input("Digite sua pergunta"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        final_bot_msg = ""

        # 1. Responde via RAG
        with st.spinner("Pensando..."):
            if st.session_state.chain:
                response = st.session_state.chain.invoke({"question": user_input})
                rag_response = response.get("answer", "Não consegui gerar uma resposta.")
            else:
                rag_response = "A base de documentos não está carregada."

        final_bot_msg = rag_response

        # 2. Fallback para busca web se a resposta for insuficiente
        # ✅ CORREÇÃO: usa função robusta no lugar da lista de strings exatas
        #    e remove a condição "sem_pdf_usuario" para funcionar com o PDF padrão também
        resposta_insuficiente = resposta_eh_insuficiente(rag_response)

        if resposta_insuficiente:
            try:
                llm_web = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    openai_api_key=OPENAI_API_KEY,
                )
                resposta_web = pesquisar_na_web(llm_web, user_input)
                if resposta_web:
                    final_bot_msg = resposta_web
            except Exception as e:
                st.warning(f"Erro na busca web: {e}")
                # mantém a resposta RAG original

        st.markdown(final_bot_msg)

    st.session_state.messages.append({"role": "assistant", "content": final_bot_msg})

    if not st.session_state.chat_name:
        st.session_state.chat_name = f"Chat - {user_input[:30]}"

    salvar_chat(
        username=username,
        chat_name=st.session_state.chat_name,
        messages=st.session_state.messages,
        pdf_paths=st.session_state.get("pdf_paths", []),
    )