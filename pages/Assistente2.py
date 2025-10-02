# main.py
import os
import uuid
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from chat_db import salvar_chat, listar_chats, carregar_chat, delete_chat
import fitz

# 🔎 DuckDuckGo
from ddgs import DDGS


def buscar_duckduckgo(query: str, max_results: int = 3):
    """Busca informações no DuckDuckGo."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({"title": r["title"], "link": r["href"], "snippet": r["body"]})
    return results


def preparar_query_externa(pergunta: str) -> str:
    """Limpa a pergunta e adiciona contexto de busca de forma inteligente."""
    saudacoes = ["ola", "olá", "oi", "e aí", "bom dia", "boa tarde", "boa noite"]
    pergunta_proc = pergunta.lower()
    for s in saudacoes:
        if pergunta_proc.startswith(s):
            pergunta_proc = pergunta_proc.replace(s, "", 1).strip(",. ")

    # 🔍 Se mencionar termos institucionais → força site oficial
    termos_institucionais = ["cefet", "professor", "disciplina", "matrícula", "editais", "curso", "varginha"]
    if any(t in pergunta_proc for t in termos_institucionais):
        if "cefet" not in pergunta_proc:
            pergunta_proc += " CEFET-MG Varginha site oficial"
        else:
            pergunta_proc = pergunta_proc.replace("cefet", "CEFET-MG Varginha") + " site oficial"

    return pergunta_proc


def ler_pagina(url: str, max_chars: int = 2000) -> str:
    """Extrai texto limpo priorizando conteúdo relevante."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove elementos irrelevantes
        for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script.extract()

        # Pega o conteúdo principal (se existir)
        main = soup.find("main") or soup.find("article")
        texto = main.get_text(separator=" ", strip=True) if main else soup.get_text(separator=" ", strip=True)

        return texto[:max_chars] + "..."
    except Exception as e:
        return f"⚠️ Erro ao acessar {url}: {e}"


def resumir_resultados(llm, resultados, pergunta):
    """Usa o LLM para resumir resultados da web em uma resposta única e clara."""
    docs = []
    for r in resultados:
        conteudo = ler_pagina(r["link"])
        docs.append(Document(page_content=conteudo, metadata={"source": r["link"]}))

    if not docs:
        return "❌ Não encontrei conteúdo relevante nos resultados da web."

    prompt_resumo = ChatPromptTemplate.from_template("""
    Você é o assistente do CEFET-MG. Use os textos a seguir para responder a pergunta do aluno.
    Resuma de forma clara, cite as fontes quando útil e dê a resposta como se fosse uma conversa rápida.
    mostre o link da pesquisa 
    Contexto:
    {context}

    Pergunta: {question}
    """)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_resumo)
    resposta = chain.invoke({"input_documents": docs, "question": pergunta})
    return resposta["output_text"]


st.set_page_config(page_title="Assistente PDF", page_icon="📄")
load_dotenv()

# =====================
# 📂 Funções auxiliares
# =====================
def get_pdf_text(pdf_paths):
    text = ""
    for pdf in pdf_paths:
        reader = PdfReader(pdf) if not isinstance(pdf, str) else PdfReader(open(pdf, "rb"))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(chunks, embedding=embeddings)

def carregar_vectorstore_default():
    path = "perguntas2.pdf"
    if not os.path.exists(path):
        st.error("Arquivo padrão 'perguntas2.pdf' não encontrado!")
        return None
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text("text") + "\n"
    document = Document(page_content=text, metadata={"source": path})
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_documents([document], embeddings)

# =====================
# 📌 Prompts
# =====================
RAG_PROMPT = """
# DIRETIVA PRINCIPAL

Você é o "Infobot", o assistente virtual gente boa do CEFET-MG, campus Varginha. Sua vibe é de um veterano de Sistemas de Informação que manja muito da matéria e tá sempre disposto a ajudar os calouros e a galera do curso.

Sua missão é transformar materiais de estudo complexos em resumos e explicações fáceis de entender.

---
# PERSONA E TOM DE VOZ

1.  **Linguagem:** Comunique-se de forma clara, moderna e acessível. A vibe é de uma conversa no WhatsApp: direta, útil e sem formalidades desnecessárias. Pode usar gírias como "tipo", "mano", "se liga", "top", "pegar a visão", mas sem exagerar pra não parecer forçado.
2.  **Atitude:** Seja sempre positivo, encorajador e paciente. A meta é fazer o aluno se sentir inteligente e capaz de aprender qualquer coisa.

---
# DIRETRIZES DE EXECUÇÃO

1.  **Fonte de Verdade:** Sua fonte primária de informação é sempre o que for fornecido no `{context}`.
2.  **Quando o Contexto não Ajudar:** Se a resposta não estiver no material, você está autorizado a usar seu conhecimento externo para complementar. **Sempre sinalize isso**, tipo: "Olha, no material não fala sobre isso, mas pesquisando aqui por fora, a visão é que...".
3.  **Análise Estruturada (O PULO DO GATO):** Ao receber um material no `{context}` para resumir ou explicar, sua primeira ação é "escanear" o conteúdo e identificar os seguintes pontos:
    * **Conceitos-chave e Definições:** Quais são os termos mais importantes?
    * **Tópicos Principais:** Quais são os grandes blocos de assunto?
    * **Exemplos e Analogias:** O texto usa exemplos práticos para ilustrar a teoria?
    * **Argumento Central:** Qual é a ideia principal que o texto quer passar?
    * **Passos ou Fórmulas:** Existe um processo passo a passo, código ou fórmula?
    * **Conclusões:** Qual o fechamento do assunto?

4.  **Como Criar RESUMOS (Sem deixar nada de fora):**
    * Use a análise do passo anterior como um checklist. Seu resumo **precisa** cobrir todos os pontos que você identificou.
    * **Organize a informação:** Use títulos, subtítulos e `bullet points` (listas com marcadores) para quebrar o conteúdo em partes menores e mais fáceis de digerir. Ninguém gosta de textão.
    * O objetivo é que o aluno leia seu resumo e sinta que "pegou a visão" completa do material.

5.  **Como Criar EXPLICAÇÕES (Didática Nível Mestre):**
    * Quando a `{question}` pedir para explicar algo, transforme o "tecniquês" em "português".
    * Use **analogias com o dia a dia**, games, séries ou o universo de tecnologia para explicar conceitos difíceis.
    * Se for um processo, quebre a explicação em um **passo a passo numerado**.
    * A meta é que até quem está "boiando" na matéria consiga entender de primeira.

---
# REGRAS DE INTERAÇÃO

* **Sem Saudações Repetitivas:** Não precisa de "Olá!" ou "Bom dia" a cada pergunta. Mantenha a conversa fluida e direta.
* **Seja Proativo:** Se a pergunta do aluno for muito vaga, ajude a refinar. Exemplo: Se ele perguntar "me explica sobre POO", você pode responder: "Claro! Pra eu te dar a melhor explicação, o que você quer saber exatamente? Os pilares tipo Herança e Polimorfismo, ou um exemplo prático de como funciona?".

Contexto: {context}
Pergunta: {question}
"""

CONDENSE_PROMPT = """
Dado o histórico da conversa e uma nova pergunta, reescreva como pergunta independente.

Histórico:
{chat_history}

Pergunta: {question}

Pergunta independente:
"""

def criar_chain(vectorstore, mensagens_anteriores=None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8) 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if mensagens_anteriores:
        for msg in mensagens_anteriores:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    condense_prompt = PromptTemplate.from_template(CONDENSE_PROMPT)
    question_generator = LLMChain(llm=llm, prompt=condense_prompt)
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        memory=memory,
    )

# =====================
#  Autenticação + Estado
# =====================


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

with st.sidebar:
    st.header("📌 Envie seus documentos PDF")
    pdf_docs = st.file_uploader("Carregue um ou mais PDFs", accept_multiple_files=True)

    if st.button("📄 Processar PDFs"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.pdf_paths = None
        st.session_state.chat_name = ""

        if not pdf_docs:
            vectorstore = carregar_vectorstore_default()
        else:
            with st.spinner("Processando PDFs..."):
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
        st.rerun() # Use rerun para limpar a tela

    st.divider()

    st.subheader("📅 Seus chats salvos")
    chats = listar_chats(username)
    if chats:
        chat_escolhido = st.selectbox("Escolha um chat", chats, format_func=lambda x: x[1])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Carregar chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chain = None
                st.session_state.pdf_paths = None
                mensagens_salvas, pdf_paths = carregar_chat(chat_escolhido[0])
                st.session_state.messages = mensagens_salvas
                st.session_state.pdf_paths = pdf_paths
                st.session_state.chat_name = chat_escolhido[1]

                if pdf_paths and all(os.path.exists(p) for p in pdf_paths):
                    texto = get_pdf_text(pdf_paths)
                    chunks = get_text_chunks(texto)
                    vectorstore = get_vectorstore(chunks)
                else:
                    vectorstore = carregar_vectorstore_default()

                if vectorstore:
                    st.session_state.chain = criar_chain(vectorstore, mensagens_salvas)
                    st.success("Chat carregado!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao carregar vectorstore.")

        with col2:
            if st.button("🗑️ Apagar chat", use_container_width=True):
                delete_chat(chat_escolhido[0])
                st.session_state.messages = []
                st.session_state.chain = None
                st.session_state.pdf_paths = None
                st.session_state.chat_name = ""
                st.success("Chat apagado!")
                st.rerun()

if st.session_state.chain is None:
    if not st.session_state.pdf_paths:
        with st.spinner("Carregando base padrão..."):
            vectorstore = carregar_vectorstore_default()
            if vectorstore:
                st.session_state.chain = criar_chain(vectorstore)
            else:
                st.error("❌ Não foi possível carregar a base padrão.")


# =====================
# 💬 Interface principal
# =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Digite sua pergunta"):
    if not st.session_state.chain:
        st.warning("📄 Carregue um PDF ou base padrão antes de perguntar.")
    else:
        # Exibe a pergunta do usuário imediatamente
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepara para receber e exibir a resposta do assistente
        with st.chat_message("assistant"):
            # Usa um spinner enquanto a resposta é gerada
            with st.spinner("Digitando..."):
                # Cria um placeholder que será atualizado com o streaming
                message_placeholder = st.empty()
                full_response = ""
                
                # Usa .stream() em vez de .invoke()
                stream = st.session_state.chain.stream({"question": user_input})

                for chunk in stream:
                    # 'answer' é a chave que geralmente contém o texto no stream da chain
                    if 'answer' in chunk:
                        full_response += chunk['answer']
                        # Atualiza o placeholder com a resposta parcial + um cursor
                        message_placeholder.markdown(full_response + "▌")
                
                # Atualiza o placeholder com a resposta final sem o cursor
                message_placeholder.markdown(full_response)
        
        bot_msg = full_response # Guarda a resposta completa

        # 🚀 Lógica de fallback (busca externa)
        gatilhos_fallback = [
            "não tenho informações", "não encontrado", "não localizei",
            "não encontrei", "não está no documento"
        ]

        if (not bot_msg) or any(frase in bot_msg.lower() for frase in gatilhos_fallback):
            with st.chat_message("assistant"):
                with st.spinner("Não achei nos documentos, buscando na web..."):
                    try:
                        query_externa = preparar_query_externa(user_input)
                        resultados = buscar_duckduckgo(query_externa, max_results=3)

                        if resultados:
                            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
                            resumo = resumir_resultados(llm, resultados[:2], user_input)
                            bot_msg = resumo
                            st.markdown(bot_msg)
                        else:
                            st.warning("❌ Não encontrei resultados relevantes na web.")
                    except Exception as e:
                        bot_msg += f"\n\n⚠️ Erro ao buscar externamente: {e}"
                        st.error(f"Erro ao buscar externamente: {e}")


        # Adiciona a resposta final (do RAG ou do fallback) ao histórico
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        # Salva o chat no banco de dados
        if not st.session_state.chat_name:
            base = user_input.strip().split("\n")[0][:30]
            st.session_state.chat_name = f"Chat - {base}"

        salvar_chat(
            username=username,
            chat_name=st.session_state.chat_name,
            messages=st.session_state.messages,
            pdf_paths=st.session_state.pdf_paths,
        )