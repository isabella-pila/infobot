# main.py
import os
import uuid
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

st.set_page_config(page_title="Assistente PDF", page_icon="📄")
load_dotenv()

# Funções auxiliares
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

# Prompts
RAG_PROMPT = """
Você é um atendente virtual amigável e prestativo de uma faculdade chamada CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) no campus de Varginha.
Você vai atender um público de universitários e pré-universitários da geração Z, portanto você deve utilizar um vocabulário voltado para esse público.
Seu trabalho é ajudar os alunos de Sistemas de Informação a aprender a matéria de maneira educada, simpática e clara.
Consultando as informações extraídas do texto, sempre seja organizado e detalhado.
Sempre ao fazer resumos pegue todos os topicos, seja legal.
Sempre seja gentil ao responder.
Você pode usar algumas fontes para pesquisar sem ser do que o aluno mandar mas nada fora do contexto.
Se a pessoa falar obrigada, respoda educadamente de nada.
Não precisa cumprimentar toda vez que o usuario fazer uma pergunta 

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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
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

# Estado inicial
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

# Sidebar
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
        st.success("Novo chat iniciado.")

    st.divider()

    st.subheader("📅 Seus chats salvos")
    chats = listar_chats(username)
    if chats:
        chat_escolhido = st.selectbox("Escolha um chat", chats, format_func=lambda x: x[1])

        if st.button("🔄 Carregar chat"):
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
                st.warning("📂 PDF(s) não encontrado(s). Usando base padrão.")
                vectorstore = carregar_vectorstore_default()

            if vectorstore:
                st.session_state.chain = criar_chain(vectorstore, mensagens_salvas)
                st.success("Chat carregado com sucesso!")
            else:
                st.error("❌ Erro ao carregar vectorstore.")

        if st.button("🗑️ Apagar chat"):
            delete_chat(chat_escolhido[0])
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.pdf_paths = None
            st.session_state.chat_name = ""
            st.success("Chat apagado com sucesso!")

# 🚀 Auto-carregamento da base padrão no início
if st.session_state.chain is None:
    if not st.session_state.pdf_paths:
        with st.spinner("Carregando base padrão..."):
            vectorstore = carregar_vectorstore_default()
            if vectorstore:
                st.session_state.chain = criar_chain(vectorstore)
                
            else:
                st.error("❌ Não foi possível carregar a base padrão.")

# Interface principal
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Digite sua pergunta"):
    if not st.session_state.chain:
        st.warning("📄 Carregue um PDF ou base padrão antes de perguntar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        resposta = st.session_state.chain.invoke({"question": user_input})
        bot_msg = resposta["answer"]

        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

        if not st.session_state.chat_name:
            base = user_input.strip().split("\n")[0][:30]
            st.session_state.chat_name = f"Chat- {base}"

        salvar_chat(
            username=username,
            chat_name=st.session_state.chat_name,
            messages=st.session_state.messages,
            pdf_paths=st.session_state.pdf_paths,
        )
