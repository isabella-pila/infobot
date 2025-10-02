import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
import fitz
import os

st.set_page_config(page_title="CEFET - Chat sobre o Cefet", page_icon="🎓")

# Botões no topo
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔐 Login"):
        st.switch_page("pages/Login.py")


# Carrega as variáveis de ambiente
_ = load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def extrai_texto_para_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

@st.cache_resource
def load_pdf_data():    
    pdf_path = "perguntas2.pdf"
    if not os.path.exists(pdf_path):
        st.error("Arquivo PDF não encontrado!")
        return None
    
    texto_extraido = extrai_texto_para_pdf(pdf_path)
    document = Document(page_content=texto_extraido, metadata={"source": pdf_path})
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents([document], embeddings)
    return vectorstore.as_retriever()

retriever = load_pdf_data()

st.title("🎓 CEFET-MG - Assistente Virtual")

prompt = ChatPromptTemplate.from_template("""
Você é um atendente virtual amigável e prestativo de uma faculdade chamada CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) no campus de Varginha.
Seu trabalho é fornecer informações sobre o curso de Sistemas de Informação de maneira educada, simpática e clara.
Consultando as informações extraídas do texto, sempre seja organizado e detalhado.
Sempre seja gentil ao responder.
Contexto: {context}
Pergunta do cliente: {question}

""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Digite sua dúvida"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response_stream = chain.stream(user_input)
    full_response = ""

    with st.chat_message("assistant"):
        response_box = st.empty()
        for partial in response_stream:
            full_response += str(partial.content)
            response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
