# main.py
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import fitz
import asyncio
import sys
import nest_asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from chat_db import salvar_chat, listar_chats, carregar_chat, delete_chat
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import CharacterTextSplitter
from utils import tratar_erro_api

# -----------------------------
# Inicialização
# -----------------------------
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

st.set_page_config(page_title="Assistente CEFET-MG", page_icon="✨")

if not GOOGLE_API_KEY or not SERPER_API_KEY:
    st.error("Chaves de API não encontradas. Verifique seu arquivo .env.")
    st.stop()

serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

# -----------------------------
# Funções auxiliares
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
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

def get_vectorstore(chunks):
    try:
     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
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
    document = Document(page_content=text, metadata={"source": path})
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        tratar_erro_api("Google Embeddings", e)
        st.stop()
    return FAISS.from_documents([document], embeddings)

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

1.  **Linguagem:** Comunique-se de forma clara, objetiva e encorajadora. seja legal pode usar algumas girias mas não exagere 
2.  **Atitude:** Mantenha sempre uma postura positiva, paciente e motivadora. O objetivo é fazer com que o estudante se sinta confiante em sua capacidade de aprender qualquer tópico.

---
# DIRETRIZES DE EXECUÇÃO

1.  **Fonte de Verdade Primária:** Sua primeira e principal fonte de informação é sempre o conteúdo fornecido no `{context}` (os arquivos que o usuário enviou).

2.  **Análise Estruturada (O Ponto-Chave):** Ao receber um material no `{context}` para resumir ou explicar, sua primeira ação é analisá-lo para identificar os seguintes pontos:
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

Contexto: {context}
Pergunta: {question}
"""

CONDENSE_PROMPT = """Dado o histórico da conversa e uma nova pergunta, reescreva a pergunta para ser uma pergunta independente, em sua língua original.

Histórico:
{chat_history}

Pergunta: {question}

Pergunta independente:"""

def criar_chain(vectorstore, mensagens_anteriores=None):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
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

    question_generator = LLMChain(llm=llm, prompt=PromptTemplate.from_template(CONDENSE_PROMPT))
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=ChatPromptTemplate.from_template(RAG_PROMPT))
    
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        memory=memory,
    )

# -----------------------------
# Busca web
# -----------------------------
def criar_query_de_busca(pergunta: str) -> str:
    try:
     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        tratar_erro_api("Gemini", e)
        st.stop()

    
    template = """
    Sua tarefa é extrair as palavras-chave essenciais da "Pergunta do Usuário" para realizar uma busca eficiente na internet.
    Adicione "CEFET-MG" 

    Pergunta do Usuário: "{pergunta}"
    Query de Busca:
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        resultado = chain.invoke({"pergunta": pergunta})
        return resultado.get('text', pergunta).strip()
    except Exception as e:
        st.warning(f"Erro ao criar query de busca: {e}")
        return " ".join(pergunta.split()[:5])

def buscar_serper(query: str, max_results: int = 4) -> list[dict]:
    try:
        res = serper.results(query)
        resultados = []
        for r in res.get("organic", [])[:max_results]:
            link, title, snippet = r.get("link"), r.get("title"), r.get("snippet")
            if link and title:
                resultados.append({"title": title, "link": link, "snippet": snippet})
        return resultados
    except Exception as e:
        tratar_erro_api("Serper", e)
        return []

def extrair_e_resumir_web(llm, resultados_busca: list[dict], pergunta: str):
    if not resultados_busca:
        return None
    urls = [r["link"] for r in resultados_busca]
    loader = AsyncChromiumLoader(urls)
    nest_asyncio.apply()
    html_docs = loader.load()
    if not html_docs:
        return None
    bs_transformer = BeautifulSoupTransformer()
    docs_transformados = bs_transformer.transform_documents(html_docs, tags_to_extract=["p","h1","h2","h3","li","span","div"])
    docs_validos = [doc for doc in docs_transformados if doc.page_content.strip()]
    if not docs_validos:
        return None

    prompt_resumo = ChatPromptTemplate.from_template("""
Você é um assistente de pesquisa especialista. Analise o 'Contexto da Web' para responder à 'Pergunta do Usuário' de forma completa.

**Contexto da Web:**
{context}

**Pergunta do Usuário:** {question}

**Sua Resposta Sintetizada:**
""")
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_resumo)
    contexto_web = "\n\n---\n\n".join([f"Fonte: {doc.metadata['source']}\nConteúdo: {doc.page_content}" for doc in docs_validos])
    resposta = chain.invoke({"input_documents": docs_validos, "question": pergunta, "context": contexto_web})
    texto_resposta = resposta.get("output_text", "")
    if texto_resposta:
        texto_resposta += "\n\n---\n**Fontes:**\n" + "\n".join(set(doc.metadata['source'] for doc in docs_validos))
    return texto_resposta

# -----------------------------
# Estado inicial e autenticação
# -----------------------------
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
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔐 Login"):
            st.switch_page("pages/Login.py")
    st.stop()

username = st.session_state.username
st.title("🎓 CEFET-MG - Assistente Virtual")

# -----------------------------
# Sidebar: PDFs + chats salvos
# -----------------------------
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
        st.success("Novo chat iniciado.")

    # Chats salvos
    st.subheader("📅 Seus chats salvos")
    chats = listar_chats(username)
    if chats:
        chat_escolhido = st.selectbox("Escolha um chat", chats, format_func=lambda x: x[1])
        if st.button("🔄 Carregar chat"):
            st.session_state.messages, st.session_state.pdf_paths = carregar_chat(chat_escolhido[0])
            st.session_state.chat_name = chat_escolhido[1]
            if st.session_state.pdf_paths and all(os.path.exists(p) for p in st.session_state.pdf_paths):
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

# -----------------------------
# Inicialização da Chain padrão
# -----------------------------
if st.session_state.chain is None:
    with st.spinner("Carregando base padrão..."):
        vectorstore = carregar_vectorstore_default()
        if vectorstore:
            st.session_state.chain = criar_chain(vectorstore)

# -----------------------------
# Histórico de chat
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Interação principal
# -----------------------------
# Interação principal
GATILHOS_FALLBACK = [
    "não encontrei",
    "não há menção",
    "não tem menção",
    "não achei",
    "não vi informação",
    "não encontrei informações",
    "não consta",
    "não há registro",
    "no material nao tem informações",
    "Parece que houve um engano"

]

if user_input := st.chat_input("Digite sua pergunta"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        final_bot_msg = ""
        with st.spinner("Pensando..."):
            if st.session_state.chain:
                response = st.session_state.chain.invoke({"question": user_input})
                rag_response = response.get("answer", "Não consegui gerar uma resposta.")
            else:
                rag_response = "A base de documentos não está carregada."

        final_bot_msg = rag_response

       

        if (
            (not st.session_state.pdf_paths or len(st.session_state.pdf_paths) == 0)
            and any(trigger in rag_response.lower() for trigger in GATILHOS_FALLBACK)
        ):
            with st.spinner("Buscando na web... 🔎"):
                query_web = criar_query_de_busca(user_input)
                st.info(f"**Buscando por:** {query_web}")

                resultados_web = buscar_serper(query_web, max_results=4)

                if resultados_web:
                    # Mostrar os 4 links principais
                    st.markdown("**🔗 Principais resultados encontrados:**")
                    for r in resultados_web:
                        st.markdown(f"- [{r['title']}]({r['link']})")


                    try:
                        llm_web = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4, google_api_key=GOOGLE_API_KEY)
                    except Exception as e:
                        tratar_erro_api("Gemini", e)
                        st.stop()


                    st.info("🕵️‍♀️ Coletando informações completas dos sites encontrados... Por favor, aguarde alguns minutos")
                    resumo_web = extrair_e_resumir_web(llm_web, resultados_web, user_input)


                    final_bot_msg = (
                         resumo_web
                    )

                else:
                    final_bot_msg = (
                        "Tentei buscar na web, mas não encontrei resultados relevantes."
                    )



        st.markdown(final_bot_msg)

    st.session_state.messages.append({"role": "assistant", "content": final_bot_msg})

    if not st.session_state.chat_name:
        st.session_state.chat_name = f"Chat - {user_input[:30]}"

    salvar_chat(
        username=username,
        chat_name=st.session_state.chat_name,
        messages=st.session_state.messages,
        pdf_paths=st.session_state.get("pdf_paths", [])
    )
