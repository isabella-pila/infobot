
import streamlit as st
import os
import uuid
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def tratar_erro_api(api_nome: str, erro: Exception):
    erro_str = str(erro).lower()

    if "quota" in erro_str or "rate limit" in erro_str:
        st.error(f"🚫 Limite de uso da API {api_nome} atingido. Tente novamente mais tarde.")
    elif "invalid api key" in erro_str or "unauthorized" in erro_str:
        st.error(f"🔑 Chave de API {api_nome} inválida ou expirada.")
    elif "timeout" in erro_str:
        st.error(f"⏱️ A API {api_nome} demorou demais para responder. Tente novamente em instantes.")
    elif "service unavailable" in erro_str or "503" in erro_str:
        st.error(f"⚙️ A API {api_nome} está temporariamente fora do ar.")
    else:
        st.error(f"❌ Ocorreu um erro inesperado com a API {api_nome}: {erro}")
