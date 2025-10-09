import streamlit as st
from auth_config import inicializar_db, verificar_credenciais

st.set_page_config(page_title="Login", page_icon="🔐")
st.title("🔐 Login")

inicializar_db()

if "auth_status" not in st.session_state:
    st.session_state.auth_status = None

with st.form("login_form"):
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    submit = st.form_submit_button("Entrar")

    if submit:
        name, auth_status, username = verificar_credenciais(usuario, senha)
        st.session_state.auth_status = auth_status
        st.session_state.username = username
        st.session_state.name = name

        if auth_status:
            st.success(f"Bem-vindo, {name}!")
            st.switch_page("pages/Chat.py")
        else:
            st.error("Usuário ou senha incorretos.")

if st.button("Não tem conta? Cadastre-se"):
    st.switch_page("Chat")
