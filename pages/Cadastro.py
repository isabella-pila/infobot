import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
from auth_config import inicializar_db, conectar_db
import streamlit_authenticator as stauth

st.set_page_config(page_title="Cadastro", page_icon="📝")
st.title("📝 Cadastro")

# Inicializa a tabela (cria se não existir)
inicializar_db()

with st.form("cadastro_form"):
    nome = st.text_input("Nome completo")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    senha_confirm = st.text_input("Confirmar senha", type="password")
    submit = st.form_submit_button("Cadastrar")

    if submit:
        if not nome or not usuario or not senha:
            st.warning("Preencha todos os campos.")
        elif len(senha) < 6:
            st.warning("A senha deve ter pelo menos 6 caracteres.")
        elif senha != senha_confirm:
            st.error("As senhas não coincidem.")
        else:
            conn = conectar_db()
            c = conn.cursor()
            try:
                # Verifica se o usuário já existe
                c.execute("SELECT * FROM users WHERE username = %s", (usuario,))
                if c.fetchone():
                    st.error("Usuário já existe.")
                else:
                    # Criptografa a senha antes de salvar
                    hashed_password = bcrypt.hashpw(senha.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                    # Insere o novo usuário
                    c.execute(
                        "INSERT INTO users (username, name, password) VALUES (%s, %s, %s)",
                        (usuario, nome, hashed_password)
                    )
                    conn.commit()
                    st.success("Usuário cadastrado com sucesso! 🎉")
                    st.switch_page("pages/Login.py")

            except Exception as e:
                st.error(f"Erro ao cadastrar usuário: {e}")
                conn.rollback()
            finally:
                conn.close()

if st.button("Já tem conta? Faça login"):
    st.switch_page("Login.py")
