#  Infobot - Assistente Virtual Inteligente do CEFET-MG

##  Sobre o Projeto

Com o avanço da Inteligência Artificial, novas soluções vêm sendo desenvolvidas para otimizar processos no ambiente educacional.  
No **Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)** – Campus Varginha, observou-se a necessidade de um sistema capaz de fornecer informações acadêmicas de forma automatizada e acessível aos estudantes.

Muitos alunos enfrentam dificuldades ao buscar dados sobre o curso de **Sistemas de Informação**, como formas de ingresso, carga horária ou participação em projetos institucionais.  
Neste contexto, o **Infobot** foi desenvolvido com o objetivo de criar um **assistente virtual inteligente** que responda de forma rápida, clara e contextualizada às dúvidas dos alunos.

O sistema utiliza **modelos de linguagem natural (LLMs)** e foi implementado com uma **interface web interativa**, oferecendo uma experiência fluida e acessível a todos os usuários.

---

##  Funcionalidades

-  **Respostas inteligentes** sobre o curso de Sistemas de Informação do CEFET-MG.  
-  **Integração com o Google Search** – realiza pesquisas em tempo real para complementar respostas.  
-  **Leitura de arquivos PDF** – o usuário pode enviar documentos, e o Infobot é capaz de interpretar o conteúdo e responder com base neles.  
-  **Busca vetorial (FAISS)** para recuperação eficiente de informações locais.  
-  **Interface web responsiva** construída com Streamlit.  
-  **Integração com MongoDB e PostgreSQL** para armazenamento e consultas.  

---

##  Tecnologias Utilizadas

- **Python**   
- **LangChain** — orquestração do fluxo de conversação e RAG  
- **Gemini (Google Generative AI)** — modelo de linguagem natural  
- **Streamlit** — desenvolvimento da interface web  
- **MongoDB** e **PostgreSQL** — bancos de dados  
- **FAISS** — busca vetorial eficiente  
- **Serper API** — pesquisa em tempo real  
- **PyMuPDF** — leitura e análise de arquivos PDF  

---

##  Como Executar o Projeto

###  Pré-requisitos
Certifique-se de ter instalado:
- Python 3.10+  
- pip
- Chaves de API configuradas ( Gemini API Key, SERPER API KEY,mongoDB, PostgeSQL )

###  Instalação

```bash
# Clone o repositório
git clone https://github.com/isabella-pila/infobot.git

# Acesse o diretório
cd infobot

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
playwright install chromium
```

### Execução 
```bash
  streamlit run Home.py
