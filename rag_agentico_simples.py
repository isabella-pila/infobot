"""
RAG Agentico - versao SIMPLES (uma execucao, pipeline linear)
==============================================================
Mesmo modelo, mesmas temperaturas, mesmo k e MESMOS PROMPTS do Rag_agentico.py.

O que foi REMOVIDO (as "regras pesadas"):
  - o loop do Verificador (ate 3 iteracoes, k crescente a cada reprovacao);
  - o fallback de busca na web;
  - o grafo LangGraph (virou chamada direta: planejar -> recuperar -> sintetizar);
  - o benchmark com repeticoes (roda a pergunta UMA vez).

O que foi CORRIGIDO:
  - load_dotenv() ANTES de ler GOOGLE_API_KEY. No Rag_agentico.py a chave era lida
    antes de o .env ser carregado, resultando em google_api_key=None; o cliente caia
    no ADC do Google Cloud e ficava minutos em retentativas silenciosas.
  - max_retries e timeout explicitos: falha rapido e mostra o erro real.
"""

import os
import re
import csv
import json
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()  # precisa vir ANTES de ler a chave

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY nao encontrada. Confira o arquivo .env.")

# ============================================================
# CONFIGURACAO (identica ao Rag_agentico.py)
# ============================================================
MODELO_GERADOR = "gemini-3.5-flash-lite"

TEMPERATURA_SINTETIZADOR = 0.7
TEMPERATURA_PLANEJADOR = 0.2

K_BASE = 4
MAX_SUBPERGUNTAS = 4
MAX_OUTPUT_TOKENS_SINTETIZADOR = 8192

# Falha rapido em vez de ficar preso em backoff exponencial.
MAX_RETRIES = 2
TIMEOUT_SEGUNDOS = 90

PERGUNTA = "O que preciso fazer para participar de um projeto de pesquisa?"
ID_PERGUNTA = "Q19"
CAMINHO_SAIDA = "resultado_agentico_simples.csv"

# ============================================================
# PROMPTS (copiados sem alteracao do Rag_agentico.py)
# ============================================================
PROMPT_PLANEJADOR = """Você é o Agente Planejador de um sistema de suporte acadêmico do CEFET-MG, Campus Varginha.
Analise a pergunta do usuário e decida se ela pode ser respondida com uma única busca vetorial
ou se precisa ser decomposta em sub-perguntas mais específicas.

Regras:
- Se a pergunta for simples e direta (ex: "Qual é o e-mail da coordenação do curso?"),
  retorne APENAS ela mesma.
- Se a pergunta exigir síntese de múltiplas informações ou raciocínio multi-etapa
  (ex: "Quais são os prazos e os procedimentos para solicitar dispensa de disciplinas e
  como isso afeta a integralização do curso?"), decomponha em 2 a 4 sub-perguntas
  independentes que, juntas, cubram o necessário para responder.
- Retorne uma sub-pergunta por linha, sem numeração e sem explicações.

Pergunta do usuário: {pergunta}

Sub-perguntas:"""

PROMPT_SINTETIZADOR = """Você é o "Infobot", assistente virtual do CEFET-MG, Campus Varginha, atuando agora
como Agente Sintetizador de um pipeline multi-agente.  Use SOMENTE o contexto abaixo para responder
de forma clara, organizada e didática, com títulos/bullet points quando fizer sentido.
Se o contexto vier de múltiplas sub-buscas, integre tudo em uma resposta coesa, sem repetições
e sem contradições.

COMPLETUDE É PRIORIDADE: extraia e inclua TODAS as informações do contexto que sejam relevantes
para a pergunta. Se houver listas (disciplinas, requisitos, documentos, etapas, prazos, exceções),
reproduza TODOS os itens presentes no contexto, nunca corte com "entre outros" ou "etc.".
Seja preciso com números, prazos, cargas horárias e nomes, reproduzindo-os exatamente.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""


def get_llm(temperature: float, max_output_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODELO_GERADOR,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=max_output_tokens,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT_SEGUNDOS,
    )


_PREFIXO_LISTA = re.compile(r"^\s*(?:[-•*]+|\d+[.)])\s*")


def planejar(pergunta: str) -> List[str]:
    """Decompoe a pergunta em sub-consultas (mesma logica, sem LLMChain)."""
    llm = get_llm(temperature=TEMPERATURA_PLANEJADOR)
    chain = PromptTemplate.from_template(PROMPT_PLANEJADOR) | llm
    texto = chain.invoke({"pergunta": pergunta}).content.strip()

    subs = [_PREFIXO_LISTA.sub("", l).strip() for l in texto.split("\n") if l.strip()]
    subs = [s for s in subs if s][:MAX_SUBPERGUNTAS]
    return subs or [pergunta]


def recuperar(vectorstore, consultas: List[str], k: int = K_BASE) -> List[Document]:
    """Busca vetorial no FAISS, uma passagem so, sem k crescente."""
    encontrados: List[Document] = []
    for sub in consultas:
        try:
            encontrados.extend(vectorstore.similarity_search(sub, k=k))
        except Exception as e:
            print(f"[Recuperador] erro na busca '{sub}': {e}")

    vistos, unicos = set(), []
    for d in encontrados:
        chave = d.page_content[:200]
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(d)
    return unicos


def sintetizar(pergunta: str, docs: List[Document]) -> str:
    """Stuffing do contexto + prompt do Sintetizador (mesmo separador do load_qa_chain)."""
    llm = get_llm(
        temperature=TEMPERATURA_SINTETIZADOR,
        max_output_tokens=MAX_OUTPUT_TOKENS_SINTETIZADOR,
    )
    contexto = "\n\n".join(d.page_content for d in docs)
    chain = PromptTemplate.from_template(PROMPT_SINTETIZADOR) | llm
    return chain.invoke({"context": contexto, "question": pergunta}).content.strip()


def main():
    cfg = {
        "modelo": MODELO_GERADOR,
        "temperatura_sintetizador": TEMPERATURA_SINTETIZADOR,
        "temperatura_planejador": TEMPERATURA_PLANEJADOR,
        "k_base": K_BASE,
        "max_subperguntas": MAX_SUBPERGUNTAS,
        "verificador": False,
        "fallback_web": False,
        "repeticoes": 1,
    }
    print("Configuracao:", json.dumps(cfg, ensure_ascii=False))

    inicio = time.time()

    print("\n[1/4] Carregando vectorstore (vetorizando o PDF)...")
    from rag_core import carregar_vectorstore_default
    vectorstore = carregar_vectorstore_default()
    print(f"      ok ({time.time() - inicio:.1f}s)")

    print("[2/4] Planejador...")
    t = time.time()
    plano = planejar(PERGUNTA)
    print(f"      {len(plano)} sub-pergunta(s) em {time.time() - t:.1f}s")
    for s in plano:
        print(f"        - {s}")

    print("[3/4] Recuperador...")
    t = time.time()
    docs = recuperar(vectorstore, plano)
    print(f"      {len(docs)} chunks unicos em {time.time() - t:.1f}s")

    print("[4/4] Sintetizador...")
    t = time.time()
    resposta = sintetizar(PERGUNTA, docs)
    print(f"      resposta gerada em {time.time() - t:.1f}s")

    latencia = time.time() - inicio

    print("\n" + "=" * 70)
    print(f"PERGUNTA: {PERGUNTA}")
    print("=" * 70)
    print(resposta)
    print("=" * 70)
    print(f"Latencia total: {latencia:.1f}s")

    campos = ["id", "pergunta", "resposta", "plano_busca", "contextos",
              "n_chunks", "latencia_segundos", "fontes"]
    fontes = sorted({d.metadata.get("source", "base institucional") for d in docs})
    with open(CAMINHO_SAIDA, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerow({
            "id": ID_PERGUNTA,
            "pergunta": PERGUNTA,
            "resposta": resposta,
            "plano_busca": json.dumps(plano, ensure_ascii=False),
            "contextos": json.dumps([d.page_content for d in docs], ensure_ascii=False),
            "n_chunks": len(docs),
            "latencia_segundos": round(latencia, 2),
            "fontes": json.dumps(fontes, ensure_ascii=False),
        })
    print(f"Salvo em {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
