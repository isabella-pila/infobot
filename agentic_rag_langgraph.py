"""
Artigo 2 - Assistente Virtual com RAG Agêntico e Orquestração Multi-Agentes (LangGraph)
========================================================================================
Implementa os quatro agentes especializados descritos na metodologia do TCC
(Seção 3.2, Objetivo Específico C): Planejador, Recuperador, Verificador e Sintetizador.

Este módulo foi desenhado para reaproveitar ao máximo o que já existe no Artigo 1:
- o mesmo vectorstore FAISS (carregar_vectorstore_default / get_vectorstore)
- o mesmo modelo de linguagem (para garantir que a comparação isole a variável arquitetural)
- a mesma função de busca web (pesquisar_na_web) como fallback

Métricas operacionais coletadas automaticamente (Objetivo Específico E):
- iteracoes_verificacao: nº de ciclos recuperação->verificação até validar o contexto
- latencia_segundos: tempo total da execução do grafo
- usou_fallback_web: se o sistema precisou recorrer à busca externa
"""

import os
import re
import time
import operator
from typing import TypedDict, Annotated, Sequence, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langgraph.graph import StateGraph, END

MAX_ITERACOES_VERIFICACAO = 3
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# 1. ESTADO COMPARTILHADO DO GRAFO
# ============================================================
class GraphState(TypedDict):
    pergunta: str
    pergunta_original: str
    plano_busca: List[str]
    # operator.add faz o LangGraph ACUMULAR documentos a cada passagem pelo
    # agente_recuperador, em vez de sobrescrever — importante quando o
    # Verificador manda recuperar de novo.
    contexto_recuperado: Annotated[Sequence[Document], operator.add]
    contexto_valido: bool
    score_relevancia: int
    justificativa_verificacao: str
    resposta_final: str
    fontes: List[str]
    iteracoes_verificacao: int
    usou_fallback_web: bool
    inicio_execucao: float
    latencia_segundos: float


# ============================================================
# 2. PROMPTS DOS AGENTES
# ============================================================
PROMPT_PLANEJADOR = """Você é o Agente Planejador de um sistema de suporte acadêmico do CEFET-MG, Campus Varginha.
Analise a pergunta do usuário e decida se ela pode ser respondida com uma única busca vetorial
ou se precisa ser decomposta em sub-perguntas mais específicas.

Regras:
- Se a pergunta for simples e direta (ex: "qual a carga horária do TCC?"), retorne APENAS ela mesma.
- Se a pergunta exigir síntese de múltiplas informações ou raciocínio multi-etapa
  (ex: "como o curso evolui ao longo dos períodos?"), decomponha em 2 a 4 sub-perguntas
  independentes que, juntas, cubram o necessário para responder.
- Retorne uma sub-pergunta por linha, sem numeração e sem explicações.

Pergunta do usuário: {pergunta}

Sub-perguntas:"""

PROMPT_VERIFICADOR = """Você é o Agente Verificador de um sistema RAG institucional do CEFET-MG.
Avalie O QUANTO o CONTEXTO abaixo ajuda a responder à PERGUNTA, sem que o modelo precise
inventar (alucinar) informações ausentes do texto. Atribua uma NOTA DE RELEVÂNCIA:

- 2 = suficiente para responder de forma completa e fiel.
- 1 = parcialmente relevante (cobre parte da pergunta, mas não tudo).
- 0 = irrelevante (o contexto não ajuda a responder a pergunta).

Responda EXATAMENTE neste formato:
RELEVANCIA: 0|1|2
JUSTIFICATIVA: <uma frase curta explicando a decisão>

Pergunta: {pergunta}

Contexto recuperado:
{contexto}
"""

PROMPT_SINTETIZADOR = """Você é o "Infobot", assistente virtual do CEFET-MG, Campus Varginha, atuando agora
como Agente Sintetizador de um pipeline multi-agente.  Use SOMENTE o contexto abaixo para responder
de forma clara, organizada e didática, com títulos/bullet points quando fizer sentido.
Se o contexto vier de múltiplas sub-buscas, integre tudo em uma resposta coesa, sem repetições
e sem contradições.

COMPLETUDE É PRIORIDADE: extraia e inclua TODAS as informações do contexto que sejam relevantes
para a pergunta. Se houver listas (disciplinas, requisitos, documentos, etapas, prazos, exceções),
reproduza TODOS os itens presentes no contexto — nunca corte com "entre outros" ou "etc.".
Seja preciso com números, prazos, cargas horárias e nomes, reproduzindo-os exatamente.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""


# ============================================================
# 3. LLM (mesmo modelo do Artigo 1 — controle experimental)
# ============================================================
def get_llm(temperature: float = 0.3, max_output_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=max_output_tokens,
    )


# ============================================================
# 4. AGENTES (NÓS DO GRAFO)
# ============================================================
def agente_planejador(state: GraphState) -> dict:
    """Decompõe a pergunta em sub-consultas quando necessário."""
    pergunta = state.get("pergunta_original") or state["pergunta"]

    llm = get_llm(temperature=0.2)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_PLANEJADOR))
    resultado = chain.invoke({"pergunta": pergunta})
    texto = resultado.get("text", pergunta).strip()

    sub_perguntas = [l.strip("-•* ").strip() for l in texto.split("\n") if l.strip()]
    if not sub_perguntas:
        sub_perguntas = [pergunta]

    return {
        "pergunta_original": pergunta,
        "plano_busca": sub_perguntas,
    }


def agente_recuperador(state: GraphState, vectorstore, k_base: int = 8) -> dict:
    """Executa busca vetorial no FAISS para cada sub-consulta do plano gerado pelo Planejador.

    (A) Em re-tentativas (quando o Verificador reprovou o contexto), o grafo volta para
    ESTE nó — não para o Planejador. Para que a nova passagem traga chunks REALMENTE
    novos, o `k` cresce a cada iteração e a pergunta original é adicionada às consultas.
    """
    iteracao = state.get("iteracoes_verificacao", 0)
    k = k_base + 6 * iteracao  # 8 -> 14 -> 20 ... a cada reprovação do Verificador

    consultas = list(state["plano_busca"])
    if iteracao > 0 and state.get("pergunta_original"):
        # Na retentativa, busca também pela pergunta original inteira (query mais ampla).
        consultas.append(state["pergunta_original"])

    docs_encontrados: List[Document] = []
    for sub_pergunta in consultas:
        try:
            docs_encontrados.extend(vectorstore.similarity_search(sub_pergunta, k=k))
        except Exception as e:
            print(f"[Recuperador] erro na busca '{sub_pergunta}': {e}")

    # Remove duplicatas, inclusive contra o que JÁ foi acumulado em passagens anteriores
    # (contexto_recuperado usa operator.add), para só adicionar chunks inéditos.
    vistos = {d.page_content[:200] for d in state.get("contexto_recuperado", [])}
    docs_unicos = []
    for d in docs_encontrados:
        chave = d.page_content[:200]
        if chave not in vistos:
            vistos.add(chave)
            docs_unicos.append(d)

    print(f"[Recuperador] iteracao={iteracao} k={k} novos_chunks={len(docs_unicos)}")
    return {"contexto_recuperado": docs_unicos}


def agente_verificador(state: GraphState) -> dict:
    """Avalia se o contexto acumulado até agora é suficiente/fiel para responder."""
    contexto_texto = "\n\n---\n\n".join(
        d.page_content for d in state["contexto_recuperado"]
    )[:12000]

    llm = get_llm(temperature=0.0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_VERIFICADOR))
    resultado = chain.invoke({
        "pergunta": state["pergunta_original"],
        "contexto": contexto_texto or "(nenhum contexto recuperado)",
    })
    texto = resultado.get("text", "")

    m_score = re.search(r"RELEVANCIA:\s*([012])", texto, re.IGNORECASE)
    score = int(m_score.group(1)) if m_score else 0
    m = re.search(r"JUSTIFICATIVA:\s*(.+)", texto, re.IGNORECASE)

    return {
        "contexto_valido": score >= 2,
        "score_relevancia": score,
        "justificativa_verificacao": m.group(1).strip() if m else "",
        "iteracoes_verificacao": state.get("iteracoes_verificacao", 0) + 1,
    }


def agente_sintetizador(state: GraphState) -> dict:
    """Gera a resposta final a partir do contexto já validado pelo Verificador."""
    llm = get_llm(temperature=0.5, max_output_tokens=8192)
    chain = load_qa_chain(
        llm, chain_type="stuff",
        prompt=ChatPromptTemplate.from_template(PROMPT_SINTETIZADOR),
    )
    resultado = chain.invoke({
        "input_documents": list(state["contexto_recuperado"]),
        "question": state["pergunta_original"],
    })

    fontes = list({
        d.metadata.get("source", "base institucional")
        for d in state["contexto_recuperado"]
    })

    return {
        "resposta_final": resultado.get("output_text", "Não foi possível gerar uma resposta."),
        "fontes": fontes,
        "latencia_segundos": time.time() - state.get("inicio_execucao", time.time()),
    }


def agente_busca_web_fallback(state: GraphState, pesquisar_na_web_fn) -> dict:
    """
    Aciona o fallback web quando, após MAX_ITERACOES_VERIFICACAO, o contexto local
    ainda é irrelevante. Reaproveita a busca web do Artigo 1
    (query inteligente -> Serper -> extração httpx -> síntese).

    (C) Passa o contexto local já recuperado para uma síntese HÍBRIDA (PDF + web).
    (D) Popula o campo `fontes` (antes ficava vazio mesmo consultando a web).
    Aceita tanto a versão estruturada (retorna dict {resposta, fontes}) quanto a
    versão antiga que retorna apenas string.
    """
    llm = get_llm(temperature=0.3)
    contexto_local = list(state.get("contexto_recuperado", []))

    try:
        resultado_web = pesquisar_na_web_fn(llm, state["pergunta_original"], contexto_local=contexto_local)
    except TypeError:
        # Compatibilidade com a assinatura antiga pesquisar_na_web(llm, pergunta)
        resultado_web = pesquisar_na_web_fn(llm, state["pergunta_original"])

    if isinstance(resultado_web, dict):
        resposta = resultado_web.get("resposta", "")
        fontes = resultado_web.get("fontes", [])
    else:
        resposta = resultado_web or ""
        fontes = list({
            d.metadata.get("source", "base institucional") for d in contexto_local
        })

    return {
        "resposta_final": resposta or "Não foi possível encontrar uma resposta confiável.",
        "fontes": fontes,
        "usou_fallback_web": True,
        "latencia_segundos": time.time() - state.get("inicio_execucao", time.time()),
    }


# ============================================================
# 5. ROTEAMENTO CONDICIONAL
# ============================================================
def decidir_proximo_passo(state: GraphState) -> str:
    # Contexto suficiente (nota 2): responde direto.
    if state["contexto_valido"]:
        return "sintetizar"

    # Ainda há tentativas: busca de novo (com query expandida e k maior no recuperador).
    if state["iteracoes_verificacao"] < MAX_ITERACOES_VERIFICACAO:
        return "recuperar_novamente"

    # Esgotou as tentativas. Se o contexto local é ao menos PARCIALMENTE relevante
    # (nota 1), responde com o que tem em vez de recorrer à web à toa. Só cai no
    # fallback web quando o contexto local é irrelevante (nota 0).
    if state.get("score_relevancia", 0) >= 1:
        return "sintetizar"
    return "busca_web_fallback"


# ============================================================
# 6. MONTAGEM DO GRAFO
# ============================================================
def construir_grafo(vectorstore, pesquisar_na_web_fn):
    """
    vectorstore: seu FAISS já carregado (ex: carregar_vectorstore_default() do Artigo 1)
    pesquisar_na_web_fn: a função pesquisar_na_web(llm, pergunta) já implementada no Artigo 1
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("planejador", agente_planejador)
    workflow.add_node("recuperador", lambda s: agente_recuperador(s, vectorstore))
    workflow.add_node("verificador", agente_verificador)
    workflow.add_node("sintetizador", agente_sintetizador)
    workflow.add_node("busca_web_fallback", lambda s: agente_busca_web_fallback(s, pesquisar_na_web_fn))

    workflow.set_entry_point("planejador")
    workflow.add_edge("planejador", "recuperador")
    workflow.add_edge("recuperador", "verificador")

    workflow.add_conditional_edges(
        "verificador",
        decidir_proximo_passo,
        {
            "sintetizar": "sintetizador",
            "recuperar_novamente": "recuperador",
            "busca_web_fallback": "busca_web_fallback",
        },
    )

    workflow.add_edge("sintetizador", END)
    workflow.add_edge("busca_web_fallback", END)

    return workflow.compile()


# ============================================================
# 7. FUNÇÃO DE EXECUÇÃO (para uso no Streamlit e nos experimentos comparativos)
# ============================================================
def responder_com_rag_agentico(app_grafo, pergunta: str) -> dict:
    """
    Executa o grafo para uma pergunta e retorna resposta + métricas operacionais,
    já no formato que você vai usar para alimentar a Tabela comparativa do Artigo 2
    (mesmo padrão da Tabela 2 do Artigo 1, mas com colunas extras de iterações/latência).
    """
    estado_inicial: GraphState = {
        "pergunta": pergunta,
        "pergunta_original": pergunta,
        "plano_busca": [],
        "contexto_recuperado": [],
        "contexto_valido": False,
        "score_relevancia": 0,
        "justificativa_verificacao": "",
        "resposta_final": "",
        "fontes": [],
        "iteracoes_verificacao": 0,
        "usou_fallback_web": False,
        "inicio_execucao": time.time(),
        "latencia_segundos": 0.0,
    }

    resultado = app_grafo.invoke(estado_inicial, config={"recursion_limit": 15})

    return {
        "resposta": resultado["resposta_final"],
        "fontes": resultado.get("fontes", []),
        "iteracoes_verificacao": resultado["iteracoes_verificacao"],
        "usou_fallback_web": resultado["usou_fallback_web"],
        "latencia_segundos": round(resultado["latencia_segundos"], 2),
    }


# ============================================================
# 8. EXEMPLO DE USO (integração com o app.py do Artigo 1)
# ============================================================
if __name__ == "__main__":
    # Ajuste este import para o nome real do seu arquivo do Artigo 1
    # from app import carregar_vectorstore_default, pesquisar_na_web

    # vectorstore = carregar_vectorstore_default()
    # grafo = construir_grafo(vectorstore, pesquisar_na_web)

    # resultado = responder_com_rag_agentico(grafo, "Como o curso evolui ao longo dos períodos?")
    # print(resultado["resposta"])
    # print("Iterações:", resultado["iteracoes_verificacao"], "| Latência:", resultado["latencia_segundos"], "s")
    pass