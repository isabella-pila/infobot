"""
Artigo 2 - Assistente Virtual com RAG Agêntico e Orquestração Multi-Agentes (LangGraph)
========================================================================================
Quatro agentes especializados: Planejador, Recuperador, Verificador e Sintetizador.

ALTERAÇÕES EM RELAÇÃO À VERSÃO ANTERIOR (para o controle experimental com o Artigo 1)
-------------------------------------------------------------------------------------
1. Modelo, temperaturas e k ficam em constantes no topo do arquivo, para serem
   alinhados ao Artigo 1 e registrados na Tabela I. O alias "-latest" foi removido.
2. k base = 4 (igual ao Artigo 1). Antes era 8, com +6 a cada retentativa.
3. Os exemplos do prompt do Planejador não pertencem mais ao conjunto de avaliação
   (antes eram as perguntas Q13 e Q16).
4. O Sintetizador pode usar o RAG_PROMPT do Artigo 1 (USAR_PROMPT_ARTIGO1), isolando
   o efeito do Planejador e do Verificador do efeito do prompt.
5. A busca web é desativada por padrão (permitir_fallback_web=False), como na
   avaliação do Artigo 1. Sem ela, o contexto insuficiente segue para a geração e é
   sinalizado em `contexto_insuficiente`.
6. O Verificador lê o mesmo contexto que o Sintetizador (limite de 12.000 caracteres
   elevado para 60.000, com aviso se for atingido).
7. O retorno inclui os contextos recuperados, o plano de busca, a nota e a
   justificativa do Verificador (os contextos são necessários para o RAGAS).
8. executar_benchmark() roda as 20 perguntas do Artigo 1 (Tabela 1), com repetições,
   e salva CSV + JSON da configuração.
"""

import os
import re
import csv
import json
import time
import operator
from typing import TypedDict, Annotated, Sequence, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langgraph.graph import StateGraph, END

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO  (registre estes valores na Tabela I do artigo)
# ============================================================
# CONFIRME: deve ser EXATAMENTE o modelo usado nos resultados do Artigo 1.
# Fixe a versão: aliases como "-latest" mudam com o tempo e prejudicam a reprodutibilidade.
MODELO_GERADOR = "gemini-3.8-flash"

# CONFIRME: temperatura de geração do Artigo 1 (o texto submetido diz 0,7).
TEMPERATURA_SINTETIZADOR = 0.7
TEMPERATURA_PLANEJADOR = 0.2
TEMPERATURA_VERIFICADOR = 0.0

K_BASE = 4                       # top-k por consulta (Artigo 1: k = 4)
K_INCREMENTO_RETENTATIVA = 4     # acréscimo de k a cada reprovação do Verificador
MAX_SUBPERGUNTAS = 4
MAX_ITERACOES_VERIFICACAO = 3    # nº máximo de verificações (= até 2 retentativas)
MAX_CHARS_VERIFICADOR = 60000
MAX_OUTPUT_TOKENS_SINTETIZADOR = 8192

# Para a comparação principal, cole aqui o RAG_PROMPT do Artigo 1 (string com {context}
# e {question}) e mude USAR_PROMPT_ARTIGO1 para True. Assim, o prompt de geração fica
# idêntico nas duas arquiteturas. Com False, usa-se o PROMPT_SINTETIZADOR abaixo
# (ênfase em completude), que é uma condição adicional e deve ser descrita como tal.
RAG_PROMPT_ARTIGO1: Optional[str] = None
USAR_PROMPT_ARTIGO1 = False


# ============================================================
# 1. ESTADO COMPARTILHADO DO GRAFO
# ============================================================
class GraphState(TypedDict):
    pergunta: str
    pergunta_original: str
    plano_busca: List[str]
    # operator.add faz o LangGraph ACUMULAR documentos a cada passagem pelo
    # agente_recuperador, em vez de sobrescrever (importante quando o Verificador
    # manda recuperar de novo).
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
# Os exemplos abaixo NÃO devem coincidir com nenhuma pergunta do conjunto de avaliação.
# Se trocar os exemplos, confira que continuam fora das 20 perguntas.
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
reproduza TODOS os itens presentes no contexto, nunca corte com "entre outros" ou "etc.".
Seja preciso com números, prazos, cargas horárias e nomes, reproduzindo-os exatamente.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""


def _prompt_sintetizador() -> ChatPromptTemplate:
    if USAR_PROMPT_ARTIGO1:
        if not RAG_PROMPT_ARTIGO1:
            raise ValueError(
                "USAR_PROMPT_ARTIGO1=True, mas RAG_PROMPT_ARTIGO1 está vazio. "
                "Cole o RAG_PROMPT do Artigo 1 (com {context} e {question})."
            )
        return ChatPromptTemplate.from_template(RAG_PROMPT_ARTIGO1)
    return ChatPromptTemplate.from_template(PROMPT_SINTETIZADOR)


# ============================================================
# 3. LLM
# ============================================================
def get_llm(temperature: float, max_output_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODELO_GERADOR,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=max_output_tokens,
    )


# ============================================================
# 4. AGENTES (NÓS DO GRAFO)
# ============================================================
_PREFIXO_LISTA = re.compile(r"^\s*(?:[-•*]+|\d+[.)])\s*")


def agente_planejador(state: GraphState) -> dict:
    """Decompõe a pergunta em sub-consultas quando necessário."""
    pergunta = state.get("pergunta_original") or state["pergunta"]

    llm = get_llm(temperature=TEMPERATURA_PLANEJADOR)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_PLANEJADOR))
    resultado = chain.invoke({"pergunta": pergunta})
    texto = resultado.get("text", pergunta).strip()

    sub_perguntas = [_PREFIXO_LISTA.sub("", l).strip() for l in texto.split("\n") if l.strip()]
    sub_perguntas = [s for s in sub_perguntas if s][:MAX_SUBPERGUNTAS]
    if not sub_perguntas:
        sub_perguntas = [pergunta]

    return {
        "pergunta_original": pergunta,
        "plano_busca": sub_perguntas,
    }


def agente_recuperador(state: GraphState, vectorstore, k_base: int = K_BASE) -> dict:
    """Executa busca vetorial no FAISS para cada sub-consulta do plano.

    Em retentativas (o Verificador reprovou o contexto), o grafo volta para ESTE nó.
    O k cresce a cada iteração e a pergunta original é adicionada às consultas, para
    que a nova passagem traga trechos realmente novos.
    """
    iteracao = state.get("iteracoes_verificacao", 0)
    k = k_base + K_INCREMENTO_RETENTATIVA * iteracao

    consultas = list(state["plano_busca"])
    if iteracao > 0 and state.get("pergunta_original"):
        consultas.append(state["pergunta_original"])

    docs_encontrados: List[Document] = []
    for sub_pergunta in consultas:
        try:
            docs_encontrados.extend(vectorstore.similarity_search(sub_pergunta, k=k))
        except Exception as e:
            print(f"[Recuperador] erro na busca '{sub_pergunta}': {e}")

    # Remove duplicatas, inclusive contra o que JÁ foi acumulado em passagens anteriores.
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
    """Avalia se o contexto acumulado até agora é suficiente para responder."""
    contexto_texto = "\n\n---\n\n".join(d.page_content for d in state["contexto_recuperado"])
    if len(contexto_texto) > MAX_CHARS_VERIFICADOR:
        print(f"[Verificador] AVISO: contexto com {len(contexto_texto)} caracteres "
              f"truncado em {MAX_CHARS_VERIFICADOR}.")
        contexto_texto = contexto_texto[:MAX_CHARS_VERIFICADOR]

    llm = get_llm(temperature=TEMPERATURA_VERIFICADOR)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_VERIFICADOR))
    resultado = chain.invoke({
        "pergunta": state["pergunta_original"],
        "contexto": contexto_texto or "(nenhum contexto recuperado)",
    })
    texto = resultado.get("text", "")

    m_score = re.search(r"RELEVANCIA:\s*([012])", texto, re.IGNORECASE)
    if not m_score:
        print(f"[Verificador] AVISO: formato inesperado, nota assumida 0. Saída: {texto[:120]!r}")
    score = int(m_score.group(1)) if m_score else 0
    m = re.search(r"JUSTIFICATIVA:\s*(.+)", texto, re.IGNORECASE)

    return {
        "contexto_valido": score >= 2,
        "score_relevancia": score,
        "justificativa_verificacao": m.group(1).strip() if m else "",
        "iteracoes_verificacao": state.get("iteracoes_verificacao", 0) + 1,
    }


def agente_sintetizador(state: GraphState) -> dict:
    """Gera a resposta final a partir de todo o contexto acumulado."""
    llm = get_llm(
        temperature=TEMPERATURA_SINTETIZADOR,
        max_output_tokens=MAX_OUTPUT_TOKENS_SINTETIZADOR,
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=_prompt_sintetizador())
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
    """Fallback web (somente se permitir_fallback_web=True em construir_grafo).

    Reaproveita a busca web do Artigo 1. Aceita a versão estruturada (retorna
    dict {resposta, fontes}) e a antiga (retorna apenas string).
    """
    llm = get_llm(temperature=TEMPERATURA_SINTETIZADOR)
    contexto_local = list(state.get("contexto_recuperado", []))

    try:
        resultado_web = pesquisar_na_web_fn(
            llm, state["pergunta_original"], contexto_local=contexto_local
        )
    except TypeError:
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
def decidir_proximo_passo(state: GraphState, permitir_fallback_web: bool = False) -> str:
    # Contexto suficiente (nota 2): responde.
    if state["contexto_valido"]:
        return "sintetizar"

    # Ainda há tentativas: busca de novo (consulta ampliada e k maior).
    if state["iteracoes_verificacao"] < MAX_ITERACOES_VERIFICACAO:
        return "recuperar_novamente"

    # Tentativas esgotadas. Nota 1 (parcial): responde com o que tem.
    if state.get("score_relevancia", 0) >= 1:
        return "sintetizar"

    # Nota 0: web só se permitido; caso contrário gera com o contexto disponível
    # (o resultado fica sinalizado em contexto_insuficiente).
    if permitir_fallback_web:
        return "busca_web_fallback"
    return "sintetizar"


# ============================================================
# 6. MONTAGEM DO GRAFO
# ============================================================
def construir_grafo(vectorstore, pesquisar_na_web_fn=None, permitir_fallback_web: bool = False):
    """
    vectorstore: FAISS já carregado (o mesmo índice do Artigo 1).
    pesquisar_na_web_fn: função de busca web do Artigo 1 (só usada se
        permitir_fallback_web=True). Para o experimento comparativo, deixe False.
    """
    if permitir_fallback_web and pesquisar_na_web_fn is None:
        raise ValueError("permitir_fallback_web=True exige pesquisar_na_web_fn.")

    workflow = StateGraph(GraphState)

    workflow.add_node("planejador", agente_planejador)
    workflow.add_node("recuperador", lambda s: agente_recuperador(s, vectorstore))
    workflow.add_node("verificador", agente_verificador)
    workflow.add_node("sintetizador", agente_sintetizador)

    workflow.set_entry_point("planejador")
    workflow.add_edge("planejador", "recuperador")
    workflow.add_edge("recuperador", "verificador")

    destinos = {
        "sintetizar": "sintetizador",
        "recuperar_novamente": "recuperador",
    }
    if permitir_fallback_web:
        workflow.add_node(
            "busca_web_fallback",
            lambda s: agente_busca_web_fallback(s, pesquisar_na_web_fn),
        )
        workflow.add_edge("busca_web_fallback", END)
        destinos["busca_web_fallback"] = "busca_web_fallback"

    workflow.add_conditional_edges(
        "verificador",
        lambda s: decidir_proximo_passo(s, permitir_fallback_web),
        destinos,
    )
    workflow.add_edge("sintetizador", END)

    return workflow.compile()


# ============================================================
# 7. EXECUÇÃO E MÉTRICAS
# ============================================================
def responder_com_rag_agentico(app_grafo, pergunta: str) -> dict:
    """Executa o grafo para uma pergunta e retorna resposta + métricas + contextos."""
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

    # Todos os trechos acumulados são os que o Sintetizador recebeu; use esta lista
    # como retrieved_contexts no RAGAS.
    contextos = [d.page_content for d in resultado.get("contexto_recuperado", [])]

    return {
        "resposta": resultado["resposta_final"],
        "fontes": resultado.get("fontes", []),
        "plano_busca": resultado.get("plano_busca", []),
        "contextos": contextos,
        "n_chunks": len(contextos),
        "iteracoes_verificacao": resultado["iteracoes_verificacao"],
        "score_relevancia": resultado.get("score_relevancia", 0),
        "justificativa_verificacao": resultado.get("justificativa_verificacao", ""),
        "contexto_insuficiente": not resultado.get("contexto_valido", False),
        "usou_fallback_web": resultado["usou_fallback_web"],
        "latencia_segundos": round(resultado["latencia_segundos"], 2),
    }


# Conjunto de avaliação do Artigo 1 (Tabela 1). Confira se o texto é idêntico ao que
# foi enviado ao sistema convencional.
PERGUNTAS = [
    ("Q1", "Qual é o objetivo do curso de Sistemas de Informação?"),
    ("Q2", "Como os eventos do CEFET-MG contribuem para a formação dos alunos de SI?"),
    ("Q3", "Como posso ingressar no curso?"),
    ("Q4", "Qual é a duração mínima para se formar em SI?"),
    ("Q5", "Quais são os pré-requisitos para fazer o estágio obrigatório?"),
    ("Q6", "O curso de SI tem disciplinas de programação?"),
    ("Q7", "É obrigatório fazer ENEM para entrar no curso?"),
    ("Q8", "Quais matérias do curso eu preciso para trabalhar com dados?"),
    ("Q9", "Como posso obter certificado de atividades complementares?"),
    ("Q10", "Qual é o perfil profissional formado pelo curso?"),
    ("Q11", "Quais áreas um aluno de SI pode atuar?"),
    ("Q12", "Quantas horas tem o curso de Sistemas de Informação?"),
    ("Q13", "Como o curso evolui ao longo dos períodos?"),
    ("Q14", "Quais são as etapas para desenvolver o TCC no CEFET-MG?"),
    ("Q15", "Como funciona o TCC no curso de Sistemas de Informação do CEFET-MG?"),
    ("Q16", "Qual é a carga horária do TCC?"),
    ("Q17", "Em qual período posso iniciar o TCC?"),
    ("Q18", "Qual a diferença entre trancamento e cancelamento de matrícula?"),
    ("Q19", "O que preciso fazer para participar de um projeto de pesquisa?"),
    ("Q20", "Quais são as responsabilidades do aluno durante o estágio?"),
]

CAMPOS_CSV = [
    "id", "pergunta", "repeticao", "resposta", "plano_busca", "contextos", "n_chunks",
    "iteracoes_verificacao", "score_relevancia", "justificativa_verificacao",
    "contexto_insuficiente", "usou_fallback_web", "latencia_segundos", "fontes", "erro",
]


def configuracao_atual(extra: Optional[dict] = None) -> dict:
    cfg = {
        "modelo": MODELO_GERADOR,
        "temperatura_sintetizador": TEMPERATURA_SINTETIZADOR,
        "temperatura_planejador": TEMPERATURA_PLANEJADOR,
        "temperatura_verificador": TEMPERATURA_VERIFICADOR,
        "k_base": K_BASE,
        "k_incremento_retentativa": K_INCREMENTO_RETENTATIVA,
        "max_subperguntas": MAX_SUBPERGUNTAS,
        "max_iteracoes_verificacao": MAX_ITERACOES_VERIFICACAO,
        "prompt_sintetizador": "artigo1" if USAR_PROMPT_ARTIGO1 else "completude",
    }
    if extra:
        cfg.update(extra)
    return cfg


def executar_benchmark(
    app_grafo,
    perguntas=PERGUNTAS,
    n_repeticoes: int = 1,
    caminho_saida: str = "resultados_agentico.csv",
    extra_config: Optional[dict] = None,
) -> str:
    """Roda todas as perguntas (n_repeticoes vezes cada) e salva CSV + JSON da configuração.

    O CSV é regravado após cada pergunta, para não perder progresso se a API falhar.
    As colunas plano_busca, contextos e fontes são JSON (use json.loads ao ler).
    """
    cfg = configuracao_atual(extra_config)
    print("Configuração:", json.dumps(cfg, ensure_ascii=False))
    with open(caminho_saida + ".config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    linhas = []
    for qid, pergunta in perguntas:
        for rep in range(1, n_repeticoes + 1):
            print(f"\n=== {qid} (repetição {rep}/{n_repeticoes}) ===")
            try:
                r = responder_com_rag_agentico(app_grafo, pergunta)
                erro = ""
            except Exception as e:  # registra e segue para a próxima
                r = {"resposta": "", "plano_busca": [], "contextos": [], "n_chunks": 0,
                     "iteracoes_verificacao": 0, "score_relevancia": "",
                     "justificativa_verificacao": "", "contexto_insuficiente": "",
                     "usou_fallback_web": "", "latencia_segundos": "", "fontes": []}
                erro = str(e)
                print(f"[ERRO] {qid}: {erro}")

            linhas.append({
                "id": qid,
                "pergunta": pergunta,
                "repeticao": rep,
                "resposta": r["resposta"],
                "plano_busca": json.dumps(r["plano_busca"], ensure_ascii=False),
                "contextos": json.dumps(r["contextos"], ensure_ascii=False),
                "n_chunks": r["n_chunks"],
                "iteracoes_verificacao": r["iteracoes_verificacao"],
                "score_relevancia": r["score_relevancia"],
                "justificativa_verificacao": r["justificativa_verificacao"],
                "contexto_insuficiente": r["contexto_insuficiente"],
                "usou_fallback_web": r["usou_fallback_web"],
                "latencia_segundos": r["latencia_segundos"],
                "fontes": json.dumps(r["fontes"], ensure_ascii=False),
                "erro": erro,
            })

        # utf-8-sig: abre corretamente no Excel
        with open(caminho_saida, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
            writer.writeheader()
            writer.writerows(linhas)

    print(f"\nConcluído. Resultados em {caminho_saida}")
    return caminho_saida


# ============================================================
# 8. EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    # AJUSTE este import para o nome real do arquivo do Artigo 1.
    from rag_core import carregar_vectorstore_default

    vectorstore = carregar_vectorstore_default()
    grafo = construir_grafo(vectorstore, pesquisar_na_web_fn=None, permitir_fallback_web=False)

    executar_benchmark(
        grafo,
        n_repeticoes=3,
        caminho_saida="resultados_agentico.csv",
        extra_config={"fallback_web": False},
    )