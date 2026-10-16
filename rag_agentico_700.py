"""
rag_agentico_700.py
===================
Artigo 2 - Assistente Virtual com RAG Agêntico e Orquestração Multi-Agentes (LangGraph).
Quatro agentes: Planejador, Recuperador, Verificador e Sintetizador.

VERSÃO PARA RODAR DE NOVO, ALINHADA AO ARTIGO 1
-----------------------------------------------
Em relação à versão anterior, mudam só as condições do experimento (a arquitetura é a mesma):
  1. Índice com a segmentação do Artigo 1 (700 caracteres, sobreposição 200), via indice_700.py,
     que confere o tamanho dos trechos antes de rodar.
  2. O Sintetizador usa o RAG_PROMPT do Artigo 1 (USAR_PROMPT_ARTIGO1 = True). Assim o prompt
     de geração é idêntico ao do convencional e o efeito do Planejador/Verificador fica isolado.
  3. O CSV registra chars_contexto e contexto_truncado (> 7.000 caracteres, limite do Context
     Relevance do RAGAS 0.2), para preencher a coluna "Trunc." da planilha.
  4. Retomada: se a cota da API acabar, rode o mesmo comando de novo; só o que falta é refeito.
     Também espera e tenta de novo quando a cota por minuto estoura.
  5. A configuração (incluindo hash do prompt e tamanho dos trechos) é gravada em .config.json
     e o script se recusa a misturar configurações diferentes no mesmo CSV.

Uso (na pasta do projeto, com indice_700.py e o .env com GOOGLE_API_KEY):
    python rag_agentico_700.py --apenas Q1 Q5                       # teste barato
    python rag_agentico_700.py                                      # as 20 perguntas
    python rag_agentico_700.py --apenas Q1 --repeticoes 3 --saida resultados_agentico_700_q1.csv
    python rag_agentico_700.py --fora-escopo                        # F1 a F5 (sem cobertura)
"""

import argparse
import csv
import hashlib
import json
import operator
import os
import re
import sys
import time
from typing import Annotated, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

csv.field_size_limit(2**31 - 1)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO  (registre estes valores na Tabela I do artigo)
# ============================================================
MODELO_GERADOR = "gemini-3.5-flash-lite"      # o mesmo do Artigo 1 e do convencional
TEMPERATURA_SINTETIZADOR = 0.7
TEMPERATURA_PLANEJADOR = 0.2
TEMPERATURA_VERIFICADOR = 0.0

K_BASE = 4                       # top-k por consulta (Artigo 1: k = 4)
K_INCREMENTO_RETENTATIVA = 4     # acréscimo de k a cada reprovação do Verificador
MAX_SUBPERGUNTAS = 4
MAX_ITERACOES_VERIFICACAO = 3    # nº máximo de verificações (= até 2 retentativas)
MAX_CHARS_VERIFICADOR = 60000
MAX_OUTPUT_TOKENS_SINTETIZADOR = 8192
LIMITE_CONTEXT_RELEVANCE = 7000  # o Context Relevance do RAGAS 0.2 lê só os 7.000 primeiros caracteres

# ----------------------------------------------------------------
# RAG_PROMPT do Artigo 1 (texto exato, com os mesmos erros de digitação do original).
# Se existir rag_convencional_com_ragas.py na pasta, o prompt é importado de lá, para
# garantir que os dois sistemas usem o MESMO texto (confira o prompt_sha256 nos dois .config.json).
# ----------------------------------------------------------------
RAG_PROMPT_ARTIGO1: Optional[str] = """
# DIRETIVA PRINCIPAL
Você é o "Infobot", o assistente virtual do CEFET-MG, campus Varginha. Sua identidade é a de um colega experiente de Sistemas de Informação, que domina os assuntos do curso e está sempre disposto a auxiliar os estudantes com clareza e objetividade.

Sua missão é traduzir materiais de estudo complexos em resumos e explicações didáticas e fáceis de compreender.
sempre traga as informações de forma organizada.
---
# PERSONA E TOM DE VOZ

    quando for sobre matriz curricular ou materias sempre mande todas sem exeção 
1.  **Linguagem:** Comunique-se de forma clara, objetiva e encorajadora. seja legal pode usar algumas girias mas não exagere 
2.  **Atitude:** Mantenha sempre uma postura positiva, paciente e motivadora. O objetivo é fazer com que o estudante se sinta confiante em sua capacidade de aprender qualquer tópico.

---
# DIRETRIZES DE EXECUÇÃO

1.  **Fonte de Verdade Primária:** Sua primeira e principal fonte de informação é sempre o conteúdo fornecido no `{{context}}` (os arquivos que o usuário enviou).

2.  **Análise Estruturada (O Ponto-Chave):** Ao receber um material no `{{context}}` para resumir ou explicar, sua primeira ação é analisá-lo para identificar os seguintes pontos:
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


# REGRAS DE INTERAÇÃO

* **Comunicação Direta:** Não há necessidade de saudações ("Olá", "Bom dia") a cada nova pergunta. Mantenha a conversa fluida.
* **Agradecimentos:** Se o usuário agradecer com "obrigado(a)" ou similar, responda de forma educada e prestativa, com algo como "De nada! Se precisar de mais alguma coisa, é só chamar." ou "Disponha!".
* **Proatividade:** Se a pergunta do estudante for muito ampla ("me explica sobre Redes de Computadores"), ajude-o a especificar. Exemplo: *"Claro! Redes é um assunto bem grande. Para te ajudar melhor, você quer saber sobre os modelos de camadas como OSI/TCP, tipos de topologia, ou talvez exemplos de protocolos como HTTP e DNS?"*.
**Quando a pessoa so cumprimentar ("ola", "bom dia") cumprimentar também e so isso
**se as pessoas perguntares coisas fora da area do cefet, falar que não pode responder.**


Contexto:
{context}

Pergunta:
{question}
"""
try:  # fonte única do prompt, se o script do convencional estiver na pasta
    from rag_convencional_com_ragas import RAG_PROMPT as _PROMPT_CONV

    RAG_PROMPT_ARTIGO1 = _PROMPT_CONV
    print("[Prompt] RAG_PROMPT importado de rag_convencional_com_ragas.py")
except Exception:
    pass

USAR_PROMPT_ARTIGO1 = True       # True = mesmo prompt do convencional (comparação principal)


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
# Os exemplos abaixo NÃO coincidem com nenhuma pergunta do conjunto de avaliação.
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

# Prompt alternativo (ênfase em completude). Só é usado se USAR_PROMPT_ARTIGO1 = False e,
# nesse caso, deve ser descrito no artigo como uma condição adicional.
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


def _texto_prompt_sintetizador() -> str:
    if USAR_PROMPT_ARTIGO1:
        if not RAG_PROMPT_ARTIGO1:
            raise ValueError("USAR_PROMPT_ARTIGO1=True, mas RAG_PROMPT_ARTIGO1 está vazio.")
        return RAG_PROMPT_ARTIGO1
    return PROMPT_SINTETIZADOR


def _prompt_sintetizador() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(_texto_prompt_sintetizador())


def _hash_prompt() -> str:
    return hashlib.sha256(_texto_prompt_sintetizador().encode("utf-8")).hexdigest()[:12]


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

    return {"pergunta_original": pergunta, "plano_busca": sub_perguntas}


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
            if _cota_esgotada(str(e)) or _cota_por_minuto(str(e)):
                raise  # deixa o laço externo esperar/parar, em vez de seguir com contexto vazio

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
        d.metadata.get("source", "base institucional") for d in state["contexto_recuperado"]
    })

    return {
        "resposta_final": resultado.get("output_text", "Não foi possível gerar uma resposta."),
        "fontes": fontes,
        "latencia_segundos": time.time() - state.get("inicio_execucao", time.time()),
    }


def agente_busca_web_fallback(state: GraphState, pesquisar_na_web_fn) -> dict:
    """Fallback web (somente se permitir_fallback_web=True em construir_grafo)."""
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
        fontes = list({d.metadata.get("source", "base institucional") for d in contexto_local})

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
    if state["contexto_valido"]:                                  # nota 2: responde
        return "sintetizar"
    if state["iteracoes_verificacao"] < MAX_ITERACOES_VERIFICACAO:  # ainda há tentativas
        return "recuperar_novamente"
    if state.get("score_relevancia", 0) >= 1:                     # esgotou, nota 1: responde
        return "sintetizar"
    if permitir_fallback_web:                                     # nota 0
        return "busca_web_fallback"
    return "sintetizar"                                           # sinaliza contexto_insuficiente


# ============================================================
# 6. MONTAGEM DO GRAFO
# ============================================================
def construir_grafo(vectorstore, pesquisar_na_web_fn=None, permitir_fallback_web: bool = False):
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

    destinos = {"sintetizar": "sintetizador", "recuperar_novamente": "recuperador"}
    if permitir_fallback_web:
        workflow.add_node(
            "busca_web_fallback",
            lambda s: agente_busca_web_fallback(s, pesquisar_na_web_fn),
        )
        workflow.add_edge("busca_web_fallback", END)
        destinos["busca_web_fallback"] = "busca_web_fallback"

    workflow.add_conditional_edges(
        "verificador", lambda s: decidir_proximo_passo(s, permitir_fallback_web), destinos
    )
    workflow.add_edge("sintetizador", END)
    return workflow.compile()


# ============================================================
# 7. EXECUÇÃO, MÉTRICAS E RETOMADA
# ============================================================
def _cota_esgotada(erro: str) -> bool:
    """Cota DIÁRIA: não adianta esperar."""
    return ("PerDay" in erro) or ("per day" in erro.lower()) or bool(re.search(r"retry in \d+h", erro))


def _cota_por_minuto(erro: str) -> bool:
    return any(s in erro for s in ("429", "RESOURCE_EXHAUSTED", "quota")) and not _cota_esgotada(erro)


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

    # Todos os trechos acumulados são os que o Sintetizador recebeu: use esta lista
    # como retrieved_contexts no RAGAS.
    contextos = [d.page_content for d in resultado.get("contexto_recuperado", [])]
    chars = sum(len(c) for c in contextos)

    return {
        "resposta": resultado["resposta_final"],
        "fontes": resultado.get("fontes", []),
        "plano_busca": resultado.get("plano_busca", []),
        "contextos": contextos,
        "n_chunks": len(contextos),
        "chars_contexto": chars,
        "contexto_truncado": chars > LIMITE_CONTEXT_RELEVANCE,
        "iteracoes_verificacao": resultado["iteracoes_verificacao"],
        "score_relevancia": resultado.get("score_relevancia", 0),
        "justificativa_verificacao": resultado.get("justificativa_verificacao", ""),
        "contexto_insuficiente": not resultado.get("contexto_valido", False),
        "usou_fallback_web": resultado["usou_fallback_web"],
        "latencia_segundos": round(resultado["latencia_segundos"], 2),
    }


# Conjunto de avaliação do Artigo 1 (Tabela 1).
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

# Consultas sem cobertura documental (Artigo 1, Tabela 2). Confira se o texto é idêntico ao
# que você enviou ao sistema convencional no Artigo 1.
FORA_DO_ESCOPO = [
    ("F1", "Qual é a nota de corte do SiSU para Sistemas de Informação em Varginha?"),
    ("F2", "Quantas horas de atividades complementares eu já cumpri?"),
    ("F3", "Como funciona o TCC no curso de Engenharia Elétrica do CEFET-MG?"),
    ("F4", "Me explique como funciona o algoritmo de Dijkstra."),
    ("F5", "Por que o TCC do curso exige defesa em inglês?"),
]

CAMPOS_CSV = [
    "id", "pergunta", "repeticao", "resposta", "plano_busca", "contextos", "n_chunks",
    "chars_contexto", "contexto_truncado", "iteracoes_verificacao", "score_relevancia",
    "justificativa_verificacao", "contexto_insuficiente", "usou_fallback_web",
    "latencia_segundos", "fontes", "erro",
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
        "prompt_sha256": _hash_prompt(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def _ler_csv(caminho: str) -> list:
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def executar_benchmark(
    app_grafo,
    perguntas=PERGUNTAS,
    n_repeticoes: int = 1,
    caminho_saida: str = "resultados_agentico_700.csv",
    extra_config: Optional[dict] = None,
    pausa: float = 3.0,
) -> str:
    """Roda as perguntas (n_repeticoes vezes cada), salva CSV + JSON da configuração e retoma
    de onde parou se a execução for interrompida (cota da API, queda de rede etc.)."""
    cfg = configuracao_atual(extra_config)
    caminho_cfg = caminho_saida + ".config.json"

    # Não mistura configurações diferentes no mesmo CSV.
    if os.path.exists(caminho_cfg) and os.path.exists(caminho_saida):
        with open(caminho_cfg, encoding="utf-8") as f:
            anterior = {k: v for k, v in json.load(f).items() if k != "info_indice"}
        atual = {k: v for k, v in cfg.items() if k != "info_indice"}
        if anterior != atual:
            sys.exit("A configuração mudou desde a execução anterior deste CSV. Apague o CSV e o "
                     f".config.json ou use outro --saida.\nAnterior: {anterior}\nAtual:    {atual}")

    print("Configuração:", json.dumps(cfg, ensure_ascii=False))
    with open(caminho_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    ordem = {(qid, rep): i for i, (qid, _) in enumerate(perguntas) for rep in range(1, n_repeticoes + 1)}
    linhas = {}
    for r in _ler_csv(caminho_saida):
        chave = (r["id"], int(r.get("repeticao") or 1))
        if r.get("resposta") and not r.get("erro"):   # só mantém o que foi concluído
            linhas[chave] = r

    def salvar():
        ordenadas = sorted(linhas.items(), key=lambda kv: (ordem.get(kv[0], 10**6), kv[0][1]))
        with open(caminho_saida, "w", newline="", encoding="utf-8-sig") as f:  # abre no Excel
            w = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
            w.writeheader()
            w.writerows([v for _, v in ordenadas])

    parar = False
    for qid, pergunta in perguntas:
        for rep in range(1, n_repeticoes + 1):
            if (qid, rep) in linhas:
                print(f"=== {qid} (rep {rep}) já concluída, pulando ===")
                continue
            print(f"\n=== {qid} (repetição {rep}/{n_repeticoes}) ===")
            r, erro = None, ""
            for tentativa in range(1, 4):
                try:
                    r = responder_com_rag_agentico(app_grafo, pergunta)
                    erro = ""
                    break
                except Exception as e:
                    erro = str(e)
                    print(f"[ERRO] {qid} (tentativa {tentativa}/3): {erro[:200]}")
                    if _cota_esgotada(erro):
                        parar = True
                        break
                    if _cota_por_minuto(erro) and tentativa < 3:
                        print("  cota por minuto: aguardando 65 s")
                        time.sleep(65)
                    else:
                        break
            if parar:
                print("\nCota diária do Gemini esgotada. O progresso foi salvo; rode o mesmo comando "
                      "de novo depois e ele retoma de onde parou.")
                break
            if r is None:
                print(f"  {qid} falhou e não foi salva; será refeita na próxima rodada.")
                continue

            linhas[(qid, rep)] = {
                "id": qid, "pergunta": pergunta, "repeticao": rep,
                "resposta": r["resposta"],
                "plano_busca": json.dumps(r["plano_busca"], ensure_ascii=False),
                "contextos": json.dumps(r["contextos"], ensure_ascii=False),
                "n_chunks": r["n_chunks"], "chars_contexto": r["chars_contexto"],
                "contexto_truncado": r["contexto_truncado"],
                "iteracoes_verificacao": r["iteracoes_verificacao"],
                "score_relevancia": r["score_relevancia"],
                "justificativa_verificacao": r["justificativa_verificacao"],
                "contexto_insuficiente": r["contexto_insuficiente"],
                "usou_fallback_web": r["usou_fallback_web"],
                "latencia_segundos": r["latencia_segundos"],
                "fontes": json.dumps(r["fontes"], ensure_ascii=False),
                "erro": "",
            }
            salvar()  # grava a cada pergunta
            if pausa > 0:
                time.sleep(pausa)
        if parar:
            break

    salvar()
    faltam = [k for k in ordem if k not in linhas]
    print(f"\nResultados em {caminho_saida} | concluídas: {len(linhas)} | faltam: {len(faltam)} {faltam if faltam else ''}")
    return caminho_saida


# ============================================================
# 8. EXECUÇÃO
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="RAG agêntico (Artigo 2) no índice de 700 caracteres")
    ap.add_argument("--apenas", nargs="*", default=[], help="IDs a rodar (ex.: Q1 Q5)")
    ap.add_argument("--repeticoes", type=int, default=1)
    ap.add_argument("--fora-escopo", action="store_true", help="roda F1 a F5 em vez de Q1 a Q20")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--reconstruir-indice", action="store_true")
    ap.add_argument("--pausa", type=float, default=3.0, help="segundos entre perguntas")
    args = ap.parse_args()

    if not GOOGLE_API_KEY:
        sys.exit("GOOGLE_API_KEY não encontrada. Coloque GOOGLE_API_KEY=... no .env.")

    base = FORA_DO_ESCOPO if args.fora_escopo else PERGUNTAS
    perguntas = base
    if args.apenas:
        sel = {x.upper() for x in args.apenas}
        perguntas = [p for p in base if p[0] in sel]
    if not perguntas:
        sys.exit("Nenhuma pergunta selecionada.")
    saida = args.saida or ("resultados_agentico_700_fora_escopo.csv" if args.fora_escopo
                           else "resultados_agentico_700.csv")

    from indice_700 import carregar_vectorstore_700, info_indice

    vectorstore = carregar_vectorstore_700(reconstruir=args.reconstruir_indice)  # confere 700 antes de rodar
    grafo = construir_grafo(vectorstore, pesquisar_na_web_fn=None, permitir_fallback_web=False)

    executar_benchmark(
        grafo,
        perguntas=perguntas,
        n_repeticoes=args.repeticoes,
        caminho_saida=saida,
        extra_config={"fallback_web": False, "info_indice": info_indice(vectorstore)},
        pausa=args.pausa,
    )


if __name__ == "__main__":
    main()