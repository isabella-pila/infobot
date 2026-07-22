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


class GraphState(TypedDict):
    pergunta: str
    pergunta_original: str
    plano_busca: List[str]
    plano_anterior: List[str]

    contexto_recuperado: Annotated[Sequence[Document], operator.add]
    contexto_valido: bool
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

PROMPT_PLANEJADOR = """<papel>
Você é o Agente Planejador de um sistema RAG de suporte acadêmico do CEFET-MG, Campus Varginha.
Sua única função é transformar a pergunta do usuário em um plano de busca vetorial eficiente.
Você NÃO responde à pergunta — apenas decide o que precisa ser buscado.
</papel>

<tarefa>
Classifique a pergunta e produza as consultas de busca:

1. SIMPLES (fato único e direto) → retorne exatamente 1 consulta: a própria pergunta,
   reescrita em termos objetivos e ricos em palavras-chave para busca vetorial.
2. COMPLEXA (exige síntese de múltiplas fontes, comparação ou raciocínio multi-etapa)
   → decomponha em 2 a 4 sub-perguntas independentes e não sobrepostas que, juntas,
   cubram tudo o que é necessário para responder.
</tarefa>

<regras>
- Cada sub-pergunta deve ser autossuficiente (não pode depender do texto de outra).
- Prefira termos institucionais concretos (ex: "carga horária TCC", "pré-requisitos estágio").
- Evite palavras vazias ("por favor", "gostaria de saber").
- NUNCA responda à pergunta nem adicione comentários.
- NÃO repita o cabeçalho "Sub-perguntas:"; escreva apenas as consultas.
- Saída: uma consulta por linha, sem numeração, sem marcadores, sem texto extra.
</regras>

<exemplos>
Pergunta: "Qual a carga horária do TCC?"
Sub-perguntas:
carga horária do Trabalho de Conclusão de Curso TCC

Pergunta: "Como o curso de Sistemas de Informação evolui do 1º ao último período?"
Sub-perguntas:
disciplinas e conteúdos dos períodos iniciais do curso Sistemas de Informação
disciplinas e conteúdos dos períodos intermediários do curso
disciplinas, TCC e estágio nos períodos finais do curso

Pergunta: "Preciso fazer estágio obrigatório e quantas horas?"
Sub-perguntas:
obrigatoriedade do estágio supervisionado no curso
carga horária mínima exigida para o estágio supervisionado
</exemplos>

Pergunta do usuário: {pergunta}

Sub-perguntas:"""


PROMPT_PLANEJADOR_REPLAN = """<papel>
Você é o Agente Planejador de um sistema RAG de suporte acadêmico do CEFET-MG, Campus Varginha.
Sua função é gerar um plano de busca vetorial. Você NÃO responde à pergunta.
</papel>

<situacao>
Uma tentativa anterior de busca FALHOU: o contexto recuperado foi julgado insuficiente.
O verificador apontou a seguinte lacuna:
<lacuna>
{justificativa}
</lacuna>

As consultas já tentadas (e que NÃO devem ser repetidas) foram:
<consultas_anteriores>
{plano_anterior}
</consultas_anteriores>
</situacao>

<tarefa>
Gere um NOVO plano de busca com 2 a 4 consultas DIFERENTES das anteriores, mais
específicas e reformuladas com sinônimos/termos institucionais alternativos, focadas
em cobrir exatamente a lacuna apontada acima.
</tarefa>

<regras>
- Cada consulta deve ser autossuficiente e rica em palavras-chave.
- NÃO repita nenhuma das consultas anteriores nem apenas troque a ordem das palavras.
- Prefira termos institucionais concretos e variações que possam casar com outros trechos.
- NUNCA responda à pergunta nem adicione comentários.
- NÃO repita o cabeçalho "Sub-perguntas:"; escreva apenas as consultas.
- Saída: uma consulta por linha, sem numeração, sem marcadores, sem texto extra.
</regras>

Pergunta do usuário: {pergunta}

Sub-perguntas:"""


PROMPT_VERIFICADOR = """<papel>
Você é o Agente Verificador de um sistema RAG institucional do CEFET-MG.
Sua função é julgar, de forma rigorosa e cética, se o CONTEXTO recuperado
contém informação SUFICIENTE e EXPLÍCITA para responder à PERGUNTA sem alucinação.
</papel>

<criterios>
Decida INSUFICIENTE se QUALQUER item abaixo for verdadeiro:
- O contexto não menciona o tema central da pergunta.
- Responder exigiria inferir, estimar ou completar dados ausentes.
- Só há informação parcial (responde parte da pergunta, não o todo).
- O contexto é ambíguo ou contraditório sobre o ponto perguntado.

Decida SUFICIENTE somente se a resposta completa puder ser extraída
diretamente do contexto, sem adicionar conhecimento externo.
</criterios>

<raciocinio>
Antes de decidir, pergunte-se: "Se eu respondesse usando apenas este contexto,
precisaria inventar ou supor algo?" Se sim, é INSUFICIENTE.
</raciocinio>

<formato_de_saida>
Responda EXATAMENTE neste formato, em duas linhas, sem nenhum texto extra:
VEREDITO: SUFICIENTE
JUSTIFICATIVA: <uma frase objetiva; quando INSUFICIENTE, aponte o que falta>
</formato_de_saida>

Pergunta: {pergunta}

Contexto recuperado:
{contexto}
"""


PROMPT_SINTETIZADOR = """<papel>
Você é o "Infobot", assistente virtual oficial do CEFET-MG, Campus Varginha,
atuando como Agente Sintetizador de um pipeline multi-agente.
</papel>

<regra_de_ouro>
Use EXCLUSIVAMENTE as informações do CONTEXTO abaixo. É terminantemente proibido usar
conhecimento externo, suposições ou generalizações. Se a informação necessária não estiver
no contexto, diga claramente: "Não encontrei essa informação nos documentos institucionais
disponíveis." — nunca invente.
</regra_de_ouro>

<instrucoes>
- Responda em português do Brasil, com tom cordial, claro e didático.
- Estruture a resposta com títulos curtos e/ou bullet points quando houver múltiplos pontos;
  use parágrafo único para respostas simples.
- Se o contexto vier de várias sub-buscas, integre tudo em UMA resposta coesa,
  sem repetições e sem contradições.
- COMPLETUDE É PRIORIDADE: extraia e inclua TODAS as informações do contexto que sejam
  relevantes para a pergunta. Se houver listas (disciplinas, requisitos, documentos, etapas,
  exceções, prazos), reproduza TODOS os itens — nunca resuma com "entre outros" ou "etc."
  quando os itens completos estiverem no contexto.
- Seja preciso com números, prazos, cargas horárias e nomes: reproduza-os exatamente como no contexto.
- Não mencione "o contexto"/"os documentos" no corpo da resposta; apenas entregue a
  informação como um atendente experiente faria. (Reformular com suas palavras é permitido,
  desde que nenhum dado seja omitido.)
- Não invente links, e-mails, telefones ou setores que não estejam no contexto.
</instrucoes>

<contexto>
{context}
</contexto>

<pergunta>
{question}
</pergunta>

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
    """Decompõe a pergunta em sub-consultas. Na primeira passagem usa o prompt base;
    em replanejamentos, usa o prompt que considera a lacuna e o plano anterior."""
    pergunta = state.get("pergunta_original") or state["pergunta"]
    iteracoes = state.get("iteracoes_verificacao", 0)
    plano_anterior = state.get("plano_busca", [])

    llm = get_llm(temperature=0.2)

    if iteracoes > 0 and plano_anterior:
        # Replanejamento: usa a justificativa do Verificador para diversificar a busca
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_PLANEJADOR_REPLAN))
        resultado = chain.invoke({
            "pergunta": pergunta,
            "justificativa": state.get("justificativa_verificacao", "(não especificada)"),
            "plano_anterior": "\n".join(f"- {p}" for p in plano_anterior),
        })
    else:
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(PROMPT_PLANEJADOR))
        resultado = chain.invoke({"pergunta": pergunta})

    texto = resultado.get("text", pergunta).strip()

    sub_perguntas = [l.strip("-•* ").strip() for l in texto.split("\n") if l.strip()]
    # Remove eventual cabeçalho ecoado pelo modelo
    sub_perguntas = [s for s in sub_perguntas if s.lower() != "sub-perguntas:"]
    if not sub_perguntas:
        sub_perguntas = [pergunta]

    return {
        "pergunta_original": pergunta,
        "plano_busca": sub_perguntas,
        "plano_anterior": plano_anterior,
    }


def agente_recuperador(state: GraphState, vectorstore, k: int = 8) -> dict:
    """Executa busca vetorial no FAISS para cada sub-consulta do plano gerado pelo Planejador."""
    docs_encontrados: List[Document] = []
    for sub_pergunta in state["plano_busca"]:
        try:
            docs_encontrados.extend(vectorstore.similarity_search(sub_pergunta, k=k))
        except Exception as e:
            print(f"[Recuperador] erro na busca '{sub_pergunta}': {e}")

    # Remove duplicatas (mesmo chunk pode ser recuperado por sub-perguntas diferentes)
    vistos, docs_unicos = set(), []
    for d in docs_encontrados:
        chave = d.page_content[:200]
        if chave not in vistos:
            vistos.add(chave)
            docs_unicos.append(d)

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

    valido = bool(re.search(r"VEREDITO:\s*SUFICIENTE", texto, re.IGNORECASE))
    m = re.search(r"JUSTIFICATIVA:\s*(.+)", texto, re.IGNORECASE)

    return {
        "contexto_valido": valido,
        "justificativa_verificacao": m.group(1).strip() if m else "",
        "iteracoes_verificacao": state.get("iteracoes_verificacao", 0) + 1,
    }


def agente_sintetizador(state: GraphState) -> dict:
    """Gera a resposta final a partir do contexto já validado pelo Verificador."""
    llm = get_llm(temperature=0.3, max_output_tokens=8192)
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
    ainda não foi validado. Reaproveita a mesma função pesquisar_na_web do Artigo 1
    (query inteligente -> Serper -> extração httpx -> síntese).
    """
    llm = get_llm(temperature=0.3)
    resposta_web = pesquisar_na_web_fn(llm, state["pergunta_original"])

    return {
        "resposta_final": resposta_web or "Não foi possível encontrar uma resposta confiável.",
        "usou_fallback_web": True,
        "latencia_segundos": time.time() - state.get("inicio_execucao", time.time()),
    }


# ============================================================
# 5. ROTEAMENTO CONDICIONAL
# ============================================================
def decidir_proximo_passo(state: GraphState) -> str:
    if state["contexto_valido"]:
        return "sintetizar"
    if state["iteracoes_verificacao"] >= MAX_ITERACOES_VERIFICACAO:
        return "busca_web_fallback"
    return "recuperar_novamente"


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
            "recuperar_novamente": "planejador",
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
    Executa o grafo para uma pergunta e retorna resposta + métricas operacion quero it embora socoorp 
    
    
    roriimd kais,
    já no formato que você vai usar para alimentar a Tabela comparativa do Artigo 2
    (mesmo padrão da Tabela 2 do Artigo 1, mas com colunas extras de iterações/latência).
    """
    estado_inicial: GraphState = {
        "pergunta": pergunta,
        "pergunta_original": pergunta,
        "plano_busca": [],
        "plano_anterior": [],
        "contexto_recuperado": [],
        "contexto_valido": False,
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

if __name__ == "__main__":
    pass