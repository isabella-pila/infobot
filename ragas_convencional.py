"""
rag_convencional_com_ragas.py
=============================
Faz, em UM comando só e sem arquivos intermediários para você montar, o que o Artigo 1 fazia:
para cada pergunta, recupera os trechos, gera a resposta e já avalia com o RAGAS.

Sistema convencional do Artigo 1, rodado sobre o MESMO índice do agêntico
(rag_core.carregar_vectorstore_default):
    modelo gemini-2.5-flash | temperatura 0,7 | top-k = 4, ciclo único | stuffing |
    prompt = RAG_PROMPT usado nas 20 perguntas do Artigo 1

Avaliação (igual à do Artigo 1): RAGAS 0.2.x | avaliador GPT-3.5-turbo, temperatura 0 |
embeddings paraphrase-multilingual-MiniLM-L12-v2 | Faithfulness, Answer Relevancy e Context Relevance.

Uso (na pasta do projeto, com o rag_core.py, o .env e o ragas 0.2.x no mesmo Python):
    python rag_convencional_com_ragas.py --apenas Q1 Q5      # teste barato (2 perguntas)
    python rag_convencional_com_ragas.py                     # as 20 perguntas

O .env precisa ter GOOGLE_API_KEY (geração) e OPENAI_API_KEY (avaliação).

Segurança:
  * Confere a versão do ragas e faz uma chamada mínima à OpenAI ANTES de gerar qualquer resposta.
  * Grava a cada etapa. Se a cota do Gemini ou o crédito da OpenAI acabarem, para e retoma depois,
    SEM gerar de novo as respostas que já existem (só avalia o que falta).
  * Bloqueia a mistura de configurações diferentes no mesmo CSV.
  * No fim, mostra a tabela: convencional (novo) x agêntico x Artigo 1.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time

from dotenv import load_dotenv

csv.field_size_limit(2**31 - 1)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------- geração (igual ao Artigo 1) ----------
MODELO_GERADOR = "gemini-2.5-flash"
TEMPERATURA_GERACAO = 0.7
K = 4
MAX_OUTPUT_TOKENS = 8192

# ---------- avaliação (igual ao Artigo 1) ----------
MODELO_AVALIADOR = "gpt-3.5-turbo"
TEMPERATURA_AVALIADOR = 0
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIMITE_CONTEXT_RELEVANCE = 7000   # o Context Relevance do ragas 0.2 lê só os 7000 primeiros caracteres

# Prompt usado nas 20 perguntas do Artigo 1 (sem a busca na web e sem {chat_history}).
# Confira se é idêntico ao do seu rag_core.py da época. O hash fica registrado na configuração.
RAG_PROMPT = """
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

# Resultados do Artigo 1 (Tabela 1): (Faithfulness, Answer Relevancy, Context Relevance)
ARTIGO1 = {
    "Q1": (1.00, 0.73, 1.00), "Q2": (0.00, 0.81, 0.50), "Q3": (0.80, 0.57, 0.50),
    "Q4": (0.50, 0.52, 0.50), "Q5": (0.00, 0.78, 1.00), "Q6": (0.83, 0.63, 1.00),
    "Q7": (0.75, 0.62, 1.00), "Q8": (0.00, 0.81, 0.00), "Q9": (0.78, 0.82, 0.75),
    "Q10": (0.50, 0.72, 0.25), "Q11": (0.87, 0.66, 0.75), "Q12": (0.67, 0.75, 1.00),
    "Q13": (0.78, 0.56, 1.00), "Q14": (1.00, 0.81, 1.00), "Q15": (1.00, 0.81, 1.00),
    "Q16": (0.50, 0.65, 1.00), "Q17": (0.50, 0.69, 1.00), "Q18": (0.75, 0.56, 1.00),
    "Q19": (0.88, 0.69, 0.25), "Q20": (0.50, 0.86, 0.25),
}

METRICAS = ("faithfulness", "answer_relevancy", "context_relevance")
CAMPOS = ["id", "pergunta", "resposta", "contextos", "n_chunks", "latencia_segundos",
          *METRICAS, "contexto_truncado", "completo", "erro"]


# ============================================================
# Auxiliares
# ============================================================
def e_nan(x):
    return x is None or x == "" or (isinstance(x, float) and math.isnan(x))


def _cota_diaria_esgotada(erro):
    return ("PerDay" in erro) or ("per day" in erro.lower()) or bool(re.search(r"retry in \d+h", erro))


def _cota_por_minuto(erro):
    return any(s in erro for s in ("429", "RESOURCE_EXHAUSTED", "quota")) and not _cota_diaria_esgotada(erro)


def _modelo_indisponivel(erro):
    e = erro.lower()
    return "404" in erro and ("no longer available" in e or "not found" in e or "is not supported" in e)


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def configuracao():
    try:
        versao = versao_ragas() or "desconhecida"
    except Exception:
        versao = "desconhecida"
    return {
        "sistema": "convencional", "modelo": MODELO_GERADOR, "temperatura": TEMPERATURA_GERACAO,
        "k": K, "estrategia": "stuffing",
        "prompt_sha256": hashlib.sha256(RAG_PROMPT.encode("utf-8")).hexdigest()[:12],
        "avaliador": MODELO_AVALIADOR, "temperatura_avaliador": TEMPERATURA_AVALIADOR,
        "embedding_ragas": MODELO_EMBEDDING, "ragas": versao,
    }


def info_indice(vectorstore):
    try:
        tam = [len(d.page_content) for d in vectorstore.docstore._dict.values()]
        return {"n_trechos": len(tam), "tamanho_medio": round(sum(tam) / len(tam)), "tamanho_max": max(tam)}
    except Exception:
        return {}


# ============================================================
# RAGAS (mesma configuração do Artigo 1)
# ============================================================
def versao_ragas():
    from importlib import metadata
    try:
        return metadata.version("ragas")
    except metadata.PackageNotFoundError:
        return None


def exigir_ragas_02(ignorar=False):
    versao = versao_ragas()
    if versao is None:
        sys.exit("O pacote ragas não está instalado neste ambiente. Instale com: pip install \"ragas==0.2.15\"")
    try:
        import ragas
    except ImportError as e:
        sys.exit(f"Falha ao importar o ragas: {e}")
    origem = getattr(ragas, "__file__", None)
    if not origem or not hasattr(ragas, "evaluate"):
        sys.exit(
            "\nO 'import ragas' NÃO carregou o pacote instalado pelo pip "
            f"(versão instalada: {versao}; origem do módulo carregado: {origem}).\n"
            "Procure um arquivo ragas.py ou uma pasta 'ragas' na pasta do projeto, ou reinstale com: "
            "pip install --force-reinstall --no-deps \"ragas==0.2.15\""
        )
    if versao.startswith("0.2") or ignorar:
        return
    sys.exit(f"\nEste ambiente tem o ragas {versao}, mas o Artigo 1 usou a versão 0.2.x. "
             "Instale a versão certa: pip install \"ragas==0.2.15\" (ou use --ignorar-versao).")


def montar_avaliador():
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    chat = ChatOpenAI(model=MODELO_AVALIADOR, temperature=TEMPERATURA_AVALIADOR)
    try:  # chamada mínima: confere chave, crédito e disponibilidade do modelo
        chat.invoke("Responda apenas: ok")
    except Exception as e:
        sys.exit(f"\nFalha ao chamar a OpenAI (chave inválida, sem crédito ou modelo indisponível):\n{e}")
    return (LangchainLLMWrapper(chat),
            LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)))


def avaliar_amostra(a, llm, emb):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerRelevancy, ContextRelevance, Faithfulness
    from ragas.run_config import RunConfig

    ds = EvaluationDataset.from_list([{
        "user_input": a["pergunta"],
        "retrieved_contexts": a["contextos"],
        "response": a["resposta"],
    }])
    res = evaluate(ds, metrics=[Faithfulness(), AnswerRelevancy(), ContextRelevance()],
                   llm=llm, embeddings=emb, raise_exceptions=False, show_progress=False,
                   run_config=RunConfig(max_workers=4, timeout=120))
    s = res.scores[0]
    nan = float("nan")
    return {"faithfulness": s.get("faithfulness", nan),
            "answer_relevancy": s.get("answer_relevancy", nan),
            "context_relevance": s.get("nv_context_relevance", nan)}


# ============================================================
# Geração (pipeline convencional)
# ============================================================
def construir_chain():
    from langchain.chains.question_answering import load_qa_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=MODELO_GERADOR, temperature=TEMPERATURA_GERACAO,
                                 google_api_key=GOOGLE_API_KEY, max_output_tokens=MAX_OUTPUT_TOKENS)
    return load_qa_chain(llm, chain_type="stuff", prompt=ChatPromptTemplate.from_template(RAG_PROMPT))


def responder(vectorstore, chain, pergunta):
    t0 = time.time()
    docs = vectorstore.similarity_search(pergunta, k=K)      # recuperação, ciclo único, top-k = 4
    res = chain.invoke({"input_documents": docs, "question": pergunta})
    return {"resposta": res.get("output_text", ""), "contextos": [d.page_content for d in docs],
            "latencia_segundos": round(time.time() - t0, 2)}


# ============================================================
# Tabela final: convencional (novo) x agêntico x Artigo 1
# ============================================================
def metricas_agentico(caminho):
    if not caminho or not os.path.exists(caminho):
        return {}
    out = {}
    for r in ler_csv(caminho):
        try:
            if str(r.get("completo")) == "True" and int(r.get("repeticao") or 1) == 1:
                out[r["id"].upper()] = tuple(float(r[m]) for m in METRICAS)
        except (KeyError, ValueError):
            continue
    return out


def resumo(linhas, ag, caminho_cmp):
    ids = [i for i, _ in PERGUNTAS if str(linhas.get(i, {}).get("completo")) == "True"]
    if not ids:
        print("Nenhuma pergunta completa para resumir.")
        return
    fmt = lambda x: f"{x:6.2f}" if x is not None else "   -  "
    print("\n" + "=" * 96)
    print(f"{'':5}|" + "|".join(f"{n:^23}" for n in ("Faithfulness", "Answer Relevancy", "Context Relevance")) + "| trunc.")
    print(f"{'ID':5}|" + "|".join(f" {'conv.':>6} {'agênt.':>6} {'A1':>6} " for _ in METRICAS) + "|")
    cols = [[] for _ in range(9)]
    linhas_cmp = []
    for qid in ids:
        r = linhas[qid]
        cv = tuple(float(r[m]) for m in METRICAS)
        agt, a1 = ag.get(qid), ARTIGO1.get(qid)
        cel, reg = [], {"id": qid}
        for i, m in enumerate(METRICAS):
            trio = (cv[i], agt[i] if agt else None, a1[i] if a1 else None)
            for j, v in enumerate(trio):
                if v is not None:
                    cols[i * 3 + j].append(v)
            cel.append(f" {fmt(trio[0])} {fmt(trio[1])} {fmt(trio[2])} ")
            reg.update({f"{m}_convencional": trio[0], f"{m}_agentico": "" if trio[1] is None else trio[1],
                        f"{m}_artigo1": "" if trio[2] is None else trio[2],
                        f"{m}_conv_menos_agentico": "" if trio[1] is None else round(trio[0] - trio[1], 3)})
        reg["n_chunks"], reg["contexto_truncado"] = r["n_chunks"], r["contexto_truncado"]
        linhas_cmp.append(reg)
        print(f"{qid:5}|" + "|".join(cel) + f"| {'sim' if str(r['contexto_truncado']) == 'True' else ''}")
    print("-" * 96)
    med = [f"{sum(c) / len(c):6.2f}" if c else "   -  " for c in cols]
    print(f"{'Média':5}|" + "|".join(f" {med[3*i]} {med[3*i+1]} {med[3*i+2]} " for i in range(3)) + f"| (n = {len(ids)})")
    with open(caminho_cmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_cmp[0].keys()))
        w.writeheader()
        w.writerows(linhas_cmp)
    print(f"\nComparação salva em {caminho_cmp}")
    if not ag:
        print("Obs.: sem a coluna 'agênt.' porque o ragas_agentico.csv não foi encontrado (use --agentico).")


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="RAG convencional (Artigo 1): gera as respostas e avalia com o RAGAS")
    ap.add_argument("--saida", default="resultados_convencional_ragas.csv")
    ap.add_argument("--agentico", default="ragas_agentico.csv", help="CSV de métricas do agêntico, para a tabela final")
    ap.add_argument("--apenas", nargs="*", default=[], help="IDs a rodar (ex.: Q1 Q5)")
    ap.add_argument("--pausa", type=float, default=3.0, help="segundos entre perguntas")
    ap.add_argument("--ignorar-versao", action="store_true")
    args = ap.parse_args()

    perguntas = PERGUNTAS
    if args.apenas:
        sel = {x.upper() for x in args.apenas}
        perguntas = [p for p in PERGUNTAS if p[0] in sel]
    if not perguntas:
        sys.exit("Nenhuma pergunta selecionada.")

    cfg = configuracao()
    caminho_cfg = args.saida + ".config.json"
    if os.path.exists(caminho_cfg) and os.path.exists(args.saida):
        anterior = {k: v for k, v in json.load(open(caminho_cfg, encoding="utf-8")).items() if k != "info_indice"}
        if anterior != cfg:
            sys.exit("A configuração mudou desde a execução anterior deste CSV. Apague o CSV e o .config.json "
                     f"ou use outro --saida.\nAnterior: {anterior}\nAtual:    {cfg}")

    linhas = {}
    if os.path.exists(args.saida):
        linhas = {r["id"]: r for r in ler_csv(args.saida)}

    precisa_gerar = [q for q, _ in perguntas if not (linhas.get(q, {}).get("resposta") and not linhas[q].get("erro"))]
    precisa_avaliar = [q for q, _ in perguntas if str(linhas.get(q, {}).get("completo")) != "True"]
    print(f"Perguntas: {len(perguntas)} | precisam de resposta: {len(precisa_gerar)} | precisam de RAGAS: {len(precisa_avaliar)}")

    def salvar():
        with open(args.saida, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows([{c: linhas[i].get(c, "") for c in CAMPOS} for i, _ in PERGUNTAS if i in linhas])

    llm = emb = vectorstore = chain = None
    info = {}
    if precisa_avaliar:
        exigir_ragas_02(args.ignorar_versao)
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY não encontrada. Coloque OPENAI_API_KEY=... no .env da pasta de execução.")
        llm, emb = montar_avaliador()          # confere chave/crédito da OpenAI ANTES de gerar qualquer coisa
    if precisa_gerar:
        if not GOOGLE_API_KEY:
            sys.exit("GOOGLE_API_KEY não encontrada. Coloque GOOGLE_API_KEY=... no .env da pasta de execução.")
        from rag_core import carregar_vectorstore_default
        vectorstore = carregar_vectorstore_default()
        chain = construir_chain()
        info = info_indice(vectorstore)

    print("Configuração:", json.dumps(cfg, ensure_ascii=False))
    if info:
        print("Índice:", info)
    json.dump({**cfg, "info_indice": info}, open(caminho_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    parar = False
    for k, (qid, pergunta) in enumerate(perguntas):
        if qid not in precisa_gerar and qid not in precisa_avaliar:
            continue
        print(f"=== {qid} ===")
        # --- 1) gerar (só se ainda não existe resposta) ---
        if qid in precisa_gerar:
            if k > 0 and args.pausa > 0:
                time.sleep(args.pausa)
            r, erro = None, ""
            for tentativa in range(1, 4):
                try:
                    r = responder(vectorstore, chain, pergunta)
                    erro = ""
                    break
                except Exception as e:
                    erro = str(e)
                    print(f"[ERRO] {qid} (tentativa {tentativa}/3): {erro[:200]}")
                    if _cota_diaria_esgotada(erro) or _modelo_indisponivel(erro):
                        parar = True
                        break
                    if _cota_por_minuto(erro) and tentativa < 3:
                        time.sleep(65)
                    else:
                        break
            if parar:
                print("\nCota diária do Gemini esgotada ou modelo indisponível. O progresso foi salvo; "
                      "rode de novo depois e ele retoma de onde parou.")
                break
            if r is None:
                linhas[qid] = {"id": qid, "pergunta": pergunta, "resposta": "", "contextos": "[]", "n_chunks": 0,
                               "latencia_segundos": "", "completo": "False", "erro": erro}
                salvar()
                continue
            ctx = sum(len(c) for c in r["contextos"])
            linhas[qid] = {"id": qid, "pergunta": pergunta, "resposta": r["resposta"],
                           "contextos": json.dumps(r["contextos"], ensure_ascii=False),
                           "n_chunks": len(r["contextos"]), "latencia_segundos": r["latencia_segundos"],
                           "contexto_truncado": ctx > LIMITE_CONTEXT_RELEVANCE, "completo": "False", "erro": ""}
            salvar()
        # --- 2) avaliar com o RAGAS ---
        linha = linhas[qid]
        s = avaliar_amostra({"pergunta": pergunta, "contextos": json.loads(linha["contextos"]),
                             "resposta": linha["resposta"]}, llm, emb)
        if all(e_nan(s[m]) for m in METRICAS):
            salvar()
            sys.exit(f"\nAs três métricas falharam em {qid}. Provável falta de crédito ou erro de chave/modelo "
                     "da OpenAI. Parei para não gastar mais; a resposta gerada foi salva e não será refeita.")
        for m in METRICAS:
            linha[m] = "" if e_nan(s[m]) else round(float(s[m]), 4)
        linha["completo"] = str(not any(e_nan(s[m]) for m in METRICAS))
        salvar()
        if linha["completo"] != "True":
            print(f"  aviso: métricas sem valor em {qid} (será refeita na próxima rodada)")

    salvar()
    resumo(linhas, metricas_agentico(args.agentico), args.saida.replace(".csv", "") + "_comparacao.csv")


if __name__ == "__main__":
    main()