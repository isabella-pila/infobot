"""
teste_saudacao_ar.py
====================
Teste de sensibilidade: quanto a saudação inicial ("Olá! Eu sou o Infobot...") pesa na
Answer Relevancy do RAGAS?

Para cada uma das 20 respostas do agêntico, calcula a Answer Relevancy DUAS vezes, na mesma
rodada e com o mesmo avaliador:
    (a) com a resposta original;
    (b) com a resposta SEM a(s) frase(s) de apresentação do início.
A comparação (a) x (b) é pareada, e (a) agora x (a) da rodada anterior mede o ruído do avaliador.
A Q20, que não tem saudação, serve de controle: o MESMO texto é avaliado duas vezes, e a diferença
entre as duas notas é ruído puro do avaliador.

Configuração igual à do Artigo 1: RAGAS 0.2.x | GPT-3.5-turbo, temperatura 0 |
embeddings paraphrase-multilingual-MiniLM-L12-v2.

Uso (na pasta do projeto, no mesmo Python em que o ragas 0.2.15 funciona; o .env precisa ter
OPENAI_API_KEY):

    python teste_saudacao_ar.py --so-limpar     # NÃO usa API: mostra o que será removido de cada resposta
    python teste_saudacao_ar.py --estimar       # NÃO usa API: estima as chamadas
    python teste_saudacao_ar.py                 # roda o teste (≈ 40 avaliações)

Arquivos lidos:   resultados_agentico_completo.csv (obrigatório) e ragas_agentico.csv (opcional,
                  com a Answer Relevancy da rodada anterior).
Arquivos gerados: respostas_sem_saudacao.csv e teste_saudacao_ar.csv

Segurança: confere a versão do ragas e faz uma chamada mínima à OpenAI antes de começar; grava a
cada resposta e retoma de onde parou; para se as métricas falharem (crédito/chave).
"""

import argparse
import csv
import json
import math
import os
import re
import statistics as st
import sys

from dotenv import load_dotenv

csv.field_size_limit(2**31 - 1)

MODELO_AVALIADOR = "gpt-3.5-turbo"
TEMPERATURA_AVALIADOR = 0
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIMIAR = 0.05   # diferenças menores que isso são tratadas como "iguais"

# Frases de apresentação a remover (só no PRIMEIRO parágrafo da resposta)
PADROES_SAUDACAO = [
    r"^\s*(olá|ola|oi)\b",
    r"\b(eu\s+)?sou\s+o\s+\"?infobot\"?",
    r"assistente\s+virtual\s+do\s+cefet",
    r"\bagente\s+sintetizador\b",
]


# ============================================================
# Limpeza da saudação
# ============================================================
def remover_saudacao(texto):
    """Remove, do primeiro parágrafo, as frases que apresentam o assistente. Devolve (novo, removido)."""
    partes = texto.strip().split("\n\n", 1)
    primeiro, resto = partes[0], (partes[1] if len(partes) > 1 else "")
    sentencas = re.split(r"(?<=[.!?])\s+", primeiro.strip())
    mantidas, removidas = [], []
    for s in sentencas:
        (removidas if any(re.search(p, s, re.I) for p in PADROES_SAUDACAO) else mantidas).append(s)
    novo_primeiro = " ".join(mantidas).strip()
    novo = (novo_primeiro + ("\n\n" if novo_primeiro and resto else "") + resto).strip()
    novo = re.sub(r"^\s*(-{3,}\s*)+", "", novo).strip()      # tira separador '---' que sobrar no início
    return novo, " ".join(removidas).strip()


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def carregar_respostas(caminho):
    out = []
    for r in ler_csv(caminho):
        qid = r["id"].strip().upper()
        if not qid.startswith("Q") or int(r.get("repeticao") or 1) != 1 or r.get("erro") or not r["resposta"].strip():
            continue
        novo, removido = remover_saudacao(r["resposta"])
        # proteção: se a limpeza apagar demais ou deixar quase nada, mantém a original
        alterada = bool(removido) and len(removido) <= 500 and len(novo) >= 20
        out.append({"id": qid, "pergunta": r["pergunta"].strip(), "original": r["resposta"],
                    "limpa": novo if alterada else r["resposta"], "removido": removido if alterada else "",
                    "alterada": alterada})
    out.sort(key=lambda a: int(a["id"][1:]))
    return out


# ============================================================
# RAGAS (mesma configuração do Artigo 1)
# ============================================================
def versao_ragas():
    from importlib import metadata
    try:
        return metadata.version("ragas")
    except metadata.PackageNotFoundError:
        return None


def exigir_ragas_02():
    versao = versao_ragas()
    if versao is None:
        sys.exit("O pacote ragas não está instalado neste ambiente: pip install \"ragas==0.2.15\"")
    try:
        import ragas
    except ImportError as e:
        sys.exit(f"Falha ao importar o ragas: {e}")
    if not getattr(ragas, "__file__", None) or not hasattr(ragas, "evaluate"):
        sys.exit("O 'import ragas' não carregou o pacote instalado (arquivo ou pasta 'ragas' na pasta do projeto, "
                 "ou instalação corrompida). Reinstale: pip install --force-reinstall --no-deps \"ragas==0.2.15\"")
    if not versao.startswith("0.2"):
        sys.exit(f"Este ambiente tem o ragas {versao}; o Artigo 1 usou a 0.2.x. Instale: pip install \"ragas==0.2.15\"")


def montar_avaliador():
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    chat = ChatOpenAI(model=MODELO_AVALIADOR, temperature=TEMPERATURA_AVALIADOR)
    try:
        chat.invoke("Responda apenas: ok")
    except Exception as e:
        sys.exit(f"\nFalha ao chamar a OpenAI (chave inválida, sem crédito ou modelo indisponível):\n{e}")
    return (LangchainLLMWrapper(chat),
            LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)))


def avaliar_ar(pergunta, resposta, llm, emb):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerRelevancy
    from ragas.run_config import RunConfig
    ds = EvaluationDataset.from_list([{"user_input": pergunta, "response": resposta}])
    res = evaluate(ds, metrics=[AnswerRelevancy()], llm=llm, embeddings=emb, raise_exceptions=False,
                   show_progress=False, run_config=RunConfig(max_workers=4, timeout=120))
    return res.scores[0].get("answer_relevancy", float("nan"))


def e_nan(x):
    return x is None or x == "" or (isinstance(x, float) and math.isnan(x))


# ============================================================
# Resumo
# ============================================================
def resumo(linhas, anterior):
    comp = [l for l in linhas.values() if not e_nan(l["ar_original_agora"]) and not e_nan(l["ar_sem_saudacao"])]
    comp.sort(key=lambda l: int(l["id"][1:]))
    if not comp:
        print("Nenhuma resposta completa para resumir.")
        return
    f = lambda x: f"{float(x):5.2f}" if not e_nan(x) else "  -  "
    print("\n" + "=" * 74)
    print(f"{'ID':5}| {'antes':>6} {'orig.':>6} {'sem sd.':>7} {'dif.':>6} | removido do início")
    for l in comp:
        dif = float(l["ar_sem_saudacao"]) - float(l["ar_original_agora"])
        rem = (l["removido"][:38] + "...") if len(l["removido"]) > 38 else (l["removido"] or "(nada: controle)")
        print(f"{l['id']:5}| {f(anterior.get(l['id'])):>6} {f(l['ar_original_agora']):>6} {f(l['ar_sem_saudacao']):>7} {dif:+6.2f} | {rem}")
    orig = [float(l["ar_original_agora"]) for l in comp]
    sem = [float(l["ar_sem_saudacao"]) for l in comp]
    difs = [s - o for s, o in zip(sem, orig)]
    com = [l for l in comp if l["alterada"] == "True"]
    print("-" * 74)
    print(f"Respostas avaliadas: {len(comp)} (com saudação removida: {len(com)})")
    print(f"AR médio  original: {st.mean(orig):.2f} | sem saudação: {st.mean(sem):.2f} | diferença: {st.mean(difs):+.2f}")
    if com:
        d2 = [float(l["ar_sem_saudacao"]) - float(l["ar_original_agora"]) for l in com]
        print(f"Só nas {len(com)} que mudaram: diferença média {st.mean(d2):+.2f} | "
              f"melhoraram {sum(x > LIMIAR for x in d2)} | pioraram {sum(x < -LIMIAR for x in d2)} | iguais {sum(abs(x) <= LIMIAR for x in d2)}")
    sem_alt = [l for l in comp if l["alterada"] != "True"]
    for l in sem_alt:
        print(f"Controle ({l['id']}, texto idêntico avaliado duas vezes): {float(l['ar_original_agora']):.2f} e {float(l['ar_sem_saudacao']):.2f} "
              f"(diferença {float(l['ar_sem_saudacao']) - float(l['ar_original_agora']):+.2f} = ruído puro do avaliador)")
    ruido = [abs(float(l["ar_original_agora"]) - anterior[l["id"]]) for l in comp if l["id"] in anterior]
    if ruido:
        print(f"Ruído do avaliador (mesma resposta, rodada anterior x esta): diferença absoluta média {st.mean(ruido):.2f}")
        print("Regra de leitura: se a diferença média 'sem saudação' não passar bem do ruído, a saudação não explica a queda.")


# ============================================================
def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Sensibilidade da Answer Relevancy à saudação inicial")
    ap.add_argument("--entrada", default="resultados_agentico_completo.csv")
    ap.add_argument("--anterior", default="ragas_agentico.csv", help="CSV de métricas da rodada anterior (opcional)")
    ap.add_argument("--saida", default="teste_saudacao_ar.csv")
    ap.add_argument("--so-limpar", action="store_true", help="não usa API: só mostra o que será removido")
    ap.add_argument("--estimar", action="store_true", help="não usa API: estima chamadas")
    args = ap.parse_args()

    respostas = carregar_respostas(args.entrada)
    if not respostas:
        sys.exit("Nenhuma resposta encontrada (confira --entrada).")
    print(f"{len(respostas)} respostas | com saudação removida: {sum(a['alterada'] for a in respostas)}")

    with open("respostas_sem_saudacao.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pergunta", "removido", "resposta_sem_saudacao"])
        w.writeheader()
        for a in respostas:
            w.writerow({"id": a["id"], "pergunta": a["pergunta"], "removido": a["removido"], "resposta_sem_saudacao": a["limpa"]})

    if args.so_limpar:
        for a in respostas:
            print(f"\n{a['id']}  REMOVIDO: {a['removido'] or '(nada)'}")
            print(f"      COMEÇA ASSIM AGORA: {a['limpa'][:110].replace(chr(10), ' ')}")
        print("\nConferência salva em respostas_sem_saudacao.csv. Se algo substantivo foi removido, me avise.")
        return

    if args.estimar:
        n = len(respostas) * 2
        tok = sum((len(a["original"]) + len(a["limpa"])) / 3.5 + 2 * 600 for a in respostas)
        print(f"Avaliações: {n} (cada uma faz de 1 a 3 chamadas) -> cerca de {n} a {3 * n} chamadas.")
        print(f"Tokens de entrada (aprox.): {tok:,.0f}. Estimativa grosseira; confira os preços no painel da OpenAI.")
        return

    exigir_ragas_02()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY não encontrada. Coloque OPENAI_API_KEY=... no .env da pasta de execução.")

    cfg = {"avaliador": MODELO_AVALIADOR, "temperatura": TEMPERATURA_AVALIADOR,
           "embedding": MODELO_EMBEDDING, "ragas": versao_ragas()}
    caminho_cfg = args.saida + ".config.json"
    if os.path.exists(caminho_cfg) and os.path.exists(args.saida):
        if json.load(open(caminho_cfg, encoding="utf-8")) != cfg:
            sys.exit("A configuração do avaliador mudou desde a rodada anterior deste arquivo. Use outro --saida.")
    json.dump(cfg, open(caminho_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Configuração do avaliador:", cfg)

    anterior = {}
    if os.path.exists(args.anterior):
        for r in ler_csv(args.anterior):
            try:
                if int(r.get("repeticao") or 1) == 1 and not e_nan(r.get("answer_relevancy")):
                    anterior[r["id"].upper()] = float(r["answer_relevancy"])
            except ValueError:
                pass

    campos = ["id", "alterada", "removido", "ar_original_agora", "ar_sem_saudacao"]
    linhas = {r["id"]: r for r in ler_csv(args.saida)} if os.path.exists(args.saida) else {}

    def salvar():
        with open(args.saida, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows([{c: linhas[a["id"]].get(c, "") for c in campos} for a in respostas if a["id"] in linhas])

    pend = [a for a in respostas if a["id"] not in linhas or e_nan(linhas[a["id"]].get("ar_original_agora"))
            or e_nan(linhas[a["id"]].get("ar_sem_saudacao"))]
    print(f"Já completas: {len(respostas) - len(pend)} | a avaliar agora: {len(pend)}")
    if pend:
        llm, emb = montar_avaliador()
        for a in pend:
            print(f"Avaliando {a['id']}...")
            o = avaliar_ar(a["pergunta"], a["original"], llm, emb)
            s = avaliar_ar(a["pergunta"], a["limpa"], llm, emb)   # sem saudação: igual à original => repete a avaliação
            if e_nan(o) and e_nan(s):
                salvar()
                sys.exit(f"\nA métrica falhou em {a['id']}. Provável falta de crédito ou erro de chave/modelo da OpenAI. "
                         "Parei para não gastar mais; o progresso foi salvo.")
            linhas[a["id"]] = {"id": a["id"], "alterada": a["alterada"], "removido": a["removido"],
                               "ar_original_agora": "" if e_nan(o) else round(float(o), 4),
                               "ar_sem_saudacao": "" if e_nan(s) else round(float(s), 4)}
            salvar()
    res = {k: {**v, "alterada": str(v["alterada"])} for k, v in linhas.items()}
    resumo(res, anterior)


if __name__ == "__main__":
    main()