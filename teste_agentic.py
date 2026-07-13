"""
teste_agentic.py
=================
Script de teste standalone (sem Streamlit) para validar o RAG Agêntico do Artigo 2.

Como usar:
1. Coloque este arquivo, rag_core.py e agentic_rag_langgraph.py na MESMA pasta do
   seu perguntas2.pdf e do seu .env (GOOGLE_API_KEY, SERPER_API_KEY).
2. No terminal (com o venv do projeto ativado):
      python teste_agentic.py
3. Acompanhe no console: sub-perguntas geradas pelo Planejador, se houve
   fallback web, quantas iterações de verificação, latência, etc.
"""

import time
from rag_core import carregar_vectorstore_default, pesquisar_na_web_estruturado
from agentic_rag_langgraph import construir_grafo, responder_com_rag_agentico

# Free tier do Gemini (gemini-2.5-flash-lite) = 10 requisições/minuto.
# Cada pergunta do RAG Agêntico pode disparar 3 a 6 chamadas de LLM
# (Planejador + Verificador x N + Sintetizador, ou + fallback web).
# Por isso esperamos alguns segundos ENTRE perguntas para não estourar a cota.
PAUSA_ENTRE_PERGUNTAS_SEGUNDOS = 45

PERGUNTAS_TESTE = [
    "Qual é a duração mínima para se formar em SI?",              # simples -> deve ir direto pro Sintetizador
]

if __name__ == "__main__":
    print("Carregando base vetorial (perguntas2.pdf)...")
    vectorstore = carregar_vectorstore_default()

    print("Montando grafo multi-agente...")
    grafo = construir_grafo(vectorstore, pesquisar_na_web_estruturado)

    # (Opcional) gera o diagrama Mermaid do grafo — ótimo para a Figura do Artigo 2
    try:
        print("\n--- Diagrama Mermaid do grafo (cole em https://mermaid.live) ---")
        print(grafo.get_graph().draw_mermaid())
    except Exception as e:
        print(f"[aviso] não foi possível gerar o diagrama: {e}")

    for i, pergunta in enumerate(PERGUNTAS_TESTE):
        if i > 0:
            print(f"\n(aguardando {PAUSA_ENTRE_PERGUNTAS_SEGUNDOS}s para respeitar a cota do free tier...)")
            time.sleep(PAUSA_ENTRE_PERGUNTAS_SEGUNDOS)

        print("\n" + "=" * 70)
        print("PERGUNTA:", pergunta)
        resultado = responder_com_rag_agentico(grafo, pergunta)
        print("-" * 70)
        print("RESPOSTA:", resultado["resposta"])
        print("Iterações de verificação:", resultado["iteracoes_verificacao"])
        print("Usou fallback web?:", resultado["usou_fallback_web"])
        print("Latência (s):", resultado["latencia_segundos"])
        print("Fontes:", resultado["fontes"])