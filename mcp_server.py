from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

app = FastMCP("buscador_externo")

@app.tool()
def buscar_duckduckgo(query: str, max_results: int = 3):
    """
    Busca informações no DuckDuckGo.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({"title": r["title"], "link": r["href"], "snippet": r["body"]})
    return results

if __name__ == "__main__":
    app.run()
