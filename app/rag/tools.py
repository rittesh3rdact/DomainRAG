"""The single tool exposed to the model: search over the ingested domain
documents. Giving the model a tool instead of pre-fetching context lets it
decide how many searches it needs and how to phrase each query -- e.g.
issuing two focused searches for a two-part question, or refining a query
that returned nothing useful. This is the core of "agentic RAG" as opposed
to a fixed single-retrieval pipeline.
"""

from google.genai import types

from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import RetrievedChunk

SEARCH_TOOL_NAME = "search_documents"

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=SEARCH_TOOL_NAME,
            description=(
                "Search the domain document collection for passages relevant "
                "to a query. Call this whenever you need information to "
                "answer the user, and call it again with a refined or "
                "different query if the first results are insufficient."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="A focused search query (not the full user question verbatim).",
                    )
                },
                required=["query"],
            ),
        )
    ]
)


class SearchTool:
    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever

    def run(self, query: str) -> tuple[dict, list[RetrievedChunk]]:
        chunks = self._retriever.retrieve(query)
        if not chunks:
            return {"results": [], "note": "No relevant passages found for this query."}, []

        results = [{"source": c.source, "text": c.text} for c in chunks]
        return {"results": results}, chunks
