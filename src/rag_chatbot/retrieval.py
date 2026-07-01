"""CrewAI tool for querying indexes built by :mod:`rag_chatbot.ingestion`."""

from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_chatbot.ingestion import OllamaEmbedder, collection_name
from rag_chatbot.settings import get_settings


class DomainSearchInput(BaseModel):
    query: str = Field(description="Question or keywords to search for")


class DomainSearchTool(BaseTool):
    name: str = "Domain knowledge search"
    description: str = "Search the pre-indexed local knowledge base with citations."
    args_schema: type[BaseModel] = DomainSearchInput
    domain: Literal["accounting", "hr", "legal"]

    def _run(self, query: str) -> str:
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(
            path=str(settings.domain_chroma_path(self.domain))
        )
        try:
            collection = client.get_collection(collection_name(self.domain))
        except Exception as exc:
            raise RuntimeError(
                "The domain index is missing. Run `rag-chatbot ingest` first."
            ) from exc
        embedding = OllamaEmbedder(settings)([query])
        result = collection.query(
            query_embeddings=embedding,
            n_results=settings.default_top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not documents:
            return "No matching knowledge-base content was found."

        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            locator = next(
                (
                    f"{key} {metadata[key]}"
                    for key in ("page", "row", "heading", "paragraph", "section")
                    if key in metadata
                ),
                f"chunk {metadata.get('chunk_index', '?')}",
            )
            matches.append(
                f"Source: {metadata['filename']} ({locator}; distance={distance:.4f})\n"
                f"{document}"
            )
        return "\n\n".join(matches)
