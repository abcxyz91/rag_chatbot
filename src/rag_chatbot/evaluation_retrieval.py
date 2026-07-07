"""Read-only embedding retrieval used by the evaluation harness."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag_chatbot.settings import Settings


Domain = Literal["accounting", "hr", "legal"]
DOMAINS: tuple[Domain, ...] = ("accounting", "hr", "legal")
SUPPORTED_EXTENSIONS = {".csv", ".docx", ".pdf"}


@dataclass(frozen=True)
class SearchMatch:
    document: str
    metadata: dict[str, str | int]
    distance: float


@dataclass(frozen=True)
class CorpusChunk:
    document: str
    metadata: dict[str, str | int]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_chunks(path: Path, root: Path) -> list[CorpusChunk]:
    """Extract auditable paragraph, page, or row chunks from one source file."""

    source_type = path.suffix.lower().lstrip(".")
    common: dict[str, str | int] = {
        "filename": path.name,
        "source_path": path.relative_to(root).as_posix(),
        "source_type": source_type,
    }
    if source_type == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                CorpusChunk(
                    " | ".join(
                        f"{name}: {_normalize(value or '')}"
                        for name, value in row.items()
                        if name and _normalize(value or "")
                    ),
                    {**common, "row": row_number},
                )
                for row_number, row in enumerate(csv.DictReader(handle), start=2)
            ]
    if source_type == "docx":
        from docx import Document

        chunks: list[CorpusChunk] = []
        heading = ""
        for paragraph_number, paragraph in enumerate(
            Document(path).paragraphs, start=1
        ):
            text = _normalize(paragraph.text)
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style else "").lower()
            if style.startswith("heading"):
                heading = text
                continue
            locator: dict[str, str | int] = {"paragraph": paragraph_number}
            if heading:
                locator["heading"] = heading
            chunks.append(CorpusChunk(text, {**common, **locator}))
        return chunks
    if source_type == "pdf":
        from pypdf import PdfReader

        return [
            CorpusChunk(text, {**common, "page": page_number})
            for page_number, page in enumerate(PdfReader(path).pages, start=1)
            if (text := _normalize(page.extract_text() or ""))
        ]
    return []


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        from ollama import Client

        self.client = Client(host=settings.ollama_url)
        self.model = settings.embed_model

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self.client.embed(model=self.model, input=list(texts))["embeddings"]
        except Exception as exc:
            raise RuntimeError(
                f"Ollama could not embed with {self.model!r}: {exc}"
            ) from exc


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


class CorpusRetriever:
    """Embed the configured corpus once, then rank chunks for golden questions."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Callable[[Sequence[str]], list[list[float]]] | None = None,
    ):
        self.embedder = embedder or OllamaEmbedder(settings)
        self.chunks: dict[Domain, list[CorpusChunk]] = {}
        self.embeddings: dict[Domain, list[list[float]]] = {}
        for domain in DOMAINS:
            root = settings.domain_knowledge_path(domain)
            if not root.is_dir():
                raise RuntimeError(f"Missing knowledge-base directory: {root}")
            chunks = [
                chunk
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                for chunk in extract_chunks(path, settings.knowledge_base_path)
            ]
            self.chunks[domain] = chunks
            self.embeddings[domain] = (
                self.embedder([chunk.document for chunk in chunks]) if chunks else []
            )

    def __call__(self, domain: Domain, query: str, top_k: int) -> list[SearchMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_embedding = self.embedder([query])[0]
        ranked = sorted(
            zip(self.chunks[domain], self.embeddings[domain]),
            key=lambda item: _cosine_distance(query_embedding, item[1]),
        )
        return [
            SearchMatch(
                chunk.document,
                chunk.metadata,
                _cosine_distance(query_embedding, embedding),
            )
            for chunk, embedding in ranked[:top_k]
        ]
