"""Deterministic document extraction and ChromaDB corpus rebuilding."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

from rag_chatbot.settings import Settings


Domain = Literal["accounting", "hr", "legal"]
DOMAINS: tuple[Domain, ...] = ("accounting", "hr", "legal")
SUPPORTED_EXTENSIONS = {".csv", ".docx", ".md", ".pdf", ".txt"}
MANIFEST_NAME = "manifest.json"
COLLECTION_PREFIX = "rag_chatbot"


class IngestionError(RuntimeError):
    """Raised when a corpus cannot be safely rebuilt."""


@dataclass(frozen=True)
class ExtractedSection:
    text: str
    locator: dict[str, str | int]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def collection_name(domain: Domain) -> str:
    return f"{COLLECTION_PREFIX}_{domain}"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_sections(path: Path) -> list[ExtractedSection]:
    sections: list[ExtractedSection] = []
    heading = ""
    buffer: list[str] = []
    section_number = 0

    def flush() -> None:
        nonlocal buffer, section_number
        text = _normalized(" ".join(buffer))
        if text:
            section_number += 1
            locator: dict[str, str | int] = {"section": section_number}
            if heading:
                locator["heading"] = heading
            sections.append(ExtractedSection(text, locator))
        buffer = []

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if path.suffix.lower() == ".md" and line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        elif line:
            buffer.append(line)
        else:
            flush()
    flush()
    return sections


def _csv_sections(path: Path) -> list[ExtractedSection]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            ExtractedSection(
                " | ".join(
                    f"{name}: {_normalized(value or '')}"
                    for name, value in row.items()
                    if name and _normalized(value or "")
                ),
                {"row": row_number},
            )
            for row_number, row in enumerate(rows, start=2)
        ]


def _docx_sections(path: Path) -> list[ExtractedSection]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestionError("DOCX ingestion requires python-docx") from exc

    document = Document(path)
    sections: list[ExtractedSection] = []
    heading = ""
    for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
        text = _normalized(paragraph.text)
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style else "").lower()
        if style.startswith("heading"):
            heading = text
            continue
        locator: dict[str, str | int] = {"paragraph": paragraph_number}
        if heading:
            locator["heading"] = heading
        sections.append(ExtractedSection(text, locator))
    return sections


def _pdf_sections(path: Path) -> list[ExtractedSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestionError("PDF ingestion requires pypdf") from exc

    return [
        ExtractedSection(text, {"page": page_number})
        for page_number, page in enumerate(PdfReader(path).pages, start=1)
        if (text := _normalized(page.extract_text() or ""))
    ]


def extract_sections(path: Path) -> list[ExtractedSection]:
    extractors = {
        ".csv": _csv_sections,
        ".docx": _docx_sections,
        ".md": _text_sections,
        ".pdf": _pdf_sections,
        ".txt": _text_sections,
    }
    try:
        return extractors[path.suffix.lower()](path)
    except (OSError, ValueError) as exc:
        raise IngestionError(f"Could not extract {path}: {exc}") from exc


def split_text(text: str, *, size: int = 1200, overlap: int = 150) -> list[str]:
    if size < 100:
        raise ValueError("chunk size must be at least 100 characters")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk overlap must be non-negative and smaller than size")

    text = _normalized(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def scan_domain(settings: Settings, domain: Domain) -> list[Path]:
    root = settings.domain_knowledge_path(domain)
    if not root.is_dir():
        raise IngestionError(f"Missing knowledge-base directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def build_chunks(
    settings: Settings,
    domain: Domain,
    path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    relative_path = path.relative_to(settings.knowledge_base_path).as_posix()
    chunks: list[Chunk] = []
    for section in extract_sections(path):
        for chunk_index, text in enumerate(
            split_text(section.text, size=chunk_size, overlap=chunk_overlap), start=1
        ):
            identity = json.dumps(
                [domain, relative_path, section.locator, chunk_index, text],
                sort_keys=True,
                ensure_ascii=False,
            )
            chunks.append(
                Chunk(
                    id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    text=text,
                    metadata={
                        "domain": domain,
                        "filename": path.name,
                        "source_path": relative_path,
                        "source_type": path.suffix.lower().lstrip("."),
                        "chunk_index": chunk_index,
                        **section.locator,
                    },
                )
            )
    return chunks


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        from ollama import Client

        self.client = Client(host=settings.ollama_url)
        self.model = settings.embed_model

    def __call__(self, documents: Sequence[str]) -> list[list[float]]:
        try:
            response = self.client.embed(model=self.model, input=list(documents))
            return response["embeddings"]
        except Exception as exc:
            raise IngestionError(
                f"Ollama could not embed documents with {self.model!r}: {exc}"
            ) from exc


def _batches(items: Sequence[Chunk], size: int) -> Iterable[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _default_client(path: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(path))


def rebuild_corpus(
    settings: Settings,
    *,
    domains: Sequence[Domain] = DOMAINS,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    batch_size: int = 64,
    embedder: Callable[[Sequence[str]], list[list[float]]] | None = None,
    client_factory: Callable[[Path], object] = _default_client,
) -> dict:
    """Rebuild selected domain indexes and return the written manifest."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    settings.validate_startup()
    embed = embedder or OllamaEmbedder(settings)
    manifest_files: list[dict] = []
    domain_summary: dict[str, dict[str, int]] = {}

    for domain in domains:
        files = scan_domain(settings, domain)
        all_chunks: list[Chunk] = []
        for path in files:
            chunks = build_chunks(
                settings,
                domain,
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
            stat = path.stat()
            manifest_files.append(
                {
                    "domain": domain,
                    "path": path.relative_to(settings.knowledge_base_path).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                    "chunk_count": len(chunks),
                }
            )

        embedded_batches = [
            (batch, embed([chunk.text for chunk in batch]))
            for batch in _batches(all_chunks, batch_size)
        ]
        client = client_factory(settings.domain_chroma_path(domain))
        name = collection_name(domain)
        try:
            client.get_collection(name)
        except Exception:
            pass
        else:
            client.delete_collection(name)
        collection = client.create_collection(
            name,
            metadata={"domain": domain, "embedding_model": settings.embed_model},
        )
        for batch, embeddings in embedded_batches:
            documents = [chunk.text for chunk in batch]
            collection.add(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=embeddings,
            )
        domain_summary[domain] = {"files": len(files), "chunks": len(all_chunks)}

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "domains": domain_summary,
        "files": manifest_files,
    }
    manifest_path = settings.chroma_path / MANIFEST_NAME
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return manifest
