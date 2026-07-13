import json
from pathlib import Path

from rag_chatbot.ingestion import (
    build_chunks,
    collection_name,
    rebuild_corpus,
    scan_domain,
    split_text,
)
from rag_chatbot.settings import Settings


def make_settings(tmp_path: Path) -> Settings:
    knowledge = tmp_path / "knowledge"
    for domain in ("accounting", "hr", "legal"):
        (knowledge / domain).mkdir(parents=True)
    return Settings(
        knowledge_base_path=knowledge,
        chroma_path=tmp_path / "chroma",
        embed_model="test-embedding",
        _env_file=None,
    )


class FakeCollection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.records = []

    def add(self, **record):
        self.records.append(record)


class FakeClient:
    def __init__(self):
        self.collections = {"rag_chatbot_accounting": FakeCollection("old", {})}
        self.deleted = []

    def get_collection(self, name):
        if name not in self.collections:
            raise ValueError(name)
        return self.collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.collections[name]

    def create_collection(self, name, metadata):
        collection = FakeCollection(name, metadata)
        self.collections[name] = collection
        return collection


def test_split_text_is_stable_and_overlapping():
    text = " ".join(f"word-{number}" for number in range(80))

    chunks = split_text(text, size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks == split_text(text, size=120, overlap=20)
    assert chunks[0][-10:] in chunks[1]


def test_csv_chunks_have_stable_ids_and_row_metadata(tmp_path):
    settings = make_settings(tmp_path)
    source = settings.domain_knowledge_path("accounting") / "ledger.csv"
    source.write_text("date,amount\n2026-01-01,125\n", encoding="utf-8")

    first = build_chunks(
        settings, "accounting", source, chunk_size=200, chunk_overlap=20
    )
    second = build_chunks(
        settings, "accounting", source, chunk_size=200, chunk_overlap=20
    )

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].metadata["row"] == 2
    assert first[0].metadata["source_path"] == "accounting/ledger.csv"


def test_scan_domain_is_recursive_sorted_and_filters_unknown_files(tmp_path):
    settings = make_settings(tmp_path)
    root = settings.domain_knowledge_path("hr")
    (root / "nested").mkdir()
    (root / "z.txt").write_text("Z", encoding="utf-8")
    (root / "nested" / "a.md").write_text("A", encoding="utf-8")
    (root / "ignored.bin").write_bytes(b"no")

    assert [path.name for path in scan_domain(settings, "hr")] == ["a.md", "z.txt"]


def test_rebuild_replaces_collection_and_writes_manifest(tmp_path):
    settings = make_settings(tmp_path)
    source = settings.domain_knowledge_path("accounting") / "policy.txt"
    source.write_text("Accrual accounting records revenue when earned.", encoding="utf-8")
    clients = {}

    def client_factory(path):
        return clients.setdefault(path, FakeClient())

    manifest = rebuild_corpus(
        settings,
        domains=["accounting"],
        chunk_size=200,
        chunk_overlap=20,
        embedder=lambda documents: [[float(len(text))] for text in documents],
        client_factory=client_factory,
    )

    client = clients[settings.domain_chroma_path("accounting")]
    collection = client.collections[collection_name("accounting")]
    assert client.deleted == [collection_name("accounting")]
    assert collection.records[0]["documents"] == [
        "Accrual accounting records revenue when earned."
    ]
    assert manifest["domains"]["accounting"] == {"files": 1, "chunks": 1}
    written = json.loads((settings.chroma_path / "manifest.json").read_text())
    assert written["files"][0]["sha256"]
    assert written["files"][0]["chunk_count"] == 1


def test_embedding_failure_preserves_existing_collection(tmp_path):
    settings = make_settings(tmp_path)
    source = settings.domain_knowledge_path("accounting") / "policy.txt"
    source.write_text("content", encoding="utf-8")
    client = FakeClient()

    def fail(_documents):
        raise RuntimeError("offline")

    try:
        rebuild_corpus(
            settings,
            domains=["accounting"],
            chunk_size=200,
            chunk_overlap=20,
            embedder=fail,
            client_factory=lambda _path: client,
        )
    except RuntimeError:
        pass

    assert client.deleted == []
    assert collection_name("accounting") in client.collections
