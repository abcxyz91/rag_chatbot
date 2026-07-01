import json
from collections import Counter

import pytest

from rag_chatbot.evaluation import (
    DEFAULT_DATASET,
    EvaluationError,
    GoldenQuestion,
    citation_metadata_is_valid,
    load_golden_questions,
    run_evaluation,
)
from rag_chatbot.evaluation_retrieval import CorpusRetriever, SearchMatch
from rag_chatbot.evaluate_cli import main as evaluation_main
from rag_chatbot.settings import Settings


def test_default_golden_set_has_twenty_questions_per_specialized_domain():
    questions = load_golden_questions()
    counts = Counter(item.expected_domain for item in questions)

    assert DEFAULT_DATASET.is_file()
    assert len(questions) == 68
    assert counts == {"accounting": 20, "hr": 20, "legal": 20, "general": 8}


def test_golden_set_rejects_duplicate_ids_and_missing_sources(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "same",
                        "question": "An invoice question",
                        "expected_domain": "accounting",
                        "expected_sources": ["ledger.csv"],
                    }
                ),
                json.dumps(
                    {
                        "id": "same",
                        "question": "A leave question",
                        "expected_domain": "hr",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="has no expected source|Duplicate"):
        load_golden_questions(dataset)


@pytest.mark.parametrize(
    ("metadata", "valid"),
    [
        (
            {
                "filename": "policy.pdf",
                "source_path": "legal/policy.pdf",
                "source_type": "pdf",
                "page": 2,
            },
            True,
        ),
        (
            {
                "filename": "ledger.csv",
                "source_path": "accounting/ledger.csv",
                "source_type": "csv",
                "row": 4,
            },
            True,
        ),
        (
            {
                "filename": "guide.docx",
                "source_path": "hr/guide.docx",
                "source_type": "docx",
                "paragraph": 3,
            },
            True,
        ),
        (
            {
                "filename": "policy.pdf",
                "source_path": "legal/policy.pdf",
                "source_type": "pdf",
            },
            False,
        ),
        ({"filename": "policy.pdf", "source_type": "pdf", "page": 1}, False),
    ],
)
def test_citation_metadata_validation_is_file_type_aware(metadata, valid):
    assert citation_metadata_is_valid(metadata) is valid


def test_evaluation_calculates_routing_retrieval_and_citation_metrics(tmp_path):
    questions = [
        GoldenQuestion("a", "accounting q", "accounting", ("ledger.csv",)),
        GoldenQuestion("h", "hr q", "hr", ("leave.docx",)),
        GoldenQuestion("l", "legal q", "legal", ("law.pdf",)),
        GoldenQuestion("g", "general q", "general"),
    ]
    routes = {
        "accounting q": "accounting",
        "hr q": "general",
        "legal q": "legal",
        "general q": "general",
    }

    def retriever(domain, question, top_k):
        assert top_k == 2
        if domain == "accounting":
            return [
                SearchMatch(
                    "other",
                    {
                        "filename": "other.docx",
                        "source_path": "accounting/other.docx",
                        "source_type": "docx",
                        "paragraph": 1,
                    },
                    0.1,
                ),
                SearchMatch(
                    "expected",
                    {
                        "filename": "ledger.csv",
                        "source_path": "accounting/ledger.csv",
                        "source_type": "csv",
                        "row": 2,
                    },
                    0.2,
                ),
            ]
        if domain == "hr":
            return [
                SearchMatch(
                    "miss",
                    {
                        "filename": "training.docx",
                        "source_path": "hr/training.docx",
                        "source_type": "docx",
                        "heading": "Training",
                    },
                    0.3,
                )
            ]
        return [
            SearchMatch(
                "expected but no page",
                {
                    "filename": "law.pdf",
                    "source_path": "legal/law.pdf",
                    "source_type": "pdf",
                },
                0.1,
            )
        ]

    settings = Settings(
        knowledge_base_path=tmp_path,
        chroma_path=tmp_path / "chroma",
        _env_file=None,
    )
    report = run_evaluation(
        questions,
        router=routes.__getitem__,
        retriever=retriever,
        settings=settings,
        top_k=2,
    )

    assert report["routing"]["accuracy"] == 0.75
    assert report["routing"]["confusion_matrix"]["hr"] == {"general": 1}
    assert report["retrieval"]["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert report["retrieval"]["mean_reciprocal_rank"] == 0.5
    assert report["citations"]["metadata_valid_rate"] == 0.75
    assert report["citations"]["expected_citation_rate"] == pytest.approx(1 / 3)


def test_corpus_retriever_ranks_chunks_and_preserves_citations(tmp_path):
    knowledge = tmp_path / "knowledge"
    for domain in ("accounting", "hr", "legal"):
        (knowledge / domain).mkdir(parents=True)
    (knowledge / "accounting" / "ledger.csv").write_text(
        "description,amount\nalpha deposit,10\nbeta invoice,20\n", encoding="utf-8"
    )
    settings = Settings(
        knowledge_base_path=knowledge,
        chroma_path=tmp_path / "chroma",
        _env_file=None,
    )

    def embed(texts):
        return [[0.0, 1.0] if "beta" in text.lower() else [1.0, 0.0] for text in texts]

    retriever = CorpusRetriever(settings, embedder=embed)
    matches = retriever("accounting", "find beta", 2)

    assert "beta invoice" in matches[0].document
    assert matches[0].metadata["filename"] == "ledger.csv"
    assert matches[0].metadata["row"] == 3
    assert matches[0].distance == 0.0


def test_cli_rejects_disabling_every_component(capsys):
    assert evaluation_main(["--skip-routing", "--skip-retrieval"]) == 1
    assert "cannot skip both" in capsys.readouterr().err
