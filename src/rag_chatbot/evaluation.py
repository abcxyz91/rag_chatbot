"""Golden-set evaluation for routing, retrieval, and citation metadata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chatbot.evaluation_retrieval import CorpusRetriever, Domain, SearchMatch
from rag_chatbot.routing import QueryClassifier, ROUTER_SYSTEM_PROMPT, Route
from rag_chatbot.settings import Settings, get_settings


DEFAULT_DATASET = Path(__file__).with_name("evaluation_data") / "golden_questions.jsonl"


class EvaluationError(RuntimeError):
    """Raised when the golden set or a live evaluator cannot be used."""


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    expected_domain: Route
    expected_sources: tuple[str, ...] = ()


def load_golden_questions(path: Path = DEFAULT_DATASET) -> list[GoldenQuestion]:
    """Load and validate a JSON Lines golden set."""

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(
            f"Could not read evaluation dataset {path}: {exc}"
        ) from exc

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
            item = GoldenQuestion(
                id=str(record["id"]),
                question=str(record["question"]),
                expected_domain=record["expected_domain"],
                expected_sources=tuple(record.get("expected_sources", [])),
            )
            QueryClassifier(query_type=item.expected_domain)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"Invalid dataset record on line {line_number}: {exc}"
            ) from exc
        if item.id in seen:
            raise EvaluationError(f"Duplicate evaluation id: {item.id}")
        if item.expected_domain != "general" and not item.expected_sources:
            raise EvaluationError(
                f"Specialized question {item.id} has no expected source"
            )
        seen.add(item.id)
        questions.append(item)
    if not questions:
        raise EvaluationError(f"Evaluation dataset is empty: {path}")
    return questions


class OllamaRouter:
    """Classify a question with the configured local router model."""

    def __init__(self, settings: Settings):
        from ollama import Client

        self.client = Client(host=settings.ollama_url)
        self.model = settings.router_model

    def __call__(self, question: str) -> Route:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                format=QueryClassifier.model_json_schema(),
                options={"temperature": 0},
            )
            return QueryClassifier.model_validate_json(
                response["message"]["content"]
            ).query_type
        except Exception as exc:
            raise EvaluationError(
                f"Ollama could not classify with {self.model!r}: {exc}"
            ) from exc


def citation_metadata_is_valid(metadata: dict[str, str | int]) -> bool:
    """Check that a retrieved chunk can produce an auditable file-type citation."""

    if not metadata.get("filename") or not metadata.get("source_path"):
        return False
    source_type = metadata.get("source_type")
    if source_type == "pdf":
        return "page" in metadata
    if source_type == "csv":
        return "row" in metadata
    if source_type == "docx":
        return "heading" in metadata or "paragraph" in metadata
    if source_type in {"md", "txt"}:
        return "heading" in metadata or "section" in metadata
    return any(
        key in metadata for key in ("page", "row", "heading", "paragraph", "section")
    )


def run_evaluation(
    questions: Sequence[GoldenQuestion],
    *,
    router: Callable[[str], Route] | None = None,
    retriever: Callable[[Domain, str, int], list[SearchMatch]] | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
    evaluate_routing: bool = True,
    evaluate_retrieval: bool = True,
) -> dict[str, Any]:
    """Evaluate the selected components and return a JSON-serializable report."""

    active_settings = settings or get_settings()
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    k = top_k or active_settings.default_top_k
    route = router or (OllamaRouter(active_settings) if evaluate_routing else None)
    retrieve = retriever or (
        CorpusRetriever(active_settings) if evaluate_retrieval else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_size": len(questions),
        "top_k": k,
        "domain_counts": dict(Counter(item.expected_domain for item in questions)),
        "cases": [],
    }

    routing_correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    retrieval_total = retrieval_hits = 0
    reciprocal_rank_total = 0.0
    retrieved_citations = valid_citations = expected_valid_citations = 0

    for item in questions:
        case: dict[str, Any] = {
            "id": item.id,
            "expected_domain": item.expected_domain,
        }
        if evaluate_routing:
            assert route is not None
            predicted = route(item.question)
            QueryClassifier(query_type=predicted)
            correct = predicted == item.expected_domain
            routing_correct += int(correct)
            confusion[item.expected_domain][predicted] += 1
            case.update({"predicted_domain": predicted, "routing_correct": correct})

        if evaluate_retrieval and item.expected_domain != "general":
            assert retrieve is not None
            domain: Domain = item.expected_domain
            matches = retrieve(domain, item.question, k)
            retrieval_total += 1
            ranked_sources = [
                str(match.metadata.get("filename", "")) for match in matches
            ]
            expected = set(item.expected_sources)
            first_rank = next(
                (
                    rank
                    for rank, source in enumerate(ranked_sources, start=1)
                    if source in expected
                ),
                None,
            )
            hit = first_rank is not None
            retrieval_hits += int(hit)
            reciprocal_rank_total += 0.0 if first_rank is None else 1.0 / first_rank

            validity = [citation_metadata_is_valid(match.metadata) for match in matches]
            retrieved_citations += len(validity)
            valid_citations += sum(validity)
            expected_valid = any(
                source in expected and valid
                for source, valid in zip(ranked_sources, validity)
            )
            expected_valid_citations += int(expected_valid)
            case.update(
                {
                    "retrieval_hit": hit,
                    "first_relevant_rank": first_rank,
                    "retrieved_sources": ranked_sources,
                    "citation_metadata_valid": validity,
                    "expected_citation_valid": expected_valid,
                }
            )
        report["cases"].append(case)

    if evaluate_routing:
        total = len(questions)
        report["routing"] = {
            "total": total,
            "correct": routing_correct,
            "accuracy": routing_correct / total,
            "confusion_matrix": {
                expected: dict(predictions)
                for expected, predictions in sorted(confusion.items())
            },
        }
    if evaluate_retrieval:
        report["retrieval"] = {
            "total": retrieval_total,
            "hits": retrieval_hits,
            "hit_rate_at_k": retrieval_hits / retrieval_total
            if retrieval_total
            else 0.0,
            "mean_reciprocal_rank": (
                reciprocal_rank_total / retrieval_total if retrieval_total else 0.0
            ),
        }
        report["citations"] = {
            "retrieved": retrieved_citations,
            "valid_metadata": valid_citations,
            "metadata_valid_rate": (
                valid_citations / retrieved_citations if retrieved_citations else 0.0
            ),
            "questions_with_expected_valid_citation": expected_valid_citations,
            "expected_citation_rate": (
                expected_valid_citations / retrieval_total if retrieval_total else 0.0
            ),
        }
    return report
