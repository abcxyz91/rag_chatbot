"""Command-line entry point for quality evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_chatbot.evaluation import (
    DEFAULT_DATASET,
    EvaluationError,
    load_golden_questions,
    run_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-chatbot-evaluate")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--skip-routing", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--min-routing-accuracy", type=float, default=0.0)
    parser.add_argument("--min-retrieval-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-citation-rate", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.skip_routing and args.skip_retrieval:
        print("Evaluation failed: cannot skip both components", file=sys.stderr)
        return 1
    if args.top_k is not None and args.top_k < 1:
        print("Evaluation failed: --top-k must be positive", file=sys.stderr)
        return 1
    thresholds = (
        args.min_routing_accuracy,
        args.min_retrieval_hit_rate,
        args.min_citation_rate,
    )
    if any(value < 0 or value > 1 for value in thresholds):
        print("Evaluation failed: thresholds must be between 0 and 1", file=sys.stderr)
        return 1

    try:
        report = run_evaluation(
            load_golden_questions(args.dataset),
            top_k=args.top_k,
            evaluate_routing=not args.skip_routing,
            evaluate_retrieval=not args.skip_retrieval,
        )
    except (EvaluationError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Report: {args.output.resolve()}")
    if "routing" in report:
        print(f"Routing accuracy: {report['routing']['accuracy']:.3f}")
    if "retrieval" in report:
        print(
            f"Retrieval hit rate@{report['top_k']}: "
            f"{report['retrieval']['hit_rate_at_k']:.3f}"
        )
        print(
            "Expected citation rate: "
            f"{report['citations']['expected_citation_rate']:.3f}"
        )

    failed = (
        report.get("routing", {}).get("accuracy", 1.0) < args.min_routing_accuracy
        or report.get("retrieval", {}).get("hit_rate_at_k", 1.0)
        < args.min_retrieval_hit_rate
        or report.get("citations", {}).get("expected_citation_rate", 1.0)
        < args.min_citation_rate
    )
    if failed:
        print("Evaluation failed: a metric threshold was missed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
