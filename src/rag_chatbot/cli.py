"""Command-line entry point for corpus management."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rag_chatbot.ingestion import DOMAINS, IngestionError, rebuild_corpus
from rag_chatbot.settings import get_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-chatbot")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="rebuild local ChromaDB indexes")
    ingest.add_argument(
        "--domain",
        action="append",
        choices=DOMAINS,
        help="index only this domain (repeatable; defaults to all domains)",
    )
    ingest.add_argument("--chunk-size", type=int, default=1200)
    ingest.add_argument("--chunk-overlap", type=int, default=150)
    ingest.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        try:
            manifest = rebuild_corpus(
                get_settings(),
                domains=args.domain or DOMAINS,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
            )
        except (IngestionError, RuntimeError, ValueError) as exc:
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            return 1
        for domain, summary in manifest["domains"].items():
            print(f"{domain}: {summary['files']} files, {summary['chunks']} chunks")
        print(f"Manifest: {get_settings().chroma_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
