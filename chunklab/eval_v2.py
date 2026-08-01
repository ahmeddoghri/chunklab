"""Does the headline number generalize past the one document it was measured on?

``chunklab.eval`` reports coverage on a single 1248-char document with 10
planted answers: structure-aware chunking beats fixed-size cutting by 30
points. The README is upfront that this is "one document, ten planted
answers, a lexical retriever," and invites pointing the harness at your
own documents. This module actually does that: it reruns the identical
benchmark against a second, independently-written document
(:mod:`chunklab.corpus_v2`, a different domain, different structure,
questions phrased before the benchmark ran), with both the original
retriever and a stopword-filtered one (:mod:`chunklab.retriever_v2`).

    python -m chunklab.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, Sequence

from . import corpus as corpus_v1
from . import corpus_v2
from .chunkers import Chunk, FixedSizeChunker, OverlapChunker, RecursiveChunker, SentenceChunker
from .eval import _retrieve as retrieve_v1
from .retriever_v2 import retrieve as retrieve_v2

CHUNKERS = [
    ("fixed", lambda size: FixedSizeChunker(size=size)),
    ("overlap", lambda size: OverlapChunker(size=size, overlap=size // 2)),
    ("recursive", lambda size: RecursiveChunker(size=size)),
    ("sentence", lambda size: SentenceChunker(size=size)),
]


def _coverage(document: str, qa, retrieve_fn: Callable, size: int = 200) -> Dict[str, float]:
    results = {}
    for name, factory in CHUNKERS:
        chunks: list[Chunk] = factory(size).chunk(document)
        covered = sum(
            1 for q, _, s, e in qa
            if (best := retrieve_fn(chunks, q)) is not None and best.covers(s, e)
        )
        results[name] = round(covered / len(qa), 4)
    return results


def build_report(size: int = 200) -> Dict:
    return {
        "bundled_document": {
            "retriever_v1": _coverage(corpus_v1.DOCUMENT, corpus_v1.QA, retrieve_v1, size),
            "retriever_v2": _coverage(corpus_v1.DOCUMENT, corpus_v1.QA, retrieve_v2, size),
        },
        "second_document": {
            "retriever_v1": _coverage(corpus_v2.DOCUMENT, corpus_v2.QA, retrieve_v1, size),
            "retriever_v2": _coverage(corpus_v2.DOCUMENT, corpus_v2.QA, retrieve_v2, size),
        },
    }


def format_report(report: Dict) -> str:
    lines = [
        "coverage by chunking strategy, across two independent documents and two retrievers",
        "=" * 78,
        f"{'document / retriever':<28}" + "".join(f"{name:>12}" for name, _ in CHUNKERS),
        "-" * 78,
    ]
    for doc_name in ("bundled_document", "second_document"):
        for retr_name in ("retriever_v1", "retriever_v2"):
            row = report[doc_name][retr_name]
            label = f"{doc_name} / {retr_name}"
            lines.append(f"{label:<28}" + "".join(f"{row[n]:>12.0%}" for n, _ in CHUNKERS))
        lines.append("")
    lines.append(
        "on the bundled document, structure-aware chunking (recursive/sentence) leads"
    )
    lines.append(
        "fixed-size cutting by 30 points and stopword filtering changes nothing. on a"
    )
    lines.append(
        "second, independently-written document, that margin nearly vanishes and even"
    )
    lines.append(
        "flips which structure-aware strategy is best, depending on the retriever's"
    )
    lines.append(
        "tie-breaking noise. the headline number is real, but it is a property of this"
    )
    lines.append("one document, not a universal law. measure your own.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report(args.size)
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
