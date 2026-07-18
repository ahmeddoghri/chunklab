"""Benchmark: which chunking strategy actually retrieves the answer?

We build a document, plant several question/answer pairs in it, and record the
exact character span of each answer. Then for each chunking strategy we:

  1. chunk the document,
  2. retrieve the best chunk for each question with a simple lexical scorer,
  3. check whether that chunk fully contains the answer span.

The metric is answer coverage: of the questions, how many retrieved a chunk that
actually holds the whole answer. A chunk that retrieves near the answer but cuts
it in half scores zero, because in a real RAG pipeline that is a wrong answer.

The retriever is deliberately fixed and simple, so the only thing that varies is
the chunking. Any difference in coverage is attributable to the split, which is
the whole point.

Run it:

    python -m chunklab.eval

Deterministic. No embeddings, no model, no network.
"""
from __future__ import annotations

import re

from chunklab.chunkers import (
    Chunk,
    FixedSizeChunker,
    OverlapChunker,
    RecursiveChunker,
    SentenceChunker,
)
from chunklab.corpus import DOCUMENT, QA

_WORD = re.compile(r"[a-z0-9]+")


def _score(query: str, chunk: Chunk) -> int:
    q = set(_WORD.findall(query.lower()))
    c = set(_WORD.findall(chunk.text.lower()))
    return len(q & c)


def _retrieve(chunks: list[Chunk], query: str) -> Chunk | None:
    best, best_score = None, -1
    for ch in chunks:
        s = _score(query, ch)
        if s > best_score:
            best, best_score = ch, s
    return best


def _coverage(chunker) -> tuple[int, int, float]:
    chunks = chunker.chunk(DOCUMENT)
    covered = 0
    for question, _, span_start, span_end in QA:
        best = _retrieve(chunks, question)
        if best is not None and best.covers(span_start, span_end):
            covered += 1
    avg_len = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
    return covered, len(chunks), avg_len


def run(size: int = 200) -> None:
    chunkers = [
        FixedSizeChunker(size=size),
        OverlapChunker(size=size, overlap=size // 2),
        RecursiveChunker(size=size),
        SentenceChunker(size=size),
    ]

    n = len(QA)
    print(f"chunking benchmark: {len(DOCUMENT)}-char document, {n} planted "
          f"question/answer pairs, target chunk size {size}\n")
    print(f"  {'strategy':>12}  {'answer coverage':>16}  {'chunks':>7}  {'avg len':>8}")

    results = {}
    for ch in chunkers:
        covered, count, avg_len = _coverage(ch)
        results[ch.name] = covered / n
        print(f"  {ch.name:>12}  {covered}/{n} = {covered / n:>6.0%}      "
              f"{count:>7}  {avg_len:>7.0f}c")

    best = max(results, key=results.get)
    worst = min(results, key=results.get)
    print(f"\n{best} retrieves the whole answer {results[best]:.0%} of the time; "
          f"{worst} only manages {results[worst]:.0%}.")
    print("the fixed cutter slices through answers that straddle a boundary.")
    print("respecting sentence and paragraph seams keeps the answer in one piece,")
    print("which is the difference between a right answer and a confident wrong one.")


if __name__ == "__main__":
    run()
