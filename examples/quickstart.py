"""Sixty-second tour of chunklab.

    python examples/quickstart.py
"""
from chunklab.chunkers import FixedSizeChunker, SentenceChunker

text = (
    "The cache expires after twenty-four hours. "
    "Deploys go out on Tuesday and Thursday. "
    "Rollbacks take about ninety seconds."
)

for chunker in [FixedSizeChunker(size=60), SentenceChunker(size=60)]:
    chunks = chunker.chunk(text)
    print(f"\n{chunker.name} -> {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c.start:>3}:{c.end:>3}] {c.text!r}")
