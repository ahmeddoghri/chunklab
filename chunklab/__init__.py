"""chunklab: your RAG is only as good as how you cut up the documents.

Everyone obsesses over the embedding model and the reranker and forgets that the
first decision, how you split the document into chunks, quietly sets the ceiling
on everything downstream. Split too coarse and each chunk is half noise. Split
too fine and you sever the sentence that held the answer. Split on a fixed
character count and you cut straight through the middle of the one paragraph
that mattered.

chunklab implements the common chunking strategies (fixed-size, fixed-with-
overlap, recursive on structure, and sentence-aware) and scores them on a
retrieval task where the answer spans are known. It measures whether the chunk
you retrieved actually contains the answer, so you can pick a strategy on
evidence instead of folklore. No embedding model required.
"""
from chunklab.chunkers import (
    Chunk,
    FixedSizeChunker,
    OverlapChunker,
    RecursiveChunker,
    SentenceChunker,
)

__all__ = [
    "Chunk",
    "FixedSizeChunker",
    "OverlapChunker",
    "RecursiveChunker",
    "SentenceChunker",
]

__version__ = "0.1.0"
