"""A retriever with stopword filtering, and the honest limits of fixing it.

``eval._score`` has no stopword list at all: "a", "how", "does", "is" all
count as matches with the same weight as a genuine content word. On the
bundled document this turns out to be harmless (see
``tests/test_adversarial.py``, stopword filtering changes zero benchmark
outcomes there), but it produces frequent, wide ties: "how long does a
rollback take" ties 6 to 10 chunks at a best score of just 1, decided only
by which chunk happens to come first. That fragility is invisible on the
lucky bundled document and very visible on a second one (see
:mod:`chunklab.corpus_v2` and ``eval_v2``), where stopword filtering
changes which strategy wins.

This is a narrow fix, not a rewrite: same bag-of-words overlap, same
first-chunk tie-break, minus the stopwords. It will not rescue a query like
"how long does a rollback take" against a document that only ever says
"Rolling back" (two words, not one) -- that's a genuine vocabulary gap no
amount of stopword filtering closes, and it's worth knowing that on
purpose rather than being surprised by it in production.
"""
from __future__ import annotations

import re

from .chunkers import Chunk

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "my", "me", "i", "you", "your",
    "it", "its", "what", "which", "who", "how", "do", "does", "did",
    "please", "can", "could", "would", "should", "will", "get", "give",
    "show", "tell", "some", "any", "here", "there", "this", "that",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def score(query: str, chunk: Chunk) -> int:
    return len(_tokens(query) & _tokens(chunk.text))


def retrieve(chunks: list[Chunk], query: str) -> Chunk | None:
    best, best_score = None, -1
    for ch in chunks:
        s = score(query, ch)
        if s > best_score:
            best, best_score = ch, s
    return best
