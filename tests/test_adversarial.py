"""Tests for retriever fragility and whether the headline finding generalizes."""

from __future__ import annotations

from chunklab.chunkers import FixedSizeChunker, OverlapChunker, RecursiveChunker, SentenceChunker
from chunklab.corpus import DOCUMENT as DOC1
from chunklab.corpus import QA as QA1
from chunklab.corpus_v2 import DOCUMENT as DOC2
from chunklab.corpus_v2 import QA as QA2
from chunklab.eval import _retrieve as retrieve_v1
from chunklab.eval_v2 import build_report
from chunklab.retriever_v2 import retrieve as retrieve_v2

_CHUNKERS = [FixedSizeChunker(200), OverlapChunker(200, 100),
             RecursiveChunker(200), SentenceChunker(200)]


def _coverage(document, qa, retrieve_fn, chunker):
    chunks = chunker.chunk(document)
    return sum(
        1 for q, _, s, e in qa
        if (best := retrieve_fn(chunks, q)) is not None and best.covers(s, e)
    )


# --- the finding: the retriever has no stopword filtering, and it's fragile -

def test_original_retriever_has_wide_ties_on_the_bundled_document():
    """"how long does a rollback take" ties many chunks at a very low score,
    which means the winner is effectively arbitrary."""
    chunks = RecursiveChunker(200).chunk(DOC1)
    from chunklab.eval import _score

    scores = [_score("how long does a rollback take", c) for c in chunks]
    best = max(scores)
    assert best <= 1
    assert scores.count(best) >= 4


def test_stopword_filtering_is_inert_on_the_bundled_document():
    """The bug is real but happens to change nothing on the one document the
    published numbers are measured on."""
    for chunker in _CHUNKERS:
        v1 = _coverage(DOC1, QA1, retrieve_v1, chunker)
        v2 = _coverage(DOC1, QA1, retrieve_v2, chunker)
        assert v1 == v2


# --- the finding: the headline margin does not generalize ------------------

def test_headline_margin_shrinks_on_a_second_document():
    """30-point structure-aware win on the bundled doc; nowhere near that on
    an independently-written second one."""
    fixed_1 = _coverage(DOC1, QA1, retrieve_v1, FixedSizeChunker(200))
    best_structured_1 = max(
        _coverage(DOC1, QA1, retrieve_v1, RecursiveChunker(200)),
        _coverage(DOC1, QA1, retrieve_v1, SentenceChunker(200)),
    )
    margin_1 = best_structured_1 - fixed_1

    fixed_2 = _coverage(DOC2, QA2, retrieve_v1, FixedSizeChunker(200))
    best_structured_2 = max(
        _coverage(DOC2, QA2, retrieve_v1, RecursiveChunker(200)),
        _coverage(DOC2, QA2, retrieve_v1, SentenceChunker(200)),
    )
    margin_2 = best_structured_2 - fixed_2

    assert margin_1 >= 3  # the published 30-point-ish margin, in count terms
    assert margin_2 < margin_1


def test_retriever_choice_flips_the_winning_strategy_on_the_second_document():
    """Stopword filtering, harmless on the bundled document, changes which
    strategy wins on the second one -- the same bug is fragile, not inert."""
    cov_v1 = {
        name: _coverage(DOC2, QA2, retrieve_v1, chunker)
        for name, chunker in zip(("fixed", "overlap", "recursive", "sentence"), _CHUNKERS)
    }
    cov_v2 = {
        name: _coverage(DOC2, QA2, retrieve_v2, chunker)
        for name, chunker in zip(("fixed", "overlap", "recursive", "sentence"), _CHUNKERS)
    }
    winner_v1 = max(cov_v1, key=cov_v1.get)
    winner_v2 = max(cov_v2, key=cov_v2.get)
    assert winner_v1 != winner_v2


# --- sanity: the second corpus is well-formed --------------------------------

def test_second_corpus_spans_are_exact():
    for _, answer, start, end in QA2:
        assert DOC2[start:end] == answer


def test_second_corpus_questions_are_not_lifted_from_the_document():
    """None of the second corpus's questions should be near-verbatim
    substrings of the document, the same circularity check applied to every
    other project's benchmark in this series."""
    doc_lower = DOC2.lower()
    for q, _, _, _ in QA2:
        assert q.lower() not in doc_lower


# --- the original benchmark is unaffected -----------------------------------

def test_original_chunkers_and_eval_untouched():
    import chunklab.eval as eval_module

    assert not hasattr(eval_module, "retrieve_v2")


def test_original_benchmark_still_reproduces():
    results = {
        name: _coverage(DOC1, QA1, retrieve_v1, chunker)
        for name, chunker in zip(("fixed", "overlap", "recursive", "sentence"), _CHUNKERS)
    }
    assert results == {"fixed": 5, "overlap": 6, "recursive": 8, "sentence": 8}


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()
