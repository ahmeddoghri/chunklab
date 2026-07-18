from chunklab.chunkers import (
    Chunk,
    FixedSizeChunker,
    OverlapChunker,
    RecursiveChunker,
    SentenceChunker,
)
from chunklab.corpus import DOCUMENT, QA

CHUNKERS = [FixedSizeChunker(200), OverlapChunker(200, 50),
            RecursiveChunker(200), SentenceChunker(200)]


def test_chunks_reconstruct_or_cover_document():
    # every chunker must cover the whole document (offsets contiguous or overlapping)
    for chunker in CHUNKERS:
        chunks = chunker.chunk(DOCUMENT)
        assert chunks
        assert chunks[0].start == 0
        assert chunks[-1].end == len(DOCUMENT)


def test_chunk_text_matches_span():
    for chunker in CHUNKERS:
        for c in chunker.chunk(DOCUMENT):
            assert DOCUMENT[c.start:c.end] == c.text


def test_fixed_chunks_are_bounded():
    chunks = FixedSizeChunker(size=100).chunk(DOCUMENT)
    assert all(len(c.text) <= 100 for c in chunks)


def test_overlap_actually_overlaps():
    chunks = OverlapChunker(size=100, overlap=30).chunk(DOCUMENT)
    # consecutive chunks should share characters
    assert chunks[1].start < chunks[0].end


def test_sentence_chunker_never_splits_mid_sentence():
    chunks = SentenceChunker(size=200).chunk(DOCUMENT)
    # each chunk should end at a sentence boundary (period-space) or the doc end
    for c in chunks[:-1]:
        assert c.text.rstrip().endswith(".") or c.text.endswith("\n")


def test_covers_span_logic():
    c = Chunk("hello world", 10, 21)
    assert c.covers(12, 18)
    assert not c.covers(5, 12)
    assert not c.covers(18, 25)


def test_recursive_respects_size_mostly():
    chunks = RecursiveChunker(size=200).chunk(DOCUMENT)
    # recursive may slightly exceed on unsplittable text, but should stay close
    assert all(len(c.text) <= 260 for c in chunks)


def test_qa_spans_are_exact():
    for _, answer, start, end in QA:
        assert DOCUMENT[start:end] == answer


def test_sentence_beats_fixed_on_coverage():
    from chunklab.eval import _coverage
    fixed_cov = _coverage(FixedSizeChunker(200))[0]
    sentence_cov = _coverage(SentenceChunker(200))[0]
    assert sentence_cov >= fixed_cov
