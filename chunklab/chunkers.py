"""The chunking strategies being compared.

Each chunker takes a document and returns a list of Chunk objects that record
their character span in the original text. Keeping the spans is what makes the
benchmark honest: we can check whether a retrieved chunk actually covers the
character range where the answer lives, instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start: int      # character offset in the source document
    end: int        # exclusive

    def covers(self, span_start: int, span_end: int) -> bool:
        """True if this chunk fully contains the given character span."""
        return self.start <= span_start and self.end >= span_end


class Chunker:
    name = "base"

    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(Chunker):
    """Cut every ``size`` characters. The naive baseline. Fast, and it will
    happily slice through the middle of the sentence holding your answer."""

    name = "fixed"

    def __init__(self, size: int = 200) -> None:
        self.size = size

    def chunk(self, text: str) -> list[Chunk]:
        return [Chunk(text[i:i + self.size], i, min(i + self.size, len(text)))
                for i in range(0, len(text), self.size)]


class OverlapChunker(Chunker):
    """Fixed size, but each chunk overlaps the previous one by ``overlap``
    characters. The overlap is a cheap hedge: an answer cut at a boundary in one
    chunk is often whole in its neighbour."""

    name = "overlap"

    def __init__(self, size: int = 200, overlap: int = 50) -> None:
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        chunks = []
        step = self.size - self.overlap
        i = 0
        while i < len(text):
            end = min(i + self.size, len(text))
            chunks.append(Chunk(text[i:end], i, end))
            if end == len(text):
                break
            i += step
        return chunks


class RecursiveChunker(Chunker):
    """Split on the biggest structural boundary that keeps chunks under ``size``:
    paragraphs first, then sentences, then a hard cut. This is the LangChain-
    style recursive splitter, and it respects the document's natural seams."""

    name = "recursive"

    def __init__(self, size: int = 200) -> None:
        self.size = size

    def chunk(self, text: str) -> list[Chunk]:
        return self._split(text, 0)

    def _split(self, text: str, offset: int) -> list[Chunk]:
        if len(text) <= self.size:
            return [Chunk(text, offset, offset + len(text))] if text else []
        # try paragraph, then sentence, then hard cut
        for sep in ("\n\n", ". ", " "):
            pieces = _split_keep(text, sep)
            if len(pieces) > 1:
                out: list[Chunk] = []
                pos = offset
                buf = ""
                buf_start = offset
                for piece in pieces:
                    if len(buf) + len(piece) > self.size and buf:
                        out.extend(self._flush(buf, buf_start))
                        buf = ""
                        buf_start = pos
                    buf += piece
                    pos += len(piece)
                if buf:
                    out.extend(self._flush(buf, buf_start))
                return out
        # nothing to split on: hard cut
        return [Chunk(text[i:i + self.size], offset + i, offset + min(i + self.size, len(text)))
                for i in range(0, len(text), self.size)]

    def _flush(self, buf: str, start: int) -> list[Chunk]:
        if len(buf) <= self.size:
            return [Chunk(buf, start, start + len(buf))]
        return self._split(buf, start)


class SentenceChunker(Chunker):
    """Group whole sentences up to ``size``, never splitting a sentence. If the
    answer is a sentence, this is the strategy least likely to sever it."""

    name = "sentence"

    def __init__(self, size: int = 200) -> None:
        self.size = size

    def chunk(self, text: str) -> list[Chunk]:
        sentences = _split_keep(text, ". ")
        out: list[Chunk] = []
        buf = ""
        start = 0
        pos = 0
        for s in sentences:
            if len(buf) + len(s) > self.size and buf:
                out.append(Chunk(buf, start, start + len(buf)))
                start = pos
                buf = ""
            buf += s
            pos += len(s)
        if buf:
            out.append(Chunk(buf, start, start + len(buf)))
        return out


def _split_keep(text: str, sep: str) -> list[str]:
    """Split on ``sep`` but keep the separator attached to each piece, so
    character offsets stay exact."""
    if sep == " ":
        parts = re.split(r"(\s+)", text)
        # rejoin tokens with their following whitespace
        pieces = []
        for i in range(0, len(parts), 2):
            token = parts[i]
            ws = parts[i + 1] if i + 1 < len(parts) else ""
            if token or ws:
                pieces.append(token + ws)
        return pieces
    out = []
    idx = 0
    while True:
        j = text.find(sep, idx)
        if j == -1:
            out.append(text[idx:])
            break
        out.append(text[idx:j + len(sep)])
        idx = j + len(sep)
    return [p for p in out if p]
