# 📐 chunklab

**Your RAG is only as good as how you cut up the documents. Most people pick a chunk size by vibes.**

![CI](https://github.com/ahmeddoghri/chunklab/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-9%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Structure-aware chunking retrieves the whole answer 80% of the time. A naive
> fixed-size cutter manages 50%.** Same document, same retriever, same everything
> except the split: `python -m chunklab.eval`.

Everyone tunes the embedding model. Everyone argues about rerankers. And almost
nobody measures the decision that quietly caps all of it: how you chop the
document into chunks in the first place.

Cut on a fixed character count and you will slice clean through the middle of
the sentence that held the answer. Now the retriever finds a chunk that is
"near" the answer, your model reads half a fact, and it confidently makes up the
other half. The chunk boundary, not the embedding, is what broke you.

chunklab puts the common chunking strategies side by side on a retrieval task
where the answer spans are known exactly, and measures the only thing that
matters: did the retrieved chunk actually contain the whole answer? No embedding
model, no network, no dependencies.

---

## The result in one command

```bash
python -m chunklab.eval
```
```
chunking benchmark: 1248-char document, 10 planted question/answer pairs, target chunk size 200

      strategy   answer coverage   chunks   avg len
         fixed  5/10 =    50%            7      178c
       overlap  6/10 =    60%           12      196c
     recursive  8/10 =    80%           11      113c
      sentence  8/10 =    80%            9      139c
```

The retriever is held fixed, so the only variable is the split. Fixed-size
cutting loses half the answers to boundaries it drew through the middle of a
sentence. Overlap buys back a little by giving each boundary a second chance.
The structure-aware strategies, which respect paragraph and sentence seams, keep
the answer in one piece and win by a full 30 points.

That 30 points is not an embedding upgrade or a bigger model. It is free, and it
is sitting in a decision most pipelines make without measuring.

## Install

```bash
git clone https://github.com/ahmeddoghri/chunklab
cd chunklab && pip install -e .
python examples/quickstart.py
```

## See the difference

```python
from chunklab.chunkers import FixedSizeChunker, SentenceChunker

text = "The cache expires after twenty-four hours. Deploys go out on Tuesday."

FixedSizeChunker(size=60).chunk(text)
# cuts mid-sentence: "...Deploys go out on" | " Tuesday."   <- answer severed

SentenceChunker(size=60).chunk(text)
# keeps each fact whole: "...twenty-four hours." | "Deploys go out on Tuesday."
```

## The strategies

| Chunker | How it splits | The tradeoff |
|---|---|---|
| `FixedSizeChunker` | every N characters | fast, and it does not care what it cuts through |
| `OverlapChunker` | fixed size with overlap | the overlap rescues some boundary casualties |
| `RecursiveChunker` | paragraphs, then sentences, then a hard cut | respects the document's natural seams |
| `SentenceChunker` | whole sentences up to N chars | never splits a sentence, so it never splits a one-sentence answer |

Every chunk records its exact character span in the source, which is what lets
the benchmark check true answer coverage instead of eyeballing it. Write your
own chunker in a few lines, keep the span honest, and its coverage shows up next
to the rest.

## An honest note on the number

This is a small, transparent benchmark: one document, ten planted answers, a
lexical retriever. It is not a claim that sentence chunking wins by exactly 30
points on your corpus. It is a claim that the chunking strategy has a large,
measurable effect that most pipelines never measure, and a harness you can point
at your own documents to find your own number. Swap in your retriever and your
data; the coverage metric does not change.

## Tests

```bash
pip install pytest && pytest -q      # 9 passing
```

## License

MIT © Ahmed Doghri
