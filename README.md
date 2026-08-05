# 📐 chunklab

**Your RAG is only as good as how you cut up the documents. Most people pick a chunk size by vibes.**

![tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Structure-aware chunking retrieves the whole answer 80% of the time. A naive
> fixed-size cutter manages 50%.** Same document, same retriever, same everything
> except the split: `python -m chunklab.eval`.
>
> **Update:** I did the thing the README below invites you to do, ran the
> same benchmark against a second, independently-written document. The
> 30-point margin nearly disappears, and which structure-aware strategy
> even wins changes depending on retriever tie-breaking noise. The headline
> number is real, but it's a property of this one document, not a law of
> chunking. `python -m chunklab.eval_v2`.

Everyone tunes the embedding model. Everyone argues about rerankers on a
podcast. And almost nobody measures the decision that quietly caps all of it:
how you chop the document into chunks in the first place, usually decided in
thirty seconds by whatever number felt right at the time.

Cut on a fixed character count and you will slice clean through the middle of
the sentence that held the answer, no ceremony, no warning. Now the retriever
finds a chunk that's "near" the answer, your model reads half a fact, and it
confidently makes up the other half like it was always going to say that. The
chunk boundary, not the embedding, is what broke you, and nobody's going to
tell you that in the incident review.

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

## I actually did the "swap in your own documents" test

The paragraph above was a caveat until I measured it. I wrote a second
document, a different domain (a REST API reference instead of an internal
wiki), different sentence lengths and structure, with questions phrased
before running the benchmark, and reran the identical measurement:

```bash
python -m chunklab.eval_v2
```
```
document / retriever               fixed     overlap   recursive    sentence
bundled_document / retriever_v1         50%         60%         80%         80%
bundled_document / retriever_v2         50%         60%         80%         80%

second_document / retriever_v1         50%         70%         60%         70%
second_document / retriever_v2         50%         50%         70%         60%
```

Two things worth knowing before you trust the headline number on your own
data. First, the 30-point structure-aware margin nearly vanishes on the
second document; fixed-size cutting is only 10-20 points behind here, not
30. Second, which structure-aware strategy even wins flips depending on a
bug in the retriever: `eval._score` has no stopword list at all, so "a",
"how", "does" count as matches with the same weight as a real content
word. On the bundled document this is completely harmless, stopword
filtering (`chunklab/retriever_v2.py`) changes zero outcomes there. On the
second document it changes which strategy wins outright (recursive moves
from 6/10 worst to 7/10 best). The same latent bug is inert on one corpus
and decisive on another, which is exactly the kind of thing "measure your
own" is supposed to catch and this repo hadn't, until now. `chunkers.py`/
`eval.py` are untouched, so the bundled-document numbers above still
reproduce exactly; `retriever_v2.py` is opt-in.

## Tests

```bash
pip install pytest && pytest -q      # 18 passing
```

## License

MIT © Ahmed Doghri
