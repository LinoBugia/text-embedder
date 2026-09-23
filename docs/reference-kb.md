# Example corpus and reference knowledge base

## The corpus

`input/` contains 149 documents (~3.2 MB): research notes on ML models with
YAML front matter, longer prose texts, and the same content once as markdown
and once as plain prose. The duplicated content is deliberate — it makes the
three chunking modes directly comparable on identical material.

## RefKB

A fixed **reference knowledge base** is built from that corpus and shipped as
a ready-to-use RAG data set under `reference-kb/`, named `RefKB_<model>`:

| Variant | Model | Dim | Size | In the repo? |
|---------|-------|-----|------|--------------|
| `RefKB_bge-m3` | `bge-m3` | 1024 | 60 MB | **yes** |
| `RefKB_qwen3-embedding-4b-fp16` | `qwen3-embedding:4b-fp16` | 2560 | 149 MB | no |

Both rest on **the same chunk set**: 4178 chunks from 149 files, mode
`markdown` with `max_tokens: 600`, `title_depth: 3`, redaction and the YAML
filter enabled.

The important property: the **chunk IDs are identical and in the same order in
both variants**, and so are the chunk texts. Only the encoder differs. That
makes the two models comparable chunk by chunk via the ID — same question,
same chunks, different encoder — which is what a retrieval benchmark needs.

Only the bge-m3 variant is small enough for a Git repository; GitHub rejects
any file above 100 MB, and the 2560-dimensional vectors come to 149 MB. The
dimension, not the corpus, is what decides this.

## Reproducing a variant

Never chunk again for a new variant — that would generate fresh random IDs and
break the coupling. Re-embed the existing run instead:

```bash
# set embedding.model in config.json to the desired model, then:
python main.py --reembed reference-kb/RefKB_bge-m3 \
               --out-dir reference-kb --out-name RefKB_qwen3-embedding-4b-fp16
```

`--reembed` copies `chunks.jsonl` verbatim, embeds it with the currently
configured model and records the origin as `reembedded_from` in `specs.json`.

Timing for orientation, measured on an M4 with 24 GB: `bge-m3` needs a few
minutes for the full corpus, `qwen3-embedding:4b-fp16` about an hour at
~1.1 chunks/s while occupying ~10 GB of RAM.

## Using it downstream

The default in `config.json` stays `bge-m3` so a fresh clone does not have to
pull 8 GB and the shipped RefKB is queryable right away.

```bash
cp -r reference-kb/RefKB_bge-m3 ../gp-qubo-rag-indexer/
```

> **Mind the query side:** query embeddings must be produced with exactly the
> model that indexed the variant you are querying — vectors of different models
> live in different spaces. For Qwen3-Embedding also note that documents are
> embedded raw (as done here) while the query side benefits from the model's
> instruct prefix.

The file format is specified in
[RAG_INDEX_CONTRACT.md](../RAG_INDEX_CONTRACT.md).
