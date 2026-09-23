# Chunking

How documents are cut into chunks, and why the rules are what they are.

- [The three modes](#the-three-modes)
- [`title_depth` — context above the chunk](#title_depth--context-above-the-chunk)
- [`max_tokens` and `hard_max_tokens`](#max_tokens-and-hard_max_tokens)
- [Text transforms](#text-transforms)

## The three modes

**`marker`** — split at a user-defined separator line (e.g. `---`). Full manual
control over chunk boundaries in the document. The mode knows nothing about
headings or token limits; you set the boundaries yourself, so its chunks carry
only `mode` and `chunk_index` as metadata.

**`token`** — roughly token-based, for prose. Cuts happen only at sentence
boundaries; tables, bullet lists and code blocks always stay together. If one
of them exceeds `max_tokens` you get a warning on stderr and an oversized
chunk rather than a broken one — a table cut in half is worthless to a
retriever, an oversized one is merely inefficient.

**`markdown`** — structure-aware, and the default. Every heading (`#` …
`######`) starts its own section. Inside a section the body is split
token-based, with the same protections as `token` mode. Each chunk is prefixed
with its heading path; when a section has to be split, the lowest heading is
repeated above every part and numbered `(Part 1 of n)`, `(Part 2 of n)`, …

The heading path is tracked with a stack, so it stays correct across level
jumps (`#` straight to `###`). Headings inside fenced code blocks are ignored —
a `# comment` in Python does not start a section.

## `title_depth` — context above the chunk

`title_depth` counts how many levels **in addition to the chunk's own heading**
are carried over, from the bottom up. For a chunk under
`# Handbook` → `## Installation` → `### Requirements`:

| Value | Header above the chunk |
|-------|------------------------|
| `0` | `### Requirements` |
| `1` | `## Installation` + `### Requirements` |
| `2` | `# Handbook` + `## Installation` + `### Requirements` |
| `3` and above | the entire path (asking for more levels than exist is harmless) |

More context makes a chunk understandable on its own and helps retrieval with
ambiguous sections — "Known issues" often appears several times in one
document, and without the path those chunks are indistinguishable. It also
costs tokens that are then missing for actual content, and the header counts
towards `max_tokens`.

The part numbering always attaches to the **lowest** heading shown:

```
# Handbook
## Installation
### Requirements (Part 2 of 3)

| Component | Minimum | Recommended |
...
```

## `max_tokens` and `hard_max_tokens`

`max_tokens` is a **target**, not a hard limit. Blocks are packed greedily
until the next one no longer fits; prose is split at sentence boundaries to
make it fit. Tables, lists and code blocks count as indivisible — if a table
is larger than `max_tokens` it gets an oversized chunk of its own rather than
being cut apart.

`hard_max_tokens` (default `6000`) is the emergency brake above that. Whatever
is still too large — giant tables, code blocks, prose without any punctuation,
blocks of link definitions — is force-split at line boundaries, and single
overlong lines are cut by character count. Without it, Ollama rejects overlong
inputs with HTTP 400 and the whole run aborts partway through.

Keep the value below the context window of your embedding model. A run over a
mixed corpus is a good sanity check: if `specs.json` reports a maximum token
count close to your model's window, lower it.

## Text transforms

These run after chunking and are all optional. Each one records itself in the
chunk metadata, so you can always tell what was applied.

### `strip_yaml_header`

Removes YAML front matter (a `---` block at the very start of a file) before
chunking. Files without one are untouched — it is a check, not a blind cut.
It runs before the marker mode so that `---` does not collide with a marker of
the same shape. Chunks get `"yaml_header_stripped": true`.

### `redact_markdown`

Strips `#` heading marks and all `*` from the finished chunks; heading text and
everything else stay verbatim. The rationale is that markdown syntax is token
noise that can bias the embedding slightly without carrying meaning. Chunks get
`"redacted": true`.

### `strip_links_for_embedding`

Removes URLs **only from the text that goes into the embedding model**. The
stored chunk — what your RAG serves later — keeps its links in full. URLs are
token noise (`arxiv`, `abs`, `2004`, `12832`) that skews the vectors without
adding meaning.

| In the chunk | In the embedding text |
|--------------|-----------------------|
| `[ColBERT](https://arxiv.org/abs/2004.12832)` | `ColBERT` (link text stays) |
| `![Diagram](image.png)` | `Diagram` |
| `[BGE][bge-ref]` | `BGE` |
| `[bge-ref]: https://huggingface.co/…` | line dropped entirely |
| `<https://example.com>` and bare URLs | dropped |
| URLs **inside code blocks** | left untouched |

Affected chunks get an extra `embed_text` field and
`"links_stripped_for_embedding": true` in their metadata; chunks without links
stay unchanged and have no `embed_text`. The GUI viewer shows the embedding
text below the chunk so you can see exactly what the model was given.
