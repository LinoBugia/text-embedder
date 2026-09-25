# text-embedder

Turns a folder of `.md`/`.txt` documents into a **portable knowledge base**:
one directory holding the text chunks, their vectors and a manifest, joined by
stable IDs and described well enough that another program can simply read it.

No database server, no index format to reverse-engineer — three files you can
copy, diff, version and hand to whatever retrieval layer you are building.

```
reference-kb/RefKB_bge-m3/
│
├── chunks.jsonl      one row per chunk — the payload plus its metadata
│   {"id": "dCrKnB4QzBKEtpL9", "source": "01_autos_8000_markdown.md",
│    "text": "Automobile: Aufbau …\nVorstufen und Dampfantrieb\n\n- 1769: …",
│    "metadata": {"mode": "markdown", "chunk_index": 1, "token_count": 71,
│                 "heading_path": ["Automobile: Aufbau, Entstehung …",
│                                  "Frühe Geschichte und Ursprünge",
│                                  "Vorstufen und Dampfantrieb"]}}
│
├── embeddings.jsonl  one row per chunk — same id, stored separately so you
│   {"id": "dCrKnB4QzBKEtpL9",           can read the texts without loading
│    "embedding": [-0.0171, -0.0022, -0.0139, …]}      55 MB of vectors
│
└── specs.json        the manifest — how it was built and what is inside
    {"chunk_count": 4178, "embedding_dim": 1024,
     "embedding": {"provider": "ollama", "model": "bge-m3"},
     "chunking": {"mode": "markdown", "max_tokens": 600, "title_depth": 3},
     "ids_by_file": {"01_autos_8000_markdown.md": ["dCrKnB4QzBKEtpL9", …]}}
```

The `id` is the join key: 16 random characters, unique enough that two
knowledge bases can be merged without collisions. `specs.json` answers
everything a consumer needs to ask — which model produced the vectors, how
long they are, how the text was cut, and which chunks came from which document
in what order.

The format is specified field by field in
[RAG_INDEX_CONTRACT.md](RAG_INDEX_CONTRACT.md), including the invariants a
reader may rely on.

## Quickstart

```bash
# 1. local embedding backend
brew install ollama            # or https://ollama.com/download
ollama serve
ollama pull bge-m3             # the model used by the default config (~1.2 GB)

# 2. put .md/.txt files into ./input, then
python main.py --dry-run       # chunk only, inspect the result, costs nothing
python main.py                 # chunk + embed
```

For the GUI: `pip install -r requirements.txt && python app.py`.

The pipeline itself is standard-library only; `customtkinter` is needed for the
GUI and `tiktoken` is optional for exact token counts.

## Commands

```bash
python main.py                    # chunk + embed
python main.py --dry-run          # chunk only, no embeddings
python main.py --config other.json

# Re-embed an existing run with a different model — chunks and IDs stay
# identical, only the vectors are new (for model comparisons):
python main.py --reembed <run-dir> [--out-dir DIR] [--out-name NAME]
```

## Chunking modes

| Mode | Splits at | Use for |
|------|-----------|---------|
| `marker` | a separator line you choose (`---`) | documents where you set the boundaries yourself |
| `token` | sentence boundaries, up to `max_tokens` | flowing prose without headings |
| `markdown` | headings, then sentences inside each section | structured documents — the default |

Tables, bullet lists and code blocks are never cut apart, and in `markdown`
mode each chunk is prefixed with its heading path:

```markdown
# Handbook                          ┌─ chunk 1 ──────────────────────┐
## Installation                     │ # Handbook                     │
                                    │ ## Installation                │
The setup script checks all         │                                │
prerequisites automatically. It     │ The setup script checks all    │
then unpacks the archive …          │ prerequisites automatically. … │
                                    └────────────────────────────────┘
### Requirements                    ┌─ chunk 2 ──────────────────────┐
                                    │ # Handbook                     │
| Component | Minimum | Recommended │ ## Installation                │
|-----------|---------|------------ │ ### Requirements               │
| RAM       | 4 GB    | 16 GB       │                                │
| Disk      | 10 GB   | 50 GB       │ | Component | Minimum | Rec… | │
                                    │ | RAM       | 4 GB    | 16 … | │
                                    └────────────────────────────────┘
```

A retrieved fragment stays interpretable on its own, and the table is not cut
in half to hit a token budget. Sections too large for one chunk are numbered
`(Part 1 of n)`. See [docs/chunking.md](docs/chunking.md) for the details and
the reasoning.

## Output

Every run writes one directory — `output/<name>_<mode>_<timestamp>/` — with the
three files shown at the top. Runs never overwrite each other, so you can
chunk the same corpus with different settings and compare the results side by
side.

`--reembed` takes an existing run and embeds it again with another model,
keeping the chunk texts and IDs byte-identical. That is how the two variants
in `reference-kb/` are built: same chunks, different encoder, directly
comparable per chunk — see [docs/reference-kb.md](docs/reference-kb.md).

## Documentation

| Document | Contents |
|----------|----------|
| [docs/configuration.md](docs/configuration.md) | Every `config.json` field, and how input files are discovered |
| [docs/chunking.md](docs/chunking.md) | The three modes, `title_depth`, token limits, text transforms |
| [docs/models.md](docs/models.md) | Choosing an embedding model, sizes and RAM, Gemini setup, adding a provider |
| [docs/gui.md](docs/gui.md) | GUI walkthrough and app bundling |
| [docs/reference-kb.md](docs/reference-kb.md) | Example corpus and the shipped RefKB variants |
| [RAG_INDEX_CONTRACT.md](RAG_INDEX_CONTRACT.md) | Normative spec of the output format |

## Project structure

| Path | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `app.py` | GUI (CustomTkinter): workspaces, chunk/embed, chunk browser |
| `pipeline.py` | Shared pipeline: file discovery, chunk IDs, run directories, re-embedding |
| `chunkers.py` | The three chunking modes plus the YAML, markdown and link filters |
| `embedders.py` | Embedding providers (base class, Ollama, Gemini) |
| `config.json` | Central configuration |
| `input/` | Example corpus, 149 documents |
| `reference-kb/` | Reference knowledge base `RefKB_<model>` |

## License

MIT — see [LICENSE](LICENSE).
