# text-embedder

Chunks `.md`/`.txt` documents and turns them into embeddings for a custom RAG
setup — structure-aware, so tables, lists and code blocks survive intact and
every chunk carries its heading path.

Runs as a CLI or a GUI, embeds locally via Ollama or through the Gemini API,
and writes a documented interchange format that downstream indexers consume.

What that looks like — a document goes in, self-contained chunks come out,
each one still knowing where it came from and with its table intact:

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

The heading path is carried into every chunk, so a retrieved fragment is still
interpretable on its own — and the table is never cut in half to hit a token
budget.

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

Tables, bullet lists and code blocks are never cut apart. In `markdown` mode
each chunk is prefixed with its heading path, and split sections are numbered
`(Part 1 of n)`. See [docs/chunking.md](docs/chunking.md) for the details and
the reasoning.

## Output

Each run produces a directory of three files:

```
output/<name>_<mode>_<timestamp>/
  chunks.jsonl        chunk texts + metadata
  embeddings.jsonl    vectors, keyed by the same chunk ID
  specs.json          manifest: config, statistics, all IDs
```

Chunks and vectors are linked by a 16-character random ID, and the two `.jsonl`
files are line-aligned. The format is specified field by field in
**[RAG_INDEX_CONTRACT.md](RAG_INDEX_CONTRACT.md)** — that document is the
contract for anything reading these files.

A ready-made knowledge base built from the example corpus ships in
`reference-kb/`; see [docs/reference-kb.md](docs/reference-kb.md).

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
