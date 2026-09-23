# Configuration and input

Everything is driven by `config.json` in the project root. The GUI writes its
settings back into the same file, so both entry points stay in sync.

- [`config.json`](#configjson)
- [Input files](#input-files)

## `config.json`

A complete example with every field. All except `mode` are optional; the
default in the table applies when a field is absent.

```json
{
  "input_dir": "./input",
  "output_dir": "./output",
  "chunking": {
    "mode": "markdown",
    "marker": "---",
    "max_tokens": 600,
    "min_tokens": 30,
    "title_depth": 3,
    "hard_max_tokens": 6000,
    "redact_markdown": true,
    "strip_yaml_header": true,
    "strip_links_for_embedding": false,
    "tokenizer": "heuristic"
  },
  "embedding": {
    "provider": "ollama",
    "model": "bge-m3",
    "base_url": "http://localhost:11434",
    "batch_size": 16,
    "num_ctx": 8192,
    "truncate": true
  }
}
```

### Paths

| Field | Default | Meaning |
|-------|---------|---------|
| `input_dir` | `./input` | Root of the document tree, searched recursively |
| `output_dir` | `./output` | Where run directories are created |

Keep these relative so the config works on any machine.

### `chunking`

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | — | `marker`, `token` or `markdown`; see [chunking.md](chunking.md) |
| `marker` | `"---"` | Separator line for marker mode |
| `max_tokens` | `350` | Target upper bound per chunk |
| `min_tokens` | `0` | A preamble without a heading below this size is dropped (`markdown` only) |
| `title_depth` | `2` | Heading levels carried above the chunk, counted upwards from its own heading; `0` = its own heading only |
| `hard_max_tokens` | `6000` | Emergency brake: anything larger is force-split at line boundaries so no chunk blows the model's context window |
| `redact_markdown` | `false` | Strip `#` and `*` from the finished chunks |
| `strip_yaml_header` | `true` | Remove YAML front matter before chunking |
| `strip_links_for_embedding` | `false` | Hide URLs from the embedding text only |
| `tokenizer` | `"heuristic"` | `heuristic` (~4 chars/token, no dependency) or `tiktoken` |

### `embedding`

| Field | Default | Meaning |
|-------|---------|---------|
| `provider` | `"ollama"` | `ollama` or `gemini`; see [models.md](models.md) |
| `model` | — | Model name as the provider knows it |
| `batch_size` | `16` | Texts per request |
| `base_url` | `http://localhost:11434` | Ollama only: server address |
| `num_ctx` | unset | Ollama only: context window of the request |
| `truncate` | `true` | Ollama only: clip overlong inputs instead of failing with HTTP 400 |
| `api_key` | unset | Gemini only: prefer the `GEMINI_API_KEY` environment variable |
| `api_key_env` | `GEMINI_API_KEY` | Gemini only: name of the environment variable to read |
| `output_dimensionality` | unset | Gemini only: truncate vectors (Matryoshka) |

An `api_key` in the config is never written into `specs.json`, but the config
file itself would still carry it — do not commit such a config.

## Input files

| | |
|---|---|
| **Formats** | `.md` and `.txt` (extension case-insensitive). Other files are ignored silently |
| **Encoding** | UTF-8; a BOM at the start of the file is stripped |
| **Discovery** | `input_dir` is searched **recursively**, to any depth |
| **Excluded** | anything whose path contains an element starting with `.` (`.git`, `.obsidian`, hidden files) |
| **Naming** | each file is identified by its **path relative to the input folder**, e.g. `notes/models/colbert.md` — identically named files in different subfolders do not collide |
| **Order** | alphabetical by relative path, so runs are reproducible |

That relative path shows up again everywhere downstream: as `source` in
`chunks.jsonl` and as the key in `ids_by_file` and `modes_by_file` in
`specs.json`.

In the GUI each file can be given its own chunking mode; all selected files
still end up in **one** run. When modes are mixed, the run directory is named
`..._mixed_<timestamp>` and `modes_by_file` records what was used where.
