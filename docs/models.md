# Embedding providers and models

Provider and model are set in `config.json` (`embedding.provider`,
`embedding.model`) or switched directly in the GUI header.

- [Ollama — local, open source](#ollama--local-open-source)
- [Size and precision](#size-and-precision)
- [Gemini — Google AI Studio API](#gemini--google-ai-studio-api)
- [Adding a provider](#adding-a-provider)

> **Switching models invalidates existing embeddings.** Vectors of different
> models live in different spaces and cannot be compared. After a switch,
> re-embed the corpus and use exactly the same model for queries.

## Ollama — local, open source

Recommended for anything sensitive, because nothing leaves the machine. All of
these are Apache 2.0 / MIT and usable commercially — pull one and put its name
into `model`:

| Model (`ollama pull …`) | Download | Dim | Strengths |
|-------------------------|----------|-----|-----------|
| `bge-m3` | 1.2 GB | 1024 | Multilingual, strong on German text, 8K context |
| `embeddinggemma` | 0.6 GB | 768 | Google DeepMind, 300M — laptop-friendly, 100+ languages, 2K context |
| `nomic-embed-text` | 0.3 GB | 768 | Popular all-rounder, Matryoshka (variable dimension) |
| `mxbai-embed-large` | 0.7 GB | 1024 | Top-tier for English (mixedbread ai) |
| `snowflake-arctic-embed2` | 1.2 GB | 1024 | Multilingual (74 languages), 8K context |
| `qwen3-embedding:4b-fp16` | 8.0 GB | 2560 | Very strong MTEB multilingual scores, 32K+ context |

The authoritative dimension is `embedding_dim` in `specs.json` after the first
run — use that to size your index, not a table.

`multilingual-e5-large` (MIT, robust for mixed DE/EN) is not in the official
Ollama library; use `sentence-transformers` or a community pull. For German
corpora, `bge-m3` (small and fast) or `qwen3-embedding:4b-fp16` (much larger,
stronger) are the natural picks.

## Size and precision

Ollama's default tag is usually **quantised** (`q4_K_M`). Full precision needs
an explicit tag, and the difference is substantial:

| Tag | Download | Note |
|-----|----------|------|
| `qwen3-embedding:0.6b` / `:0.6b-fp16` | 0.6 / 1.2 GB | smallest variant |
| `qwen3-embedding:4b` / `:4b-fp16` | 2.5 / 8.0 GB | **good middle ground** on 24 GB RAM |
| `qwen3-embedding:8b` / `:8b-fp16` | 4.7 / 15 GB | F16 gets tight on 24 GB |

Rule of thumb for Apple Silicon: the model shares unified memory with the
system, so roughly **two thirds of RAM** are realistically usable. On a 24 GB
machine, models up to ~10 GB run comfortably, 15 GB is borderline, and
candidates such as `KaLM-Embedding-Gemma3-12B-2511` (11.8B parameters, 24 GB in
F16) do **not** fit — they are only available quantised, which in turn makes a
clean quality comparison awkward.

Throughput matters as much as size. Measured on an M4 with 24 GB over a real
corpus (chunks of ~190 tokens on average): `qwen3-embedding:4b-fp16` embeds
about 1.1 chunks/s, so 4178 chunks take roughly an hour. `bge-m3` is seven
times smaller and finishes the same corpus in minutes. Lowering `num_ctx` does
not help measurably — the model itself is the bottleneck.

## Gemini — Google AI Studio API

No subscription needed, just an API key from
<https://aistudio.google.com/apikey>:

```bash
export GEMINI_API_KEY="your-key"    # e.g. in ~/.zshrc
```

```json
"embedding": {
  "provider": "gemini",
  "model": "gemini-embedding-001",
  "batch_size": 16,
  "output_dimensionality": 768
}
```

| Model | Free tier | Paid tier | Context / dim |
|-------|-----------|-----------|----------------|
| `gemini-embedding-001` | free, ~1500 RPM | $0.15 / 1M tokens | 2K / up to 3072 dim |
| `text-embedding-004` (predecessor) | free | $0.025 / 1M tokens | 2K / 768 dim |

`output_dimensionality` is optional (Matryoshka: 3072 truncated to e.g. 768,
saving storage). Instead of the environment variable you can put
`"api_key": "..."` into the config — but then **do not commit** that config.
The key is never written into `specs.json`.

> ⚠️ **Privacy:** on the free tier Google trains on your inputs ("Used to
> improve our products: Yes"). For patient data, NDA material or anything
> sensitive, **never use the free tier** — use the paid tier or stay local with
> Ollama.

## Adding a provider

Subclass `Embedder` in `embedders.py` and register it:

```python
class OpenAIEmbedder(Embedder):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...  # API call, returns list[list[float]]

PROVIDERS["openai"] = OpenAIEmbedder
```

Then set `"provider": "openai"` in `config.json`. Batching and output handling
come from the base class. Optionally extend `check_provider()` and
`list_models()` so the GUI can show its status and offer its models.

> If the new provider should also be usable for RAG querying, add the query
> embedding in `rag_gui.py` (`embed_query`) in the gp-qubo-rag-indexer project
> as well.
