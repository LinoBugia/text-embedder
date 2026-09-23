---
title: jina-embeddings-v3
category: embedding
models_covered: [jina-embeddings-v3]
last_updated: 2026-07-11
sources: [https://huggingface.co/jinaai/jina-embeddings-v3, https://arxiv.org/abs/2409.10173, https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/]
---

# jina-embeddings-v3

## Übersicht
jina-embeddings-v3 (Jina AI, Sept. 2024, Paper arXiv:2409.10173 "Multilingual Embeddings With Task LoRA") ist ein mehrsprachiges Frontier-Embedding-Modell mit ~570 M Parametern. Sein Alleinstellungsmerkmal ist die Kombination aus **task-spezifischen LoRA-Adaptern** und **Matryoshka-Embeddings** auf einem 8192-Token-Long-Context-Backbone. Zum Release übertraf es laut Paper OpenAI text-embedding-3-large auf englischen MTEB-Tasks und multilingual-e5-large-instruct über alle evaluierten mehrsprachigen Tasks.

## Architektur
Der Backbone ist **Jina-XLM-RoBERTa** (eine XLM-RoBERTa-Variante mit Flash-Attention-Implementierung) und nutzt **Rotary Position Embeddings (RoPE)** für lange Eingaben bis 8192 Token. Encoder-only, Mean-Pooling.

Die zentrale Innovation sind **fünf task-spezifische LoRA-Adapter** (Low-Rank Adaptation), die zur Inferenzzeit gewählt werden und denselben eingefrorenen Backbone spezialisieren, ohne die Parameterzahl wesentlich zu erhöhen:
- `retrieval.query` — Query-Embeddings (asymmetrisches Retrieval)
- `retrieval.passage` — Passage/Dokument-Embeddings (asymmetrisches Retrieval)
- `separation` — Clustering und Re-Ranking
- `classification` — Klassifikationsaufgaben
- `text-matching` — Ähnlichkeit/symmetrisches Retrieval

Zusätzlich ist das Modell mit **Matryoshka Representation Learning** trainiert: die Ausgabedimension kann von 1024 auf 768/512/256/128/64/32 gekürzt werden, ohne die Qualität stark zu beeinträchtigen (Speicher-/Latenz-Tradeoff).

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Org | Jina AI |
| Release | Sept. 2024 |
| Basis-Modell | Jina-XLM-RoBERTa (+ Flash Attention, RoPE) |
| Parameter | ~570 M (0.6B) |
| Embedding-Dimension | 1024 (Matryoshka: 1024→32) |
| Kontextlänge | 8192 Token |
| Sprachen | 100 (tiefes Tuning für 30 Zielsprachen inkl. Deutsch) |
| Task-Adapter | 5 LoRA (retrieval.query/passage, separation, classification, text-matching) |
| Lizenz | CC BY-NC 4.0 (Model Card) / CC BY-NC-SA 4.0 (Paper) — nicht-kommerziell; kommerziell via Jina AI |

**Lizenz-Hinweis:** Die Model Card nennt CC BY-NC 4.0, das arXiv-Paper CC BY-NC-SA 4.0 — in beiden Fällen **nicht-kommerziell**; kommerzielle Nutzung erfordert eine Jina-AI-Lizenz/API.

## Benchmarks
- **MTEB (Englisch):** übertrifft laut Paper OpenAI text-embedding-3-large.
- **MTEB (Multilingual):** übertrifft multilingual-e5-large-instruct über alle evaluierten mehrsprachigen Tasks (Paper).
- Beispiel-Einzelwert (Model Card, self-reported): ArguAna 43.29. (Aggregierte MTEB-Average-Zahl liegt in der ausklappbaren Evaluations-Liste der Card, nicht als Rohwert im Haupttext — daher hier nicht als exakter Average zitiert.)

## Besonderheiten / Trivia
- LoRA-Task-Adapter erlauben, aus einem Basismodell je nach Aufgabe optimierte Embeddings zu ziehen — ein Weg, das "query:/passage:"-Prefix-Konzept (E5) architektonisch zu erweitern.
- 30 tief getunte Zielsprachen inkl. Deutsch, Französisch, Spanisch, Chinesisch, Japanisch, Arabisch etc.
- Nicht-kommerzielle Lizenz ist ein wichtiger Unterschied zu MIT/Apache-Modellen (BGE, E5, GTE, Nomic) für Produktivnutzung.

## Quellen
1. jina-embeddings-v3 Model Card — https://huggingface.co/jinaai/jina-embeddings-v3
2. jina-embeddings-v3: Multilingual Embeddings With Task LoRA (arXiv:2409.10173) — https://arxiv.org/abs/2409.10173
3. Jina AI Blog Announcement — https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
