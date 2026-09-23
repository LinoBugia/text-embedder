---
title: BGE (BAAI General Embedding) — bge-large-en-v1.5 & bge-m3
category: embedding
models_covered: [bge-large-en-v1.5, bge-m3]
last_updated: 2026-07-11
sources: [https://huggingface.co/BAAI/bge-large-en-v1.5, https://huggingface.co/BAAI/bge-m3, https://arxiv.org/abs/2309.07597, https://arxiv.org/abs/2402.03216, https://github.com/FlagOpen/FlagEmbedding]
---

# BGE (BAAI General Embedding)

## Übersicht
BGE ("BAAI General Embedding") ist eine Familie von Open-Source-Embedding-Modellen der Beijing Academy of Artificial Intelligence (BAAI), die über das FlagEmbedding-Projekt veröffentlicht wird. Sie zählt zu den einflussreichsten offenen Embedding-Modell-Serien und dominierte 2023–2024 die MTEB-Leaderboards. Zwei zentrale Vertreter:

- **bge-large-en-v1.5** (Sept. 2023): englisches Dense-Embedding-Modell auf BERT-Basis, lange Zeit Top-Referenz auf MTEB (Englisch). Die v1.5-Revision behob eine problematische Ähnlichkeits-Verteilung und verbesserte Retrieval ohne Instruction-Prefix.
- **bge-m3** (Feb. 2024): "Multi-Lingual, Multi-Functionality, Multi-Granularity"-Modell, das Dense-, Sparse- (Lexical) und Multi-Vector-Retrieval (ColBERT-Stil) in einem einzigen Modell vereint und Kontextlängen bis 8192 Token unterstützt.

Die zugehörigen Papers sind C-Pack (arXiv:2309.07597) für die BGE-v1.5-Familie und BGE M3-Embedding (arXiv:2402.03216) für bge-m3.

## Architektur
**bge-large-en-v1.5** ist ein Encoder-only Transformer auf **BERT**-Architektur (bge-large ≈ BERT-large: 24 Layer). Trainiert wird zunächst mit **RetroMAE** (masked auto-encoder Pre-Training für Retrieval), danach folgt kontrastives Fine-Tuning auf großskaligen Textpaaren mit einer Temperatur von 0.01. Das Embedding wird über den [CLS]-Token gebildet; für Queries wird optional das Prefix "Represent this sentence for searching relevant passages:" empfohlen (v1.5 funktioniert aber auch ohne Instruction).

**bge-m3** baut auf **XLM-RoBERTa-large** auf, dessen maximale Sequenzlänge auf 8192 Token erweitert und via RetroMAE nach-pretrained wurde (BAAI/bge-m3-retromae). Die zentrale Innovation ist **Multi-Functionality**: ein einziges Modell erzeugt gleichzeitig
1. **Dense Retrieval** — ein 1024-dim Embedding (Standard-DPR/BGE-Stil, [CLS]-Pooling),
2. **Sparse / Lexical Retrieval** — On-the-fly-Token-Gewichte (SPLADE-artig, ähnlich BM25) ohne zusätzliche Generierungskosten,
3. **Multi-Vector / Late Interaction** — ColBERT-artige Token-für-Token-Interaktion.

Trainiert wird mit **Self-Knowledge Distillation**: Die kombinierten Relevanz-Scores aus Dense-, Sparse- und ColBERT-Modus dienen als Lehrer-Signal, um jeden einzelnen Modus zu schärfen. Zusätzlich verbessert **MCLS** (Multiple CLS) die Long-Text-Verarbeitung ohne spezielles Fine-Tuning.

## Specs-Tabelle

| Merkmal | bge-large-en-v1.5 | bge-m3 |
|---|---|---|
| Org | BAAI | BAAI |
| Release | 12. Sept. 2023 | 5. Feb. 2024 |
| Basis-Architektur | BERT (Encoder-only) | XLM-RoBERTa-large + RetroMAE |
| Parameter | ~335 M (0.3B) | ~568 M (0.6B, XLM-R-large-Basis) |
| Embedding-Dimension | 1024 | 1024 (Dense) |
| Kontextlänge | 512 Token | 8192 Token |
| Sprachen | Englisch | 100+ |
| Lizenz | MIT | MIT |
| Retrieval-Modi | Dense | Dense + Sparse + Multi-Vector (Hybrid) |
| Pooling | [CLS] | [CLS] (Dense) |

## Benchmarks
**bge-large-en-v1.5** (MTEB, 56 Tasks, gemäß offizieller Model Card):
- Average: **64.23**
- Retrieval (15 Tasks): 54.29
- Clustering (11): 46.08
- Pair Classification (3): 87.12
- Reranking (4): 60.03
- STS (10): 83.11
- Classification (12): 75.97
- Summarization (1): 31.61

**bge-m3** (gemäß Paper/Model Card): Multilingualer State-of-the-Art zum Release-Zeitpunkt auf **MIRACL** (mehrsprachiges Retrieval) und **MKQA** (Cross-Lingual); übertraf in community-evaluierten Benchmarks OpenAI-Embedding-Modelle im mehrsprachigen und Long-Doc-Bereich. Für Long-Document-Retrieval wurde der eigene **MLDR**-Datensatz (13 Sprachen, LLM-generiert) erstellt. (Exakte MIRACL-nDCG@10-Zahlen liegen in der Model Card als Grafiken vor, nicht als Rohtext — daher hier nicht als exakte Zahl zitiert.)

## Besonderheiten / Trivia
- bge-m3 wird oft als "mother of all embedding models" bezeichnet, weil ein einziges Modell Dense-, Sparse- und ColBERT-Retrieval liefert und damit Hybrid-Suche in Vektor-DBs (Milvus, Vespa) ohne zusätzliche Modelle ermöglicht.
- Zur BGE-Familie gehören außerdem populäre Reranker (bge-reranker-v2-m3), die häufig als zweite Stufe hinter dem Dense-Retriever eingesetzt werden.
- MIT-Lizenz macht beide kommerziell frei nutzbar.

## Quellen
1. bge-large-en-v1.5 Model Card — https://huggingface.co/BAAI/bge-large-en-v1.5
2. bge-m3 Model Card — https://huggingface.co/BAAI/bge-m3
3. C-Pack: Packed Resources For General Chinese Embeddings (BGE Paper) — https://arxiv.org/abs/2309.07597
4. BGE M3-Embedding Paper (arXiv:2402.03216) — https://arxiv.org/abs/2402.03216
5. FlagEmbedding GitHub — https://github.com/FlagOpen/FlagEmbedding
6. Modal MTEB Leaderboard Article — https://modal.com/blog/mteb-leaderboard-article
