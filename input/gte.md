---
title: GTE (General Text Embeddings) — Alibaba
category: embedding
models_covered: [gte-large, gte-large-en-v1.5, gte-Qwen2-1.5B-instruct, gte-Qwen2-7B-instruct]
last_updated: 2026-07-11
sources: [https://huggingface.co/thenlper/gte-large, https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5, https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct, https://arxiv.org/abs/2308.03281]
---

# GTE (General Text Embeddings)

## Übersicht
GTE ("General Text Embeddings") ist eine Embedding-Familie des Institute for Intelligent Computing / Tongyi Lab von Alibaba (ursprünglich DAMO Academy). Das Grundlagen-Paper "Towards General Text Embeddings with Multi-stage Contrastive Learning" (arXiv:2308.03281, Aug. 2023) führte die erste Generation ein (gte-small/base/large auf BERT-Basis). Die Familie entwickelte sich in zwei Richtungen weiter:

- **GTE v1.5** (Encoder-only, long-context): gte-base-en-v1.5 und gte-large-en-v1.5 mit 8192-Token-Kontext.
- **GTE-Qwen2** (Decoder-only / LLM-basiert): gte-Qwen2-1.5B-instruct und gte-Qwen2-7B-instruct, die einen Qwen2-LLM als Backbone nutzen und zu den stärksten offenen MTEB-Modellen ihrer Zeit gehörten.

## Architektur
Die **erste Generation** (thenlper/gte-*) sind **Encoder-only Transformer** auf **BERT**-Basis, trainiert mit **Multi-Stage Contrastive Learning**: zunächst kontrastives Pre-Training auf großem, schwach überwachtem Textpaar-Korpus, dann Fine-Tuning auf hochwertigen, gelabelten Relevanz-Paaren. Mean-Pooling erzeugt das Satz-Embedding.

Die **v1.5-Encoder** (Alibaba-NLP/gte-*-en-v1.5) erweitern den BERT-Backbone auf **8192 Token** Kontext (via verbesserter Positional-Encoding-Strategie/RoPE-artige Skalierung) bei kompakter Größe. Sie benötigen `flash_attn` für effiziente Inferenz.

Die **GTE-Qwen2-Modelle** sind **Decoder-only LLMs** (Qwen2-Backbone) mit **bidirektionaler Attention** (das kausale Masking wird für Embedding aufgehoben) plus **query-seitigem Instruction-Tuning**. Sie liefern hochdimensionale Embeddings (1536 bzw. 3584) und 32k-Kontext — repräsentativ für den Trend, LLMs als Embedding-Backbones zu nutzen.

## Specs-Tabelle

| Modell | Arch. | Params | Dim | Kontext | Sprachen | Lizenz | MTEB(56) |
|---|---|---|---|---|---|---|---|
| gte-large (v1) | BERT enc-only | ~335 M (0.3B) | 1024 | 512 | EN | MIT | 63.13 |
| gte-large-en-v1.5 | BERT enc-only | ~434 M (1.74 GB FP32) | 1024 | 8192 | EN | Apache 2.0 | 65.39 |
| gte-base-en-v1.5 | BERT enc-only | ~137 M (0.51 GB FP32) | 768 | 8192 | EN | Apache 2.0 | 64.11 |
| gte-Qwen2-1.5B-instruct | Qwen2 dec-only | ~1.5B | 1536 | 32k | multilingual | Apache 2.0 | 67.16 |
| gte-Qwen2-7B-instruct | Qwen2 dec-only | ~7B | 3584 | 32k | multilingual | Apache 2.0 | 70.24 |

Weitere v1-Größen: gte-base (768 dim, 512 Kontext), gte-small (384 dim, 512 Kontext).

## Benchmarks (MTEB Average, 56 Tasks)
- gte-large (v1): **63.13** (Retrieval 52.22, STS 83.35, Classification 73.33 — Model Card)
- gte-base-en-v1.5: **64.11**
- gte-large-en-v1.5: **65.39**
- gte-Qwen2-1.5B-instruct: **67.16** | C-MTEB(35) 67.65 | MTEB-fr(26) 66.60
- gte-Qwen2-7B-instruct: **70.24** | C-MTEB(35) 72.05 | MTEB-fr(26) 68.25 | MTEB-pl(26) 67.86

## Besonderheiten / Trivia
- GTE demonstriert innerhalb einer Familie beide dominierenden Embedding-Paradigmen: kompakte BERT-Encoder und LLM-basierte Decoder-Embedder.
- gte-Qwen2-7B-instruct war zeitweise Spitzenreiter bzw. Top-Modell im offenen MTEB-Ranking (>70 Average).
- Die erste GTE-Generation (thenlper) ist nur Englisch; die Qwen2-Varianten sind multilingual.
- Kommerzielle Versionen (text-embedding-v1/v2/v3, gte-rerank) laufen über Alibaba Cloud Model Studio.

## Quellen
1. gte-large (v1) Model Card — https://huggingface.co/thenlper/gte-large
2. gte-large-en-v1.5 Model Card — https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
3. gte-Qwen2-1.5B-instruct Model Card — https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
4. Towards General Text Embeddings with Multi-stage Contrastive Learning (arXiv:2308.03281) — https://arxiv.org/abs/2308.03281
