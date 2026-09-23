---
title: Nomic Embed (nomic-embed-text-v1 / v1.5)
category: embedding
models_covered: [nomic-embed-text-v1, nomic-embed-text-v1.5, nomic-embed-text-v2-moe]
last_updated: 2026-07-11
sources: [https://huggingface.co/nomic-ai/nomic-embed-text-v1, https://huggingface.co/nomic-ai/nomic-embed-text-v1.5, https://arxiv.org/abs/2402.01613, https://github.com/nomic-ai/contrastors, https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models]
---

# Nomic Embed

## Übersicht
Nomic Embed (Nomic AI) ist als **erstes vollständig reproduzierbares** Long-Context-Embedding-Modell bekannt: das Paper "Nomic Embed: Training a Reproducible Long Context Text Embedder" (arXiv:2402.01613, akzeptiert bei TMLR) veröffentlicht **offene Gewichte, offenen Trainingscode UND die vollständigen Trainingsdaten** unter Apache 2.0. Das war ein wichtiger Kontrapunkt zu proprietären Modellen wie OpenAI text-embedding-ada-002, die es auf MTEB und dem Long-Context-Benchmark LoCo übertrifft — bei nur ~137 M Parametern.

- **nomic-embed-text-v1** (Feb. 2024): 8192-Token-Kontext, 768-dim.
- **nomic-embed-text-v1.5**: fügt **Matryoshka Representation Learning** hinzu (Embedding-Dimension zur Laufzeit auf 512/256/128/64 kürzbar bei geringem Qualitätsverlust).
- **nomic-embed-text-v2-moe**: erstes Text-Embedding-Modell mit **Mixture-of-Experts (MoE)**-Architektur; mehrsprachig, trainiert auf 1.6 Mrd. kontrastiven Paaren über ~100 Sprachen (mC4, multilingual CC News).

## Architektur
nomic-embed-text-v1 ist ein **Long-Context-BERT** (basierend auf dem eigenen `nomic-bert-2048`). Statt absoluter Positional Embeddings nutzt es **Rotary Position Embeddings (RoPE)**, was Skalierung über die Trainings-Kontextlänge hinaus erlaubt (dynamische RoPE-Konfiguration: `rope_theta = 1000`, `rope_type = "dynamic"`, `factor = 2.0`, um von 2048 auf 8192 Token zu skalieren). Weitere Effizienz-Bausteine der nomic-bert-Architektur umfassen typischerweise SwiGLU-Aktivierungen und Flash-Attention. Satz-Embedding via Mean-Pooling.

**Task-Prefixes** (verpflichtend): `search_document:` (Dokumente/RAG-Index), `search_query:` (Fragen/Queries), `clustering:`, `classification:`.

**Training:** mehrstufige Pipeline — unsupervised **contrastive pretraining** auf großem Web-Paar-Korpus, dann hochwertiges **supervised finetuning** mit **Hard-Example-Mining**. Der vollständige kuratierte Datensatz und Code liegen offen im Repo `nomic-ai/contrastors`.

v1.5 ergänzt **Matryoshka Representation Learning**: das Modell wird so trainiert, dass die ersten k Dimensionen des Embeddings für sich genommen brauchbar sind → truncierbare Embeddings (768 → 512/256/128/64) für Speicher-/Latenz-Tradeoffs.

## Specs-Tabelle (nomic-embed-text-v1)

| Merkmal | Wert |
|---|---|
| Org | Nomic AI |
| Release | Feb. 2024 |
| Basis | nomic-bert-2048 (Long-Context BERT) |
| Parameter | ~137 M (0.1B) |
| Embedding-Dimension | 768 (v1.5: Matryoshka 768→64) |
| Kontextlänge | 8192 Token |
| Positional Encoding | Rotary (RoPE, dynamisch skalierbar) |
| Sprachen | Englisch (v2-moe: ~100 mehrsprachig) |
| Lizenz | Apache 2.0 (Gewichte + Code + Daten) |
| Prefix | search_document / search_query / clustering / classification |

## Benchmarks (nomic-embed-text-v1, Model Card)
- **MTEB:** 62.39 (vs. OpenAI ada-002 60.99, text-embedding-3-small 62.26)
- **LoCo (Long-Context):** 85.53 (vs. ada-002 52.7, text-embedding-3-small 82.40)
- **Jina Long Context:** 54.16 (vs. ada-002 55.25, text-embedding-3-small 58.20)

## Besonderheiten / Trivia
- Alleinstellungsmerkmal: **Full reproducibility** — offene Daten + Code + Gewichte, ein 5-Mio-Sample-Ausschnitt der Pretraining-Daten ist auf Nomic Atlas visualisiert.
- v1.5 popularisierte truncierbare Matryoshka-Embeddings im Open-Source-Bereich.
- v2-moe ist laut Nomic das erste Text-Embedding-Modell mit MoE-Routing (aktive vs. Gesamt-Parameter-Trennung wie bei MoE-LLMs).
- Sehr klein (137 M) → gut für lokale/CPU-nahe Deployments und lange Dokumente.

## Quellen
1. nomic-embed-text-v1 Model Card — https://huggingface.co/nomic-ai/nomic-embed-text-v1
2. nomic-embed-text-v1.5 Model Card — https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
3. Nomic Embed Technical Report (arXiv:2402.01613) — https://arxiv.org/abs/2402.01613
4. contrastors Trainingscode/Daten — https://github.com/nomic-ai/contrastors
5. BentoML: Open-Source Embedding Models 2026 (v2-moe) — https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
