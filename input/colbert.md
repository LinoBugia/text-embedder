---
title: ColBERT / ColBERTv2 — Late Interaction Retrieval
category: embedding
models_covered: [ColBERT (v1), ColBERTv2]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2004.12832, https://arxiv.org/abs/2112.01488, https://arxiv.org/abs/2205.09707, https://github.com/stanford-futuredata/ColBERT]
---

# ColBERT / ColBERTv2 (Late Interaction)

## Übersicht
ColBERT ("Contextualized Late Interaction over BERT") ist ein Retrieval-Paradigma von Omar Khattab und Matei Zaharia (Stanford Future Data). Im Gegensatz zu Single-Vector-Dense-Embeddings (BGE, E5, GTE, Nomic), die Query und Dokument je auf **einen** Vektor komprimieren, repräsentiert ColBERT beide als **Matrix von Token-Level-Embeddings** und berechnet Relevanz über eine feingranulare **Late-Interaction** (MaxSim). Das verbindet die Qualität voller Cross-Encoder-Interaktion mit der Skalierbarkeit vorab berechneter Repräsentationen.

- **ColBERTv1** (SIGIR 2020, arXiv:2004.12832): Einführung der Late Interaction.
- **ColBERTv2** (NAACL 2022, arXiv:2112.01488): fügt aggressive Residual-Kompression und Denoised Supervision/Distillation hinzu, reduziert den Speicherbedarf um das 6–10-fache und erreicht State-of-the-Art auf MS MARCO (in-domain) und BEIR (out-of-domain).

## Architektur
ColBERT nutzt einen **BERT-Backbone** (typischerweise `bert-base-uncased`). Query und Dokument werden getrennt encodiert; jedes Token erhält ein eigenes, kontextualisiertes Embedding der Dimension **128** (statt eines einzigen Pool-Vektors).

**Late Interaction / MaxSim:** Der Relevanz-Score zwischen Query Q (mit Token-Embeddings q_i) und Dokument D (mit Token-Embeddings d_j) ist die Summe über die maximale Ähnlichkeit jedes Query-Tokens zu irgendeinem Dokument-Token:

`Score(Q, D) = Σ_i  max_j  (q_i · d_j)`

Dieses MaxSim erlaubt es, Dokument-Embeddings **offline vorzuberechnen** und zu indexieren; die eigentliche Interaktion passiert erst spät (daher "late"), günstig zur Query-Zeit.

**ColBERTv2-Verbesserungen:**
- **Residual Compression:** Token-Vektoren werden als (Zentroid-ID + komprimiertes Residuum) gespeichert, konfigurierbar via Quantisierung (`nbits=2`) → 6–10× kleinerer Index.
- **Denoised Supervision / Distillation:** Training mit destillierten Relevanz-Signalen (Cross-Encoder-Teacher) und Hard Negatives verbessert In- und Out-of-Domain-Qualität.
- **PLAID Engine** (CIKM 2022, arXiv:2205.09707): effizienter Late-Interaction-Suchmaschinen-Kern, der Kandidaten über Zentroide vorfiltert und so die Latenz drastisch senkt.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Org | Stanford Future Data (Khattab, Santhanam, Saad-Falcon, Potts, Zaharia) |
| Release | ColBERTv1: SIGIR 2020 · ColBERTv2: NAACL 2022 |
| Backbone | BERT (z.B. bert-base-uncased) |
| Repräsentation | Multi-Vector, Token-Level (dim=128 pro Token) |
| Scoring | Late Interaction / MaxSim |
| Kompression (v2) | Residual + Quantisierung (nbits=2), 6–10× kleiner |
| Such-Engine | PLAID |
| Lizenz | MIT (Code-Repo) / Paper CC BY 4.0 |
| Ökosystem | RAGatouille, DSPy, integriert in bge-m3 (Multi-Vector-Modus) |

## Benchmarks
- **ColBERTv2** erreicht laut Paper **State-of-the-Art** auf **MS MARCO Passage Ranking** (in-domain) und übertrifft über ein breites Set von **BEIR**-Out-of-Domain-Benchmarks andere Retriever, bei gleichzeitig 6–10× reduziertem Speicherbedarf gegenüber ColBERTv1. (Exakte MRR@10-/nDCG@10-Zahlen stehen in den Ergebnistabellen des Papers; hier nicht einzeln zitiert, da nur aus Volltext-Tabellen entnehmbar.)

## Besonderheiten / Trivia
- Late Interaction ist ein eigenes Paradigma zwischen bi-Encodern (schnell, ein Vektor) und Cross-Encodern (genau, teuer) — es liefert nahezu Cross-Encoder-Qualität bei vorab-indexierbaren Dokumenten.
- Der Multi-Vector/ColBERT-Modus von **bge-m3** übernimmt genau dieses Prinzip.
- **RAGatouille** (bclavie) macht ColBERT für RAG-Pipelines leicht nutzbar; **DSPy** (ebenfalls Khattab) integriert ColBERT als Retriever.
- Nachteil: Multi-Vector-Indizes sind trotz Kompression größer und komplexer als Single-Vector-Dense-Indizes.

## Quellen
1. ColBERT (SIGIR 2020, arXiv:2004.12832) — https://arxiv.org/abs/2004.12832
2. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (NAACL 2022, arXiv:2112.01488) — https://arxiv.org/abs/2112.01488
3. PLAID: An Efficient Engine for Late Interaction Retrieval (CIKM 2022, arXiv:2205.09707) — https://arxiv.org/abs/2205.09707
4. ColBERT GitHub (Stanford Future Data) — https://github.com/stanford-futuredata/ColBERT
