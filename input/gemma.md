---
title: Gemma 2 / Gemma 3 (Google)
category: dense-decoder
models_covered: [Gemma 2 2B, Gemma 2 9B, Gemma 2 27B, Gemma 3 1B, Gemma 3 4B, Gemma 3 12B, Gemma 3 27B]
last_updated: 2026-07-11
sources: [https://arxiv.org/html/2503.19786v1, https://huggingface.co/blog/gemma3, https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/, https://namangoyal.com/blog/2025/gemma3/]
---

# Gemma 2 / Gemma 3 (Google)

## Übersicht
Gemma ist Googles Familie offener Modelle, abgeleitet aus der Technologie der geschlossenen Gemini-Modelle. **Gemma 2** (Juni 2024, 2B/9B/27B) setzte auf Wissensdistillation aus größeren Modellen und lieferte für seine Größe sehr starke Ergebnisse. **Gemma 3** (März 2025, 1B/4B/12B/27B) ist der große Sprung: **multimodal** (Bild + Text ab 4B), **128K Kontext**, über 140 Sprachen — und laut Google auf der LMSys Chatbot Arena unter den Top-Modellen (Elo ~1339 für 27B-IT), vergleichbar mit deutlich größeren Modellen.

## Architektur
Beide Generationen sind Dense-Decoder-only-Transformer mit RMSNorm, RoPE, GeGLU/GQA. Ihr charakteristisches Merkmal ist die **Verschränkung lokaler und globaler Attention-Layer**, um langen Kontext speichereffizient zu handhaben:

- **Gemma 2**: alterniert lokale (Sliding-Window, Fenster 4096) und globale Attention im Verhältnis **1:1**. Zusätzlich **logit soft-capping** (Attention- und Final-Logits werden per tanh begrenzt) und **RMSNorm sowohl vor als auch nach** jedem Sub-Layer (Pre- und Post-Norm).
- **Gemma 3**: verschiebt das Verhältnis auf **5 lokale zu 1 globalen Layer** und verkleinert das lokale Fenster auf **1024 Tokens** — das reduziert den KV-Cache-Speicher stark, ohne die Perplexität nennenswert zu verschlechtern. Die **RoPE-Basisfrequenz** wird auf den globalen Layern von 10k (Gemma 2) auf **1M** angehoben (lokale Layer bleiben bei 10k), um 128K Kontext zu ermöglichen. Gemma 3 ersetzt logit soft-capping durch **QK-Norm**.

**Multimodalität (Gemma 3, ab 4B)**: Ein **SigLIP**-Vision-Encoder verarbeitet auf 896x896 skalierte Bilder; ein "Pan-and-Scan"-Verfahren erlaubt adaptives Zuschneiden hochauflösender/nicht-quadratischer Bilder. Bilder erhalten volle bidirektionale Attention, Text bleibt kausal. Der **Tokenizer** ist der Gemini-2.0-SentencePiece-Tokenizer mit **262K** Einträgen, was CJK-Sprachen (Chinesisch/Japanisch/Koreanisch) deutlich besser kodiert.

## Specs

| Attribut | Gemma 2 | Gemma 3 |
|---|---|---|
| Organisation | Google (DeepMind) | Google (DeepMind) |
| Release | Juni 2024 | 12. März 2025 |
| Lizenz | Gemma License (eigene, kommerziell nutzbar mit Nutzungsrichtlinien) | Gemma License |
| Größen | 2B, 9B, 27B | 1B, 4B, 12B, 27B |
| Kontextlänge | 8K | 128K (1B: 32K) |
| Modalität | Text | Multimodal (Bild+Text ab 4B), 140+ Sprachen |
| Trainingsdaten | 2B: 2T · 9B: 8T · 27B: 13T Tokens | mehr multilinguale Daten (≈2x) |
| Vokabular | 256K (SentencePiece) | 262K (Gemini-2.0 SentencePiece) |
| Attention | lokal/global 1:1, Fenster 4096, logit soft-cap | lokal/global 5:1, Fenster 1024, QK-Norm |
| PosEnc | RoPE (Basis 10k) | RoPE (global 1M / lokal 10k) |
| Vision-Encoder | — | SigLIP (896x896, Pan-and-Scan) |

## Benchmarks
Gemma 3 27B-IT (aus HuggingFace/Google-Angaben):

- **LMSys Chatbot Arena Elo**: 1339 (Top-10 global, vergleichbar mit o1-preview unter Text-Modellen).
- **MMLU-Pro**: 67,5
- **MATH**: 69,0
- **GPQA Diamond**: 42,4
- **LiveCodeBench**: 29,7
- **Bird-SQL**: 54,4
- **FACTS Grounding**: 74,9
- **MMMU (multimodal)**: 64,9
- **SimpleQA**: 10,0

Google gibt an, dass **Gemma-3-4B-IT das Gemma-2-27B-IT** übertrifft und **Gemma-3-27B-IT in mehreren Benchmarks Gemini 1.5-Pro** schlägt.

## Besonderheiten / Trivia
- Gemma nutzt eine **eigene "Gemma License"** (nicht Apache/MIT) mit Nutzungsrichtlinien — kommerziell erlaubt, aber mit Einschränkungen bei bestimmten Anwendungen.
- Das lokal/global-Attention-Schema (5:1 in Gemma 3) ist der Schlüssel zu geringem KV-Cache-Bedarf bei 128K Kontext.
- Gemma 3 1B ist bewusst text-only und auf 32K begrenzt (Edge-fokussiert).
- Distillation aus Gemini-Lehrer-Modellen ist ein Kern-Trainingsrezept beider Generationen.

## Quellen
1. Gemma 3 Technical Report — https://arxiv.org/html/2503.19786v1
2. Welcome Gemma 3 (HuggingFace Blog) — https://huggingface.co/blog/gemma3
3. Gemma explained (Google Developers Blog) — https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/
4. Gemma 3 Technical Deep Dive (Naman Goyal) — https://namangoyal.com/blog/2025/gemma3/
