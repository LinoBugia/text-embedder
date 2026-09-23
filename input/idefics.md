---
title: Idefics2 / Idefics3
category: vision-language
models_covered: [Idefics2-8B, Idefics3-8B-Llama3]
last_updated: 2026-07-11
sources: [https://huggingface.co/blog/idefics2, https://arxiv.org/abs/2405.02246, https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3, https://huggingface.co/papers/2408.12637]
---

# Idefics2 / Idefics3

## Übersicht
Idefics (**I**mage-aware **D**ecoder **E**nhanced à la **F**lamingo with **I**nterleaved **C**ross-attention**S**) ist die offene VLM-Reihe von **HuggingFace**, die als Community-Reproduktion und Weiterentwicklung von DeepMinds Flamingo begann. Die Reihe legt besonderen Wert auf **vollständig offene Trainingsdaten und Pipeline** (OBELICS, The Cauldron, WebSight, Docmatix).

**Idefics2** (April 2024, 8B) verließ das Flamingo-artige Gated-Cross-Attention-Design zugunsten einer einfacheren Architektur mit Perceiver-Pooling und direkter Konkatenation. **Idefics3** (August 2024, 8B) wechselte auf Llama-3.1-8B als Backbone, ersetzte das Perceiver-Pooling durch Pixel-Shuffle und verbesserte durch den Docmatix-Datensatz Dokumenten-/OCR-Fähigkeiten drastisch (DocVQA 74.0 → 87.7).

## Architektur
**Idefics2-8B:**
- **Vision Encoder:** **SigLIP-So400m-patch14-384** (0,9B Parameter), verarbeitet Bilder in nativer Auflösung/Seitenverhältnis (bis 980×980) via **NaViT-Strategie** statt fixem Resizing. Optionales Sub-Image-Splitting (adaptiert von SPHINX/LLaVA-NeXT) zerlegt hochauflösende Bilder in 4 Sub-Images.
- **Bridge:** Abkehr von Idefics1's Gated-Cross-Attention. Visuelle Features werden extrahiert, mit einem **lernbaren Perceiver-Resampler** auf eine kleinere feste Token-Menge gepoolt, durch ein **MLP** projiziert und **direkt mit den Text-Embeddings konkateniert** (Self-Attention-Fusion im LLM statt separater Cross-Attention).
- **LLM-Backbone:** **Mistral-7B-v0.1**.

**Idefics3-8B-Llama3:**
- **Vision Encoder:** weiterhin **SigLIP-So400m**.
- **LLM-Backbone:** **Meta Llama-3.1-8B-Instruct**.
- **Bridge:** **Pixel Shuffle** statt Perceiver-Resampler. Bilder werden in Kacheln von **364×364** zerlegt, jede Kachel wird zu **169 visuellen Tokens** encodiert; große Bilder werden in mehrere Sub-Images ≤364×364 zerlegt und einzeln encodiert.
- Optimierte visuelle Token-Strategie + Docmatix-Training → starker Sprung bei OCR, Dokumentenverständnis und visuellem Reasoning.

## Specs

| Modell | Release | Größe | Vision-Encoder | LLM-Backbone | Lizenz | Bild-Auflösung |
|---|---|---|---|---|---|---|
| Idefics2-8B | 2024-04 | 8B | SigLIP-So400m (0.9B) | Mistral-7B-v0.1 | Apache-2.0 | nativ bis 980×980 (NaViT), opt. 4 Sub-Images |
| Idefics3-8B-Llama3 | 2024-08 | 8B | SigLIP-So400m | Llama-3.1-8B-Instruct | Apache-2.0 | Kacheln 364×364 → 169 Tokens/Kachel |

- **Kontextlänge:** Idefics2 ~32k (Mistral); Idefics3 profitiert vom Llama-3.1-Kontext.
- **Trainingsdaten:** Wikipedia, **OBELICS** (interleaved Web-Dokumente), Image-Caption-Paare (PMD, LAION-COCO), OCR/Dokumenten-Daten (PDFA, IDL, RenderedText), Image-to-Code (**WebSight**). Instruction-Tuning auf **The Cauldron** (kuratierte Sammlung von 50 Datensätzen) + Text-Instruktionsdaten. Idefics3 zusätzlich auf **Docmatix** (dokumentfokussiert) und erweitertem Cauldron.
- **Lizenz:** Apache-2.0 für Gewichte sowie die zugrunde liegenden Mistral-/SigLIP-Basismodelle (Idefics2). Idefics3-8B-Llama3 ebenfalls Apache-2.0 gelistet (Llama-3.1-Backbone unterliegt zusätzlich der Llama-Community-Lizenz — Model Card beachten).

## Benchmarks

| Benchmark | Idefics2-8B (ohne / mit Split) | Idefics3-8B |
|---|---|---|
| MMMU (val) | 43.5 / 43.0 | 46.6 |
| DocVQA (test) | 67.3 / 74.0 | **87.7** |
| MathVista (testmini/test) | 51.6 / 51.4 | 58.4 |
| TextVQA (val) | 70.4 / 73.0 | 74.9 |
| MMStar (val) | — | 55.9 |
| MMBench (test) | 76.8 / 76.7 | — |
| VQAv2 (test-dev) | 80.8 / 81.2 | — |

Idefics3 vs. Idefics2 zeigt vor allem auf dokument-/OCR-lastigen Benchmarks (DocVQA +13.7 Punkte) und MathVista große Fortschritte, primär durch Docmatix und optimierte visuelle Token.

## Architektur-Besonderheiten
- Konsequente Offenheit: alle Trainingsdatensätze (OBELICS, The Cauldron, WebSight, Docmatix) sind öffentlich → Referenz für reproduzierbare VLM-Forschung.
- Übergang von Flamingo-Cross-Attention (Idefics1) zu Self-Attention-Konkatenation (Idefics2) — folgt dem LLaVA-Trend zu einfacheren Bridges.
- Idefics3: Pixel-Shuffle + Docmatix als OCR-/Dokumenten-Booster.
- NaViT-Strategie für native Auflösung bereits in Idefics2.

## Quellen
1. Idefics2 Blog (Architektur, SigLIP, Perceiver, NaViT, Datensätze, Benchmarks, Apache-2.0) — https://huggingface.co/blog/idefics2
2. Idefics2 Paper — https://arxiv.org/abs/2405.02246
3. Idefics3-8B-Llama3 Model Card (Llama-3.1-Backbone, Pixel Shuffle, 169 Tokens/Kachel, Docmatix, Benchmarks) — https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
4. Idefics3 Technical Report — https://huggingface.co/papers/2408.12637
5. OBELICS / The Cauldron / Docmatix Datensätze — https://huggingface.co/datasets/HuggingFaceM4/OBELICS , https://huggingface.co/datasets/HuggingFaceM4/the_cauldron , https://huggingface.co/datasets/HuggingFaceM4/Docmatix
