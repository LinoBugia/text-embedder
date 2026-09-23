---
title: PaliGemma
category: vision-language
models_covered: [PaliGemma (3B)]
last_updated: 2026-07-11
sources: [https://huggingface.co/blog/paligemma, https://arxiv.org/abs/2310.09199, https://github.com/google-research/big_vision, https://huggingface.co/docs/transformers/model_doc/paligemma]
---

# PaliGemma

## Übersicht
PaliGemma (Release **Mai 2024**) ist Googles kompaktes offenes Vision-Language-Model mit ~3 Mrd. Parametern. Es kombiniert den SigLIP-So400m-Vision-Encoder mit dem Gemma-2B-Sprachmodell und folgt dem **PaLI-3-Trainingsrezept**. Anders als Chat-orientierte VLMs (LLaVA, Qwen-VL) ist PaliGemma primär ein **Base-/Transfer-Modell**: Es ist darauf ausgelegt, per Fine-Tuning auf konkrete Aufgaben (VQA, Captioning, Segmentierung, Detection, OCR, Remote-Sensing) spezialisiert zu werden, statt out-of-the-box als Assistent zu chatten. Es ist eines der stärksten VLMs seiner Größenklasse und dank Kompaktheit gut für ressourcenbeschränkte Deployments geeignet.

## Architektur
PaliGemma besteht aus drei Komponenten und einem charakteristischen Attention-Muster:

- **Vision Encoder:** **SigLIP-So400m-patch14-384** — ein per SigLIP-Loss vortrainierter ViT. Erzeugt 1152-dimensionale Embeddings pro Patch (Patch-Größe 14).
- **Linearer Adapter (Bridge):** Ein **linearer Projektor** hebt die 1152-dim SigLIP-Embeddings auf 2048 Dimensionen an, um sie an den Token-Einbettungsraum von Gemma-2B anzugleichen. Bilder werden per bikubischem Resampling auf die Zielauflösung gebracht.
- **Sprachmodell:** **Gemma-2B**, ein Decoder-Transformer.
- **Prefix-LM-Attention:** PaliGemma nutzt **kein rein kausales** Attention-Muster. Der gesamte Eingabe-Prefix — Bild-Embeddings + `<bos>`-Token + Prompt-Text + `\n`-Token — erhält **volle (bidirektionale) Block-Attention**, während nur die autoregressiv generierte Textausgabe eine **kausale Maske** verwendet. Dadurch kann das Modell beim "Lesen" der Aufgabe bidirektional über Bild und Prompt attendieren.

**Auflösung & visuelle Tokens:** PaliGemma erscheint in drei quadratischen Auflösungen, mit entsprechend unterschiedlicher Tokenzahl:

| Auflösung | Bild-Tokens |
|---|---|
| 224×224 | 256 |
| 448×448 | 1024 |
| 896×896 | 4096 |

Höhere Auflösungen sind vor allem für dichte Aufgaben (OCR, Detection, Segmentierung) sinnvoll.

## Specs

| Attribut | Wert |
|---|---|
| Org | Google |
| Release | 2024-05 |
| Gesamt-Parameter | ~3B (z.B. 2,93B im QLoRA-Setup) |
| Vision-Encoder | SigLIP-So400m (Patch 14) |
| Sprachmodell | Gemma-2B |
| Adapter | linear (1152 → 2048) |
| Attention | Prefix-LM (voll auf Prefix, kausal auf Output) |
| Auflösungen | 224 / 448 / 896 (quadratisch) |
| Präzisionen | bfloat16, float16, float32 |
| Kontextlänge | 8k (Gemma) |
| Trainingsrezept | PaLI-3 |
| Lizenz | Gemma-Lizenz (Zustimmung auf HF nötig) |

**Checkpoint-Varianten:**
- **PT (Pretrained):** für nachgelagertes Fine-Tuning.
- **Mix:** auf einer Aufgabenmischung feinjustiert, für allgemeine Inferenz (Forschungsgebrauch).
- **FT (Fine-tuned):** auf spezifische akademische Benchmarks spezialisierte Modelle.

## Benchmarks (ausgewählte fine-tuned Checkpoints)
Da PaliGemma ein Transfer-Modell ist, werden Ergebnisse pro aufgabenspezifischem FT-Checkpoint berichtet:

| Aufgabe / Checkpoint | Metrik | Wert |
|---|---|---|
| VQAv2 (paligemma-3b-ft-vqav2-448) | Accuracy | 85.64 |
| COCO Captions (ft-cococap-448) | CIDEr | 144.6 |
| ScienceQA (ft-science-qa-448, Bild-Subset, ohne CoT) | Accuracy | 95.93 |
| RefCOCO Segmentation (ft-refcoco-seg-896) | Mean IoU | 76.94 |
| RSVQA-HR (ft-rsvqa-hr-224, test) | Accuracy | 92.61 |

## Architektur-Besonderheiten
- Kompaktes 3B-Modell — deutlich kleiner als die meisten Chat-VLMs, gut für Edge/Fine-Tuning.
- **Prefix-LM** statt reiner Causal-Attention: bidirektionale Aufmerksamkeit über Bild + Prompt.
- Base-/Transfer-Philosophie (PaLI-3-Rezept): stark nach aufgabenspezifischem Fine-Tuning, weniger als Zero-Shot-Chatbot.
- Deckt ungewöhnlich breites Aufgabenspektrum ab (inkl. Segmentierung, Detection, Remote-Sensing) via spezialisierte FT-Checkpoints.

## Quellen
1. PaliGemma Blog (SigLIP-So400m + Gemma-2B, linearer Adapter, Prefix-LM, Auflösungen/Tokens, FT-Benchmarks, Gemma-Lizenz) — https://huggingface.co/blog/paligemma
2. PaLI-3 Paper (Trainingsrezept) — https://arxiv.org/abs/2310.09199
3. Big Vision GitHub (offizielle JAX-Implementierung) — https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
4. PaliGemma Transformers-Doku — https://huggingface.co/docs/transformers/model_doc/paligemma
5. PaliGemma Model Collection — https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda
