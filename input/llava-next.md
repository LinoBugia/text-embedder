---
title: LLaVA / LLaVA-NeXT / LLaVA-OneVision
category: vision-language
models_covered: [LLaVA-1.5, LLaVA-NeXT (1.6), LLaVA-OneVision, LLaVA-Video]
last_updated: 2026-07-11
sources: [https://github.com/LLaVA-VL/LLaVA-NeXT, https://llava-vl.github.io/blog/2024-01-30-llava-next/, https://huggingface.co/docs/transformers/en/model_doc/llava_next, https://arxiv.org/abs/2408.03326, https://learnopencv.com/llava-training-a-visual-assistant/]
---

# LLaVA / LLaVA-NeXT / LLaVA-OneVision

## Übersicht
LLaVA (**L**arge **L**anguage **a**nd **V**ision **A**ssistant) ist das einflussreichste offene VLM-Rezept. Die Originalarbeit "Visual Instruction Tuning" (Liu et al., April 2023) zeigte, dass man mit GPT-4-generierten visuellen Instruktionsdaten und einem minimalen Design — eingefrorener CLIP-Encoder + Projektor + LLM — einen leistungsfähigen visuellen Assistenten trainieren kann. LLaVA-1.5 (Oktober 2023) verbesserte dies mit einem MLP-Projektor und akademischen VQA-Daten.

**LLaVA-NeXT (Version 1.6, Januar 2024)** erhöhte die Eingabeauflösung um das 4-fache, führte AnyRes-Kachelung ein und nutzte stärkere LLM-Backbones. **LLaVA-OneVision (August 2024)** vereinheitlichte Single-Image-, Multi-Image- und Video-Verständnis in einem Modell (0.5B/7B/72B) und wechselte auf einen SigLIP-Encoder. **LLaVA-Video** (Oktober 2024) fokussiert Video mit dem synthetischen LLaVA-Video-178K-Datensatz. Die Familie dient bis heute als Referenz-Baseline und Ausgangspunkt für viele abgeleitete Modelle.

## Architektur
Das Kernprinzip ist bewusst einfach ("a model of simplicity"):

- **Vision Encoder:** In LLaVA-1.5/NeXT ein eingefrorener **CLIP ViT-L/14 @ 336px**. Die visuellen Features werden aus der vorletzten Schicht (`vision_feature_layer = -2`) entnommen; das CLS-Token wird standardmäßig entfernt. LLaVA-OneVision wechselte auf **SigLIP-So400m**.
- **Projektor (Bridge):** Ursprünglich ein einzelner linearer Layer, der CLIP-Embeddings (1024-dim) in den LLM-Einbettungsraum (z.B. 5120-dim für Vicuna-13B) abbildet. Ab LLaVA-1.5 ein **zweischichtiges MLP** mit GeLU-Aktivierung. Dies ist der einzige "neu trainierte" Kernbaustein der ursprünglichen Alignment-Stufe.
- **LLM-Backbone:** Austauschbar — Vicuna, Mistral-7B, Nous-Hermes-2-Yi-34B, Llama-3-8B, Qwen-1.5 (72B/110B), Qwen2 (OneVision). Der Backbone bestimmt Größe, Kontextlänge und Lizenz.
- **AnyRes / Dynamic High Resolution (NeXT):** Statt Bilder auf ein festes Quadrat zu skalieren, bewahrt LLaVA-NeXT das Seitenverhältnis und zerlegt hochauflösende Bilder in Kacheln, die einzeln vom Encoder verarbeitet und dann konkateniert werden. Unterstützte Seitenverhältnisse bis 672×672, 336×1344, 1344×336; Grid-Pinpoints z.B. `[[336,672],[672,336],[672,672],[1008,336],[336,1008]]`. Pro Kachel entstehen ~576 visuelle Tokens (24×24 Patches). Dies verbessert insbesondere OCR- und Detail-Aufgaben.

Training erfolgt zweistufig: (1) **Feature Alignment** — nur der Projektor wird trainiert, Encoder und LLM eingefroren; (2) **Visual Instruction Tuning** — Projektor + LLM werden auf Instruktionsdaten feinjustiert (Encoder je nach Variante eingefroren oder mitgetunt).

## Specs

| Variante | Release | Größen | Vision-Encoder | LLM-Backbone | Lizenz |
|---|---|---|---|---|---|
| LLaVA-1.5 | 2023-10 | 7B, 13B | CLIP ViT-L/336 | Vicuna | Llama-Community (Backbone) |
| LLaVA-NeXT (1.6) | 2024-01 | 7B, 13B, 34B | CLIP ViT-L/336 | Vicuna, Mistral-7B, Nous-Hermes-Yi-34B | Code: Apache-2.0; Weights je Backbone |
| LLaVA-NeXT (Stronger) | 2024-05 | 8B, 72B, 110B | CLIP ViT-L/336 | Llama-3-8B, Qwen-1.5-72B/110B | je Backbone |
| LLaVA-OneVision | 2024-08 | 0.5B, 7B, 72B | SigLIP-So400m | Qwen2 | Apache-2.0 |
| LLaVA-Video | 2024-10 | 7B, 72B | SigLIP-So400m | Qwen2 | Apache-2.0 |

- **Kontextlänge:** abhängig vom LLM-Backbone (Vicuna ~4k; Mistral/Qwen2 32k).
- **Visuelle Tokens:** ~576 pro Kachel/Bild (24×24 Patches bei 336px); AnyRes multipliziert dies mit der Kachelzahl.
- **Trainingsdaten:** GPT-4/GPT-4V-generierte Visual-Instruction-Daten (LLaVA-Instruct-150K), akademische VQA-Datensätze; OneVision nutzt eine große kuratierte Single/Multi-Image/Video-Mischung über 47 Benchmarks; LLaVA-Video nutzt LLaVA-Video-178K (synthetisch).

## Architektur-Besonderheiten
- Extrem einfaches, modulares "ViT + MLP + LLM"-Design → leicht reproduzierbar und Referenz für die gesamte offene VLM-Community.
- AnyRes-Kachelung als frühe, einflussreiche Lösung für hochauflösende Eingaben.
- Backbone-Agnostik: dasselbe Rezept skaliert von 0.5B bis 110B.
- OneVision führte "Task Transfer" ein: durch gemeinsames Training auf Bild/Multi-Image/Video emergieren Video-Fähigkeiten teils zero-shot aus reinem Bildtraining.

## Benchmarks
- **LLaVA-NeXT-34B** übertraf bei Release Gemini Pro auf mehreren Benchmarks (laut LLaVA-NeXT-Blog, 2024-01-30).
- **LLaVA-OneVision-72B** erreicht laut Paper (arXiv:2408.03326) SOTA-nahe Werte unter offenen Modellen über die 47 evaluierten Benchmarks; MMMU im Bereich ~48–56 (val) je nach Setup. *Genaue Einzelwerte je Benchmark sind im OneVision-Paper tabelliert — hier nicht vollständig verifiziert, daher als Größenordnung angegeben.*
- Ursprüngliches LLaVA erreichte auf ScienceQA (fine-tuned, Synergie mit GPT-4) 92.53% Accuracy.

## Quellen
1. LLaVA-NeXT GitHub (Release Notes, Größen, Lizenz Apache-2.0) — https://github.com/LLaVA-VL/LLaVA-NeXT
2. LLaVA-NeXT Blog (AnyRes, 4x Auflösung) — https://llava-vl.github.io/blog/2024-01-30-llava-next/
3. LLaVA-NeXT Transformers-Doku (CLIP, Feature-Layer -2, Grid-Pinpoints, ~576 Tokens) — https://huggingface.co/docs/transformers/en/model_doc/llava_next
4. LLaVA-OneVision Paper — https://arxiv.org/abs/2408.03326
5. LLaVA-Video Paper — https://arxiv.org/abs/2410.02713
6. LearnOpenCV — LLaVA-Architektur (linearer vs. MLP-Projektor) — https://learnopencv.com/llava-training-a-visual-assistant/
7. LLaVA Projektseite — https://llava-vl.github.io/
