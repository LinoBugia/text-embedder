---
title: Pixtral 12B
category: vision-language
models_covered: [Pixtral 12B]
last_updated: 2026-07-11
sources: [https://mistral.ai/news/pixtral-12b/, https://huggingface.co/mistralai/Pixtral-12B-2409, https://arxiv.org/abs/2410.07073]
---

# Pixtral 12B

## Übersicht
Pixtral 12B (Release **17. September 2024**) ist das erste native Vision-Language-Model von **Mistral AI**. Es kombiniert einen von Grund auf trainierten Vision-Encoder mit dem Mistral-Nemo-12B-Decoder und zeichnet sich durch starkes Instruction-Following, variable Bildauflösung, mehrere Bilder pro Prompt und ein großes 128k-Kontextfenster aus. Bei Release übertraf Pixtral vergleichbare offene Modelle (Qwen2-VL 7B, LLaVA-OneVision 7B, Phi-3.5 Vision) deutlich beim Instruction-Following. Das Modell ist offen unter Apache 2.0 lizenziert. *(Stand 2026 im Mistral-Katalog als "deprecated" markiert und durch neuere Vision-Modelle abgelöst, bleibt aber ein prominentes offenes VLM.)*

## Architektur
Pixtral folgt dem "Encoder + Decoder"-Muster, unterscheidet sich aber durch den **von Grund auf trainierten** Vision-Encoder:

- **Vision Encoder (400M, from scratch):** Trainiert von Grund auf, unterstützt **nativ variable Bildgrößen und Seitenverhältnisse**. Das Bild wird in **16×16-Patches** bei nativer Auflösung zerlegt und in Tokens umgewandelt. Die Patches werden zu einer Sequenz geflacht, wobei **`[IMG BREAK]`-Tokens zwischen Zeilen** und ein **`[IMG END]`-Token am Bildende** eingefügt werden, damit das Modell die 2D-Struktur rekonstruieren kann. Der Encoder nutzt **2D-RoPE** (Rotary Position Embeddings in Höhe und Breite).
- **Decoder-Backbone:** Ein **12B-Parameter-Multimodal-Transformer-Decoder auf Basis von Mistral Nemo**. Verarbeitet interleaved Text- und Bild-Tokens mit Next-Token-Prediction — beliebig viele Bilder beliebiger Größe können innerhalb des Kontextfensters eingestreut werden.
- **Bridge:** Die Bild-Tokens des Encoders werden direkt in die Decoder-Sequenz eingespeist (vision-language-Fusion im Decoder), ohne separaten Perceiver.

## Specs

| Attribut | Wert |
|---|---|
| Org | Mistral AI |
| Release | 2024-09-17 |
| Decoder-Parameter | 12B (Mistral Nemo) |
| Vision-Encoder | 400M, von Grund auf trainiert |
| Patch-Größe | 16×16, native Auflösung |
| Positional Encoding | 2D-RoPE |
| Kontextfenster | 128k Tokens (mehrere Bilder beliebiger Größe) |
| Lizenz | Apache 2.0 |
| HF-Repo | mistralai/Pixtral-12B-2409 |
| API-Name | pixtral-12b-2409 |

## Benchmarks (Mistral-Angaben)

| Benchmark | Pixtral 12B |
|---|---|
| MMMU (CoT) | 52.5% |
| MathVista (CoT) | 58.0% |
| ChartQA (CoT) | 81.8% |
| DocVQA (ANLS) | 90.7% |
| VQAv2 (VQA Match) | 78.6% |

- **Instruction Following:** Übertrifft Qwen2-VL 7B, LLaVA-OneVision 7B und Phi-3.5 Vision auf IF-Eval und MT-Bench (laut Mistral um ~20% auf Schlüssel-Benchmarks).

## Architektur-Besonderheiten
- Einer der wenigen offenen VLMs mit **von Grund auf trainiertem** Vision-Encoder (statt vortrainiertem CLIP/SigLIP).
- Explizite `[IMG BREAK]`/`[IMG END]`-Tokens zur Kodierung der 2D-Bildstruktur in einer 1D-Sequenz.
- Sehr großes 128k-Kontextfenster erlaubt viele Bilder + langen Text gemeinsam.
- Starkes Instruction-Following bei vollständiger Apache-2.0-Offenheit.

## Quellen
1. Pixtral 12B Ankündigung (Encoder 400M from scratch, IMG BREAK/END, 2D-RoPE, Mistral Nemo, 128k, Apache 2.0, Benchmarks) — https://mistral.ai/news/pixtral-12b/
2. Pixtral 12B HF Model Card — https://huggingface.co/mistralai/Pixtral-12B-2409
3. Pixtral 12B Technical Report — https://arxiv.org/abs/2410.07073
4. Mistral Vision Capabilities (Nachfolger) — https://docs.mistral.ai/capabilities/vision
