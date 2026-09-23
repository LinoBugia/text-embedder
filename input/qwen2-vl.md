---
title: Qwen-VL / Qwen2-VL / Qwen2.5-VL
category: vision-language
models_covered: [Qwen-VL, Qwen2-VL, Qwen2.5-VL]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2409.12191, https://arxiv.org/html/2409.12191v1, https://qwenlm.github.io/blog/qwen2.5-vl/, https://github.com/QwenLM/Qwen2-VL, https://debuggercafe.com/qwen2-5-vl/]
---

# Qwen-VL / Qwen2-VL / Qwen2.5-VL

## Übersicht
Die Qwen-VL-Reihe von Alibaba (Qwen-Team) gehört zu den stärksten offenen VLMs, besonders bei Dokumenten-/OCR-, Diagramm-, Video- und Agenten-Aufgaben. **Qwen-VL** (2023) etablierte das Framework aus Vision-Encoder + Adapter + Qwen-LLM. **Qwen2-VL** (September 2024) führte "Naive Dynamic Resolution" und Multimodal-RoPE (M-RoPE) ein und erreichte in den Größen 2B/7B/72B State-of-the-Art unter offenen Modellen. **Qwen2.5-VL** (26. Januar 2025) verbesserte den ViT (Window Attention, RMSNorm/SwiGLU), führte absolute Zeitkodierung für langes Video ein und erschien in 3B/7B/72B.

Qwen2-VL-72B war bei Release über mehrere Benchmarks konkurrenzfähig mit bzw. besser als GPT-4o und Claude 3.5 Sonnet auf dokumentzentrierten Aufgaben.

## Architektur
Qwen2-VL behält das grundlegende Qwen-VL-Framework (Vision-Encoder + LLM) bei, mit drei Kernkomponenten: Vision Encoder, MLP-basierter Vision-Language-Merger und Qwen2-LLM.

- **Vision Encoder:** Ein **ViT mit konstant ~675M Parametern** über alle Modellgrößen hinweg — der visuelle Rechenaufwand bleibt also gleich, während nur der LLM skaliert.
- **Naive Dynamic Resolution:** Bilder beliebiger Auflösung werden dynamisch in eine variable Zahl visueller Tokens umgewandelt, statt auf eine feste Größe skaliert. Statt absoluter Positions-Embeddings nutzt der ViT **2D-RoPE**. Eine abschließende MLP-Schicht **komprimiert benachbarte 2×2-Patches zu einem Token** (z.B. ein 224×224-Bild → 66 Tokens). Das hält die Tokenzahl beherrschbar und erlaubt native Auflösung.
- **Multimodal Rotary Position Embedding (M-RoPE):** Zerlegt das übliche 1D-Rotary-Embedding in **temporale, Höhen- und Breiten-Komponenten**. Bei Bildern bleibt die temporale ID konstant, während räumliche Komponenten variieren; bei Videos inkrementiert die temporale ID pro Frame. So werden Bild-, Video- und Textpositionen einheitlich modelliert.
- **Einheitliche Bild-/Video-Verarbeitung:** Video wird mit **2 FPS** gesampelt und mit 3D-Convolutions (Tiefe 2) als "3D-Tubes" statt 2D-Patches verarbeitet. Frame-Auflösungen werden dynamisch angepasst, um Trainingseingaben auf 16.384 Tokens pro Video zu begrenzen; zur Inferenzzeit kann bis ~80K Tokens extrapoliert werden.

**Qwen2.5-VL-Änderungen:** Der ViT nutzt jetzt **Window Attention** (nur 4 Layer Full Attention, Rest Fenster mit max. 8×8, kleinere Bereiche ohne Padding) zur Effizienzsteigerung, sowie **RMSNorm und SwiGLU** im ViT, um dessen Struktur mit dem LLM zu harmonisieren. Für Video kommt **dynamisches FPS-Training + absolute Zeitkodierung** hinzu (mRoPE-IDs an reale Zeit gekoppelt), was sekundengenaue Lokalisierung in über einstündigen Videos ermöglicht. Bounding-Boxes/Punkte werden in echter Bildgröße (ohne Normalisierung) repräsentiert.

## Specs

| Variante | Release | Größen (LLM) | Vision-Encoder | Kontext | Lizenz |
|---|---|---|---|---|---|
| Qwen2-VL-2B | 2024-09 | 1.5B | ViT ~675M | 32k (bis ~80k extrapoliert) | Apache-2.0 |
| Qwen2-VL-7B | 2024-09 | 7.6B | ViT ~675M | 32k | Apache-2.0 |
| Qwen2-VL-72B | 2024-09 | 72B | ViT ~675M | 32k | Qwen-Lizenz (tongyi-qianwen) |
| Qwen2.5-VL-3B | 2025-01 | 3B | ViT (Window Attn) | 32k+ | Apache-2.0 (Research/Non-commercial für 3B laut Qwen-Lizenz — bitte Model Card prüfen) |
| Qwen2.5-VL-7B | 2025-01 | 7B | ViT (Window Attn) | 32k+ | Apache-2.0 |
| Qwen2.5-VL-72B | 2025-01 | 72B | ViT (Window Attn) | 32k+ (Video >1h) | Qwen-Lizenz |

- **Trainingsdaten (Qwen2-VL):** 1,4 Billionen Tokens Pre-Training (600 Mrd. in Stage 1, 800 Mrd. in Stage 2). Trainingstokens umfassen visuelle und Text-Tokens, Supervision nur auf Text-Tokens. Training auf Alibaba Cloud PAI-Lingjun.
- **Vision-Encoder-Parameter:** ~675M konstant über alle Qwen2-VL-Größen.
- *Hinweis:* Die genaue Lizenz-Zuordnung der Qwen2.5-VL-3B-Variante bitte in der HuggingFace Model Card verifizieren; die 7B-Varianten sind Apache-2.0, die 72B-Varianten stehen unter der Qwen-Lizenz.

## Benchmarks (Qwen2-VL, aus dem Technical Report)

| Benchmark | 72B | 7B | 2B | Referenz (GPT-4o / Claude 3.5 Sonnet) |
|---|---|---|---|---|
| DocVQA (test) | 96.5 | 94.5 | 90.1 | 92.8 / 95.2 |
| MMMU (val) | 64.5 | 54.1 | 41.1 | 69.1 / 68.3 |
| MathVista (testmini) | 70.5 | 58.2 | 43.0 | 63.8 / 67.7 |

Qwen2.5-VL: Laut Qwen-Blog übertrifft das 7B-Instruct-Modell GPT-4o-mini auf mehreren Aufgaben, und das 3B-Modell übertrifft das ältere Qwen2-VL-7B. Qwen2.5-VL-72B ist konkurrenzfähig in Dokumentenverständnis, Mathematik, College-Level-Aufgaben, Video und visuellen Agenten. *Die exakten Qwen2.5-VL-Einzelwerte stehen in den Report-Tabellen (Bilder), wurden hier nicht zahlenweise verifiziert.*

## Architektur-Besonderheiten
- Konstanter ~675M-Vision-Encoder über alle Größen → planbare visuelle Rechenkosten.
- M-RoPE als einheitliche 3D-Positionskodierung für Text/Bild/Video.
- Naive Dynamic Resolution + 2×2-Patch-Merging: native Auflösung bei kontrollierter Tokenzahl.
- Qwen2.5-VL: Window Attention im ViT, absolute Zeitkodierung für Stunden-Video, GUI-/Agenten-Fähigkeiten (Mobile/Computer Use), Objekt-Grounding in echten Koordinaten.

## Quellen
1. Qwen2-VL Technical Report (Architektur, Größen, 1.4T Tokens, 32k Kontext, Benchmarks) — https://arxiv.org/abs/2409.12191 / https://arxiv.org/html/2409.12191v1
2. Qwen2.5-VL Blog (Window Attention, RMSNorm/SwiGLU, absolute Zeitkodierung, Größen, Release 2025-01-26) — https://qwenlm.github.io/blog/qwen2.5-vl/
3. Qwen2-VL GitHub — https://github.com/QwenLM/Qwen2-VL
4. DebuggerCafe — Qwen2.5-VL Deep Dive (drei Komponenten, MLP-Merger) — https://debuggercafe.com/qwen2-5-vl/
