---
title: Llama 3 / 3.1 / 3.2 / 3.3 (Meta)
category: dense-decoder
models_covered: [Llama 3 8B, Llama 3 70B, Llama 3.1 8B, Llama 3.1 70B, Llama 3.1 405B, Llama 3.2 1B, Llama 3.2 3B, Llama 3.3 70B]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2407.21783, https://ai.meta.com/blog/meta-llama-3-1/, https://huggingface.co/blog/llama31, https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/]
---

# Llama 3 / 3.1 / 3.2 / 3.3 (Meta)

## Übersicht
Die Llama-3-Reihe von Meta AI ist die einflussreichste offene Dense-Decoder-LLM-Familie der Jahre 2024–2025 und dient als De-facto-Referenz für offene Modelle. Llama 3 (April 2024, 8B & 70B) etablierte ein 128K-Token-Vokabular und Training auf ~15T Tokens. Llama 3.1 (Juli 2024) erweiterte den Kontext von 8K auf **128K** über alle Größen und fügte das Flaggschiff **405B** hinzu — das erste offene Modell auf GPT-4-Niveau. Llama 3.2 (September 2024) brachte kleine Edge-Modelle (1B, 3B) sowie Vision-Modelle (11B, 90B). Llama 3.3 (Dezember 2024) lieferte ein 70B-Instruct-Modell, dessen Qualität nahe an das 405B heranreicht, zu deutlich geringeren Kosten.

## Architektur
Llama 3 ist ein reiner Dense-Decoder-only-Transformer mit dem heute üblichen Standard-Stack: **Pre-Norm mit RMSNorm**, **RoPE** als Positionskodierung, **SwiGLU**-FFN und **Grouped-Query Attention (GQA)** über alle Größen (auch das 8B nutzt GQA, anders als Llama 2 7B). Der wesentliche Sprung gegenüber Llama 2 war ein neuer **Tiktoken-basierter BPE-Tokenizer mit 128.256 Einträgen** (statt 32K), der die Kompressionsrate und Multilingualität stark verbessert.

Für den 128K-Kontext in Llama 3.1 wird die **RoPE-Basisfrequenz erhöht** und ein mehrstufiges Long-Context-Training genutzt. Llama 3.1 ist bewusst als "einfache" dense Architektur gehalten — Meta verzichtete auf MoE, um Trainings- und Inferenzstabilität zu maximieren. Llama 3.2 Vision fügt einen separaten Vision-Encoder über Cross-Attention-Adapter hinzu, ohne die Text-Backbone-Gewichte zu ändern (die 1B/3B Text-Modelle wurden per Pruning + Distillation aus größeren Modellen gewonnen).

## Specs

| Attribut | Wert |
|---|---|
| Organisation | Meta AI |
| Release | Llama 3: Apr 2024 · 3.1: Jul 2024 · 3.2: Sep 2024 · 3.3: Dez 2024 |
| Lizenz | Llama 3.x Community License (permissiv, mit Namensnennung "Built with Llama"; >700 Mio. MAU brauchen Extra-Lizenz) |
| Größen | 8B, 70B, 405B (3.1); 1B, 3B Text + 11B, 90B Vision (3.2); 70B (3.3) |
| Kontextlänge | 8K (Llama 3) → 128K (3.1/3.2/3.3) |
| Trainingsdaten | >15 Billionen Tokens (mehrsprachig, Web, Code, Fachtexte) |
| Vokabular | 128.256 (Tiktoken-BPE) |
| Attention | Grouped-Query Attention (GQA) |
| Norm / PosEnc / Aktivierung | RMSNorm (Pre-Norm) / RoPE / SwiGLU |
| Instruct-Tuning | SFT + RLHF, >25 Mio. synthetische Beispiele (3.1) |
| GPU-Stunden (3.1) | 8B: 1,46M · 70B: 7,0M · 405B: 30,84M (gesamt 39,3M H100-h) |

## Benchmarks
Ausgewählte Instruct-Benchmarks (Meta / HuggingFace-Angaben):

- **Llama 3.1 405B Instruct**: MMLU ~88,6; HumanEval ~89; GSM8K ~96,8; MATH ~73,8 — auf Augenhöhe mit GPT-4o und Claude 3.5 Sonnet in vielen Aufgaben.
- **Llama 3.1 70B Instruct**: MMLU ~86; HumanEval ~80,5; GSM8K ~95,1.
- **Llama 3.1 8B Instruct**: MMLU ~69,4; HumanEval ~72,6; GSM8K ~84,5.
- **Llama 3.3 70B Instruct**: laut Meta nahe an 405B-Niveau bei Reasoning/Instruction-Following; unabhängig oft MMLU ~86, MATH deutlich verbessert ggü. 3.1 70B.

(Hinweis: Exakte Benchmark-Zahlen schwanken je nach Eval-Setup/Shot-Zahl; die HuggingFace-Llama-3.1-Seite listet nicht alle Werte direkt — obige Zahlen stammen aus Metas Model-Cards/Blog und sind als Größenordnung zu verstehen.)

Praktische VRAM-Anhaltspunkte (Inferenz, HuggingFace): 8B ~16 GB (FP16) / 4 GB (INT4); 70B ~140 GB (FP16) / 35 GB (INT4); 405B ~810 GB (FP16) / 203 GB (INT4).

## Besonderheiten / Trivia
- Erstes offenes Modell (405B), das mit den besten geschlossenen Frontier-Modellen konkurrierte.
- Llama 3.1 erlaubt ausdrücklich die Nutzung der Modell-Outputs zur Erzeugung synthetischer Trainingsdaten und Distillation in andere Modelle — ein Bruch mit der restriktiveren Llama-2-Lizenz.
- Unterstützt Tool-Calling (Brave Search, Wolfram Alpha, Python Interpreter) plus benutzerdefinierte JSON-Funktionsaufrufe.
- Die 1B/3B-Edge-Modelle (3.2) sind für On-Device-Betrieb (Handy, Laptop) optimiert.

## Quellen
1. The Llama 3 Herd of Models (Technical Report) — https://arxiv.org/abs/2407.21783
2. Introducing Llama 3.1 (Meta AI) — https://ai.meta.com/blog/meta-llama-3-1/
3. Llama 3.1 — 405B, 70B & 8B (HuggingFace Blog) — https://huggingface.co/blog/llama31
4. Llama 3.2: Edge AI and Vision (Meta AI) — https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
5. Meta releases Llama 3.1 (IBM Think) — https://www.ibm.com/think/news/meta-releases-llama-3-1-models-405b-parameter-variant
