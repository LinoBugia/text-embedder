---
title: Code Llama
category: code
models_covered: [Code Llama, Code Llama - Python, Code Llama - Instruct]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2308.12950, https://huggingface.co/blog/codellama, https://ai.meta.com/blog/code-llama-large-language-model-coding/, https://github.com/meta-llama/codellama, https://huggingface.co/codellama/CodeLlama-70b-hf]
---

# Code Llama

## Übersicht

Code Llama ist Metas Familie offener Code-LLMs, veröffentlicht am **25. August 2023**; die 70B-Variante folgte im **Januar 2024**. Die Modelle sind als code-spezialisierte Fortsetzung von **Llama 2** initialisiert und decken drei Varianten ab:
- **Code Llama** (Foundation, allgemeine Code-Synthese/-Verständnis),
- **Code Llama - Python** (auf Python spezialisiert, +100 B Python-Token),
- **Code Llama - Instruct** (instruktionsfolgend, sicherere Deployment-Eigenschaften).

Größen: **7B, 13B, 34B, 70B**. Code Llama war eines der ersten wirklich starken offenen Code-Modelle und lange Referenzpunkt (v. a. 34B), bevor DeepSeek-Coder, StarCoder2 und Qwen2.5-Coder es 2024 überholten.

## Architektur

Autoregressiver **Decoder-only Transformer**, initialisiert aus Llama-2-Basismodellen und anschließend auf einem code-lastigen Korpus fortgesetzt. Zentrale Besonderheiten:

- **Kontext & RoPE:** Training auf 16 K-Kontextfenster; durch Skalierung der **RoPE-Basisfrequenz auf 1e6** ("long context fine-tuning") extrapoliert das Modell zuverlässig bis zu **100 000 Token** Eingabe.
- **Fill-in-the-Middle (FIM):** Unterstützt bei **7B- und 13B-Base/Instruct-Modellen** (Token `<FILL_ME>` bzw. `<PRE>/<SUF>/<MID>`). **Nicht** bei den 34B-Modellen und **nicht** bei den Python-Spezialisten.
- **Trainingsstufen:** Basis-Code-Training (500 B Token) → optionale Python-Spezialisierung (+100 B) → Long-Context-Fine-Tuning → Instruct-Tuning (Llama-2-Chat-Instruktionen + self-instruct-Datensatz aus synthetischen Programmierfragen und Unit-Tests).

## Specs

| Merkmal | Wert |
|---|---|
| Organisation | Meta AI |
| Release | 25. Aug 2023 (70B: Jan 2024) |
| Parametergrößen | 7B, 13B, 34B, 70B (dense) |
| Basis-Initialisierung | Llama 2 |
| Trainingsdaten | 500 B Code-Token (Base); 70B: 1 T Token; Python-Variante +100 B; Gesamt-Pretraining-Länge 7/13/34B ≈ 2 500–2 620 B inkl. Llama-2-Basis |
| Kontextlänge | 16 K nativ, bis 100 K via RoPE-Extrapolation (RoPE-Base 1e6) |
| FIM | Ja (nur 7B/13B Base+Instruct), nicht 34B/Python |
| Sprachen | u. a. Python, C++, Java, PHP, TypeScript/JavaScript, C#, Bash |
| Lizenz | Llama 2 Community License (kommerzielle Nutzung erlaubt) |

### Benchmarks (MultiPL-E, pass@1; HF-Blog-Tabelle)

| Modell | Python | JavaScript | Leaderboard Avg |
|---|---|---|---|
| Code Llama 7B | 29.98 | 31.80 | 24.36 |
| Code Llama 7B-Python | 40.48 | 36.34 | 23.50 |
| Code Llama 7B-Instruct | 45.65 | 33.11 | 26.45 |
| Code Llama 13B | 35.07 | 38.26 | 28.35 |
| Code Llama 13B-Python | 42.89 | 40.66 | 28.67 |
| Code Llama 13B-Instruct | 50.60 | 40.91 | 31.29 |
| Code Llama 34B | 45.11 | 41.66 | 33.89 |
| Code Llama 34B-Python | 53.29 | 44.72 | 33.87 |
| Code Llama 34B-Instruct | 50.79 | 45.85 | 35.09 |

- **70B (HumanEval, Metas Ankündigung):** Code Llama 70B-Instruct erreichte laut Meta ~**67.8 %** HumanEval pass@1; 70B-Base ~48 %, 70B-Python ~65.6 %. Diese Zahlen stammen aus der Meta-Ankündigung/Modellkarte und sind nicht Teil der oben zitierten MultiPL-E-Tabelle (Quelle 3/5).
- Besonderheit: 70B wurde auf **1 T** statt 500 B Token trainiert.

## Besonderheiten / Trivia

- Erstes großes offenes Code-Modell mit expliziter Long-Context-(100 K)-Fähigkeit über RoPE-Base-Skalierung — Muster, das viele Nachfolger übernahmen.
- Asymmetrische FIM-Unterstützung (nur kleine Modelle) ist eine häufige Fehlerquelle bei IDE-Integrationen.
- Die Python-Variante opfert FIM zugunsten reiner Autoregression und höherer Python-Genauigkeit.

## Quellen

1. Code Llama: Open Foundation Models for Code (Paper) — https://arxiv.org/abs/2308.12950
2. Code Llama: Llama 2 learns to code (HF Blog, Benchmark-Tabelle) — https://huggingface.co/blog/codellama
3. Introducing Code Llama (Meta Blog) — https://ai.meta.com/blog/code-llama-large-language-model-coding/
4. Code Llama Inference Code (GitHub) — https://github.com/meta-llama/codellama
5. CodeLlama-70b-hf Model Card — https://huggingface.co/codellama/CodeLlama-70b-hf
