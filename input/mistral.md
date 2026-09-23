---
title: Mistral (7B, Nemo, Small, Large 2)
category: dense-decoder
models_covered: [Mistral 7B, Mistral NeMo 12B, Mistral Small, Mistral Large 2]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2310.06825, https://huggingface.co/mistralai/Mistral-7B-v0.1, https://encord.com/blog/mistral-large-explained/]
---

# Mistral (7B, NeMo, Small, Large 2)

## Übersicht
Mistral AI (Paris, 2023 gegründet) setzte mit **Mistral 7B** (September 2023) einen Meilenstein für effiziente offene Dense-Decoder-Modelle: Das 7B-Modell übertraf Llama 2 13B in allen getesteten Benchmarks und näherte sich teils Llama 34B — bei rund halber Größe. Die Familie umfasst heute u. a. **Mistral NeMo** (12B, gemeinsam mit NVIDIA, 128K Kontext), **Mistral Small** und das Flaggschiff **Mistral Large 2** (123B, Juli 2024). Mistrals Markenzeichen sind architektonische Effizienztricks (Sliding Window Attention) und ein Fokus auf mehrsprachige (EU-Sprachen) und Code-Fähigkeiten.

## Architektur
**Mistral 7B** ist ein Dense-Decoder-only-Transformer mit dem Standard-Stack (RMSNorm, RoPE, SwiGLU), erweitert um zwei zentrale Effizienzmechanismen:

- **Grouped-Query Attention (GQA)**: reduziert KV-Cache und beschleunigt Inferenz.
- **Sliding Window Attention (SWA)**: Jedes Token beachtet nur ein Fenster der letzten W Tokens (bei Mistral 7B W=4096). Durch die Schichtung über mehrere Layer wächst der effektive Rezeptionsbereich linear mit der Tiefe (theoretisch bis ~131K), ohne die quadratischen Kosten voller Attention. Kombiniert mit einem **Rolling-Buffer-KV-Cache** (fester Cache-Speicher, der zyklisch überschrieben wird) ermöglicht das effiziente Verarbeitung langer Sequenzen.
- **Byte-fallback-BPE-Tokenizer**: garantiert, dass jedes Byte kodierbar ist (keine "unknown"-Tokens).

Spätere Modelle (NeMo, Large 2) nutzen längere native Kontexte (128K) und größere Vokabulare; **Mistral NeMo** verwendet den **Tekken-Tokenizer** (auf tiktoken basierend), der ~30 % effizienter komprimiert. Mistral Large 2 ist ein dense 123B-Modell mit 128K Kontext, ausgelegt auf Reasoning, Code und Mehrsprachigkeit.

## Specs

| Attribut | Mistral 7B | Mistral NeMo 12B | Mistral Large 2 |
|---|---|---|---|
| Organisation | Mistral AI | Mistral AI + NVIDIA | Mistral AI |
| Release | Sep 2023 | Jul 2024 | Jul 2024 |
| Lizenz | Apache 2.0 | Apache 2.0 | Mistral Research License (nicht-kommerziell frei; kommerziell lizenzpflichtig) |
| Parameter | 7,3B | 12B | 123B |
| Kontextlänge | 32K (SWA-Fenster 4096) | 128K | 128K |
| Attention | GQA + Sliding Window Attention | GQA | GQA |
| Norm / PosEnc / Aktivierung | RMSNorm / RoPE / SwiGLU | RMSNorm / RoPE / SwiGLU | RMSNorm / RoPE / SwiGLU |
| Tokenizer | Byte-fallback BPE | Tekken (tiktoken-basiert) | — |

## Benchmarks
- **Mistral 7B** übertraf laut Paper Llama 2 13B in allen getesteten Kategorien (Reasoning, Mathematik, Code-Generierung) und näherte sich CodeLlama 7B bei Code, ohne allgemeine Fähigkeiten zu verlieren. Typische Werte: MMLU ~60, HumanEval ~30, GSM8K ~52 (base; Instruct-Varianten höher).
- **Mistral Large 2 (123B)**: laut Mistral MMLU ~84; sehr stark bei Code (nahe an führenden Modellen) und Multilingualität (EN, FR, DE, ES, IT + weitere). 128K Kontext für präzises Retrieval aus langen Dokumenten.

(Hinweis: Benchmark-Zahlen aus Paper/Herstellerangaben; Instruct-vs-Base und Shot-Setup beachten.)

## Besonderheiten / Trivia
- Mistral 7B popularisierte Sliding Window Attention als praktikables Effizienzmittel in offenen Modellen.
- Mistral 7B und NeMo stehen unter **Apache 2.0** (voll kommerziell nutzbar) — ein wichtiger Lizenzvorteil ggü. der Llama-Community-Lizenz.
- Mistral betreibt daneben auch echte MoE-Modelle (Mixtral 8x7B, 8x22B) — diese gehören jedoch zur MoE-Klasse, nicht in dieses Dense-Doku.
- Mistral Large 2 ist bewusst dense gehalten (kein MoE) für vorhersehbare Latenz.

## Quellen
1. Mistral 7B Paper — https://arxiv.org/abs/2310.06825
2. Mistral-7B-v0.1 Model Card — https://huggingface.co/mistralai/Mistral-7B-v0.1
3. Mistral Large Explained (Encord) — https://encord.com/blog/mistral-large-explained/
4. Mistral AI — https://mistral.ai/
