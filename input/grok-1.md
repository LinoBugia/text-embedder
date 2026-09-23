---
title: Grok-1 (xAI)
category: moe
models_covered: [Grok-1]
last_updated: 2026-07-11
sources: [https://github.com/xai-org/grok-1, https://x.ai/news/grok-os, https://huggingface.co/xai-org/grok-1, https://artificialanalysis.ai/models/grok-1, https://www.promptingguide.ai/models/grok-1]
---

# Grok-1 (xAI)

## Übersicht

Grok-1 ist das erste große Sprachmodell von **xAI** (Elon Musks KI-Unternehmen). Am **17. März 2024** veröffentlichte xAI die **Gewichte und Architektur** von Grok-1 unter **Apache-2.0-Lizenz** — zum damaligen Zeitpunkt das **größte offen gewichtete MoE-Modell** mit 314B Parametern. Es handelt sich um das **Basismodell** (Rohgewichte, aus dem Pre-Training vom Oktober 2023), **nicht** um die feinabgestimmte/Chat-Version, die im Produkt Grok verwendet wird. Es ist nicht instruct-getuned und nicht auf ein konkretes Anwendungsszenario (z.B. Dialog) spezialisiert.

Die Veröffentlichung erfolgte über GitHub (JAX-basierter Beispielcode) und Hugging Face; wegen der Größe (314B) ist eine Multi-GPU-Maschine zum Ausführen erforderlich.

## Architektur

Grok-1 ist ein **Mixture-of-Experts (MoE)**-Decoder-Transformer mit klassischem 8-Experten-Design (ähnlich dem Mixtral-Muster):

- **Experten:** 8 Experten, **2 aktiv pro Token** (Top-2-Routing).
- **Layer:** 64 Transformer-Layer.
- **Attention:** 48 Query-Heads und 8 Key/Value-Heads (also eine GQA-artige Aufteilung).
- **Embedding-Größe:** 6.144.
- **Tokenizer:** SentencePiece mit **131.072 Tokens** Vokabular.
- **Zusatzfeatures:** unterstützt Activation-Sharding und 8-bit-Quantisierung.
- **Kontextlänge:** **8.192 Tokens**.

## Specs

| Spezifikation | Grok-1 |
|---|---|
| Org | xAI |
| Release | 17. März 2024 (Pre-Training-Stand Okt 2023) |
| Lizenz | Apache 2.0 |
| Gesamtparameter | 314B |
| Aktive Params/Token | ~78B (2 von 8 Experten) |
| Experten / aktiv | 8 / 2 (Top-2) |
| Layer | 64 |
| Attention-Heads | 48 Query / 8 KV |
| Embedding-Größe | 6.144 |
| Tokenizer | SentencePiece, 131.072 Tokens |
| Kontextlänge | 8.192 |
| Typ | Basismodell (kein Instruct/Chat-Tuning) |
| Trainingsdaten | offiziell nicht veröffentlicht |

Die aktive Parameterzahl von ~78B stammt aus Drittanalysen (Artificial Analysis); xAI nennt im offiziellen Release primär 314B gesamt, 8 Experten und 2 aktive Experten pro Token. Die restlichen Architekturzahlen (64 Layer, 48/8 Heads, 6144 Embedding, 131.072-Token-Tokenizer, 8.192 Kontext) stammen direkt aus der offiziellen GitHub-Model-Specifications-Sektion.

## Benchmarks

xAI veröffentlichte Benchmark-Zahlen primär beim früheren Grok-1-Blog-Announcement (nicht im Open-Weights-Release). Berichtete Werte des (internen, feinabgestimmten) Grok-1 zum Announcement-Zeitpunkt umfassten u.a. GSM8K, MMLU, HumanEval und MATH und lagen zwischen GPT-3.5 und GPT-4. **Wichtig:** Diese Zahlen beziehen sich auf xAIs damalige interne Evaluations-Version, nicht zwingend eins-zu-eins auf die veröffentlichten Rohgewichte des Basismodells. Für das offen veröffentlichte Basismodell hat xAI keine offizielle, verifizierte Benchmark-Tabelle im Open-Weights-Release bereitgestellt — daher hier keine konkreten Zahlen zum Open-Weight-Basismodell, um Fehlangaben zu vermeiden.

## Besonderheiten / Trivia

- Größtes offenes MoE-Gewicht zum Release (März 2024, 314B).
- Reines Basismodell — für praktische Nutzung ist eigenes Fine-Tuning nötig; kein RLHF/Instruct.
- Beispielcode ist in **JAX** geschrieben (untypisch ggü. dem PyTorch-Ökosystem der meisten offenen LLMs).
- Nachfolger (Grok-1.5, Grok-2, …) sind proprietär und nicht offen gewichtet (Stand der offenen Grok-Releases: Grok-1).

## Quellen

1. Grok-1 GitHub — xAI: https://github.com/xai-org/grok-1
2. Open Release of Grok-1 — xAI: https://x.ai/news/grok-os
3. Grok-1 — Hugging Face: https://huggingface.co/xai-org/grok-1
4. Grok-1 Intelligence, Performance & Price — Artificial Analysis: https://artificialanalysis.ai/models/grok-1
5. Grok-1 — Prompt Engineering Guide: https://www.promptingguide.ai/models/grok-1
