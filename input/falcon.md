---
title: Falcon (TII)
category: dense-decoder
models_covered: [Falcon-7B, Falcon-40B, Falcon-180B, Falcon 2 11B, Falcon 3]
last_updated: 2026-07-11
sources: [https://huggingface.co/tiiuae/falcon-180B, https://falconllm.tii.ae/falcon-models.html, https://blog.paperspace.com/introducing-falcon/, https://sam-solutions.com/blog/falcon-llm-architecture/]
---

# Falcon (TII)

## Übersicht
Falcon ist die offene LLM-Familie des **Technology Innovation Institute (TII)** in Abu Dhabi. Bei Release im Jahr 2023 waren **Falcon-40B** und später **Falcon-180B** (September 2023) unter den stärksten offenen Modellen und führten zeitweise das HuggingFace Open LLM Leaderboard an. Ihr Alleinstellungsmerkmal war das Training auf dem hochqualitativen, deduplizierten Web-Korpus **RefinedWeb**. Spätere Generationen sind **Falcon 2** (11B, teils multimodal) und **Falcon 3** (kleinere, effizientere Größen). Der historische Beitrag der Familie: Nachweis, dass sorgfältig gefilterte reine Web-Daten (statt kuratierter Mischungen) für Spitzenleistung ausreichen.

## Architektur
Falcon (v1) ist ein **causal Decoder-only-Transformer** mit einigen von späteren Standardmodellen abweichenden Designentscheidungen, die auf Inferenzeffizienz zielten:

- **Multi-Query Attention (MQA)**: Alle Attention-Heads teilen sich **ein einziges** Key/Value-Paar (Vorläufer von GQA). Das minimiert den KV-Cache-Speicher extrem — wichtig für ein 180B-Modell.
- **Parallele Attention/MLP-Blöcke**: Attention- und Feed-Forward-Sub-Layer werden **parallel** statt sequenziell berechnet (wie bei GPT-J), was die Trainingsdurchsatzrate erhöht.
- **RoPE**-Positionskodierung; **head_dim 64** (bewusst auf FlashAttention optimiert).
- Trainiert mit ZeRO-Memory-Optimierung auf großen verteilten GPU-Clustern.

Der **Kontext war bei Falcon v1 kurz** (Sequenzlänge 2048 Tokens) — ein deutlicher Nachteil gegenüber späteren 128K-Modellen. Falcon 2/3 modernisierten die Architektur (GQA, längerer Kontext).

## Specs

| Attribut | Falcon-7B | Falcon-40B | Falcon-180B | Falcon 2 11B |
|---|---|---|---|---|
| Organisation | TII (Abu Dhabi) | TII | TII | TII |
| Release | 2023 | 2023 | Sep 2023 | Mai 2024 |
| Lizenz | Apache 2.0 | Apache 2.0 | Falcon-180B TII License (eigene, kommerzielle Nutzung eingeschränkt) | TII Falcon License 2.0 |
| Parameter | 7B | 40B | 180B | 11B |
| Kontextlänge | 2048 | 2048 | 2048 | 8192 |
| Trainingsdaten | 1.500B Tokens (RefinedWeb + Kuratiertes) | 1.000B Tokens | 3.500B Tokens (RefinedWeb + Kuratiertes) | ~5.500B Tokens |
| Attention | MQA | MQA | MQA, 80 Layer, d_model 14848, Vokabular 65024 | GQA |
| PosEnc | RoPE | RoPE | RoPE | RoPE |

## Benchmarks
- **Falcon-180B** führte bei Release das HuggingFace Open LLM Leaderboard an und übertraf Llama 2 70B sowie (in einigen Tests) frühe GPT-3.5-Niveaus; typische Angaben: MMLU ~70. TII testete Falcon-180B aus technischen Gründen nicht auf allen Benchmarks (z. B. QuAC, OBQA ausgelassen).
- **Falcon-40B** war 2023 monatelang das stärkste offene Modell auf dem Leaderboard vor Erscheinen von Llama 2.
- **Falcon 2 11B**: laut TII konkurrenzfähig mit Llama 3 8B / Gemma 7B in seiner Größenklasse; eine Vision-Language-Variante (VLM) wurde ebenfalls veröffentlicht.

(Hinweis: Falcon-Benchmarkzahlen aus TII-/HuggingFace-Angaben; die frühen Modelle sind gegenüber der 2024/25-Generation inzwischen überholt.)

## Besonderheiten / Trivia
- **RefinedWeb**: TII zeigte, dass strikt gefilterte und deduplizierte CommonCrawl-Daten (~85 % des Falcon-Korpus) für Spitzenleistung genügen — ein einflussreicher Datenbefund.
- Falcon-7B und -40B sind **Apache 2.0**; **Falcon-180B** hat eine eigene, restriktivere Lizenz (Hosting/kommerzielle Nutzung mit Auflagen).
- Multi-Query Attention war bei Falcon ein früher, aggressiver Effizienz-Schritt, der später von GQA (weniger extrem) abgelöst wurde.
- Kurzer Kontext (2048) ist die größte historische Schwäche der v1-Reihe.

## Quellen
1. Falcon-180B Model Card — https://huggingface.co/tiiuae/falcon-180B
2. Falcon Models (TII offiziell) — https://falconllm.tii.ae/falcon-models.html
3. Introducing Falcon 180B (Paperspace) — https://blog.paperspace.com/introducing-falcon/
4. Falcon LLM Architecture Deep Dive (SaM Solutions) — https://sam-solutions.com/blog/falcon-llm-architecture/
