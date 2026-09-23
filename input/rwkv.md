---
title: RWKV (Eagle v5 & Finch v6)
category: state-space
models_covered: [RWKV-5 Eagle, RWKV-6 Finch]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2404.05892, https://wiki.rwkv.com/basic/architecture.html, https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers, https://blog.rwkv.com/p/rwkv-v6-finch-14b-is-here, https://huggingface.co/RWKV]
---

# RWKV (Eagle v5 & Finch v6)

## Übersicht
RWKV ("**R**eceptance **W**eighted **K**ey **V**alue", ausgesprochen "RwaKuv") ist eine offene, RNN-basierte Architektur (Hauptautor **Bo Peng**), die die parallele Trainierbarkeit von Transformern mit der linearen Inferenz-Effizienz von RNNs vereint. Das Projekt ist von der **Linux Foundation (LF AI & Data)** anerkannt und wird community-getrieben entwickelt. Das Paper *"Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"* (arXiv:2404.05892, **April 2024**) beschreibt die Generationen **v5 (Eagle)** und **v6 (Finch)**, beide Nachfolger von RWKV-4. Eagle 7B wurde als "Transformer-schlagendes" mehrsprachiges Modell mit 1 Billion+ Trainings-Tokens über 100+ Sprachen beworben.

## Architektur
RWKV ist formal ein **RNN**, dessen Rekurrenz sich aber als paralleler Formel-Ausdruck über die gesamte Sequenz umschreiben lässt (daher trainierbar wie ein Transformer, aber Inferenz in O(1) Speicher pro Token). Jeder Block besteht aus zwei Sub-Modulen:
- **Time-Mixing:** übernimmt die Rolle der Attention. Es kombiniert einen aktuellen und einen zeitversetzten ("token-shift") Input über Receptance-, Key- und Value-Projektionen mit einem gelernten Zeit-Decay (WKV-Mechanismus).
- **Channel-Mixing:** eine positionsweise Feed-Forward-artige Mischung mit Token-Shift.

Neuerungen ggü. v4:
- **v5 (Eagle): matrix-wertige Zustände** statt vektor-wertiger — deutlich mehr Zustandskapazität und Ausdrucksstärke pro Layer (Multi-Head-artige Struktur).
- **v6 (Finch): dynamische, datenabhängige Rekurrenz** — die Decay-/Token-Shift-Parameter werden per LoRA-artiger Projektion **input-abhängig** (analog zur "Selektivität" bei Mamba), was das Modell adaptiver macht.
- **Tokenizer:** eigener schneller **RWKV-World-Tokenizer** auf Basis von Greedy-Matching, optimiert für Mehrsprachigkeit.
- **Keine Positional Embeddings** (Position steckt in der Rekurrenz); konstanter Speicher, theoretisch unbegrenzte Kontextlänge.

## Specs

| Merkmal | Wert |
|---|---|
| Organisation | RWKV / Bo Peng et al.; Linux Foundation (LF AI & Data) |
| Release | April 2024 (Paper); Eagle 7B Jan/Feb 2024, Finch 14B 2024 |
| Lizenz | Apache-2.0 |
| Eagle (v5) Größen | 0.46B, 1.5B, 3B, 7.5B (vier Modelle, 0.46B–7.5B) |
| Finch (v6) Größen | 1.6B, 3.1B (im Paper); zusätzlich Finch 7B & 14B (Blog-Releases) |
| Trainingsdaten | mehrsprachiger Korpus, 1.12T Tokens (Paper); Eagle 7B auf ~1.1T+, Ziel 2T; 100+ Sprachen |
| Kontextlänge | linear/unbeschränkt (RNN, fester Zustand) |
| Architektur | RNN mit Time-/Channel-Mixing; v5 matrix-wertige Zustände; v6 dynamische Rekurrenz |

## Benchmarks
- **Eagle 7B (v5):** als mehrsprachiges Modell auf 100+ Sprachen beworben; laut RWKV-Blog konkurrenzfähig mit bzw. teils vor Transformern ähnlicher Größe bei mehrsprachigen Benchmarks; Ziel war der direkte Vergleich mit LLaMA-2 7B nach 2T Tokens [blog.rwkv.com].
- **Finch (v6):** Finch 7B verbesserte sich **+5,38%** und Finch 14B zusätzlich **+7,14%** über alle Benchmarks relativ zu Eagle 7B [blog.rwkv.com/p/rwkv-v6-finch-14b-is-here].
- Auf **Long Range Arena (LRA)** (Sequenzen 1.000–16.000 Tokens) schneidet RWKV stark ab, laut Analyse zweitbester Wert hinter S4 über die fünf Datensätze [hunterheidenreich.com].
- Die vier Eagle- (0.46B–7.5B) und zwei Finch-Modelle (1.6B, 3.1B) erreichen laut Paper "competitive performance" über eine breite Benchmark-Palette [arXiv:2404.05892].

Hinweis zu Zahlen: Die exakten per-Task-Accuracy-Tabellen stehen im Paper-PDF und den Blog-Posts; die prozentualen Verbesserungswerte (+5,38% / +7,14%) sind Aggregat-Angaben des RWKV-Blogs, keine Einzeltask-Zahlen.

## Besonderheiten / Trivia
- RWKV ist eines der wenigen großen LLM-Projekte, das **vollständig community-/open-source-getrieben** und von der Linux Foundation anerkannt ist.
- Starke Betonung von **Mehrsprachigkeit** (100+ Sprachen) und Energie-Effizienz (RNN-Inferenz auf CPU/Edge möglich).
- Nachfolgeversionen (v7 "Goose") wurden nach dem Eagle/Finch-Paper weiterentwickelt; dieses Dokument fokussiert auf die im Paper dokumentierten v5/v6.

## Quellen
1. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence — https://arxiv.org/abs/2404.05892
2. RWKV Architecture History (Wiki) — https://wiki.rwkv.com/basic/architecture.html
3. Eagle 7B Blog — https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers
4. RWKV v6 Finch 14B Blog — https://blog.rwkv.com/p/rwkv-v6-finch-14b-is-here
5. HuggingFace RWKV — https://huggingface.co/RWKV
