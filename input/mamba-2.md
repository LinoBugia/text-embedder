---
title: Mamba-2 (State Space Duality, SSD)
category: state-space
models_covered: [Mamba-2]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2405.21060, https://github.com/state-spaces/mamba, https://tridao.me/blog/2024/mamba2-part1-model/, https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems]
---

# Mamba-2 (State Space Duality, SSD)

## Übersicht
Mamba-2 ist die im **Mai 2024** von **Tri Dao und Albert Gu** vorgestellte Weiterentwicklung von Mamba, präsentiert im Paper *"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"* (arXiv:2405.21060). Die zentrale Idee ist das **State Space Duality (SSD)**-Framework, das eine theoretische Brücke zwischen SSMs und (kausaler linearer) Attention schlägt: Beide lassen sich als spezielle strukturierte Matrix-Transformationen (semiseparable Matrizen) auffassen. Praktisch macht das Mamba-2 **deutlich schneller trainierbar** (bessere Ausnutzung von Tensor Cores durch Matmul-lastige Berechnung) und erlaubt **viel größere Zustandsdimensionen** bei gleicher Effizienz.

## Architektur
Mamba-2 ersetzt den sequenziellen selektiven Scan von Mamba-1 durch den **SSD-Algorithmus**: Die SSM-Berechnung wird in **Chunks** zerlegt, innerhalb derer eine quadratische ("Attention-artige") Form genutzt wird und zwischen denen eine lineare rekurrente Form die Zustände weiterreicht. Dieser "chunkweise" Hybrid nutzt effiziente Matrixmultiplikationen und ist damit hardware-freundlicher als der reine Scan.

Weitere Änderungen ggü. Mamba-1:
- **Größerer SSM-Zustand** (`d_state` typ. 64 oder 128 statt 16) — mehr Speicherkapazität pro Layer.
- **Multi-head-artige Struktur** analog zu Multi-Head-Attention (die `A`-Matrix wird auf einen Skalar pro Head vereinfacht, was SSD ermöglicht).
- Parallel/kompatibel zu Tensor-Parallelismus, wodurch Mamba-2 sich besser für großes Training eignet.
- Es gibt auch **hybride Varianten** wie `mamba2attn-2.7b` (Mamba-2 + einige Attention-Layer), die zeigen, dass wenige Attention-Layer die In-Context-Fähigkeiten verbessern.

## Specs

| Merkmal | Wert |
|---|---|
| Organisation | Tri Dao, Albert Gu |
| Release | Mai 2024 (arXiv 2405.21060) |
| Lizenz | Apache-2.0 |
| Modellgrößen | 130M, 370M, 780M, 1.3B, 2.7B (plus `mamba2attn-2.7b`, `transformerpp-2.7b` als Baselines) |
| Kontextlänge | linear/unbeschränkt (fester Zustand) |
| Trainingsdaten | The Pile, 300B Tokens (Standard-Checkpoints) |
| Architektur | SSD (State Space Duality), chunkweise Berechnung, `d_state` 64/128, Multi-head-SSM |
| Vorteil | schnelleres Training als Mamba-1, größere Zustände, Tensor-Core-freundlich |

Checkpoints (HuggingFace `state-spaces`): `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`, `mamba2attn-2.7b`, `transformerpp-2.7b`.

## Benchmarks
- Mamba-2 erreicht **vergleichbare bis bessere Sprach-Modeling-Qualität** wie Mamba-1 bei gleicher Parameterzahl, bei **deutlich höherem Trainings-Durchsatz** (dank SSD/Matmul) [arXiv:2405.21060; tridao.me Blog].
- Größere Zustandsdimensionen verbessern insbesondere Aufgaben, die Erinnern/Copying über lange Distanzen verlangen (Multi-Query Associative Recall).

Hinweis zu Zahlen: Exakte Perplexity-/Accuracy-Tabellen und Durchsatz-Faktoren stehen im vollständigen Paper (arXiv:2405.21060) und in den Blog-Serien von Tri Dao / Princeton PLI; die hier verwendeten Extraktionsquellen bestätigen die qualitativen Aussagen und die Modell-/Konfigurationsdaten.

## Besonderheiten / Trivia
- Kernaussage des Titels: **"Transformers are SSMs"** — SSD zeigt, dass lineare Attention und selektive SSMs zwei Seiten derselben Medaille sind.
- Bildet die Basis für spätere Hybride und Nachfolger; das Repo referenziert bereits **Mamba-3** (`mamba3.py`, `d_state=128`, `headdim=64`, `is_mimo=True`, `mimo_rank=4`, `chunk_size=16`) als weitere Iteration.

## Quellen
1. Transformers are SSMs (Mamba-2, SSD) — https://arxiv.org/abs/2405.21060
2. state-spaces/mamba (GitHub) — https://github.com/state-spaces/mamba
3. Tri Dao Blog: State Space Duality (Mamba-2) Part I — https://tridao.me/blog/2024/mamba2-part1-model/
4. Princeton PLI: Mamba-2 Algorithms and Systems — https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems
