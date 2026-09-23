---
title: Mamba (Selektives State-Space-Modell)
category: state-space
models_covered: [Mamba]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2312.00752, https://github.com/state-spaces/mamba, https://huggingface.co/state-spaces, https://www.ibm.com/think/topics/mamba-model]
---

# Mamba (Selektives State-Space-Modell)

## Übersicht
Mamba ist ein von **Albert Gu (CMU) und Tri Dao (Princeton)** im **Dezember 2023** vorgestelltes Sequenzmodell, das den Transformer-Attention-Mechanismus vollständig durch ein **selektives strukturiertes State-Space-Modell (SSM)** ersetzt. Motivation: Attention skaliert quadratisch in der Sequenzlänge und braucht einen wachsenden KV-Cache; Mamba erreicht **lineare Zeitkomplexität** und einen **festen Zustand**, bietet damit ~5× höheren Inferenz-Durchsatz und lineares Scaling auf sehr lange Sequenzen. Mamba-3B übertrifft Transformer gleicher Größe und matcht Transformer doppelter Größe in Pretraining und Downstream-Evaluation; SOTA-Ergebnisse wurden über Sprache, Audio und Genomik gezeigt.

## Architektur
Ein klassisches SSM bildet eine 1-D-Eingabe über einen latenten Zustand ab: `h_t = A h_{t-1} + B x_t`, `y_t = C h_t`, wobei `A` typischerweise eine (diagonal-plus-low-rank / diagonale) strukturierte Matrix ist. Der Kern von Mamba ist der **selektive Mechanismus ("S6")**: Die Parameter `B`, `C` und der diskrete Zeitschritt `Δ` werden **input-abhängig** gemacht (Funktionen des aktuellen Tokens). Dadurch kann das Modell kontextabhängig entscheiden, welche Information in den Zustand aufgenommen, propagiert oder vergessen wird — was reine zeitinvariante SSMs (S4) nicht können.

Weil input-abhängige Parameter die effiziente Faltungsdarstellung brechen, nutzt Mamba einen **hardware-bewussten parallelen Scan-Algorithmus** (kernel fusion, Recomputation, kein Materialisieren des expandierten Zustands im HBM), der im rekurrenten Modus effizient auf GPUs läuft. Der Mamba-Block vereint das SSM mit einem Gated-MLP-artigen Zweig in einer einzigen homogenen Blockstruktur — es gibt **keine separaten Attention- oder MLP-Blöcke**. Da Position implizit über die Rekurrenz kodiert wird, braucht Mamba **keine Positional Embeddings**. Hinweis: Ein Mamba-Modell hat etwa die doppelte Layer-Zahl eines Transformers ähnlicher Größe, da zwei Mamba-Blöcke einem Transformer-Layer entsprechen.

## Specs

| Merkmal | Wert |
|---|---|
| Organisation | Albert Gu (CMU), Tri Dao (Princeton) |
| Release | Dezember 2023 (arXiv 2312.00752) |
| Lizenz | Apache-2.0 (Code & Weights); Paper CC-BY-4.0 |
| Modellgrößen | 130M, 370M, 790M, 1.4B, 2.8B |
| Kontextlänge | linear/unbeschränkt (fester Zustand); getestet bis Million-Länge |
| Trainingsdaten | The Pile, 300B Tokens (Standard-Checkpoints); `mamba-2.8b-slimpj`: SlimPajama, 600B Tokens |
| Architektur | Selektives SSM (S6), kein Attention/MLP, hardware-bewusster Scan |
| Durchsatz | ~5× höher als Standard-Transformer bei Inferenz |

Konfigurationen (Layer / Model-Dim):
- 130M: 24 Layer / 768
- 370M: 48 Layer / 1024
- 790M: 48 Layer / 1536
- 1.4B: 48 Layer / 2048
- 2.8B: 64 Layer / 2560

Checkpoints (HuggingFace `state-spaces`): `mamba-130m`, `mamba-370m`, `mamba-790m`, `mamba-1.4b`, `mamba-2.8b`, `mamba-2.8b-slimpj`.

## Benchmarks
- **Mamba-3B übertrifft Transformer gleicher Größe** und **matcht Transformer doppelter Größe** in Pretraining-Perplexity und Downstream-Zero-Shot-Tasks [arXiv:2312.00752].
- **~5× Inferenz-Durchsatz** vs. Standard-Transformer; lineares Scaling in Sequenzlänge [arXiv:2312.00752].
- SOTA über mehrere Modalitäten (Sprache, Audio, Genomik) laut Paper.
- Zum direkten Vergleich: DeepMinds Griffin-Paper berichtet Hawk-3B MMLU 31,3% > Mamba-3B 26,2% (Mamba auf 600B Tokens), was zeigt, dass reine SSMs bei wissenslastigen Benchmarks wie MMLU noch hinter gated-recurrent/hybriden Ansätzen liegen können [arXiv:2402.19427]. (Nur als Vergleichspunkt; exakte Mamba-Benchmark-Tabellen stehen im Original-PDF.)

Hinweis zu Zahlen: Die genauen Perplexity- und Zero-Shot-Accuracy-Tabellen der einzelnen Mamba-Größen stehen im vollständigen PDF (arXiv:2312.00752); die Abstract-Seite nennt nur die qualitativen Ergebnisse oben.

## Besonderheiten / Trivia
- Mamba war das erste reine SSM, das breit als ernsthafte Transformer-Alternative galt und löste eine Welle von SSM/Hybrid-Forschung aus (Jamba, Zamba etc.).
- Der Name spielt auf die Schlange an; das GitHub-Repo (`state-spaces/mamba`) hat >18.500 Sterne.
- Nachfolger: **Mamba-2** (Mai 2024, State Space Duality) und **Mamba-3** (Repo referenziert `mamba3.py`, `headdim=64`, MIMO-Modus). Siehe `mamba-2.md`.

## Quellen
1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces — https://arxiv.org/abs/2312.00752
2. state-spaces/mamba (GitHub) — https://github.com/state-spaces/mamba
3. HuggingFace state-spaces — https://huggingface.co/state-spaces
4. IBM: What is a Mamba model? — https://www.ibm.com/think/topics/mamba-model
