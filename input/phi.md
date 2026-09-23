---
title: Phi-3 / Phi-4 (Microsoft)
category: dense-decoder
models_covered: [Phi-3-mini 3.8B, Phi-3-small 7B, Phi-3-medium 14B, Phi-4 14B]
last_updated: 2026-07-11
sources: [https://arxiv.org/pdf/2404.14219, https://arxiv.org/html/2412.08905v1, https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/phi-4/-]
---

# Phi-3 / Phi-4 (Microsoft)

## Übersicht
Die Phi-Reihe von Microsoft Research verfolgt die These "**Datenqualität schlägt Datenmenge**": Durch stark kuratierte, "textbook-artige" und **synthetische** Trainingsdaten erreichen vergleichsweise kleine Dense-Modelle Leistungen, die sonst deutlich größere Modelle erfordern. **Phi-3-mini** (April 2024, 3,8B) war klein genug, um lokal auf einem Smartphone zu laufen, und erreichte dennoch MMLU-Werte um 69. **Phi-4** (Dezember 2024, 14B) fokussiert stark auf Reasoning und Mathematik und übertrifft in mehreren Reasoning-Benchmarks sogar deutlich größere Modelle wie Llama 3.3 70B.

## Architektur
Beide Modelle sind klassische **Dense-Decoder-only-Transformer** ohne exotische Struktur — der Fokus liegt bewusst auf Daten- und Trainingsrezept, nicht auf Architektur.

- **Phi-3-mini**: Decoder-only Transformer, Standard-Kontext 4K, per **LongRoPE** auf **128K** erweiterbar (128K-Variante). Nutzt denselben Tokenizer wie Llama 2 (Vokabular 32K), um Kompatibilität mit dem Llama-Ökosystem zu wahren. Größere Geschwister: Phi-3-small (7B, Vokabular 100K) und Phi-3-medium (14B).
- **Phi-4**: 14B Decoder-only Transformer, **tiktoken-Tokenizer** mit gepolstertem Vokabular von **100.352**. Standard-Kontext im Pretraining 4K, per Midtraining-Stufe (250B Long-Context-Tokens) auf **16K** erweitert; dabei **RoPE-Basisfrequenz auf 250K** erhöht. Phi-4 nutzt **volle Attention** über den Kontext (ersetzt die Sliding-Window-Attention von Phi-3-medium).

Das entscheidende Merkmal ist die **Trainingsdaten-Zusammensetzung** von Phi-4: ~40 % synthetische Daten (per Multi-Agent-Prompting, Self-Revision und "Pivotal Token Search"), 20 % Code, 15 % gefilterte Web-Daten, 15 % Web-Rewrites, 10 % akademische/lizenzierte Quellen. Insgesamt ~10T Tokens Pretraining + 250B Midtraining.

## Specs

| Attribut | Phi-3-mini | Phi-4 |
|---|---|---|
| Organisation | Microsoft Research | Microsoft Research |
| Release | April 2024 | 12. Dezember 2024 |
| Lizenz | MIT | MIT (arXiv-Report gibt CC BY 4.0 für das Paper an; Modellgewichte MIT) |
| Parameter | 3,8B (+ small 7B, medium 14B) | 14B |
| Kontextlänge | 4K (128K-Variante via LongRoPE) | 4K → 16K (Midtraining) |
| Trainingsdaten | ~3,3T Tokens (kuratiert/synthetisch) | ~10T Pretraining + 250B Midtraining |
| Vokabular | 32K (mini, Llama-2-Tokenizer); 100K (small) | 100.352 (tiktoken) |
| Attention | Transformer, teils SWA (medium) | volle Attention |
| PosEnc | RoPE / LongRoPE | RoPE (Basis 250K nach Extension) |
| Training-HW (Phi-4) | — | 1920x H100-80G, ~21 Tage |

## Benchmarks
**Phi-4 (14B, simple-evals)** — aus dem Technical Report:

- **MMLU**: 84,8
- **GPQA**: 56,1
- **MATH**: 80,4
- **HumanEval**: 82,6 (HumanEval+ 82,8)
- **MGSM**: 80,6
- **DROP**: 75,5
- **MMLU-Pro**: 70,4
- **ArenaHard**: 75,4
- **IFEval**: 63,0
- **SimpleQA**: 3,0 (schwach bei reinem Faktenwissen — typisch für kleine Modelle)

Zum Vergleich nennt der Report **Llama-3.3-70B**: MMLU 86,3; GPQA 49,1; MATH 66,3; HumanEval 78,9 — d. h. Phi-4 (14B) übertrifft das ~5x größere Llama-3.3-70B in GPQA, MATH und HumanEval, liegt bei MMLU knapp darunter.

**Phi-3-mini (3,8B)**: MMLU ~68,8; laut Microsoft auf dem Niveau von Mixtral 8x7B und GPT-3.5 — bemerkenswert für ein Handy-taugliches Modell.

## Besonderheiten / Trivia
- Kernthese der Phi-Reihe: Modellgröße lässt sich durch Datenqualität ("textbooks") drastisch reduzieren, ohne Performance zu opfern.
- **MIT-Lizenz** — eine der freizügigsten unter allen hier behandelten Modellen.
- Phi-4 ist bei reinem Weltwissen (SimpleQA) schwach, glänzt aber bei Reasoning/Mathematik — direkte Folge des synthetik-lastigen Trainings.
- "Pivotal Token Search" (PTS) ist ein Microsoft-eigenes Verfahren zur Auswahl besonders lernrelevanter Tokens für die Datensynthese.

## Quellen
1. Phi-3 Technical Report — https://arxiv.org/pdf/2404.14219
2. Phi-4 Technical Report — https://arxiv.org/html/2412.08905v1
3. Phi-4 (NVIDIA NGC Catalog) — https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/phi-4/-
