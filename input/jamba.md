---
title: Jamba (Hybrid Transformer-Mamba + MoE, AI21)
category: state-space
models_covered: [Jamba v0.1, Jamba 1.5 Mini, Jamba 1.5 Large]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2403.19887, https://www.ai21.com/blog/announcing-jamba/, https://www.ai21.com/blog/announcing-jamba-model-family, https://developer.nvidia.com/blog/jamba-1-5-llms-leverage-hybrid-architecture-to-deliver-superior-reasoning-and-long-context-handling/]
---

# Jamba (Hybrid Transformer-Mamba + MoE, AI21)

## Übersicht
Jamba ist das erste **produktionsreife** Sprachmodell, das auf einer neuartigen **Hybrid-Architektur aus Transformer- und Mamba-(SSM-)Layern mit Mixture-of-Experts (MoE)** basiert. Entwickelt von **AI21 Labs** und im **März 2024** als Basismodell (Jamba v0.1) veröffentlicht, kombiniert es die Qualität/In-Context-Fähigkeit von Attention mit dem Speicher- und Durchsatzvorteil von Mamba. Im **August 2024** folgte die Instruktions-getunte Familie **Jamba 1.5 Mini** und **Jamba 1.5 Large**. Jamba bietet ein **256K-Token-Kontextfenster** und ist besonders auf lange Kontexte und RAG ausgelegt.

## Architektur
Grundeinheit ist der **"Jamba-Block"**, der Layer im Verhältnis **1:7 (Attention : Mamba)** interleavt — d.h. auf jeweils sieben Mamba-Layer kommt ein Transformer-(Attention-)Layer. Ein Block umfasst 8 Layer. **MoE** wird auf jedem zweiten Layer angewandt, mit **16 Experten insgesamt** und **2 aktiven Experten pro Token** (top-2 Routing). Diese Kombination liefert:
- **Mamba-Layer:** liefern den Großteil der Kapazität bei linearer Komplexität und kleinem Zustand → kleiner KV-Cache, langer Kontext.
- **Attention-Layer (wenige):** liefern die für In-Context-Learning/Retrieval nötige globale Aufmerksamkeit.
- **MoE:** erhöht die Gesamt-Parameterzahl (Kapazität), während die pro Token **aktiven** Parameter niedrig bleiben (Effizienz).

Die Architektur ist so konfiguriert, dass ein Jamba-Block auf eine **einzelne 80GB-GPU (NVIDIA H100/A100)** passt; die 256K-Konfiguration erlaubt bis ~140K Kontext auf einer einzigen 80GB-GPU. Native Unterstützung für **Function Calling** und **JSON**.

## Specs

| Modell | Total Params | Aktive Params | Experten | Kontext | Lizenz | Release |
|---|---|---|---|---|---|---|
| Jamba v0.1 | 52B | 12B | 16 (top-2) | 256K | Apache-2.0 | März 2024 |
| Jamba 1.5 Mini | 52B | 12B | 16 (top-2) | 256K | Jamba Open Model License | Aug 2024 |
| Jamba 1.5 Large | 398B | 94B | 16 (top-2) | 256K | Jamba Open Model License | Aug 2024 |

Weitere Details:
- **Architektur:** Hybrid Transformer + Mamba + MoE, 1:7 Attention:Mamba, MoE auf jedem 2. Layer.
- **Kontext:** 256K Tokens (≈ 800 Seiten Text) — zum Release das längste Kontextfenster eines offenen Modells.
- **Hardware:** v0.1 / 1.5 Mini auf einer 80GB-GPU; 1.5 Large für Multi-GPU (mit Quantisierung "ExpertsInt8" auf 8×80GB lauffähig).
- **Trainingsdaten:** proprietärer Web-/Code-/Text-Mix von AI21 (genaue Token-Zahl nicht öffentlich spezifiziert).

## Benchmarks
- Jamba liefert laut AI21 **3× höheren Durchsatz auf langen Kontexten** gegenüber vergleichbaren reinen Transformern (z.B. Mixtral 8x7B) [ai21.com/blog/announcing-jamba].
- **Jamba 1.5 Large** ist laut AI21 zum Release das **stärkste offene Modell mit hybrider Architektur** und schneidet auf **Arena-Hard** über vergleichbaren offenen Modellen ab; beide 1.5-Modelle führen laut AI21 die **RULER-Long-Context-Benchmark** unter offenen Modellen an (effektive 256K-Kontextnutzung) [ai21.com/blog/announcing-jamba-model-family].
- Passt mit 256K-Kontext-Konfig auf eine einzelne 80GB-GPU (v0.1) [arXiv:2403.19887].

Hinweis zu Zahlen: Die exakten Benchmark-Scores (Arena-Hard, RULER, MMLU etc.) stehen in der AI21-Modellfamilien-Ankündigung und den Model Cards; die NVIDIA- und arXiv-Extraktionen bestätigen Architektur, Kontextlänge und Hardware-Angaben. Die Parameter-Splits (52B/12B, 398B/94B) sind die von AI21 offiziell kommunizierten Werte.

## Besonderheiten / Trivia
- **Jamba Open Model License:** Die 1.5-Modelle stehen unter einer eigenen permissiven Lizenz (nicht mehr reines Apache-2.0 wie v0.1), die kommerzielle Nutzung erlaubt.
- Jamba war das erste **Mamba-basierte Modell in Produktionsqualität** und ein wichtiger Beweis, dass Hybride SSM+Attention praxistauglich skalieren.
- "ExpertsInt8"-Quantisierung wurde von AI21 speziell für effizientes Serving von Jamba 1.5 Large entwickelt.

## Quellen
1. Jamba: A Hybrid Transformer-Mamba Language Model — https://arxiv.org/abs/2403.19887
2. AI21 Introducing Jamba — https://www.ai21.com/blog/announcing-jamba/
3. AI21 Jamba 1.5 Model Family — https://www.ai21.com/blog/announcing-jamba-model-family
4. NVIDIA: Jamba 1.5 LLMs Hybrid Architecture — https://developer.nvidia.com/blog/jamba-1-5-llms-leverage-hybrid-architecture-to-deliver-superior-reasoning-and-long-context-handling/
