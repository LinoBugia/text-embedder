---
title: DeepSeek MoE-Familie (DeepSeekMoE 16B, DeepSeek-V2, DeepSeek-V3)
category: moe
models_covered: [DeepSeekMoE 16B, DeepSeek-V2, DeepSeek-V3]
last_updated: 2026-07-11
sources: [https://github.com/deepseek-ai/DeepSeek-MoE, https://arxiv.org/abs/2401.06066, https://arxiv.org/abs/2405.04434, https://arxiv.org/html/2412.19437v1, https://github.com/deepseek-ai/DeepSeek-V3, https://fireworks.ai/blog/deepseek-model-architecture]
---

# DeepSeek MoE-Familie (DeepSeekMoE 16B → V2 → V3)

## Übersicht

DeepSeek (chinesisches KI-Labor, ursprünglich aus dem Quant-Fonds High-Flyer) hat die MoE-Architektur mit einer eigenen, sehr einflussreichen Designlinie geprägt: **fein-granulare Experten + isolierte geteilte Experten (shared experts)** in Kombination mit **Multi-head Latent Attention (MLA)**. Die Familie entwickelte sich in drei Hauptstufen:

- **DeepSeekMoE 16B** (Jan 2024) — Beweis-of-Concept der Architektur: erreicht Llama-2-7B-Qualität mit nur ~40 % der Rechenlast.
- **DeepSeek-V2** (Mai 2024) — 236B-Modell, führt MLA zur radikalen KV-Cache-Kompression ein; extrem günstig zu betreiben.
- **DeepSeek-V3** (Dez 2024) — 671B-Frontier-Modell mit nur 37B aktiven Parametern, auxiliary-loss-free Load Balancing, Multi-Token-Prediction und FP8-Training; trainiert für ~2,79 Mio. H800-GPU-Stunden — bemerkenswert günstig für ein Modell dieser Klasse.

Die Linie ist bekannt dafür, Frontier-nahe Qualität bei drastisch reduzierten Trainings- und Inferenzkosten zu liefern, und wurde Grundlage für die späteren Reasoning-Modelle (DeepSeek-R1 baut auf V3 auf).

## Architektur

**DeepSeekMoE-Prinzip (alle drei Modelle).** Zwei Kernideen:
1. **Fein-granulare Experten-Segmentierung:** Statt weniger großer Experten werden viele kleine Experten verwendet. Das erhöht die Zahl möglicher Experten-Kombinationen pro Token drastisch und verbessert die Spezialisierung.
2. **Shared-Expert-Isolation:** Einige Experten sind *immer* aktiv (shared) und kapseln gemeinsames/allgemeines Wissen, sodass die routed experts sich auf spezialisiertes Wissen konzentrieren und Redundanz vermeiden.

**Multi-head Latent Attention (MLA)** (ab V2): komprimiert Keys und Values in einen niedrigdimensionalen latenten Vektor (in V3 auf d_c=512 komprimiert), wodurch der KV-Cache massiv schrumpft (bei V2 um 93,3 % gegenüber DeepSeek 67B). Das ist der Haupthebel für die günstige Inferenz bei langem Kontext.

**Weitere Bausteine:** klassischer Transformer-Block mit SwiGLU, RoPE, RMSNorm.

**V3-Spezifika:**
- **Auxiliary-loss-free Load Balancing:** dynamische Bias-Terme steuern das Routing und verhindern routing collapse ohne den qualitätsmindernden Hilfsverlust.
- **Multi-Token Prediction (MTP):** sequentiell mit Tiefe D=1 — sagt zusätzlich zum nächsten auch ein weiteres zukünftiges Token voraus; verbessert das Trainingssignal und ermöglicht ~1,8x schnelleres speculative decoding.
- **FP8-Mixed-Precision-Training** mit tile-/block-weiser Gruppierung; trainiert auf 2048 NVIDIA H800 GPUs mit HAI-LLM, DualPipe-Scheduling, 16-way Pipeline-Parallelism, 64-way Expert-Parallelism.

## Specs

| Spezifikation | DeepSeekMoE 16B | DeepSeek-V2 | DeepSeek-V3 |
|---|---|---|---|
| Org | DeepSeek | DeepSeek | DeepSeek |
| Release | Jan 2024 | Mai 2024 | 27. Dez 2024 |
| Gesamtparameter | 16,4B | 236B | 671B |
| Aktive Params/Token | ~2,8B | 21B | 37B |
| Experten-Konfig | 64 routed + 2 shared, 6 routed aktiv | 160 routed + 2 shared (fein-granular, MLA) | 256 routed + 1 shared, 8 routed aktiv (MLA) |
| Layer | 28 | 60 | 61 |
| Kontextlänge | 4K | 128K | 128K (YaRN-Erweiterung) |
| Trainingsdaten | 2T Tokens (EN+ZH) | 8,1T Tokens | 14,8T Tokens (multilingual) |
| Attention | MHA | MLA | MLA |
| Lizenz (Code / Weights) | MIT / DeepSeek Model License (kommerz.) | DeepSeek Model License | MIT / DeepSeek Model License (kommerz.) |

Hinweise zur Genauigkeit: Die Experten-Detailkonfiguration von DeepSeekMoE 16B (64 routed + 2 shared, 6 aktiv, ~2,8B aktive Params) steht im PDF/HTML des Papers (arXiv 2401.06066), nicht im Abstract; die GitHub-Seite bestätigt 16,4B gesamt, ~40 % Rechenlast von Llama-2-7B. Für V3 nennt die GitHub-README nicht explizit "61 Layer / 256+1 Experten" im Fließtext, aber der Konvertierungscode referenziert `--n-experts 256`; das Technical Report bestätigt 671B/37B, MLA, 256 routed + 1 shared, 8 aktiv.

## Benchmarks

**DeepSeekMoE 16B (Base):** Erreicht vergleichbare Leistung wie das dichte DeepSeek 7B mit nur ~40,5 % der Rechenlast; übertrifft Llama 2 7B auf der Mehrheit der Benchmarks mit nur ~39,6 % der Rechenlast (Open-LLM-Leaderboard).[Quelle: github.com/deepseek-ai/DeepSeek-MoE]

**DeepSeek-V2:** Top-Tier-Leistung unter offenen Base- und Chat-Modellen trotz nur 21B aktiver Params; ggü. DeepSeek 67B: 42,5 % geringere Trainingskosten, 93,3 % kleinerer KV-Cache, 5,76x höherer max. Generierungsdurchsatz.[Quelle: arXiv 2405.04434]

**DeepSeek-V3** (offizielle Zahlen aus GitHub-README):
| Benchmark | Base | Chat |
|---|---|---|
| MMLU | 87,1 (5-shot) | 88,5 (EM) |
| MMLU-Pro | 64,4 (5-shot) | 75,9 (EM) |
| GPQA-Diamond | — | 59,1 (Pass@1) |
| MATH-500 | — | 90,2 (EM) |
| AIME 2024 | — | 39,2 (Pass@1) |
| HumanEval | 65,2 (Pass@1, 0-shot) | 82,6 (HumanEval-Mul Pass@1) |
| Codeforces | — | 51,6 (Percentile) |
| SWE-bench Verified | — | 42,0 (% resolved) |

Trainingskosten V3: gesamt **2,788 Mio. H800-GPU-Stunden** — außergewöhnlich niedrig für ein 671B-Frontier-Modell.

## Besonderheiten / Trivia

- MLA (statt GQA/MQA) ist DeepSeeks Signatur-Innovation für günstige Long-Context-Inferenz.
- V3 popularisierte auxiliary-loss-free Load Balancing und großflächiges FP8-Training.
- DeepSeek-R1 (Reasoning) baut direkt auf der V3-Basis auf.
- Sehr großzügige Lizenzierung (V3-Code MIT), Weights unter kommerziell nutzbarer DeepSeek Model License.

## Quellen

1. DeepSeek-MoE GitHub: https://github.com/deepseek-ai/DeepSeek-MoE
2. DeepSeekMoE — arXiv 2401.06066: https://arxiv.org/abs/2401.06066
3. DeepSeek-V2 — arXiv 2405.04434: https://arxiv.org/abs/2405.04434
4. DeepSeek-V3 Technical Report — arXiv 2412.19437: https://arxiv.org/html/2412.19437v1
5. DeepSeek-V3 GitHub: https://github.com/deepseek-ai/DeepSeek-V3
6. DeepSeek v3/R1 Model Architecture — Fireworks: https://fireworks.ai/blog/deepseek-model-architecture
