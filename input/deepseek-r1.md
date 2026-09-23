---
title: DeepSeek-R1 (inkl. destillierte Varianten)
category: reasoning
models_covered: [DeepSeek-R1, DeepSeek-R1-Zero, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B, DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Llama-70B]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2501.12948, https://huggingface.co/deepseek-ai/DeepSeek-R1, https://github.com/deepseek-ai/DeepSeek-R1, https://arxiv.org/html/2501.12948v1]
---

# DeepSeek-R1 (inkl. destillierte Varianten)

## Übersicht
DeepSeek-R1 ist das im Januar 2025 von DeepSeek-AI veröffentlichte Reasoning-Modell, das die Open-Source-Reasoning-Landschaft neu definierte. Es demonstrierte, dass fortgeschrittenes Reasoning-Verhalten (lange Gedankenketten, Selbstverifikation, "Aha-Momente") **rein durch Reinforcement Learning** aus einem Basismodell emergieren kann — der Vorläufer **DeepSeek-R1-Zero** wurde vollständig ohne supervised fine-tuning (SFT) trainiert. Das Modell erreicht Leistung auf Augenhöhe mit OpenAIs o1 bei Mathematik-, Code- und wissenschaftlichem Reasoning, ist aber unter der **permissiven MIT-Lizenz** veröffentlicht, was kommerzielle Nutzung und Weiterdestillation ausdrücklich erlaubt. Zusätzlich veröffentlichte DeepSeek sechs **destillierte Varianten** (1.5B–70B), die o1-mini-Niveau auf kleiner, lokal ausführbarer Hardware brachten.

## Architektur
DeepSeek-R1 ist ein **Mixture-of-Experts (MoE) Decoder-only Transformer**, trainiert auf Basis des **DeepSeek-V3-Base**-Modells. Es hat **671 Milliarden Gesamtparameter**, von denen pro Token nur **37 Milliarden aktiviert** werden (Sparse-MoE-Routing), und unterstützt eine **Kontextlänge von 128K Token**. Es erbt die V3-Architektur: Multi-head Latent Attention (MLA) zur KV-Cache-Kompression und feingranulares MoE-Routing mit geteilten und geroutet Experten.

Der eigentliche Kern ist die Trainingsmethodik:
- **DeepSeek-R1-Zero:** Direktes, großskaliges RL auf dem Basismodell mit **GRPO** (Group Relative Policy Optimization — eine PPO-Variante ohne separates Value/Critic-Netz, die den Vorteil aus einer Gruppe gesampelter Antworten relativ schätzt). Belohnt werden verifizierbare Ergebnisse (korrekte Mathe-Antwort in `\boxed{}`, bestandene Code-Tests) plus Format-Belohnungen. R1-Zero entwickelte starkes Reasoning, litt aber unter schlechter Lesbarkeit, Wiederholungen und Sprachmischung.
- **DeepSeek-R1 (voll):** Mehrstufige Pipeline: (1) kleine **Cold-Start-SFT** mit kuratierten langen CoT-Daten für Lesbarkeit/Sprache, (2) reasoning-orientiertes RL, (3) Rejection-Sampling + erneute SFT für breitere Fähigkeiten, (4) finales RL zur Alignment mit menschlichen Präferenzen.
- **Distillation:** R1-generierte Reasoning-Traces wurden per reiner SFT (kein RL) in Qwen2.5- und Llama-3.x-Basismodelle transferiert.

Inferenzempfehlungen laut Model Card: Temperatur 0,5–0,7 (0,6 empfohlen), **kein System-Prompt** (alle Instruktionen in den User-Prompt), Antwort mit `<think>\n` erzwingen, um die Reasoning-Kette nicht zu überspringen.

## Specs (DeepSeek-R1, Hauptmodell)
| Attribut | Wert |
|---|---|
| Org | DeepSeek-AI |
| Release | Januar 2025 |
| Lizenz | MIT |
| Architektur | MoE Decoder-only Transformer (auf DeepSeek-V3-Base) |
| Gesamtparameter | 671 Mrd. |
| Aktive Parameter | 37 Mrd. / Token |
| Kontextlänge | 128K Token |
| Paper | arXiv 2501.12948 |

### Benchmarks — DeepSeek-R1 (Pass@1, aus Model Card)
| Benchmark | Wert |
|---|---|
| AIME 2024 | 79,8 % |
| MATH-500 | 97,3 % |
| CNMO 2024 | 78,8 % |
| GPQA-Diamond | 71,5 % |
| LiveCodeBench (COT) | 65,9 % |
| Codeforces Rating / Perzentil | 2029 / 96,3 % |
| SWE-bench Verified | 49,2 % |
| MMLU | 90,8 % |
| MMLU-Pro | 84,0 % |
| IF-Eval (Prompt Strict) | 83,3 % |
| AlpacaEval 2.0 (LC-winrate) | 87,6 % |
| ArenaHard | 92,3 % |
| SimpleQA | 30,1 % |

### Destillierte Varianten & Benchmarks
Basismodelle und Leistung (aus Model Card):

| Distilled Modell | Basis | AIME24 Pass@1 | AIME24 cons@64 | MATH-500 | GPQA-Diamond | LiveCodeBench | CF Rating |
|---|---|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28,9 % | 52,7 % | 83,9 % | 33,8 % | 16,9 % | 954 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55,5 % | 83,3 % | 92,8 % | 49,1 % | 37,6 % | 1189 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 50,4 % | 80,0 % | 89,1 % | 49,0 % | 39,6 % | 1205 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69,7 % | 80,0 % | 93,9 % | 59,1 % | 53,1 % | 1481 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 72,6 % | 83,3 % | 94,3 % | 62,1 % | 57,2 % | 1691 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 70,0 % | 86,7 % | 94,5 % | 65,2 % | 57,5 % | 1633 |

Die destillierten Modelle erben die MIT-Lizenz von DeepSeek, unterliegen aber zusätzlich der jeweiligen Basis-Lizenz (Qwen-Lizenz bzw. Meta Llama-Lizenz).

## Besonderheiten / Trivia
- **R1-Zero als Beweis:** Erster großer öffentlicher Beleg, dass Reasoning ohne SFT rein durch RL emergieren kann — der "Aha-Moment", bei dem das Modell lernt, innezuhalten und neu zu bewerten, wurde als emergent beschrieben.
- **Kostendisruption:** Die Veröffentlichung löste breite Diskussion über Trainingskosten aus, da DeepSeek deutlich günstigeres Training als westliche Frontier-Labs meldete.
- **GRPO** wurde durch R1 zur populären RL-Methode für Reasoning-Modelle und ist die Basis vieler Nachfolger (u. a. Skywork-OR1).
- Kontextlänge auf manchen Hosting-Plattformen (z. B. Azure) auf 40K begrenzt, obwohl das Originalmodell 128K unterstützt.

## Quellen
1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv 2501.12948) — https://arxiv.org/abs/2501.12948
2. DeepSeek-R1 Model Card (Hugging Face) — https://huggingface.co/deepseek-ai/DeepSeek-R1
3. DeepSeek-R1 GitHub (LICENSE, README) — https://github.com/deepseek-ai/DeepSeek-R1
4. DeepSeek-R1 Paper HTML — https://arxiv.org/html/2501.12948v1
