---
title: Skywork-o1 (und Nachfolger Skywork-OR1)
category: reasoning
models_covered: [Skywork-o1-Open-Llama-3.1-8B, Skywork-o1-Open-PRM-Qwen-2.5-1.5B, Skywork-o1-Open-PRM-Qwen-2.5-7B, Skywork-OR1-32B, Skywork-OR1-7B, Skywork-OR1-Math-7B]
last_updated: 2026-07-11
sources: [https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B, https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B, https://github.com/SkyworkAI/Skywork-OR1, https://arxiv.org/abs/2505.22312, https://github.com/SkyworkAI/skywork-o1-prm-inference]
---

# Skywork-o1 (und Nachfolger Skywork-OR1)

## Übersicht
Skywork-o1 ist eine Reasoning-Modellreihe des **Skywork-Teams bei Kunlun Inc.** (Tiangong), veröffentlicht am **27. November 2024** — eine der frühesten offenen o1-Repliken, noch vor DeepSeek-R1. Der Fokus liegt auf **"slow thinking" / o1-style Reasoning** und insbesondere auf **Process Reward Models (PRMs)**, die einzelne Reasoning-Schritte bewerten statt nur das Endergebnis. Die Reihe umfasst ein Chat-Modell (8B, Llama-basiert) und zwei separate PRM-Modelle. Im **April/Mai 2025** folgte die deutlich stärkere Nachfolgeserie **Skywork-OR1 (Open Reasoner 1)** unter Apache 2.0, die per Reinforcement Learning auf DeepSeek-R1-Distill-Basismodellen aufbaut und DeepSeek-R1 auf AIME übertrifft.

## Architektur
**Skywork-o1-Serie (Nov 2024)** — dichte Decoder-only Transformer:
- **Skywork-o1-Open-Llama-3.1-8B:** Chat-Modell mit **8 Mrd. Parametern**, feingetunt auf **Llama-3.1-8B-Instruct**, angereichert mit o1-style Slow-Thinking-Daten. BF16, Safetensors. Kontextlänge erbt vom Llama-3.1-Basis (nativ bis 128K möglich, praktisch als 8B-Chat-Modell genutzt).
- **Skywork-o1-Open-PRM-Qwen-2.5-1.5B** und **-7B:** Zwei **Process Reward Models** auf Qwen2.5-Basis, die schrittweise Reasoning-Belohnungen liefern (kein Chat, sondern Bewertungsmodelle für die Suche/Verifikation).

Dreistufige Trainingsmethodik der o1-Serie:
1. **Reflective Reasoning Training:** Ein proprietäres Multi-Agent-System erzeugt vielfältige, hochwertige Long-Thinking-Daten, gefolgt von Continued Pre-Training und SFT.
2. **RL mit PRM:** Reinforcement Learning, das das Skywork-o1-PRM nutzt, um die Qualität einzelner Reasoning-Schritte zu belohnen (proprietäre Reasoning-RL-Algorithmen).
3. **Reasoning Planning:** Erste öffentliche Implementierung des **Q\***-Online-Reasoning-Algorithmus (Tiangongs proprietäres Q*) plus modellbasierte Suche, um optimale Reasoning-Pfade zu finden.

**Skywork-OR1-Serie (April/Mai 2025)** — dichte Modelle, aufgebaut auf **DeepSeek-R1-Distill-Qwen-7B** bzw. **-32B**. Trainingskern ist **großskaliges regelbasiertes RL** über eine mehrstufige Pipeline (Entropy-Collapse wird gezielt vermieden), umgesetzt mit einem Custom-Fork des **verl**-Frameworks. RL-Datensatz: `Skywork-OR1-RL-Data`, difficulty-gefiltert relativ zu DeepSeek-R1-Distill-Qwen-Modellen. Kontextlänge erbt von der DeepSeek-R1-Distill-Basis (128K).

## Specs
### Skywork-o1-Serie (Nov 2024)
| Attribut | Wert |
|---|---|
| Org | Skywork / Kunlun Inc. (Tiangong) |
| Release | 27. November 2024 |
| Lizenz | Skywork Community License (kommerzielle Nutzung erlaubt, mit Bedingungen) |
| o1-Open-Llama-3.1-8B | 8B, Basis Llama-3.1-8B-Instruct, BF16 |
| o1-Open-PRM-Qwen-2.5-1.5B | PRM, Basis Qwen2.5-1.5B |
| o1-Open-PRM-Qwen-2.5-7B | PRM, Basis Qwen2.5-7B |
| Training | Reflective Reasoning SFT + RL mit PRM + Q*-Planning |

### Skywork-OR1-Serie (April/Mai 2025)
| Modell | Basis | Release | Lizenz |
|---|---|---|---|
| Skywork-OR1-Math-7B | DeepSeek-R1-Distill-Qwen-7B | 13. April 2025 (Preview) | Apache 2.0 |
| Skywork-OR1-7B (final) | DeepSeek-R1-Distill-Qwen-7B | 13. Mai 2025 | Apache 2.0 |
| Skywork-OR1-32B (final) | DeepSeek-R1-Distill-Qwen-32B | 13. Mai 2025 | Apache 2.0 |

### Benchmarks
**Skywork-o1-Open-Llama-3.1-8B:** Laut Model Card übertrifft es Qwen-2.5-7B-Instruct auf Mathematik- und Code-Benchmarks. Konkrete Zahlen liegen primär in den Benchmark-Bildern der Model Card (main_result_math.png / main_result_code.png); **belastbare Einzelzahlen für AIME/MATH sind im Fließtext nicht angegeben** — daher hier nicht als konkrete Werte behauptet.

**PRM-Modelle:** o1-Open-PRM-Qwen2.5-1.5B erreicht das Niveau von 8B-PRMs (RLHFlow Llama3.1-8B-PRM, OpenR Math-psa-7B); die 7B-PRM-Version erreicht/übertrifft Qwen2.5-Math-RM-72B auf den meisten Benchmarks.

**Skywork-OR1 (Metrik: Avg@K, nicht Pass@1):**
| Modell | AIME24 (Avg@32) | AIME25 (Avg@32) | LiveCodeBench (Avg@4) |
|---|---|---|---|
| Skywork-OR1-7B | 70,2 | 54,6 | 47,6 |
| Skywork-OR1-32B | 82,2 | 73,3 | 63,0 |

Skywork-OR1-32B übertrifft damit DeepSeek-R1 (AIME24 79,8 / AIME25 70,0) und Qwen3-32B (AIME24 81,4 / AIME25 72,9) auf AIME.

## Besonderheiten / Trivia
- **PRM-Fokus:** Skywork-o1 ist eine der ersten offenen Serien, die dedizierte Process Reward Models mitliefert — nützlich für Best-of-N-Sampling und suchbasierte Inferenz.
- **Q\*-Planung:** Skywork bezeichnet die o1-Serie als erste öffentliche Implementierung des Q*-Algorithmus für Online-Reasoning-Planung.
- **Zwei getrennte Serien nicht verwechseln:** "Skywork-o1" (Nov 2024, Skywork Community License, Llama/Qwen-Basis, PRM+Q*) ≠ "Skywork-OR1 / Open Reasoner 1" (April/Mai 2025, Apache 2.0, DeepSeek-R1-Distill-Basis, RL per verl). OR1 ist die stärkere, permissiver lizenzierte Nachfolgeserie mit vollständig offenen Gewichten, Trainingscode und Datensätzen.
- Avg@K (Durchschnitt über K unabhängige Versuche) misst Reasoning-Konsistenz strenger als Pass@1.

## Quellen
1. Skywork-o1-Open-Llama-3.1-8B Model Card (Hugging Face) — https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B
2. Skywork-o1-Open-PRM-Qwen-2.5-1.5B Model Card — https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
3. Skywork-OR1 GitHub (README, Benchmarks, Lizenz) — https://github.com/SkyworkAI/Skywork-OR1
4. Skywork Open Reasoner 1 Technical Report (arXiv 2505.22312) — https://arxiv.org/abs/2505.22312
5. Skywork-o1 PRM Inference Code — https://github.com/SkyworkAI/skywork-o1-prm-inference
