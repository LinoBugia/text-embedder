---
title: Qwen MoE-Modelle (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B)
category: moe
models_covered: [Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B]
last_updated: 2026-07-11
sources: [https://qwenlm.github.io/blog/qwen-moe/, https://qwen.ai/blog?id=qwen-moe, https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B, https://qwenlm.github.io/blog/qwen2/]
---

# Qwen MoE-Modelle (Alibaba / Qwen-Team)

## Übersicht

Das Qwen-Team von Alibaba Cloud betreibt neben seinen dichten Modellen eine MoE-Linie unter permissiver **Apache-2.0-Lizenz**. Zwei zentrale offene MoE-Modelle:

- **Qwen1.5-MoE-A2.7B** (März 2024) — ein sehr effizientes **Klein-MoE**: nur 2,7B aktive Parameter, erreicht aber die Leistung starker 7B-Dense-Modelle (Mistral 7B, Qwen1.5-7B). Namenskonvention "A2.7B" = "activated 2.7B".
- **Qwen2-57B-A14B** (Juni 2024) — ein mittelgroßes MoE aus der Qwen2-Serie: 57,4B gesamt, 14B aktiv, Apache 2.0.

Diese Modelle zeigen Qwens Fokus darauf, mit MoE das Verhältnis aus Qualität und Inferenzkosten zu optimieren. (Alibaba betreibt darüber hinaus deutlich größere, teils proprietäre MoE-Modelle wie Qwen2.5-Max; hier fokussieren wir auf die offen gewichteten Qwen-MoE-Modelle.)

## Architektur

**Qwen1.5-MoE-A2.7B** übernimmt das DeepSeek-artige **fein-granulare + shared-Experten**-Design:
- **64 fein-granulare Experten** gesamt, entstanden durch Partitionierung eines Standard-FFN in mehrere Segmente.
- Davon **4 shared experts** (immer aktiv) + **60 routing experts**, von denen **4 pro Token aktiviert** werden.
- **Upcycling:** Das Modell wurde nicht from scratch trainiert, sondern aus dem bestehenden dichten **Qwen-1.8B** initialisiert ("upcycled"), mit zusätzlicher Zufälligkeit bei der Initialisierung zur Beschleunigung der Konvergenz.
- 2,0B aktive Non-Embedding-Parameter (~1/3 der Größe von Qwen1.5-7B).

**Qwen2-57B-A14B** ist Teil der Qwen2-Serie mit deren Standard-Stack: Decoder-only-Transformer, SwiGLU, RoPE, RMSNorm und **Grouped-Query-Attention (GQA)**. Es aktiviert 14B von 57,4B Parametern pro Token (Non-Embedding: 56,32B). Die exakte Experten-Zahl wird im Qwen2-Blog nicht explizit genannt (nicht auffindbar in der offiziellen Quelle — daher hier nicht angegeben, statt zu raten).

## Specs

| Spezifikation | Qwen1.5-MoE-A2.7B | Qwen2-57B-A14B |
|---|---|---|
| Org | Alibaba (Qwen-Team) | Alibaba (Qwen-Team) |
| Release | 28. März 2024 | 7. Juni 2024 |
| Lizenz | Apache 2.0 | Apache 2.0 |
| Gesamtparameter | 14,3B | 57,41B (Non-Emb. 56,32B) |
| Aktive Params/Token | 2,7B (2,0B non-emb.) | 14B |
| Experten-Konfig | 60 routed + 4 shared, 4 routed aktiv (64 gesamt) | fein-granular (exakte Zahl offiziell n.v.) |
| Attention | — | GQA |
| Kontextlänge | 8K | 32K pretrained (bis 64K, Instruct) |
| Trainingsdaten | upcycled aus Qwen-1.8B | Teil des Qwen2-Korpus |

## Benchmarks

**Qwen1.5-MoE-A2.7B** (offizielle Zahlen):
| Benchmark | Score |
|---|---|
| MMLU | 62,5 |
| GSM8K | 61,5 |
| HumanEval | 34,2 |
| Multilingual (aggr.) | 40,8 |
| MT-Bench (Chat) | 7,17 |

Effizienz: ~75 % geringere Trainingskosten und ~1,74x schnellere Inferenz vs. Qwen1.5-7B; auf einer NVIDIA A100-80G (vLLM, 1000/1000 Tokens) ~2,01 Requests/s bzw. 4010 TPS.[Quelle: qwenlm.github.io/blog/qwen-moe]

**Qwen2-57B-A14B — Base** (Auswahl):
| Benchmark | Score |
|---|---|
| MMLU (5-shot) | 76,5 |
| MMLU-Pro (5-shot) | 43,0 |
| GPQA (5-shot) | 34,3 |
| GSM8K (4-shot) | 80,7 |
| MATH (4-shot) | 43,0 |
| HumanEval (0-shot) | 53,0 |
| MBPP (0-shot) | 71,9 |
| BBH (3-shot) | 67,0 |
| C-Eval (5-shot) | 87,7 |
| CMMLU (5-shot) | 88,5 |

**Qwen2-57B-A14B-Instruct** (Auswahl): MMLU 75,4 · MMLU-Pro 52,8 · MT-Bench 8,55 · HumanEval 79,9 · MBPP 70,9 · GSM8K 79,6 · MATH 49,1 · LiveCodeBench 25,5.[Quelle: qwenlm.github.io/blog/qwen2]

## Besonderheiten / Trivia

- Qwen1.5-MoE ist ein Paradebeispiel für **upcycling** (Dense → MoE) statt Training from scratch.
- "A2.7B"/"A14B" in den Namen bezeichnet stets die **aktiven** Parameter.
- Qwen2-57B-A14B ist Apache 2.0, während das größere dichte Qwen2-72B unter der eigenen Qianwen-Lizenz steht.

## Quellen

1. Qwen1.5-MoE Blog (qwenlm.github.io): https://qwenlm.github.io/blog/qwen-moe/
2. Qwen1.5-MoE Blog (qwen.ai Spiegel): https://qwen.ai/blog?id=qwen-moe
3. Qwen1.5-MoE-A2.7B — Hugging Face: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
4. Qwen2 Blog: https://qwenlm.github.io/blog/qwen2/
