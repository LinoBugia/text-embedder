---
title: DeepSeek-Coder & DeepSeek-Coder-V2
category: code
models_covered: [DeepSeek-Coder-1.3B, DeepSeek-Coder-6.7B, DeepSeek-Coder-33B, DeepSeek-Coder-V2-Lite (16B), DeepSeek-Coder-V2 (236B)]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2401.14196, https://arxiv.org/html/2401.14196v1, https://arxiv.org/abs/2406.11931, https://arxiv.org/html/2406.11931v1, https://deepseekcoder.github.io/, https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct]
---

# DeepSeek-Coder & DeepSeek-Coder-V2

## Übersicht
DeepSeek-Coder ist die Code-Modellfamilie von **DeepSeek AI**. Die erste Generation (**DeepSeek-Coder**, Paper arXiv:2401.14196, 25. Jan 2024) wurde **from scratch** trainiert (nicht als Continued-Pretraining) und war 2024 das leistungsstärkste offene Code-Modell, das damals auch Closed-Source-Modelle wie Codex und GPT-3.5 übertraf. Die zweite Generation (**DeepSeek-Coder-V2**, arXiv:2406.11931, 17. Jun 2024) ist ein **Mixture-of-Experts**-Modell auf Basis von DeepSeek-V2 und war eines der ersten offenen Code-Modelle, das GPT-4-Turbo auf zentralen Coding-Benchmarks erreichte bzw. übertraf – bei 128K Kontext und 338 Programmiersprachen.

## Specs

### DeepSeek-Coder (V1)
| Eigenschaft | Wert |
|---|---|
| Organisation | DeepSeek AI |
| Release | 25. Jan 2024 |
| Größen | 1,3B, 5,7B, 6,7B, 33B |
| Kontext | 16K |
| Trainings-Tokens | 2T (from scratch), 87 % Code / 13 % NL (Englisch+Chinesisch) |
| Sprachen | 87 Programmiersprachen |
| FIM | Ja (fill-in-the-blank, repo-level Pretraining) |
| Lizenz | MIT (Code-Repo) + DeepSeek Model License (kommerzielle Nutzung erlaubt) |

### DeepSeek-Coder-V2 (MoE)
| Eigenschaft | V2-Lite | V2 |
|---|---|---|
| Gesamt-Parameter | 16B | 236B |
| Aktive Parameter | 2,4B | 21B |
| Architektur | DeepSeekMoE | DeepSeekMoE |
| Basis | DeepSeek-V2 (Zwischen-Checkpoint) | DeepSeek-V2 |
| Kontext | 128K | 128K |
| FIM | Ja (PSM, Rate 0,5) | Nein (deaktiviert) |
| Sprachen | 338 | 338 |
| Trainings-Tokens | 10,2T total (4,2T DeepSeek-V2 + 6T zusätzlich); 60 % Code / 10 % Math / 30 % NL |
| Lizenz | DeepSeek License (Paper CC-BY-4.0) | — |

## Architektur
**DeepSeek-Coder (V1):** Decoder-only Transformer, from scratch trainiert. Innovation war das **repository-level Pretraining**: Dateien eines Repos werden per Topologie/Dependency sortiert und mit Trennern konkateniert, damit cross-file-Abhängigkeiten gelernt werden. Zusätzlich ein **Fill-in-the-blank (FIM)**-Ziel zur Verbesserung von Infilling/Completion. Kontext 16K.

**DeepSeek-Coder-V2:** **Mixture-of-Experts (DeepSeekMoE)**-Architektur, weiter-pretrainiert aus einem Zwischen-Checkpoint von DeepSeek-V2. Nur ein kleiner Teil der Experten ist pro Token aktiv (2,4B von 16B bzw. 21B von 236B), was hohe Kapazität bei moderater Inferenzkosten bietet. Kontext wird per **YaRN** von 16K auf **128K** erweitert (Scale s=40, α=1, β=32). V2-Lite nutzt FIM (Prefix-Suffix-Middle, Rate 0,5); das große V2 hat FIM deaktiviert. Optimierung mit AdamW (β1=0,9, β2=0,95, weight decay 0,1); konventionelle statt exponentieller Normalisierung zur Vermeidung von Gradient-Spikes.

## Benchmarks

### DeepSeek-Coder (V1), Pass@1
- **DeepSeek-Coder-33B-Instruct:** HumanEval **79,3 %**, MBPP ~70,1 %.
- **LeetCode Contest:** 6,7B-Instruct 19,4 %, 33B-Instruct 27,8 %.
- Laut offizieller Projektseite übertrifft **DeepSeek-Coder-Base-33B** CodeLlama-34B um 7,9 % (HumanEval Python), 9,3 % (HumanEval multilingual), 10,8 % (MBPP) und 5,9 % (DS-1000).

### DeepSeek-Coder-V2, Pass@1 (Greedy) – Vergleich mit GPT-4-Turbo-0409
| Benchmark | V2-Lite-Instruct | V2-Instruct | GPT-4-Turbo-0409 |
|---|---|---|---|
| HumanEval (Python) | 81,1 % | **90,2 %** | 88,2 % |
| HumanEval (13 Sprachen avg) | 65,6 % | 75,3 % | 72,3 % |
| MBPP+ | 68,8 % | 76,2 % | 72,2 % |
| LiveCodeBench (Dez23–Jun24) | 24,3 % | 43,4 % | 45,7 % |
| USACO | 6,5 % | 12,1 % | 12,3 % |
| SWE-Bench | 0,0 % | 12,7 % | 18,3 % |
| Defects4J (Repair) | 9,2 % | 21,0 % | 24,3 % |

Weitere V2-Lite-Base-Werte: RepoBench v1.1 (Python) 38,9 %, Single-Line Infilling (FIM) Mean 86,4 % (> Codestral 22B mit 83,0 %).

## Besonderheiten / Trivia
- DeepSeek-Coder (V1) war eines der wenigen großen Code-Modelle, das **komplett from scratch** (nicht auf einem General-LLM) trainiert wurde.
- DeepSeek-Coder-V2 sprengte die Sprachabdeckung von 86 auf **338 Programmiersprachen** und erreichte GPT-4-Turbo-Niveau auf HumanEval/MBPP+ – ein Meilenstein für offene Code-Modelle.
- V2 nutzt YaRN für 128K-Kontext, ausgelegt auf repository-level Abhängigkeiten.
- Auf agentischen/harten Benchmarks (SWE-Bench, LiveCodeBench) lag V2 noch hinter GPT-4-Turbo – ehrlich im Paper ausgewiesen.

## Quellen
1. DeepSeek-Coder (V1) – https://arxiv.org/abs/2401.14196 (HTML: https://arxiv.org/html/2401.14196v1)
2. DeepSeek-Coder-V2 – https://arxiv.org/abs/2406.11931 (HTML: https://arxiv.org/html/2406.11931v1)
3. DeepSeek Coder Projektseite (Benchmark-Vergleiche) – https://deepseekcoder.github.io/
4. DeepSeek-Coder-33B-Instruct HF Model Card – https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct
