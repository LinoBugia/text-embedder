---
title: StarCoder2 (und StarCoder / The Stack v2)
category: code
models_covered: [StarCoder2-3B, StarCoder2-7B, StarCoder2-15B, StarCoder (v1)]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2402.19173, https://arxiv.org/html/2402.19173v1, https://huggingface.co/bigcode/starcoder2-15b, https://www.bigcode-project.org/]
---

# StarCoder2 (und The Stack v2)

## Übersicht
StarCoder2 ist die zweite Generation der offenen Code-Modelle des **BigCode**-Projekts (Kollaboration von Hugging Face, ServiceNow und – bei StarCoder2 – NVIDIA). Vorgestellt am **29. Februar 2024** (Paper "StarCoder 2 and The Stack v2: The Next Generation", arXiv:2402.19173). Das Projekt zeichnet sich durch **radikale Transparenz** aus: Trainingsdaten (The Stack v2), Trainingscode und Modelle sind offen, Datenherkunft ist über **Software Heritage persistent IDentifiers (SWHIDs)** nachvollziehbar. StarCoder (v1, Mai 2023) war der Vorgänger; StarCoder2 vergrößerte das Datenset auf ~4× und deckt **619 Programmiersprachen** ab.

## Specs
| Eigenschaft | StarCoder2-3B | StarCoder2-7B | StarCoder2-15B |
|---|---|---|---|
| Organisation | BigCode (HF/ServiceNow/NVIDIA) | — | — |
| Release | 29. Feb 2024 | — | — |
| Parameter | 3B | 7B | 15B |
| Layer | 30 | 32 | 40 |
| Hidden Dim | 3072 | 4608 | 6144 |
| Query Heads | 24 | 36 | 48 |
| KV Heads (GQA) | 2 | 4 | 4 |
| Vokabular | 49.152 | 49.152 | 49.152 |
| Kontext (base) | 4.096 | 4.096 | 4.096 |
| Kontext (long) | 16.384 | 16.384 | 16.384 |
| Trainings-Tokens | 3,1T + 200B = 3,3T | 3,5T + 200B = 3,7T | 4,1T + 200B = 4,3T |
| Lizenz | BigCode OpenRAIL-M | — | — |
| Trainingsdaten | The Stack v2 (Software Heritage), 619 Sprachen | — | — |

## Architektur
Decoder-only Transformer mit folgenden konkreten Entscheidungen (aus dem Paper):
- **Grouped-Query Attention (GQA)** statt Multi-Query Attention – 2 KV-Heads beim 3B, 4 KV-Heads bei 7B/15B – für effizientere Inferenz bei besserer Qualität als MQA.
- **Rotary Position Embeddings (RoPE):** Ersetzten die gelernten Positional Embeddings von StarCoder v1. Basisperiode θ = 1e5 (Ausnahme 15B: durch einen Config-Parsing-Bug θ=1e4 im Base-Pretraining). Für Long-Context: θ=1e6 (3B/7B) bzw. θ=1e5 (15B).
- **Sliding-Window Attention:** Fenster von 4.096 Tokens während des Long-Context-Stages (16.384 Tokens).
- **Fill-in-the-Middle (FIM):** Repo-Context file-level FIM. Repository-Beispiele werden mit 50 % Wahrscheinlichkeit FIM-Kandidaten; ausgewählte Dateien werden mit Spezialtokens (`<|endoftext|>`, `<file_sep>`) getrennt, mit 50 % FIM-Transformationsrate. **Hinweis:** StarCoder2-15B hatte einen Implementierungs-Bug, der die FIM-Rate reduzierte.
- **Zweistufiges Training:** Base-Pretraining bei 4K Kontext, dann Long-Context-Fine-Tuning auf zusätzlichen 200B Tokens bei 16K.

## Benchmarks (Pass@1, Greedy Decoding)
| Benchmark | 3B | 7B | 15B |
|---|---|---|---|
| HumanEval | 31,7 % | 35,4 % | 46,3 % |
| HumanEval+ | 27,4 % | 29,9 % | 37,8 % |
| MBPP | 57,4 % | 54,4 % | 66,2 % |
| MBPP+ | 47,4 % | 45,6 % | 53,1 % |

FIM Exact Match (Auszug, StarCoder2-3B): Java 75,0 %, JavaScript 73,0 %, Python 59,1 %.

Relative Einordnung laut Paper: StarCoder2-3B übertrifft vergleichbar große Code-LLMs und sogar StarCoderBase-15B. StarCoder2-15B übertrifft Modelle vergleichbarer Größe deutlich, erreicht/übertrifft CodeLlama-34B und übertrifft DeepSeek-Coder-33B in Mathematik, Reasoning und Low-Resource-Sprachen.

## Besonderheiten / Trivia
- **Transparenz-Vorreiter:** The Stack v2 basiert auf dem Software-Heritage-Archiv; jede Trainingsdatei ist per SWHID rückverfolgbar – einzigartig in der Klasse.
- Deckt **619 Programmiersprachen** ab, inkl. vieler Low-Resource-Sprachen.
- Der 15B-FIM-Bug und der RoPE-θ-Config-Bug sind im Paper offen dokumentiert – Beispiel für die transparente Kultur des Projekts.
- Lizenz **BigCode OpenRAIL-M** enthält Use-Case-Restriktionen (Responsible AI License), ist also nicht so uneingeschränkt wie Apache-2.0.

## Quellen
1. StarCoder 2 and The Stack v2: The Next Generation – https://arxiv.org/abs/2402.19173
2. StarCoder2 arXiv HTML (Architektur, Benchmarks) – https://arxiv.org/html/2402.19173v1
3. StarCoder2-15B HF Model Card – https://huggingface.co/bigcode/starcoder2-15b
4. BigCode Project – https://www.bigcode-project.org/
