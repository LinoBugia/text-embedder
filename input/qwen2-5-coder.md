---
title: Qwen2.5-Coder
category: code
models_covered: [Qwen2.5-Coder-0.5B, Qwen2.5-Coder-1.5B, Qwen2.5-Coder-3B, Qwen2.5-Coder-7B, Qwen2.5-Coder-14B, Qwen2.5-Coder-32B]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2409.12186, https://arxiv.org/html/2409.12186v3, https://github.com/QwenLM/Qwen2.5-Coder, https://qwenlm.github.io/blog/qwen2.5-coder-family/]
---

# Qwen2.5-Coder

## Übersicht
Qwen2.5-Coder ist die Code-Modellfamilie von **Alibaba Qwen** (Technical Report arXiv:2409.12186, v1 18. Sep 2024). Sie ist der Nachfolger von CodeQwen1.5 und basiert auf der Qwen2.5-Architektur mit fortgesetztem code-lastigem Pretraining. Herausragend ist die **breiteste Größenpalette der Klasse** (sechs Größen von 0,5B bis 32B), was von latenzsensitivem On-Device-Autocomplete bis zu high-end Code-Generierung alles abdeckt. Das 32B-Instruct-Modell war Ende 2024 das leistungsstärkste offene Code-Modell und erreichte auf mehreren Benchmarks GPT-4o-Niveau.

## Specs
| Modell | Layer | Hidden | Q-Heads | KV-Heads | Intermediate | Embedding Tying |
|---|---|---|---|---|---|---|
| 0,5B | 24 | 896 | 14 | 2 | 4.864 | Ja |
| 1,5B | 28 | 1.536 | 12 | 2 | 8.960 | Ja |
| 3B | 36 | 2.048 | 16 | 2 | 4.864 | Ja |
| 7B | 28 | 3.584 | 28 | 4 | 18.944 | Nein |
| 14B | 48 | 5.120 | 40 | 8 | 13.824 | Nein |
| 32B | 64 | 5.120 | 40 | 8 | 27.648 | Nein |

| Eigenschaft | Wert |
|---|---|
| Organisation | Alibaba Qwen |
| Release | 18. Sep 2024 (v1); 32B im Nov 2024 |
| Kontext | 8K native → 32K (repo-level) → 128K (131.072) via YaRN |
| Trainings-Tokens | 5,5T total (5,2T in finaler Mixed-Stage); 70 % Code / 20 % Text / 10 % Math |
| Code-Sprachen | 92 (GitHub-Repos vor Feb 2024) |
| Vokabular | 151.646 (inkl. FIM-Tokens) |
| Lizenz | Apache-2.0 für die meisten Größen; 3B und 32B unter abweichender (Qwen-Research/Qwen-)Lizenz |

## Architektur
Decoder-only Transformer, abgeleitet von Qwen2.5:
- **Grouped-Query Attention (GQA)** mit größenspezifischen Query/KV-Head-Verhältnissen (z. B. 7B: 28 Q / 4 KV; 32B: 40 Q / 8 KV).
- **Rotary Position Embeddings (RoPE):** Basisfrequenz wird während des repo-level Pretrainings von 10.000 auf 1.000.000 erhöht, um längere Kontexte zu unterstützen.
- **Kontext-Staging:** File-level Pretraining bei 8.192 Tokens, repo-level bei 32.768, Extrapolation auf 131.072 (128K) mittels **YaRN**.
- **Fill-in-the-Middle (FIM):** Spezialtokens `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` (u. a.) im Vokabular; FIM ist zentrales Trainingsziel für Autocomplete.
- **Datenpipeline:** sorgfältiges Data-Cleaning, skalierbare synthetische Datengenerierung (via CodeQwen1.5) und balanciertes Data-Mixing; zusätzlich Text-Code-Grounding aus Common Crawl, Pull Requests, Commits, Jupyter Notebooks.

## Benchmarks (Pass@1)
| Benchmark | 7B-Instruct | 32B-Instruct |
|---|---|---|
| HumanEval | 88,4 % | **92,7 %** |
| HumanEval+ | 84,1 % | 87,2 % |
| MBPP | 83,5 % | 90,2 % |
| MBPP+ | 71,7 % | 75,1 % |
| LiveCodeBench | 18,2 % (Tab.16; Text nennt 37,6 % auf 2407–2409-Datensatz) | 31,4 % |

Laut Report erreicht Qwen2.5-Coder SOTA auf über 10 code-bezogenen Benchmarks (Generierung, Completion, Reasoning, Repair) und behält dabei allgemeine Sprach- und Math-Fähigkeiten.

## Besonderheiten / Trivia
- **Sechs Modellgrößen (0,5B–32B)** – die feinste Abstufung aller Code-Modell-Familien; ideal, um Größe an Latenz/Qualität anzupassen.
- Kleinere Größen (0,5B–3B) nutzen **Embedding Tying** (geteilte Input/Output-Embeddings) zur Parameterersparnis; ab 7B nicht mehr.
- 32B-Instruct galt Ende 2024 als bestes offenes Code-Modell und schlug in mehreren Coding-Metriken GPT-4o.
- Lizenz-Hinweis: Nicht alle Größen sind Apache-2.0 – **3B und 32B** haben abweichende Lizenzbedingungen; vor kommerzieller Nutzung die jeweilige Model-Card prüfen.

## Quellen
1. Qwen2.5-Coder Technical Report – https://arxiv.org/abs/2409.12186 (HTML v3: https://arxiv.org/html/2409.12186v3)
2. Qwen2.5-Coder GitHub – https://github.com/QwenLM/Qwen2.5-Coder
3. Qwen2.5-Coder Family Blog – https://qwenlm.github.io/blog/qwen2.5-coder-family/
