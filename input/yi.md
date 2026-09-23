---
title: Yi / Yi-1.5 (01.AI)
category: dense-decoder
models_covered: [Yi-6B, Yi-9B, Yi-34B, Yi-34B-200K, Yi-1.5 6B, Yi-1.5 9B, Yi-1.5 34B]
last_updated: 2026-07-11
sources: [https://huggingface.co/01-ai/Yi-34B, https://github.com/01-ai/Yi, https://openlaboratory.com/models/yi-1_5-34b/, https://arxiv.org/abs/2403.04652]
---

# Yi / Yi-1.5 (01.AI)

## Übersicht
Yi ist die offene LLM-Familie des chinesischen Start-ups **01.AI** (gegründet von Kai-Fu Lee), erstmals veröffentlicht im November 2023. **Yi-34B** war bei Release eines der leistungsstärksten offenen Modelle seiner Größe und rangierte auf offenen Leaderboards zeitweise vor deutlich größeren Modellen; die **Yi-34B-200K**-Variante bot einen für die Zeit außergewöhnlich langen **200K-Token-Kontext**. Die Nachfolgegeneration **Yi-1.5** (Mai 2024) wurde auf mehr Tokens nachtrainiert und verbesserte insbesondere Code, Mathematik und Reasoning. Yi ist zweisprachig stark (Englisch + Chinesisch).

## Architektur
Yi verwendet bewusst eine **Llama-kompatible Dense-Decoder-only-Architektur** (RMSNorm, RoPE, SwiGLU, GQA), um vom bestehenden Llama-Software-Ökosystem (Tooling, Inferenz-Stacks) zu profitieren. 01.AI betont, dass der eigentliche Wert weniger in der Architektur als im **Data-Engineering** liegt: ein aufwendiger Daten-Cleaning- und Deduplizierungs-Pipeline für das Pretraining-Korpus (~3,1T Tokens, überwiegend EN/ZH).

Der **lange Kontext** (200K bei Yi-34B-200K) wird durch Fortsetzungstraining auf langen Sequenzen mit angepasster RoPE-Basisfrequenz erreicht. Yi nutzt einen eigenen Tokenizer mit einem für Chinesisch optimierten Vokabular (~64K).

## Specs

| Attribut | Yi-34B | Yi-34B-200K | Yi-1.5 34B |
|---|---|---|---|
| Organisation | 01.AI | 01.AI | 01.AI |
| Release | Nov 2023 | Nov 2023 | Mai 2024 |
| Lizenz | Apache 2.0 (Yi-1.5); Yi-Series-Lizenz für frühe Yi-1 (kommerziell nutzbar nach Registrierung) | wie Yi-34B | Apache 2.0 |
| Parameter | 34B (+ 6B, 9B) | 34B | 34B (+ 6B, 9B) |
| Kontextlänge | 4K | 200K | 4K (bis 16K/32K je Variante) |
| Trainingsdaten | ~3,1T Tokens (EN/ZH) | ~3,1T Tokens | zusätzliche 500B Tokens ggü. Yi-1 |
| Attention | GQA | GQA | GQA |
| Norm / PosEnc / Aktivierung | RMSNorm / RoPE / SwiGLU | RMSNorm / RoPE / SwiGLU | RMSNorm / RoPE / SwiGLU |
| Vokabular | ~64K | ~64K | ~64K |

## Benchmarks
- **Yi-34B (base)** erreichte bei Release Spitzenwerte unter offenen Modellen, u. a. MMLU ~76 — höher als Llama 2 70B — und war damit bemerkenswert parametereffizient. Getestet wurden u. a. GSM8K (8-shot), MATH (4-shot), HumanEval (0-shot), MBPP (3-shot).
- **Yi-1.5 34B** verbesserte laut 01.AI Code (HumanEval), Mathematik (GSM8K/MATH) und Instruction-Following deutlich gegenüber Yi-1 und positionierte sich in der Nähe von Llama 3 8B–70B (je nach Benchmark) sowie Qwen1.5-Modellen.

(Hinweis: Zahlen aus 01.AI-Model-Card/GitHub; Falcon-180B wurde in derselben Vergleichstabelle bei einigen Benchmarks wie QuAC/OBQA aus technischen Gründen ausgelassen. Exakte Werte je nach Eval-Setup.)

## Besonderheiten / Trivia
- Yi-34B-200K bot einen der **längsten Kontexte** (200K) unter offenen Modellen Ende 2023 — lange bevor 128K+ Standard wurde.
- Die Architektur ist absichtlich Llama-kompatibel; 01.AI kommunizierte offen, dass der Fokus auf **Datenqualität statt Architektur-Neuheit** liegt.
- Gründer Kai-Fu Lee (ehemals Google/Microsoft China) machte Yi zu einem prominenten Vertreter der chinesischen Open-Weight-Szene.
- Yi-1.5 steht unter **Apache 2.0** (voll kommerziell nutzbar).

## Quellen
1. Yi-34B Model Card — https://huggingface.co/01-ai/Yi-34B
2. Yi GitHub (01-ai/Yi) — https://github.com/01-ai/Yi
3. Yi Technical Report (arXiv) — https://arxiv.org/abs/2403.04652
4. Yi-1.5 34B (Open Laboratory) — https://openlaboratory.com/models/yi-1_5-34b/
