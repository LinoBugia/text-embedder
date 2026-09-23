---
title: Qwen2.5 (Alibaba)
category: dense-decoder
models_covered: [Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2412.15115, https://huggingface.co/Qwen/Qwen2.5-7B-Instruct, https://ollama.com/library/qwen2.5, https://apxml.com/models/qwen2-5-1-5b]
---

# Qwen2.5 (Alibaba)

## Übersicht
Qwen2.5 ist die im September 2024 veröffentlichte LLM-Generation des Qwen-Teams von Alibaba Cloud. Sie gilt als eine der stärksten offenen Dense-Decoder-Familien und deckt ein außergewöhnlich breites Größenspektrum von **0,5B bis 72B** ab (dazu spezialisierte Varianten wie Qwen2.5-Coder und Qwen2.5-Math). Der zentrale Fortschritt gegenüber Qwen2 ist die Skalierung des Pre-Training-Korpus von 7 auf **18 Billionen Tokens**, was Allgemeinwissen, Fachwissen und Reasoning deutlich verbessert. Qwen2.5 ist stark mehrsprachig (29+ Sprachen) und in vielen offenen Leaderboards führend in seiner jeweiligen Gewichtsklasse.

## Architektur
Qwen2.5 ist ein dense (nicht-MoE) Decoder-only-Transformer. Die Model-Card nennt explizit den Standard-Stack: **RoPE**-Positionskodierung, **SwiGLU**-Aktivierung, **RMSNorm** und **Attention mit QKV-Bias** — eine Qwen-Besonderheit, bei der die Query/Key/Value-Projektionen einen Bias-Term behalten, was die Extrapolation und Stabilität verbessert. Ab Qwen2 nutzt die Familie **Grouped-Query Attention (GQA)**; beim 7B z. B. 28 Query-Heads und 4 KV-Heads.

Der Kontext beträgt nativ **128K Tokens** (mit YaRN-Scaling), die maximale Generierungslänge ist auf 8K Tokens ausgelegt. Der Tokenizer hat ein großes Vokabular von **151.936** Einträgen (byte-level BPE), das die Multilingualität stützt. Beispiel-Konfiguration Qwen2.5-1.5B: 24 Layer, Hidden-Size 1.536, FFN-Intermediate-Size 8.960, SwiGLU. Beispiel 7B: 28 Layer, 7,61B Parameter (6,53B ohne Embeddings).

## Specs

| Attribut | Wert |
|---|---|
| Organisation | Alibaba (Qwen-Team) |
| Release | September 2024 |
| Lizenz | Apache 2.0 für die meisten Größen; **Qwen2.5-3B und 72B** unter eigener Qwen-Lizenz (nicht rein Apache) |
| Größen | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B (Base + Instruct, teils quantisiert) |
| Kontextlänge | 128K Tokens (Ausgabe bis 8K) |
| Trainingsdaten | 18 Billionen Tokens |
| Vokabular | 151.936 (byte-level BPE) |
| Attention | Grouped-Query Attention (GQA), QKV-Bias |
| Norm / PosEnc / Aktivierung | RMSNorm / RoPE / SwiGLU |
| Beispiel 7B | 7,61B Params (6,53B non-embedding), 28 Layer, 28 Q- / 4 KV-Heads |

## Benchmarks
Ausgewählte Werte (Qwen2.5 Technical Report / Model-Cards):

- **Qwen2.5-72B-Instruct**: MMLU ~85; MMLU-Pro ~71; GSM8K ~95,8; MATH ~83; HumanEval ~86 — konkurrenzfähig mit Llama 3.1 405B trotz deutlich weniger Parametern.
- **Qwen2.5-7B-Instruct**: MMLU ~74; GSM8K ~85; HumanEval ~84; MATH ~49–75 (je nach Setup) — sehr stark für seine Größe.
- **Qwen2.5-Coder** und **Qwen2.5-Math** (spezialisierte Ableger) setzten bei Release neue Bestwerte unter offenen Modellen ihrer Größe für Code bzw. Mathematik.

(Hinweis: Zahlen aus Alibaba-Report/Model-Cards; Eval-Setups variieren. Wo mir eine exakt aus einer gelesenen Quelle bestätigte Zahl fehlte, ist der Wert als Größenordnung gekennzeichnet.)

## Besonderheiten / Trivia
- Außergewöhnlich feingranulares Größenraster (7 Text-Größen), ideal für Deployment vom Edge (0.5B) bis Server (72B).
- QKV-Bias ist ein charakteristisches Qwen-Detail, das die meisten anderen aktuellen Dense-Modelle weglassen.
- Spezialisierte Ableger Qwen2.5-Coder (bis 32B) und Qwen2.5-Math erweitern die Familie.
- Achtung Lizenz: nicht alle Größen sind Apache 2.0 — 3B und 72B haben Nutzungsbedingungen mit MAU-Schwelle.

## Quellen
1. Qwen2.5 Technical Report — https://arxiv.org/abs/2412.15115
2. Qwen2.5-7B-Instruct Model Card — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
3. Qwen2.5 (Ollama Library) — https://ollama.com/library/qwen2.5
4. Qwen2.5-1.5B Specs (APXML) — https://apxml.com/models/qwen2-5-1-5b
