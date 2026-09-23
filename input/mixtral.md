---
title: Mixtral 8x7B & 8x22B (Mistral AI)
category: moe
models_covered: [Mixtral 8x7B, Mixtral 8x7B Instruct, Mixtral 8x22B, Mixtral 8x22B Instruct]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2401.04088, https://mistral.ai/news/mixtral-of-experts/, https://mistral.ai/news/mixtral-8x22b/, https://huggingface.co/blog/mixtral, https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745]
---

# Mixtral 8x7B & 8x22B (Mistral AI)

## Übersicht

Mixtral ist Mistral AIs Familie offener **Sparse-Mixture-of-Experts (SMoE)**-Modelle unter permissiver **Apache-2.0-Lizenz**. Mixtral 8x7B (veröffentlicht am 11. Dezember 2023) war das erste breit adaptierte, hochwertige offene MoE-Modell und wurde zum De-facto-Referenzdesign der Klasse. Es schlägt oder erreicht Llama 2 70B und GPT-3.5 über die meisten Standard-Benchmarks bei **6x schnellerer Inferenz** als Llama 2 70B, weil es pro Token nur ~12,9B statt der vollen ~46,7B Parameter aktiviert.

**Mixtral 8x22B** (17. April 2024) ist die größere Variante mit 141B Gesamtparametern und 39B aktiven Parametern, optimiert für Reasoning, Mathematik und Code, mit nativem Function-Calling und 64K-Kontext.

Beide Modelle gibt es als Base- und als Instruct-Variante (SFT + DPO). Mixtral war zentral für die Popularisierung von MoE in der offenen Community und lief lange als starke Baseline für lokale Deployments.

## Architektur

Mixtral hat dieselbe Grundarchitektur wie das dichte Mistral 7B (Decoder-only-Transformer, SwiGLU, RoPE, RMSNorm, Grouped-Query-Attention), ersetzt aber in **jedem** Transformer-Layer den Feed-Forward-Block (FFN) durch **8 unabhängige Experten-FFNs**. Für jedes Token wählt an jedem Layer ein Router-Netzwerk (lineares Gating + Softmax) die **Top-2 Experten** aus; deren Ausgaben werden gewichtet mit den Router-Scores summiert. Als Experten-Funktion E_i(x) dient dieselbe SwiGLU-Architektur wie in Mistral 7B, mit K=2.

Da nur 2 von 8 Experten pro Token laufen, aber die Attention-Schichten geteilt werden, ergibt sich für 8x7B: ~46,7B Gesamtparameter (nicht 8×7B=56B, weil sich Attention/Embeddings teilen), aber nur ~12,9B aktive Parameter — also Kosten/Latenz eines ~13B-Modells. Analog bei 8x22B: 141B gesamt, 39B aktiv.

Wichtig: Der Name "8x7B" ist etwas irreführend — es sind nicht 8 komplette 7B-Modelle, sondern 8 Experten pro FFN-Layer bei geteilter Attention.

## Specs

| Spezifikation | Mixtral 8x7B | Mixtral 8x22B |
|---|---|---|
| Org | Mistral AI | Mistral AI |
| Release | 11. Dez 2023 | 17. Apr 2024 |
| Lizenz | Apache 2.0 | Apache 2.0 |
| Gesamtparameter | ~46,7B | 141B |
| Aktive Parameter/Token | ~12,9B (13B) | 39B |
| Experten / aktiv | 8 / 2 (Top-2) | 8 / 2 (Top-2) |
| Kontextlänge | 32K | 64K |
| Attention | GQA | GQA |
| Sprachen | EN, FR, IT, DE, ES | EN, FR, IT, DE, ES |
| Trainingsdaten | offiziell nicht veröffentlicht | offiziell nicht veröffentlicht |

## Benchmarks

**Mixtral 8x7B (Base):** Übertrifft oder erreicht Llama 2 70B und GPT-3.5 über nahezu alle bewerteten Benchmarks; besonders stark in Mathematik, Code-Generierung und mehrsprachigen Benchmarks im Vergleich zu Llama 2 70B.[Quelle: arXiv 2401.04088]

**Mixtral 8x7B Instruct:** MT-Bench-Score **8,30** (SFT + DPO), vergleichbar mit GPT-3.5; übertrifft laut Mistral GPT-3.5 Turbo, Claude-2.1, Gemini Pro und Llama-2-70B-Chat auf menschlichen Bewertungs-Benchmarks. Zeigt weniger Bias (BBQ) und positivere Sentiments (BOLD) als Llama 2.

**Mixtral 8x22B:** Auf Reasoning/Wissen optimiert; übertrifft Vergleichsmodelle auf MMLU, HellaSwag, WinoGrande, ARC Challenge, TriviaQA und NaturalQS. Schlägt Llama 2 70B in FR/DE/ES/IT auf HellaSwag, ARC Challenge und MMLU. In Math/Code stark auf HumanEval, MBPP, GSM8K und MATH. Instruct-Version: **90,8 % auf GSM8K (maj@8)** und **44,6 % auf MATH (maj@4)**.[Quelle: mistral.ai/news/mixtral-8x22b]

Hinweis: Mistral hat die exakten numerischen Scores für 8x7B (MMLU/GSM8K/HumanEval) primär als Grafiken im Blog/Paper veröffentlicht statt als Zahlen im Fließtext; die obigen exakten Zahlen für 8x22B (GSM8K 90,8 %, MATH 44,6 %) stammen direkt aus dem offiziellen Blog.

## Besonderheiten / Trivia

- Erstes wirklich populäres offenes SMoE — prägte das "8x"-Namensschema.
- Apache 2.0 auf Base UND Instruct — voll kommerziell nutzbar.
- 8x7B wurde initial per Torrent-Magnet-Link angekündigt (typischer Mistral-Stil) bevor der offizielle Blog erschien.
- 8x22B unterstützt natives Function-Calling und einen Constrained-Output-Modus.

## Quellen

1. Mixtral of Experts — arXiv 2401.04088: https://arxiv.org/abs/2401.04088
2. Mixtral of experts (Blog) — Mistral AI: https://mistral.ai/news/mixtral-of-experts/
3. Cheaper, Better, Faster, Stronger (Mixtral 8x22B) — Mistral AI: https://mistral.ai/news/mixtral-8x22b/
4. Welcome Mixtral — Hugging Face: https://huggingface.co/blog/mixtral
5. Papers Explained 95: Mixtral 8x7B — Medium: https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745
