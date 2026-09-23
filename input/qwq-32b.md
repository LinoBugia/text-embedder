---
title: QwQ-32B
category: reasoning
models_covered: [QwQ-32B]
last_updated: 2026-07-11
sources: [https://qwenlm.github.io/blog/qwq-32b/, https://huggingface.co/Qwen/QwQ-32B, https://arxiv.org/abs/2412.15115, https://arxiv.org/abs/2309.00071, https://medium.com/data-science-in-your-pocket/qwq-32b-vs-deepseek-r1-f573cb341b83]
---

# QwQ-32B

## Übersicht
QwQ-32B ist ein reasoning-fokussiertes Modell des Alibaba-Qwen-Teams, veröffentlicht am **6. März 2025** unter der **Apache-2.0-Lizenz**. Es ist bemerkenswert, weil es mit nur **32,5 Milliarden Parametern** (dicht) eine Reasoning-Leistung erreicht, die mit dem rund 20-mal größeren DeepSeek-R1 (671B MoE) konkurriert. QwQ-32B belegt damit die These, dass gut ausgeführtes, mehrstufiges Reinforcement Learning die Reasoning-Fähigkeit eines mittelgroßen Modells massiv steigern kann, ohne extreme Parameterzahlen. Der Name "QwQ" steht für "Qwen-with-Questions" (Qwens Reasoning-Linie), und das Modell ist der Nachfolger des früheren QwQ-32B-Preview (Nov 2024).

## Architektur
QwQ-32B ist ein **Causal / Decoder-only Transformer**, aufgebaut auf dem **Qwen2.5-32B**-Basismodell. Konkrete Bausteine:
- **RoPE** (Rotary Position Embeddings) für die Positionskodierung.
- **SwiGLU**-Aktivierung in den MLP-Blöcken.
- **RMSNorm** für die Layer-Normalisierung.
- **Attention QKV-Bias** und **Grouped-Query Attention (GQA)** mit **40 Query-Heads und 8 Key/Value-Heads**.
- **64 Transformer-Layer**.

Parameterzahlen: **32,5 Mrd. gesamt**, davon **31,0 Mrd. non-embedding**. Die native Positions-Basis liegt bei **32.768 Token**; per **YaRN** (Yet another RoPE extensioN, arXiv 2309.00071) wird die volle Kontextlänge auf **131.072 Token** erweitert (YaRN-Konfiguration nötig für Prompts über 8.192 Token).

Trainingsmethodik (mehrstufiges RL ausgehend von einem Cold-Start-Checkpoint):
- **Stufe 1 (Mathe & Code):** Outcome-based RL. Statt klassischer Reward-Modelle nutzt QwQ einen **Accuracy-Verifier** (prüft mathematische Korrektheit) und einen **Code-Execution-Server** (validiert generierten Code gegen Testfälle). So skaliert das RL-Signal mit verifizierbarem Erfolg.
- **Stufe 2 (Allgemeine Fähigkeiten):** Kurze RL-Phase mit allgemeinen Reward-Modellen und regelbasierten Verifiern für Instruction-Following, Präferenz-Alignment und Agent-Fähigkeiten — ohne die Mathe/Code-Leistung zu verschlechtern.

## Specs
| Attribut | Wert |
|---|---|
| Org | Alibaba (Qwen-Team) |
| Release | 6. März 2025 |
| Lizenz | Apache 2.0 |
| Typ | Causal Language Model (dicht) |
| Basismodell | Qwen2.5-32B |
| Gesamtparameter | 32,5 Mrd. |
| Non-Embedding-Parameter | 31,0 Mrd. |
| Layer | 64 |
| Attention | GQA, 40 Q-Heads / 8 KV-Heads, QKV-Bias |
| Bausteine | RoPE, SwiGLU, RMSNorm |
| Kontextlänge | 131.072 Token (via YaRN; Basis 32.768) |
| Training | Pretraining + Post-training (SFT + mehrstufiges RL) |

### Benchmarks
QwQ-32B wurde offiziell gegen DeepSeek-R1, R1-Distill-Qwen-32B, R1-Distill-Llama-70B und o1-mini verglichen (Qwen-Blog; die exakten Zahlen liegen primär im offiziellen Benchmark-Bild). Belegte Vergleichswerte:

| Benchmark | QwQ-32B | Vergleich |
|---|---|---|
| LiveBench | 73,1 | > DeepSeek-R1 (71,6) |
| LiveCodeBench (2024.08–2025.02) | 63,4 | < DeepSeek-R1 (65,9) |
| AIME 2024 | ~79,5 (Qwen-Report) | ≈ DeepSeek-R1 (79,8) |
| MMLU-Pro (Community-Eval) | 69,07 | — |
| LEXam (Open Question / MCQ) | 44,36 / 47,83 | — |

Hinweis zur Genauigkeit: LiveBench (73,1) und LiveCodeBench (63,4) sind aus dem Vergleich mit DeepSeek-R1 belegt. Die AIME-2024-Zahl (~79,5) wird in Qwens Kommunikation genannt, ist aber im Fließtext des Blogs nicht als Zahl aufgeführt (nur im Benchmark-Bild); daher als "laut Qwen-Report" gekennzeichnet. MMLU-Pro/LEXam stammen aus Community-Evaluierungen auf der HF-Modellseite, nicht aus offiziellen Qwen-Angaben.

Benchmark-Bild (offiziell): https://huggingface.co/Qwen/QwQ-32B/resolve/main/figures/benchmark.jpg

## Besonderheiten / Trivia
- **Effizienz-Story:** Der zentrale Reiz von QwQ-32B ist, R1-nahe Reasoning-Leistung auf einem einzelnen High-End-GPU-Setup lokal ausführbar zu machen (32B dicht statt 671B MoE).
- **Verifier-basiertes RL** (Accuracy-Checker + Code-Sandbox) statt gelernter Reward-Modelle in Stufe 1 — sauberes, weniger hackbares Belohnungssignal.
- Standardmäßig generiert das Modell einen langen `<think>`-Reasoning-Block; die Transformers-Beispiele nutzen `max_new_tokens=32768`, um genug Platz zum "Denken" zu geben.
- Verfügbar auf Hugging Face, ModelScope und Qwen Chat; Apache 2.0 erlaubt uneingeschränkte kommerzielle Nutzung.

## Quellen
1. QwQ-32B: Embracing the Power of Reinforcement Learning (Qwen-Blog) — https://qwenlm.github.io/blog/qwq-32b/
2. QwQ-32B Model Card (Hugging Face) — https://huggingface.co/Qwen/QwQ-32B
3. Qwen2.5 Technical Report (arXiv 2412.15115) — https://arxiv.org/abs/2412.15115
4. YaRN: Efficient Context Window Extension (arXiv 2309.00071) — https://arxiv.org/abs/2309.00071
5. QwQ-32B vs DeepSeek-R1 Benchmark-Vergleich — https://medium.com/data-science-in-your-pocket/qwq-32b-vs-deepseek-r1-f573cb341b83
