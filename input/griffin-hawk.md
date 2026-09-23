---
title: Griffin & Hawk (RG-LRU, Google DeepMind)
category: state-space
models_covered: [Griffin, Hawk]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2402.19427, https://arxiv.org/html/2402.19427v1, https://gonzoml.substack.com/p/griffin-mixing-gated-linear-recurrences]
---

# Griffin & Hawk (RG-LRU, Google DeepMind)

## Übersicht
Griffin und Hawk sind zwei im **Februar 2024** von **Google DeepMind** vorgestellte Architekturen (Paper: *"Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"*, arXiv:2402.19427). Sie zeigen, dass **RNN-artige Modelle mit gated linearen Rekurrenzen so effizient wie Transformer skalieren** — mit schneller Inferenz und gutem Verhalten auf langen Sequenzen. **Hawk** ist ein reines RNN, **Griffin** ein Hybrid aus Rekurrenz und lokaler Attention. Es wurden **keine offenen Gewichte** dieser Forschungs-Modelle veröffentlicht; die Architektur bildet aber die Grundlage des offenen **RecurrentGemma** (siehe `recurrentgemma.md`).

## Architektur
Kernbaustein ist die **RG-LRU (Real-Gated Linear Recurrent Unit)** — eine diagonale lineare Rekurrenz mit zwei Gates:
- **Recurrence-Gate:** `r_t = σ(W_a x_t + b_a)`
- **Input-Gate:** `i_t = σ(W_x x_t + b_x)`
- **Scale-Parameter:** `a_t = a_c^{r_t}` mit Konstante `c = 8`, wobei `a = σ(Λ)` ein diagonaler lernbarer Parameter ist.
- **State-Update:** `h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)`

Wichtig: Die Gates hängen **nur vom aktuellen Input** ab (nicht vom vorherigen Zustand `h_{t-1}`), was hoch-parallelisierbare Berechnung erlaubt. Für numerische Stabilität wird im **Log-Space** gerechnet: `log a_t = −c · softplus(Λ) ⊙ r_t`.

- **Hawk:** interleavt Gated-MLP-Blöcke mit RG-LRU-Rekurrenzblöcken (reines RNN, kein Attention).
- **Griffin:** interleavt in einer Schicht **zwei RG-LRU-Rekurrenzblöcke mit einem lokalen (Sliding-Window-) Multi-Query-Attention-Block**. Das lokale Fenster ist **fest 1024 Tokens** groß, wodurch der KV-Cache beschränkt bleibt und nicht quadratisch mit der Sequenzlänge wächst.

## Specs

| Merkmal | Wert |
|---|---|
| Organisation | Google DeepMind |
| Release | Februar 2024 (arXiv 2402.19427) |
| Lizenz | nur Paper, keine offiziell veröffentlichten Gewichte |
| Modellgrößen | 100M, 200M, 400M, 1.3B, 3B, 7B, 14B (skaliert für Experimente) |
| Trainingsdaten | MassiveText, 300B Tokens (für Downstream-Eval; skaliert auch auf mehr) |
| Kontext / Fenster | Griffin: lokales Attention-Fenster 1024; trainiert auf 2048, extrapoliert ≥4× (bis ≥8192) |
| Architektur | Hawk = RNN (RG-LRU); Griffin = RG-LRU + lokale MQA (2:1 Rekurrenz:Attention) |

## Benchmarks (bei 300B Trainings-Tokens, MMLU)
- **Hawk-3B:** MMLU 31,3% (> Mamba-3B mit 26,2% trotz Mambas 600B Tokens) [arXiv:2402.19427].
- **Hawk-7B:** MMLU 35,0%.
- **Griffin-3B:** MMLU 32,6%.
- **Griffin-7B:** MMLU 39,3% (vs. Llama-2 7B 45,3%, aber bei >6× weniger Trainings-Tokens).
- **Griffin-14B:** MMLU 49,5%.

Held-out-Vergleich (Durchschnitt über Downstream-Tasks: HellaSwag, PIQA, WinoGrande, ARC-E/C):
- **Griffin-7B:** 65,8% Ø vs. Llama-2 7B 65,3% — **besser trotz 6,6× weniger Tokens** (300B vs. 2T) [arXiv:2402.19427].
- **Griffin-14B:** 69,5% Ø vs. Llama-2 13B 69,3%.

Weitere Ergebnisse:
- **Extrapolation:** Hawk und Griffin können auf Sequenzen ≥4× ihrer Trainingslänge extrapolieren (trainiert 2048, evaluiert bis ≥8192).
- Deutlich höherer Inferenz-Durchsatz und niedrigere Latenz als Transformer-Baselines auf langen Sequenzen (fester Zustand statt wachsendem KV-Cache).

## Besonderheiten / Trivia
- "**RNNs strike back**": Das Paper war ein wichtiger Beleg, dass gated lineare Rekurrenzen mit sehr wenigen Attention-Layern Transformer-Qualität erreichen können.
- Griffin (Hybrid) schlägt in den Experimenten durchgehend das reine Hawk-RNN — ein wiederkehrendes Muster in dieser Modellklasse: **wenige Attention-Layer helfen deutlich** bei Retrieval/Copying.
- Direkter Praxis-Transfer: **RecurrentGemma** (April 2024) ist die offene, produktreife Umsetzung der Griffin-Architektur.

## Quellen
1. Griffin: Mixing Gated Linear Recurrences with Local Attention — https://arxiv.org/abs/2402.19427
2. Griffin HTML-Version (Architektur-Details, Formeln) — https://arxiv.org/html/2402.19427v1
3. GonzoML: Griffin Zusammenfassung — https://gonzoml.substack.com/p/griffin-mixing-gated-linear-recurrences
