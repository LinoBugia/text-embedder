---
title: RecurrentGemma (Google, Griffin-Architektur)
category: state-space
models_covered: [RecurrentGemma 2B, RecurrentGemma 9B]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2404.07839, https://arxiv.org/html/2404.07839v1, https://ai.google.dev/gemma/docs/recurrentgemma/model_card, https://huggingface.co/google/recurrentgemma-9b, https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/]
---

# RecurrentGemma (Google, Griffin-Architektur)

## Übersicht
RecurrentGemma ist eine im **April 2024** von **Google** (Griffin-Team von DeepMind) veröffentlichte, **offene** Modellfamilie, die die **Griffin-Architektur** produktreif umsetzt (Paper: *"RecurrentGemma: Moving Past Transformers for Efficient Open Language Models"*, arXiv:2404.07839). Sie nutzt dieselben Trainingsdaten und Rezepte wie die reguläre Gemma-Familie, ersetzt aber die globale Attention durch **lineare Rekurrenzen + lokale Attention**. Das ergibt einen **festen (konstant großen) Zustand**, geringeren Speicherbedarf und **schnellere Inferenz auf langen Sequenzen** als das gleich große Standard-Gemma — bei vergleichbarer Qualität.

## Architektur
RecurrentGemma basiert auf **Griffin**: Es mischt **gated lineare Rekurrenzen (RG-LRU)** mit **lokaler (Sliding-Window-)Attention**. Der zentrale Vorteil gegenüber einem Standard-Transformer ist der **fixed-size state**: Anders als bei globaler Attention wächst der KV-Cache nicht mit der Sequenzlänge, wodurch beliebig lange Prompts mit konstantem Speicher verarbeitet werden können und der Durchsatz bei langen Generierungen deutlich steigt. Das lokale Attention-Fenster begrenzt zusätzlich die Attention-Kosten. Ansonsten teilt es den Gemma-Tokenizer und die Gemma-Trainingspipeline.

Details:
- **Fixed-size state** durch RG-LRU-Rekurrenz → memory-effizient für lange Kontexte.
- **Lokale Sliding-Window-Attention** statt globaler Attention.
- Getunte Instruktions-Varianten (IT) verfügbar, englischsprachig.

## Specs

| Merkmal | RecurrentGemma 2B | RecurrentGemma 9B |
|---|---|---|
| Organisation | Google (DeepMind Griffin-Team) | Google |
| Release | April 2024 | 2024 (mit 9B-Release) |
| Lizenz | Gemma Terms of Use | Gemma Terms of Use |
| Kontext / Fenster | lokale Sliding-Window-Attention (Griffin, ~2K-Fenster); fester Zustand | dito |
| Trainingsdaten | wie Gemma (Web/Code/Mathe, mehrsprachig gefiltert); 2B laut Paper mit ~50% weniger Tokens vergleichbar zu Gemma-2B | wie Gemma |
| Hardware/Framework | TPUv5e, JAX + ML Pathways | TPUv5e, JAX |
| Architektur | Griffin (RG-LRU + lokale Attention) | Griffin |
| Varianten | pretrained + instruction-tuned (IT) | pretrained + IT |

Hinweis: Exakte Parameter-Zahlen jenseits der "2B"/"9B"-Bezeichnung und genaue Trainings-Token-Zahlen werden auf der Model Card nicht explizit genannt; das Paper gibt an, dass RecurrentGemma-2B trotz **~50% weniger Trainings-Tokens** vergleichbare Leistung wie Gemma-2B erreicht.

## Benchmarks
| Benchmark | RecurrentGemma 2B | RecurrentGemma 9B |
|---|---|---|
| MMLU (5-shot, top-1) | 38,4 | 60,5 |
| HellaSwag (0-shot) | 71,0 | 80,4 |

[Quelle: offizielle Google Model Card, https://ai.google.dev/gemma/docs/recurrentgemma/model_card]

- RecurrentGemma-2B zeigt **vergleichbare Leistung zu Gemma-2B**, obwohl auf **~50% weniger Tokens** trainiert [Paper arXiv:2404.07839; Ritvik Rastogi Summary].
- Hauptvorteil ggü. Standard-Gemma: **höherer Durchsatz bei langen Sequenzen** und **konstanter Speicherbedarf** durch den festen Zustand.

## Besonderheiten / Trivia
- RecurrentGemma ist die **erste offen verfügbare, produktreife Umsetzung der Griffin-Architektur** — praktischer Zugang zu einer Nicht-Transformer-Architektur für die Community.
- Titel des Papers ("Moving Past Transformers") ist programmatisch: Google positioniert es als Beleg, dass rekurrente/hybride Modelle bei Effizienz Transformer schlagen können, ohne Qualität einzubüßen.
- Lizenz: **Gemma Terms of Use** (kommerzielle Nutzung erlaubt, mit Nutzungsbeschränkungen laut Gemma-Prohibited-Use-Policy).

## Quellen
1. RecurrentGemma: Moving Past Transformers for Efficient Open Language Models — https://arxiv.org/abs/2404.07839
2. RecurrentGemma HTML (Architektur) — https://arxiv.org/html/2404.07839v1
3. Offizielle Model Card — https://ai.google.dev/gemma/docs/recurrentgemma/model_card
4. HuggingFace google/recurrentgemma-9b — https://huggingface.co/google/recurrentgemma-9b
5. Google Developers Blog: RecurrentGemma Architecture — https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/
