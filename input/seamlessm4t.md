---
title: SeamlessM4T (v1 & v2)
category: audio-speech
models_covered: [SeamlessM4T, SeamlessM4T v2]
last_updated: 2026-07-11
sources: [https://ai.meta.com/blog/seamless-m4t/, https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2, https://www.nature.com/articles/s41586-024-08359-z, https://github.com/facebookresearch/seamless_communication, https://www.emergentmind.com/topics/seamlessm4t-models, https://ai.meta.com/research/publications/seamlessm4t-massively-multilingual-multimodal-machine-translation/]
---

# SeamlessM4T (v1 & v2)

## Übersicht

**SeamlessM4T** (Massively Multilingual & Multimodal Machine Translation) ist Metas Foundation-Modell, das **fünf Aufgaben in einem einzigen Modell** vereint: automatische Spracherkennung (ASR), Speech-to-Text-Translation (S2TT), Text-to-Text-Translation (T2TT), Text-to-Speech-Synthese (TTS) und Speech-to-Speech-Translation (S2ST) — für bis zu **~100 Sprachen**. v1 erschien im **August 2023**, die verbesserte **v2** im **Dezember 2023** mit der neuen **UnitY2**-Architektur (bessere Qualität + niedrigere Latenz bei der Sprachgenerierung). Die Arbeit wurde 2024 in **Nature** publiziert ("Joint speech and text machine translation for up to 100 languages").

Das Modell ist unter **CC-BY-NC 4.0** (nur Forschung/nicht-kommerziell) verfügbar. Sprachumfang v2: **101 Sprachen** für Sprach-Input, **96 Sprachen** für Text-Input/Output, **35–36 Sprachen** für Sprach-Output. SeamlessM4T ist die Basis der breiteren "Seamless"-Familie (SeamlessExpressive für Prosodie-Erhalt, SeamlessStreaming für niedrige Latenz).

## Architektur

SeamlessM4T basiert auf dem **UnitY**- (v1) bzw. **UnitY2**- (v2) Multitask-Transformer-Framework. Der Verarbeitungspfad ist zweistufig:

1. **Speech-Encoder — w2v-BERT 2.0:** ein Conformer-basierter Encoder (kombiniert Convolution + Self-Attention), selbstüberwacht vortrainiert auf massiven unbeschrifteten Sprachdaten. Für SeamlessM4T-Large hat der Encoder **~635 M Parameter / 24 Conformer-Layer** (Medium: ~311 M / weniger Layer). Er wandelt Audio in kontextualisierte Repräsentationen.
2. **Text-Decoder (Übersetzung):** erzeugt Zieltext in der gewünschten Sprache — dies ist gemeinsame Grundlage für ASR, S2TT und T2TT.
3. **Text-to-Unit (T2U) — Unit-Decoder:** wandelt den Übersetzungstext in diskrete **akustische "Unit-Tokens"** um. In **v1** war dies ein autoregressiver Decoder; in **v2 (UnitY2)** wird er durch einen **nicht-autoregressiven (NAR)** Unit-Decoder ersetzt, der die FastSpeech2-Decoder-Architektur adaptiert und hierarchisch von Subwörtern zu Units upsampelt (mit "glancing"-Trainingsziel). Ergebnis: ~**3× schnellere** Sprachgenerierung.
4. **Vocoder:** synthetisiert aus den Unit-Tokens die finale Audio-Wellenform.

Diese Struktur (Text als Zwischen-Repräsentation + diskrete Units für Audio) erlaubt es, alle Modalitätskombinationen (Speech↔Text) mit gemeinsam genutzten Komponenten abzudecken und Kaskadierungsfehler klassischer Pipelines zu reduzieren.

**Trainingsdaten:** ~1–4,5 Mio. Stunden unbeschriftetes Sprach-Pretraining (71–143 Sprachen) sowie **SeamlessAlign** — ~470.000 Stunden automatisch geminte, gefilterte parallele Daten (speech–text, text–text, speech–speech) über ~100 Sprachen; ergänzt durch CommonVoice, VoxPopuli, GigaSpeech, CoVoST2, CVSS, MuST-C, LibriTTS u. a.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Organisation | Meta AI |
| Release | v1 Aug 2023 · v2 Dez 2023 |
| Lizenz | CC-BY-NC 4.0 (nicht-kommerziell) |
| Parameter — Large | 2,3 B |
| Parameter — Medium | 1,2 B |
| Speech-Encoder | w2v-BERT 2.0 Conformer (~635 M Large / ~311 M Medium, 24 Conformer-Layer) |
| Unit-Decoder | UnitY (v1, AR) → UnitY2 (v2, NAR, FastSpeech2-basiert) |
| Aufgaben | ASR, S2TT, T2TT, TTS, S2ST |
| Sprachen (v2) | 101 Sprach-Input · 96 Text-I/O · 35–36 Sprach-Output |
| Trainingsdaten | 1–4,5 Mio. h unbeschriftet + ~470k h SeamlessAlign parallel |
| Repo | facebookresearch/seamless_communication |

## Benchmarks

Auf **FLEURS** (X→Englisch), Vergleich mit Whisper-Large-v2 als Baseline:

| Modell | ASR WER (Fleurs) ↓ | S2TT BLEU (X→EN) ↑ | S2ST ASR-BLEU ↑ |
|---|---|---|---|
| Whisper-Large-v2 (1,5 B) | 41,7 % | 22,7 | 23,2 |
| SeamlessM4T-Medium (1,2 B) | 21,9 % | — | — |
| SeamlessM4T-Large v1 (2,3 B) | 23,1 % | 24,0 | 25,8 |
| **SeamlessM4T v2 Large (2,3 B)** | **18,5 %** | **26,6** | **29,7** |

Human-Eval (XSTS-Score, 0–5): Übersetzungen *aus* dem Englischen erreichen für 24 Sprachen konsistent > 4,0; *ins* Englische deutliche Verbesserung ggü. Whisper-Large-v2 in 7 von 24 Sprachen (v1-Paper).

*Hinweis:* Die genauen Sprachzahlen (101/96/36) sind Meta-Standardangaben zu v2; die hier verwendete Benchmark-Tabelle stammt aus EmergentMinds Aufbereitung der FLEURS-Ergebnisse und deckt sich mit den in der Nature-Publikation berichteten Gewinnen. Zahlen pro Sprache variieren; die primären Tabellen stehen in Paper/Nature-Artikel.

## Quellen

1. Meta AI SeamlessM4T Blog — https://ai.meta.com/blog/seamless-m4t/
2. Hugging Face SeamlessM4T-v2 Doc — https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2
3. Nature "Joint speech and text machine translation for up to 100 languages" — https://www.nature.com/articles/s41586-024-08359-z
4. facebookresearch/seamless_communication (GitHub) — https://github.com/facebookresearch/seamless_communication
5. EmergentMind SeamlessM4T Models (Parameter/Encoder/Benchmark-Details) — https://www.emergentmind.com/topics/seamlessm4t-models
6. SeamlessM4T v1 Publikation (Meta Research) — https://ai.meta.com/research/publications/seamlessm4t-massively-multilingual-multimodal-machine-translation/
