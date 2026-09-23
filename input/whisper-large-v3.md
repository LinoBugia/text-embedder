---
title: Whisper large-v3
category: audio-speech
models_covered: [Whisper large-v3]
last_updated: 2026-07-11
sources: [https://huggingface.co/openai/whisper-large-v3, https://arxiv.org/abs/2212.04356, https://github.com/openai/whisper, https://github.com/openai/whisper/discussions/1762, https://inferencebench.io/models/openai/whisper-large-v3/, https://openai.com/index/whisper/]
---

# Whisper large-v3

## Übersicht

**Whisper large-v3** ist OpenAIs Flaggschiff-Modell für automatische Spracherkennung (ASR) und Sprachübersetzung, veröffentlicht im **November 2023** unter der **MIT-Lizenz** (vollständig offene Gewichte + Code). Whisper ist ein *general-purpose*-Modell: ein einzelnes Netz erledigt multilinguale Transkription, Übersetzung nach Englisch (S2TT), Sprachidentifikation (LID) und Timestamp-Vorhersage. Der Schlüssel ist die **weakly-supervised**-Trainingsstrategie: statt sorgfältig gelabelter Datensätze wurden hunderttausende Stunden schwach beschrifteten Web-Audios genutzt, was zu hoher Robustheit gegenüber Akzenten, Hintergrundgeräuschen und Fachsprache führt.

large-v3 ist die dritte große Generation (nach large / large-v2). Es unterstützt **99 Sprachen** (large-v3 ergänzte u. a. Kantonesisch) und erreichte gegenüber large-v2 laut OpenAI eine Fehlerreduktion von 10–20 %. Später erschien eine schnellere **large-v3-turbo**-Variante (809 M Parameter, minimaler Genauigkeitsverlust). Whisper ist wegen Offenheit und Qualität der De-facto-Standard für offene Transkription und dient auch als Encoder-Basis für Audio-LLMs (z. B. Qwen2-Audio).

## Architektur

Whisper large-v3 ist ein **Encoder-Decoder-Transformer** (Sequence-to-Sequence). Der Verarbeitungsweg:

- **Frontend:** Rohaudio wird auf 16 kHz resampled und in ein **log-Magnituden-Mel-Spektrogramm** mit **128 Mel-Bins** umgewandelt (25 ms Analysefenster, 10 ms Stride). *Hinweis:* large-v3 erhöhte die Mel-Bins von 80 (large-v2) auf 128 — die wesentliche Architektur-Änderung ggü. large-v2.
- **Encoder:** stapelt Self-Attention-Transformer-Layer über die Spektrogramm-Frames und erzeugt eine akustische Repräsentation über ein festes **30-Sekunden-Fenster**.
- **Decoder:** autoregressiver Transformer-Decoder, der Text-Tokens erzeugt. Aufgaben werden über **spezielle Steuertokens** im Multitask-Format kodiert: `<|startoftranscript|>`, Sprach-Token, `<|transcribe|>` / `<|translate|>`, `<|notimestamps|>` etc. So werden Transkription, Übersetzung, LID und Zeitstempel in einem einheitlichen Sequenzformat abgebildet.
- **Positional Encoding / Kontext:** Der Decoder hat eine maximale Kontextlänge von **448 Tokens**; längere Audios werden in 30-s-Fenster zerlegt und sequentiell (oder gechunkt) verarbeitet.

Architektur-Details (large-v3, dense): **32 Layer**, **Hidden-Dimension 1280**, **20 Attention-Heads** (KV-Heads 20, Head-Dim 64), **Vokabulargröße 51.866**. Insgesamt **1,55 Mrd. Parameter**. Die Architektur ist identisch zu large-v2, bis auf die 128 Mel-Bins und einen zusätzlichen Sprach-Token.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Organisation | OpenAI |
| Release | November 2023 |
| Lizenz | MIT |
| Parameter (gesamt = aktiv, dense) | 1,55 B (1550 M) |
| Layer | 32 |
| Hidden-Dimension | 1.280 |
| Attention-Heads | 20 (KV 20, Head-Dim 64) |
| Vokabular | 51.866 |
| Mel-Bins | 128 |
| Audio-Fenster | 30 s |
| Decoder-Kontext | 448 Tokens |
| Sprachen | 99 |
| Trainingsdaten | 5 Mio. h (1 Mio. schwach beschriftet + 4 Mio. pseudo-beschriftet via large-v2), 2,0 Epochen |
| Turbo-Variante | 809 M Parameter |

## Benchmarks

Word Error Rate (WER, niedriger = besser):

| Benchmark | Metrik | Wert | Quelle |
|---|---|---|---|
| LibriSpeech test-clean | WER | ~2,0 % | [13] Local AI Master / OpenAI-Paper |
| LibriSpeech test-other (noisy) | WER | ~4,2 % | [13] |
| Fleurs (99-Sprachen-Schnitt) | WER | ~10,6 % | [13] |
| Short-form Transkription | WER | 8,4 % | Groq/HF Model Card |
| Sequential long-form | WER | 10,0 % | Groq Docs |
| Chunked long-form | WER | 11,0 % | Groq Docs |
| Englisch (gesamt, v3) | WER | ~2,4 % | Fact-Check-Zusammenfassung (Sekundärquelle) |

*Hinweis:* Genaue WER-Zahlen variieren je nach Evaluations-Setup (short-form vs. long-form, chunked vs. sequential, VAD/Chunking). LibriSpeech-Werte ~2,0/4,2 % stammen aus Sekundärquellen, die sich auf das OpenAI-Paper und die HF-Modellkarte berufen; die primäre WER-Aufschlüsselung pro Sprache liefert OpenAI als Grafik im GitHub-Repo (Common Voice 15 + Fleurs). Ressourcen: ~10 GB VRAM (GPU), ~2,9 GiB (GGML), GPU stark empfohlen.

## Quellen

1. OpenAI Whisper large-v3 Model Card — https://huggingface.co/openai/whisper-large-v3
2. Whisper Paper "Robust Speech Recognition via Large-Scale Weak Supervision" (arXiv:2212.04356) — https://arxiv.org/abs/2212.04356
3. Whisper GitHub Repository — https://github.com/openai/whisper
4. large-v3 Release Discussion #1762 — https://github.com/openai/whisper/discussions/1762
5. InferenceBench Architektur-Details (Layer/Head/Vokabular) — https://inferencebench.io/models/openai/whisper-large-v3/
6. Introducing Whisper (OpenAI Blog) — https://openai.com/index/whisper/
7. Groq Docs Whisper large-v3 (WER-Zahlen) — https://console.groq.com/docs/model/whisper-large-v3
8. Local AI Master Whisper large-v3 (Benchmark-Zusammenfassung) — https://localaimaster.com/models/whisper-large-v3
