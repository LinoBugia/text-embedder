---
title: Moshi
category: audio-speech
models_covered: [Moshi, Mimi, Helium]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2410.00037, https://arxiv.org/html/2410.00037v2, https://kyutai.org/blog/2024-09-18-moshi-release/, https://github.com/kyutai-labs/moshi, https://kyutai.org/Moshi.pdf, https://huggingface.co/docs/transformers/en/model_doc/moshi]
---

# Moshi

## Übersicht

**Moshi** ist ein **Speech-Text-Foundation-Modell** und **full-duplex** Sprachdialog-Framework von **Kyutai** (Paris; finanziert u. a. von Iliad, CMA CGM, Schmidt Sciences), veröffentlicht am **18. September 2024** (Paper arXiv:2410.00037). Es ist das **erste offene full-duplex Echtzeit-Sprach-Large-Language-Model**: Es hört und spricht gleichzeitig, erlaubt Überlappungen, Backchanneling ("mhm") und Unterbrechungen ohne feste Sprecherwechsel — mit einer theoretischen Latenz von **160 ms** (praktisch ~**200 ms**). Der Code (PyTorch, Rust/Candle, MLX) ist unter Apache/MIT offen; die Modellgewichte werden unter **CC-BY 4.0** bereitgestellt.

Zwei Sprachvarianten wurden mit künstlich erzeugten Stimmen veröffentlicht: **Moshiko** (männlich) und **Moshika** (weiblich). Moshi kann in Echtzeit auf einer NVIDIA-L4-GPU oder einem Apple-M3-MacBook-Pro laufen. Ein späterer Ableger, **MoshiVis**, erweitert Moshi um Bildverständnis.

## Architektur

Moshi ist ein **Multi-Stream Speech-to-Speech Transformer** aus drei Hauptkomponenten:

1. **Helium (Text-LLM-Backbone):** ein von Grund auf trainiertes **7-Mrd.-Parameter**-Sprachmodell, trainiert auf **2,1 Billionen Tokens** Text. Es liefert das Sprach-/Weltwissen und die Reasoning-Fähigkeit.
2. **Mimi (neuronaler Streaming-Audio-Codec):** wandelt 24-kHz-Audio in diskrete Tokens bei nur **12,5 Hz** Framerate und **1,1 kbps** Bandbreite. Mimi modelliert **semantische und akustische** Information gemeinsam (via Distillation aus einem selbstüberwachten Modell) und ist **vollständig kausal** (streaming-fähig). Es nutzt einen **Residual-Vector-Quantizer (RVQ)** und übertrifft laut Kyutai frühere Codecs (SoundStream, Encodec, SpeechTokenizer, RVQGAN, SemantiCodec).
3. **Multi-Stream-Modellierung / RQ-Transformer:** Für jeden Zeitschritt werden die Tokens von **Moshis eigener Stimme** und der **Nutzerstimme** parallel gestapelt und modelliert (getrennte Streams). Die Architektur ist eine Variante des **RQ-Transformers**: ein großer **Temporal Transformer** (der 7B-Helium-Backbone) plus ein kleinerer **Depth Transformer**, der die semantische/akustische Token-Hierarchie je Zeitschritt verarbeitet, *ohne* die Sequenzlänge zu erhöhen. Für 1 Sekunde Audio sind nur **12,5 Durchläufe** durch den 7B-Backbone nötig.

**Inner Monologue:** Ein Schlüsseltrick — Moshi sagt zunächst **zeitalignierten Text** für seine eigene Sprache voraus, *bevor* es die semantischen/akustischen Audiotokens generiert. Das verbessert die Sprachqualität deutlich. Durch gezieltes Verzögern der Audio- bzw. Text-Tokens lässt sich dasselbe Prinzip auch als **Streaming-TTS** (Text→Audio) oder **Streaming-ASR mit Alignment** (Audio→Text) nutzen.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Organisation | Kyutai |
| Release | 18. September 2024 |
| Paper | arXiv:2410.00037 |
| Lizenz | Code Apache/MIT · Gewichte CC-BY 4.0 |
| Backbone (Helium) | 7 B Parameter, 2,1 T Tokens Text |
| Gesamtgröße (on-device) | ~7,6 B (inkl. Audio-Komponenten) |
| Audio-Codec (Mimi) | 24 kHz → 12,5 Hz, 1,1 kbps, RVQ, kausal (CC-BY) |
| Kern-Architektur | Multi-Stream RQ-Transformer (Temporal + Depth) |
| Latenz | 160 ms theoretisch / ~200 ms praktisch |
| Trainingsdaten (Fine-tune) | 20.000 h synthetische full-duplex Dialoge (via Helium-Skripte + Multi-Stream-TTS) |
| Varianten | Moshiko (männl.) / Moshika (weibl.) |
| Inferenz-Hardware | NVIDIA L4 GPU / Apple M3 MacBook Pro (Echtzeit) |
| Aufgabe | Full-duplex Speech-to-Speech Dialog (+ ableitbar TTS/ASR) |

## Benchmarks

Moshis Hauptbeitrag ist qualitativer/architektonischer Natur (erstes offenes full-duplex Echtzeit-Sprach-LLM), daher stehen **Latenz** und **Codec-Qualität** im Vordergrund:

- **Latenz:** 160 ms theoretisch, ~200 ms praktisch — deutlich unter der ~80-ms-pro-Schritt-Grenze für Echtzeit bei 12,5-Hz-Mimi.
- **Mimi-Codec:** übertrifft SoundStream, Encodec, SpeechTokenizer, RVQGAN und SemantiCodec bei semantisch-akustischer Audio-Rekonstruktion (Kyutai-Paper).
- **Sprachverständnis / Wissen:** Das Moshi-Paper berichtet Ergebnisse auf gesprochenen QA- und Sprachverständnis-Benchmarks sowie Text-Benchmarks des Helium-Backbones.

*Hinweis:* Konkrete numerische Benchmark-Tabellen (z. B. gesprochene QA-Scores, MOS-Werte für Audioqualität) sind im vollständigen Paper (arXiv:2410.00037) enthalten; die hier zusammengefassten Werte (Latenz, Codec-Bandbreite, Frame-Rate) stammen aus dem Kyutai-Release-Blog und dem GitHub-Repo. Präzise MOS-/QA-Zahlen wurden in dieser Recherche nicht aus den Primärtabellen extrahiert und sollten bei Bedarf direkt aus dem PDF verifiziert werden.

## Quellen

1. Moshi Paper (arXiv:2410.00037) — https://arxiv.org/abs/2410.00037
2. Moshi Paper HTML (Modellabschnitt) — https://arxiv.org/html/2410.00037v2
3. Kyutai Moshi Release Blog — https://kyutai.org/blog/2024-09-18-moshi-release/
4. kyutai-labs/moshi (GitHub) — https://github.com/kyutai-labs/moshi
5. Moshi Technical PDF — https://kyutai.org/Moshi.pdf
6. Hugging Face Transformers — Moshi Doc — https://huggingface.co/docs/transformers/en/model_doc/moshi
7. MoshiVis (Bilderweiterung) — https://kyutai.org/moshivis/
