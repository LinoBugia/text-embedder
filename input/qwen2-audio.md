---
title: Qwen2-Audio
category: audio-speech
models_covered: [Qwen2-Audio, Qwen2-Audio-7B, Qwen2-Audio-7B-Instruct]
last_updated: 2026-07-11
sources: [https://arxiv.org/html/2407.10759v1, https://arxiv.org/abs/2407.10759, https://qwenlm.github.io/blog/qwen2-audio/, https://github.com/QwenLM/Qwen2-Audio, https://localaimaster.com/models/qwen-2-audio-7b]
---

# Qwen2-Audio

## Übersicht

**Qwen2-Audio** ist ein **Large Audio-Language Model (Audio-LLM)** von **Alibaba Cloud (Qwen-Team)**, veröffentlicht im **Juli 2024** (Technical Report arXiv:2407.10759, 15.07.2024). Es kann verschiedene Audiosignale — Sprache, Umgebungsgeräusche, Musik und Mischungen — analysieren und darauf in natürlicher Sprache antworten. Die Modellgewichte (Qwen2-Audio-7B und Qwen2-Audio-7B-Instruct) sind offen unter **Apache-2.0** verfügbar (der arXiv-Report selbst steht unter CC BY 4.0 — das ist die Paper-, nicht die Modelllizenz).

Das Besondere: Qwen2-Audio unterstützt **zwei nahtlos integrierte Modi ohne Hotword oder Systemprompt-Umschaltung**:
- **Voice Chat:** freie, rein sprachbasierte Konversation (Nutzer spricht, Modell antwortet).
- **Audio Analysis:** analytische Aufgaben über beliebiges Audio, gesteuert per Sprach- oder Textinstruktion.

Es erreichte State-of-the-Art auf mehreren Benchmarks (u. a. Aishell2, FLEURS-zh, VocalSound, AIR-Bench Chat) und ist Nachfolger von Qwen-Audio.

## Architektur

Qwen2-Audio brückt einen Audio-Encoder an ein Text-LLM ("Audio-LLM-Bridge"-Prinzip):

- **Audio-Encoder:** basiert auf dem **Whisper-large-v3**-Modell. Audio wird auf **16 kHz** resampled und in ein **128-Kanal-Mel-Spektrogramm** umgewandelt (25 ms Fenster, 10 ms Hop). Eine **Pooling-Schicht mit Stride 2** reduziert die Auflösung, sodass jeder Ausgabe-Frame ~**40 ms** Originalaudio repräsentiert.
- **LLM-Backbone:** **Qwen-7B**. Die Audio-Repräsentationen werden in den Embedding-Raum des LLM projiziert; das LLM generiert die textuelle/verbale Antwort autoregressiv.
- **Gesamtgröße:** **8,2 Mrd. Parameter** (Qwen-7B + Audio-Encoder + Projektion).

**Dreistufiges Training:**
1. **Pre-training:** ersetzt komplexe hierarchische Tags (wie in Qwen-Audio) durch **natürliche Sprach-Prompts** → bessere Generalisierung und Instruktionsbefolgung.
2. **Supervised Fine-Tuning (SFT):** gemeinsames Training auf hochwertigen SFT-Daten für Voice-Chat und Audio-Analysis.
3. **Direct Preference Optimization (DPO):** Ausrichtung an menschlichen Präferenzen über Triplet-Präferenzdaten → bessere Faktentreue und Verhalten.

Dadurch entfällt der explizite Task-Switch: das Modell erkennt aus dem Input, ob es sich um Voice-Chat oder eine Analyse-Instruktion handelt.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Organisation | Alibaba Cloud (Qwen) |
| Release | Juli 2024 (arXiv:2407.10759) |
| Lizenz | Apache-2.0 (Modell) · CC BY 4.0 (Paper) |
| Parameter (gesamt) | 8,2 B |
| LLM-Basis | Qwen-7B |
| Audio-Encoder | Whisper-large-v3-basiert, 128-Mel, Pooling-Stride 2 (~40 ms/Frame) |
| Audio-Sampling | 16 kHz |
| Modi | Voice Chat + Audio Analysis (ohne Hotword/Systemprompt) |
| Training | 3 Stufen: Pre-training → SFT → DPO |
| Checkpoints | Qwen2-Audio-7B, Qwen2-Audio-7B-Instruct |

## Benchmarks

**ASR — Word Error Rate (WER ↓):**

| Datensatz | Split | WER |
|---|---|---|
| LibriSpeech | dev-clean | 1,3 % |
| LibriSpeech | dev-other | 3,4 % |
| LibriSpeech | test-clean | 1,6 % |
| LibriSpeech | test-other | 3,6 % |
| Common Voice 15 | en | 8,6 % |
| Common Voice 15 | zh | 6,9 % |
| Common Voice 15 | yue (Kant.) | 5,9 % |
| Common Voice 15 | fr | 9,6 % |
| Fleurs | zh (zero-shot) | 7,5 % |
| Aishell2 | Mic / iOS / Android | 3,0 / 3,0 / 2,9 % |

**Speech-to-Text-Translation — CoVoST2 (BLEU ↑):** en-de 29,9 · de-en 35,2 · en-zh 45,2 · zh-en 24,4 · es-en 40,0 · fr-en 38,5 · it-en 36,3.

**Weitere:**
- Speech Emotion Recognition (MELD, Accuracy): **55,3 %**
- Vocal Sound Classification (VocalSound, Accuracy): **93,92 %**
- **AIR-Bench Chat** (GPT-4-Score, 0–10): Speech **7,18** · Sound **6,99** · Music **6,79** · Mixed-Audio **6,77**

Laut Report erreicht Qwen2-Audio SOTA u. a. auf Aishell2, FLEURS-zh, VocalSound und dem AIR-Bench-Chat-Benchmark. Alle Zahlen aus dem Technical Report (arXiv:2407.10759v1).

## Quellen

1. Qwen2-Audio Technical Report HTML (arXiv:2407.10759v1) — https://arxiv.org/html/2407.10759v1
2. Qwen2-Audio Technical Report (arXiv Abstract) — https://arxiv.org/abs/2407.10759
3. Qwen2-Audio Blog "Chat with Your Voice!" — https://qwenlm.github.io/blog/qwen2-audio/
4. QwenLM/Qwen2-Audio (GitHub) — https://github.com/QwenLM/Qwen2-Audio
5. Local AI Master Qwen2-Audio 7B (Benchmark-Zusammenfassung) — https://localaimaster.com/models/qwen-2-audio-7b
