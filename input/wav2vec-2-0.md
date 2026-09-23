---
title: Wav2Vec 2.0
category: audio-speech
models_covered: [Wav2Vec 2.0]
last_updated: 2026-07-11
sources: [https://huggingface.co/docs/transformers/en/model_doc/wav2vec2, https://huggingface.co/papers/2006.11477, https://arxiv.org/abs/2006.11477, https://www.quantumrun.com/consulting/wav2vec-2-0-statistics/, https://ritvik19.medium.com/papers-explained-485-wav2vec-2-0-fe05d2379da1]
---

# Wav2Vec 2.0

## Übersicht

**Wav2Vec 2.0** ist ein Framework von **Meta AI (FAIR)** für **selbstüberwachtes Lernen von Sprach-Repräsentationen**, vorgestellt im **Juni 2020** von Alexei Baevski, Henry Zhou, Abdelrahman Mohamed und Michael Auli (Paper "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", arXiv:2006.11477). Es war ein Meilenstein, weil es zeigte, dass man aus großen Mengen **unbeschrifteten** Audios starke Repräsentationen lernen und anschließend mit sehr wenig gelabelten Daten hochwertige ASR erreichen kann: mit nur **10 Minuten** beschrifteten Daten (plus Pretraining auf 53k Stunden) erzielt es 4,8/8,2 % WER — bei voller Beschriftung (960 h) sogar 1,8/3,3 %.

Wav2Vec 2.0 begründete das dominante ASR-Paradigma "pretrain on raw audio, fine-tune with CTC" und ist Grundlage vieler Nachfolger (XLS-R für Multilingualität, w2v-BERT / Wav2Vec2-BERT 2.0, das u. a. in SeamlessM4T als Encoder dient). Die Original-Implementierung liegt in Metas fairseq (MIT-Lizenz); die auf Hugging Face gehosteten Checkpoints (`facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-960h-lv60-self`) sind unter **Apache-2.0** verfügbar.

## Architektur

Wav2Vec 2.0 kombiniert vier Bausteine und ein kontrastives Trainingsziel:

- **CNN-Feature-Encoder:** ein 1D-Convolutional-Netz, das direkt aus der Roh-Wellenform latente Sprach-Features extrahiert. Standard: **7 Conv-Layer** mit Dimensionen (512, 512, 512, 512, 512, 512, 512), Strides (5, 2, 2, 2, 2, 2, 2) und Kernels (10, 3, 3, 3, 3, 2, 2). Ergebnis: ein Latent-Vektor je ~25 ms Audio.
- **Transformer-Kontextnetz:** baut kontextualisierte Repräsentationen über die Latents. Base: 12 Layer, Hidden 768, 12 Heads, Intermediate 3072. Large: 24 Layer, Hidden 1024, 16 Heads.
- **Quantisierungsmodul (Produkt-Quantisierung):** diskretisiert die Latents in Codevektoren. Standard: **G = 2** Codebooks mit je **V = 320** Einträgen (theoret. max. 320² ≈ 102,4k Codewörter), `codevector_dim = 256`. Die Auswahl erfolgt differenzierbar über **Gumbel-Softmax** (Temperatur τ von 2 auf 0,5 (Base) bzw. 0,1 (Large) angehalten).
- **Trainingsziel:** Teile der Latents werden **maskiert**; das Modell löst einen **kontrastiven Task** (contrastive loss, Temperatur κ = 0,1, 100 Negatives) — es muss die korrekte quantisierte Zielrepräsentation an maskierten Positionen von Ablenkern unterscheiden. Ein **Diversity-Loss** (Gewicht 0,1) fördert die gleichmäßige Nutzung aller Codebook-Einträge.

Nach dem Pretraining wird für ASR ein linearer Kopf aufgesetzt und mit **CTC-Loss** auf gelabelten Daten fine-getunt. Für Sprach-LM-Boosting kann ein n-Gramm-Sprachmodell beim Decoding kombiniert werden.

## Specs-Tabelle

| Merkmal | Wert |
|---|---|
| Organisation | Meta AI (FAIR) |
| Autoren | Baevski, Zhou, Mohamed, Auli (2020) |
| Release | Juni 2020 (arXiv:2006.11477) |
| Lizenz | MIT (fairseq) / Apache-2.0 (HF-Checkpoints) |
| Parameter — Base | ~95 M |
| Parameter — Large | ~317 M |
| Feature-Encoder | 7-Layer 1D-CNN (512 Kanäle) |
| Transformer Base | 12 Layer / 768 Hidden / 12 Heads |
| Transformer Large | 24 Layer / 1024 Hidden / 16 Heads |
| Quantisierung | Produkt-Quant., G=2, V=320, Gumbel-Softmax |
| Frame-Rate | ~25 ms je Latent |
| Pretraining-Daten | LibriSpeech LS-960 (960 h) bzw. Libri-Light (53–60k h) |
| Fine-tuning | CTC |

*Hinweis:* Die HF-Doku listet die Default-Konfiguration (Base-Stil) auf, nennt aber die exakten Gesamt-Parameterzahlen nicht direkt. Die Werte ~95 M (Base) und ~317 M (Large) sind etablierte, in der Literatur/Community verbreitete Zahlen; das Paper selbst nennt die genauen Layer-Konfigurationen (Large = 24 Transformer-Blöcke).

## Benchmarks

LibriSpeech WER (niedriger = besser):

| Konfiguration | test-clean WER | test-other WER | gelabelte Daten |
|---|---|---|---|
| Wav2Vec2-Base-960h | 3,4 % | 8,6 % | 960 h |
| Wav2Vec2-Large-960h | 1,8 % | 3,3 % | 960 h |
| Wav2Vec2-Large-LV60 | 1,9 % | 3,9 % | 960 h |
| Wav2Vec2 (1 h gelabelt) | 2,7 % | 5,8 % | 1 h |
| Wav2Vec2 (10 min gelabelt) | 4,8 % | 8,2 % | 10 min (+53k h Pretraining) |

Kernaussage des Papers: Mit **10 Minuten** gelabelten Daten + Pretraining auf 53k h übertrifft Wav2Vec 2.0 den vorherigen State-of-the-Art auf dem 100-Stunden-Subset — bei **100× weniger** gelabelten Daten.

## Quellen

1. Hugging Face Transformers — Wav2Vec2 Doc — https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
2. Wav2Vec 2.0 Paper (HF Papers) — https://huggingface.co/papers/2006.11477
3. Wav2Vec 2.0 Paper (arXiv:2006.11477) — https://arxiv.org/abs/2006.11477
4. Quantumrun — LibriSpeech Benchmark-Tabelle — https://www.quantumrun.com/consulting/wav2vec-2-0-statistics/
5. Papers Explained 485: wav2vec 2.0 (Quantisierungs-/Gumbel-Details) — https://ritvik19.medium.com/papers-explained-485-wav2vec-2-0-fe05d2379da1
6. NVIDIA NGC wav2vec2 (Fairseq-Basis) — https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/wav2vec2_pyt
