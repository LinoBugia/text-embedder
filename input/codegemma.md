---
title: CodeGemma
category: code
models_covered: [CodeGemma-2B, CodeGemma-7B (PT), CodeGemma-7B-IT]
last_updated: 2026-07-11
sources: [https://arxiv.org/abs/2406.11409, https://arxiv.org/html/2406.11409v2, https://huggingface.co/google/codegemma-7b, https://ai.google.dev/gemma/docs/codegemma]
---

# CodeGemma

## Übersicht
CodeGemma ist Googles Familie offener Code-Modelle, aufgebaut auf **Gemma** (Paper "CodeGemma: Open Code Models Based on Gemma", arXiv:2406.11409, v1 17. Jun 2024). Die Modelle entstehen durch code-lastiges Continued-Pretraining der Gemma-Basismodelle und zielen besonders auf **schnelles, latenzsensitives Code-Infilling (Autocomplete)** ab. Die 2B-Variante ist explizit als State-of-the-Art-Autocomplete-Modell für latenzkritische Umgebungen positioniert; die 7B-Varianten (PT/IT) bieten stärkere Generierung und Instruction-Following bei erhaltener natürlichsprachlicher und mathematischer Kompetenz.

## Specs
| Eigenschaft | CodeGemma-2B | CodeGemma-7B (PT) | CodeGemma-7B-IT |
|---|---|---|---|
| Organisation | Google | — | — |
| Release | 17.–19. Jun 2024 (Paper); erste Modelle Apr 2024 | — | — |
| Basis | Gemma | Gemma | Gemma |
| Trainings-Tokens | +500B (Englisch, Math-Datensätze, synthetischer Code) | +500B | (Instruction-Tuning) |
| Kontext | 8K (von Gemma-Basis) | 8K | 8K |
| FIM | Ja (80 % FIM-Rate, 50–50 PSM/SPM) | Ja | eingeschränkt |
| Lizenz | Gemma License | Gemma License | Gemma License |
| Fokus | schnelles Code-Infilling | Generierung + NL/Math | Instruct/Chat |

FIM-Spezialtokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|file_separator|>`.

## Architektur
Decoder-only Transformer auf Basis von **Gemma** (RoPE, GeGLU-Aktivierung, RMSNorm, große Vokabulargröße ~256K). Code-spezifische Anpassungen:
- **Code-Continued-Pretraining:** +500B Tokens (Mix aus englischem Text, Open-Source-Mathematik-Datensätzen und synthetisch generiertem Code) auf den Gemma-Checkpoints.
- **Fill-in-the-Middle (FIM):** Sehr hohe FIM-Rate von **80 %**, davon 50–50 aufgeteilt in PSM (Prefix-Suffix-Middle) und SPM (Suffix-Prefix-Middle). Multi-File-Training mit `<|file_separator|>`-Token für repository-Kontext. Diese hohe FIM-Rate macht CodeGemma besonders stark im Autocomplete.
- **Kontext:** erbt das 8K-Fenster der Gemma-Basis.

## Benchmarks (Pass@1, HF Model Card)
| Benchmark | 2B | 7B (PT) | 7B-IT |
|---|---|---|---|
| HumanEval (Overall) | 31,1 | 44,5 | 56,1 |
| MBPP (Overall) | 43,6 | 56,2 | 54,2 |
| HumanEval Single Line (FIM) | 78,41 | 76,09 | 68,25 |
| HumanEval Multi Line (FIM) | 51,44 | 58,44 | 20,05 |

BabelCode (mehrsprachig, Auszug HumanEval): Python 2B 21,7 / 7B 42,2 / 7B-IT 48,4; Java 2B 29,2 / 7B 41,0 / 7B-IT 48,4; C++ 2B 24,2 / 7B 32,9 / 7B-IT 42,2; JavaScript 2B 21,7 / 7B 39,8 / 7B-IT 46,0.

Beobachtung: Das 2B-Modell glänzt bei Single-Line-Infilling (78,41), fällt bei Multi-Line/Generierung aber deutlich ab – konsistent mit seiner Autocomplete-Ausrichtung.

## Besonderheiten / Trivia
- **Höchste FIM-Rate der Klasse (80 %)** – bewusst auf Autocomplete/Infilling optimiert statt auf reine Chat-Generierung.
- 2B-Variante ist als latenzoptimiertes On-Device-Autocomplete-Modell konzipiert.
- Nutzt Gemmas großes ~256K-Vokabular (breite mehrsprachige/Code-Abdeckung).
- **Lizenz:** Gemma License (nicht OSI-Open-Source); Nutzung erfordert Akzeptanz der Google-Nutzungsbedingungen, erlaubt aber breite (auch kommerzielle) Verwendung unter Auflagen.

## Quellen
1. CodeGemma: Open Code Models Based on Gemma – https://arxiv.org/abs/2406.11409 (HTML v2: https://arxiv.org/html/2406.11409v2)
2. CodeGemma-7B HF Model Card (Benchmarks, FIM-Details) – https://huggingface.co/google/codegemma-7b
3. Google AI CodeGemma Docs – https://ai.google.dev/gemma/docs/codegemma
