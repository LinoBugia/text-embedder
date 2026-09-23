---
title: InternVL (InternVL 2.5)
category: vision-language
models_covered: [InternVL 1.5, InternVL 2.0, InternVL 2.5]
last_updated: 2026-07-11
sources: [https://internvl.github.io/blog/2024-12-05-InternVL-2.5/, https://huggingface.co/papers/2412.05271, https://github.com/OpenGVLab/InternVL, https://arxiv.org/abs/2404.16821]
---

# InternVL (InternVL 2.5)

## Übersicht
InternVL ist eine Reihe multimodaler LLMs von **OpenGVLab (Shanghai AI Laboratory)**, die sich durch einen ungewöhnlich großen, eigens skalierten Vision-Encoder (InternViT-6B) und die breiteste Größenskala der offenen VLM-Welt auszeichnet. InternVL 1.0 (CVPR 2024) skalierte erstmals einen Vision-Foundation-Encoder auf 6B Parameter und richtete ihn auf ein LLM aus. InternVL 1.5 (April 2024) führte dynamisches hochauflösendes Tiling und einen starken bilingualen (EN/ZH) Datensatz ein.

**InternVL 2.5 (Dezember 2024)** ist ein Meilenstein: Mit **InternVL2.5-78B** überschritt erstmals ein offenes VLM die 70%-Marke auf MMMU (70.1% val mit CoT) und schloss weitgehend zu GPT-4o und Claude 3.5 Sonnet auf. Die Reihe umfasst sieben Größen von 1B bis 78B.

## Architektur
InternVL 2.5 baut auf InternVL 2.0 auf und folgt dem "ViT-MLP-LLM"-Paradigma:

- **Vision Encoder (InternViT):** Zwei Varianten — **InternViT-300M-448px-V2_5** (304M Parameter) für kleine Modelle (1B–8B) und **InternViT-6B-448px-V2_5** (5,54 Mrd. Parameter) für große Modelle (26B–78B). Der 6B-Encoder ist deutlich größer als die üblichen ~300M–900M-Encoder anderer VLMs und liefert entsprechend reichhaltigere visuelle Features.
- **MLP-Projektor:** Verbindet InternViT mit dem LLM (12M–172M Parameter je nach Größe).
- **LLM-Backbones:** **Qwen2.5** und **InternLM2.5** in verschiedenen Größen. *(Die genaue Backbone-Zuordnung pro Größe ist im Technical Report / auf den Model Cards dokumentiert; der InternVL-2.5-Blog nennt die Backbone-Namen nicht explizit.)*
- **Dynamisches Tiling / Auflösung:** Bilder werden in bis zu **36 Kacheln à 448×448** während des Trainings zerlegt (bis zu **128 Kacheln** zur Testzeit). Zusätzlich wird ein Thumbnail des Gesamtbilds beigefügt, um globalen Kontext zu bewahren.
- **Pixel Shuffle:** InternVL nutzt (aus früheren Versionen bekannt) Pixel-Shuffle zur Reduktion der visuellen Token pro Kachel (Faltung benachbarter Patches). *Hinweis: Der 2.5-Blogtext erwähnt "pixel shuffle" nicht explizit, das Verfahren ist jedoch fester Bestandteil der InternVL-Serie.*

**Trainings-Innovationen (2.5):**
- **Progressive Scaling Strategy:** InternViT wird zunächst mit kleineren, günstigen LLMs aligned und trainiert, dann via Shared-Weight-Mechanismus ohne Neutraining auf größere LLMs übertragen — reduziert Datenbedarf und Kosten großer Modelle.
- **Random JPEG Compression** (Qualität 75–100) simuliert Web-Bilddegradation.
- **Loss Reweighting** ("square averaging") balanciert Gradienten über verschieden lange Antworten.
- LLM-basiertes Qualitäts-Scoring + regelbasiertes Filtern gegen anomale Daten (weniger repetitive CoT-Generierungen) und dynamisches Data-Packing für höhere GPU-Effizienz.

## Specs

| Modell | Gesamt | ViT | MLP | LLM | Vision-Encoder |
|---|---|---|---|---|---|
| InternVL2.5-1B | 938,19M | 304,01M | 4,48M | 629,70M | InternViT-300M |
| InternVL2.5-2B | 2,21B | 304,01M | 12,60M | 2,21B* | InternViT-300M |
| InternVL2.5-4B | 3,71B | 304,01M | 12,60M | 3,40B | InternViT-300M |
| InternVL2.5-8B | 8,08B | 304,01M | 33,57M | 7,74B | InternViT-300M |
| InternVL2.5-26B | 25,51B | 5,54B | 116,43M | 19,86B | InternViT-6B |
| InternVL2.5-38B | 38,39B | 5,54B | 91,79M | 32,76B | InternViT-6B |
| InternVL2.5-78B | 78,41B | 5,54B | 172,01M | 72,70B | InternViT-6B |

(*Der im Blog gelistete LLM-Wert für die 2B-Variante entspricht dem Gesamtwert; interpretiere die LLM-Spalte für 2B mit Vorsicht.)

- **Lizenz:** **MIT** (InternVL2.5-Modelle auf HuggingFace stehen überwiegend unter MIT; bei Qwen2.5-basierten großen Backbones sind zusätzliche Backbone-Bedingungen möglich — Model Card prüfen).
- **Kontextlänge:** abhängig vom LLM-Backbone (bis ~32k bei Qwen2.5/InternLM2.5).
- **Auflösung:** bis 128 Kacheln × 448px (Testzeit).

## Benchmarks (InternVL2.5-78B)
- **MMMU (val, CoT):** **70.1%** — erstes offenes MLLM über 70%; MMMU (test): 61.8%.
- **MathVista (mini):** **72.3%** (schlägt Qwen2-VL-72B mit 70.5% und GPT-4o mit 63.8%).
- **OCRBench:** 854 (Qwen2-VL-72B leicht höher mit 877).

## Architektur-Besonderheiten
- Größter Vision-Encoder der offenen VLM-Welt (InternViT-6B, 5,54 Mrd. Parameter).
- Breiteste Größenskala: 1B bis 78B in einer konsistenten Familie.
- Progressive Scaling mit Shared-Weight-Transfer des Encoders → deutlich günstigeres Training großer Modelle.
- Erstes offenes VLM, das die 70%-MMMU-Schwelle knackte.

## Quellen
1. InternVL 2.5 Blog (Encoder, Tiling, Parametertabelle, Benchmarks, Training) — https://internvl.github.io/blog/2024-12-05-InternVL-2.5/
2. InternVL 2.5 Technical Report — https://huggingface.co/papers/2412.05271
3. InternVL GitHub — https://github.com/OpenGVLab/InternVL
4. InternVL 1.5 Paper — https://arxiv.org/abs/2404.16821
5. InternVL 1.0 Paper (CVPR 2024) — https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_InternVL_Scaling_up_Vision_Foundation_Models_and_Aligning_for_Generic_CVPR_2024_paper.pdf
6. Model Cards — https://huggingface.co/OpenGVLab/InternVL2_5-78B , https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5
