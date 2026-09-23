---
title: Snowflake Arctic
category: moe
models_covered: [Snowflake Arctic Base, Snowflake Arctic Instruct]
last_updated: 2026-07-11
sources: [https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/, https://github.com/Snowflake-Labs/snowflake-arctic, https://huggingface.co/Snowflake/snowflake-arctic-instruct, https://www.flexera.com/blog/finops/snowflake-arctic/]
---

# Snowflake Arctic

## Übersicht

Snowflake Arctic ist eine Familie von **Enterprise-fokussierten** offenen LLMs von **Snowflake AI Research**, veröffentlicht am **24. April 2024** unter **Apache-2.0-Lizenz** (inkl. offengelegter Datenrezeptur/Trainingsdetails). Arctic wurde gezielt auf **Enterprise-Aufgaben** — SQL-Generierung, Coding, Instruction-Following — optimiert und erreicht dabei laut Snowflake vergleichbare "Enterprise Intelligence" wie Llama 3 70B bei **~17x geringerem Trainings-Compute-Budget** (Trainingskosten unter 2 Mio. $, weniger als 3.000 GPU-Wochen).

Die zentrale Innovation ist eine **Dense-MoE-Hybrid-Architektur**, die einen kompakten dichten Transformer mit einem sehr breiten, fein-granularen MoE-MLP kombiniert.

## Architektur

Arctic nutzt eine einzigartige **Dense-MoE-Hybrid-Transformer**-Architektur:

- Ein **10B dichter Transformer** läuft **parallel** zu einem **residualen MoE-MLP** mit **128 Experten × 3,66B**.
- Ergebnis: **480B Gesamtparameter**, aber nur **17B aktive Parameter** pro Token, ausgewählt via **Top-2-Gating** über die 128 fein-granularen Experten.
- Die dichte und die MoE-Komponente werden kombiniert (residual), sodass jeder Token sowohl die dichte Bahn als auch die 2 gewählten Experten durchläuft.

Diese Aufteilung soll das Verhältnis aus Trainings-/Inferenz-Effizienz und Qualität für Enterprise-Workloads optimieren. Das hybride Design reduziert Speicherlesevorgänge deutlich (bis zu 4x weniger als Code-Llama 70B und 2,5x weniger als Mixtral 8x22B).

**Kontext:** natives Attention-Fenster von **4K** Tokens; Varianten mit Sliding-Window/Attention-Sinks für 32K-Erweiterung waren zum Release in Entwicklung.

**Training:** 3-Phasen-Daten-Curriculum über insgesamt **3,5T Tokens** — Phase 1: 1T Tokens (generische Fähigkeiten/Common Sense), Phase 2: 1,5T Tokens (Enterprise: Code, Math, SQL), Phase 3: 1T Tokens (Enterprise). Unterstützt FP6/FP8-Runtime, integriert mit DeepSpeed.

## Specs

| Spezifikation | Snowflake Arctic |
|---|---|
| Org | Snowflake AI Research |
| Release | 24. April 2024 |
| Lizenz | Apache 2.0 (inkl. Data Recipe) |
| Gesamtparameter | 480B |
| Aktive Params/Token | 17B |
| Architektur | Dense-MoE-Hybrid (10B dense + 128×3,66B MoE-MLP) |
| Experten / aktiv | 128 / 2 (Top-2) |
| Kontextlänge | 4K (32K-Erweiterung in Entwicklung) |
| Trainingsdaten | 3,5T Tokens (3-Phasen-Curriculum) |
| Trainings-Compute | < 2 Mio. $ / < 3.000 GPU-Wochen |
| Runtime | FP8/FP6, DeepSpeed |

## Benchmarks

Snowflake evaluierte Arctic vor allem auf einer aggregierten **"Enterprise Intelligence"**-Metrik (Coding: HumanEval+/MBPP+; SQL: Spider; Instruction-Following: IFEval):

- **Enterprise Intelligence:** auf Augenhöhe mit **Llama 3 70B**, bei ~17x geringerem Trainings-Compute-Budget.
- Übertrifft/matcht Llama 3 8B und Llama 2 70B auf Enterprise-Metriken bei **<1/2** ihres Trainings-Compute-Budgets.
- **Academic/Reasoning:** konkurrenzfähig mit DBRX (über 11 Metriken für Language Understanding & Reasoning); übertrifft DBRX in **Math (GSM8K)** bei ~7x geringerem Compute.
- **MMLU:** bewusst niedrigeres World-Knowledge-MMLU-Ergebnis "by design", da das Trainingsbudget stark auf Kosten-Effizienz und Enterprise-Skills optimiert wurde.

Inferenz: bei Batch-Size 1 mit FP8-Quantisierung passt Arctic auf einen einzelnen GPU-Knoten und generiert **70+ Tokens/Sekunde**.

Hinweis: Snowflake publizierte die exakten numerischen Einzelscores primär als Grafiken; die obigen Aussagen sind die im Blog verbal genannten, verifizierten Vergleichsaussagen.

## Besonderheiten / Trivia

- Eines der wenigen prominenten **Dense-MoE-Hybrid**-Designs (statt reinem MoE).
- Besonders "offen": Neben Gewichten wurde auch die **Datenrezeptur/Trainings-Cookbook** veröffentlicht.
- Klar auf **Enterprise-/DB-Aufgaben** (SQL via Spider) zugeschnitten — passt zu Snowflakes Data-Cloud-Geschäft.
- Extrem niedriges Trainingsbudget (<2 Mio. $) als zentrales Verkaufsargument.

## Quellen

1. Snowflake Arctic Blog: https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/
2. Snowflake Arctic GitHub: https://github.com/Snowflake-Labs/snowflake-arctic
3. snowflake-arctic-instruct — Hugging Face: https://huggingface.co/Snowflake/snowflake-arctic-instruct
4. Snowflake Arctic 101 (480B) — Flexera: https://www.flexera.com/blog/finops/snowflake-arctic/
