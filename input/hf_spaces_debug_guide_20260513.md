# Hugging Face Spaces Debugging Guide

_Debugging ZeroGPU / GPU Spaces step by step, updated for Blackwell-era ZeroGPU behavior._

This guide is designed to help you debug Hugging Face Spaces without assuming that the failure is simple, local, or stable across versions. It favors diagnostic leverage, searchability, and traceable sources over brevity.

## Table of contents

- [How to use this guide](#how-to-use-this-guide)
- [How this large guide is layered](#how-this-large-guide-is-layered)
- [Label conventions](#label-conventions)
- [Search index and aliases](#search-index-and-aliases)
- [Cross-reference map](#cross-reference-map)
- [Suggested reading order](#suggested-reading-order)
- [0. Quick diagnosis](#0-quick-diagnosis)
  - [0.1 Quick TL;DR](#01-quick-tldr)
  - [0.2 Symptom → likely bucket](#02-symptom--likely-bucket)
  - [0.3 First 5 minutes checklist](#03-first-5-minutes-checklist)
  - [0.4 Where to look first: status, hardware, logs](#04-where-to-look-first-status-hardware-logs)
- [1. Mental models](#1-mental-models)
  - [1.1 What a Space really is](#11-what-a-space-really-is)
  - [1.2 ZeroGPU and GPU Spaces](#12-zerogpu-and-gpu-spaces)
  - [1.3 What can break](#13-what-can-break)
  - [1.4 Storage mental model: ephemeral disk vs dataset repos vs Buckets](#14-storage-mental-model-ephemeral-disk-vs-dataset-repos-vs-buckets)
    - [1.4.1 Ephemeral disk](#141-ephemeral-disk)
    - [1.4.2 Dataset repos](#142-dataset-repos)
    - [1.4.3 Storage Buckets](#143-storage-buckets)
    - [1.4.4 A simple rule of thumb](#144-a-simple-rule-of-thumb)
  - [1.5 ZeroGPU 2026 update](#15-zerogpu-2026-update)
    - [1.5.1 Hardware and tiers](#151-hardware-and-tiers)
    - [1.5.2 Duration, queue priority, and quotas](#152-duration-queue-priority-and-quotas)
    - [1.5.3 Over-quota and account-type behavior](#153-over-quota-and-account-type-behavior)
    - [1.5.4 What changed from older guides](#154-what-changed-from-older-guides)
    - [1.5.5 A standing rule for HF libraries](#155-a-standing-rule-for-hf-libraries)
- [2. Core debugging workflows](#2-core-debugging-workflows)
  - [2.0 Formatting conventions used in this guide](#20-formatting-conventions-used-in-this-guide)
  - [2.1 A standard debugging loop](#21-a-standard-debugging-loop)
  - [2.2 Reproduce the problem with the smallest failing input](#22-reproduce-the-problem-with-the-smallest-failing-input)
  - [2.3 Simplify until a minimal version works](#23-simplify-until-a-minimal-version-works)
  - [2.4 Add logging that proves or disproves assumptions](#24-add-logging-that-proves-or-disproves-assumptions)
  - [2.5 Work locally and in Dev Mode when needed](#25-work-locally-and-in-dev-mode-when-needed)
  - [2.6 Debugging LLM-generated code](#26-debugging-llm-generated-code)
  - [2.7 Minimal example patterns](#27-minimal-example-patterns)
- [3. Build and startup failures](#3-build-and-startup-failures)
  - [3.1 Build never finishes or fails](#31-build-never-finishes-or-fails)
  - [3.2 Common root causes](#32-common-root-causes)
  - [3.3 A decision tree for “stuck on Building”](#33-a-decision-tree-for-stuck-on-building)
    - [3.3.1 Empty logs or only “Build queued”](#331-empty-logs-or-only-build-queued)
    - [3.3.2 Dependency / import failures](#332-dependency--import-failures)
    - [3.3.3 Clone or repo-history failures](#333-clone-or-repo-history-failures)
    - [3.3.4 Build succeeds but startup never becomes healthy](#334-build-succeeds-but-startup-never-becomes-healthy)
  - [3.4 Large downloads at build time vs startup time](#34-large-downloads-at-build-time-vs-startup-time)
  - [3.5 Platform-level incidents](#35-platform-level-incidents)
- [4. Runtime, API, and auth failures](#4-runtime-api-and-auth-failures)
  - [4.1 App crashes at runtime](#41-app-crashes-at-runtime)
  - [4.2 Python exceptions and stack traces](#42-python-exceptions-and-stack-traces)
  - [4.3 Dependency mismatches at runtime](#43-dependency-mismatches-at-runtime)
  - [4.4 HTTP / API errors](#44-http--api-errors)
  - [4.5 Authenticated vs unauthenticated calls on ZeroGPU](#45-authenticated-vs-unauthenticated-calls-on-zerogpu)
  - [4.6 Browser works but API fails](#46-browser-works-but-api-fails)
  - [4.7 Custom frontends, app-to-app calls, and identity-path pitfalls](#47-custom-frontends-app-to-app-calls-and-identity-path-pitfalls)
- [5. Storage and persistence](#5-storage-and-persistence)
  - [5.0 What belongs in this chapter vs elsewhere](#50-what-belongs-in-this-chapter-vs-elsewhere)
  - [5.1 Disk, cache, and local filesystem problems](#51-disk-cache-and-local-filesystem-problems)
    - [5.1.1 “No space left on device”](#511-no-space-left-on-device)
    - [5.1.2 HF_HOME, caches, and startup downloads](#512-hf_home-caches-and-startup-downloads)
    - [5.1.3 Why local fixes disappear after restart](#513-why-local-fixes-disappear-after-restart)
    - [5.1.4 Repo size vs runtime disk vs cache size](#514-repo-size-vs-runtime-disk-vs-cache-size)
  - [5.2 Persistent storage choices](#52-persistent-storage-choices)
    - [5.2.1 When ephemeral disk is enough](#521-when-ephemeral-disk-is-enough)
    - [5.2.2 When to use a dataset repo](#522-when-to-use-a-dataset-repo)
    - [5.2.3 When to use a Storage Bucket](#523-when-to-use-a-storage-bucket)
    - [5.2.4 Dataset repo vs Bucket: practical decision rules](#524-dataset-repo-vs-bucket-practical-decision-rules)
- [6. ZeroGPU deep dive](#6-zerogpu-deep-dive)
  - [6.0 What belongs in this chapter vs elsewhere](#60-what-belongs-in-this-chapter-vs-elsewhere)
  - [6.1 Hardware and tiers](#61-hardware-and-tiers)
    - [6.1.1 Blackwell-era compatibility check](#611-blackwell-era-compatibility-check)
  - [6.2 Duration, queue priority, and quotas](#62-duration-queue-priority-and-quotas)
  - [6.3 Over-quota and account-type behavior](#63-over-quota-and-account-type-behavior)
  - [6.4 Root-level CUDA placement vs import-time side effects](#64-root-level-cuda-placement-vs-import-time-side-effects)
  - [6.5 Why `torch.compile` is a bad fit for ZeroGPU](#65-why-torchcompile-is-a-bad-fit-for-zerogpu)
  - [6.6 AoTI as an advanced path](#66-aoti-as-an-advanced-path)
  - [6.7 Root-level CUDA is allowed, but bad side effects still happen](#67-root-level-cuda-is-allowed-but-bad-side-effects-still-happen)
  - [6.8 What changed from older guides](#68-what-changed-from-older-guides)
- [7. Cookbook: common problems and fixes](#7-cookbook-common-problems-and-fixes)
  - [7.0 How to read the Cookbook](#70-how-to-read-the-cookbook)
  - [7.1 `ModuleNotFoundError` / missing dependencies](#71-modulenotfounderror--missing-dependencies)
  - [7.2 Version conflicts and ABI issues](#72-version-conflicts-and-abi-issues)
  - [7.3 “Disk full” / cache pressure / startup downloads](#73-disk-full--cache-pressure--startup-downloads)
  - [7.4 Timeouts and slow responses](#74-timeouts-and-slow-responses)
  - [7.5 Authentication / private models / private datasets](#75-authentication--private-models--private-datasets)
  - [7.6 Where should I save outputs from a Space?](#76-where-should-i-save-outputs-from-a-space)
  - [7.7 How to debug “No space left on device”](#77-how-to-debug-no-space-left-on-device)
  - [7.8 Dataset repo vs Bucket: which one should I choose?](#78-dataset-repo-vs-bucket-which-one-should-i-choose)
  - [7.9 High-frequency writes, logs, checkpoints, and intermediate artifacts](#79-high-frequency-writes-logs-checkpoints-and-intermediate-artifacts)
  - [7.10 Space disk is ephemeral: what that means in practice](#710-space-disk-is-ephemeral-what-that-means-in-practice)
  - [7.11 Build succeeds, startup fails: what to check next](#711-build-succeeds-startup-fails-what-to-check-next)
  - [7.12 Empty logs or only “Build queued”](#712-empty-logs-or-only-build-queued)
  - [7.13 Space works in browser but fails through API](#713-space-works-in-browser-but-fails-through-api)
  - [7.14 ZeroGPU quota looks wrong or PRO seems ignored](#714-zerogpu-quota-looks-wrong-or-pro-seems-ignored)
  - [7.15 Duplicate works differently from the original Space](#715-duplicate-works-differently-from-the-original-space)
  - [7.16 Secrets were not copied during duplication](#716-secrets-were-not-copied-during-duplication)
  - [7.17 Gradio 6 messages format vs tokenizer input normalization](#717-gradio-6-messages-format-vs-tokenizer-input-normalization)
  - [7.18 `apply_chat_template()` crashes on list-based content](#718-apply_chat_template-crashes-on-list-based-content)
  - [7.19 `hf_transfer` advice is outdated: use Xet correctly](#719-hf_transfer-advice-is-outdated-use-xet-correctly)
  - [7.20 Xet environment variables that still matter](#720-xet-environment-variables-that-still-matter)
  - [7.21 `torch.compile` is not supported on ZeroGPU](#721-torchcompile-is-not-supported-on-zerogpu)
  - [7.22 AoTI on ZeroGPU: when it helps and when it hurts](#722-aoti-on-zerogpu-when-it-helps-and-when-it-hurts)
  - [7.23 Root-level CUDA is allowed, but eager side effects still break things](#723-root-level-cuda-is-allowed-but-eager-side-effects-still-break-things)
  - [7.24 Bucket for logs/checkpoints vs dataset repo for publishable artifacts](#724-bucket-for-logscheckpoints-vs-dataset-repo-for-publishable-artifacts)
  - [7.25 When to keep using a dataset repo instead of adopting Buckets](#725-when-to-keep-using-a-dataset-repo-instead-of-adopting-buckets)
  - [7.26 Authenticated vs unauthenticated ZeroGPU API calls](#726-authenticated-vs-unauthenticated-zerogpu-api-calls)
  - [7.27 Browser path vs custom frontend path](#727-browser-path-vs-custom-frontend-path)
  - [7.28 Build-time downloads vs startup-time downloads](#728-build-time-downloads-vs-startup-time-downloads)
  - [7.29 HF_HOME, cache growth, and disk pressure](#729-hf_home-cache-growth-and-disk-pressure)
  - [7.30 Before you trust any fix: check docs, changelog, and open issues](#730-before-you-trust-any-fix-check-docs-changelog-and-open-issues)
  - [7.31 How to navigate Hugging Face when you do not know where to start](#731-how-to-navigate-hugging-face-when-you-do-not-know-where-to-start)
  - [7.32 How to choose the right Hugging Face documentation page](#732-how-to-choose-the-right-hugging-face-documentation-page)
  - [7.33 How to search for the right Space, issue, or model](#733-how-to-search-for-the-right-space-issue-or-model)
  - [7.34 How to use this guide itself as context for an LLM](#734-how-to-use-this-guide-itself-as-context-for-an-llm)
  - [7.35 Datasets 4.x migration: dataset scripts, TorchCodec, and media pitfalls](#735-datasets-4x-migration-dataset-scripts-torchcodec-and-media-pitfalls)
  - [7.36 Transformers v5 migration: what breaks most often in real apps](#736-transformers-v5-migration-what-breaks-most-often-in-real-apps)
  - [7.37 ZeroGPU Blackwell / CUDA wheel mismatch](#737-zerogpu-blackwell--cuda-wheel-mismatch)
  - [7.38 Gradio event chains run after failure](#738-gradio-event-chains-run-after-failure)
  - [7.39 Hub upload hits 429 or 503](#739-hub-upload-hits-429-or-503)
- [8. Appendices](#8-appendices)
  - [8.0 How to use the appendices](#80-how-to-use-the-appendices)
  - [Appendix A. Deprecated or historical storage paths](#appendix-a-deprecated-or-historical-storage-paths)
    - [A.1 The old persistent storage setting](#a1-the-old-persistent-storage-setting)
    - [A.2 Why old permanent-local-storage advice is now a dead end](#a2-why-old-permanent-local-storage-advice-is-now-a-dead-end)
  - [Appendix B. ZeroGPU operational notes](#appendix-b-zerogpu-operational-notes)
    - [B.1 Root-level CUDA placement vs import-time CUDA side effects](#b1-root-level-cuda-placement-vs-import-time-cuda-side-effects)
    - [B.2 Why `torch.compile` is a bad fit for ZeroGPU](#b2-why-torchcompile-is-a-bad-fit-for-zerogpu)
    - [B.3 AoTI as an advanced exception path](#b3-aoti-as-an-advanced-exception-path)
    - [B.4 Duration, queue priority, and cold-start trade-offs](#b4-duration-queue-priority-and-cold-start-trade-offs)
    - [B.5 Blackwell-era CUDA wheel checks](#b5-blackwell-era-cuda-wheel-checks)
  - [Appendix C. Gradio / Transformers migration notes](#appendix-c-gradio--transformers-migration-notes)
    - [C.1 Gradio 6 content blocks](#c1-gradio-6-content-blocks)
    - [C.2 Why tokenizer inputs still often need plain strings](#c2-why-tokenizer-inputs-still-often-need-plain-strings)
    - [C.3 Common `apply_chat_template()` failure patterns](#c3-common-apply_chat_template-failure-patterns)
    - [C.4 Event-chain continuation: `.then()` vs `.success()`](#c4-event-chain-continuation-then-vs-success)
  - [Appendix D. Download stack and cache notes](#appendix-d-download-stack-and-cache-notes)
    - [D.1 `hf_transfer` is old advice now](#d1-hf_transfer-is-old-advice-now)
    - [D.2 Xet environment variables](#d2-xet-environment-variables)
    - [D.3 HF_HOME and cache placement](#d3-hf_home-and-cache-placement)
    - [D.4 Upload retries, 429, and 503](#d4-upload-retries-429-and-503)
  - [Appendix E. Duplicate Space troubleshooting](#appendix-e-duplicate-space-troubleshooting)
    - [E.1 What duplication copies](#e1-what-duplication-copies)
    - [E.2 What duplication does not copy](#e2-what-duplication-does-not-copy)
    - [E.3 Why a faithful duplicate still behaves differently](#e3-why-a-faithful-duplicate-still-behaves-differently)
  - [Appendix F. HF source taxonomy and update discipline](#appendix-f-hf-source-taxonomy-and-update-discipline)
    - [F.1 Official docs and changelog](#f1-official-docs-and-changelog)
    - [F.2 HF Staff-maintained Spaces](#f2-hf-staff-maintained-spaces)
    - [F.3 Blogs and posts](#f3-blogs-and-posts)
    - [F.4 Discussions and forum threads](#f4-discussions-and-forum-threads)
    - [F.5 GitHub issues](#f5-github-issues)
    - [F.6 Why you should always check the latest docs and issue trackers](#f6-why-you-should-always-check-the-latest-docs-and-issue-trackers)
    - [F.7 Support contact notes](#f7-support-contact-notes)
  - [Appendix G. Using LLMs as debugging assistants](#appendix-g-using-llms-as-debugging-assistants)
    - [G.1 What LLMs are good at](#g1-what-llms-are-good-at)
    - [G.2 What LLMs are bad at](#g2-what-llms-are-bad-at)
    - [G.3 A safe workflow for LLM-assisted debugging](#g3-a-safe-workflow-for-llm-assisted-debugging)
    - [G.4 Prompt patterns that work well for Spaces](#g4-prompt-patterns-that-work-well-for-spaces)
    - [G.5 Failure modes of LLM-assisted debugging](#g5-failure-modes-of-llm-assisted-debugging)
    - [G.6 How to use this guide itself with an LLM](#g6-how-to-use-this-guide-itself-with-an-llm)
  - [Appendix H. How to navigate Hugging Face when you do not know where information lives](#appendix-h-how-to-navigate-hugging-face-when-you-do-not-know-where-information-lives)
    - [H.1 A practical map of Hugging Face](#h1-a-practical-map-of-hugging-face)
    - [H.2 Where ZeroGPU information tends to appear first](#h2-where-zerogpu-information-tends-to-appear-first)
    - [H.3 Where storage and account issues tend to appear](#h3-where-storage-and-account-issues-tend-to-appear)
  - [Appendix I. Search and research basics for Spaces debugging](#appendix-i-search-and-research-basics-for-spaces-debugging)
    - [I.1 Turn failures into good search queries](#i1-turn-failures-into-good-search-queries)
    - [I.2 Search order](#i2-search-order)
    - [I.3 How to ask ChatGPT or Gemini useful questions](#i3-how-to-ask-chatgpt-or-gemini-useful-questions)
    - [I.4 When other AI communities can help](#i4-when-other-ai-communities-can-help)
  - [Appendix J. Curated external clue clusters](#appendix-j-curated-external-clue-clusters)
    - [J.1 Why keep external clue clusters at all?](#j1-why-keep-external-clue-clusters-at-all)
    - [J.2 Good external clue candidates](#j2-good-external-clue-candidates)
    - [J.3 How to use a clue cluster safely](#j3-how-to-use-a-clue-cluster-safely)
  - [Appendix K. Canonical search phrases for common failures](#appendix-k-canonical-search-phrases-for-common-failures)
    - [K.1 Build and startup](#k1-build-and-startup)
    - [K.2 Runtime and API](#k2-runtime-and-api)
    - [K.3 Storage and cache](#k3-storage-and-cache)
    - [K.4 ZeroGPU](#k4-zerogpu)
    - [K.5 Migration](#k5-migration)
- [9. Resources](#9-resources)
  - [9.0 How to use resources](#90-how-to-use-resources)
  - [9.1 Current docs to check first](#91-current-docs-to-check-first)
  - [9.2 Migration and change-tracking sources](#92-migration-and-change-tracking-sources)
  - [9.3 Live issue and forum sources](#93-live-issue-and-forum-sources)
  - [9.4 Curated link clusters by problem type](#94-curated-link-clusters-by-problem-type)
  - [9.5 Hugging Face navigation quick map](#95-hugging-face-navigation-quick-map)
  - [9.6 Support and contact notes](#96-support-and-contact-notes)
  - [9.7 Resource tiers: what to trust first](#97-resource-tiers-what-to-trust-first)
  - [9.8 Resource triage: keep, historical clue, or drop](#98-resource-triage-keep-historical-clue-or-drop)
  - [9.9 Example triage from the current link packs](#99-example-triage-from-the-current-link-packs)

## How to use this guide

### How to use the table of contents

Use the table of contents in one of two ways:

- Start with a top-level chapter if you already know the failure bucket.
- Start with a subsection if you already have a concrete symptom, migration issue, or search phrase.

The table of contents is synchronized to the current headings in this file and should be treated as the navigation baseline for the guide.

Choose the entry path that matches how you are working.

### 1. I just need the shortest route
Start here:
1. **Search index and common aliases**
2. **0. Quick diagnosis**
3. **Cross-reference map**
4. the matching **Cookbook** recipe

### 2. I want the official/default explanation first
Start here:
1. **0. Quick diagnosis**
2. the matching main chapter:
   - Build/startup
   - Runtime/API/auth
   - Storage/persistence
   - ZeroGPU
3. then the matching **Cookbook** recipe
4. then the relevant **Appendix** if migration or edge cases are involved

### 3. I already have an error string
Use:
1. **Search index and common aliases**
2. **Appendix K. Canonical search phrases**
3. **Resources 9.7 / 9.8 / 9.9**

### 4. I am attaching this file to an LLM
Attach:
- this guide,
- the failing logs or stack trace,
- `README.md` metadata,
- `requirements.txt`,
- and any reproduction notes.

Then ask the LLM to:
- classify the failure bucket,
- point to the most relevant sections,
- generate search phrases,
- and suggest only fixes consistent with current migration notes.

## How this large guide is layered

This guide is intentionally large. To keep it usable, each layer has a different job.

- **Quick diagnosis** tells you which failure bucket you are in.
- **Mental models** define the current assumptions and vocabulary.
- **Main body chapters** explain the official or default troubleshooting path.
- **Cookbook** translates recurring symptoms into concrete “what to check next” recipes.
- **Appendices** hold migration notes, operational quirks, historical dead ends, and search aids.
- **Resources** point outward to current docs, changelogs, issues, forum threads, and clue clusters.

When the same topic appears in more than one place, read it this way:
- **main body** for the default explanation,
- **Cookbook** for concrete symptom-driven action,
- **Appendix** for edge cases, history, and exceptions,
- **Resources** for live external follow-up.

## Label conventions

This guide uses four labels consistently.

- **Current**: the current default guidance or the current official model
- **Operational note**: practical nuance that matters in real debugging
- **Historical**: older behavior or context that may still help explain a failure
- **Dead end**: advice that still circulates but should not be followed as a default path

Use these labels to separate live guidance from migration context and historical traps.

## Search index and aliases

Use this section when you are searching the file with `Ctrl+F`, feeding it to an LLM, or trying to route a raw error to the right chapter quickly.

### Build / startup phrases
Search for these phrases when the Space never becomes healthy:
- `Building`
- `Build queued`
- `stuck on Building`
- `startup never becomes healthy`
- `empty logs`
- `Job failed with exit code 1`
- `repo history`
- `clone failure`

Main destinations:
- **3. Build and startup failures**
- **7.11 / 7.12 Cookbook**
- **Appendix E**
- **Resources 9.9**

### Runtime / API / auth phrases
Search for these phrases when browser and API behavior differ:
- `No API found`
- `browser works but API fails`
- `private Space token`
- `api_visibility`
- `footer_links`
- `authenticated vs unauthenticated`
- `custom frontend`
- `identity path`

Main destinations:
- **4. Runtime, API, and auth failures**
- **7.13 / 7.26 / 7.27 Cookbook**
- **Appendix C**
- **Resources 9.9**

### Storage / cache / disk phrases
Search for these phrases when files disappear or disk pressure grows:
- `No space left on device`
- `HF_HOME`
- `cache growth`
- `ephemeral disk`
- `dataset repo vs Bucket`
- `storage bucket`
- `local fixes disappear`
- `startup downloads`

Main destinations:
- **5. Storage and persistence**
- **7.6 / 7.7 / 7.8 / 7.29 Cookbook**
- **Appendix A / D**
- **Resources 9.9**

### ZeroGPU phrases
Search for these phrases when GPU behavior is the main variable:
- `ZeroGPU`
- `quota exceeded`
- `PRO seems ignored`
- `duration`
- `queue priority`
- `CUDA must not be initialized in the main process`
- `torch.compile`
- `AoTI`
- `root-level CUDA`
- `import-time side effects`

Main destinations:
- **1.5 ZeroGPU 2026 update**
- **6. ZeroGPU deep dive**
- **7.14 / 7.21 / 7.22 / 7.23 Cookbook**
- **Appendix B**
- **Resources 9.9**

### Migration phrases
Search for these phrases when something used to work and stopped after an upgrade:
- `Gradio 6`
- `Transformers v5`
- `huggingface_hub v1.0`
- `hf_transfer`
- `Xet`
- `additional_chat_templates`
- `dataset scripts are no longer supported`
- `TorchCodec`

Main destinations:
- **Appendix C / D**
- **7.17 / 7.18 / 7.19 / 7.35 / 7.36 Cookbook**
- **Resources 9.7**

## Cross-reference map

Use this map when the guide feels large and you need the shortest route.

### If the Space is stuck on `Building`
Read in this order:
1. **0. Quick diagnosis**
2. **3. Build and startup failures**
3. **7.11 / 7.12 Cookbook**
4. **Appendix E. Duplicate Space troubleshooting** if the problem started after duplication
5. **Resources** for status, forum threads, and recent build regressions

### If the Space is `Running` but acts wrong
Read in this order:
1. **0. Quick diagnosis**
2. **4. Runtime, API, and auth failures**
3. **5. Storage and persistence** if files disappear, cache explodes, or outputs are missing
4. **6. ZeroGPU deep dive** if the failure depends on GPU path or queue/quota behavior
5. **7. Cookbook** for symptom-specific recipes

### If browser works but API fails
Read:
1. **4.5 Authenticated vs unauthenticated calls on ZeroGPU**
2. **4.6 Browser works but API fails**
3. **4.7 Custom frontends, app-to-app calls, and identity-path pitfalls**
4. **7.13 / 7.26 / 7.27 Cookbook**
5. **Appendix C** if the failure appears after a Gradio migration

### If the issue appeared after an upgrade or restart
Read:
1. **1.5 ZeroGPU 2026 update**
2. **6.8 What changed from older guides**
3. **Appendix A** for deprecated or dead-end assumptions
4. **Appendix C / D** for Gradio, Transformers, Xet, and cache migrations
5. **Resources 9.7** for migration guides and changelogs

### If you do not know where the relevant information lives
Read:
1. **Appendix H. How to navigate Hugging Face**
2. **Appendix I. Search and research basics**
3. **Resources 9.6 / 9.7**
4. **Cookbook 7.31 / 7.32 / 7.33 / 7.34**

## Suggested reading order

Use this order unless you already know exactly where the failure belongs.

1. **Search index and common aliases**
2. **0. Quick diagnosis**
3. **Cross-reference map**
4. One main chapter
5. One Cookbook recipe
6. One Appendix only if needed
7. Resources only after you know what you are looking for

This avoids getting lost in links too early.

## 0. Quick diagnosis

### 0.1 Quick TL;DR

When a Space breaks, classify it first. In practice, most failures fit one of these six buckets.

1. **Build never finishes or fails**  
   The Space stays on `Building...`, or the build logs fail during dependency installation, cloning, package compilation, or Docker steps. Start with `requirements.txt`, `packages.txt`, `Dockerfile`, repository history, and platform status.

2. **App crashes at runtime**  
   The Space reaches `Running`, but the UI is blank, returns 500 errors, or the container keeps restarting. Start with runtime logs and the first useful stack trace.

3. **HTTP / API errors**  
   The browser path works, but the API path fails, or you see 4xx / 5xx responses. Private Spaces require a token with the right permissions, and API failures often involve schema, auth, or path mismatches.

4. **ZeroGPU quota / queue / auth-path confusion**  
   The quota seems too small, PRO seems ignored, or behavior changes depending on whether the request is authenticated or routed through a custom frontend. On ZeroGPU, quota, queue priority, and request path all matter.

5. **Disk, cache, and storage-destination mistakes**  
   You see `No space left on device`, caches keep growing, or files written locally disappear after a restart. Local Space disk is ephemeral, so durable outputs need a different destination.

6. **Platform-level incidents**  
   Multiple Spaces become unstable at the same time, or even a minimal Space fails without any relevant code change. In that case, suspect the platform before you over-debug your own app.

### 0.2 Symptom → likely bucket

Use this table as a fast first-pass map.

| Symptom | Likely bucket | First place to look |
|---|---|---|
| `Building...` takes unusually long | Build / startup | build logs, queue state, repo history |
| Build succeeds, but the Space never becomes healthy | Startup health | startup logs, preload behavior, timeout |
| Browser works, but the API fails | API / auth | token, endpoint path, auth scope |
| `quota exceeded` appears too early | ZeroGPU auth / quota path | logged-in browser path, account tier, frontend path |
| `No space left on device` | Disk / cache | HF cache, startup downloads, temp files |
| The duplicate fails, but the original works | Runtime contract drift | hardware, secrets, YAML, dependency pins |

### 0.3 First 5 minutes checklist

The first five minutes should be consistent.

1. **Check the stage**  
   Confirm whether the Space is `BUILDING`, `RUNNING`, or `FAILED`. That tells you whether to follow the build path or the runtime path.

2. **Check hardware and runtime assumptions**  
   Confirm whether the Space is using CPU, a dedicated GPU, or ZeroGPU. ZeroGPU is Gradio-only and has tier-specific quota and runtime behavior.

3. **Check secrets and auth**  
   If the Space depends on private models, private datasets, or private API access, missing tokens and secrets are a top-priority suspicion. Private Space API access requires the right token permissions.

4. **Check the README YAML**  
   Review `sdk`, `python_version`, `sdk_version`, `suggested_hardware`, `preload_from_hub`, and related settings. Duplication and later rebuilds often expose drift here.

5. **Do the first reproduction on the standard Hugging Face Space page**  
   For ZeroGPU in particular, the first test should be the logged-in standard Space page, not a custom frontend or app-to-app path. That makes auth-path and identity-path issues easier to isolate.

6. **Start with the smallest successful workload**  
   Do not begin with the largest image, the longest prompt, or the heaviest generation settings. ZeroGPU behavior depends on declared duration, queueing, and quota, so smallest-first testing is more reliable.

### 0.4 Where to look first: status, hardware, logs

Look at three things first.

#### Runtime status
Check whether you are dealing with a build problem, a startup health problem, or a runtime problem. Mixing those layers wastes time.

#### Hardware and storage assumptions
CPU, dedicated GPU, and ZeroGPU do not fail in the same way. ZeroGPU adds queue, duration, and quota behavior. Storage assumptions matter too, because local disk is ephemeral.

#### Logs
At minimum, separate logs into three layers:

- **build logs**  
  Look for dependency installation failures, cloning failures, OS package failures, and Docker build failures.
- **runtime logs**  
  Look for import-time errors, model-loading failures, startup health failures, and early crashes.
- **application logs**  
  Look for your own timings, branches, input sizes, request metadata, and device-placement logs.

A lot of debugging gets easier as soon as you stop mixing those three layers.

---

## 1. Mental models

### 1.1 What a Space really is

A Hugging Face Space is not just a web app. At minimum, it is four layers at once.

- **Git repository**  
  It stores code, configuration, and lightweight assets.
- **Managed runtime**  
  Hugging Face builds, deploys, and health-checks it.
- **Selected hardware**  
  It runs on CPU, a dedicated GPU, or a shared GPU runtime such as ZeroGPU.
- **One or more access paths**  
  It can be used through the browser UI, an API endpoint, an embedded path, or a custom frontend.

If you misidentify which layer is failing, you can spend a long time changing application code when the real problem is configuration, runtime drift, or platform behavior.

### 1.2 ZeroGPU and GPU Spaces

ZeroGPU is not just “free GPU access.” It is a different runtime model.

ZeroGPU uses shared infrastructure that dynamically allocates and releases GPU capacity as needed. Current docs describe NVIDIA RTX Pro 6000 Blackwell-backed ZeroGPU sizes rather than the older H200 wording. A dedicated GPU Space keeps a specific GPU attached. A ZeroGPU Space does not. ZeroGPU is also tied to a narrower runtime model, with queueing, duration declarations, supported-version boundaries, and quota behavior that do not behave like a normal dedicated GPU Space.

In practice, this means ZeroGPU has a few properties that are easy to underestimate:

- queue and quota affect user-visible behavior,
- `@spaces.GPU` duration affects scheduling and quota perception,
- request path and authentication path can change observed behavior,
- `torch.compile` is not the normal path, while AoTI is an advanced exception path.

Treat ZeroGPU as a specific managed runtime, not as a simple GPU checkbox.

### 1.3 What can break

Most failures eventually reduce to five broad classes.

1. **Dependency / build problems**  
   `pip install`, cloning, package compilation, or Docker build steps fail.
2. **Runtime logic problems**  
   The code starts, but request handling fails with type, shape, state, or control-flow errors.
3. **Auth / path problems**  
   The browser path works, but the API path fails, or a custom frontend behaves differently from the default path.
4. **Disk / storage design problems**  
   Caches grow, outputs disappear, startup downloads fill disk, or the wrong storage destination is used.
5. **ZeroGPU-specific operational problems**  
   Quota, duration, queue, auth path, or compile strategy become the real source of failure.

This classification is not perfect, but it is usually good enough to tell you where to start.

### 1.4 Storage mental model: ephemeral disk vs dataset repos vs Buckets

#### 1.4.1 Ephemeral disk

Every Space has local disk, but that disk is **ephemeral**. If the Space restarts or stops, local files can disappear.

That makes local disk appropriate for temporary downloads, decompression, caches, and short-lived intermediate files. It is the wrong place to treat as the only durable store for outputs you care about later.

#### 1.4.2 Dataset repos

A dataset repo is a **versioned, shareable Hub repository**. It is a good default when you want to keep outputs, share them, revisit them later, or make them easy to consume with Hub and `datasets` tooling.

In practice, dataset repos fit “publishable” or “reusable” artifacts better than ephemeral local disk does.

#### 1.4.3 Storage Buckets

A Storage Bucket is **S3-like object storage**, not a git-based repo. It is designed for non-versioned, mutable, operational storage.

That makes Buckets a good fit for training checkpoints, logs, intermediate artifacts, and other large working sets where mutability and operational convenience matter more than repository history.

#### 1.4.4 A simple rule of thumb

When in doubt, use this rule.

- **Temporary work** → ephemeral disk  
- **Shareable, versioned, reusable outputs** → dataset repo  
- **Fast mutable working storage, logs, checkpoints** → Bucket

### 1.5 ZeroGPU 2026 update

#### 1.5.1 Hardware and tiers

ZeroGPU documentation now describes NVIDIA RTX Pro 6000 Blackwell-backed `large` and `xlarge` sizes.

- `large` is the default size and maps to half an NVIDIA RTX Pro 6000 Blackwell with 48 GB VRAM.
- `xlarge` maps to a full NVIDIA RTX Pro 6000 Blackwell with 96 GB VRAM and costs 2× quota.

This matters because old H200-era assumptions can hide a real hardware / CUDA / PyTorch-wheel mismatch. If a Space suddenly fails with a CUDA kernel error after rebuild, duplication, or hardware movement, do not debug the model first. Check the effective ZeroGPU hardware, the installed PyTorch version, and whether the CUDA wheel supports the active GPU architecture, including Blackwell-class `sm_120` devices when that capability is visible in logs or runtime probes.

At the time of this update, the ZeroGPU docs list PyTorch support beginning at `2.8.0` and include `2.8.0`, `2.9.1`, `2.10.0`, and `2.11.0` in the supported-version list. Treat those versions as part of the hosted runtime contract.

#### 1.5.2 Duration, queue priority, and quotas

The default `@spaces.GPU` duration is 60 seconds. If your function is likely to run longer, you can declare a custom duration. Shorter declared duration can improve queue priority. Dynamic duration is also supported.

In practice, perceived ZeroGPU behavior depends not only on “GPU seconds,” but also on declared duration and queue behavior.

#### 1.5.3 Over-quota and account-type behavior

ZeroGPU daily quota depends on account tier. Authenticated and unauthenticated paths do not behave the same way, and higher tiers have higher daily limits. PRO and above may also have over-quota options, so quota behavior should be checked against current docs, not memory.

#### 1.5.4 What changed from older guides

Older guides often mislead readers in four ways.

- **They assume the older ZeroGPU hardware wording**  
  Recheck any guide that assumes H200-backed ZeroGPU, A100-like CUDA behavior, or old PyTorch wheels. Current ZeroGPU docs describe RTX Pro 6000 Blackwell-backed sizes.
- **They assume persistent local storage**  
  The safe default now is to treat local disk as ephemeral.
- **They assume the old download stack**  
  `huggingface_hub` v1.0 moved away from older assumptions such as `Repository`-based advice and older transfer guidance.
- **They mix UI-layer schemas with tokenizer-layer schemas**  
  Gradio 6 changed component-level message formats, while many tokenizer and chat-template examples still assume plain string content.

#### 1.5.5 A standing rule for HF libraries

Hugging Face libraries and services change quickly. That is especially true for Gradio major upgrades, `huggingface_hub` migration, Transformers chat templating, and ZeroGPU operational guidance.

As a standing rule, check current docs and migration guides first. Then check current issues, discussions, and forum threads. Do not trust older blog posts, older forum answers, or stale generated summaries by default.

## 2. Core debugging workflows

### 2.0 Formatting conventions used in this guide

The guide follows a simple formatting rule.

- Use `code formatting` for statuses, parameters, environment variables, error strings, and API paths.
- Use **bold** for decision words, label words, and contrasts that matter operationally.
- Use `-` bullets for unordered lists.
- Use numbered lists only when order changes the debugging path.

This keeps the guide easier to scan and easier to translate.

Use this chapter for repeatable debugging habits that apply across build, runtime, API, storage, and ZeroGPU issues.

The goal is not to memorize tricks. The goal is to reuse the same small loop until the failure becomes narrow enough to explain.

### 2.1 A standard debugging loop

Use this loop consistently. Start simple, reduce variables, and repeat.

#### Step 1: Identify the failure type

- Use the **status** (build vs runtime).
- Use the **symptom** (HTTP 4xx/5xx, crash, blank UI).
- Use the **logs** to confirm which path is failing.

#### Step 2: Reproduce the problem in the simplest possible way

- Trigger the app exactly as the user would:
  - Send the same input via the UI.
  - Call the same HTTP endpoint with the same payload (using `curl` / `requests`).
- Try to isolate a **minimal input** that causes the failure.
  - If a long prompt breaks, test shorter prompts.
  - If a large image breaks, test with a tiny image.

#### Step 3: Simplify the code until it works

- Comment out or remove:
  - Optional features.
  - Unnecessary logging.
  - Extra calls to other APIs.
- Replace the main function with a **minimal script**:
  - Load the model.
  - Run one simple inference.
  - Return a constant or very simple result.

If this minimal version fails, you know the problem is in the **environment, dependencies, or core model logic**, not in your UI wrappers.

#### Step 4: Add logging to confirm assumptions

- Log:
  - Inputs (in a safe and privacy-preserving way).
  - Shapes and dtypes of tensors.
  - Device placement (CPU/GPU).
  - Start/end of each major step.
- Make logs **specific**:
  - “Loading model...” vs “Loading `meta-llama/Llama-3-8B-Instruct` using `AutoModelForCausalLM` on device `cuda`”.

#### Step 5: Work locally and in Dev Mode when needed

- Reproduce issues locally using:
  - A small **conda** / **venv** environment with same `requirements.txt`.
  - Similar Python versions.
- Use **Dev Mode** to connect via SSH or VS Code to the live Space:
  - Inspect files and logs with your usual tools.
  - Run commands interactively in the same environment the Space uses.

#### Step 6: Rebuild, test, and document

- Commit small changes and redeploy.
- Re-run:
  - Minimal test.
  - One or two realistic test cases.
- Keep a short note:
  - “Problem → Hypothesis → Experiment → Result → Fix”.
- This history helps when the same problem appears again months later.

---

### 2.2 Reproduce the problem with the smallest failing input

The fastest way to waste time is to reproduce with the *largest* input first. For ZeroGPU and API troubleshooting, always start with the smallest prompt, smallest image, smallest audio file, or smallest configuration that still triggers the bug. This reduces queue time, lowers quota usage, and separates payload-size issues from infrastructure issues.

### 2.3 Simplify until a minimal version works

A good minimal reproduction is not a separate goal. It is the shortest path to learning whether the problem is in your app logic, your environment, your configuration, or the platform. Remove optional acceleration, custom frontends, chained API calls, analytics, and extra file handling until the core path either works or fails in an obvious way.

### 2.4 Add logging that proves or disproves assumptions

Prefer logs that answer a yes/no question. Examples:

- Did startup reach model load?
- Did the request path use an authenticated call?
- Did a download happen during build or only during startup?
- Did the code path normalize Gradio 6 content blocks before `apply_chat_template()`?

### 2.5 Work locally and in Dev Mode when needed

Dev Mode allows you to:

- **SSH into the Space runtime**.
- Connect via **VS Code** remote development.
- Inspect files, run commands, and debug interactively **inside** the same environment as your Space.

Typical workflow:

1. Enable **Dev Mode** for your Space.
2. Connect via:
   - SSH (terminal).
   - VS Code Remote SSH or the dedicated integration.
3. In the remote shell:
   - Run `python` scripts manually.
   - Inspect `pip list` and confirm versions.
   - Tail logs and add new logging.
4. Once you identify the fix:
   - Update the repository.
   - Disable Dev Mode if not needed.

Dev Mode is especially valuable for:

- Subtle environment issues.
- Debugging large models that behave differently in local vs remote environments.
- Inspecting disk usage and cache files.

---

### 2.6 Debugging LLM-generated code

LLM-generated Space code is powerful but:

- It often **assumes** libraries or APIs that are not installed.
- It may mix patterns from different frameworks.
- It may silently ignore errors or implement fragile logic.

Treat LLM code like code from a junior collaborator:

1. **Read it end to end**
   - Understand the high-level flow:
     - Input → preprocessing → model → postprocessing → output.
   - Ensure it matches what you want the app to do.

2. **Check imports and dependencies**
   - Compare imports with your `requirements.txt`.
   - Remove unused imports and unused dependencies to minimize build complexity.

3. **Isolate the core model call**
   - Extract the “model inference” into a small function or script.
   - Run that alone locally and/or in Dev Mode.
   - Confirm:
     - Inputs.
     - Outputs.
     - Performance and memory.

4. **Rewrite suspicious sections**
   - Replace overly clever abstractions with simple, explicit code.
   - Add defensive checks:
     - Input validation.
     - Timeouts.
     - Clear error messages.

5. **Ask the LLM for smaller, targeted snippets**
   - Instead of “write the whole app”, ask:
     - “Write a small function that converts this JSON input into a list of token IDs.”
     - “Write a simple Gradio interface that calls this already-working function.”

The more you understand and simplify LLM-generated code, the easier it is to debug.

---

### 2.7 Minimal example patterns

Having a simple, known-good Space is extremely useful.

### 2.7.1 Minimal Gradio “ping” Space

A tiny Space that just echoes input lets you:

- Verify basic deployments.
- Confirm build, runtime, and logs all behave as expected.

Example:

```python
# Minimal Gradio echo app
# Docs: https://gradio.app/getting_started/
import gradio as gr  # pip install gradio

def echo(text: str) -> str:
    return f"Echo: {text}"

demo = gr.Interface(fn=echo, inputs="text", outputs="text", title="Echo Space")

if __name__ == "__main__":
    demo.launch()
```

Use this for:

- Testing new hardware tiers.
- Checking if the platform is healthy.
- Confirming basic configuration like port, runtime, etc.

### 2.7.2 Minimal HF Transformers inference snippet

A tiny model call you can test locally and in Dev Mode:

```python
# Minimal Transformers text completion
# Docs: https://huggingface.co/docs/transformers/index
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "gpt2"  # replace with your model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

def generate(prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(generate("Hello, Spaces! "))
```

Once this minimal snippet works:

- Wire it into Gradio.
- Add logging and error handling.
- Only then add more complex logic.

---

## 3. Build and startup failures

Use this chapter when a Space stays on `Building...`, fails during rebuild, or never becomes healthy after the image build completes.

The key distinction is simple: some failures happen **during build**, while others happen **after build** when the app still cannot become ready. That distinction decides which logs and which fixes matter first.

### 3.1 Build never finishes or fails

Treat this as a build/startup problem first, not as an application-logic problem.

In Spaces, a commit triggers a rebuild and restart, and the README YAML controls runtime parameters such as `python_version`, `sdk_version`, and startup behavior. That means the same repository can behave differently after duplication or after time has passed, even when the visible app code did not change. ([Hugging Face](https://huggingface.co/docs/hub/spaces-overview), [Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

The first question is simple: **did the build itself fail, or did the app fail to become healthy after the build completed?** If the logs stop during dependency installation, cloning, package compilation, or Docker steps, you are still in the build phase. If the image builds successfully but the Space never stabilizes as `Running`, you are usually dealing with startup health, late downloads, bad runtime assumptions, or a platform issue. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

### 3.2 Common root causes

The most common causes are still the unglamorous ones.

First, **dependency installation failures**. These include missing wheels, incompatible Python versions, binary-extension build failures, and version conflicts that install in one environment but not another. Since Spaces uses the `python_version` declared in the README YAML during build, a Python-version mismatch is often enough to break a rebuild. The default Python version is `3.10` if you do not specify one. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

Second, **large downloads at the wrong time**. A Space can build successfully and still fail during startup because it pulls large models or assets only after the container starts. The `preload_from_hub` setting exists to move selected Hub downloads into build time instead of runtime. Use it when startup health or first-request latency is dominated by cold downloads. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

Third, **bad repository shape or history baggage**. A duplicate preserves Git history for Spaces, and a repository with large-file baggage, broken LFS/Xet history, or storage-limit issues can fail in ways that do not show up in the current `app.py` alone. In practice, clone failures and strange rebuild behavior often come from repository shape rather than current runtime logic. ([Hugging Face](https://huggingface.co/docs/hub/spaces-overview), [Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

Fourth, **startup health failures after a successful build**. The README YAML supports `startup_duration_timeout`, which defaults to `30m`. If the app needs longer to become healthy because of model loading, slow initialization, or late downloads, the Space can be marked unhealthy even though the image itself built correctly. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

### 3.3 A decision tree for “stuck on Building”

Use this as a strict order of operations.

#### 3.3.1 Empty logs or only “Build queued”

If logs are empty, mostly queue noise, or stop at something like `Build queued`, do **not** jump straight into rewriting dependencies. Treat this as a possible scheduler, capacity, or platform-state issue first. Before changing code, check whether other users are reporting similar symptoms, whether multiple Spaces are affected, and whether a minimal Space on the same hardware behaves the same way. The practical reason is simple: a platform queueing problem and a broken `requirements.txt` can look similar from the UI but require completely different responses. ([Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

In this bucket, the cheapest actions are: one restart, one factory rebuild, and one minimal reproduction check. If those do not change anything and the logs remain empty, stop treating it like an app-code issue until you have evidence that it is one. ([Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

#### 3.3.2 Dependency / import failures

If logs clearly show `pip` failures, missing system packages, extension-build errors, or import-time crashes, stay in the dependency lane. Compare `python_version`, `sdk_version`, `requirements.txt`, `pre-requirements.txt`, and `packages.txt` before you touch app logic. For Gradio Spaces, the YAML may also be selecting a different Gradio version than you expect. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference), [Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

A useful rule is: **pin first, simplify second, rewrite last**. If a duplicate or rebuild suddenly breaks without a deliberate app-code change, version drift is more likely than a new logic bug. ([Hugging Face Forums](https://discuss.huggingface.co/t/duplicated-space-not-working-as-original/175072))

#### 3.3.3 Clone or repo-history failures

If the Space fails while cloning, checking out files, or restoring repository contents, suspect repository shape and history. Space duplication preserves full Git history, so old large-file baggage can follow a duplicate even when the current commit looks light. Errors about repository storage limits, cloning, or repository state belong here. ([Hugging Face](https://huggingface.co/docs/hub/spaces-overview), [Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

The practical fixes in this bucket are different from dependency fixes: slim the repository, move heavy assets to model or dataset repos, or create a cleaner reproduction that does not inherit the same history burden. ([Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

#### 3.3.4 Build succeeds but startup never becomes healthy

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 3.3.5 503 after restart: split build, crash, and health

A `503 Service Unavailable` is an outer symptom, not a root cause. Split it before applying fixes:

1. **Build fails** — dependency installation, clone, package, OS-level, or wheel problem.
2. **Container starts then crashes** — app import error, missing env/secrets, runtime exception, or process exits.
3. **Container runs but never becomes healthy** — wrong host/port, late model download, slow startup, missing listener, Docker `app_port` mismatch, or healthcheck timeout.

For Docker and custom stacks, also check whether the app binds the expected interface and port. For large models, check whether files should move from startup-time download to `preload_from_hub` or another build-time/cache strategy.

Safe wording: “503 means the Space endpoint is unavailable; it does not tell us whether the failure is build, container crash, or healthcheck.”

Primary references:
- https://huggingface.co/docs/hub/spaces-config-reference
- https://discuss.huggingface.co/t/my-space-was-paused-and-503-when-restart/171474
- https://discuss.huggingface.co/t/launch-timed-out-space-was-not-healthy-after-30-min-space-keeps-on-building/63685

If the image builds but the Space never stabilizes, treat this as a startup-health problem. Common causes are late downloads, import-time model loading that takes too long, incorrect startup assumptions, or health-check failure because the app never becomes ready inside the allowed startup window. `startup_duration_timeout` and `preload_from_hub` are the first two config knobs to check. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

Also check whether you are debugging a real duplicate or an accidental migration. A duplicate created on free CPU instead of matching the original hardware is no longer a faithful reproduction of the source runtime contract. ([Hugging Face](https://huggingface.co/docs/hub/spaces-overview), [Gradio](https://www.gradio.app/docs/python-client/client))

### 3.4 Large downloads at build time vs startup time

Do not treat all downloads the same. Some downloads belong in build time, some do not.

If the app needs a known set of Hub assets to become ready, `preload_from_hub` can move those downloads into build time so startup is less fragile. This is especially useful when a Space looks healthy only after the first warm run or when a duplicate behaves worse simply because it has a cold cache while the source Space had already stabilized long ago. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

By contrast, pulling large files lazily during startup or on the first user request increases the chance of startup timeout, first-request latency spikes, and misleading “works sometimes” behavior. In other words: **if startup always needs it, prefer preloading; if only some code paths need it, load it later and consciously**. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

### 3.5 Platform-level incidents

Sometimes the issue is not your Space.

If multiple unrelated Spaces fail at the same time, if a minimal Space on the same hardware also fails, or if your logs stay empty despite rebuilds and restarts, platform conditions become the primary suspect. In that case, do not burn time rewriting code until you have ruled out an incident or capacity problem. The forum guidance for “stuck on Building” explicitly treats this as a first-class branch of the decision tree, not as an afterthought. ([Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

The right escalation package here is concise: Space URL, hardware type, exact timestamps, stage transitions, and whether a minimal reproduction on the same tier also fails. That is much more useful than a long description of speculative code changes. ([Hugging Face Forums](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119))

## 4. Runtime, API, and auth failures

Use this chapter when the Space reaches `Running`, but real usage still fails.

This includes blank UIs, repeated restarts, 4xx / 5xx API responses, browser-path vs API-path mismatches, and auth-path problems that look like model failures at first glance.

### 4.1 App crashes at runtime

A runtime failure means the Space reaches `Running`, but real usage still breaks.

That includes blank UIs, repeated restarts, 500 responses, stuck interactions, or errors that appear only after a request reaches application code. This is a different class of problem from build failure, even when both feel like “the Space is broken.”

Start by asking where the failure appears: in the **browser path**, the **API path**, or only in a **custom or embedded path**. In Spaces, those paths can differ in authentication context, headers, quota handling, and how frontend state reaches the backend. ([Hugging Face](https://huggingface.co/docs/hub/spaces-api-endpoints))

### 4.2 Python exceptions and stack traces

The basic rule still holds: find the **first useful stack trace**, then find the **topmost line in your own code**, then inspect the exact state at that point. Type mismatches, shape mismatches, missing environment variables, and bad path assumptions still explain a large share of runtime crashes.

In 2026, one especially common source of confusion is message-shape drift across libraries. A Gradio 6 chat UI may hand you structured content blocks, while your tokenizer or prompt builder still expects a plain string. If you do not normalize at the app boundary, a runtime crash can surface far downstream in templating or tokenization code. ([Gradio](https://www.gradio.app/guides/gradio-6-migration-guide), [Hugging Face](https://huggingface.co/docs/transformers/chat_templating))

### 4.3 Dependency mismatches at runtime

Some dependency problems do not fail the build. They install successfully and fail only when imported or exercised.

This is common when one library changed its API surface, when one layer now emits a different data contract than another expects, or when old examples still assume a pre-migration schema.

Three modern examples matter here.

First, **Gradio major-version changes**. Gradio 6 migration guidance documents breaking changes in chat interfaces and related behavior, so code written against Gradio 4/5 assumptions can build and then fail only when the chat path is executed. ([Gradio](https://www.gradio.app/guides/gradio-6-migration-guide), [Gradio](https://www.gradio.app/changelog))

Second, **Transformers chat templating**. `apply_chat_template()` behavior depends on the model’s template contract, and many templates still assume plain-string `content`. Passing list-based structured content directly can fail in ways that look like model bugs but are really schema mismatches. ([Hugging Face](https://huggingface.co/docs/transformers/chat_templating))

Third, **download-stack changes in `huggingface_hub`**. Old advice around `hf_transfer` is no longer current in the v1.0 migration era, and environment-variable assumptions may be stale if you are debugging download behavior, cold starts, or cache placement. ([Hugging Face](https://huggingface.co/docs/huggingface_hub/concepts/migration), [Hugging Face](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables))

### 4.4 HTTP / API errors

Keep the basic distinction simple.

- **4xx** usually means the request is wrong, incomplete, unauthorized, or pointed at the wrong path.
- **5xx** usually means the request reached the server and the server-side path failed, timed out, or crashed.

For 4xx, validate the URL, the endpoint shape, the token scope, and the request body. For 5xx, inspect the runtime logs immediately and match timestamps. The same minimal reproduction principle applies here: reproduce with the smallest possible `gradio_client`, `requests`, or `curl` call before changing your application logic. ([Hugging Face](https://huggingface.co/docs/hub/spaces-api-endpoints))

### 4.5 Authenticated vs unauthenticated calls on ZeroGPU

This distinction matters more than many users expect.

For ZeroGPU Spaces, authenticated requests consume **your account’s** GPU quota. Unauthenticated requests use a **shared pool with stricter limits**. The current docs also list different included daily quotas by account type and explain that PRO, Team, and Enterprise users can extend beyond the included daily quota with pre-paid credits. ([Hugging Face](https://huggingface.co/docs/hub/spaces-api-endpoints), [Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

That means a Space can appear to “have broken quotas” when the real issue is simply that the request path is no longer authenticated the way the browser path is. This is why the first clean test for ZeroGPU should be the standard Hugging Face Space page while logged in. ([Hugging Face](https://huggingface.co/docs/hub/spaces-api-endpoints))

### 4.6 Browser works but API fails

Treat this as an auth-or-path problem before you treat it as a model problem.

Private Spaces require a token with READ permissions. If the browser works but API calls fail, check whether the browser session is giving you authentication context that your API client is not replicating. Also check organization token policy if you are working inside an org with stricter rules around which token types are accepted. ([Hugging Face](https://huggingface.co/docs/hub/spaces-api-endpoints), [Hugging Face](https://huggingface.co/docs/hub/security-tokens))

A good rule is: **prove the Space works in the browser first, then prove the same payload works through an authenticated API client, then debug any custom path**. That order avoids mixing model failures with identity failures.

### 4.7 Custom frontends, app-to-app calls, and identity-path pitfalls

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 4.7.1 ZeroGPU identity path checklist

When ZeroGPU quota looks wrong, classify the request path before classifying the quota.

Check the path in this order:

1. normal `huggingface.co/spaces/...` page while logged in,
2. direct `*.hf.space` URL,
3. `gradio_client` or HTTP API call,
4. custom frontend,
5. app-to-app call,
6. third-party website that wraps or proxies a Space.

For ZeroGPU, the rate limit can depend on an `X-IP-Token` request header inserted by Hugging Face infrastructure. If that identity path is missing or not forwarded, a logged-in or PRO user can be treated like an unauthenticated user. Custom frontends and third-party wrappers are therefore identity-boundary risks, not merely UI risks.

Use this practical test: if the standard HF page works while logged in but API/custom/third-party usage fails, debug identity forwarding before debugging the model or the user's billing state.

Primary references:
- https://www.gradio.app/docs/python-client/using-zero-gpu-spaces
- https://github.com/gradio-app/gradio/issues/13209

ZeroGPU is especially sensitive to request path details.

A custom frontend, embedded path, or app-to-app call can change which headers are present, how the user is identified, and whether the backend sees the request as authenticated. So the first validation for a ZeroGPU Space should always be the standard Hugging Face Space page while logged in, with the smallest possible workload. Only after that should you test custom frontends, embedding, or chained app-to-app flows.

This matters because a path-level identity problem can masquerade as a quota bug, a performance bug, or even a model bug. If the standard browser path works and the custom path fails, the custom path is guilty until proven otherwise.

## 5. Storage and persistence

Use this chapter to decide where data should live and how to interpret disk-related failures correctly.

Many Space failures that look like model, framework, or startup bugs are actually storage-model mistakes. The main split is between **ephemeral local disk**, **versioned dataset repos**, and **mutable Storage Buckets**.

### 5.0 What belongs in this chapter vs elsewhere

Use this chapter for the **storage model itself**:
- what local disk is,
- what dataset repos are,
- what Buckets are,
- how cache pressure differs from repository problems.

Use other parts of the guide for:
- **specific storage symptoms** → Cookbook
- **dead-end old storage assumptions** → Appendix A
- **Xet / cache migration details** → Appendix D
- **docs / forum / issue follow-up** → Resources

### 5.1 Disk, cache, and local filesystem problems

Many Space failures that look like model or framework problems are actually filesystem problems.

Every Space comes with local disk storage, but that storage is **ephemeral**. If the Space restarts or is stopped, local files can disappear. Local disk is therefore appropriate for temporary downloads, decompression, caches, and short-lived intermediate files, but it is the wrong place to treat as your only durable source of truth. ([Hugging Face](https://huggingface.co/docs/hub/spaces-storage))

This distinction matters because three very different things often get mixed together:

- **repository contents** committed to Git
- **runtime-local files** created after the Space starts
- **persistent external storage** such as dataset repos or Storage Buckets

If you do not separate those three in your head, you can easily misread the symptom. A file that “exists locally” during one run but disappears after restart is not the same type of problem as a repository checkout failure or a missing secret.

#### 5.1.1 “No space left on device”

Treat this error literally before you invent a deeper explanation.

When a Space reports `No space left on device`, the usual causes are:

- model or dataset downloads that exceed the available runtime disk
- cache growth across repeated rebuilds or repeated startup downloads
- temporary files that are never cleaned up
- archives or converted assets that are much larger than the originals

The practical response is not “upgrade everything and hope.” It is to identify **which class of files is consuming disk** and then move or remove the right class.

Good immediate questions are:

- Is the Space downloading large assets at startup that could be preloaded or moved elsewhere?
- Is the repository itself carrying files that belong in a model repo, dataset repo, or Bucket instead?
- Is the app repeatedly generating large temporary files and never deleting them?

If the file is needed only during one execution path, leave it in ephemeral storage and clean it up. If it must persist or be shared, move it to a proper long-lived destination instead of expanding local-disk assumptions.

#### 5.1.2 HF_HOME, caches, and startup downloads

Many disk issues are really cache-placement issues.

`huggingface_hub` uses `HF_HOME` as the root for several local storage paths. By default, repository cache lives under `HF_HUB_CACHE`, Xet chunks live under `HF_XET_CACHE`, and downstream-library assets live under `HF_ASSETS_CACHE`, all under `HF_HOME` by default. ([Hugging Face](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables))

That means a Space can slowly fill its local disk without any single giant artifact in your repository. Instead, the pressure comes from:

- repeated Hub downloads
- Xet chunk caching
- preprocessed assets from downstream libraries
- logs and temporary conversion outputs

This is also why “works once, fails later” can happen: the startup path and cache state are part of the runtime contract.

Two simple rules help:

1. **If startup always needs a known Hub asset, consider preloading it deliberately** instead of discovering it indirectly through repeated cold starts.
2. **If the asset is large, mutable, or long-lived, do not let local cache be its only home.**

#### 5.1.3 Why local fixes disappear after restart

A common beginner trap is to connect through Dev Mode or a shell, edit files, download assets, test successfully, and then assume the fix is permanent.

It is not, unless the change was committed to the right persistent layer.

- A Git change is only durable if it is committed and pushed.
- A generated artifact is only durable if it is uploaded or saved to a real persistent destination.
- A local downloaded file is only durable for as long as the runtime stays alive.

This is one reason the line between “debugging” and “storage design” matters so much on Spaces. A fix that only exists in the current runtime is not really a fix.

#### 5.1.4 Repo size vs runtime disk vs cache size

These are three separate constraints.

- **Repository size** concerns what must be cloned, checked out, or tracked in Git/Xet.
- **Runtime disk** concerns what exists locally inside the live Space.
- **Cache size** concerns what libraries keep under `HF_HOME` and related paths.

You can have a small repository and still blow up runtime disk with model downloads. You can have a modest runtime workload and still fail because the repository history is unhealthy. And you can have a healthy repository and still accumulate local cache pressure until startup becomes fragile.

If you do not classify the failure at the right layer, your fix will probably land in the wrong place.

### 5.2 Persistent storage choices

The correct storage destination depends on **how long the data must live**, **how often it changes**, and **whether it should behave like a Hub dataset/repository or like mutable working storage**.

For most Space authors, the cleanest mental model is:

- local disk for temporary work
- dataset repos for reusable and shareable data
- Storage Buckets for large mutable working data

#### 5.2.1 When ephemeral disk is enough

Ephemeral disk is enough when the file is:

- temporary
- reproducible
- cheap to regenerate
- not the final output you care about

Examples:

- unzip directories
- temporary image/audio conversions
- per-request scratch files
- one-run caches during debugging

If losing the file on restart is acceptable, ephemeral storage is fine.

#### 5.2.2 When to use a dataset repo

Use a dataset repo when you want the output to behave like a first-class dataset artifact on the Hub.

A dataset repo is the right choice when the data should be:

- **versioned**
- **shareable**
- easy to revisit later
- compatible with Hub dataset tooling such as dataset cards and `load_dataset()` workflows

This makes dataset repos the natural default for:

- publishable evaluation outputs
- curated artifacts you want others to inspect
- reusable structured data generated by a Space
- semi-final outputs where history and reproducibility matter

In practice, if your instinct is “I may want to treat this like a dataset later,” a dataset repo is usually the correct default.

#### 5.2.3 When to use a Storage Bucket

Use a Storage Bucket when the data is **large, mutable, and operational**, rather than publication-oriented.

Storage Buckets are a separate repo type on the Hub that provide **S3-like object storage**. Unlike Git-based repositories, they are **non-versioned and mutable**, and the docs explicitly describe them as **simple, fast storage** for checkpoints, logs, intermediate artifacts, and other large collections of files that do not need version control. ([Hugging Face](https://huggingface.co/docs/hub/storage-buckets))

That makes Buckets a good fit for:

- rolling logs
- training or fine-tuning checkpoints
- intermediate artifacts in multi-stage workflows
- frequently overwritten files
- large working data that is not meant to behave like a published dataset

For this guide, the safest way to explain Buckets is through the **Python API and CLI first**, because those are the most stable, explicit interfaces. Mount-based usage inside Spaces can be useful, but the conceptual model should start with Buckets as mutable object storage, not as “permanent local disk.” ([Hugging Face](https://huggingface.co/docs/huggingface_hub/en/guides/buckets))

#### 5.2.4 Dataset repo vs Bucket: practical decision rules

Use this rule when choosing between them.

Choose a **dataset repo** if you want:

- version history
- Hub-style shareability
- dataset semantics
- Viewer / `load_dataset()` friendliness
- stable, publishable artifacts

Choose a **Bucket** if you want:

- fast mutable storage
- overwrite-in-place behavior
- operational working data
- logs, checkpoints, intermediate artifacts
- high-frequency writes without Git history concerns

If you are unsure, default to **dataset repos** for publishable or reusable outputs and **Buckets** for mutable working storage.

---

## 6. ZeroGPU deep dive

Use this chapter when ZeroGPU behavior is the main variable.

This chapter covers the ZeroGPU runtime model itself: hardware tiers, duration, queue behavior, quota, root-level CUDA placement, import-time side effects, `torch.compile`, and AoTI. If the main question is “what kind of runtime is ZeroGPU really?”, start here.

### 6.0 What belongs in this chapter vs elsewhere

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 6.0.1 2026-05 triage update: four tracks before fixes

For current ZeroGPU debugging, classify the case into four tracks before changing code:

1. **Quota / billing / identity** — who is making the request, which account tier is applied, whether the route is authenticated, and whether credits or run-count limits are involved.
2. **Duration / queue / decorator design** — what `@spaces.GPU(...)` requests, whether the requested GPU section is too broad, and whether `large` / `xlarge` changes quota cost.
3. **Duplication / runtime reconstruction** — whether a duplicated Space copied only the repository or also recreated hardware, secrets, README YAML, dependency pins, startup behavior, and auth path.
4. **CUDA / compile / runtime side effects** — whether CUDA is touched too early, whether a normal persistent-GPU assumption leaked into ZeroGPU, and whether AoTI is the right advanced optimization path.

This matters because the same visible error, especially a quota error or 503, can come from different tracks. Treat ZeroGPU as a managed shared-GPU runtime, not as a normal always-on GPU Space.

Use this chapter for the **ZeroGPU runtime model**:
- hardware and tiers,
- duration and quota behavior,
- root-level CUDA vs side effects,
- `torch.compile` vs AoTI.

Use other parts of the guide for:
- **symptom-first troubleshooting** → Cookbook
- **historical and operational quirks** → Appendix B
- **auth-path and API-path failures** → Chapter 4
- **live external references** → Resources

### 6.1 Hardware and tiers

ZeroGPU is not a generic “free GPU toggle.” It is a specific runtime model with specific hardware, version, and queueing assumptions.

Current docs describe ZeroGPU as shared infrastructure that dynamically allocates and releases NVIDIA RTX Pro 6000 Blackwell GPUs as needed. In practice, your Space does not keep a dedicated GPU alive all the time. GPU work is scheduled into a managed shared system instead. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

The public docs describe two ZeroGPU sizes:

- **`large`** — half NVIDIA RTX Pro 6000 Blackwell, 48 GB VRAM, 1× quota cost
- **`xlarge`** — full NVIDIA RTX Pro 6000 Blackwell, 96 GB VRAM, 2× quota cost

The hardware details matter because an apparent “model problem” can actually be a tier-selection problem, a quota-cost problem, or a CUDA wheel compatibility problem. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

#### 6.1.1 Blackwell-era compatibility check

When a ZeroGPU Space was written or last debugged under older assumptions, classify the failure before changing model logic.

Check these in order:

1. **Hardware expectation** — Did the code or old notes assume A100, H200, or an unspecified GPU?
2. **Effective ZeroGPU size** — Is the call using `large` or `xlarge`?
3. **PyTorch version** — Is the installed `torch` one of the currently supported ZeroGPU versions?
4. **CUDA wheel capability** — Does the wheel support the active Blackwell-class GPU architecture, including `sm_120` where applicable?
5. **Error shape** — Does the log include `CUDA error: no kernel image is available for execution on the device`?

If the answer to the last question is yes, treat it as substrate drift first. A practical recovery path is to move to a ZeroGPU-supported PyTorch version with a CUDA wheel suitable for current Blackwell-backed ZeroGPU, then re-test the smallest workload before changing the model code.

### 6.2 Duration, queue priority, and quotas

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 6.2.1 Requested duration is not visible runtime

Do not read `60s requested` as “the app visibly ran for 60 seconds.” It means the ZeroGPU-managed call requested a GPU duration budget. The remaining daily quota must be large enough for that requested duration, even if the user-visible work appears shorter.

Keep these separate:

- daily GPU-time quota,
- per-call requested duration,
- Free-account run-count limits,
- `large` vs `xlarge` quota cost,
- authenticated vs unauthenticated quota,
- prepaid-credit overage for eligible paid tiers,
- queue priority.

A useful explanation for users is: “Your visible runtime may be short, but the Space may be reserving or requesting a larger GPU budget for each call.”

Primary references:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://discuss.huggingface.co/t/it-seems-use-60-sec-gpu-quota-instead-of-real-time-usage/175130

One of the most important ZeroGPU details is also one of the easiest to miss: **declared duration matters**.

ZeroGPU uses `@spaces.GPU(...)` to mark GPU sections. The default GPU duration is **60 seconds** unless you specify another duration, and the docs state that shorter durations improve queue priority. That means two identical models can feel very different to the user if one app declares realistic durations and another leaves everything to worst-case runtime behavior. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

Quota also depends on account type. Current documentation distinguishes unauthenticated usage, free usage, PRO, Team, and Enterprise tiers, and notes that remaining quota affects queue priority as well. Quotas reset **24 hours after first GPU usage**, not on an arbitrary wall-clock boundary. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

In practice, “I only ran this a few times” is not enough information. You also need to ask:

- What duration was declared?
- Which tier was the request actually using?
- Was the request authenticated?
- Was the Space using `large` or `xlarge`?

### 6.3 Over-quota and account-type behavior

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 6.3.1 Credits and overage: current rule vs historical clues

Keep current specification and forum history separate.

Current public docs describe prepaid-credit overage for eligible paid tiers after the included daily ZeroGPU quota is exhausted. Historical forum reports from rollout periods are useful operational evidence, but they should not be treated as timeless billing specification.

When a user says credits did not help, ask:

- what account tier is active,
- whether prepaid credits are actually available,
- whether the failing route is authenticated as that account,
- whether the error is really quota/overage rather than requested duration,
- the date of the referenced forum workaround or report.

Safe wording: “Current behavior and older rollout reports can differ; check current docs and the request identity path before concluding that billing is broken.”

Primary references:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/hub/llms-full.txt
- https://discuss.huggingface.co/t/getting-you-have-exceeded-your-pro-gpu-quota-error-even-with-credits-loaded/174850

Older ZeroGPU advice often stops at “you ran out of quota.” That is no longer enough.

Current docs state that PRO users have higher daily usage quota, higher queue priority, and can go beyond their included daily quota with pre-paid credits. Team and Enterprise plans have their own included usage and over-quota behavior as well. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu))

For troubleshooting, this means two things.

First, do not assume a user is really seeing the tier they think they are seeing. If the request path is effectively unauthenticated, the behavior can resemble a much lower quota tier.

Second, do not write user-facing messages that imply a single universal quota model. ZeroGPU behavior is tier-sensitive.

### 6.4 Root-level CUDA placement vs import-time side effects

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### 6.4.1 Practical CUDA-side-effect checklist

Use this checklist before declaring ZeroGPU incompatible:

- Is `spaces` imported before CUDA-touching work?
- Does any import call `torch.cuda.*`?
- Does any import move a model to CUDA?
- Does a dependency initialize CUDA during import?
- Is `model.to("cuda")` inside the managed path?
- Is the app assuming a persistent CUDA process?
- Is `torch.compile` used directly or indirectly?
- Is a CUDA extension trying to build from source instead of using a compatible wheel?

For one-off `No CUDA GPUs are available`, retry/restart once before rewriting the app. For repeated failures, look for eager CUDA side effects and native-extension assumptions.

Use this distinction carefully:

**root-level CUDA placement is allowed on ZeroGPU, but import-time or eager side effects can still make a Space fragile.**

That difference matters in practice.

The official AoTI blog explains that ZeroGPU uses the `spaces` package to intercept PyTorch API calls, postpone CUDA operations, and execute decorated GPU functions in a fork. The blog even shows a pattern where a pipeline is created at module level and moved to CUDA before the decorated function is called. ([Hugging Face](https://huggingface.co/blog/zerogpu-aoti))

So the rule is **not** “root-level CUDA is forbidden.”

The real problem is this: some libraries or code paths do CUDA-touching work too early or in ways that do not compose well with the ZeroGPU process model. In practice, fragile patterns include:

- import-time CUDA initialization with side effects
- library imports that implicitly touch CUDA
- eager initialization paths that do more than ordinary device placement
- code that assumes a long-lived CUDA process model

So the safe wording is:

- root-level CUDA placement can be valid
- but import-time CUDA side effects are still a first-class suspect when a ZeroGPU Space behaves strangely

### 6.5 Why `torch.compile` is a bad fit for ZeroGPU

This is one of the clearest “looks clever, performs badly” traps on ZeroGPU.

The official ZeroGPU guidance is explicit: **ZeroGPU does not support `torch.compile`**. The AoTI blog also explains why on-the-fly compilation is a poor fit for ZeroGPU’s process model. ZeroGPU forks a process for GPU work and releases that process when the GPU is released, so compile-heavy strategies that assume a stable long-lived process can end up re-paying compilation cost in exactly the wrong place. ([Hugging Face](https://huggingface.co/blog/zerogpu-aoti))

In practice, this means that code paths that **implicitly** rely on `torch.compile` semantics can be just as problematic as explicit `torch.compile(...)` calls. Even if the app “works,” the performance profile may become worse instead of better.

### 6.6 AoTI as an advanced path

AoTI is the exception path.

The same official guidance that rules out `torch.compile` points to **Ahead-of-Time compilation** with `torch.export` + `AOTInductor` as the supported advanced optimization route. The point is to compile once in a controlled path and then reuse the compiled artifact, instead of doing expensive just-in-time compilation during ordinary ZeroGPU execution. ([Hugging Face](https://huggingface.co/blog/zerogpu-aoti))

This can be valuable, but it should be treated as **advanced operational work**, not as a default first step.

AoTI is a good fit only when:

- the model and shapes are stable enough
- the compilation cost is worth paying once
- you have a fallback path if compilation or export breaks
- you understand that this is an optimization layer, not a basic correctness fix

A good maintainer rule is: **correctness first, reproducibility second, performance third**.

### 6.7 Root-level CUDA is allowed, but bad side effects still happen

This section exists to make the practical nuance explicit.

A reader should not leave this guide thinking either of these oversimplifications is true:

- “root-level CUDA is always forbidden on ZeroGPU”
- “if root-level CUDA is allowed, then import-time CUDA behavior is never a problem”

Both are wrong.

The better operational rule is:

- allow root-level CUDA placement as a valid pattern
- but if the Space is unstable, one of the first experiments should be to move CUDA-touching work closer to the controlled GPU execution boundary and see whether the failure disappears

This is not because the original pattern is “illegal.” It is because ZeroGPU is a specialized runtime, and specialized runtimes often expose edge cases in library initialization order.

### 6.8 What changed from older guides

Several older assumptions now need explicit correction.

First, old advice about permanent local Space storage is now a dead end. Current config docs state that the old persistent storage feature is no longer available and that `suggested_storage` is ignored. ([Hugging Face](https://huggingface.co/docs/hub/spaces-config-reference))

Second, old download-stack advice can be stale. In the `huggingface_hub` v1.0 migration era, the older `hf_transfer`-centric advice is no longer the best reference point, and Xet-related environment variables are the current knob surface. ([Hugging Face](https://huggingface.co/docs/huggingface_hub/concepts/migration), [Hugging Face](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables))

Third, old Gradio chat-history assumptions are dangerous. Gradio major-version changes and Transformers chat templating still require readers to distinguish between UI-side structured content and tokenizer-side plain-string expectations. ([Gradio](https://www.gradio.app/guides/gradio-6-migration-guide), [Hugging Face](https://huggingface.co/docs/transformers/chat_templating))

## 7. Cookbook: common problems and fixes

Use the Cookbook when you already recognize the symptom and want the shortest practical next-step list.

The recipes here do not replace the main chapters. They translate recurring real-world failures into checklists, search terms, and routing hints.

### 7.0 How to read the Cookbook

The Cookbook is where repeated real-world failures get translated into practical checklists.

Use it when:
- you already know the symptom,
- you want the shortest likely-next-step list,
- or you need a searchable entry for a recurring error shape.

If you need the formal definitions first, read the main body chapters before the recipe.
If you need migration nuance, read the relevant Appendix after the recipe.

This section is intentionally pragmatic. The goal is not to explain every subsystem in full, but to give a repeatable “symptom -> first checks -> likely cause -> next move” pattern that works under real Spaces conditions.

### 7.1 `ModuleNotFoundError` / missing dependencies

**Symptom**

- Build succeeds, but the app crashes at startup with `ModuleNotFoundError`.
- Or build fails because `pip` cannot resolve a package.

**First checks**

1. Confirm whether the error appears in **build logs** or **runtime logs**.
2. Compare imports in your code against `requirements.txt`, `pre-requirements.txt`, and `packages.txt`.
3. Check whether the import name differs from the package name.

**Likely causes**

- The dependency is missing from `requirements.txt`.
- The package name and import name differ.
- The dependency installs, but only fails at import time because of version drift.

**Next move**

- Add the package explicitly and pin the version if the stack is fragile.
- If the package is optional, remove the import entirely until the minimal app works.
- If the failure only appears after duplication, compare the duplicate’s YAML and dependency pins against the source Space.

### 7.2 Version conflicts and ABI issues

**Symptom**

- Import-time crashes mentioning missing symbols, incompatible libraries, or extension load failures.
- Problems often involve `torch`, `xformers`, `bitsandbytes`, FA kernels, or quantization backends.

**First checks**

1. Compare `torch`, `transformers`, `accelerate`, `gradio`, and low-level extension versions.
2. Check whether the failure appears only after a rebuild or duplicate.
3. Remove optional acceleration libraries and test the simplest stack.

**Likely causes**

- A compatible version set was replaced by a newer but incompatible one.
- An extension library was built or published for a different CUDA / Python / torch combination.
- A ZeroGPU rebuild now exposes Blackwell-backed behavior while the installed PyTorch / CUDA wheel was chosen under older A100 or H200 assumptions.

**Next move**

- Prefer the simplest working stack first.
- Reintroduce performance packages only after correctness is restored.
- Treat FA3 / quantization / custom kernels as optional feature flags, not baseline requirements.
- If the error is `CUDA error: no kernel image is available for execution on the device`, check the ZeroGPU supported PyTorch versions and CUDA wheel before rewriting model code.

### 7.3 “Disk full” / cache pressure / startup downloads

**Symptom**

- `No space left on device`
- Model download fails halfway through.
- A restart “fixes” the app temporarily, but the issue comes back.

**First checks**

1. Estimate what is consuming space: repo checkout, model cache, dataset cache, temp files, upload cache, outputs.
2. Check `HF_HOME` and the caches under it.
3. Check whether heavy downloads happen at startup on every cold run.

**Likely causes**

- Startup downloads are filling the ephemeral disk.
- The cache path is growing without cleanup.
- Large files were committed directly into the Space repo.

**Next move**

- Move durable artifacts out of the Space filesystem.
- Use `preload_from_hub` or another controlled prefetch path when appropriate.
- Re-evaluate whether the files belong in a dataset repo, a model repo, or a Bucket.

### 7.4 Timeouts and slow responses

**Symptom**

- First request is extremely slow.
- API clients time out even though the Space eventually works.
- Queue wait dominates user-visible latency.

**First checks**

1. Separate queue time, startup time, model load time, and inference time.
2. Check whether the workload fits the declared ZeroGPU duration.
3. Confirm whether large downloads or compile steps are happening on the hot path.

**Likely causes**

- ZeroGPU cold-start and queue delay.
- Model load and cache miss at startup.
- Compile or quantization setup running during request handling.

**Next move**

- Test the smallest successful workload first.
- Move heavyweight setup out of the request path when possible.
- Keep optional acceleration behind flags and fallbacks.

### 7.5 Authentication / private models / private datasets

**Symptom**

- Works locally with your token, fails in the Space.
- Browser works, but an API call or background fetch fails.
- Private assets load in one environment but not another.

**First checks**

1. Check whether the resource is private or gated.
2. Confirm that the required token exists as a secret in the Space.
3. For private Space API calls, confirm that the caller uses a READ-capable token.

**Likely causes**

- Secret missing after duplication.
- Wrong token scope.
- A private dependency was assumed to be public.

**Next move**

- Recreate secrets manually after duplication.
- Narrow the first successful test to one authenticated path.
- Only after that should you add more API clients or chained Spaces.

### 7.6 Where should I save outputs from a Space?

Save outputs based on how durable, shareable, and mutable they need to be.

Use this rule:
- **ephemeral disk** for scratch work only
- **dataset repo** for versioned, shareable, reusable outputs
- **Bucket** for fast, mutable working storage such as logs, checkpoints, and intermediate artifacts

If the output should still make sense later as a Hub asset, default to a dataset repo. If it changes often and does not need Git history, Bucket is usually the better fit.

**Related reading:** 1.4 Storage mental model → 5.2 Persistent storage choices → Appendix A / Appendix D → Resources 9.6 / 9.9

### 7.7 How to debug “No space left on device”

Treat this error literally first. Most of the time, the problem is real disk pressure, not a hidden model bug.

1. Confirm the error is truly disk pressure, not quota or timeout.
2. Identify whether the space is being consumed by:
   - repo checkout
   - model cache
   - dataset cache
   - temporary uploads / generated files
   - startup downloads
3. Remove or externalize the largest category first.
4. Do not assume a local fix will survive restart.

A common mistake is to “fix” a disk error interactively in Dev Mode and then forget that the local filesystem is still ephemeral.

**Related reading:** 5.1 Disk, cache, and local filesystem problems → 5.2 Persistent storage choices → 7.29 HF_HOME, cache growth, and disk pressure → Appendix D → Resources 9.6 / 9.9

### 7.8 Dataset repo vs Bucket: which one should I choose?

Use a dataset repo when the output should behave like a Hub artifact. Use a Bucket when the output should behave like mutable working storage.

Choose a **dataset repo** when you want:
- version history
- sharing and browsing on the Hub
- compatibility with dataset tools and `load_dataset()`
- a stable, publishable artifact

Choose a **Bucket** when you want:
- high-frequency writes
- mutable working storage
- logs, checkpoints, rolling backups, intermediate artifacts
- storage that behaves more like object storage than Git history

### 7.9 High-frequency writes, logs, checkpoints, and intermediate artifacts

This is the clearest Bucket use case.

If your Space emits lots of files over time, rewrites the same logical artifact, or keeps operational logs and checkpoints, Git-style repos are usually the wrong primary destination. Use a Bucket or another external durable store for the changing files, then publish selected stable artifacts elsewhere if needed.

### 7.10 Space disk is ephemeral: what that means in practice

“Ephemeral” means more than “could disappear someday.”

In practice it means:

- a restart can remove local fixes
- cold-start can re-trigger downloads
- generated outputs saved only locally are not durable
- cache cleanup and destination planning matter as much as code correctness

### 7.11 Build succeeds, startup fails: what to check next

Treat this as a startup-health problem, not as a build problem.

If the build is green but the Space never becomes healthy:
1. check runtime logs, not build logs
2. look for model load failures, import-time crashes, or startup health timeouts
3. check `startup_duration_timeout`
4. check whether startup is doing too much work that should have happened earlier or elsewhere

**Related reading:** 3.4 Large downloads at build time vs startup time → 5.1 Disk, cache, and local filesystem problems → Resources 9.9 (Build / startup)

### 7.12 Empty logs or only “Build queued”

Treat this as a platform-state or provisioning classification problem first, not as proof that your code is wrong.

Likely buckets:
- scheduler / capacity issue
- transient platform issue
- wedged Space state
- less commonly, a repo or provisioning failure that does not produce a useful build log yet

Before rewriting dependencies, try restart / rebuild and check whether a minimal Space on the same hardware behaves similarly.

**Related reading:** 3.3 A decision tree for “stuck on Building” → 3.5 Platform-level incidents → Appendix E if duplication triggered the problem → Resources 9.9 (Build / startup)

### 7.13 Space works in browser but fails through API

Treat this as an auth-or-path problem before you treat it as a model problem.

Default suspicion order:
1. token / auth scope
2. wrong endpoint path
3. private resource assumptions
4. different request schema
5. different execution path from the browser path

Do not start by blaming the model.

**Related reading:** 4.5 Authenticated vs unauthenticated calls on ZeroGPU → 4.6 Browser works but API fails → 4.7 Custom frontends, app-to-app calls, and identity-path pitfalls → Appendix C → Resources 9.9 (Runtime / API / Gradio migration)

### 7.14 ZeroGPU quota looks wrong or PRO seems ignored

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
Extra quick check:

```text
quota-looking failure
→ check request identity path
→ check requested duration
→ check daily quota / run-count / large-xlarge cost
→ check paid-tier credits or overage only after identity and duration are clear
```

Do not diagnose billing before confirming that the request is actually being made as the expected account.

Treat this as an identity-path question before you treat it as a raw quota question.

Check:
1. does the Space work while logged in on the standard HF page?
2. does it fail only through API, custom frontend, or embedded path?
3. is the request authenticated or unauthenticated?
4. is the declared duration too large relative to the task?

**Related reading:** 1.5 ZeroGPU 2026 update → 6.2 Duration, queue priority, and quotas → 6.3 Over-quota and account-type behavior → Appendix B → Resources 9.9 (ZeroGPU)

### 7.15 Duplicate works differently from the original Space
**Related reading:** 0.3 First 5 minutes checklist → 3.3 A decision tree for “stuck on Building” → Appendix E. Duplicate Space troubleshooting → Resources 9.9 (Build / startup)

This usually means the **repo was copied but the runtime contract was not fully recreated**.

Check, in order:

1. hardware
2. secrets
3. README YAML
4. dependency pins
5. startup path
6. auth / frontend path

### 7.16 Secrets were not copied during duplication

This is one of the most common duplication failures.

Public variables may be copied into duplicates. Secrets are not. If the source depends on `HF_TOKEN`, API keys, or private resources, the duplicate can look normal while failing immediately for missing-runtime-state reasons.

### 7.17 Gradio 6 messages format vs tokenizer input normalization

Treat this as a boundary problem between two different contracts.

- At the **UI boundary**, Gradio 6 expects the newer structured messages / content-block model.
- At the **tokenizer boundary**, many text-only model templates still expect plain strings.

Do not mix these layers. Normalize at the boundary instead.

Also check event chaining when a Gradio app starts doing work after a failed prior step. `.then()` is continuation-oriented; `.success()` is the safer choice when the downstream step must run only after the previous step succeeds.

**Related reading:** 2.6 Debugging LLM-generated code → 4.3 Dependency mismatches at runtime → Appendix C. Gradio / Transformers migration notes → Resources 9.7 / 9.9

### 7.18 `apply_chat_template()` crashes on list-based content
**Related reading:** 2.6 Debugging LLM-generated code → Appendix C. Gradio / Transformers migration notes → Cookbook 7.17 / 7.36 → Resources 9.7

If `message["content"]` arrives as a list of blocks and you pass it directly into a text-only chat template, failures like string-concatenation or template-render errors are expected.

First fix:

- flatten or normalize to the form the model’s tokenizer template actually expects

### 7.19 `hf_transfer` advice is outdated: use Xet correctly

Treat old `hf_transfer` advice as historical unless you have a very specific reason not to.

Current guidance:
- Xet is the default backend
- `hf_transfer` support is removed
- use current Xet-related settings instead of old transfer toggles

**Related reading:** 5.1 Disk, cache, and local filesystem problems → Appendix D. Download stack and cache notes → Resources 9.7 / 9.9

### 7.20 Xet environment variables that still matter

Troubleshooting downloads and cache behavior now often means checking:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_XET_CACHE`
- `HF_ASSETS_CACHE`
- `HF_XET_HIGH_PERFORMANCE`

If your cold-start behavior changed after a library upgrade, re-check these before blaming the model or the network.

### 7.21 `torch.compile` is not supported on ZeroGPU

Do not treat `torch.compile` as a normal optimization path on ZeroGPU.

Even beyond the explicit support constraint, ZeroGPU’s process model is a poor fit for compile-on-demand strategies that re-run too often.

**Related reading:** 6.4 Root-level CUDA placement vs import-time side effects → 6.5 Why `torch.compile` is a bad fit for ZeroGPU → 6.6 AoTI as an advanced path → Appendix B → Resources 9.9 (ZeroGPU)

### 7.22 AoTI on ZeroGPU: when it helps and when it hurts

AoTI is an advanced optimization path, not the default fix.

It can help when:
- you already have correctness
- the model and shapes are stable enough
- compile cost can be amortized or artifacts can be reused

It hurts when:
- you are still debugging correctness
- shape variability is high
- queue / quota / compile cost dominate actual inference

**Related reading:** 6.5 Why `torch.compile` is a bad fit for ZeroGPU → 6.6 AoTI as an advanced path → Appendix B → Resources 9.9 (ZeroGPU)

### 7.23 Root-level CUDA is allowed, but eager side effects still break things

Keep the distinction explicit: allowed CUDA placement is not the same thing as safe import-time behavior.

These two statements can both be true:
- root-level CUDA placement can work on ZeroGPU
- import-time or eager side effects can still make a Space fragile or nonfunctional

So the debugging rule is not “never touch CUDA early.” The debugging rule is “differentiate between allowed placement and bad side effects.”

**Related reading:** 6.4 Root-level CUDA placement vs import-time side effects → Appendix B. ZeroGPU operational notes → Resources 9.9 (ZeroGPU)

### 7.24 Bucket for logs/checkpoints vs dataset repo for publishable artifacts

Use Bucket as the operational store. Use dataset repo as the curated store.

That split prevents working-state churn from polluting a publishable, versioned artifact history.

### 7.25 When to keep using a dataset repo instead of adopting Buckets

Keep using a dataset repo when:

- you want history and diffs
- you want Hub-native discoverability
- the artifact is intended for later consumption as data
- you want the object to be a first-class dataset asset, not just a file blob

### 7.26 Authenticated vs unauthenticated ZeroGPU API calls

Do not treat these as equivalent.

They can consume different quota paths and produce different user-visible behavior. Always test both intentionally if your app depends on both.

### 7.27 Browser path vs custom frontend path

A Space can work on the standard HF page and still fail through a custom frontend.

That does not automatically mean the model is broken. It may mean the request identity, auth, or frontend integration path is different.

### 7.28 Build-time downloads vs startup-time downloads

This distinction matters more than many people think.

A Space that “worked once” may only have looked stable because the source runtime was already warm. Duplicates and rebuilds often re-expose startup download costs and startup health limits.

### 7.29 HF_HOME, cache growth, and disk pressure
**Related reading:** 5.1 Disk, cache, and local filesystem problems → Appendix D. Download stack and cache notes → Resources 9.6 / 9.9

Disk pressure is often a cache-placement problem.

If the cache lives on the ephemeral Space filesystem and grows unchecked, the app can fail long before the repo itself is large.

### 7.30 Before you trust any fix: check docs, changelog, and open issues
**Related reading:** Appendix F. HF source taxonomy and update discipline → Appendix I. Search and research basics → Resources 9.7 / 9.8 / 9.9

Hugging Face, Gradio, Transformers, and `huggingface_hub` change quickly.

Before you trust a snippet, a forum answer, or even an old note:

1. check the latest official docs
2. check the changelog or migration guide
3. check open GitHub issues and repo discussions
4. only then decide whether the advice is still current

This is especially important for:

- Gradio major version changes
- `huggingface_hub` migration and download stack behavior
- ZeroGPU operational guidance
- chat templating and message-format expectations

### 7.31 How to navigate Hugging Face when you do not know where to start
Use a simple route map:
- **Hub docs** when you need product concepts and official feature behavior.
- **Spaces docs** when the problem is about build, runtime, config, hardware, or Dev Mode.
- **Changelog / migration guides** when something used to work and stopped working after an upgrade.
- **Forum / Discussions / GitHub issues** when the symptom looks current, niche, or rollout-related.

For fast-changing topics such as Spaces, ZeroGPU, Gradio, Transformers, and `huggingface_hub`, prefer:
1. official docs,
2. migration guides / changelogs,
3. current GitHub issues,
4. current forum threads.

### 7.32 How to choose the right Hugging Face documentation page
A rough rule:
- **What is this feature?** → product overview / index page.
- **Which config field or parameter changed?** → config reference or migration guide.
- **Why does a specific error happen?** → issue tracker + forum + docs.
- **How do I deploy or serve this?** → task / engine / deployment docs.
- **What changed recently?** → changelog or release notes.

### 7.33 How to search for the right Space, issue, or model
Turn errors into search units:
- exact error phrase in quotes
- one product or library name
- one runtime qualifier (`Spaces`, `ZeroGPU`, `Gradio 6`, `Transformers v5`, `Xet`)
- one symptom qualifier (`Building`, `No API found`, `quota`, `duplicate`, `additional_chat_templates`)

Examples:
- `"No API found" Gradio 6 Spaces`
- `"additional_chat_templates" Transformers v5`
- `"Build queued" Hugging Face Spaces`
- `"CUDA must not be initialized in the main process" ZeroGPU`

### 7.34 How to use this guide itself as context for an LLM
A practical workflow:
1. Attach this guide.
2. Attach the failing logs / stack trace / README YAML / `requirements.txt`.
3. Ask the LLM to:
   - classify the failure bucket,
   - list the most likely root causes,
   - point to the most relevant sections of this guide,
   - generate a minimal search plan.

Good requests are specific:
- “Classify this failure into build/runtime/API/platform.”
- “Which sections of the guide are most relevant to this stack trace?”
- “Turn this log into 5 good search queries.”
- “List only fixes that are consistent with Gradio 6 and Transformers v5.”

### 7.35 Datasets 4.x migration: dataset scripts, TorchCodec, and media pitfalls

Treat this as a migration bucket, not as a generic dataset-loading bug.

If a Space loads older Hub datasets and suddenly fails after a `datasets` upgrade, check whether the repo still depends on dataset scripts. The `datasets` 4.x line removed dataset scripts from the loading path and moved more strongly toward Parquet/Arrow and streaming-friendly data layouts. Audio/video decoding also moved to TorchCodec + FFmpeg, which introduces new environment requirements and a new class of failures. Treat:
- old dataset script assumptions,
- `trust_remote_code` style expectations,
- and missing TorchCodec / FFmpeg
as a separate migration bucket.

**Related reading:** Appendix C / D for migration framing → Resources 9.7 / 9.9 → official Datasets docs and migration-related notes

### 7.36 Transformers v5 migration: what breaks most often in real apps

Treat Transformers v5 as a migration event, not as a normal patch bump.

Common breakpoints worth checking first:
- `apply_chat_template()` return type expectations
- `dtype` vs older `torch_dtype` style usage in examples
- `Trainer(..., tokenizer=...)` vs `processing_class`
- removed / renamed pipeline and generation paths
- `quantization_config` migration

**Related reading:** Appendix C. Gradio / Transformers migration notes → Resources 9.7 → current Transformers migration guide and issue tracker

### 7.37 ZeroGPU Blackwell / CUDA wheel mismatch

**Symptom**

- A ZeroGPU Space rebuilds or duplicates successfully but fails when GPU work begins.
- Logs mention `CUDA error: no kernel image is available for execution on the device`.
- The same code previously worked under older ZeroGPU assumptions.

**First checks**

1. Confirm the Space is actually using ZeroGPU, not CPU or a paid dedicated GPU tier.
2. Check whether the app or its notes assumed A100, H200, or an unspecified GPU.
3. Check the installed `torch` version and CUDA wheel.
4. Compare against the current ZeroGPU supported PyTorch versions.
5. Reproduce with the smallest GPU call before changing the model pipeline.

**Likely causes**

- The installed PyTorch wheel does not include kernels for the active Blackwell-class architecture, for example `sm_120`.
- The app inherited an old pin that was valid for a previous hardware assumption.
- Optional acceleration packages were installed for a different CUDA / torch boundary.

**Next move**

Use a ZeroGPU-supported PyTorch version with a CUDA wheel suitable for current Blackwell-backed ZeroGPU. Then re-test the smallest workload. Do not treat `torch==2.9.1+cu128` as the only possible answer; treat it as one observed working pin among the supported-version space.

### 7.38 Gradio event chains run after failure

**Symptom**

- A later Gradio step runs even though an earlier step failed.
- Cleanup, upload, API call, or UI update executes after an exception.
- Debugging is confusing because the visible error comes from the second or third step, not the original failure.

**First checks**

1. Look for `.then()` chains.
2. Decide whether the downstream step should run after failure.
3. If it should only run after success, use `.success()`.
4. If it should only run after failure, use `.failure()`.

**Next move**

Treat event chaining as part of the app contract. On Spaces, this matters because failed GPU calls, upload retries, or quota errors can trigger misleading downstream behavior if the chain continues unconditionally.

### 7.39 Hub upload hits 429 or 503

**Symptom**

- Bulk uploads fail with `429`, `503`, or repeated transient HTTP errors.
- A script repeatedly calls `upload_file()` or many small commits.
- The failure looks like network instability but correlates with request volume.

**First checks**

1. Confirm the script uses `HF_TOKEN` and passes it downstream.
2. Record whether the failing requests are API, resolver, or commit/upload operations.
3. Check the installed `huggingface_hub` version.
4. For large folders, consider `upload_large_folder()` instead of repeated ad-hoc upload loops.
5. Treat rate-limit headers as operational signals, not noise.

**Next move**

Use current `huggingface_hub` behavior for rate-limit handling where possible, reduce request burstiness, and make large uploads resumable rather than restarting the entire transfer after every transient failure.

## 8. Appendices

Use the appendices for migration notes, historical dead ends, operational quirks, and search aids.

The appendices are not less important than the main chapters. They are separate because they are either narrower, more version-sensitive, more historical, or more search-oriented than the default troubleshooting path.

### 8.0 How to use the appendices

Read an appendix when one of these is true:
- the main chapter explains the default path, but your case still does not fit,
- the problem appeared after a migration or version change,
- you need historical context to avoid an old dead end,
- or you need search phrases, source taxonomy, or clue clusters rather than more narrative explanation.

If you do not already know which appendix you need, start with the main chapter first and return here only when the boundary becomes clear.

### Appendix A. Deprecated or historical storage paths

Use this appendix when older storage advice still appears in forum threads, copied templates, or generated answers.

This appendix exists to mark storage-related dead ends clearly, especially when they still look plausible but no longer match the current Spaces storage model.

#### A.1 The old persistent storage setting

Older Spaces guidance and older examples may refer to persistent local storage or settings that imply a durable local filesystem. Treat those references with caution.

At the time of this guide’s update, the old persistent storage path described through `suggested_storage`-style config is not the current model to rely on. The safer modern assumption is:

- local Space disk is ephemeral
- long-lived data belongs in a different destination

#### A.2 Why old permanent-local-storage advice is now a dead end

This dead end matters because it still appears in old answers and stale generated summaries.

If an old answer tells you to rely on local Space disk as if it were durable storage, the safest response is:

1. verify the current Spaces storage docs
2. re-evaluate whether the data belongs in a dataset repo, Bucket, or another external store

### Appendix B. ZeroGPU operational notes

Use this appendix when the default ZeroGPU explanation is not enough and you need the operational nuance.

This appendix collects the sharp edges that matter in practice: allowed placement vs bad side effects, process-model mismatches, and performance ideas that look reasonable but fit ZeroGPU poorly.

#### B.1 Root-level CUDA placement vs import-time CUDA side effects

Root-level CUDA placement can be acceptable. That does not make all import-time CUDA interactions safe.

The practical distinction is:

- **placement** may be fine
- **eager side effects** may still destabilize the app

Examples of risky patterns include:

- import-time CUDA checks with side effects
- libraries that initialize GPU state while importing
- hidden setup that assumes a traditional long-lived GPU process

#### B.2 Why `torch.compile` is a bad fit for ZeroGPU

Even if compile-on-demand is attractive elsewhere, ZeroGPU’s runtime model makes it risky as a default strategy.

Common failure modes include:

- repeated compile cost instead of amortized benefit
- wasted GPU window / quota
- complicated debugging because correctness and optimization are entangled

#### B.3 AoTI as an advanced exception path

AoTI belongs in an advanced maintenance path, not in the default “make the app work” path.

Use it only after:

- correctness is stable
- baseline performance is measured
- fallback is implemented
- compile artifacts and lifecycle are understood

#### B.4 Duration, queue priority, and cold-start trade-offs

Shorter declared durations can improve queue priority, but overly aggressive durations can also make the app fragile if the real workload does not fit.

The right question is not “what is the maximum allowed duration?”
The right question is “what is the smallest realistic duration that still matches the workload?”

#### B.5 Blackwell-era CUDA wheel checks

Use this check when a ZeroGPU Space has a CUDA kernel error after rebuild, duplication, or a dependency refresh.

Minimum useful evidence:

- effective Space hardware / ZeroGPU size,
- `torch.__version__`,
- CUDA wheel tag,
- `torch.version.cuda`,
- device name and compute capability when visible,
- exact CUDA error text.

Do not assume that an old H200-era or A100-era pin is still the safest runtime contract. Current ZeroGPU docs describe RTX Pro 6000 Blackwell-backed `large` and `xlarge` sizes and a current supported PyTorch list.

### Appendix C. Gradio / Transformers migration notes

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### C.0.1 Generation mismatch: Gradio, huggingface_hub, Transformers

A duplicated or rebuilt Space can become a silent migration.

Common generation-boundary failures include:

- old Gradio app code resolving against a newer Gradio major version,
- old downstream code importing removed `huggingface_hub` APIs,
- `huggingface_hub` 1.x behavior meeting libraries that still expect 0.x,
- Transformers v4/v5 migration boundaries interacting with hub-client versions,
- examples copied from old Spaces without `sdk_version`, `python_version`, or dependency pins.

A safe first move is to compare the resolved Gradio, `huggingface_hub`, Transformers, Python, and Space SDK versions before changing app logic.

Primary references:
- https://www.gradio.app/guides/gradio-6-migration-guide
- https://huggingface.co/docs/huggingface_hub/concepts/migration
- https://huggingface.co/blog/huggingface-hub-v1

Use this appendix when a Space used to work and stopped working after a Gradio or Transformers upgrade.

This appendix is about migration boundaries. It helps separate UI-layer contract changes, tokenizer-layer assumptions, and chat-template breakpoints that are easy to mix together.

#### C.1 Gradio 6 content blocks

Gradio 6 standardized the newer message/content-block model for chat interfaces. This is a UI-layer contract change, not automatically a tokenizer-layer change.

#### C.2 Why tokenizer inputs still often need plain strings

Many text-oriented chat templates and tokenizer paths still assume string content. That means structured UI data often needs explicit normalization before prompt construction.

#### C.3 Common `apply_chat_template()` failure patterns

Watch for these patterns:

- list-based `content` passed into text-only templates
- mixed structured blocks without normalization
- assumptions copied from older model families that no longer match the current tokenizer template

#### C.4 Event-chain continuation: `.then()` vs `.success()`

Gradio event chains are easy to misread during incident debugging.

Use this rule:

- use `.then()` when the next step may run after either success or failure,
- use `.success()` when the next step must run only after success,
- use `.failure()` when the next step is failure handling.

This is especially important on Spaces when the first step is a ZeroGPU call, Hub upload, private-resource access, or any other operation that can fail for runtime-contract reasons.

### Appendix D. Download stack and cache notes

Use this appendix when download behavior, cache layout, or transfer advice seems inconsistent across old and new guidance.

This appendix focuses on Xet-era assumptions, cache placement, and older advice that still circulates even though the current `huggingface_hub` stack changed.

#### D.1 `hf_transfer` is old advice now

Older advice that says “turn on `hf_transfer`” is no longer the right baseline for current `huggingface_hub`.

#### D.2 Xet environment variables

When investigating download performance, cache behavior, or upload/download regressions, check the current Xet-related variables first.

#### D.3 HF_HOME and cache placement

Remember that cache pressure is partly a placement problem. If the caches live on a tight ephemeral filesystem, failures may appear as storage problems rather than explicit download-stack problems.

#### D.4 Upload retries, 429, and 503

For upload-heavy workflows, separate the problem type before retrying:

- `429` usually means rate limiting or request burstiness,
- `503` often means transient service or commit-path instability,
- repeated small uploads can be worse than a resumable large-folder strategy.

Prefer authenticated requests, current `huggingface_hub`, resumable upload paths, and logs that preserve status code, request type, and retry context.

### Appendix E. Duplicate Space troubleshooting

<!-- HFMG-20260509-ZEROGPU-SPACES-OPS-DELTA -->
#### E.0.1 Runtime reconstruction order for duplicates

Use this order before editing application code:

1. hardware and Space type,
2. secrets and variables,
3. token/access to private or gated resources,
4. README YAML,
5. dependency pins,
6. startup downloads and cache assumptions,
7. standard logged-in HF page,
8. API/custom/third-party route.

Key README YAML fields to compare include `sdk`, `sdk_version`, `python_version`, `app_file`, `suggested_hardware`, `startup_duration_timeout`, and `preload_from_hub`.

A duplicate can copy the repository correctly and still fail because secrets, hardware, dependency resolution, or frontend identity path were not reconstructed.

Primary references:
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-config-reference
- https://www.gradio.app/docs/python-client/client

Use this appendix when a duplicated Space behaves differently from its source.

This appendix explains why duplication can copy the repository accurately while still failing to reproduce the runtime contract, secrets, hardware path, or provisioning assumptions that made the original Space work.

#### E.1 What duplication copies

A duplicate can reproduce the repository very well while still failing to reproduce the original runtime contract.

Treat duplication as two layers:

1. copy the Space
2. recreate the runtime conditions that made the source work

#### E.2 What duplication does not copy

In practice, secrets and some runtime assumptions are the main missing pieces.

That is why a duplicate can look healthy while still failing in ways the source does not.

#### E.3 Why a faithful duplicate still behaves differently

Common reasons:

- different hardware in practice
- unknown original hardware assumptions such as A100, H200, or RTX Pro 6000 Blackwell
- missing secrets
- current dependency resolution differs from the old stable build
- startup cache state differs
- frontend or auth path differs

### Appendix F. HF source taxonomy and update discipline

Use this appendix when you need to decide which Hugging Face source to trust first.

This appendix is about source discipline: current docs, changelogs, staff-maintained Spaces, discussions, issues, and fallback contact paths. It is the guide’s map of evidence quality.

#### F.1 Official docs and changelog

Use these as the default source of truth for current behavior.

#### F.2 HF Staff-maintained Spaces

Treat actively maintained official or staff-operated Spaces as high-value operational references.

#### F.3 Blogs and posts

Use these for rationale, direction, and rollout context.

#### F.4 Discussions and forum threads

Use these for real-world breakage patterns, edge cases, and staff replies that have not yet made it into formal docs.

#### F.5 GitHub issues

Use these to confirm whether a problem is already known, version-specific, or still unresolved.

#### F.6 Why you should always check the latest docs and issue trackers

Hugging Face, Gradio, Transformers, and `huggingface_hub` evolve quickly.

For troubleshooting, the safe order is:

1. official docs
2. changelog / migration guide
3. issue tracker
4. discussions / forum

Do not rely on old snippets alone.

#### F.7 Support contact notes

If Hugging Face documentation gives a dedicated support address for a specific issue type, use that address first.

Examples:

- billing: `billing@huggingface.co`
- security: `security@huggingface.co`

If no dedicated route is clearly documented and the issue is primarily website / account / access related, `website@huggingface.co` is a practical fallback used in community guidance.

### Appendix G. Using LLMs as debugging assistants

Use this appendix when you want to attach this guide, logs, and configs to an LLM without over-trusting the result.

This appendix focuses on workflow, not hype. It separates the parts LLMs are good at from the parts that still need current docs, issue trackers, and human verification.

#### G.1 What LLMs are good at

LLMs are unusually good at compressing long build logs, normalizing stack traces, comparing two Space configurations, proposing search queries, and turning a noisy failure report into a short list of hypotheses. They are especially useful when you already collected the raw evidence but need help organizing it.

#### G.2 What LLMs are bad at

LLMs are bad at silently up-to-date package knowledge, exact quota behavior, undocumented product changes, and library APIs that changed recently. They can invent environment variables, misremember deprecations, or suggest stale fixes from older major versions.

#### G.3 A safe workflow for LLM-assisted debugging

A safe order is:

1. collect logs, YAML, dependency pins, and exact error messages
2. check latest docs and changelogs
3. check current GitHub issues and relevant forum threads
4. only then ask the LLM to summarize, compare, or suggest the next diagnostic step

#### G.4 Prompt patterns that work well for Spaces

Useful prompts include:

- “Classify this Space failure as build, startup, runtime, API/auth, storage, or ZeroGPU quota/path.”
- “Turn this stack trace into the next five checks I should perform.”
- “Compare these two README YAML blocks and tell me what could make the duplicate behave differently.”
- “Normalize this Gradio 6 chat history into the text-only message format expected by a tokenizer chat template.”

#### G.5 Failure modes of LLM-assisted debugging

Common failure modes are hallucinated APIs, fake environment variables, stale package advice, and overconfident recommendations that skip direct evidence from logs or docs. Treat LLM suggestions as hypotheses, not as proof.

#### G.6 How to use this guide itself with an LLM

This guide can be attached alongside logs, YAML, or stack traces as a classification aid. A useful pattern is to attach: the failing logs, the README YAML, your dependency files, and this guide, then ask the LLM to classify the failure first before proposing changes.

### Appendix H. How to navigate Hugging Face when you do not know where information lives

Use this appendix when you do not yet know whether the answer belongs in Docs, a forum thread, a repo discussion, a changelog, or an official Space.

This appendix gives a practical map of Hugging Face as a site and documentation system so you can find the right layer faster.

#### H.1 A practical map of Hugging Face

When debugging Spaces, the main places to look are:

- **Docs** for current official behavior
- **Spaces Changelog** for recent product changes
- **Forum** for real-world breakage reports and staff replies
- **Repo Discussions** for model- or Space-specific issues
- **Staff-maintained Spaces** for operational examples
- **GitHub issues** for current library regressions

#### H.2 Where ZeroGPU information tends to appear first

ZeroGPU information may appear in docs, but operational details also show up in staff-maintained Spaces, blog posts, community organizations like ZeroGPU Explorers, Discussions, and GitHub issues for Gradio or related libraries.

#### H.3 Where storage and account issues tend to appear

Storage behavior is primarily documented in Spaces storage docs and Buckets docs. Account and billing issues may appear in docs, forum staff replies, and support guidance. If a dedicated address is documented, use it. If not, `website@huggingface.co` is a practical fallback for website/account issues.

### Appendix I. Search and research basics for Spaces debugging

Use this appendix when you have a symptom but not yet a good search plan.

This appendix turns raw failures into better queries, better evidence gathering, and better follow-up habits. It is about research method more than platform behavior.

#### I.1 Turn failures into good search queries

A good search query contains the exact error string, the library name, the product surface, and the likely version boundary. Example patterns:

- `"No space left on device" Hugging Face Spaces HF_HOME`
- `Gradio 6 apply_chat_template list content TypeError`
- `ZeroGPU browser works API fails token path`
- `Space stuck on Building Build queued startup_duration_timeout`

#### I.2 Search order

A reliable search order is:

1. official docs
2. changelog / migration guide
3. GitHub issues
4. forum / repo discussions
5. broader web search

#### I.3 How to ask ChatGPT or Gemini useful questions

Provide raw evidence, not just a summary. Include:

- exact error text
- build vs runtime stage
- README YAML
- requirements / packages files
- whether the failure occurs on the standard HF page or only through API/custom frontend
- the smallest failing input

#### I.4 When other AI communities can help

Other AI communities can be useful for package-level or model-level failures, but they should usually come after the official docs, repo issues, and Hugging Face-specific discussions. They are best treated as clue sources, not as the final authority on product behavior.

### Appendix J. Curated external clue clusters

Use this appendix when you need pattern samples, search accelerators, or community clue clusters rather than primary specifications.

This appendix exists because real debugging often starts from imperfect external signals. The key is to use them as navigation aids, not as final authority.

#### J.1 Why keep external clue clusters at all?
Some links are not primary specifications, but they are still worth keeping because they:
- reflect what users actually get stuck on,
- reveal emerging failure patterns,
- and point quickly toward the right docs, issue trackers, or staff/community workarounds.

Treat them as:
- clue clusters,
- navigation aids,
- and search accelerators,
not as direct authority by themselves.

#### J.2 Good external clue candidates
Useful categories:
- current forum threads with staff replies,
- active GitHub issues on Gradio / Transformers / huggingface_hub,
- staff-managed Spaces and posts,
- ecosystem migration notes,
- model / Space Discussions with concrete reproduction steps.

#### J.3 How to use a clue cluster safely
1. Extract the exact error phrase.
2. Extract the product/runtime qualifier.
3. Find the official docs or migration page.
4. Check whether the issue is current and still open.
5. Only then adopt a workaround into your own Space.

### Appendix K. Canonical search phrases for common failures

Use this appendix when you already have a failure shape and want a fast query you can paste into search.

This appendix is intentionally mechanical. Its value is not explanation. Its value is fast retrieval.

#### K.1 Build and startup
- `"Build queued" Hugging Face Spaces`
- `"stuck on Building" Hugging Face Spaces`
- `"startup never becomes healthy" Spaces`
- `"Job failed with exit code 1" Hugging Face Space`

#### K.2 Runtime and API
- `"No API found" Gradio Spaces`
- `"browser works but API fails" Hugging Face Space`
- `"api_visibility" Gradio 6`
- `"footer_links" Gradio 6`
- `"private Space" "READ token"`

#### K.3 Storage and cache
- `"No space left on device" Hugging Face Space`
- `"HF_HOME" cache Hugging Face`
- `"ephemeral disk" Spaces`
- `"Storage Bucket" Hugging Face Spaces`

#### K.4 ZeroGPU
- `"CUDA must not be initialized in the main process" ZeroGPU`
- `"quota exceeded" ZeroGPU`
- `"torch.compile" ZeroGPU`
- `"AoTI" ZeroGPU`
- `"root-level CUDA" ZeroGPU`

#### K.5 Migration
- `"additional_chat_templates" Transformers`
- `"dataset scripts are no longer supported" datasets`
- `"hf_transfer" Xet migration`
- `"Transformers v5" migration`
- `"Gradio 6" migration`

## 9. Resources

Use Resources after you know what kind of failure you are dealing with.

This chapter is for current docs, migration guides, changelogs, issue trackers, forum threads, and clue clusters that help you verify or extend the main guide.

### 9.0 How to use resources

Do not start here by default.

Start with diagnosis first. Then use Resources to confirm current behavior, check migrations, or follow live issue threads that match your symptom closely.

### 9.1 Current docs to check first

Use these links first when you want the current official explanation.

- Docs home: [Hugging Face Docs](https://huggingface.co/docs)
- Hub docs index: [Hub documentation](https://huggingface.co/docs/hub/index)
- Spaces overview: [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- Spaces configuration reference: [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- Spaces as API endpoints: [Spaces as API Endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- Spaces storage: [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- ZeroGPU docs: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- Hub rate limits: [Hub Rate limits](https://huggingface.co/docs/hub/rate-limits)
- Storage Buckets: [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)

### 9.2 Migration and change-tracking sources

Use these when something used to work and stopped working after an upgrade, rebuild, or infrastructure change.

- Hugging Face changelog: [Hugging Face Changelog](https://huggingface.co/changelog)
- Gradio 6 migration: [Gradio 6 Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)
- Gradio Blocks and event listeners: [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners)
- Gradio changelog: [Gradio Changelog](https://www.gradio.app/changelog)
- `huggingface_hub` migration: [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- Transformers v5 migration: [Transformers MIGRATION_GUIDE_V5.md](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)

### 9.3 Live issue and forum sources

Use these when the official docs explain the model but not the current failure.

- Hugging Face forum: [Hugging Face Forums](https://discuss.huggingface.co/)
- Gradio issues: [gradio-app/gradio issues](https://github.com/gradio-app/gradio/issues)
- Transformers issues: [huggingface/transformers issues](https://github.com/huggingface/transformers/issues)
- `huggingface_hub` issues: [huggingface/huggingface_hub issues](https://github.com/huggingface/huggingface_hub/issues)

Use current issue and forum sources for:
- rollout bugs,
- version-sensitive regressions,
- documentation gaps,
- and fast-moving ZeroGPU or Gradio behavior.

### 9.4 Curated link clusters by problem type

#### Build / startup / stuck on Building
- [How to Fix “Space Stuck on Building” in Hugging Face Spaces](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119)
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Hugging Face Status](https://status.huggingface.co/)

#### Runtime / API / Gradio migration
- [Spaces as API Endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Gradio 6 Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)
- [Gradio Changelog](https://www.gradio.app/changelog)
- [Error: No API Found – forum thread](https://discuss.huggingface.co/t/error-no-api-found/146226)
- [Gradio issue: NO API Found](https://github.com/gradio-app/gradio/issues/8410)

#### Storage / cache / repo layout / data
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)
- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Datasets docs](https://huggingface.co/docs/datasets/index)
- [Tokenizers docs](https://huggingface.co/docs/tokenizers/index)

#### ZeroGPU / quota / queue / runtime behavior
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell forum update](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960/6)
- [ZeroGPU AoTI](https://huggingface.co/blog/zerogpu-aoti)
- [ZeroGPU Explorers](https://huggingface.co/zero-gpu-explorers)
- [ZeroGPU Explorers discussion #104](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104)
- [ZeroGPU Explorers discussion #123](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/123)

### 9.5 Hugging Face navigation quick map

Use this map when you do not yet know where the answer belongs.

- **Hub docs** when you need the official product model.
- **Spaces docs** when the problem is about build, runtime, config, hardware, storage, or API behavior.
- **Migration guides and changelogs** when a dependency or platform change is the likely trigger.
- **Forum / Discussions / GitHub issues** when the symptom looks current, niche, or rollout-related.
- **Staff-managed Spaces and posts** when you need strong operational examples.

### 9.6 Support and contact notes

Use dedicated support addresses first when the docs name them explicitly.

- Billing support: `billing@huggingface.co`
- Security support: `security@huggingface.co`

If no dedicated path is clearly documented and the issue is website/account related, `website@huggingface.co` is a practical fallback.

### 9.7 Resource tiers: what to trust first

Use resources in this order.

#### Tier 1. Current primary sources
Use these first:
- official docs,
- migration guides,
- changelogs,
- official API references,
- official billing/security/support pages.

#### Tier 2. Current secondary sources
Use these when Tier 1 is not enough:
- current GitHub issues,
- current forum threads,
- repo Discussions,
- staff-managed Spaces and posts.

These are useful for rollout bugs, regressions, and edge cases that docs may not explain yet.

#### Tier 3. Historical clues and broader link mines
Use these last:
- older posts,
- broad link collections,
- issue packs,
- historical notes.

These are good for:
- spotting recurring patterns,
- finding search terms,
- finding threads worth following up,
but they should not override current primary sources.

### 9.8 Resource triage: keep, historical clue, or drop

When you add a new external link to this guide, sort it into one of three buckets.

#### Current
Keep the link as a normal resource when it is:
- an official doc, API reference, migration guide, changelog, billing/security/support page,
- a current GitHub issue with a clear reproduction or maintainer signal,
- a current forum thread with strong symptom similarity and useful replies,
- or a staff-managed Space / post that still matches the current platform behavior.

#### Historical
Keep the link, but label it as historical or clue-only, when it is:
- older but still useful for search terms,
- useful mainly because it contains the exact error phrase,
- useful as a pattern sample for a server-side regression,
- or useful as a way to discover related issues and docs.

#### Dead end
Do not surface the link in the main guide when it is:
- a low-quality summary page,
- an extracted link with broken or nonsensical summarization,
- weakly related to Spaces debugging,
- stale without clear diagnostic value,
- or too broad to help the reader take a next step.

### 9.9 Example triage from the current link packs

#### Current
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Spaces as API Endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Gradio 6 Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)
- [Transformers MIGRATION_GUIDE_V5.md](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Billing](https://huggingface.co/docs/hub/billing)

#### Historical
- [How to Fix “Space Stuck on Building” in Hugging Face Spaces](https://discuss.huggingface.co/t/how-to-fix-space-stuck-on-building-in-hugging-face-spaces/175119)
- older ZeroGPU Explorers discussions,
- older Gradio “No API found” issue threads,
- older community posts that expose a now-recognizable failure pattern.

#### Dead end
- link-collection entries whose summaries are malformed or clearly off-topic,
- generic AI links without a direct Spaces-debugging role,
- broad community posts that do not help with diagnosis, search, or next-step routing.
