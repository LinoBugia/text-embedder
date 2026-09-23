# Hugging Face-Centered Migration and Drift-Recovery Guide

---

## 0. Reader Contract

This guide is **not** a general debugging manual.

It is for cases like these:

- a notebook worked a few weeks ago and now fails
- a Space breaks after restart, rebuild, duplication, or hardware change
- a tutorial still looks plausible but no longer matches the current stack
- the same repo behaves differently on Colab, Kaggle, Spaces, and local Jupyter
- a skipped upgrade path causes many small incompatibilities at once
- a library migration guide explains some changes, but not the hosted runtime behavior that actually broke you

This guide is intentionally **non-exhaustive**.

Its priorities are:

1. classify the likely drift layer quickly
2. verify the hypothesis with the smallest credible check
3. separate **short-term recovery** from **durable migration**
4. separate **current fixes** from **historical clues**
5. make rollback vs forward migration an explicit decision, not an accident

The center of gravity is this idea:

> **Official migration guides are necessary, but they are not sufficient.**

They matter. You should read them. But they usually do not fully capture runtime-image drift, platform-side behavior changes, tutorial drift, or skipped-version migration.

---

## 1. How to Use This Guide

### If you need the fastest path

1. Start with **Fast Triage Workflow**.
2. Find the matching **Symptom-First Playbook**.
3. Identify the likely drift layer.
4. Decide whether you need **Recovery First**, **Migration First**, or **Rollback Now, Migrate Next**.
5. Read only the relevant **Layer-First Chapter** after you have a hypothesis.

### If you need a durable fix

Read in this order:

1. **What Counts as Time-Caused Breakage**
2. **Declared Drift vs. Implicit Drift**
3. **Runtime Contract Drift**
4. **Rollback vs. Forward Migration**
5. the relevant **Layer-First Chapter**
6. the relevant **Casebook / Appendix**

---

## 2. What This Guide Covers

### In scope

- HF library / API drift
- dependency drift
- runtime image drift
- Python drift
- Torch / CUDA / kernel / hardware drift
- HF product / platform drift
- notebook / tutorial drift
- external runtime drift
- skipped-version migration
- minor-version accumulated drift

### Out of scope

- simple typos
- generic syntax errors with no time-change component
- beginner debugging unrelated to drift
- exhaustive API references
- one-off private infrastructure incidents that cannot be generalized

### The practical boundary

This guide is for environments where Hugging Face code is rarely “just Python code.”

In practice, behavior is shaped by a moving combination of:

- the library version you installed
- the packages the host preinstalled
- the Python version the host chose
- the current GPU / CUDA reality
- the provider or endpoint model used for inference
- the runtime contract of the host platform

That is why this guide is organized around **drift classification** first and library names second.

---

## 3. What Counts as Time-Caused Breakage

### Working definition

Time-caused breakage means:

> the code, notebook, app, or workflow previously worked, but now fails or behaves differently because some relevant assumption changed over time.

### Typical signs

- “The repo did not change, but the Space broke after rebuild.”
- “This notebook ran on Colab before and now errors at import time.”
- “A tutorial still compiles mentally, but the current library rejects the call shape.”
- “The same model loads locally but fails on Kaggle.”
- “A large upgrade introduced many unrelated-seeming failures at once.”
- “An old issue explains the symptom perfectly, but the workaround no longer works.”

### What this definition deliberately includes

This guide includes failures caused by:

- declared product or library migrations
- hidden environment refreshes
- changed defaults
- changed dependency floors or ceilings
- provider / router model changes
- changed storage assumptions
- runtime resets that erase previously relied-on state
- accumulated minor-version drift

### What it deliberately excludes

If the code never worked at all, drift may still be involved, but this guide is not the best first stop.

---

## 4. Declared Drift vs. Implicit Drift

### Declared drift

Declared drift is the easier class.

It appears in places such as:

- official migration guides
- release notes
- changelogs
- deprecation notices
- updated official docs

Examples:

- `huggingface_hub` v1.0 raises the Python floor to 3.9+ and switches its HTTP stack from `requests`/`aiohttp` to `httpx`.
- Spaces configuration explicitly documents `python_version` and `sdk_version`.
- the Inference Providers docs explicitly frame routed inference around provider-aware calls rather than a single generic mental model.

### Implicit drift

Implicit drift is the harder class.

It is real drift that users encounter, but not always in the place they naturally expect to look.

Examples:

- your host image changed
- your package resolver now selects a different transitive dependency
- your notebook tutorial assumes a loader path that the current library no longer supports
- your Space depended on ephemeral disk contents surviving a restart
- your inference code assumes an older serverless mental model while the current docs center providers and routing
- a migration guide exists, but current task docs or examples still lag behind it

### Why this split matters

A guide that only handles declared drift will miss many of the failures users actually hit.

This guide therefore treats **implicit drift as first-class**.

---

## 5. Runtime Contract Drift

### Definition

Runtime contract drift happens when your code still assumes an older execution contract than the one your current host actually provides.

That contract can include:

- Python version
- preinstalled packages
- allowed package versions
- CUDA and driver expectations
- startup command path
- file-system persistence behavior
- available hardware
- provider-routing behavior
- host-side SDK semantics
- environment variables and secret injection

### Why it deserves first-class treatment

Many breakages that look like “a library problem” are actually runtime-contract mismatches.

This is especially common on:

- Hugging Face Spaces
- Google Colab
- Kaggle Notebooks
- local notebooks trying to mimic hosted environments
- bridge runtimes such as provider-backed inference clients, Spaces-as-API, and external servers

### Core rule

Never assume the repo alone defines the runtime.

The runtime contract is often partially encoded outside the repo, or encoded incompletely.

### What runtime-contract drift looks like in practice

- a Space works until rebuild, because ephemeral disk contents disappear on restart
- a Dev Mode tweak seems to fix the app, but disappears later because it was never committed
- a Colab notebook behaves differently from local Jupyter because the runtime image or accelerator differs
- a Kaggle notebook breaks because the new image changed package or CUDA combinations
- the same inference code changes behavior because provider routing or client defaults changed

---

## 6. Recovery First vs. Migration First

### Recovery

Recovery means:

> get the previously working workload running again with the smallest justified change.

Recovery is often the right first move when:

- the system is user-facing and currently broken
- the failure is clearly recent
- a known-good version or runtime still exists
- a full migration would be too risky during the incident window

### Migration

Migration means:

> move the workload onto a healthier, current, and maintainable path.

Migration is often the right move when:

- the old path is already brittle
- the platform moved on structurally
- the old behavior is no longer documented or supported
- the rollback path is shallow and temporary at best
- the breakage exposed a long-standing hidden assumption

### Editorial rule

Every substantial case study in this guide must separate:

- **Short-Term Recovery**
- **Durable Migration Path**

Those are not the same thing.

A pin that restores a Space today may still leave you on a dead-end runtime contract.

---

## 7. Rollback vs. Forward Migration

### Rollback

Rollback is a tactical tool.

Use it when:

- service restoration matters more than stack health this minute
- the older version is still reproducibly available
- the newer failure surface is too wide to absorb safely right now

### Forward migration

Forward migration is strategic.

Use it when:

- the older stack is already unstable or unavailable
- the host platform changed its model
- you skipped too many versions for selective rollback to stay clean
- your current workaround depends on historical behavior that is fading away

### Recommended discipline

If rollback restores service but clearly preserves a dead path, record that explicitly:

> **Recovered by rollback. Durable fix still pending.**

That wording prevents temporary recovery from being mistaken for a completed migration.

---

## 8. Adjacent Migration, Skipped-Version Migration, and Minor-Version Drift

### Adjacent migration

This is the best-documented class.

Examples:

- `4.x → 5.x`
- `0.x → 1.x`

Official migration guides are most helpful here.

### Skipped-version migration

This is harder and more failure-prone.

Examples:

- `3.x → 5.x`
- tutorial-era code → current stack
- old Space repo snapshot → current hosted runtime

Here, adjacent migration guides are useful but incomplete.

### Minor-version accumulated drift

Do not treat major versions as the only drift source.

Minor and patch updates can still change:

- defaults
- transitive dependencies
- runtime compatibility boundaries
- provider behavior
- auth expectations
- startup behavior
- tutorial validity

### Core rule

This guide treats **skipped-version migration** and **minor-version accumulated drift** as first-class topics, not footnotes.

---

## 9. Fast Triage Workflow

### Step 0 — Freeze evidence before changing anything

Capture:

- full error text
- Python version
- package versions
- Torch / CUDA state
- host name (`Spaces`, `Colab`, `Kaggle`, local, etc.)
- hardware type
- whether the failure followed restart, rebuild, duplication, or hardware change
- whether the same code still works somewhere else

### Step 1 — Ask what changed

Classify the change source as one or more of:

- library/API change
- dependency resolution change
- runtime image change
- Python / CUDA / hardware change
- platform / product model change
- tutorial drift
- skipped upgrade path

### Step 2 — Pick the first likely drift layer

Choose the first best-fit layer:

- HF library/API drift
- dependency drift
- runtime image drift
- Python drift
- CUDA / hardware drift
- HF product / platform drift
- notebook / tutorial drift
- external runtime drift

### Step 3 — Reduce to the smallest still-failing workload

Do not debug the full stack first.

Find the smallest reproduction that still fails.

### Step 4 — Verify the hypothesis with the shortest credible check

Examples:

- compare host package versions
- pin a single version boundary
- test the same call locally and on-host
- duplicate the Space with explicit runtime metadata
- swap provider routing mode
- remove the outdated tutorial-only call path

### Step 5 — Choose the path explicitly

- **Recover now**
- **Migrate now**
- **Rollback now, migrate next**

---

## 10. Symptom-First Playbooks

---

### 10.1 A Hugging Face Space broke after restart, rebuild, duplication, or hardware change

#### Symptom

The same repo used to work, but now the Space fails after a rebuild, restart, duplication, hardware switch, or SDK upgrade.

#### Likely drift layer

- runtime contract drift
- Spaces platform drift
- Gradio / SDK drift
- dependency drift
- hardware / CUDA / PyTorch wheel drift
- duplication drift

#### What changed

Spaces are not just Git repos. The runtime contract also includes Space metadata, SDK selection, Python version, hardware tier, ephemeral disk behavior, and host-side rebuild behavior.

The official docs make several parts explicit:

- Space configuration lives in the README YAML.
- `python_version` defaults to `3.10` unless you set it.
- `sdk_version` controls the Gradio version for Gradio Spaces.
- each new commit triggers rebuild and restart.
- Space disk is ephemeral unless you attach durable storage.
- Dev Mode changes are not automatically persisted.
- ZeroGPU has its own constraints and is only compatible with Gradio Spaces.
- ZeroGPU hardware, PyTorch support, GPU size, and quota details can change independently from your app code.
- duplication copies a repository, but it does not automatically prove that the full runtime contract was recreated.

A repo can therefore remain unchanged while the effective runtime contract still changes.

A current ZeroGPU-specific example is the move from older H200 expectations to NVIDIA RTX Pro 6000 Blackwell-backed ZeroGPU documentation. This kind of platform-side hardware change can surface as CUDA / PyTorch wheel compatibility drift, especially when an older wheel lacks kernels for the active GPU architecture, such as Blackwell-class `sm_120` devices. In that situation, errors such as `CUDA error: no kernel image is available for execution on the device` should be treated as substrate drift first, not as a model-code bug first.

At the time of this update, the ZeroGPU documentation lists PyTorch support beginning at `2.8.0` and includes `2.8.0`, `2.9.1`, `2.10.0`, and `2.11.0` in its supported-version list. Treat those version boundaries as part of the hosted runtime contract, not as an application preference.

#### Was it declared?

Partly.

The host features are documented, but many user-visible failures still appear as implicit drift because users often rely on host defaults, ephemeral files, warm container state, or Dev Mode edits without encoding them into the repo.

#### Fastest verification

Check, in order:

1. the README YAML (`sdk`, `sdk_version`, `python_version`, `app_file`, `app_port`, OAuth flags if relevant)
2. whether the app relied on ephemeral disk contents or files created interactively after startup
3. whether the failure appears only after rebuild, duplication, or hardware change
4. whether Dev Mode was used and uncommitted changes were lost
5. whether a Gradio major or SDK bump happened without a deprecation-cleanup pass
6. whether the hardware changed, especially if GPU-specific or ZeroGPU-specific logic is involved
7. if ZeroGPU is involved, whether the current PyTorch version is one of the ZeroGPU-supported versions and whether the installed CUDA wheel supports the active GPU architecture
8. if the Space was duplicated, whether hardware, secrets, variables, storage, and README YAML were recreated rather than assuming repo copy equals runtime copy

#### Short-term recovery

- pin `python_version`
- pin `sdk_version`
- restore the last known-good Gradio-compatible startup path
- remove reliance on files created at runtime unless they are recreated on startup
- re-test without Dev Mode-only edits
- if the Space is Gradio-based, test locally under the same Python and Gradio version
- if the Space depends on persistent files, attach supported storage or move those artifacts to a supported persistent location
- for ZeroGPU CUDA kernel errors after a hardware refresh, try a ZeroGPU-supported PyTorch version with a CUDA wheel compatible with the active platform before rewriting model logic
- for duplicated Spaces, recreate secrets and verify actual hardware before debugging application code

#### Durable migration path

- encode the runtime contract explicitly in README YAML
- move persistent state to supported storage instead of ephemeral disk
- treat Dev Mode as a debugging aid, not a persistence mechanism
- keep startup deterministic and reproducible from a clean rebuild
- if using GPU-specific logic, document hardware assumptions explicitly
- if using ZeroGPU, align the app with the documented Gradio-compatible execution model instead of treating it as a drop-in GPU tier
- document whether the workload expects ZeroGPU `large` or `xlarge`, and avoid treating those as equivalent to a fixed traditional GPU allocation
- make duplication reproducible as a runtime-contract recreation task, not only as a repository-copy task

#### What not to trust

- “The repo did not change, so nothing changed.”
- “It worked in Dev Mode once, therefore it is fixed.”
- “The file exists on disk, so it will still exist after restart.”
- “Duplicating the Space recreates the same runtime automatically.”
- “A ZeroGPU hardware refresh cannot matter because the repo did not change.”
- “A CUDA kernel error must be a model bug.”

#### References

- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell update PR](https://github.com/huggingface/hub-docs/pull/2474)
- [ZeroGPU Blackwell forum note](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960/6)
### 10.2 Code worked before, but now fails on Colab and not locally

#### Symptom

The notebook still works in local Jupyter or another environment, but breaks in Colab after reconnect, runtime change, or accelerator switch.

#### Likely drift layer

- hosted runtime drift
- dependency drift
- Python drift
- hardware / accelerator drift

#### What changed

Colab is not a stable shared image in the sense many users assume. The docs explicitly distinguish hosted runtimes from local runtimes and provide Docker images for the latter. Colab release notes also surface runtime image changes over time.

“Works on Colab” is incomplete unless you know the runtime type, accelerator, package set, and whether the notebook depends on session-local state.

#### Was it declared?

Partly.

Some changes are reflected in local runtime docs and release notes, but many breakages still feel implicit from the notebook author’s point of view.

#### Fastest verification

Check:

1. hosted runtime vs local runtime
2. current accelerator and whether the notebook assumes GPU state
3. Python and key package versions
4. whether the failure disappears in a fresh local virtual environment matching the host versions
5. whether the notebook depended on session-local files, installs, or restarts
6. whether the same notebook fails immediately after a clean Colab runtime restart

#### Short-term recovery

- reinstall the explicitly needed versions at notebook start
- avoid assuming prior runtime state
- freeze critical versions near the top of the notebook
- re-test after a clean runtime restart
- if the notebook is actually targeting a local contract, use a local Colab runtime instead of guessing against hosted state

#### Durable migration path

- move from “ambient Colab state” to explicit environment checks
- capture versions programmatically at runtime start
- keep a minimal environment bootstrap cell
- test the notebook in at least one second environment so that Colab-specific drift is easier to detect
- keep the notebook valid from a cold start instead of a warm session

#### What not to trust

- “It used to work on Colab, so the notebook is still valid.”
- “Local success proves the hosted runtime is fine.”
- “The runtime reconnect preserved everything that mattered.”

#### References

- [Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)
### 10.3 Code worked before, but now fails on Kaggle and not elsewhere

#### Symptom

The same notebook or repo works locally or on another host, but fails on Kaggle after the environment changed.

#### Likely drift layer

- runtime image drift
- dependency drift
- Python / CUDA drift
- hardware drift

#### What changed

Kaggle notebooks run on maintained Docker images, and Kaggle exposes the image repository and release history publicly. That makes Kaggle a classic case of runtime-image drift: your notebook code may stay the same while the image underneath it moves.

#### Was it declared?

Partly.

The image and releases are public, but many failures still appear to notebook authors as “nothing changed in my code.”

#### Fastest verification

Check:

1. which Kaggle image / release you are effectively on
2. whether package or CUDA expectations changed
3. whether import failures match known image-level transitions
4. whether the same notebook succeeds in a clean local or Colab environment with pinned versions
5. whether your notebook accidentally relied on whatever Kaggle happened to preinstall before

#### Short-term recovery

- install a smaller compatibility shim in the notebook rather than rewriting everything at once
- pin the critical packages that moved under you
- reduce to the smallest reproducible import / load / inference failure
- compare the current runtime against the last known-good host snapshot if you have one

#### Durable migration path

- stop relying on whatever the image happens to preinstall
- explicitly bootstrap the versions your notebook needs
- track host-sensitive libraries and CUDA-sensitive packages as a set rather than one by one
- keep a minimal runtime-report cell so future image drift is easier to spot immediately

#### What not to trust

- “Kaggle is just Python in the cloud.”
- “The notebook did not change, so the environment cannot be the problem.”

#### References

- [Kaggle docker-python repository](https://github.com/Kaggle/docker-python)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell update PR](https://github.com/huggingface/hub-docs/pull/2474)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
### 10.4 An old tutorial, notebook, or forum snippet no longer works

#### Symptom

You copy code from an older tutorial, task page, forum answer, or example notebook. The code looks reasonable, but the current stack rejects it or fails in a different place.

#### Likely drift layer

- notebook / tutorial drift
- skipped-version migration
- library/API drift
- platform / product drift

#### What changed

Old material may still describe the original symptom correctly while no longer prescribing a current solution.

This matters especially when:

- task docs lag behind migration guides
- loader behavior changed
- a tutorial assumes a historical API path
- the host runtime changed around the tutorial

Two strong recurring examples are:

- Transformers task docs or examples lagging the current v5 migration surface
- Datasets examples that still imply historical script-based loading, even though recent issues show `RuntimeError: Dataset scripts are no longer supported`

#### Was it declared?

Sometimes, but not always in the exact place readers copy from.

This is one of the clearest places where **official migration guides are necessary but not sufficient**.

#### Fastest verification

Check, in order:

1. the current official docs for the exact feature
2. the migration guide for the relevant major version
3. whether the copied example predates a removed loader, renamed argument, or old endpoint model
4. whether the current task docs themselves still lag behind the migration guidance
5. whether the code only starts working if you move back to a historical bridge version

#### Short-term recovery

- stop copying the historical snippet verbatim
- replace only the outdated call boundary first
- keep the rest of the example minimal until the current path works
- if the stack is truly historical, use the smallest bridge version only long enough to confirm the failure class or convert the asset you need

#### Durable migration path

- rewrite the example using the current officially documented primitive
- label the old snippet as a historical clue, not a solution
- if the official docs still lag, anchor your migration on the migration guide plus a version-bounded issue rather than on the stale task page alone
- remove historical loaders, arguments, and endpoint assumptions one boundary at a time

#### What not to trust

- upvoted forum answers with no version boundary
- examples that omit dates or versions
- historical docs as though they were current docs
- a bridge version as though it were a destination state

#### References

- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers issue: task docs still reference `pipeline()` in places after v5 migration](https://github.com/huggingface/transformers/issues/43827)
- [Datasets issue: dataset scripts are no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Datasets issue: HC3 docs drift after script-loader removal](https://github.com/huggingface/datasets/issues/8012)
- [Datasets v2.19 loading methods docs (historical)](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods)
### 10.5 Inference calls, auth, router, or provider behavior changed

#### Symptom

Inference calls that used to feel generic now fail, require different parameters, route differently, or behave differently across providers and hosts.

#### Likely drift layer

- HF product / platform drift
- client API drift
- auth drift
- provider-routing drift

#### What changed

The current Hugging Face inference surface is explicitly provider-aware.

The docs describe:

- Inference Providers as the top-level model
- `InferenceClient` as a unified interface across multiple services
- `hf-inference` as the service formerly called “Inference API (serverless)”
- provider selection and request routing as part of the current mental model

Older “one generic serverless endpoint” assumptions now drift conceptually even when the call still looks superficially similar.

It is also easy to confuse:

- the dead legacy `api-inference.huggingface.co` endpoint
- Hub API endpoints
- Spaces as API endpoints
- routed inference providers
- dedicated endpoints

Those are related surfaces, but not identical ones.

#### Was it declared?

Yes, substantially — but users still drift when older assumptions linger in examples or internal abstractions.

#### Fastest verification

Check:

1. whether the code still targets `api-inference.huggingface.co`
2. whether you are using Hub APIs, a Space API, routed provider inference, or a dedicated endpoint
3. whether provider choice is explicit or left on auto behavior
4. whether auth scope / token expectations changed
5. whether you are using the current `InferenceClient` surface or a historical wrapper assumption
6. whether the request still sends legacy nested `parameters={...}` payload assumptions where the current helper expects flat kwargs

#### Short-term recovery

- stop using the dead legacy endpoint
- make provider choice explicit
- reduce the call to the smallest current `InferenceClient` example
- verify the surface type first: Space API vs Hub API vs provider inference
- avoid mixing historical serverless wording with current provider-routing expectations

#### Durable migration path

- adopt the current provider-aware mental model explicitly
- isolate inference-surface logic behind your own wrapper with surface-type checks
- document auth scope and endpoint type instead of relying on memory
- migrate old ad-hoc payload shapes to current helper arguments or router-compatible request shapes

#### What not to trust

- “Inference API” as a single timeless concept
- code that does not distinguish routed provider inference from Space APIs or dedicated endpoints
- auth assumptions copied from a different HF surface
- wrappers that hide the actual target surface

#### References

- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [InferenceClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [Hub Rate limits](https://huggingface.co/docs/hub/rate-limits)
- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [HF Inference provider page](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub API Endpoints](https://huggingface.co/docs/hub/api)
### 10.6 A large version jump broke multiple things at once

#### Symptom

After a big upgrade, many failures appear together: renamed arguments, changed defaults, missing imports, different auth behavior, tutorial breakage, and new runtime assumptions.

#### Likely drift layer

- skipped-version migration
- dependency drift
- changed defaults across adjacent releases
- tutorial-era assumptions collapsing together

#### What changed

This is the signature of skipped-version migration.

The underlying problem is often not one bug, but the loss of many small compatibility assumptions at the same time.

Typical mixed clusters include:

- library major-version drift
- dependency-floor changes
- host runtime changes
- old tutorial assumptions that now fail
- inference-surface model changes

#### Was it declared?

Partly.

Adjacent migration guides may exist. They are still useful. But if you skipped multiple eras, no single migration guide usually covers the full gap.

#### Fastest verification

Pick one compatibility boundary at a time:

1. library major
2. dependency set
3. Python floor
4. host runtime contract
5. inference surface

A practical order is:

- restore or identify the last known-good checkpoint
- move only one major boundary at a time
- verify the smallest failing example at each checkpoint

#### Short-term recovery

- temporarily restore a smaller known-good version interval
- validate one layer at a time
- stop trying to solve all regressions in one edit pass
- use bridge versions when they help confirm the failing boundary, but do not mistake the bridge for the durable target

#### Durable migration path

- migrate by checkpoints
- validate each boundary separately
- update code to the current supported primitive, not to the oldest minimally passing workaround
- document the checkpoint sequence so later upgrades do not collapse into another blind jump

#### What not to trust

- “One giant upgrade will be faster.”
- “The adjacent migration guide fully covers a multi-era jump.”
- “A bridge version that loads once is the final fix.”

#### References

- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Hub Rate limits](https://huggingface.co/docs/hub/rate-limits)
- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Datasets releases](https://github.com/huggingface/datasets/releases)
### 10.7 Hub uploads, downloads, or metadata calls started hitting `429`, `503`, or retry loops

#### Symptom

A script that used to upload or sync Hub assets now fails intermittently, stalls under load, or starts receiving `429 Too Many Requests`, `503`, connection errors, or repeated transient failures.

#### Likely drift layer

- Hub API / resolver rate-limit drift
- client retry behavior drift
- upload strategy drift
- repository-size or large-folder workflow drift

#### What changed

Hub access is not a single unlimited channel. The effective contract includes the request class, authentication state, account or organization plan, client library version, repository shape, upload method, and whether the workload is doing many small API calls, resolver downloads, or large file transfers.

For large uploads, the important distinction is between ordinary one-shot upload methods and resumable / retry-oriented large-folder workflows. A workflow that was acceptable for a few files can become brittle when scaled to many files, many commits, or large LFS objects.

#### Was it declared?

Partly.

Rate-limit headers and current upload helpers are documented, but user failures often appear as generic networking or Hub instability unless the script records status codes, headers, request class, and client version.

#### Fastest verification

1. confirm whether requests are authenticated with `HF_TOKEN`
2. record the HTTP status code and any `RateLimit` header information
3. separate API calls, resolver downloads, and commit/upload operations
4. check the installed `huggingface_hub` version before implementing custom retry logic
5. for large folders, test whether `upload_large_folder()` is a better fit than repeated ad-hoc commits
6. distinguish transient `429` / `503` behavior from permanent permission or repository-structure errors

#### Short-term recovery

- authenticate every script and downstream library that touches the Hub
- reduce request burstiness
- prefer resolver paths where appropriate
- use current `huggingface_hub` retry behavior instead of hand-written sleeps when possible
- for large folders, switch to the resumable large-folder upload path before retrying the whole transfer manually

#### Durable migration path

- treat Hub rate limits and transient upload failures as part of the operational contract
- use resumable upload primitives for large or long-running transfers
- keep a local record of completed work for large sync jobs
- avoid designing workflows that require many unnecessary commits or metadata calls
- separate permanent permission failures from transient network and rate-limit failures in logs

#### What not to trust

- “A failed upload means the file is corrupt.”
- “Retrying the entire folder from scratch is the safest recovery.”
- “Unauthenticated access is equivalent to token-authenticated access.”
- “A fixed sleep is better than using the client’s rate-limit-aware behavior.”

#### References

- [Hub Rate limits](https://huggingface.co/docs/hub/rate-limits)
- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [HfApi Client reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)

## 11. Layer-First Chapters

### 11.1 HF Library / API Drift

Use this layer when the breakage is primarily caused by a changed library surface rather than by the host image alone.

#### Typical signals

- renamed or removed arguments
- changed defaults
- changed object roles (`tokenizer` vs `processing_class`, config vs generation config, etc.)
- task examples or docs that still reflect an older API surface
- custom overrides failing because the framework now passes more information than before

#### Fastest verification

- compare the failing call against the current migration guide or package reference
- reduce the failure to the smallest API boundary that still fails
- check whether the error disappears only when you move back to an older library line

#### Recovery bias

- pin the smallest version interval that restores the boundary
- replace the obviously removed or renamed argument first
- avoid widening the refactor before the smallest failing API call passes

#### Migration bias

- update to the current primitive, not a historical workaround
- follow the official migration guide first, then use selected issues only to close documented gaps
- retest adjacent libraries as a set when the migration surface is shared

#### Library-specific notes

##### `huggingface_hub`

`huggingface_hub` v1.0 is a good example of declared drift that still cascades into environment drift. The migration guide explicitly raises the Python floor to 3.9+ and changes the HTTP backend to `httpx`. That is not only an API concern; it can also surface as dependency drift, proxy behavior drift, and compatibility drift inside hosted notebooks and apps.

Hub upload and rate-limit behavior also belongs in this layer. Current Hub docs document `429 Too Many Requests`, `RateLimit` headers, and `huggingface_hub` smart retry behavior for rate-limit errors. Current upload docs also distinguish ordinary folder uploads from `upload_large_folder()`, which is designed to be resumable and retry-oriented for large transfers.

##### `transformers`

Transformers v5 has an official migration guide, but the live issue tracker already shows why migration guides are not the whole story. Some issues expose gaps or edge cases after the migration surface changed — for example, task docs lagging behind migration guidance, tokenizer refactor fallout, or initialization-related migration edge cases. In practice, the most discoverable task recipe is not always the most current one for the installed major version; see Appendix F.1.

##### `datasets`

Datasets is a major tutorial-drift hotspot because historical examples can remain discoverable long after the preferred loading path changed. Recent issues reporting “dataset scripts are no longer supported” are exactly the kind of drift this guide treats as first-class. Media-heavy paths also deserve a separate backend check rather than being treated as generic dataset corruption; see Appendix F.2.

##### `gradio`

Gradio can create combined library + host-runtime drift, especially on Spaces. A Gradio migration can change launch behavior, footer/API visibility, chat formatting assumptions, and host interaction in one move.

#### What not to trust

- a migration issue comment as though it replaced the migration guide
- stale task-page examples as though they were versionless
- a local-only success as proof that a hosted Gradio app is migrated

#### References

- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers issue: docs lag after migration](https://github.com/huggingface/transformers/issues/43827)
- [Transformers issue: tokenizer refactor edge case](https://github.com/huggingface/transformers/issues/44361)
- [Transformers issue: `_is_hf_initialized` edge case](https://github.com/huggingface/transformers/issues/43632)
- [Datasets issue: dataset scripts no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Gradio 5 migration issue](https://github.com/gradio-app/gradio/issues/9463)
- [Gradio Blocks and event listeners](https://www.gradio.app/guides/blocks-and-event-listeners)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell update PR](https://github.com/huggingface/hub-docs/pull/2474)

---

### 11.2 Dependency Drift

Use this layer when the breakage comes from version combinations, resolver behavior, optional extras, or transitive packages rather than from one obvious API rename.

#### Typical signals

- import failures after a library upgrade that look unrelated to the upgraded library itself
- code that works only because a host happened to preinstall a compatible combination before
- breakage that appears only on one host image or only after a clean reinstall
- proxies, HTTP, serialization, or media packages suddenly behaving differently after a seemingly unrelated upgrade

#### Fastest verification

- capture Python and package versions before changing them
- compare a critical-package subset rather than the full environment first
- test whether the smallest failing import or call succeeds once the adjacent versions are aligned
- check whether the failure disappears only when the host-preinstalled stack is bypassed

#### Recovery bias

- pin the smallest set needed to verify the boundary
- prefer narrow and explainable compatibility shims over large blind downgrades
- keep a minimal reproduction that proves the dependency boundary was the problem

#### Migration bias

- align the dependency set deliberately rather than adding one-off pins forever
- treat dependency drift as part of the migration plan, not as a temporary nuisance
- document the version set that actually forms the working contract

#### What not to trust

- “Only one package changed, so only one package matters.”
- “The host preinstalled it, so it must be part of the stable contract.”
- “A giant freeze file is the same as understanding the compatibility boundary.”

#### References

- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)
- [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)

---

### 11.3 Runtime Image Drift

Use this layer when your code stayed the same, but the host image or container changed under it.

#### Typical signals

- the repo did not change, but the failure started after restart, rebuild, or reconnect
- the same notebook fails on one hosted platform but not another
- imports fail only on the maintained host image
- a Space or notebook works only in a warm session and not from a clean start

#### Fastest verification

- identify the host and effective image/release era
- compare a cold start with a warm session result
- compare a minimal environment snapshot across hosts
- check whether the code relied on preinstalled packages, warm caches, or files created outside the repo

#### Recovery bias

- verify the image-era hypothesis before refactoring application code
- restore the smallest known-good runtime contract if possible
- keep the failing case minimal so the image boundary stays visible

#### Migration bias

- stop assuming the maintained image is part of your stable contract unless you encode it
- bootstrap the required environment explicitly
- make the workload cold-start reproducible

#### What not to trust

- “The code didn’t change, so the environment didn’t matter.”
- “A warm container result proves the rebuild path is healthy.”
- “Works on Kaggle/Colab/Spaces” without version capture.

#### References

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)

---

### 11.4 Python, Torch, CUDA, Kernel, and Hardware Drift

Use this layer when the real break is the execution substrate: interpreter floor, wheel availability, CUDA compatibility, or hardware-sensitive behavior.

#### Typical signals

- a library suddenly refuses the active Python version
- a package imports on CPU-only local environments but fails on GPU hosts
- a model path works on one accelerator tier and fails on another
- installation or import behavior changes only when CUDA is present
- `CUDA error: no kernel image is available for execution on the device`
- the visible GPU name or CUDA capability changes after a host or hardware refresh, for example to a Blackwell-class `sm_120` device
- a PyTorch wheel built for older CUDA / GPU architectures lands on a newer hosted GPU

#### Fastest verification

- capture Python version first
- capture Torch / CUDA / hardware state second
- capture the CUDA device name and compute capability when CUDA is available
- compare the same minimal repro across CPU and GPU or across two GPU tiers if possible
- check whether the package line you are on still supports the active interpreter and runtime assumptions
- for ZeroGPU, check the current supported PyTorch versions before treating a CUDA failure as application logic

#### Recovery bias

- return to the last known-good Python / Torch / CUDA trio if service restoration matters immediately
- for ZeroGPU hardware-refresh failures, test a documented ZeroGPU-supported PyTorch version with a compatible CUDA wheel
- keep the reproduction below the application layer until the substrate is verified
- avoid changing model logic before you know the runtime substrate is valid

#### Migration bias

- migrate onto a supported interpreter + Torch + CUDA combination deliberately
- keep hardware architecture, CUDA wheel support, and hosted-runtime constraints together as one substrate contract
- document the supported substrate instead of relying on ambient host luck
- retest hardware-sensitive workloads after host changes even when the code looks unchanged

#### What not to trust

- “It imports locally, so the GPU host is fine.”
- “The error mentions a model, so the substrate cannot be the issue.”
- “Python/Torch/CUDA can be debugged independently in every case.”
- “The GPU name is a cosmetic detail.”
- “Old CUDA wheels will remain valid across hosted GPU refreshes.”

#### References

- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)

---

### 11.5 HF Product / Platform Drift

Use this layer when the problem is not “inside the model code” at all, but in the Hugging Face surface you are actually calling or deploying on.

#### Common sub-surfaces

- Hub APIs
- routed inference providers
- Space runtime management
- Spaces as API endpoints
- auth scopes or OAuth-related assumptions
- ZeroGPU-specific execution assumptions

#### Typical signals

- auth works for one HF surface and not another
- the same logical inference task behaves differently once routed through a different provider surface
- a Space works in the browser but your API path or wrapper logic breaks
- code still assumes an older serverless endpoint model

#### Fastest verification

- identify the exact HF surface first
- verify the auth scope and endpoint type for that surface
- confirm whether your wrapper hides provider choice, endpoint type, or API family
- reduce to the smallest current official example for that surface

#### Recovery bias

- make the surface type explicit before changing payload logic
- stop using dead or historical endpoint assumptions immediately
- verify auth on the smallest current call shape first

#### Migration bias

- document product-surface type, provider choice, and auth expectations explicitly
- keep Hub APIs, Spaces APIs, and routed provider inference separate in your abstraction layers
- treat platform drift as first-class, not as noise around model code

#### What not to trust

- “Hugging Face API” as though all surfaces were one thing
- auth fixes copied across surfaces without checking scope
- wrappers that hide which HF surface they actually target

#### References

- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [InferenceClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [HF Inference provider page](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub API Endpoints](https://huggingface.co/docs/hub/api)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)

---

### 11.6 Notebook / Tutorial Drift

Use this layer when the instructional artifact is the thing that went stale.

#### Typical signals

- the example still looks plausible, but the current stack rejects it
- an old notebook works only if you downgrade into a bridge version
- task docs and migration guides disagree on the currently valid primitive
- multiple users report the same “example broke” pattern in issues or forums

#### Fastest verification

- check the date and version boundary of the example
- compare the example against the current official docs and migration guide
- isolate the first historical boundary: loader, argument name, endpoint model, startup assumption, or host behavior

#### Recovery bias

- replace only the stale boundary first
- keep the rest of the example minimal until the current primitive works
- use a bridge version only to prove the failure class or convert an old asset

#### Migration bias

- rewrite the example on the current primitive
- label old material as a historical clue, not current authority
- remove hidden host assumptions from the instructional path

#### What not to trust

- upvotes with no version boundary
- old task pages treated as current truth
- a bridge version as a durable migration target

#### References

- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers issue: docs lag after migration](https://github.com/huggingface/transformers/issues/43827)
- [Datasets issue: dataset scripts are no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Datasets v2.19 loading methods docs (historical)](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods)

---

### 11.7 External Runtime Drift

Use this layer when Hugging Face code is only one side of the contract and another runtime or bridge layer is the real moving part.

#### Examples

- local inference servers
- provider-backed wrappers
- third-party bridges used from notebooks and apps
- custom frontends calling Spaces or Hub surfaces indirectly

#### Typical signals

- Hugging Face code looks correct in isolation, but the surrounding runtime wrapper changed
- the same request shape behaves differently when routed through a bridge layer
- local and hosted inference behave differently even when the model and prompt are the same

#### Fastest verification

- reduce the repro until the external runtime boundary is unmistakable
- test the same logical request against the smallest official HF surface you can
- separate “HF surface drift” from “bridge runtime drift” before changing both at once

#### Recovery bias

- bypass the extra bridge layer temporarily if you can
- make the runtime boundary explicit in logs and wrappers
- keep the repro below the application orchestration layer until the contract is visible

#### Migration bias

- treat the external runtime as part of the contract, not as invisible plumbing
- document which layer owns routing, auth, formatting, and retry behavior
- keep your own adapters thin enough that surface drift is easy to localize

#### What not to trust

- “The wrapper abstracts this away, so the boundary does not matter.”
- “Different runtimes returning different behavior means the model changed.”
- “If the local server passes, the hosted route must be equivalent.”

#### References

- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [InferenceClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub API Endpoints](https://huggingface.co/docs/hub/api)

---
## 12. Host-Specific Chapters

### 12.1 Hugging Face Spaces

#### What the runtime contract includes

- README YAML
- SDK selection
- `python_version`
- `sdk_version`
- hardware tier
- storage behavior
- startup path
- API / OAuth / secret expectations
- rebuild / restart behavior
- ZeroGPU-specific constraints where relevant
- duplication behavior: repository copy, variables, secrets, hardware selection, and actual assigned runtime
- ZeroGPU `large` / `xlarge` sizing and quota cost where relevant
- current PyTorch / CUDA wheel compatibility where GPU architecture matters

#### Diagnostic prompts

- Did the failure start after rebuild, restart, duplication, hardware change, SDK change, or visibility change?
- Is the runtime contract fully encoded in README YAML, or partially living in memory / Dev Mode / warm state?
- Does the app depend on files that were created after startup or outside the repo?
- Is the failure really an app bug, or a `public` / `protected` / `private` visibility or auth-surface mismatch?
- Is the API path different from the browser path?
- Is the app assuming ordinary GPU behavior while actually running on ZeroGPU?
- Is the app assuming historical ZeroGPU hardware or quota details after a hardware documentation refresh?
- Does the duplicate have the same secrets, hardware request, storage assumptions, and README YAML as the source?
- Is `sdk_version` acting as the Gradio version authority while `requirements.txt` also tries to pin `gradio`?
- Is the config still treating `suggested_storage` as if it provisions storage for the Space?
- Does a custom component or frontend bundle create an extra rebuild-specific failure surface beyond ordinary Python dependencies?
- Do the documented contract and the actual runtime image from this run appear to disagree?

#### Recovery patterns

- verify YAML before refactoring app logic
- verify the current visibility and auth surface before treating access failures as application regressions
- pin `python_version` and `sdk_version`
- if ZeroGPU is involved, verify the current documented PyTorch support and GPU size before changing model code
- keep one clear authority for the Gradio version on Gradio Spaces instead of letting README YAML and `requirements.txt` disagree
- treat `suggested_storage` as a hint field, not as current storage provisioning
- test from a clean rebuild, not only from a warm container
- if custom components are involved, verify clean rebuild and loader behavior separately from basic browser-path success
- when docs and behavior diverge, capture code snapshot + build log + container/runtime log from the same run
- treat the default disk as ephemeral and move only the necessary state to a currently supported persistent location when persistence is the issue
- reduce launch to the smallest working startup path

#### Migration patterns

- make clean-rebuild success part of the contract
- treat Dev Mode as a debugging tool, not persistence
- separate browser-path assumptions from API-path assumptions
- separate the documented Spaces contract from the actual runtime image you are diagnosing
- encode hardware, visibility, and storage assumptions explicitly
- treat duplication docs and UI defaults as part of the runtime contract, not as a guarantee of equivalence

#### What not to trust

- “The repo is the environment.”
- “A duplicate should behave identically by default.”
- “The browser path worked, so the API path is equivalent.”
- “`suggested_storage` means storage is already provisioned.”
- “A startup warning alone proves platform incompatibility.”
- “ZeroGPU `large`, ZeroGPU `xlarge`, and standard GPU Spaces are interchangeable.”
- “A duplicate that opened successfully recreated secrets, hardware, and quota behavior.”

#### Seed references

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [Using GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Managing your Space runtime](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime)

---

### 12.2 Google Colab

#### What the runtime contract includes

- hosted vs local runtime
- runtime image / package set
- accelerator choice
- file-system and session assumptions
- restart behavior
- notebook bootstrap logic

#### Diagnostic prompts

- Is this a hosted runtime problem or a local runtime mismatch?
- Did the notebook rely on packages installed earlier in the session?
- Does the issue appear only after reconnect or accelerator change?
- Does the notebook still work from a cold start with the bootstrap cells rerun from the top?

#### Recovery patterns

- capture environment state at notebook start
- reinstall the explicitly required versions in a single bootstrap cell
- retest from a clean runtime restart
- compare against a matching local environment when the host looks suspicious

#### Migration patterns

- make the notebook valid from a cold start
- keep environment capture and bootstrap near the top
- stop depending on ambient state left over from an earlier session
- maintain at least one second environment for comparison

#### What not to trust

- “It works after a few retries, so the notebook is healthy.”
- “Local success proves the hosted runtime is unchanged.”
- “Session state is part of the durable contract.”

#### Seed references

- [Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)

---

### 12.3 Kaggle

#### What the runtime contract includes

- image release cadence
- preinstalled package set
- accelerator-sensitive differences
- CUDA-sensitive differences
- startup-time bootstrap behavior

#### Diagnostic prompts

- Which image / release am I effectively on?
- Is the failure image-specific or generally reproducible?
- Did the notebook rely on something Kaggle happened to preinstall before?
- Does the issue disappear when the same package set is recreated elsewhere?

#### Recovery patterns

- compare against the image/release history first
- add the smallest compatibility shim that restores the failing boundary
- keep the repro below the full notebook when possible

#### Migration patterns

- stop treating the maintained image as your stable environment definition
- bootstrap the packages your notebook truly needs
- preserve a minimal environment snapshot so future image drift is easier to identify

#### What not to trust

- “Kaggle is just another Python shell.”
- “If it used to import, the host image must still be equivalent.”

#### Seed references

- [Kaggle docker-python repository](https://github.com/Kaggle/docker-python)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)

---

### 12.4 Cross-Host Comparison

When the same code behaves differently across hosts, compare these first:

- Python version
- critical package subset
- Torch / CUDA state
- visible hardware
- startup command path
- whether files are assumed to persist
- auth / endpoint surface type
- whether the failure requires a warm session to reproduce or disappear

Use cross-host comparison to confirm a runtime-contract hypothesis, not to skip classification.

---

## 13. Recovery Patterns

### 13.1 Freeze, Minimize, Verify

This is the default opening move.

- **Freeze:** preserve errors, versions, host, and hardware facts before changing them away.
- **Minimize:** reduce the failing workload until the boundary is obvious.
- **Verify:** choose the smallest test that can falsify the current guess.

### 13.2 Known-Good Restoration

A rollback is acceptable when it restores service and is named honestly as temporary.

Use it when:

- a last known-good boundary exists
- the current incident window is too small for a forward migration
- the rollback target is still available and reproducible

Do not let “service restored” silently become “migration completed.”

### 13.3 Smallest Justified Pinning

Prefer narrow, explainable pins to large unexamined environment dumps.

Good pinning:

- proves a specific boundary
- is easy to reverse later
- makes the migration debt visible

Bad pinning:

- hides the real failing layer
- locks the stack blindly
- becomes the permanent state without explanation

### 13.4 Host Swap as Verification, Not as Proof

If the code works on one host and not another, that is evidence for runtime-contract drift. It is not proof that the code is healthy.

Use host swaps to:

- distinguish host drift from pure application-code bugs
- compare runtime contracts
- confirm whether a failure needs cold-start reproduction

Do not use host success as a substitute for classification.

### 13.5 Temporary Workarounds Must Not Erase Migration Debt

Every temporary fix should make the pending durable migration clearer, not blurrier.

A good temporary workaround should leave behind:

- what it bypasses
- what it does not solve
- what durable migration still remains

### 13.6 Recovery Decision Template

When writing or reviewing a recovery note, force these three lines:

- **Recovered by:** [exact short-term action]
- **Likely drift layer:** [best current classification]
- **Durable migration still needed:** [yes / no, with why]

---

## 14. Durable Migration Patterns

### 14.1 Move from Historical Path to Supported Path

Replace stale tutorial-era primitives with the current documented ones.

Do not preserve a historical path merely because it still half-works under a bridge version. For a concrete version-bound example, see Appendix F.2.

### 14.2 Collapse Implicit Assumptions into Explicit Config

If the host matters, encode the host assumptions.

Examples:

- Python version
- SDK version
- startup path
- storage expectations
- provider choice
- auth surface

This is the opposite of “works because the environment happened to be warm.”

### 14.3 Align Adjacent Libraries as a Set

Do not migrate one library in isolation if the real contract boundary is shared. For adapter-targeting drift and supervised-token-span drift in training stacks, see Appendix F.5.

Examples:

- Hub client + inference surface
- Transformers + adjacent training stack
- Gradio + Spaces runtime behavior

### 14.4 Migrate by Checkpoints, Not One Giant Leap

This is especially important for skipped-version migration.

Checkpoint migration means:

- pick a stable intermediate target
- validate the smallest still-failing example there
- move one major boundary at a time
- document what broke and what changed at each checkpoint

### 14.5 Replace Convenience Myths with Explicit Surface Types

Examples:

- Space API vs Hub API vs routed provider inference
- hosted image vs local runtime
- ephemeral disk vs durable storage
- browser path vs API path

Many migration failures persist only because the surface type stayed implicit.

### 14.6 Migration Checkpoint Template

When planning a durable migration, force these lines:

- **Current failing checkpoint:** [version / host / surface]
- **Target checkpoint:** [next version / host / surface]
- **Boundary being tested:** [API, dependency set, runtime image, substrate, platform surface]
- **Proof of success:** [smallest test that must pass]

---
## 15. What Not to Trust

Drift recovery fails as much from **misplaced trust** as from missing information.

### High-risk sources of false confidence

Do not over-trust:

- old forum answers with no version boundary
- the most discoverable task example treated as the current one without checking the installed major-version guidance
- issue comments treated as universal solutions
- tutorials with no date or version context
- adjacent-version migration guides as complete answers for skipped migrations
- “works on my Colab” evidence with no runtime metadata
- “the repo did not change” as proof that the environment did not change
- Dev Mode edits as proof of persistent fixes
- bridge versions that restore one path as though they were durable endpoints

### What stale advice still helps with

Old advice can still explain the symptom, point to the likely layer, or reveal the older mental model your code still assumes.

That makes old advice a clue, not authority.

### Trust hierarchy

1. current official docs / migration guides / changelogs
2. current runtime or environment docs
3. releases and selected issues
4. forum threads and community fixes
5. historical clues, clearly labeled as historical

### Operational rule

When a source is not clearly current, ask:

- what version or date boundary does it belong to?
- is it explaining the symptom, or prescribing the current fix?
- does it match the exact HF surface and host I am using?

---

## 16. Section Template for Major Case Studies

Use this exact structure whenever possible.

### Symptom

Describe the failure pattern in plain language first.

### Likely drift layer

List the most likely layers in order of probability, not alphabetically.

### What changed

State the most plausible change class and the boundary where it likely happened.

### Was it declared?

Use one of:

- **Yes**
- **Partly**
- **No**
- **Unknown**

Then state where the declaration actually lives.

### Fastest verification

Give the smallest credible check that can falsify the current hypothesis.

### Short-term recovery

Give the smallest justified restoration path.

### Durable migration path

Give the forward path that removes the hidden assumption rather than preserving it.

### What not to trust

List stale assumptions, misleading historical guidance, and tempting false equivalences.

### References

- Tier 1 first
- Tier 2 next
- Tier 3 only where it closes a real gap
- Tier 4 only when it adds verification value
- Tier 5 only as explicitly labeled historical clue

### Writing rules for this template

- keep recovery and migration separate
- keep current fixes and historical clues separate
- keep rollback and forward migration separate
- keep the section standalone

---

## 17. References Discipline

### Per-section rules

Each substantial section should include:

- at least one Tier 1 or Tier 2 source
- explicit version / date boundaries where relevant
- historical sources labeled as historical
- issue threads used only to close a verified gap
- source ordering that makes current guidance easier to see than old clues

### Source tiers

- **Tier 1:** official docs / changelog / migration docs
- **Tier 2:** runtime docs / environment docs / image releases
- **Tier 3:** GitHub releases / issues / release threads
- **Tier 4:** forum threads / community fixes
- **Tier 5:** historical clues

### Current-Fix Lane vs Historical-Clue Lane

When using this guide, keep two lanes in mind:

#### Current-Fix Lane

Use this lane for:

- the fix a reader should try now
- the migration path you want readers to adopt
- the current supported model of the product or library

This lane should be anchored on Tier 1 and Tier 2 whenever possible.

#### Historical-Clue Lane

Use this lane for:

- explaining why old code or old tutorials looked reasonable
- explaining why the symptom still appears in search results
- mapping a failure to an older era or model

This lane should never be allowed to masquerade as the current general solution.

### Editorial warning

Do not let a historical clue become prose that sounds like the current general solution.

### Citation hygiene

- prefer current primary sources over commentary
- use issues to expose gaps, not to replace primary docs
- keep the cited surface exact: Hub API, Space API, provider-routed inference, runtime docs, or library migration docs
- when the source only explains the symptom historically, say so explicitly

---
## 18. Appendices

### Appendix A — Symptom Index

| Symptom | Most likely drift layer | Fastest verification | Primary chapter | Source lane |
|---|---|---|---|---|
| Space breaks after restart or rebuild | runtime contract drift / Spaces platform drift | inspect README YAML, storage assumptions, SDK/Python pins | 10.1, 12.1 | Tier 1 + Tier 2 |
| Notebook works locally but fails on Colab | hosted runtime drift / dependency drift | compare hosted vs local runtime, accelerator, package versions | 10.2, 12.2 | Tier 2 + Tier 1 |
| Notebook fails on Kaggle and nowhere else | runtime image drift | inspect Kaggle image / release era and package set | 10.3, 12.3 | Tier 2 + Tier 1 |
| Old tutorial code no longer runs | tutorial drift / skipped-version migration | compare current docs and migration guide with copied snippet | 10.4, 11.6 | Tier 1 + Tier 3 |
| Inference or auth calls changed behavior | HF product/platform drift | identify the exact surface: Hub API, Space API, provider-routed inference, dedicated endpoint | 10.5, 11.5 | Tier 1 |
| Many unrelated failures appear after one upgrade | skipped-version migration / dependency drift | split by boundary: library major, dependency set, Python floor, host runtime | 10.6, 8 | Tier 1 + Tier 3 |
| UI or launch behavior changed after Gradio migration | library/API drift + host-runtime drift | verify Gradio major, SSR expectations, Spaces behavior | 11.1, 12.1 | Tier 1 + Tier 3 |
| Dataset loading snippet that used to work now throws loader errors | tutorial drift / library drift | check whether the loader model itself changed | 10.4, 11.1 | Tier 1 + Tier 3 |

---

### Appendix B — Drift Timeline / Breakage Atlas

This appendix is not a history lesson. It is a diagnostic map.

| Change era / cluster | What tends to break | Likely drift layer | Recovery bias | Durable migration bias |
|---|---|---|---|---|
| Historical tutorial / script-loader era | copied examples, dataset loading, stale task pages | tutorial drift / skipped-version migration | replace only the stale boundary first | rewrite on top of current documented primitive |
| `huggingface_hub` v1.0 era | auth/client behavior, HTTP stack assumptions, Python floor issues | library/API drift + dependency drift | verify Python and client path first | migrate to current Hub client model |
| Transformers v5 migration era | renamed args, docs lag, tokenizer or init edge cases | library/API drift + skipped-version migration | isolate the exact migrated surface | follow migration guide, then close gaps with issue-backed fixes |
| Gradio 5 / Spaces SSR era | launch behavior, frontend/runtime expectations, Space rebuild surprises | library/API drift + runtime contract drift | pin Gradio and verify Space metadata | encode runtime assumptions explicitly |
| Inference Providers era | generic serverless mental model no longer fits routed provider surface | HF product/platform drift | reduce to current `InferenceClient` examples | make endpoint/surface type explicit in your code |
| Colab / Kaggle image churn | imports, package conflicts, GPU-sensitive behavior | runtime image drift / dependency drift | reproduce with version capture | move from ambient host state to explicit bootstrap |

---

### Appendix C — Historical Clues and Dead Ends

Historical clues are useful when they explain **why a symptom feels familiar**. They are dangerous when they are copied into the present without a version boundary.

#### Historical clue

Older dataset-loading material that still assumes a script-based loading path.

#### Why it remains useful

It explains why users still search for or remember the old workflow.

#### Why it is dangerous

Recent issues reporting `RuntimeError: Dataset scripts are no longer supported` show that the old path can no longer be treated as current advice.

#### Historical clue

Older “Inference API” wording that encourages a single generic serverless mental model.

#### Why it remains useful

It explains why many codebases still use one abstraction for unrelated inference surfaces.

#### Why it is dangerous

Current docs are explicitly provider-aware and distinguish routed provider inference from Hub APIs and Spaces-as-API.

#### Historical clue

A Dev Mode tweak that made a Space work once.

#### Why it remains useful

It can reveal the real missing configuration or file dependency.

#### Why it is dangerous

Dev Mode does not automatically persist those changes, so the observed success can be diagnostic without being durable.

#### Operational rule

When a historical clue appears in this guide, it must be labeled as one of:

- **Historical clue**
- **Still diagnostically useful**
- **Not a current general fix**

#### References

- [Datasets v2.19 loading methods docs (historical)](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods)
- [Datasets issue: dataset scripts are no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)

---

### Appendix D — Recovery Checklists

#### D.1 General recovery checklist

- preserve the full error text
- record Python and package versions before changing them
- record host name and hardware type
- ask what changed outside the repo
- reduce to the smallest still-failing workload
- test one hypothesis at a time
- separate “recovered” from “migrated” in your notes

#### D.2 Space recovery checklist

- inspect README YAML
- verify `python_version` and `sdk_version`
- verify `app_file` and startup path
- check for ephemeral-disk dependence
- check whether Dev Mode-only edits were lost
- check hardware tier and any GPU-only assumptions

#### D.3 Hosted notebook recovery checklist

- capture runtime package versions at notebook start
- check accelerator and CUDA-sensitive packages
- restart cleanly and confirm the failure is reproducible
- compare against a local environment with explicit pins

#### D.4 Large-upgrade recovery checklist

- stop editing multiple layers at once
- identify the first broken boundary
- restore a known-good checkpoint if needed
- migrate in checkpoints rather than one giant pass

---

### Appendix E — Migration Checklists

#### E.1 Library/API migration checklist

- read the official migration guide first
- identify renamed or removed arguments
- identify changed defaults
- verify adjacent dependency expectations
- verify whether task docs or examples still lag behind the migration guide

#### E.2 Host-runtime migration checklist

- encode runtime assumptions explicitly
- stop relying on ambient preinstalled state
- capture Python, key packages, and hardware at runtime start
- treat host restarts and rebuilds as routine, not exceptional

#### E.3 Tutorial-to-current-path migration checklist

- treat old examples as clues, not authority
- replace the stale boundary first
- confirm the new primitive in current official docs
- only then widen back out to the full workflow

#### E.4 Inference-surface migration checklist

- identify whether you are using Hub API, Space API, routed provider inference, or a dedicated endpoint
- make provider choice explicit when useful
- document auth scope and token assumptions
- isolate inference-surface logic behind your own small wrapper

---

### Appendix F — Version-Bound Casebooks

#### F.1 Transformers v5 drift and migration cluster

##### Why this cluster matters

Transformers v5 is one of the clearest examples of this guide’s central thesis.

There is an official migration guide, and it matters. But real breakage still spills across task docs, tokenizer behavior, custom Trainer overrides, and adjacent dependency boundaries. A reader can still fail by following the most discoverable task recipe rather than the most current major-version guidance. This is exactly the kind of case where an official migration guide is necessary but not sufficient.

##### Typical symptoms

- `Trainer(..., tokenizer=...)` starts failing or warning because the migration path moved to `processing_class`
- `trainer.train(model_path=...)` no longer matches the current training entry path
- a custom `Trainer.compute_loss` override breaks because new kwargs are passed
- generation or config code still uses `model.config` or `torch_dtype` assumptions that the migration surface changed
- older task pages or copied snippets still point at v4-era patterns

##### Version-bounded changes to expect

- `Trainer(..., tokenizer=...)` → `processing_class`
- `trainer.train(model_path=...)` → `resume_from_checkpoint=...`
- `torch_dtype` deprecates in favor of `dtype`
- `load_in_4bit` / `load_in_8bit` give way to `quantization_config`
- generation-related behavior shifts further toward `generation_config`
- multiple pipeline and training defaults are simplified or removed

##### Likely drift layers

- library/API drift
- skipped-version migration
- tutorial drift
- dependency drift in adjacent libraries

##### Fastest verification

1. compare your code against the official v5 migration guide line by line
2. isolate the exact failing boundary: `Trainer`, tokenizer / processor handling, generation config, quantization config, or pipeline usage
3. if you overrode `Trainer.compute_loss`, check whether your signature accepts extra kwargs such as `num_items_in_batch`
4. if the code was copied from task docs or older examples, confirm that the example itself reflects the current v5 surface

##### Short-term recovery

- reduce to the smallest v5-relevant failing snippet
- if service restoration matters immediately, pin to the last known-good 4.x or early 5.x boundary while preserving the failing repro
- replace the most obviously removed argument first before changing unrelated logic
- add `**kwargs` to custom `compute_loss` overrides if the version boundary requires it

##### Durable migration path

- migrate to the official v5 primitive first, not to an issue-comment workaround
- replace `tokenizer=` with `processing_class=` in `Trainer`
- migrate deprecated loading and quantization calls to the current config-based model
- move generation assumptions from `model.config` to `model.generation_config` where relevant
- retest adjacent libraries as a set, especially if PEFT or other training-time integrations are involved

##### What not to trust

- a passing pin as proof that the migration is complete
- old task docs or examples as versionless truth
- a custom Trainer override that “used to work” as proof that its signature is still current

##### References

- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers v5 release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0)
- [Transformers issue: docs lag after v5 migration](https://github.com/huggingface/transformers/issues/43827)
- [Transformers issue: tokenizer refactor edge case](https://github.com/huggingface/transformers/issues/44361)
- [Transformers issue: `_is_hf_initialized` edge case](https://github.com/huggingface/transformers/issues/43632)
- [Transformers issue: `compute_loss` needs to tolerate extra kwargs](https://github.com/huggingface/transformers/issues/36331)

---

#### F.2 Datasets script-loader and tutorial-drift cluster

##### Why this cluster matters

The transition from Datasets 3.6.x to 4.x is not a routine version bump. It is a migration from “Hub-hosted builder scripts” toward “Hub-hosted data plus metadata.” That makes it one of the cleanest examples of tutorial drift turning into runtime breakage. It also shows that media-heavy paths can fail at a backend boundary even when the dataset itself looks superficially normal.

##### Typical symptoms

- `RuntimeError: Dataset scripts are no longer supported`
- code copied from older dataset docs no longer works
- audio or video handling behaves differently after upgrading because decoding expectations changed
- a task tutorial points at a dataset-loading path that no longer matches the current library behavior

##### Version-bounded changes to expect

- 3.6.x acts as a bridge line where older script-based workflows can still be converted
- 4.0.0 is the breaking line where builder scripts are removed from the loading path
- 4.x pushes toward data + metadata rather than arbitrary Python on load
- audio / video handling shifts toward more explicit media decode/backend expectations

##### Likely drift layers

- tutorial drift
- library/API drift
- skipped-version migration
- runtime dependency drift for media-heavy paths

##### Fastest verification

1. verify whether the copied example depends on a dataset script or remote builder path
2. check whether the current dataset needs a Parquet / Arrow / static-metadata path instead
3. if media is involved, verify the runtime dependencies expected by the current stack
4. treat any old loading-methods doc as historical unless its version matches your environment
5. if audio is involved, verify the current decode/backend expectation before treating the dataset as malformed

##### Short-term recovery

- if you must recover a historical path temporarily, use the smallest compatibility checkpoint that still loads the data cleanly
- replace only the loader boundary first before touching the rest of the workflow
- keep the example minimal until the current loading primitive succeeds

##### Durable migration path

- move the dataset path onto the current 4.x loading model
- treat 3.6.x as a bridge for conversion, not as a comfortable forever-version
- migrate dataset packaging toward data + metadata instead of Hub-hosted Python loader logic
- if audio or video are involved, validate media dependencies explicitly rather than assuming old behavior still holds

##### What not to trust

- historical loading-method pages without a version boundary
- older tutorial code that still “looks normal” but depends on script loading
- a successful one-off conversion run as proof the whole pipeline is migrated

##### References

- [Datasets releases](https://github.com/huggingface/datasets/releases)
- [Datasets issue: dataset scripts are no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Datasets issue: HC3 docs drift after script-loader removal](https://github.com/huggingface/datasets/issues/8012)
- [Datasets v2.19 loading methods docs (historical)](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods)

---

#### F.3 Gradio + Spaces runtime-contract cluster

##### Why this cluster matters

This cluster shows why library migration and host-runtime drift must be handled together. A Space can fail because Gradio changed, because Spaces changed, or because the interaction between them changed.

##### Typical symptoms

- launch behavior changes after a Gradio major bump
- a Space rebuilds cleanly but the app behavior changes
- local success does not reproduce Spaces behavior
- API visibility or footer/API behavior changes after migration
- chat or event-listener behavior changes after a Gradio 5→6 migration
- chained events run after a failure because `.then()` was used where `.success()` was intended
- ZeroGPU CUDA behavior changes after the backing hardware and supported PyTorch versions move

##### Version-bounded changes to expect

- the safest migration path is `5.50` first, fix deprecation warnings, then move to `6.x`
- some app-level parameters move from `Blocks(...)` to `launch(...)`
- `show_api` moves toward `footer_links`
- event-listener API exposure moves toward `api_visibility`
- Spaces interaction can surface these changes more sharply than local-only testing
- `.then()` continues the chain even after the previous event errors; `.success()` is the safer choice when a later step should only run after a successful previous step
- ZeroGPU now documents NVIDIA RTX Pro 6000 Blackwell-backed `large` and `xlarge` sizes, so old H200-era assumptions should be rechecked

##### Likely drift layers

- library/API drift
- runtime contract drift
- platform drift
- dependency drift

##### Fastest verification

1. verify the Space YAML and runtime metadata first
2. confirm the exact Gradio major version and whether the app already passed through a `5.50` deprecation-cleanup step
3. compare the Space YAML `sdk_version` against any `gradio` pin in `requirements.txt`
4. inspect whether old `Blocks(...)` constructor parameters now belong in `launch(...)`
5. inspect whether API visibility assumptions still match the current footer / endpoint settings
6. compare local and Space behavior under the same Python and Gradio versions
7. if custom components are present, test them under clean rebuild rather than assuming browser-path success proves compatibility
8. inspect chained event listeners: use `.success()` for success-only continuation, `.failure()` for failure-only handling, and `.then()` only when continuation after failure is acceptable
9. if ZeroGPU is involved, compare the current documented PyTorch support, GPU size, and runtime behavior against the app's pinned wheel assumptions

##### Short-term recovery

- pin `sdk_version` and `python_version`
- keep one authoritative Gradio version declaration on Gradio Spaces instead of letting YAML and `requirements.txt` drift apart
- if the app jumped directly to 6.x, step back to 5.50, remove deprecations, then retry the 6.x migration
- move obviously relocated parameters to `launch(...)`
- if custom components break under rebuild, reduce to the smallest non-custom-component path before deeper migration
- keep the API surface minimal until the current Gradio launch path works again
- replace critical `.then()` chains with `.success()` when downstream work must not run after an upstream failure
- move CUDA compatibility checks before model-level refactors when a ZeroGPU hardware refresh is suspected

##### Durable migration path

- encode runtime assumptions explicitly in the repo and README YAML
- make the app reproducible from a clean rebuild rather than a warm container
- treat Spaces-specific runtime behavior as part of the migration, not as noise around it
- migrate API visibility and chat/message handling deliberately rather than letting defaults drift under you
- keep Gradio event-chain semantics explicit in code review, especially around error handling and cleanup
- keep ZeroGPU hardware / PyTorch support notes close to the Space runtime documentation

##### What not to trust

- “The repo is the environment.”
- “Local Gradio success proves Spaces is fine.”
- “Directly jumping from early 5.x to 6.x is equivalent to going through the deprecation step.”
- “`.then()` means success-only continuation.”
- “A ZeroGPU runtime refresh is unrelated to Gradio migration because Gradio code did not change.”

##### References

- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [Gradio 5 migration issue](https://github.com/gradio-app/gradio/issues/9463)

---

#### F.4 Hub v1.0 and inference-surface drift cluster

##### Why this cluster matters

This cluster captures a recurring failure pattern: a client/library migration and a platform-surface migration happen close enough together that users misdiagnose one as the other.

##### Typical symptoms

- auth or request behavior changes after a time gap
- code still assumes the legacy `api-inference.huggingface.co` endpoint
- a wrapper hides whether it is calling Hub API, Space API, routed provider inference, or the old serverless model
- legacy nested `parameters={...}` assumptions survive inside current client code

##### Version-bounded changes to expect

- `huggingface_hub` v1.0 raises the Python floor to 3.9+ and moves to `httpx`
- the legacy `api-inference.huggingface.co` endpoint is deprecated in favor of the router / provider model
- HF Inference is the documented successor of the old “Inference API (serverless)” naming
- `InferenceClient` prefers task helpers and flat keyword arguments rather than older ad-hoc JSON contracts

##### Likely drift layers

- library/API drift
- HF product/platform drift
- dependency drift
- skipped-version migration in wrappers and internal abstractions

##### Fastest verification

1. confirm whether the code still targets `api-inference.huggingface.co` directly
2. identify the exact surface type: Hub API, Space API, routed provider inference, or dedicated endpoint
3. verify Python floor and dependency assumptions if `huggingface_hub` changed under you
4. reduce to the smallest current `InferenceClient` or provider-aware example

##### Short-term recovery

- stop calling the dead legacy endpoint
- switch the minimal failing path onto a current provider-aware or `InferenceClient` example
- make the surface type explicit in logs and wrappers before changing unrelated payload logic
- if auth is involved, verify the token scope against the actual surface you are using

##### Durable migration path

- adopt the provider-aware, task-aware model explicitly
- keep Hub-client migration and inference-surface migration as separate concerns in your code
- move old nested `parameters` payload assumptions to current helper arguments or router-compatible calls
- document surface type, provider choice, and auth assumptions so they stop living only in memory

##### What not to trust

- “Inference API” as a timeless singular surface
- wrappers that hide which HF surface they actually target
- auth fixes copied from a different product surface

##### References

- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [InferenceClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [HF Inference provider page](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub API Endpoints](https://huggingface.co/docs/hub/api)

---

#### F.5 Training-stack adjacency drift cluster

##### Why this cluster matters

Some fine-tuning failures look like generic trainer instability, but the real boundary sits in adjacent training surfaces: adapter injection rules and supervised token-span rules. This cluster exists to stop readers from diagnosing architecture-bound PEFT issues or labeling-bound TRL issues as vague generic training failures.

##### Typical symptoms

- a LoRA recipe copied from one model family does not attach cleanly to another
- PEFT behaves differently across architectures because `target_modules` is not portable
- chat SFT runs, but the model learns from the wrong token spans
- `assistant_only_loss` or template assumptions do not match the dataset structure

##### Version-bounded changes to expect

- current PEFT recipes remain architecture-sensitive even when the high-level LoRA recipe looks similar
- current TRL chat fine-tuning surfaces require explicit checking of supervised token spans rather than assuming the loss target is implicit

##### Likely drift layers

- adjacent-library drift
- training-stack drift
- tutorial drift
- defaults / labeling drift

##### Fastest verification

1. inspect whether `target_modules` was copied from a different architecture
2. verify which tokens actually receive loss during chat fine-tuning
3. confirm whether the chat template and dataset structure align
4. reduce to the smallest still-failing adapter or SFT configuration

##### Short-term recovery

- keep the failing setup minimal
- fix adapter targeting before touching unrelated hyperparameters
- fix supervised token-span / template alignment before retuning training settings

##### Durable migration path

- treat `target_modules` as architecture-bound, not recipe-portable
- treat `assistant_only_loss` and supervised token-span control as explicit training-surface configuration
- migrate copied training recipes onto the current model family and current chat-format assumptions

##### What not to trust

- a LoRA recipe copied from another model family as though it were portable
- "the trainer ran" as proof that the supervised token span is correct
- hyperparameter retuning before verifying adapter targeting or label span boundaries

##### References

- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [PEFT issue: target_modules portability / architecture mismatch](https://github.com/huggingface/peft/issues/2155)
- [TRL SFT Trainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [TRL issue: assistant_only_loss / token-span behavior](https://github.com/huggingface/trl/issues/3827)
- [Forum: CompletionOnlyLM / multi-turn chat confusion](https://discuss.huggingface.co/t/best-practice-for-usage-of-data-collator-for-completiononlylm-in-multi-turn-chat/99263)

---

#### F.6 Hub upload and rate-limit drift cluster

##### Why this cluster matters

Hub uploads and downloads are operational workflows, not just file operations. A workflow can fail because request volume, authentication, client retry behavior, repository shape, or upload method stopped matching the current Hub contract.

##### Typical symptoms

- intermittent `429 Too Many Requests`
- `503` or connection errors during long uploads
- a large folder upload fails late and restarts too much work
- unauthenticated scripts hit stricter limits than expected
- many small commits or metadata calls become the real bottleneck

##### Version-bounded changes to expect

- current Hub docs expose `RateLimit` headers for rate-limit diagnosis
- current `huggingface_hub` includes smarter retry handling for rate-limit errors in recent versions
- `upload_large_folder()` is explicitly designed for large, resumable, retry-oriented upload jobs

##### Likely drift layers

- Hub API drift
- operational rate-limit drift
- client behavior drift
- repository-structure drift

##### Fastest verification

1. pass `HF_TOKEN` explicitly
2. record whether the failure is API, resolver, or upload/commit traffic
3. record status code and rate-limit headers
4. test the smallest upload that still reproduces the failure
5. switch large-folder jobs to the large-folder upload path before writing custom retry loops

##### Short-term recovery

- authenticate
- slow down bursty workflows
- avoid unnecessary metadata calls
- use the current Hub client retry behavior
- use resumable upload tools for large folders

##### Durable migration path

- design large upload jobs as resumable operations
- log transient and permanent failures differently
- avoid treating every `429` or `503` as corruption
- structure repositories so large mutable artifacts do not require fragile repeated commits

##### What not to trust

- “The network failed once, so the whole upload must be restarted.”
- “A fixed sleep is enough rate-limit handling.”
- “Anonymous and token-authenticated Hub access behave the same.”

##### References

- [Hub Rate limits](https://huggingface.co/docs/hub/rate-limits)
- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [HfApi Client reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)

### Appendix G — Fastest Verification Snippets

The goal of these snippets is not to solve everything. The goal is to falsify the wrong hypothesis quickly.

#### G.1 Minimal environment snapshot

```bash
python -V
python -c "import sys, platform; print(sys.executable); print(platform.platform())"
pip freeze | sort > current_requirements_freeze.txt
```

Use this first when the failure appeared after time passed, a host changed, or a rebuild happened.

#### G.2 Torch / CUDA / hardware snapshot

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    print('device 0:', torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print('device 0 capability:', f'sm_{major}{minor}')
PY

nvidia-smi || true
```

On hosted GPU runtimes, record both the human-readable device name and the compute capability. A failure that appears only after the device changes can be a wheel / architecture mismatch even when the application code did not change. For Blackwell-class `sm_120` devices, old CUDA wheels that only contain older kernels can fail before model logic is relevant.

Use this when imports, model loading, or runtime behavior differ across hosts.

#### G.3 HF package boundary snapshot

```bash
python - <<'PY'
packages = [
    'huggingface_hub',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'trl',
    'gradio',
]
for name in packages:
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'unknown'))
    except Exception as e:
        print(name, 'IMPORT-FAILED', repr(e))
PY
```

Use this before widening the investigation into application logic.

#### G.4 Notebook bootstrap cell

```python
import os, sys, platform

print('python', sys.version)
print('executable', sys.executable)
print('platform', platform.platform())
print('cwd', os.getcwd())

try:
    import torch
    print('torch', torch.__version__)
    print('cuda available', torch.cuda.is_available())
    print('cuda version', torch.version.cuda)
except Exception as e:
    print('torch IMPORT-FAILED', repr(e))

for pkg in ['huggingface_hub', 'transformers', 'datasets', 'gradio']:
    try:
        mod = __import__(pkg)
        print(pkg, getattr(mod, '__version__', 'unknown'))
    except Exception as e:
        print(pkg, 'IMPORT-FAILED', repr(e))
```

Put this near the top of a notebook whenever you are debugging host drift.

#### G.5 Two-host comparison checklist

Capture this from both hosts and compare side by side:

- Python version
- `pip freeze` subset for critical packages
- Torch / CUDA state
- visible hardware
- startup command path
- whether files are assumed to persist across restarts
- the exact HF surface being used for inference

#### G.6 Space rebuild verification checklist

Before blaming application code, verify:

- README YAML values
- pinned `python_version`
- pinned `sdk_version`
- whether the app depends on files created outside the repo
- whether a prior fix lived only in Dev Mode
- whether the same repo launches cleanly from a fresh rebuild

#### G.7 Tutorial-drift verification checklist

Before copying more code from the same source, verify:

- the date and version context of the tutorial
- the current official docs for the exact primitive
- the official migration guide for the major version boundary
- whether the symptom already appears in a selected issue or release thread

---

#### G.8 Hub rate-limit / upload snapshot

```bash
python - <<'PY'
import huggingface_hub
print('huggingface_hub:', huggingface_hub.__version__)
PY

# In the failing script or wrapper, log:
# - whether HF_TOKEN is set
# - status code
# - RateLimit / RateLimit-Policy headers when present
# - whether the traffic is API, resolver, or upload/commit traffic
```

Use this when Hub operations fail intermittently or only at scale. The goal is to separate rate limiting, transient service errors, permanent permission failures, and poor upload strategy.

### Appendix H — Chapter Seed References

#### H.1 Hub / client migration

- [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration)
- [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/guides/inference)
- [InferenceClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)

#### H.2 Transformers migration

- [Transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers issue: docs lag after v5 migration](https://github.com/huggingface/transformers/issues/43827)
- [Transformers issue: tokenizer refactor edge case](https://github.com/huggingface/transformers/issues/44361)
- [Transformers issue: `_is_hf_initialized` edge case](https://github.com/huggingface/transformers/issues/43632)

#### H.3 Datasets drift

- [Datasets releases](https://github.com/huggingface/datasets/releases)
- [Datasets issue: dataset scripts are no longer supported](https://github.com/huggingface/datasets/issues/7693)
- [Datasets issue: HC3 docs drift after script-loader removal](https://github.com/huggingface/datasets/issues/8012)
- [Datasets v2.19 loading methods docs (historical)](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/loading_methods)

#### H.4 Spaces / Gradio runtime contract

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Disk usage on Spaces](https://huggingface.co/docs/hub/spaces-storage)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell update PR](https://github.com/huggingface/hub-docs/pull/2474)
- [ZeroGPU Blackwell forum note](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960/6)
- [Gradio 5 migration issue](https://github.com/gradio-app/gradio/issues/9463)
- [Gradio Blocks and event listeners](https://www.gradio.app/guides/blocks-and-event-listeners)

#### H.5 Hosted notebook drift

- [Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)
- [Kaggle docker-python repository](https://github.com/Kaggle/docker-python)
- [Kaggle docker-python releases](https://github.com/Kaggle/docker-python/releases)

#### H.6 Inference surface drift

- [Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [HF Inference provider page](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub API Endpoints](https://huggingface.co/docs/hub/api)

#### H.7 Training-stack adjacency drift

- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [PEFT issue: target_modules portability / architecture mismatch](https://github.com/huggingface/peft/issues/2155)
- [TRL SFT Trainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [TRL issue: assistant_only_loss / token-span behavior](https://github.com/huggingface/trl/issues/3827)
- [Forum: CompletionOnlyLM / multi-turn chat confusion](https://discuss.huggingface.co/t/best-practice-for-usage-of-data-collator-for-completiononlylm-in-multi-turn-chat/99263)

---
