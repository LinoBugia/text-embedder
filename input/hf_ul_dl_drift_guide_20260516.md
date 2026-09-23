# HF Hub Upload, Download, and Xet Transfer Drift Runbook

---

## 0. Reader Contract

This runbook is for maintainers who see Hugging Face Hub transfers become slow, unstable, confusing, or different from older instructions. It covers uploads, downloads, command-line transfers, `huggingface_hub` Python transfers, Xet-backed transfers, HTTP fallback, cache behavior, authentication state, rate limits, and dataset format choices such as auto-converted Parquet.

This runbook is intentionally narrow. It is not a complete Hugging Face Hub guide, not a Git tutorial, not a full dataset engineering guide, not an account-plan guide, and not a replacement for official documentation. It is a recovery-oriented checklist for transfer drift.

The central idea is simple: do not debug every slow Hub transfer as a generic internet problem first. The Hub transfer stack has moved from older `hf_transfer`-era advice toward Xet-backed transfer paths. During that transition, several things can drift at the same time: client version, `hf_xet` version, Xet enablement, HTTP fallback, cache location, file size, file count, operating system, authentication state, account plan, and dataset access path.

The priorities are:

1. get the transfer to complete;
2. record enough context to explain why it completed;
3. separate transfer-path problems from auth, rate-limit, cache, filesystem, and dataset-layout problems;
4. tune for speed only after the stable path is repeatable.

The fastest path is only useful after it is repeatable and attributable.

## 1. Fast Triage

If a transfer is slow, stalled, or behaving differently from an older guide, do this before changing many variables:

```bash
hf --version
hf env
hf auth whoami
python - <<'PY'
import os, platform
print('python', platform.python_version())
print('platform', platform.platform())
print('HF_TOKEN present:', bool(os.environ.get('HF_TOKEN')))
import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__)
try:
    import hf_xet
    print('hf_xet', getattr(hf_xet, '__version__', 'unknown'))
except Exception as e:
    print('hf_xet unavailable:', repr(e))
PY
```

Set environment variables before starting the command, worker, or notebook process whenever possible. In Python, set them before importing `huggingface_hub`, `datasets`, `transformers`, or another library that may import them. Then test transfer paths separately:

```bash
# Default path.
hf download REPO_ID FILENAME

# Xet high-performance path.
HF_XET_HIGH_PERFORMANCE=1 hf download REPO_ID FILENAME

# Xet-disabled path, where HTTP fallback is possible.
HF_HUB_DISABLE_XET=1 hf download REPO_ID FILENAME
```

Use equivalent `hf upload` commands for upload failures. For cache checks on current `hf` CLI versions, prefer `hf cache ls`, `hf cache prune --dry-run`, and `hf cache verify`; older cache-scan references may come from pre-v1 instructions.

Do not rely on this older flag as proof that a high-speed transfer mode was tested:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1
```

In the v1-era Hub client, `hf_transfer` support was removed and this variable is ignored. Use `HF_XET_HIGH_PERFORMANCE` and `HF_HUB_DISABLE_XET` as the main Xet-era diagnostic switches.

## 2. Choose the Failure Class

When a Hub transfer is slow or stuck, classify the problem before changing many variables.

| Failure class | Typical symptom | First useful test |
|---|---|---|
| Old transfer advice | Older guides recommend `HF_HUB_ENABLE_HF_TRANSFER=1`; nothing changes. | Check `huggingface_hub` version and replace old `hf_transfer` assumptions with Xet-era tests. |
| Xet upload drift | Upload starts, bursts, then stalls or times out. | Test default, `HF_XET_HIGH_PERFORMANCE=1`, and `HF_HUB_DISABLE_XET=1` separately. |
| Xet download drift | Download reaches some point, then appears frozen, extremely slow, or silent. | Test the same file with and without Xet; record cache path, disk type, route, and client versions. |
| Misleading burst speed | Progress begins at hundreds of MB/s, then stops after a small amount. | Treat the burst as a local-buffer or path artifact until sustained network and disk activity are confirmed. |
| HTTP fallback limit | Xet is disabled, but a very large file cannot download at all. | Treat regular HTTP fallback as a diagnostic path, not a guaranteed large-file path; retry with current `hf_xet`. |
| Dataset format drift | Raw dataset download is slow, but a Parquet path is available. | Try auto-converted Parquet, split-level access, streaming, DuckDB, Polars, or column selection. |
| Cache/local state drift | Repeated attempts behave inconsistently or stall at similar points. | Check `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, free disk, cache health, and filesystem type. |
| Account/auth drift | 401, 403, 429, rate-limit messages, anonymous-request warnings, or inconsistent CLI/API login state. | Check CLI auth, Python API auth, token propagation, account plan, shared IPs, and worker count. |
| Large folder drift | Many small files upload slowly or trigger 429/errors. | Reduce request count, use folder-oriented upload, group files, or package shards. |
| Runtime wrapper drift | `datasets`, `transformers`, notebooks, or worker processes call the Hub differently than the CLI test. | Test CLI and Python paths separately and record which process receives the token and env vars. |

Do not compare two tests unless you also record the client version, `hf_xet` version, OS, Python version, authentication state, account plan, file size, file count, cache path, and environment variables.

## 3. Mental Model: Hub Transfer Is Now a Stack

A Hub transfer is not one pipe. A slow or stuck transfer may involve several layers:

```text
user command or library call
  -> huggingface_hub / datasets / transformers / CLI / git
  -> auth and account tier
  -> Xet path or HTTP fallback
  -> CAS / range requests / chunking / staging
  -> local cache and filesystem
  -> network route, proxy, VPN, cloud provider, or campus network
  -> file layout and dataset format
```

A useful runbook does not ask, "Is the internet slow?" first. It asks, "Which layer changed?"

Common examples:

```text
Old flag changed:
  HF_HUB_ENABLE_HF_TRANSFER no longer tests the intended path.

Transfer path changed:
  The same file behaves differently with and without Xet.

Account context changed:
  Anonymous, Free, PRO, Team, and Enterprise traffic may not see the same limits.

Data layout changed:
  Raw JSONL or thousands of files are slow; Parquet or split-level reads are fast.

Local state changed:
  A warm cache, stale Xet cache, NFS mount, or nearly full disk changes behavior.
```

## 4. Symptom-First Recovery Recipes

### Problem: old tutorials say to enable `HF_HUB_ENABLE_HF_TRANSFER`

Older instructions often say:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download ...
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload ...
```

In `huggingface_hub` v1.x, this is stale advice. The official migration guide says all repositories are Xet-enabled, `hf_xet` is the default way to download and upload files, support for the `hf_transfer` optional package was removed, and `HF_HUB_ENABLE_HF_TRANSFER` is ignored. The current environment-variable docs also describe `HF_HUB_ENABLE_HF_TRANSFER` as deprecated and point users toward `HF_XET_HIGH_PERFORMANCE` for high-performance Xet behavior. Use Xet controls instead.

Use this version check first:

```bash
hf --version
python - <<'PY'
import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__)
try:
    import hf_xet
    print('hf_xet', getattr(hf_xet, '__version__', 'unknown'))
except Exception as e:
    print('hf_xet import failed:', repr(e))
PY
```

Then test current transfer modes explicitly:

```bash
# Default behavior.
hf download REPO_ID --repo-type model

# Xet high-performance path.
HF_XET_HIGH_PERFORMANCE=1 hf download REPO_ID --repo-type model

# Xet disabled / HTTP fallback.
HF_HUB_DISABLE_XET=1 hf download REPO_ID --repo-type model
```

For uploads:

```bash
# Default behavior.
hf upload REPO_ID LOCAL_PATH PATH_IN_REPO

# Xet high-performance path.
HF_XET_HIGH_PERFORMANCE=1 hf upload REPO_ID LOCAL_PATH PATH_IN_REPO

# Xet disabled / HTTP fallback.
HF_HUB_DISABLE_XET=1 hf upload REPO_ID LOCAL_PATH PATH_IN_REPO
```

If a legacy flag is left in place, you may think you have already tested a fast transfer mode when you have not.

Use this interpretation table for the first pass:

| Test result | Safe interpretation | Do not overclaim |
|---|---|---|
| Default fails, `HF_HUB_DISABLE_XET=1` completes | The Xet path, Xet cache, client version, route, or filesystem is suspicious. | Do not say Xet is globally broken. |
| Default fails, `HF_XET_HIGH_PERFORMANCE=1` completes | Default/adaptive behavior may be unsuitable for that machine or route. | Do not make high-performance mode the default everywhere. |
| `HF_HUB_DISABLE_XET=1` fails for a very large file | HTTP fallback may not support that file or route. | Do not treat Xet disablement as a universal final fix. |
| All paths fail the same way | Auth, rate limit, local disk, proxy/VPN, file layout, or backend behavior may be involved. | Do not keep toggling transfer flags without checking the rest of the stack. |


### Problem: upload starts quickly, then stalls or slows to almost zero

This is a common large-file drift symptom. It is suspicious when it appears with large `.safetensors`, GGUF files, model shards, large dataset shards, or first-time uploads.

First capture a baseline:

```bash
hf --version
hf auth whoami
hf env
```

Then test three paths separately:

```bash
# Baseline: current default.
hf upload REPO_ID LOCAL_PATH PATH_IN_REPO

# Xet high-performance mode.
HF_XET_HIGH_PERFORMANCE=1 hf upload REPO_ID LOCAL_PATH PATH_IN_REPO

# Xet disabled / HTTP fallback.
HF_HUB_DISABLE_XET=1 hf upload REPO_ID LOCAL_PATH PATH_IN_REPO
```

If the fallback is slower but completes, use it to recover the upload first. Treat speed tuning as a second phase.

If the Xet path bursts and then fails, look for Xet/CAS timeout signatures such as `upload_xorb`, `IncompleteBody`, timeout near a fixed duration, or repeated retry behavior. In Xet issue threads, large upload behavior has been tied to timeout and request behavior, not just raw bandwidth.

### Problem: download bursts fast, then appears frozen

A very fast initial burst does not prove the transfer path is healthy. Some Xet-backed download reports show high apparent early throughput followed by a complete stall after a small amount of transferred data.

Use paired tests:

```bash
# Xet default or enabled path.
hf download REPO_ID FILENAME --repo-type model

# Disable Xet to isolate the path.
HF_HUB_DISABLE_XET=1 hf download REPO_ID FILENAME --repo-type model
```

If `HF_HUB_DISABLE_XET=1` is stable but slower, the problem is likely not a generic internet failure. It may be Xet path, cache, filesystem, OS, client version, or route-specific behavior.

Do not stop at one observation. Record:

```text
repo_id
filename
file size
huggingface_hub version
hf_xet version
Python version
OS and filesystem
cache location
free disk space
authenticated or anonymous
account plan
environment variables
whether browser/wget/API behavior differs
```

### Problem: progress reaches 99-100%, then slows dramatically

Late-stage slowness can be confusing because most bytes appear complete. Do not assume the file is healthy until the command exits successfully and the local file is usable.

Check:

```text
- is network still active?
- is disk still writing?
- is CPU busy hashing or reconstructing?
- is cache or staging growing?
- does the same point repeat after restart?
- does disabling Xet change the final-stage behavior?
```

If the file is a shard in a multi-file model, verify all shards completed. One missing shard can later look like a model-loading problem rather than a transfer problem.

### Problem: disabling Xet fixes one file but breaks another

`HF_HUB_DISABLE_XET=1` is an isolation test, not a universal final solution. For very large files, the regular HTTP path may be unavailable, inefficient, or route-limited, and `hf_xet` may be required. Treat exact size thresholds as issue-specific unless confirmed in current official documentation or by a current maintainer response.

The safe interpretation is:

```text
HF_HUB_DISABLE_XET=1 works:
  Xet path is suspicious for this file, route, cache, or client version.

HF_HUB_DISABLE_XET=1 fails on a very large file:
  the HTTP fallback path may not support that file size; test with current hf_xet.
```

Do not write a runbook or script that permanently disables Xet for every case. Use it to isolate a failure, then decide whether the stable path or the Xet path is appropriate for that file.

### Problem: dataset download is slow but not obviously stuck

Dataset transfer speed can be dominated by file format and access path. A raw repository download of CSV, JSON, JSONL, or many small files can be much slower than reading auto-converted Parquet shards or only the split and columns you need.

Try this diagnostic path:

```bash
# Inspect repository files through the Python API.
python - <<'PY'
from huggingface_hub import HfApi
for path in HfApi().list_repo_files('DATASET_REPO', repo_type='dataset'):
    print(path)
PY
```

Then check whether auto-converted Parquet exists. Hugging Face Dataset Viewer exposes Parquet conversion endpoints for supported datasets. In some workflows, the `@~parquet` revision can access the converted Parquet branch (`refs/convert/parquet`). Remember that conversion may be unavailable or partial; for large non-Parquet datasets, the generated Parquet representation may cover only part of the dataset.

```bash
# List auto-converted Parquet files if available.
hf datasets parquet DATASET_REPO
```

Practical alternatives:

```text
- Use datasets streaming when you do not need the whole dataset locally.
- Use auto-converted Parquet when available.
- Read only the needed split.
- Read only the needed columns.
- Use DuckDB or Polars against Parquet when appropriate.
- Avoid cloning or downloading the entire raw repository if the task is analytical.
```

Key point: not every slow dataset download is a network problem. It can be a dataset layout problem.

### Problem: `load_dataset()` hangs but direct Hub paths differ

This is a cross-layer failure. `datasets.load_dataset()` can involve Hub metadata, file resolution, cache, Arrow/Parquet materialization, streaming choices, and Xet-backed file transfer.

Test separately:

```bash
python - <<'PY'
import datasets, huggingface_hub
print('datasets', datasets.__version__)
print('huggingface_hub', huggingface_hub.__version__)
PY
```

```bash
# Direct Hub test.
hf download DATASET_REPO --repo-type dataset

# Xet disabled direct test.
HF_HUB_DISABLE_XET=1 hf download DATASET_REPO --repo-type dataset
```

Then test the dataset library path with the same environment variables:

```bash
HF_HUB_DISABLE_XET=1 python - <<'PY'
from datasets import load_dataset
# Replace with the minimal split/config that reproduces the issue.
ds = load_dataset('DATASET_REPO', split='train', streaming=True)
print(ds)
PY
```

If direct Hub download works but `load_dataset()` hangs, record both results. The issue may sit at the intersection of transfer, dataset script, format conversion, cache, and streaming/materialization.

### Problem: 429 or rate-limit behavior appears

Rate-limit problems are different from Xet stalls, but they often appear in the same workflows.

First check authentication:

```bash
hf auth whoami
python - <<'PY'
from huggingface_hub import HfApi
try:
    print(HfApi().whoami())
except Exception as e:
    print(type(e).__name__, e)
PY
```

Then check whether your token is actually present in the process that performs the transfer:

```bash
python - <<'PY'
import os
print('HF_TOKEN present:', bool(os.environ.get('HF_TOKEN')))
PY
```

If many workers download the same model or dataset, do not let every worker hit the Hub independently. Pre-download once to a shared cache or local directory, then use local files:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='org/repo',
    local_dir='/srv/hf-cache/org-repo',
)
```

Then load locally:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '/srv/hf-cache/org-repo',
    local_files_only=True,
)
```

For repeated jobs, prefer controlled pre-download, shared cache, and `local_files_only=True` over many independent Hub requests.

### Problem: CLI auth and Python auth disagree

Do not assume that the CLI and Python API see the same token. In notebooks, Colab, Kaggle, CI, containers, and servers, the token source can differ between shell commands and Python processes.

Run both:

```bash
hf auth whoami
```

```bash
python - <<'PY'
from huggingface_hub import HfApi
try:
    print(HfApi().whoami())
except Exception as e:
    print(type(e).__name__, e)
PY
```

If they disagree, record:

```text
HF_TOKEN
HF_TOKEN_PATH
HF_STORED_TOKENS_PATH
HF_HOME
HF_HUB_CACHE
shell type
notebook/runtime type
whether the command runs in a subprocess
whether secrets are injected only into Python
```

### Problem: many small files upload slowly or hit 429

A huge number of files can stress the Hub through request count, metadata operations, commit design, and local filesystem overhead. This is not the same problem as one 40 GB file stalling.

Try:

```text
- reduce file count if possible;
- use shards rather than thousands of tiny files;
- group tiny artifacts into archives only when that does not damage downstream usability;
- use folder-oriented upload rather than repeated single-file upload;
- avoid every worker uploading independently;
- use fewer commits for large initial imports;
- record whether the issue is API rate limit, upload throughput, or commit overhead.
```

Do not blindly split one large file into thousands of small files. That can trade a transfer-size problem for a request-count problem.

## 5. What Changed: From hf_transfer-Era Advice to Xet-Era Transfer

Older Hub performance advice often revolved around `hf_transfer`, especially for faster downloads or uploads. That advice became stale as Hub storage and transfer moved toward Xet-backed paths.

The important boundary is `huggingface_hub` v1.0. The v1 migration documentation says that all Hub repositories are Xet-enabled, `hf_xet` is the default way to download and upload files, the optional `hf_transfer` package was removed, and `HF_HUB_ENABLE_HF_TRANSFER` is ignored. The practical replacement for fast Xet operation is `HF_XET_HIGH_PERFORMANCE`.

This is why old snippets can mislead users:

```bash
# Old mental model. In v1-era clients, do not assume this does anything useful.
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload ...
```

Use current controls instead:

```bash
# Try the high-performance Xet path.
HF_XET_HIGH_PERFORMANCE=1 hf upload ...
HF_XET_HIGH_PERFORMANCE=1 hf download ...

# Isolate the Xet path by falling back to regular HTTP where possible.
HF_HUB_DISABLE_XET=1 hf upload ...
HF_HUB_DISABLE_XET=1 hf download ...
```

## 6. Why Xet Drift Can Feel Strange

Xet-backed transfers can feel unintuitive because the system is not just streaming one file from one URL. It may involve chunking, range requests, deduplication, local staging, cache reuse, reconstruction, and adaptive transfer behavior.

That means several observations can be true at the same time:

```text
- the first seconds look fast;
- the progress bar later appears stuck;
- the cache grows but the final file is not complete;
- disabling Xet makes one transfer stable but makes another very large file unavailable;
- upgrading hf_xet fixes one environment but not another route;
- a browser download behaves differently from huggingface_hub;
- a dataset Parquet path is faster than raw repository download.
```

This is why the runbook focuses on paired tests and recorded context rather than one magic flag.

## 7. Required Diagnostic Capture

Before opening an issue, changing account plan, reinstalling clients, or rewriting the repository layout, capture a minimal diagnostic block.

```bash
hf --version
hf env
hf auth whoami
```

```bash
python - <<'PY'
import os, platform
print('python', platform.python_version())
print('platform', platform.platform())
print('HF_TOKEN present', bool(os.environ.get('HF_TOKEN')))

import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__)
try:
    import hf_xet
    print('hf_xet', getattr(hf_xet, '__version__', 'unknown'))
except Exception as e:
    print('hf_xet unavailable:', repr(e))

for key in [
    'HF_HOME',
    'HF_HUB_CACHE',
    'HF_XET_CACHE',
    'HF_HUB_DISABLE_XET',
    'HF_XET_HIGH_PERFORMANCE',
    'HF_XET_FIXED_UPLOAD_CONCURRENCY',
    'HF_XET_FIXED_DOWNLOAD_CONCURRENCY',
    'HF_XET_NUM_CONCURRENT_RANGE_GETS',
    'HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY',
    'HF_XET_CHUNK_CACHE_SIZE_BYTES',
    'HF_XET_SHARD_CACHE_SIZE_LIMIT',
    'HF_HUB_DOWNLOAD_TIMEOUT',
    'HF_HUB_ETAG_TIMEOUT',
]:
    print(key, os.environ.get(key))
PY
```

Also record:

```text
operation: upload / download / dataset load / snapshot_download / hf CLI / Git
repo type: model / dataset / space
repo visibility: public / private / gated
file size: largest file and total size
file count: approximate number of files
file layout: one huge file / shards / many small files / dataset raw files / Parquet
OS: Windows / Linux / macOS / WSL / Colab / Kaggle / CI / server
filesystem: local SSD / HDD / NFS / network drive / container volume
network: home / cloud / campus / corporate proxy / VPN / GCP / AWS / Azure
account state: anonymous / Free / PRO / Team / Enterprise
worker count: one process / many workers / distributed job
cache state: new cache / warm cache / moved cache / nearly full disk
```

## 8. Environment Variable Checklist

| Variable | Use | Caution |
|---|---|---|
| `HF_HUB_DISABLE_XET=1` | Disable `hf-xet` when available, to isolate Xet path problems. | Diagnostic switch, not a universal final fix; very large files may need Xet. |
| `HF_XET_HIGH_PERFORMANCE=1` | Ask `hf-xet` to use more aggressive network, CPU, and disk settings. | Best for high-bandwidth machines with enough memory; test stability before making it a default. |
| `HF_XET_FIXED_UPLOAD_CONCURRENCY` | Pin upload concurrency for experiments that bypass adaptive concurrency. | Advanced Xet tuning; use small controlled values first. |
| `HF_XET_FIXED_DOWNLOAD_CONCURRENCY` | Pin download concurrency for experiments that bypass adaptive concurrency. | Advanced Xet tuning; higher is not always better. |
| `HF_XET_NUM_CONCURRENT_RANGE_GETS` | Tune concurrent range GETs per file for Xet downloads. | Can help only when bandwidth, disk, and route can absorb more parallelism. |
| `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY` | Make reconstruction writes sequential, useful to test HDD or slow-disk behavior. | Mainly relevant when disk seek behavior is the bottleneck. |
| `HF_XET_CHUNK_CACHE_SIZE_BYTES` | Enable or size the local chunk cache used on the download path. | Current docs say the chunk cache is disabled by default; enabling it can help reuse but may not speed one-off transfers. |
| `HF_XET_SHARD_CACHE_SIZE_LIMIT` | Size the shard cache used on the upload path. | Useful for repeated or incremental uploads; not a magic fix for first-time failures. |
| `HF_XET_CACHE` | Move Xet cache to a suitable disk. | Prefer fast local SSD/NVMe; avoid unstable network mounts for heavy transfers. |
| `HF_HOME` | Move the whole HF home/cache root. | Changes token/cache locations; document it before comparing tests. |
| `HF_HUB_CACHE` | Move standard Hub cache. | Do not confuse Hub cache and Xet cache. |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Adjust HTTP download timeout. | Does not solve every Xet/CAS timeout. |
| `HF_HUB_ETAG_TIMEOUT` | Adjust metadata timeout. | Metadata timeout is different from large file transfer. |
| `HF_TOKEN` | Pass token to the actual process. | Do not print the token value in logs. |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Avoid implicit token use in some tests. | Can change auth behavior; document it. |
| `HF_DEBUG=1` | Emit more debug information and cURL-like request logs. | Logs can include sensitive URLs or headers; sanitize before sharing. |

Set environment variables before starting the command or worker process. In Python scripts and notebooks, set them before importing `huggingface_hub`, `datasets`, `transformers`, or any library that may import the Hub client, so every layer sees the same transfer settings.


## 9. Upload Drift Details

### Single large file

Large model files, `.safetensors`, GGUF files, and dataset shards can trigger Xet path issues, route-specific stalls, timeout behavior, or fallback limitations.

Recommended order:

```text
1. Try default upload once and record behavior.
2. Try HF_XET_HIGH_PERFORMANCE=1.
3. Try HF_HUB_DISABLE_XET=1 as a stability test.
4. If Xet fails and HTTP fallback completes, recover with the stable path.
5. If HTTP fallback cannot handle the file, update hf_xet and retry the Xet path.
6. If all paths fail, reduce variables: different OS, different network, smaller test file, fresh cache.
```

### Large folder or many files

A folder with many small files may fail for reasons unrelated to raw bandwidth. You may be hitting request count, commit overhead, local filesystem overhead, or metadata handling.

Recommended order:

```text
1. Count files and total size.
2. Try folder-oriented upload rather than repeated single-file upload.
3. Reduce commit count for initial imports.
4. Avoid per-worker upload.
5. Consider sharding small files into larger logical groups.
6. If 429 occurs, treat it as rate-limit/request-count drift first.
```

### Resume and restart behavior

For large transfers, resume behavior is part of the diagnosis. If a restart loses pre-upload progress or re-uploads data that appeared complete, record that separately from transfer speed.

Use a small reproducible test whenever possible:

```text
- same file;
- same client versions;
- same env vars;
- same cache path;
- interrupted once at a known point;
- restarted once;
- note whether data is reused or resent.
```

### Commit design

For large repositories, transfer may complete but the commit step may fail or time out. Distinguish these states:

```text
bytes uploaded but commit fails:
  likely commit/request/server-timeout or repository-scale problem;

bytes never finish uploading:
  likely transfer path, network, cache, disk, or file-size problem;

many small files finish slowly:
  likely request-count, metadata, or filesystem overhead;

same large file re-uploads after restart:
  likely resume/staging/cache behavior.
```

## 10. Download Drift Details

### Large model files

Large model downloads may involve `snapshot_download`, `hf_hub_download`, `transformers.from_pretrained`, `datasets.load_dataset`, or the `hf` CLI. Each route can load authentication, cache, and Xet settings differently.

Compare these paths:

```bash
# CLI.
hf download REPO_ID FILENAME

# Python direct Hub API.
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('REPO_ID', 'FILENAME'))
PY

# Xet disabled.
HF_HUB_DISABLE_XET=1 hf download REPO_ID FILENAME
```

If a browser or `wget` path behaves differently from `huggingface_hub`, treat this as client/transfer-path drift, not just a network issue.

### Progress bar looks stuck

Some Xet progress behavior may update in large jumps. A progress bar that appears silent is not always a dead transfer, but it should be checked.

Check system activity:

```text
- network receive/send rate;
- disk write rate;
- CPU usage;
- cache directory growth;
- whether the same byte count repeats for many minutes;
- debug logs if safe to enable.
```

If there is no network activity, no disk activity, and no cache growth, it is more likely to be stuck.

### Browser, CLI, and Python disagree

Treat disagreement as useful evidence.

```text
Browser succeeds, huggingface_hub fails:
  suspect client version, Xet path, auth path, cache, or Python runtime.

CLI succeeds, Python fails:
  suspect Python process env vars, token source, library version, or wrapper library.

Python succeeds, CLI fails:
  suspect shell token state, PATH, virtualenv mismatch, or CLI using another installation.

HTTP fallback succeeds, Xet fails:
  suspect Xet path, cache, route, hf_xet version, or local filesystem.

Xet succeeds, HTTP fallback fails:
  suspect large-file requirements or HTTP path limitation.
```

## 11. Dataset and Parquet Path Drift

Dataset transfers have two separate questions:

```text
1. How are bytes transferred?
2. Which bytes are you asking for?
```

A raw dataset repository can contain many JSON, CSV, image, archive, or shard files. A task may not require the raw repository layout. It may be faster and more stable to access an auto-converted Parquet representation, a specific split, a subset of columns, or a streaming view.

Try this sequence:

```text
1. Check whether the dataset has auto-converted Parquet files.
2. If using analytics, prefer Parquet with DuckDB, Polars, pandas, or datasets streaming.
3. If training, read only the split and columns needed.
4. Avoid full repo clone/download unless the raw layout is required.
5. If load_dataset hangs only with Xet, test HF_HUB_DISABLE_XET=1 and record datasets version.
```

Example diagnostic:

```bash
python - <<'PY'
import datasets, huggingface_hub
print('datasets', datasets.__version__)
print('huggingface_hub', huggingface_hub.__version__)
PY
```

If the Parquet path is much faster, document that the fix is a format/access-path change, not a network-speed fix.

Useful access patterns:

```text
hf datasets parquet DATASET_REPO
hf://datasets/namespace/dataset@~parquet/path/to/file.parquet
Dataset Viewer /parquet endpoint
streaming=True in datasets when local materialization is unnecessary
```

## 12. Cache and Local Filesystem Drift

Hub transfers are not stateless. The standard Hub cache and Xet cache can affect speed, retries, deduplication, and apparent progress.

Important paths:

```text
HF_HOME
HF_HUB_CACHE
HF_XET_CACHE
~/.cache/huggingface/hub
~/.cache/huggingface/xet
```

Xet cache includes local state such as chunk cache, shard cache, and staging areas. If cache is on a slow disk, network filesystem, nearly full disk, antivirus-scanned directory, or unstable mount, transfers can behave badly.

Checklist:

```bash
# Show configured environment.
hf env

# Inspect cache summary on current hf CLI versions.
hf cache ls
hf cache ls --revisions

# Verify cached files when corruption is suspected.
hf cache verify REPO_ID

# Dry-run cache pruning. Do not delete blindly.
hf cache prune --dry-run
```

Rules:

```text
- Prefer local SSD cache for large transfers.
- Avoid NFS/network mounts for high-volume Xet cache when possible.
- Do not modify files inside the standard Hub cache manually.
- For Xet-cache isolation, prefer moving the Xet cache aside or using a fresh `HF_XET_CACHE`; official cache docs also describe removing the `xet` cache directory as a way to reclaim space or debug Xet-cache issues.
- Always keep enough disk headroom for temporary/staging files.
```

## 13. Account, Authentication, and Rate-Limit Drift

Authentication and plan tier can affect rate limits and bandwidth. Official rate-limit tiers distinguish Anonymous, Free, PRO, Team, Enterprise, and Enterprise Plus traffic, and PRO documentation lists higher bandwidth and API rate limits as a benefit. Therefore, record account state in every transfer report.

Do not use account tier as the first explanation for deterministic stalls. A logged-in or paid account can still hit Xet path bugs, cache problems, OS issues, client regressions, file-layout problems, or local disk bottlenecks.

Use this separation:

| Symptom | More likely class |
|---|---|
| `429 Too Many Requests` | Rate limit / request count / worker count / authentication. |
| `401` or `403` | Token, gated repo, private repo, or auth path. |
| Warning about unauthenticated requests | Missing token in that process. |
| Same transfer stalls after login | Xet/cache/client/network/file-layout may still be active. |
| Many workers fail but one worker succeeds | Request fan-out and cache design. |
| PRO improves throughput but stall remains | Rate/bandwidth improved; transfer path still suspect. |

Use shared cache and local loads for distributed jobs:

```python
from huggingface_hub import snapshot_download

snapshot_download('org/repo', local_dir='/mnt/shared/hf/org-repo')
```

Then:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/mnt/shared/hf/org-repo', local_files_only=True)
```

## 14. OS, Runtime, and Native Component Drift

`hf_xet` is not just ordinary Python code. It involves native components and Xet/CAS transfer behavior. OS, libc, Python version, packaging method, and runtime environment can matter.

Record:

```text
Windows / Linux / macOS / WSL
Colab / Kaggle / CI / Docker / bare metal
Python version
pip vs conda
x86_64 vs arm64
glibc vs musl
corporate proxy or VPN
local SSD vs HDD vs network mount
```

Examples of useful distinctions:

```text
- CLI command fails, Python API works.
- Python API fails, browser download works.
- Windows upload stalls, Linux upload completes.
- Colab token is visible to Python but not to shell CLI.
- Xet disabled works for 10 GB but not for 60 GB.
```

## 15. Recommended Troubleshooting Order

Use this order when the cause is unclear.

### Phase 1: Capture context

```text
- command used;
- exact error or stall point;
- file size and file count;
- client versions;
- auth/account state;
- OS and cache path;
- env vars;
- whether this is upload, download, dataset load, or folder transfer.
```

### Phase 2: Test path isolation

```bash
# Default.
hf download REPO_ID FILENAME

# Xet high-performance.
HF_XET_HIGH_PERFORMANCE=1 hf download REPO_ID FILENAME

# Xet disabled.
HF_HUB_DISABLE_XET=1 hf download REPO_ID FILENAME
```

Use equivalent upload commands for upload failures.

### Phase 3: Test auth and rate class

```bash
hf auth whoami
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().whoami())
PY
```

If rate limits are suspected, reduce worker count and pre-download to shared cache before changing account plan.

### Phase 4: Test cache and disk

```bash
hf cache ls
hf cache ls --revisions
hf cache prune --dry-run
```

Try a fresh cache on local SSD if cache corruption or filesystem speed is plausible.

### Phase 5: Test file layout

```text
- single huge file vs shards;
- many tiny files vs fewer shards;
- raw dataset vs Parquet;
- full repo download vs split/columns only;
- repeated upload_file vs upload_folder.
```

### Phase 6: Change one variable at a time

Do not upgrade the client, change cache, disable Xet, change token, move OS, and split files in one attempt. You will lose the explanation of what fixed it.

## 16. Troubleshooting Stories

These are not universal diagnoses. They are examples of how to think.

### Story 1: The old fast flag that did nothing

A maintainer follows an old snippet and sets `HF_HUB_ENABLE_HF_TRANSFER=1`. Upload remains slow. The correct interpretation is not "fast mode failed." In v1-era clients, that variable may be ignored. The maintainer has not actually tested a current fast path until `HF_XET_HIGH_PERFORMANCE=1` and default Xet behavior are tested.

### Story 2: The burst that looked healthy

A download shows a very high initial speed, then stops. The first burst may reflect local buffering, cache behavior, or an early stage of transfer rather than sustained end-to-end progress. A useful test compares default Xet with `HF_HUB_DISABLE_XET=1`, while also watching network, disk, and cache growth.

### Story 3: The dataset that was not a bandwidth problem

A raw dataset repository downloads slowly. The Parquet path is fast. The root problem was not the network link; it was the data access path. The fix is to read the auto-converted Parquet branch, a split, or selected columns instead of downloading the whole raw repository.

### Story 4: The paid account that did not fix the stall

A PRO account can provide higher bandwidth and rate limits, but Xet path bugs, cache problems, and filesystem bottlenecks can remain. If login or plan upgrade changes 429 behavior but not a deterministic stall, continue with transfer-path and cache diagnostics.

### Story 5: The many-worker cluster that looked like a Hub outage

A training job starts many workers. Each worker downloads the same model independently. Rate limits, repeated metadata calls, duplicated network traffic, and cache contention appear. The better design is to pre-download once, use a shared local path, and run workers with `local_files_only=True`.

## 17. Timeline of Relevant Transfer-Stack Drift

This timeline is not a complete product history. It is included to explain why older advice and newer behavior can disagree. Treat issue reports as examples of observed failure classes, not proof that every local failure has the same root cause.

| Date | Event or report | Use in this runbook |
|---|---|---|
| 2024-08 | Hugging Face acquired XetHub to replace Git LFS on the Hub. | Explains why Xet became a normal storage and transfer concept rather than an optional edge feature. |
| 2024-2025 | Older guidance often referenced `hf_transfer` and `HF_HUB_ENABLE_HF_TRANSFER`. | Useful historical context, but not sufficient for v1-era diagnosis. |
| 2025-10-27 | `huggingface_hub` v1.0 was released. | Major boundary: Python 3.9+ requirement, HTTPX migration, `hf_transfer` removal, `hf_xet` default transfer path, and ignored `HF_HUB_ENABLE_HF_TRANSFER`. |
| 2026-03 | A Windows report described uploads with new data above 2 GB slowing or stalling near 1.98 GB. | Useful direct example of upload drift; do not generalize it to all OSes without testing. |
| 2026-03 | A report described regular HTTP download limitations for very large files. | Explains why `HF_HUB_DISABLE_XET=1` is a diagnostic tool, not a universal final mode. |
| 2026-04 | Reports described Xet-backed downloads stalling in Colab or cloud routes, sometimes changing when Xet was disabled. | Supports paired default/Xet-disabled testing for download drift. |
| 2026-04 | Xet issues discussed timeout and stall behavior. | Supports recording Xet version, cache path, route, timeout, and retry behavior. |
| 2026-05 | A report described `HF_HUB_ENABLE_HF_TRANSFER=1` as a silent no-op on v1 upload paths. | Reinforces that old flags can make a test look meaningful when it is not. |

## 18. Known Issue Map

Use these links as starting points, not as proof that every local failure has the same root cause. Prefer the official documentation anchors first, then compare issue reports only after matching the command path, operating system, client versions, file shape, and failure signature.

### Official documentation anchors

| Anchor | Scope | What to use it for |
|---|---|---|
| [Migrating to huggingface_hub v1.0](https://huggingface.co/docs/huggingface_hub/concepts/migration) | v1 boundary, HTTPX, `hf_transfer` removal, `hf_xet` default | Decide whether old instructions are stale. |
| [huggingface_hub environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) | `HF_HUB_DISABLE_XET`, `HF_XET_HIGH_PERFORMANCE`, Xet cache vars, deprecated vars | Verify exact env-var spelling and meaning. |
| [Using Xet Storage](https://huggingface.co/docs/hub/xet/using-xet-storage) | Xet workflow, Git Xet, adaptive/fixed concurrency | Explain why transfer behavior can vary by client, cache, and route. |
| [huggingface_hub cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) | standard cache and Xet cache layout | Diagnose cache, shard, chunk, and staging behavior. |
| [huggingface_hub CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) | `hf auth`, `hf download`, `hf upload`, `hf cache`, `hf env`, `hf datasets parquet` | Keep commands current. |
| [Hub rate limits](https://huggingface.co/docs/hub/rate-limits) | Anonymous, Free, PRO, Team, Enterprise tiers | Separate rate-limit symptoms from transfer-path stalls. |
| [HF PRO account benefits](https://huggingface.co/docs/hub/pro) | PRO bandwidth and rate-limit benefits | Avoid overclaiming PRO as a fix for client/cache/Xet bugs. |
| [Dataset Viewer Parquet endpoint](https://huggingface.co/docs/dataset-viewer/parquet) | converted Parquet files, partial conversion, API endpoint | Diagnose dataset access-path drift. |
| [DuckDB and the `@~parquet` revision](https://huggingface.co/docs/hub/datasets-duckdb) | querying auto-converted Parquet through `hf://` | Use Parquet directly instead of raw repo download when appropriate. |

### Transfer-stack issue examples

| Link | Main scope | Use carefully |
|---|---|---|
| [`HF_HUB_ENABLE_HF_TRANSFER=1` no-op on v1 upload path](https://github.com/huggingface/huggingface_hub/issues/4219) | stale env-var / v1 upload-path confusion | Good evidence for warning users away from legacy `hf_transfer` assumptions. |
| [Upload speeds extremely slow / stalling since April 1st](https://discuss.huggingface.co/t/upload-speeds-extremely-slow-stalling-since-april-1st/174910) | large upload burst-then-stall reports | Forum evidence; use as symptom context, not final root cause. |
| [Uploads with new data over 2 GB get stuck on Windows](https://github.com/huggingface/huggingface_hub/issues/3871) | Windows large-upload drift | Do not generalize to Linux/macOS without reproducing. |
| [Large file upload failures](https://github.com/huggingface/huggingface_hub/issues/3747) | large upload failure examples | Compare command, OS, version, and file shape before applying. |
| [Large folder upload pain points](https://github.com/huggingface/huggingface_hub/issues/2612) | many-file / large-repo uploads | Use for file-count and commit-design diagnosis. |
| [Xet upload timeout / `upload_xorb` timeout discussion](https://github.com/huggingface/xet-core/issues/807) | timeout and retry behavior | Useful for timeout vocabulary and retry-capture fields. |
| [Xet downloads stalling](https://github.com/huggingface/xet-core/issues/789) | download stalls | Use for paired Xet/default/fallback thinking. |
| [Xet-backed downloads stall on GCP](https://github.com/huggingface/xet-core/issues/800) | route/cloud-specific download drift | Good reminder to record cloud provider and route. |
| [Large downloads via Hub stuck in Colab](https://github.com/huggingface/huggingface_hub/issues/4085) | notebook/cloud download drift | Good reminder to record runtime and auth path. |
| [HTTP download fails for files above 50 GB](https://github.com/huggingface/huggingface_hub/issues/3868) | large-file fallback limitation | Treat as issue-specific evidence that HTTP fallback may not be enough. |
| [Downloading not working with hf_xet](https://github.com/huggingface/huggingface_hub/issues/3960) | incomplete or failing download path | Compare against current `hf_xet` and client versions. |
| [`HF_HUB_DISABLE_XET` version-specific behavior](https://github.com/huggingface/huggingface_hub/issues/3266) | env-var behavior drift | Useful reminder to record exact client versions. |
| [CLI auth and Python auth can differ in Colab](https://github.com/huggingface/huggingface_hub/issues/3774) | auth path mismatch | Use when CLI and Python behavior disagree. |
| [Many-file upload and rate-limit discussion](https://discuss.huggingface.co/t/how-to-get-around-rate-limits/151947) | many workers / many requests / rate limits | Separate request fan-out from raw bandwidth. |
| [Download speed improved by Parquet access path](https://discuss.huggingface.co/t/download-speed-way-too-slow/169824/12) | dataset format-path drift | Use as a prompt to test Parquet/streaming. |
| [Kaggle dataset download stuck with Xet path](https://discuss.huggingface.co/t/huggingface-dataset-download-stuck-in-kaggle/175183) | hosted notebook / route / Xet path | Record platform and cache details. |
| [datasets `load_dataset` hang with hf_xet path](https://github.com/huggingface/datasets/issues/8129) | datasets wrapper plus Xet path | Keep direct Hub tests separate from `datasets` behavior. |

## 19. Minimal Issue Template

When reporting or handing off a Hub transfer problem, include this block.

```text
Problem summary:
  upload/download/dataset load/folder upload is slow/stuck/failing

Operation:
  hf CLI / huggingface_hub Python / datasets / transformers / git

Command or code:
  <paste sanitized command or minimal code>

Repo:
  repo_id:
  repo_type:
  public/private/gated:

Data shape:
  largest file size:
  total size:
  file count:
  format:
  raw files or Parquet:

Client stack:
  huggingface_hub:
  hf_xet:
  hf CLI:
  datasets:
  transformers:
  Python:
  OS:
  pip/conda/container:

Authentication:
  anonymous/free/pro/team/enterprise:
  hf auth whoami result:
  Python HfApi().whoami result:
  token passed to the actual process: yes/no/unknown

Environment variables:
  HF_HUB_DISABLE_XET:
  HF_XET_HIGH_PERFORMANCE:
  HF_XET_FIXED_UPLOAD_CONCURRENCY:
  HF_XET_FIXED_DOWNLOAD_CONCURRENCY:
  HF_XET_NUM_CONCURRENT_RANGE_GETS:
  HF_HOME:
  HF_HUB_CACHE:
  HF_XET_CACHE:

Cache and disk:
  cache location:
  disk type:
  free disk:
  shared or local:
  antivirus/proxy/VPN:

Path tests:
  default result:
  HF_XET_HIGH_PERFORMANCE result:
  HF_HUB_DISABLE_XET result:
  browser/wget result if tested:
  Parquet path result if dataset:

Observed failure:
  exact error:
  stall point:
  timeout duration:
  retry behavior:
  progress bar behavior:
```

## 20. Practical Defaults

These defaults are intentionally conservative. They are meant to produce a useful first recovery path without hiding the cause of the failure.


For a first recovery attempt:

```text
- Record versions first.
- Verify auth in the actual process.
- Use current `hf cache ls`, `hf cache prune --dry-run`, and `hf cache verify` commands rather than older cache-scan examples when using the v1 CLI.
- Test default, Xet high-performance, and Xet-disabled paths separately.
- Use HTTP fallback if it is slower but completes and the file size allows it.
- Do not permanently disable Xet without checking large-file requirements.
- Use local SSD cache when possible.
- For datasets, check Parquet or streaming before downloading raw repositories.
- For clusters, pre-download once and reuse a shared cache or local directory.
- For many-file uploads, reduce request count rather than only changing bandwidth settings.
- When a progress bar looks stuck, check network, disk, and cache activity before killing the process.
```

## 21. Closing Rule

A Hub transfer failure is rarely explained by one variable alone. Treat `huggingface_hub`, `hf_xet`, Xet mode, HTTP fallback, authentication state, account plan, cache, filesystem, OS, file layout, and dataset format as one transfer stack. Change one variable at a time, and keep the test record.
