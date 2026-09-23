---
source: "huggingface+chat+files+web"
topic: "Hugging Face Spaces Git 1GB hard limit and workarounds"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T02:20:01Z"
---

# Hugging Face Spaces Git 1GB hard limit and workarounds

## 1. Background and overview

### 1.1 What changed

Starting in 2024–2025, Hugging Face introduced a **hard ~1 GB size limit for the Git repository of each Space**. When the total size of all Git objects and Git‑LFS blobs reachable from the current history crosses this threshold, operations that need to upload or clone the repo fail with messages such as:

- `batch response: Repository storage limit reached (Max: 1 GB)` when pushing from Git.
- `Build error: Error while cloning repository` or similar messages in the Space logs when the build system tries to clone a too‑large Space repo.

This limit is:

- **Per Space repository** (code repo), not per account.
- **Independent from** your account’s global storage quota and from any **persistent storage** that you may have attached to the Space.
- **Currently not user‑upgradeable** – forum replies from Hugging Face staff explicitly confirm there is no way to increase the Git repo cap for Spaces.

The intent is to keep Space repos focused on **app code and small assets**, not as general-purpose storage for large models or datasets.

### 1.2 Two different “storages” you must distinguish

There are two important, separate limits that affect Spaces:

1. **Space Git repository (this topic)**  
   - Hard cap of about **1 GB** on the Git repo used for the Space.  
   - Includes all Git objects and Git‑LFS blobs that are reachable from your current history.  
   - Exceeding this limit blocks pushes and can cause **build-time clone errors**.

2. **Runtime disk (VM filesystem)**  
   - Ephemeral root disk (for free CPU Spaces typically **~50 GB**; larger on some GPU SKUs).  
   - Optional **persistent storage**, usually mounted at `/data`, with its own paid quota.  
   - Overfilling the ephemeral root disk triggers eviction messages like `Workload evicted, storage limit exceeded (50G)` and causes the Space to be restarted, but this has nothing to do with the 1 GB repo cap.

A common source of confusion is mixing the two:

- The **1 GB Git limit** is about how much data can be stored in the Space’s Git history. It is enforced during `git push` and clone.
- The **50 GB+ runtime limit** is about how much disk your app uses at runtime (caches, temporary files, downloaded models). It is enforced by the underlying Kubernetes cluster.

The correct mitigation depends on which one you hit.

### 1.3 Why Hugging Face enforces a 1 GB repo limit

Official storage‑limit documentation plus forum responses highlight several reasons:

- Spaces are designed to host **apps and demos**, not large artifacts. Large weights belong in **model** or **dataset** repos.
- Git, especially with Git‑LFS, is inefficient for storing huge binary blobs across many commits; a few large files or a long history of large model checkpoints can quickly bloat the repo.
- Keeping Space repos small improves **clone times**, **build reliability**, and **resource usage** for the shared Spaces infrastructure.

Instead of lifting the cap, Hugging Face recommends architectural patterns that keep Spaces lean and move heavy files to more appropriate locations.

## 2. What the official docs and forums say

### 2.1 Forum thread: “1GB storage limit in Spaces?”

The forum thread “1GB storage limit in spaces?” clarifies that the error:

> `batch response: Repository storage limit reached (Max: 1 GB)`

means you hit the **Space repo cap**, not the 50 GB runtime limit. Staff and community replies note that:

- **1 GB applies specifically to Spaces’ Git repos** (code).  
- The runtime disk can still be much larger (50 GB ephemeral root, plus optional `/data` persistent volume).  
- The recommended pattern is: keep the Space repo small, and put large model weights in a **separate model or dataset repo**, then download them at runtime.

The thread also lists related topics like “Lfs Storage cap” and “Spaces force push getting ‘Repository storage limit reached’”, which all converge on the same conclusion: the repo cap is fixed, so the design has to adapt.

### 2.2 Forum threads: push failures and build errors

Multiple threads show the same failure patterns:

- **Push failures**:  
  When pushing from local Git, users see:

  ```text
  batch response: Repository storage limit reached (Max: 1 GB)
  error: failed to push some refs to ...
  ```

  Even after deleting large files from the working tree, the error persists. The root cause is that **Git history still contains old commits referencing large LFS blobs**.

- **Build-time clone errors**:  
  Spaces stuck in **“Building”** with a “Build error: Error while cloning repository” log line are often hitting the same 1 GB limit during clone. The builder cannot fetch the overweight repo, so the build fails before user code runs.

- **Cloned Spaces and duplicates**:  
  Duplicating a Space or cloning its repo locally and then pushing to a new Space can **carry over the heavy history**, causing the new Space to hit the repo limit immediately, even when the current working tree is small.

### 2.3 “Wanted to know how to increase the storage limit in git lfs …”

Another forum answer spells out the situation succinctly:

- The **Spaces Git repo limit is 1 GB** and cannot be increased.  
- However, after startup, the Space can use up to **50 GB of runtime disk** (more on larger hardware tiers).  
- Therefore, you should:
  - keep the Git repo small (ideally tens of MB, not hundreds),
  - download big files at runtime (via `pip`, `huggingface_hub`, `wget`, etc.), and
  - store them on runtime disk or persistent `/data` storage.

The same reply suggests using **model or dataset repos** for large files, then using runtime downloads rather than committing those files into the Space repo.

### 2.4 Docs: storage limits and persistent storage

The general **Storage limits** doc and the **Spaces storage** doc add a few key points:

- Model and dataset repos have different per‑file limits (e.g., 50 GB per file) and can be scaled to much larger total storage; this is where you should host large models and datasets.  
- Spaces with paid persistent storage get a `/data` mount that survives restarts and can hold tens or hundreds of GB of artifacts and caches.  
- None of these change the **1 GB Git cap** for Space repos themselves.

Putting it together: your long‑term design must treat the Space repo as a **small code repository** and keep heavy assets elsewhere.

## 3. How the 1 GB limit shows up in practice

### 3.1 Classic symptoms

Common symptoms that indicate you hit the Git repo limit:

- `git push` fails with `Repository storage limit reached (Max: 1 GB)`.
- The Space UI build status is **stuck in “Building”** with logs showing “Error while cloning repository” and no further details.
- Storage usage in the account/org **looks fine** (for example, “59 GB used of 1 TB”), which can be misleading because that quota is for **Hub storage in general**, not for a single Space repo.

### 3.2 Why deleting files from the working tree often does not help

Deleting large files in the current commit but keeping the old commits in history does **not** free repo space for the purpose of the 1 GB cap. The server still counts all LFS objects and Git blobs referenced by any commit you push.

Typical pitfalls:

- You tracked large files (e.g., `*.pt`, `*.safetensors`, `.pth`) with Git‑LFS for a while, then removed them from the working tree but never **rewrote history**.  
- You deleted a Space and created a new one, but reused your old local repo with its heavy history; pushing still sends the same large LFS objects and hits the limit again.  
- You used `git lfs push --all`, which explicitly uploads all historic LFS objects, including those you no longer use.

### 3.3 LFS and remote storage accounting

For Spaces, the server counts **all LFS objects reachable from your pushed refs** against the 1 GB limit. As a result:

- Simply removing `.gitattributes` lines or running `git rm` on large files in the latest commit is insufficient.  
- You must either:
  - push a brand‑new repo with a clean history, or  
  - rewrite the current repo history (drop or shrink large blobs) and then force‑push.

Hugging Face’s “Storage usage” and “List LFS files” pages for the Space can help you identify which paths and commits consume most of the quota.

## 4. Inspecting and debugging repo size

### 4.1 Check usage from the Hub UI

For a given Space:

1. Go to the Space page.  
2. Open **Settings → Storage** (or “Storage usage”).  
3. Use the **“List LFS files”** view to see which LFS paths and commits are taking space.  
4. If available, use “delete” actions or **super_squash_history** tools to drop unused history.

These tools operate on the remote repository and are the safest first step if you are not comfortable rewriting Git history locally.

### 4.2 Inspect local history and LFS objects

On your local clone, you can run:

```bash
git remote -v

# List LFS-tracked files and their sizes
git lfs ls-files

# Approximate local LFS cache size
du -sh .git/lfs/objects

# Summary of Git object sizes
git count-objects -Hv
```

If `.git/lfs/objects` is already hundreds of MB or more, or if `git lfs ls-files` lists big binary files (models, datasets, media), your push will likely hit the 1 GB limit unless you clean the history.

### 4.3 Distinguish repo limit from runtime disk issues

If you suspect you may also be hitting the **runtime disk** limit (50 GB+), check from a running Space (via the logs or SSH if enabled):

```bash
df -h /
df -h /data  # if persistent storage is attached
```

- `df -h /` near 100% points to ephemeral root disk pressure, not Git.  
- Eviction messages like `Workload evicted, storage limit exceeded (50G)` come from the runtime, not the Git quota.

It is common to run into both issues in large projects, but they are solved differently.

## 5. Recommended design patterns and workarounds

### 5.1 Treat the Space repo as “code-only”

A robust long‑term pattern is:

- Keep the Space repo under **a few hundred MB at most**, ideally under 100 MB.  
- Do **not** store:
  - Large model checkpoints (`.bin`, `.pt`, `.safetensors`).  
  - Big datasets (CSV/JSONL/Parquet above a few MB).  
  - Generated outputs or logs.
- Restrict Git‑LFS patterns in `.gitattributes` to things you truly need in the repo (e.g., a few small logo images or icons). Avoid overly broad patterns like `* filter=lfs` applied to the entire repo.

Instead, host heavy assets in:

- **Model repos** (for model weights, LoRAs, adapters).  
- **Dataset repos** (for training/eval data or large static assets).  
- External object storage (S3, GCS, etc.), if needed.

### 5.2 Download large assets at runtime

From your Space code (e.g., `app.py` or `app.py` + `requirements.txt`), download the required models or data at runtime using the Hub APIs:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download a single file (e.g., a model checkpoint)
ckpt_path = hf_hub_download(
    repo_id="your-org/your-model-repo",
    filename="model.safetensors",
)

# Or, download an entire repo snapshot (weights + config)
local_dir = snapshot_download(
    repo_id="your-org/your-model-repo",
    revision="main",
    local_dir="/data/models/your-model",
    local_dir_use_symlinks=False,
)
```

Guidelines:

- If you have persistent storage, point `local_dir` or `HF_HOME` under `/data` so downloads survive restarts and do not count toward the Git repo limit.  
- If you only have ephemeral storage, be ready to re‑download on restart (and use caching wisely).  
- Use `local_dir_use_symlinks=False` when storing in `/data` to avoid confusing path structures and to keep disk usage predictable.

### 5.3 Use Docker / startup scripts to install dependencies

Instead of committing bulky packages or wheels into the Space repo, let the Space **install them at build or startup time**:

- For Gradio/Streamlit Spaces: list Python packages in `requirements.txt` or `pyproject.toml`.  
- For Docker Spaces: install in your `Dockerfile` using `pip install` or system package managers.

This approach uses runtime disk instead of Git for storing dependencies, keeping the repo small.

### 5.4 Move caches and temp files off root (runtime concern)

Although separate from the 1 GB repo limit, it is good practice to redirect caches and temporary files away from the root filesystem and into `/data` when you have persistent storage:

```bash
# Example in bash (e.g., at startup)
export HF_HOME=/data/.cache
export HF_HUB_CACHE=/data/.cache/huggingface/hub
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
export XDG_CACHE_HOME=/data/.cache
export TORCH_HOME=/data/.cache/torch
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR" /data/.cache
```

This prevents runtime caches from filling the 50 GB root disk and is especially important for large model downloads.

## 6. Concrete mitigation recipes for the 1 GB repo cap

### 6.1 Fastest fix: start from a clean repo

If you do not need to preserve commit history, the simplest and safest method is:

1. Create a fresh directory that contains only the files you currently need for the Space.  
2. Initialize a new Git repo and commit once.  
3. Connect it to the Space and push.

Example:

```bash
# 1) Start from your current project folder
mkdir /tmp/clean-space
rsync -a --delete --exclude '.git' ./ /tmp/clean-space/
cd /tmp/clean-space

# 2) New Git repo with lean history
git init --initial-branch=main
git lfs uninstall  # optional: avoid LFS entirely in the new repo
git add .
git commit -m "Clean Space: code only"

# 3) Connect to your Space and push
git remote add origin https://huggingface.co/spaces/your-username/your-space
git push -u origin main
```

By pushing a repo with a brand‑new, tiny history, you guarantee that no large historic blobs are counted against the 1 GB limit.

### 6.2 Rewrite history to drop big files (git‑filter‑repo / BFG)

If you need to keep the existing repo name and some history, you can **rewrite the Git history** to remove large files and commits:

1. Install `git-filter-repo` (recommended) or BFG Repo‑Cleaner.  
2. Remove large files or file types from all commits.  
3. Run `git lfs prune` and garbage collection locally.  
4. Force‑push the cleaned history.

Example with `git-filter-repo`:

```bash
pip install git-filter-repo

# Remove all large model files from history
git filter-repo --path-glob '*.pt' --path-glob '*.bin' --path-glob '*.safetensors' --invert-paths

# Clean up LFS cache and Git objects
git lfs prune
git gc --prune=now --aggressive

# Force-push rewritten history
git push origin main --force
```

After this, the remote Space repo only sees the cleaned history and should fall below the 1 GB limit (assuming remaining assets are small).

### 6.3 Tighten `.gitattributes` for LFS

Avoid tracking more files than necessary with LFS. For example, instead of a broad rule like:

```gitattributes
* filter=lfs diff=lfs merge=lfs -text
```

prefer more selective patterns:

```gitattributes
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
```

And whenever possible, keep large binary artifacts **out of the Space repo entirely**.

### 6.4 Use Git LFS wisely when cloning into Spaces

If you need to clone model or dataset repos (or even other Spaces) inside a Space and you want to avoid pulling their large LFS blobs onto the 50 GB root disk:

```bash
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/your-org/your-model-repo
```

This prevents large LFS files from being downloaded automatically during clone; you can fetch specific files you need later using the Hub APIs or `git lfs pull` on a subset of paths. This is mainly a **runtime disk** optimization but can also help keep temporary clones from bloating your own Space repo if you accidentally commit them.

## 7. Limitations, caveats, and open questions

### 7.1 Hard 1 GB limit

As of late 2025:

- The 1 GB Git repo limit for Spaces is **hard**; there is no setting or paid plan to raise it.  
- Account‑level storage upgrades affect model/dataset/private storage quotas, not the Spaces Git limit.

Any design that depends on storing multi‑GB artifacts in the Space repo is therefore brittle and will eventually break.

### 7.2 UI vs quota discrepancies

Some users report cases where:

- The UI shows low storage usage for the Space, but pushes still fail with `Repository storage limit reached (Max: 1 GB)`.  
- Or account‑level quotas show plenty of remaining storage, but a single Space is blocked.

These are usually due to **historic LFS blobs** still being referenced by the repo, or to temporary inconsistencies between UI and backend. If cleaning history and LFS objects does not resolve the discrepancy, contacting Hugging Face support is the next step.

### 7.3 Multiple branches, tags, and forks

Remember that **all branches and tags you push** count toward the 1 GB limit. Pushing extra branches with heavy history, or tags that point to large past commits with huge artifacts, can unexpectedly push you over the cap. Best practices:

- Keep only necessary branches in the Space repo.  
- Avoid heavyweight tags that point to old experimental commits with large artifacts.  
- Keep experimental work in separate repos whenever possible.

### 7.4 Interaction with runtime disk issues

It is easy to fix the Git repo problem but still hit runtime disk limits later, or vice versa. A Space can be:

- Under 1 GB Git repo size but evicted due to filling /tmp, caches, or `/` to 50 GB.  
- Under 50 GB runtime disk usage but still blocked on pushes due to heavy Git/LFS history.

Always diagnose which limit you are actually hitting before applying a fix.

## 8. References and further reading

### 8.1 Hugging Face documentation

- Storage limits overview (models, datasets, Spaces, quotas):  
  <https://huggingface.co/docs/hub/en/storage-limits>
- Spaces overview (hardware, runtime disk, persistent storage):  
  <https://huggingface.co/docs/hub/en/spaces-overview>
- Spaces persistent storage and `/data` usage:  
  <https://huggingface.co/docs/hub/en/spaces-storage>
- `huggingface_hub` download guides (runtime downloads via `hf_hub_download` and `snapshot_download`):  
  <https://huggingface.co/docs/huggingface_hub/en/guides/download>

### 8.2 Key forum threads

- “1GB storage limit in spaces?” — discussion and explanation of the 1 GB repo cap:  
  <https://discuss.huggingface.co/t/1gb-storage-limit-in-spaces/170196>
- “Build error: Error while cloning repository” — examples of build failures caused by repo size and other issues:  
  <https://discuss.huggingface.co/t/build-error-error-while-cloning-repository/113801>
- “Error: Repository storage limit reached (Max: 1 GB)” — multiple reports and staff answer confirming the cap and recommending clean histories:  
  <https://discuss.huggingface.co/t/error-repository-storage-limit-reached-max-1-gb/128910>
- “Spaces force push getting ‘Repository storage limit reached’” — push failures and LFS‑history pitfalls:  
  <https://discuss.huggingface.co/t/spaces-force-push-getting-repository-storage-limit-reached/130269>
- “Freeing memory of my space” and “Wanted to know how to increase the storage limit in git lfs …” — symptoms, partial fixes, and official guidance on using runtime disk instead of repo storage:  
  <https://discuss.huggingface.co/t/freeing-memory-of-my-space/128894>  
  <https://discuss.huggingface.co/t/wanted-to-know-how-to-increase-the-storage-limit-in-git-lfs-which-is-set-to-1-gb/160300>

### 8.3 Git and Git‑LFS resources

- Git LFS push behavior (which objects get uploaded):  
  <https://man.archlinux.org/man/extra/git-lfs/git-lfs-push.1.en>
- GitHub docs: removing large files or sensitive data from history with `git-filter-repo`:  
  <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>
- BFG Repo-Cleaner:  
  <https://rtyley.github.io/bfg-repo-cleaner/>

### 8.4 Related internal notes

- Internal notes on **Space runtime disk quotas** (50 GB ephemeral root, `/data` persistent storage patterns) and redirecting caches to `/data`.  
- Internal notes on **Space deletion and recreation** not resetting local Git history, explaining why pushing to a new Space can still trigger the 1 GB limit if the local repo history is heavy.
