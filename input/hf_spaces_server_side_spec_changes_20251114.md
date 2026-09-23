---
source: "huggingface+chat"
topic: "Hugging Face Spaces server-side spec changes: frequent errors and mitigations"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T03:02:02Z"
---

# Hugging Face Spaces server-side spec changes: frequent errors and mitigations (2024–2025)

This note focuses on **errors that appear or change behavior because Hugging Face updated the Spaces runtime on the server side**, not because a user changed their own code. It is specifically about:

- **ZeroGPU renovations and stateless GPU changes**
- **Base image / OS updates for Docker & Python Spaces**
- **Platform-wide library stack changes (Gradio, Pydantic, Transformers, `huggingface_hub`, etc.)**
- **Build / deploy pipeline issues such as “forever building” or new build errors**

The aim is to make it easier to recognize “this is a Hugging Face Spaces spec change” vs. “this is my bug”, and to collect practical mitigations.

## 1. Background: how Spaces evolve on the server side

Hugging Face Spaces are not a static environment. The platform continuously updates:

- **Container base images and OS** (e.g. Debian variants, preinstalled system packages)
- **Preinstalled Python libraries** (Gradio, FastAPI, Pydantic, Torch, `huggingface_hub`, etc.)
- **GPU infrastructure**, especially for **ZeroGPU** (stateless GPU slices on NVIDIA H200)  
  – documented in the ZeroGPU overview and advanced compute docs.  
  See: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [Advanced compute options](https://huggingface.co/docs/hub/advanced-compute-options).

These changes can **break previously working Spaces without any change in your repo**, typically after:

- Restarting or un-pausing a Space
- Switching hardware (CPU → ZeroGPU or vice versa)
- Creating a new Space from an older template
- Large platform upgrades (ZeroGPU “renovations”, updated base images, library resets)

Community discussions (especially in the **ZeroGPU Explorers** Space and forum threads) show recurring patterns where many users hit the **same error signatures immediately after a platform change**, then Hugging Face staff or power users share workarounds or confirm server-side fixes later.

Key references (2024–2025):

- John6666’s Space post summarizing Space failures after a server update and common version-pin workarounds:  
  [HF post: “If your Space stops working after restarting…”](https://huggingface.co/posts/John6666/369491746519704)
- ZeroGPU Explorers org discussions documenting **ZeroGPU renovations**, new hardware, and associated bugs:  
  [Multiple errors associated with HF-wide ZeroGPU space renovation (Sept 2024)](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104)  
  [“Getting 500 internal error on any ‘outside’ ZeroGPU space”](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/123)
- Forum threads around **build pipeline** and **Docker template** breakages:  
  [Streamlit Docker Space permanently in “Building” state](https://discuss.huggingface.co/t/streamlit-docker-space-permanently-in-building-state/168910/3)  
  [Build error: “Job failed with exit code 1” while creating a Space](https://discuss.huggingface.co/t/build-error-job-failed-with-exit-code-1-while-creating-a-space-at-hugging-face/166760)

This document collects those patterns and adds concrete mitigation checklists.

## 2. ZeroGPU & stateless GPU spec changes

### 2.1 What changed on ZeroGPU

From the ZeroGPU docs and advanced compute pages:

- ZeroGPU is now a **stateless, shared GPU infrastructure** that dynamically allocates **NVIDIA H200 GPUs** to Spaces.  
  See: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [Advanced compute options](https://huggingface.co/docs/hub/advanced-compute-options).
- GPU slices are attached only when needed and managed by the `spaces.zero` runtime and decorators like `@spaces.GPU`.
- The platform enforces **stronger constraints**:
  - **No direct CUDA initialization in the main process**.
  - GPU work must happen inside functions wrapped by the Spaces runtime (e.g. `@spaces.GPU`).
  - The runtime intercepts Torch CUDA calls and raises specific errors if you violate these rules.

The key consequence: **old ZeroGPU code that preloaded models or called CUDA at import time** can suddenly break when the underlying ZeroGPU runtime changes, even if you never touched your app code.

### 2.2 Error: “CUDA must not be initialized in the main process on Spaces with Stateless GPU environment”

**Typical symptom**

Many ZeroGPU Spaces started failing with a runtime error like:

> `RuntimeError: CUDA must not be initialized in the main process on Spaces with Stateless GPU environment.`

You can see this error in:

- ZeroGPU Explorers discussions and reproduction Spaces.  
  Example: [ZeroGPU Explorers – discussion #72 and #104](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104)
- Example Spaces that crash when importing or preloading models (e.g. Bark, vLLM demos, custom diffusion Apps).  
  Example issues and traces:  
  [vLLM does not work with Hugging Face ZeroGPU Spaces](https://github.com/vllm-project/vllm/issues/3510)  
  [Example Bark Space with ZeroGPU error](https://huggingface.co/spaces/devilent2/bark)

**Root cause (server + app interaction)**

- The **ZeroGPU runtime wraps PyTorch CUDA entry points**; when it sees CUDA being initialized outside a managed context, it throws this RuntimeError.
- Before the ZeroGPU “stateless GPU” rollout and H200 migration, some patterns (global `model.to("cuda")`, `torch.cuda.is_available()` at import time, etc.) might have worked accidentally.
- After the upgrade, the **same code now violates the new spec**.

**Common triggers**

- Global or module-level calls to:
  - `torch.cuda.is_available()`
  - `torch.cuda.set_device(...)`
  - `model.to("cuda")` / `model.cuda()` in `__init__` or at import time
  - Library internals that touch CUDA when importing or preloading models (e.g. Bark’s `preload_models()`, custom model wrappers)
- Doing GPU work in code paths **not wrapped** with `@spaces.GPU` (or equivalent) or **before** the ZeroGPU runtime has attached a GPU slice.

**Mitigations**

1. **Move CUDA calls into `@spaces.GPU`-decorated functions**

   ```python
   # app.py
   import spaces
   import torch
   from diffusers import StableDiffusionPipeline  # https://huggingface.co/docs/diffusers

   pipe = None  # keep a CPU-only or lazy placeholder

   @spaces.GPU  # ZeroGPU-managed function
   def generate(prompt: str):
       global pipe
       if pipe is None:
           # All heavy CUDA work happens here
           pipe = StableDiffusionPipeline.from_pretrained(
               "runwayml/stable-diffusion-v1-5",
               torch_dtype=torch.float16,  # HF docs: use half precision on GPUs
           ).to("cuda")  # safe: within @spaces.GPU
       image = pipe(prompt).images[0]
       return image
   ```

   - Avoid `to("cuda")` or CUDA checks at import time.
   - If a third-party library preloads to CUDA, look for a “device” or “dtype” option to keep it on CPU until inside `@spaces.GPU`.

2. **Patch or wrap third‑party libraries that initialize CUDA too early**

   - For Bark-like libraries, look for flags to **disable preload** or to set `device="cpu"` during init.
   - Wrap heavy “preload” functions inside a `@spaces.GPU` entrypoint where possible.
   - When that’s impossible, consider **forking** the library to remove top-level CUDA calls.

3. **Upgrade PyTorch to a version that supports H200 GPUs**

   - ZeroGPU now uses H200 hardware; some older PyTorch versions do not fully support it.
   - Community posts (and John6666’s summary) suggest using **PyTorch ≥ 2.2.x** for ZeroGPU Spaces when you hit low-level CUDA / driver issues.
   - Pin a compatible PyTorch version in `requirements.txt` and keep it consistent with any CUDA or `torchvision` requirements.

4. **Avoid global “cleanup” calls that touch CUDA**

   - Functions such as `torch.cuda.empty_cache()` or `torch.cuda.synchronize()` in module-level code can trigger the same error.
   - Move them into your `@spaces.GPU` logic or remove them entirely if not essential.

### 2.3 Bugs during the ZeroGPU “renovation” period

The ZeroGPU Explorers community documents a cluster of **platform-triggered bugs** during the zero-GPU “renovation” in 2024:

- Spaces sometimes entered an **endless “Building” state**, even without code changes.
- Some apps crashed when **Gradio “Examples” contained `None` values**, especially in checkboxes.
- Gradio `Progress` components behaved differently on ZeroGPU vs CPU Spaces (crashes on ZeroGPU only).
- Certain patterns of calling class methods or using `yield` inside `@spaces.GPU`-decorated functions exposed race conditions in the runtime.

In those threads, workarounds included:

- **Removing `None` from `Examples` data** for Gradio apps.
- Refactoring class methods referenced from Gradio events so that **GPU work lives in simple functions** wrapped with `@spaces.GPU`, often using `yield from` to stream results.
- Temporarily **restarting** affected Spaces and waiting for HF-side fixes when the problem was clearly platform-related and reproducible across many users.

Key discussions:

- [Multiple errors associated with HF-wide ZeroGPU space renovation (Sept 2024)](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104)
- ZeroGPU error reports with detailed traces and reproduction Spaces from community members and HF staff.

## 3. Build / deployment pipeline changes

### 3.1 “Job failed with exit code 1” when creating a new Docker Space

**Symptoms**

- Creating a new **Docker-type Space** using the default template fails immediately with a build error such as:

  > `Build error: Job failed with exit code 1`  
  > Logs show `Unable to locate package software-properties-common` or other missing Debian packages.

- The underlying log shows that `apt-get install` cannot find certain packages that the template expects.

**Root cause**

A forum thread documents that **Hugging Face updated the base Docker image to a newer Debian release (e.g. Trixie)**, which removed some packages used in older templates such as `software-properties-common` or `python3-distutils`.

Reference:

- [Build error: Job failed with exit code 1 while creating a Space](https://discuss.huggingface.co/t/build-error-job-failed-with-exit-code-1-while-creating-a-space-at-hugging-face/166760)

**Mitigations**

1. **Edit the Dockerfile to match the new base OS**

   - Remove obsolete packages like `software-properties-common` if they’re no longer available.
   - Replace old patterns with newer, supported ones (e.g. using `apt-get install -y python3-venv` instead of older Python packaging utilities).
   - Keep the Dockerfile minimal and rely on Python packages rather than OS packages when possible.

2. **Use current official templates and docs**

   - Re-generate a Dockerfile from the latest examples in [Spaces documentation](https://huggingface.co/docs/hub/spaces-overview) or from an up-to-date template.
   - Compare your failing Dockerfile with a new working example and apply the differences.

3. **Add CI or local Docker builds**

   - Build the same Dockerfile locally using `docker build` to catch OS-level issues before pushing to Spaces.
   - This reduces surprises when Hugging Face upgrades the build base image.

### 3.2 Spaces stuck forever in “Building” with no logs

**Symptoms**

- A Space (often Streamlit or Docker) gets stuck in the **“Building”** state.
- No useful logs appear in the build output, and the build never completes.
- Restarting doesn’t resolve the issue.

Forum/Space references:

- [Streamlit Docker Space permanently in “Building” state](https://discuss.huggingface.co/t/streamlit-docker-space-permanently-in-building-state/168910/3)
- ZeroGPU Explorers threads referencing an “eternal stack of builds” as a platform-level regression.

**Root cause**

This pattern is usually **a platform bug in the build scheduler or metadata**:

- Some internal flag or cached state marks the Space as building even though the build has already finished or failed.
- Changing the SDK type (e.g. from Docker to Static and back) can sometimes clear the bad state.

**Mitigations**

1. **Make a minimal edit and push again**

   - Touch `app.py`, `requirements.txt`, or `Dockerfile` and push a trivial change.
   - Sometimes a new build request clears the stale state.

2. **Toggle SDK type in `README.md`**

   - Temporarily switch `sdk:` in `README.md` to another supported type (e.g. Static), push, then switch back.
   - Community reports indicate this can force a rebuild with fresh metadata for some stuck Spaces.

3. **Escalate to Hugging Face staff**

   - If the Space remains stuck with **no logs** and other Spaces build fine, collect:
     - Space URL
     - Last known working commit
     - A note that the Space builds locally or in a fresh test Space
   - Open a topic in the forums or a GitHub issue and reference the existing stuck-building threads.

### 3.3 Git / repository limits and “error while cloning repository”

Although covered in more detail in a separate note about the **1 GB Space repo limit**, this class of build error is also directly tied to **server-side enforcement of new repository constraints**:

- Builds fail with messages like:

  > `Build error: Error while cloning repository`  
  > `Repository storage limit reached (Max: 1 GB)`

- The root cause is often **LFS-heavy history** exceeding the 1 GB per-Space repo cap, which is enforced on the server side.

Key points:

- Fixes require **cleaning or rewriting Git history** and moving large artifacts to model/dataset repos, not changes to the HF build system.
- Storage limits and behavior are documented in:  
  [Hub storage limits](https://huggingface.co/docs/hub/storage-limits) and multiple forum threads.

(See the separate KB on the 1 GB Space repo limit for detailed cleanup steps.)

## 4. Library stack changes on platform images

Several widely reported error patterns are best explained by **Hugging Face updating the default library stack** in Spaces images. The update itself may be safe, but **old code or old expectations** then break.

John6666’s April 2025 post collects several such breakages and practical version pins:  
[If your Space stops working after restarting mainly for the last 5 days](https://huggingface.co/posts/John6666/369491746519704).

### 4.1 Pydantic, Gradio, and “Error: No API found” / 500 Internal Server Error

**Symptoms**

- On Gradio-based Spaces, after a server update or restart, the app returns:

  - `Error: No API found.` on the front-end, or
  - 500 “Internal Server Error” with stack traces inside Gradio / `gradio_client` (sometimes `TypeError: argument of type 'bool' is not iterable`).

Examples:

- Gradio’s own docs show “Error: No API found” when the Gradio app metadata cannot be parsed.  
  See: [Gradio map component example (“Error: No API found”)](https://www.gradio.app/4.44.1/guides/plot-component-for-maps)
- Forum threads note that Gradio version **specified in `README.md`** can override `requirements.txt` on Spaces, leading to subtle mismatches.  
  Example: [Agent Course – First Agent Template](https://discuss.huggingface.co/t/agent-course-first-agent-template/148170/4).

**Root causes**

- A combination of:
  - Gradio changes in how it exposes API schemas.
  - Changes in `pydantic` / `pydantic-core` and their integration with Gradio and `gradio_client`.
  - Hugging Face Spaces images shipping a newer `pydantic` / Gradio stack than the one your code expected.
- When Gradio parses the app’s IO schema and hits unexpected types or malformed definitions, it may emit “No API found” or internal 500 errors.

**Mitigations**

1. **Pin a tested Gradio + Pydantic combo**

   - In `requirements.txt`, specify explicit versions, e.g.:

     ```txt
     # requirements.txt
     gradio==5.23.3           # see Gradio changelog / forums
     pydantic==2.10.6         # version known to work with that Gradio
     pydantic-core==2.27.0    # if needed, keep consistent
     ```

     Add comments referencing the issues you’re tracking.  
     For example:
     ```txt
     # See: https://github.com/gradio-app/gradio/issues/10662
     ```

   - Keep those versions up to date with Gradio’s own fixes as they stabilize.

2. **Let README.md and Space metadata agree with `requirements.txt`**

   - On Spaces, **the Gradio version in `README.md` can override `requirements.txt`** for Gradio apps.
   - Make sure the badges or metadata in README are consistent with the versions you actually want.
   - If the README indicates `sdk: gradio` with an old `gradio` version tag, update it or remove conflicting hints.

3. **Disable server-side rendering (SSR) if necessary**

   - Some 500 errors on Spaces have been tied to Gradio’s SSR mode.
   - Community fixes sometimes involve launching Gradio with `ssr_mode=False`:

     ```python
     # app.py
     import gradio as gr

     with gr.Blocks() as demo:
         ...  # define your UI

     if __name__ == "__main__":
         demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
         # ssr_mode=False: workaround for certain Spaces/ZeroGPU SSR bugs
     ```

   - Consult the latest Gradio and HF forum threads before relying on SSR-related flags; long-term fixes may change behavior again.

4. **Check for mismatched dependencies**

   - Use `pip freeze` locally or in Dev Mode to confirm that `gradio`, `gradio_client`, `pydantic`, `fastapi`, and `starlette` are on a known good combination.
   - When an HF image silently updates one of these, it may invalidate an older pinned set.

### 4.2 Transformers, Diffusers, and ZeroGPU / CUDA regressions

When Hugging Face refreshes its GPU stack (drivers, CUDA, Torch), Spaces using **Transformers** or **Diffusers** can regress:

- After certain updates, some Spaces reported:

  - New CUDA errors or GPU tasks being aborted.
  - NVML assertion failures in `c10/cuda/CUDACachingAllocator.cpp` when using specific LoRA or multi-GPU patterns.
  - Crashes only on ZeroGPU (H200) but not on CPU Spaces.

In John6666’s summary and ZeroGPU Explorers discussions, common mitigation patterns include:

- **Pinning Transformers / Diffusers to a known-good version** for your app, e.g. `transformers<=4.49.0` for certain SD/Flux pipelines around 2024–2025, while waiting for upstream fixes.
- Matching **Torch + CUDA + Transformers** versions that are validated together (consult model cards, HF docs, and GitHub issues).
- Reducing batch sizes, disabling multi-GPU features, or avoiding advanced memory optimizations that rely on specific allocator behavior.

Because these details are **time-sensitive**, treat any specific version pins as **historical snapshots**. Before adopting them, check:

- Model cards for your chosen models.
- Transformers / Diffusers GitHub issues around your error message.
- Recent HF forum posts mentioning the same error string.

### 4.3 `huggingface_hub` and deprecations (e.g. `cached_download`)

**Symptoms**

- Code using older `huggingface_hub` APIs such as `cached_download` starts failing after a Spaces image update:

  > `AttributeError: module 'huggingface_hub' has no attribute 'cached_download'`

- This often appears after restarting Spaces, even if the code has not changed.

**Root cause**

- `huggingface_hub` removed or deprecated certain functions (`cached_download`), replacing them with newer APIs like `hf_hub_download` and `snapshot_download`.
- When Hugging Face updates the base image to a newer `huggingface_hub`, your Space now runs with an incompatible version relative to your code.

**Mitigations**

1. **Modernize your Hub calls**

   ```python
   # app.py
   # Old:
   # from huggingface_hub import cached_download

   # New (see: https://huggingface.co/docs/huggingface_hub)
   from huggingface_hub import hf_hub_download

   model_path = hf_hub_download(
       repo_id="bert-base-uncased",
       filename="config.json",
       # local_dir="/data/models/bert-base-uncased",  # optional: keep on /data in Spaces
       # local_dir_use_symlinks=False,
   )
   ```

2. **Optionally pin `huggingface_hub` if you cannot migrate immediately**

   - As a **short-term** workaround, pin an older `huggingface_hub` version that still has `cached_download` (e.g. `0.25.x`), and explicitly comment why:

     ```txt
     huggingface_hub==0.25.2  # legacy API cached_download used; TODO: migrate to hf_hub_download
     ```

   - Treat this as temporary; long-term, update your code to the current Hub APIs.

## 5. App-level patterns vs server-level issues

When you encounter a new error after a Spaces restart or platform change, it’s important to decide whether it’s **purely a server bug** or a **server-induced app bug**.

### 5.1 Pure server incidents

Examples:

- All Spaces (or many unrelated Spaces) suddenly return **500 Internal Server Error** for a period of time.
- ZeroGPU Explorers and the forums show **multiple users** reporting the same 500s or build stuck states, with no shared code.
- Errors disappear after Hugging Face staff acknowledges and fixes an infrastructure issue.

In such cases:

- There is often **nothing you can fix in your repo**.
- The best you can do is:
  - Monitor [Hugging Face status / forums](https://discuss.huggingface.co/) for incident reports.
  - Collect logs and repro steps and share them in the relevant threads.
  - Use alternative infrastructure (e.g. a CPU Space, Inference Endpoints, or local Docker) for critical demos until the incident is resolved.

### 5.2 Server-induced app bugs

This is the more common middle ground, where **new platform constraints reveal a latent app assumption**. Examples:

- ZeroGPU now forbids initializing CUDA in the main process → old code that did `model.to("cuda")` at import time must be refactored.
- New Gradio / Pydantic stack changes how API schemas are generated → old patterns for specifying inputs/outputs or `Examples` need cleanup.
- Base image updates remove certain OS packages → Dockerfiles that assumed the old OS need updating.

In practice:

- The error appears “after a server update”, but the **long-term fix is in your app**.
- Once you adapt to the new spec, the app becomes more robust and future-proof.

### 5.3 Quick triage checklist

When a Space breaks after a restart or platform change:

1. **Check whether other Spaces are also broken**  
   - If many unrelated Spaces are down: suspect an HF incident.
2. **Check HF forums and ZeroGPU Explorers for the exact error string**  
   - Search `site:huggingface.co` and `site:discuss.huggingface.co` with the error text.
3. **Compare library versions**  
   - In Dev Mode or locally, run `pip freeze` and compare to a previously working snapshot.
4. **Look for ZeroGPU-specific hints**  
   - Error mentions `spaces.zero` or “Stateless GPU environment” → likely a ZeroGPU spec change.
5. **Identify whether the error happens before your app loads**  
   - “Error: No API found” or 500 on root without logs may indicate Gradio / schema / build issues, not your main business logic.

## 6. Implementation patterns and hardening tips

### 6.1 Version pinning and stack snapshots

- Treat your Spaces as **deployments** with a tested stack:
  - Pin versions of **Gradio, Pydantic, Transformers, Diffusers, Torch, `huggingface_hub`** in `requirements.txt`.
  - Keep a small `STACK.md` or comment block listing the versions you tested.
- When a server-side update breaks your Space:
  - Try **reproducing locally** with the same versions, then gradually upgrade one library at a time to find the minimal change that restores compatibility.
  - Once you have a working set, **update the pins** and redeploy.

### 6.2 Separate CPU and ZeroGPU code paths

- For ZeroGPU Spaces:
  - Design your app so that **all GPU work is triggered through `@spaces.GPU` or equivalent mechanisms**.
  - Keep model definitions and heavy CUDA operations inside those entrypoints, or in functions they call.
- For CPU Spaces:
  - Avoid assumptions that will break if the Space is later switched to ZeroGPU (no global `.cuda()`, no unconditional `torch.cuda.is_available()` logic).

### 6.3 Keep heavy data and caches off root

- Many “No space left on device” or “Error while cloning repository” issues are tied to **storage limits** and Space repo caps.
- Use environment variables like `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME`, and `TMPDIR` to redirect caches to `/data`.  
  See official docs for environment variables and Datasets cache behavior:
  - [Hugging Face Hub env vars](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
  - [Datasets cache docs](https://huggingface.co/docs/datasets/en/cache)
- Store models and large artifacts in **model/dataset repos** and download them into `/data` at runtime instead of shipping them in the Space repo.

### 6.4 Use Dev Mode and local Docker to absorb platform changes

- **Spaces Dev Mode** lets you SSH / VS Code into the Space container and inspect the real environment:  
  [Spaces Dev Mode docs](https://huggingface.co/docs/hub/spaces-dev-mode)
- Rebuild and run your Docker image locally when Hugging Face changes the base image:
  - This makes OS-level errors easier to debug.
  - Once the Dockerfile works locally, you can push with more confidence.

### 6.5 Document known platform assumptions

- In your repo (e.g. `README.md` or `docs/`), keep a short section titled “Hugging Face Spaces Assumptions”:
  - “Target: ZeroGPU H200, PyTorch ≥2.2, Gradio 5.x, Pydantic 2.10.x”
  - “No CUDA init in main process; all GPU work in `@spaces.GPU` functions”
  - “No `None` in Gradio `Examples` data”
- This helps future maintainers understand why certain patterns (like deferred `.to("cuda")` or specific version pins) exist.

## 7. Limitations, caveats, and open questions

- **Rapid evolution**: ZeroGPU, Dev Mode, and Spaces infra are actively evolving. Specific version pins and workarounds from 2024–2025 may become obsolete or even harmful later.
- **Partial visibility**: Most information about platform changes comes from **community posts, forum threads, and HF staff replies**, not always formal release notes.
- **Per-project tradeoffs**:
  - Some apps may prefer “fully pinned, stable stack” even if that lags behind the latest HF features.
  - Others may choose “stay on latest” and accept occasional breakage in exchange for new capabilities.
- **ZeroGPU semantics**: As stateless GPU concepts mature, new restrictions or features (multi-GPU, dev mode integration, per-org hosting) may change best practices for where and how you initialize models.
- **Build pipeline behavior**: Incidents of stuck builds and misreported status suggest that Spaces’ build scheduler still has edge cases; future changes may alter how often these occur.

For ongoing work, keep a habit of:

- Watching **ZeroGPU Explorers** and HF forums for known issues with the exact error strings you see.
- Periodically reviewing your **version pins** and comparing them with model cards and official docs.
- Treating **Spaces as part of a larger deployment story**, with CI, local testing, and explicit documentation of platform assumptions.

## 8. References and links

### 8.1 Official Hugging Face docs

- Spaces overview and types: <https://huggingface.co/docs/hub/spaces-overview>
- Using GPU Spaces: <https://huggingface.co/docs/hub/spaces-gpus>
- Spaces ZeroGPU: <https://huggingface.co/docs/hub/spaces-zerogpu>
- Advanced compute options (ZeroGPU details): <https://huggingface.co/docs/hub/advanced-compute-options>
- Spaces Dev Mode: <https://huggingface.co/docs/hub/spaces-dev-mode>
- Hub storage limits: <https://huggingface.co/docs/hub/storage-limits>
- Hub environment variables: <https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables>
- Datasets cache docs: <https://huggingface.co/docs/datasets/en/cache>

### 8.2 Key community posts and discussions

- HF post – **Spaces failing after server update (John6666)**:  
  <https://huggingface.co/posts/John6666/369491746519704>
- ZeroGPU Explorers – multiple errors after ZeroGPU renovation:  
  <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104>
- ZeroGPU Explorers – 500 errors on “outside” ZeroGPU Spaces:  
  <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/123>
- Forum – Streamlit Docker Space stuck in “Building”:  
  <https://discuss.huggingface.co/t/streamlit-docker-space-permanently-in-building-state/168910/3>
- Forum – Build error “Job failed with exit code 1” (Docker template vs new Debian base):  
  <https://discuss.huggingface.co/t/build-error-job-failed-with-exit-code-1-while-creating-a-space-at-hugging-face/166760>
- ZeroGPU error threads – `CUDA must not be initialized in the main process`:  
  <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/72>  
  <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104>
- Example Spaces showing ZeroGPU CUDA-init issues (Bark, vLLM demos, etc.):  
  <https://huggingface.co/spaces/devilent2/bark>  
  <https://github.com/vllm-project/vllm/issues/3510>

### 8.3 Supporting docs and curated links

- Using GPU Spaces (general GPU guidance): <https://huggingface.co/docs/hub/spaces-gpus>
- Inference Providers overview (alternative hosting when Spaces is unstable): <https://huggingface.co/docs/inference-providers/en/index>
- Additional curated links (courses, docs, Spaces-related resources) are tracked in the user’s own support list and can be consulted alongside this note.
