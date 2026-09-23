---
source: "hf-kb-mode (chat+files+web)"
topic: "Debugging ZeroGPU Spaces on Hugging Face"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-12-03T00:00:00Z"
---

# Debugging ZeroGPU Spaces on Hugging Face

This note is a focused knowledge base on **debugging ZeroGPU Spaces** on Hugging Face.  
It assumes you already know the basics of Spaces and Gradio, and concentrates on:

- What is *special* about ZeroGPU compared to normal GPU Spaces.
- How to classify and debug **ZeroGPU-specific failures**.
- Concrete **error recipes** for common messages (`CUDA must not be initialized…`, `illegal duration`, `ZeroGPU worker error`, etc.).
- **Implementation patterns** that make ZeroGPU apps more robust and easier to debug over time.

Where possible, it reconciles official documentation with real-world reports from Spaces, forums, GitHub issues, and blog posts.

---

## 1. Background and mental model

### 1.1 What ZeroGPU actually is

At a high level, a ZeroGPU Space is:

- A normal **Gradio Space** (Python app running in a managed container).
- Using a special **ZeroGPU hardware type** instead of a dedicated GPU.
- Backed by a pool of **NVIDIA H200 GPUs**, sliced and scheduled across many users and Spaces.

Key properties:

- **Dynamic GPU allocation**  
  - A GPU slice is attached only while a ZeroGPU job is running.
  - After the job finishes or times out, the slice is released back to the pool.
- **Stateless GPU environment**  
  - You cannot rely on a persistent CUDA context across requests.
  - CUDA must not be initialized in the main process; GPU work is allowed only inside managed contexts.
- **Timeboxed and quota-limited**  
  - Each job has a **maximum GPU duration** (controlled via `@spaces.GPU(duration=...)`).
  - Each user has **short-term buckets** and **daily time quotas** (larger for Pro).

Implication: debugging ZeroGPU always mixes **normal Space debugging** (build logs, runtime exceptions, HTTP codes) with **scheduler and quota behavior**.

### 1.2 How ZeroGPU changes debugging compared to “normal” GPU Spaces

On a dedicated GPU Space, many patterns are permissible but fragile. On ZeroGPU, some of them become hard errors:

- **No CUDA at import time**  
  - Any call that initializes CUDA at module import or in the main process can trigger  
    `RuntimeError: CUDA must not be initialized in the main process on Spaces with Stateless GPU environment.`  
  - Examples: `torch.cuda.is_available()`, `model.to("cuda")`, `torch.load(..., map_location="cuda")`, CUDA checks in `__init__` methods.
- **Explicit GPU entrypoints**  
  - You are expected to wrap GPU-using functions in `@spaces.GPU`.  
  - The runtime uses these decorators to schedule and timebox GPU work.
- **Extra failure modes from the scheduler**  
  - `ZeroGPU illegal duration. The requested GPU duration (...) is larger than the maximum allowed.`  
  - Errors that talk about remaining seconds or retrying “in -1 days”.
  - Quota-related messages (“unlogged user”, “no CUDA GPUs are available”, “retry later”).
- **Tighter coupling to versions**  
  - Torch must support **H200 (sm_90)** or you get architecture errors.
  - Gradio and the `spaces` library must be recent enough to integrate with ZeroGPU’s quotas and scheduler.

You still need the usual debugging skills (Python stack traces, dependency resolution, HTTP error handling), but ZeroGPU adds an extra dimension: **stateless GPU rules + quotas + scheduler behavior**.

### 1.3 Roles: Space maintainer vs Space user

The same error text can mean different things depending on your role:

- As a **Space user**:
  - You cannot change code or durations.
  - All you can do is pick which Spaces to use, avoid wasting quota on obviously broken apps, and file good bug reports for maintainers.
- As a **Space maintainer**:
  - You control **requirements**, **code structure**, **`@spaces.GPU` usage**, and **logging**.
  - This KB primarily targets you, because that is where actual fixes live.
  - However, it also includes user-facing interpretations so you can write better READMEs and discussions.

---

## 2. Signals from official docs and examples

This section distills the parts of official documentation and example Spaces that matter most for debugging.

### 2.1 ZeroGPU architecture and constraints

From the official **Spaces ZeroGPU** and **advanced compute options** docs:

- ZeroGPU is a **shared, serverless GPU layer** for Spaces, offering access to H200 GPUs via dynamic allocation.
- Only supported on **Gradio Spaces**, with Python ≥ 3.10.
- GPU work is orchestrated by the **ZeroGPU runtime**, which:
  - attaches a GPU slice to a job when needed,
  - timeboxes the job to the configured `duration`,
  - and then tears the slice down again.

The runtime explicitly enforces that:

- CUDA **must not be initialized in the main process**.
- GPU work must be done inside decorated functions (`@spaces.GPU` or closely related helpers).
- Misuse of CUDA (e.g., top-level `torch.cuda` calls) is turned into clear runtime errors, not silent failures.

### 2.2 Duration and quotas (high-level)

Combining docs and community threads:

- Each ZeroGPU call has a **requested duration** via `@spaces.GPU(duration=...)`.
- The scheduler checks this against:
  - A **per-call maximum** (documented as around 120 seconds; longer values sometimes partially work but are unreliable).
  - The **user’s remaining quota**, often described in seconds (e.g., a daily bucket of ~25 minutes for Pro).
- If `duration` is too large or exceeds remaining quota, you see errors like:
  - `ZeroGPU illegal duration. The requested GPU duration (300s) is larger than the maximum allowed.`
  - Messages combining requested duration and “remaining seconds”, sometimes with confusing “-1 day” phrasing.

In practice:

- Treat **120 seconds** as a conservative upper bound for `duration` unless you have strong evidence otherwise.
- Prefer **shorter durations (15–60s)** plus careful model design to keep jobs inside those budgets.
- Do not assume that “Pro” removes all limits; it mostly increases bucket sizes and priority.

### 2.3 Version and SDK guidance

As of late 2025:

- **Gradio**  
  - Spaces support multiple Gradio versions; you control the version via `sdk_version` in the Space README.
  - Older guidance said “ZeroGPU only supports Gradio 4.x”; newer examples show **recent Gradio 5.x** working fine and official docs no longer restrict versions.
  - Safe pattern: follow the versions used in **recent, actively maintained ZeroGPU example Spaces** and keep Gradio reasonably up to date.
- **`spaces` library**  
  - Your app should depend on the `spaces` library to use `@spaces.GPU` and ZeroGPU helpers.
  - Keep this pinned to a recent minor version, since ZeroGPU behavior evolves.
- **PyTorch**  
  - You must use a **build that supports H200 (sm_90)**.
  - Very old Torch versions (e.g., 2.0.x) frequently cause “no kernel image is available” or similar architecture errors.
  - Many public ZeroGPU Spaces pin Torch in the `2.4–2.8` range, often with matching `torchaudio` and optional `xformers` or other acceleration libraries.

When debugging, always check:

- The **Space README** (`sdk`, `sdk_version`, `python_version`, hardware).
- The exact versions in `requirements.txt` and any library that pulls in its own Torch dependency.

---

## 3. ZeroGPU-specific failure modes

This section catalogs common ZeroGPU errors and what they usually mean, independent of any single Space.

### 3.1 “CUDA must not be initialized in the main process on Spaces with Stateless GPU environment”

Symptoms:

- Appears as a Python `RuntimeError` in runtime logs.
- Stack trace leads into `torch.cuda.__init__` and `spaces.zero` or similar.

Core meaning:

- Some part of your code (or a dependency) initialized CUDA in the **main process**, outside a ZeroGPU-managed context.
- Typical culprits:
  - Global calls to `torch.cuda.is_available()` or `torch.cuda.set_device`.
  - Global `model.to("cuda")` or `model.cuda()` at import time.
  - `torch.load(..., map_location="cuda")` run at import.
  - Libraries that preload models and move them to GPU as a side effect of import.

### 3.2 “ZeroGPU illegal duration…” / “requested GPU duration (…) is larger than the maximum allowed”

Symptoms:

- Error appears immediately when calling a Space (often before any model logs appear).
- The message explicitly mentions:
  - Requested duration (e.g. `300s`),
  - Maximum allowed duration, or remaining seconds.

Core meaning:

- Your `@spaces.GPU(duration=...)` argument is **too large** for current platform limits or for the caller’s remaining quota.
- No user code runs; the scheduler rejects the job up front.
- This is **Space configuration**, not a bug in the caller or their Pro account.

### 3.3 “ZeroGPU worker error RuntimeError” and similar

Symptoms:

- Users see a generic “ZeroGPU worker error RuntimeError” popup.
- Pro minutes are consumed but no images/text are returned.
- Maintainer logs may show:
  - Internal ZeroGPU messages,
  - Torch errors,
  - Or nothing obvious if the failure is in infrastructure.

Core meaning:

- An error happened inside the GPU job worker.
- Sometimes this is truly **infrastructure-level**, e.g. partial outages or node-level issues.
- Often, it is caused by **application-level issues**:
  - Torch architecture mismatch.
  - Out-of-memory on GPU.
  - Model code raising exceptions after CUDA starts.

### 3.4 “ZeroGPU has not been initialized” / “Error while initializing ZeroGPU: Unknown”

Symptoms:

- Startup errors mentioning ZeroGPU initialization.
- Logs may show almost no user code before the error.

Core meaning:

- The ZeroGPU runtime failed to initialize for this Space.
- This can be:
  - Misconfiguration (bad `@spaces.GPU` usage, missing `spaces` library).
  - An infrastructure issue that sometimes resolves after restart or duplicate.
  - An interaction between Dev Mode and ZeroGPU lifecycle.

### 3.5 “No CUDA GPUs are available”

Symptoms:

- Appears when users try to run heavy Spaces.
- Retry sometimes succeeds after waiting.

Core meaning:

- ZeroGPU **could not allocate a GPU slice** at that moment.
- Possible reasons:
  - Fleet is saturated (too many jobs).
  - User’s short-term bucket is empty (too many long runs in a row).
  - Rarely, a misconfiguration on the Space or a transient infra glitch.

### 3.6 Architecture / Torch issues on H200

Symptoms:

- Errors like:
  - `no kernel image is available for execution on the device`.
  - “device-side assert” or low-level CUDA kernel errors at model load or first inference.

Core meaning:

- The installed Torch build was compiled for older architectures and does not support **H200 (sm_90)**.
- This often happens when:
  - A dependency (e.g. `xformers`) **pins an older Torch** as a transitive dependency.
  - Requirements specify Torch by accident or conflict.

---

## 4. ZeroGPU debugging workflow for maintainers

This is the practical, repeatable flow you can apply whenever a ZeroGPU Space misbehaves.

### 4.1 Step 0 – Confirm you are really debugging ZeroGPU

Before diving in, verify:

1. In the Space’s **“Settings → Hardware”**, the hardware type is **ZeroGPU**, not CPU or another GPU flavor.
2. The Space README’s YAML block has:
   - `sdk: gradio`,
   - a sensible `sdk_version` (recent and supported),
   - `python_version` explicitly set if you require a specific minor version.

If any of these are wrong, fix them first and redeploy.

### 4.2 Step 1 – Check versions and dependencies

In `requirements.txt` (or your Dockerfile):

1. **Pin Torch explicitly** to a version known to work on H200, for example:

   ```text
   torch>=2.4.0,<2.7.0
   torchaudio>=2.4.0,<2.7.0
   ```

2. Add or update:

   ```text
   spaces
   gradio>=5.49.1
   ```

3. Search for dependencies that might pull in older Torch:

   - Libraries like `xformers`, custom CUDA packages, or legacy model libraries.
   - If you absolutely need them, find versions compatible with Torch on H200.

4. Build locally in a fresh virtual environment:

   ```bash
   pip install -r requirements.txt
   python -c "import torch; print(torch.__version__)"
   ```

If you cannot install a clean, H200-compatible Torch stack locally, ZeroGPU is unlikely to work reliably.

### 4.3 Step 2 – Distinguish “general bug” vs “ZeroGPU-specific bug”

Toggle hardware to isolate the issue:

1. Temporarily change the Space hardware to **CPU** (or comment out `@spaces.GPU` and run everything on CPU).
2. Run a minimal repro (simple prompt, few steps).

Interpretation:

- **Fails on CPU and ZeroGPU**  
  → This is a **general code or dependency bug**; debug as a normal Space.
- **Works on CPU, fails only on ZeroGPU**  
  → This is a **ZeroGPU-specific** issue (stateless GPU rules, quotas, or H200-specific problems).

This prevents you from blaming ZeroGPU when the underlying bug is simply broken Python code.

### 4.4 Step 3 – Use logs correctly

Use the right logs for the right questions:

- **Build logs**  
  - If the Space never leaves `Building`, inspect build logs for:
    - Pip resolution errors,
    - Missing system packages,
    - Repo-size or Git LFS issues (1GB Space repo limit).
- **Runtime logs**  
  - When the Space is `Running` but actions fail:
    - Trigger the failing action once.
    - Open runtime logs and scroll to the last 50–100 lines.
    - Look for Python tracebacks, ZeroGPU messages, and Torch errors.
- **Error overlays and HTTP responses**  
  - For API calls, capture status code and response body.
  - For UI errors, copy the full error text and anything the Space prints.

If you do not see any Python traceback for your app code but only errors from `spaces.zero` or the scheduler, suspect **configuration or infrastructure** first.

### 4.5 Step 4 – Reproduce locally and in Dev Mode (if available)

1. **Local reproduction**

   ```bash
   git clone https://huggingface.co/spaces/<owner>/<space-name>
   cd <space-name>

   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scriptsctivate

   pip install -r requirements.txt
   python app.py  # or `streamlit run app.py`, etc.
   ```

   - If the error reproduces locally, you can debug with full IDE support.
   - If it does not, the difference is likely **ZeroGPU or container-specific**.

2. **Dev Mode (paid)**

   - Enable **Spaces Dev Mode**.
   - SSH or use browser-based VS Code into the Space.
   - Run small scripts:
     - `python -c "import torch; print(torch.cuda.device_count())"`
     - Minimal `@spaces.GPU` tests.
   - Check installed versions and environment variables.

### 4.6 Step 5 – Add a minimal `@spaces.GPU` smoke test

Create a very small GPU test function inside your app:

```python
import spaces
import torch

@spaces.GPU(duration=30)
def gpu_smoke_test() -> str:
    x = torch.ones((256, 256), device="cuda")
    return f"sum={x.sum().item()}"
```

Wire it to a simple Gradio button:

```python
import gradio as gr

with gr.Blocks() as demo:
    out = gr.Textbox(label="ZeroGPU test output")
    btn = gr.Button("Run ZeroGPU smoke test")

    btn.click(fn=gpu_smoke_test, inputs=None, outputs=out)

if __name__ == "__main__":
    demo.launch()
```

Use this to distinguish:

- **If the smoke test fails** with ZeroGPU-specific errors → the problem is with **ZeroGPU or the environment**, not your main model.
- **If the smoke test succeeds**, but your main pipeline fails → focus on your model loading and CUDA usage.

---

## 5. Error-specific debugging recipes

This section gives concrete “if you see X, try Y” recipes for the most common ZeroGPU error patterns.

### 5.1 Recipe: “CUDA must not be initialized in the main process…”

**Goal:** Remove all CUDA work from global scope and route it through `@spaces.GPU`.

#### 5.1.1 Typical “wrong” pattern

```python
# app.py (anti-pattern!)
import torch
from diffusers import StableDiffusionPipeline

# Global model, loaded directly onto GPU
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("cuda")

def generate(prompt: str):
    return pipe(prompt).images[0]
```

Why this breaks:

- The global `.to("cuda")` initializes CUDA in the main process.
- ZeroGPU intercepts this and raises the stateless-GPU error before any request is handled.

#### 5.1.2 Safer pattern: lazy GPU move inside `@spaces.GPU`

```python
# app.py (ZeroGPU-friendly)
import torch
from diffusers import StableDiffusionPipeline
import spaces

# Load the pipeline on CPU at import time (if memory allows)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe.to("cpu")

@spaces.GPU(duration=60)
def generate_image(prompt: str):
    global pipe

    # Move to GPU only inside the decorated function
    pipe = pipe.to("cuda")

    with torch.inference_mode():
        image = pipe(prompt).images[0]

    # Optional: move back to CPU to reduce GPU memory pressure between calls
    pipe = pipe.to("cpu")

    return image
```

Key ideas:

- Global initialization is CPU-only (or lazy).
- Every CUDA touch (`to("cuda")`) is inside the `@spaces.GPU` function.
- You can experiment with keeping the model on GPU for the duration of a single call, but not beyond.

#### 5.1.3 If third-party libraries trigger CUDA at import

Sometimes you cannot control the code that calls CUDA. Options:

- Look for environment variables or flags that disable CUDA preloading.
- Wrap imports in functions executed only inside `@spaces.GPU`.
- If necessary, fork or wrap the library so that it delays CUDA operations until after ZeroGPU attaches a GPU.

### 5.2 Recipe: “ZeroGPU illegal duration…” / duration too large

**Symptoms:**

- Error mentions requested duration and maximum allowed.
- No Python traceback from your code.

**Checklist:**

1. Search for `@spaces.GPU` in your repo:

   ```bash
   rg "@spaces.GPU" .
   ```

2. Inspect `duration` arguments:

   - Avoid values like 240–900 seconds.
   - Start with **60 seconds or less**.
   - If needed, move up to **120 seconds**, but treat anything beyond that as experimental.

3. Consider a dynamic duration function:

   ```python
   import spaces

   def estimate_duration(prompt: str, steps: int) -> float:
       # Rough heuristic: seconds per step
       return min(120.0, steps * 3.5)

   @spaces.GPU(duration=estimate_duration)
   def generate(prompt: str, steps: int):
       ...
   ```

4. After lowering durations, redeploy and test with **small, fast inputs** (short prompts, few steps).

If users still report illegal-duration errors despite conservative durations, gather:

- Space link,
- Exact error messages,
- Example inputs,

and raise the issue on the Spaces forum or ZeroGPU Explorers discussions as a potential scheduler bug.

### 5.3 Recipe: “ZeroGPU worker error RuntimeError”

**Symptoms:**

- Users’ Pro minutes decrease, but they only see a generic “worker error”.
- Errors sometimes appear in waves across multiple Spaces.

**Approach as a maintainer:**

1. **Check runtime logs** around the error time:
   - Look for Torch OOM, device asserts, or model-specific errors.
   - If you see obvious exceptions, fix them:
     - Reduce resolution, steps, or batch size.
     - Use smaller models.
     - Reduce VRAM usage (float16, CPU offload when possible).
2. **Test the smoke test** (`gpu_smoke_test`) in the same Space:
   - If even the smoke test fails with worker errors, suspect infrastructure or a severe H200/Torch mismatch.
3. **Check Torch and `spaces` versions**:
   - Ensure you are on a supported Torch version.
   - Upgrade `spaces` and Gradio to recent versions if you are several minor releases behind.
4. **If everything looks correct but errors persist:**
   - Restart the Space (“Factory reboot”).
   - If the problem affects multiple unrelated Spaces at once, it may be a **ZeroGPU outage or regression**; check official status and forums.
   - When posting a bug report:
     - Include a minimal Space link (e.g. a tiny repro built around `gpu_smoke_test`),
     - Logs,
     - Timestamps,
     - And your account type (free, Pro, etc.).

**Advice for users (what to write in Discussions):**

- Mention:
  - That you are Pro (if applicable),
  - The exact error string,
  - How many retries you tried,
  - And whether other ZeroGPU Spaces still work for you.

This gives maintainers enough context to distinguish quota issues from true worker bugs.

### 5.4 Recipe: “ZeroGPU has not been initialized” / “Error while initializing ZeroGPU: Unknown”

**Symptoms:**

- Space fails at startup with ZeroGPU initialization messages.
- Logs show almost no stack trace from your code.

**Checklist:**

1. Ensure you **import `spaces`** and actually use `@spaces.GPU` in your app.
2. Double-check `requirements.txt` includes `spaces` (latest version).
3. Look for unusual patterns:
   - Creating threads or processes that call CUDA before `@spaces.GPU` is entered.
   - Heavy initialization inside global code that might confuse the ZeroGPU runtime.
4. Restart the Space a couple of times.
5. If the error persists:
   - Create a minimal Space whose only purpose is to run `gpu_smoke_test`.
   - If *that* fails with the same message, it is likely an infrastructure issue.
   - File a report with the minimal Space link and logs.

### 5.5 Recipe: “No CUDA GPUs are available”

**Symptoms:**

- Users intermittently see “No CUDA GPUs are available” on otherwise healthy Spaces.
- Retrying after some time often works.

**Interpretation:**

- The ZeroGPU fleet or your user bucket is temporarily unable to allocate a GPU slice.

**Mitigations:**

- As a maintainer:
  - Keep durations conservative and throughput high (short jobs).
  - Avoid synchronous chains of multiple long GPU calls inside one request.
- As a user:
  - Avoid hammering the same Space with repeated retries; wait a few minutes.
  - Try other Spaces to see if ZeroGPU overall is overloaded or it is specific to one Space.

### 5.6 Recipe: Torch / H200 architecture errors

**Symptoms:**

- CUDA errors referencing kernel images or architecture mismatches at model load or first inference.

**Checklist:**

1. In Dev Mode or local env, run:

   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. If Torch is < 2.4 or clearly not built for H200, adjust `requirements.txt`:

   ```text
   torch>=2.4.0,<2.7.0
   torchaudio>=2.4.0,<2.7.0
   ```

3. Remove or upgrade libraries that pin older Torch versions.
4. Rebuild the Space and retest.
5. If errors persist, verify that no second, incompatible Torch build is being installed as a transitive dependency.

---

## 6. Implementation patterns and design tips

Designing your Space carefully reduces the chance of ZeroGPU-specific issues and makes debugging easier when they do occur.

### 6.1 Structure around a clear GPU boundary

Design your app with a narrow, well-defined GPU entrypoint:

- Global / module-level code:
  - Load configs.
  - Optionally load models **on CPU**.
  - Set up Gradio UI.
- `@spaces.GPU` function(s):
  - Move needed tensors or models to GPU.
  - Perform the actual inference.
  - Optionally move models back to CPU or free buffers.

Example layout:

```python
import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "some/7b-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cpu")

def format_prompt(history, user_input):
    # build chat prompt from history and user input
    ...

@spaces.GPU(duration=45)
def chat_fn(history, user_input):
    prompt = format_prompt(history, user_input)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return history + [(user_input, text)]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user_submit(user_message, history):
        return "", history + [[user_message, None]]

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        chat_fn, [chatbot, msg], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
```

Notes:

- Only `chat_fn` touches CUDA.
- The rest of the app runs on CPU and is compatible with CPU-only hardware.

### 6.2 Choose realistic durations and workloads

Guidelines:

- Start with **small models and short jobs**:
  - 7B models instead of 34B or 70B.
  - Image resolutions like 512×512 instead of 1024×1024+.
  - Fewer diffusion steps, limited max tokens for LLMs.
- Set **duration** slightly above your measured worst-case runtime:
  - If most jobs finish in 10–20 seconds, set `duration=30` or `45`, not `120`.
  - This helps the scheduler pack jobs efficiently and avoids unnecessary rejections.
- Add **sensible defaults** in the UI:
  - Limit max steps / resolution.
  - Warn users when they set extremely heavy parameters.

### 6.3 Logging and observability

Add lightweight logging so you can correlate user reports with runtime behavior:

- Log the start and end of each `@spaces.GPU` call:

  ```python
  import time
  import logging

  logger = logging.getLogger("zerogpu_app")

  @spaces.GPU(duration=60)
  def generate_image(prompt: str, steps: int = 20):
      start = time.time()
      logger.info("generate_image start prompt_len=%d steps=%d", len(prompt), steps)
      try:
          ...
      finally:
          logger.info("generate_image end elapsed=%.2f", time.time() - start)
  ```

- Log key configuration at startup:
  - Torch version,
  - `spaces` version,
  - Gradio version,
  - Hardware (from environment variables, if available).

This makes it easier to compare behavior before and after upgrades.

### 6.4 Keep a dedicated ZeroGPU smoke-test Space

Maintain a small, public or internal Space that:

- Uses ZeroGPU hardware.
- Runs only the `gpu_smoke_test` function and a simple small model or tensor operation.
- Logs version info clearly.

Whenever you suspect a platform regression:

- Test this smoke Space first.
- If it breaks with a clearly minimal repro, it is strong evidence of an **infrastructure or server-side change**, not app-specific debt.

### 6.5 Plan for server-side spec changes

ZeroGPU and Spaces evolve regularly:

- Base images and Python versions can change.
- ZeroGPU runtime and quotas may be adjusted.
- Library stacks (Gradio, Torch, `spaces`) are periodically updated.

To reduce surprise breakage:

- Pin your core dependencies (`torch`, `gradio`, `spaces`, main ML libs).
- Avoid over-constraining minor versions; allow patch updates, but not major jumps.
- Periodically:
  - Upgrade locally,
  - Run your test suite (or at least core flows),
  - Deploy to a staging Space before touching production Spaces.

---

## 7. Limitations, caveats, and open questions

Debugging ZeroGPU is constrained by several realities:

1. **Opaque scheduler and quotas**  
   - Exact quota refill rates and per-call limits can change and are not always fully documented.
   - Error messages (e.g., “retry in -1 day”) can be misleading or buggy.

2. **Infrastructure vs app bugs are sometimes hard to distinguish**  
   - The same “worker error” can mean:
     - A temporary glitch in the fleet,
     - A memory leak in your code,
     - Or a Torch kernel issue.
   - This is why **smoke tests, local reproduction, and good logs** are essential.

3. **Some frameworks are not ZeroGPU-friendly yet**  
   - Libraries that assume persistent GPUs or that heavily customize CUDA behavior (e.g. some custom runtime frameworks) may need adaptation.
   - When debugging such stacks, you sometimes have to follow both the library’s GitHub issues and ZeroGPU discussions.

4. **Production workloads may need dedicated GPUs**  
   - ZeroGPU is optimized for **demos and light workloads**, not steady, high-throughput production.
   - For serious production use, you should plan to:
     - Move heavy workloads to **Inference Endpoints** or **dedicated GPU Spaces**,
     - Use ZeroGPU primarily as a try-out / demo layer.

5. **Documentation and examples lag behind changes**  
   - Always cross-check multiple sources:
     - Official docs,
     - Recent ZeroGPU example Spaces,
     - Discussions where staff and power users respond.

Keeping these caveats in mind helps avoid over-optimizing for behavior that may be temporary or incidental.

---

## 8. References and useful links

This section lists key URLs grouped by type. Always check the publication dates and discussion timestamps when applying any advice.

### 8.1 Official Hugging Face docs

- Spaces ZeroGPU: Dynamic GPU Allocation  
  - <https://huggingface.co/docs/hub/spaces-zerogpu>
- Advanced compute options (ZeroGPU, GPU upgrades, etc.)  
  - <https://huggingface.co/docs/hub/advanced-compute-options>
- Gradio Spaces overview  
  - <https://huggingface.co/docs/hub/spaces-sdks-gradio>
- Spaces configuration reference (README YAML fields, `sdk_version`, `python_version`, etc.)  
  - <https://huggingface.co/docs/hub/spaces-config-reference>
- Spaces Dev Mode  
  - <https://huggingface.co/docs/hub/spaces-dev-mode>
- Using GPU Spaces  
  - <https://huggingface.co/docs/hub/spaces-gpus>

### 8.2 ZeroGPU-focused blog posts and examples

- ZeroGPU NaN / pickle error debugging article  
  - <https://huggingface.co/blog/rrg92/zero-gpu-nan-and-pickle-errors>
- Tutorial: Run ComfyUI workflows for free with Gradio on Spaces ZeroGPU  
  - <https://huggingface.co/blog/run-comfyui-workflows-on-spaces>
- Example ZeroGPU migration notes (FlashWorld-ZeroGPU)  
  - <https://huggingface.co/spaces/jbilcke-hf/FlashWorld-ZeroGPU>

### 8.3 Community discussions: CUDA-init and stateless GPU

- Discussion: “CUDA must not be initialized in the main process on Spaces with Stateless GPU environment”  
  - <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/45>
- ZeroGPU Explorers threads on stateless GPU and platform renovations  
  - <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/72>  
  - <https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/104>
- GitHub issue: vLLM does not work with ZeroGPU (illustrates CUDA-init issues)  
  - <https://github.com/vllm-project/vllm/issues/3510>

### 8.4 Community discussions: illegal duration and quotas

- “ZeroGPU illegal duration. The requested GPU duration (300s) is larger than the maximum allowed”  
  - <https://discuss.huggingface.co/t/zerogpu-illegal-duration-the-requested-gpu-duration-300s-is-larger-than-the-maximum-allowed/140849>
- “The requested GPU duration (240s) is larger than the maximum allowed…”  
  - <https://discuss.huggingface.co/t/the-requested-gpu-duration-240s-is-larger-than-the-maximum-allowed-retry-in-1-day-2359/106988>
- Qwen image Spaces “too long” / illegal duration thread  
  - <https://discuss.huggingface.co/t/qwen-image-free-space-s-not-useable-because-too-long/170454>
- Quota and Pro-limit discussions (ZeroGPU minutes, unlogged user issues)  
  - <https://discuss.huggingface.co/t/issue-with-zerogpu-quota-on-hugging-face-pro-account/154576>  
  - <https://discuss.huggingface.co/t/how-to-get-more-zerogpu-minutes-as-a-pro-user/163662>

### 8.5 Community discussions: worker errors and initialization

- “Zero GPU Worker Error” (generic worker failures and quota burn)  
  - <https://discuss.huggingface.co/t/zero-gpu-worker-error/166246>
- “Error while initializing ZeroGPU”  
  - <https://discuss.huggingface.co/t/error-while-initializing-zerogpu/158578>
- “ZeroGPU has not been initialized Error after disabling dev mode”  
  - <https://discuss.huggingface.co/t/zerogpu-has-not-been-initialized-error-after-disabling-dev-mode/91994>
- “Does ZeroGPU not work for all spaces?” (No CUDA GPUs are available)  
  - <https://discuss.huggingface.co/t/does-zerogpu-not-work-for-all-spaces/113888>

Using these links together with the structured debugging flows above should give you a robust, reusable way to debug and maintain ZeroGPU Spaces over time.
