---
source: "chat+files+web"
topic: "Debugging Hugging Face Spaces (ZeroGPU, GPU, CPU)"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-15T00:00:00Z"
---

# Debugging Hugging Face Spaces (ZeroGPU / GPU / CPU)

## 1. Background and overview

Hugging Face Spaces run user code inside managed containers. A Space is essentially:

- A Git repository hosted on the Hugging Face Hub.
- Deployed into a container with fixed CPU, RAM, and (optionally) GPU resources.
- Exposed as a web app (Gradio / Streamlit / static / Docker) and often as an HTTP API.

When a Space is "broken", almost every problem fits into one of four buckets:

1. **Build-time failures**  
   - Status shows `Building` or `Build error`.  
   - The UI never appears.  

2. **Runtime failures (app starts but actions break)**  
   - Status shows `Running`.  
   - The UI loads, but buttons or requests produce errors.  

3. **Inference / HTTP errors when calling the Space as an API**  
   - 4xx or 5xx responses when invoked from Python/JS/curl/routers.  

4. **Platform / outage issues**  
   - You did not change code, but several Spaces regress around the same time.

A repeatable approach is:

1. Classify the failure type.  
2. Look at the right logs (build vs runtime vs HTTP client).  
3. Create a minimal reproducible example.  
4. Reproduce locally or in Dev Mode.  
5. Iterate with small code/config changes.  
6. Escalate only after you have logs and a minimal repro.


---

## 2. Classifying failures: build, runtime, HTTP, platform

### 2.1 Build-time failure (`Build error` / stuck on `Building`)

**Symptoms**

- Status badge shows `Building` or `Build error`.
- App UI never loads.

**Typical root causes**

- Invalid / conflicting entries in `requirements.txt` (pip cannot resolve versions).
- Missing entry file (`app.py`, `main.py`, or framework-specific app file).
- Dockerfile issues for Docker Spaces.
- Repository over the **1 GB Space repo limit**, causing clone/build to fail.
- Transient platform issues where no build logs appear at all.

### 2.2 Runtime failure (Space runs, but actions fail)

**Symptoms**

- Status is `Running`, UI loads.
- Clicking a button or calling an endpoint produces a red error or HTTP 500.

**Typical root causes**

- Python exceptions in app code or model loading.
- `ModuleNotFoundError` at import time for libraries not in `requirements.txt`.
- CUDA / memory errors, especially on GPU or ZeroGPU Spaces.
- Logic errors (wrong shapes, missing keys, invalid file paths).

### 2.3 HTTP / API failures

**Symptoms**

- Calling the Space from a client produces 4xx/5xx HTTP errors.

**Interpretation**

- **4xx (400/401/403/404/422/429)**: usually client-side (wrong URL, payload, or auth).  
- **5xx (500/502/503/504)**: usually server-side (uncaught exception in the Space, backend overload).

### 2.4 Platform / outage suspicion

Indicators:

- Multiple Spaces regress without any repo changes.
- Logs are empty or clearly inconsistent with your code.
- Recent forum threads show the same error message from other users.

In this case:

- Check the official Spaces docs and status page.
- Try a minimal hello-world Space on the same hardware.
- Use “Factory reboot” to force a clean rebuild if advised by HF staff.


---

## 3. Observability: status, logs, local runs, Dev Mode

### 3.1 Space status and lifecycle

Spaces go through several stages:

- `BUILDING`: installing dependencies and building the container image.
- `RUNNING`: app is live.
- `PAUSED` / `STOPPED`: runtime is inactive.
- `FAILED`: build or startup crashed.

Use the badge near the like count on the Space page to see status and open logs.

The Spaces overview docs describe Space types, GPU options, persistent storage, and configuration reference in detail. Use them when deciding whether a problem is caused by your code or by configuration/hardware choices.

### 3.2 Build logs

Use build logs when the UI **never** appears.

How to access:

1. Visit `https://huggingface.co/spaces/<owner>/<space>`.
2. Click the status badge or “Open logs”.
3. Inspect the build log stream.

Common patterns:

- Pip error: `ERROR: Could not find a version that satisfies the requirement ...`  
  → fix or relax version constraints in `requirements.txt`.

- Import errors during build: `ModuleNotFoundError` when import happens in build-time hooks.  
  → ensure dependencies are installed before imports.

- Git LFS errors like `Repository storage limit reached (Max: 1 GB)`  
  → move large artifacts (models, datasets) into separate model/dataset repos.

- Missing app file: errors about “No application file”  
  → ensure the correct file is present and the Space type matches the SDK (Gradio, Streamlit, etc.).

### 3.3 Runtime logs

Use runtime logs when the app loads but specific actions fail.

Steps:

1. Confirm the Space is `Running`.
2. Trigger the failing action once (click the button, call the endpoint).
3. Re-open logs and scroll to the last 20–50 lines.

You should see:

- A Python traceback for the latest error.
- `print()` / `logging` output from your code.
- Warnings about GPU, memory, or network.

You can add simple debug prints:

```python
def generate(prompt: str):
    print("DEBUG: generate() called with prompt:", repr(prompt))
    # ...
```

For Gradio, running locally with `debug=True` prints stack traces both in the browser and in the console:

```python
import gradio as gr

def echo(text: str) -> str:
    return f"You said: {text}"

demo = gr.Interface(fn=echo, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(debug=True)
    # Docs: https://www.gradio.app/guides/using-blocks
```

### 3.4 Logs for Inference Endpoints / Providers

If a Space calls an Inference Endpoint or you use Inference Providers, remember there is a **separate Logs page** for the endpoint itself. Combine:

- Space logs,
- Inference Endpoint logs,
- Client HTTP logs (status code, response body)

to find which layer is failing.

### 3.5 Local reproduction

Cloning and running the Space locally is one of the fastest debugging tools:

```bash
git clone https://huggingface.co/spaces/<user>/<space-name>
cd <space-name>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py  # or `streamlit run app.py`, etc.
# Reference: https://huggingface.co/docs/hub/spaces-overview
```

If the error reproduces locally, you can use local tools (debuggers, IDEs) to isolate the failure. If it does not, the root cause is probably environment-specific (ZeroGPU, memory limits, OS differences).

### 3.6 Dev Mode (paid plans)

Spaces Dev Mode lets you:

- SSH or use a browser-based VS Code into the Space container.
- Edit `app.py` and other files live.
- Run shell commands (`pip`, Python REPL, debugging tools).
- Push commits back to the repo once you are done.

Use Dev Mode to shorten the debug loop when it is available. It is particularly useful for complex Docker Spaces or GPU/ZeroGPU apps.


---

## 4. A reusable debugging loop

Use the same loop for almost all problems:

1. **Freeze a minimal reproducible example**
   - Decide on 1–2 specific inputs that always trigger the bug.
   - Keep these fixed while you debug so logs and experiments are comparable.

2. **Collect full logs and tracebacks**
   - Build-time: copy relevant build logs from first error onward.
   - Runtime: copy the full traceback and the last 20–50 lines of logs after one failing run.
   - HTTP: record status code, URL, headers (without secrets), request payload, and response body.

3. **Reproduce locally**
   - Clone the Space, install dependencies, and run the app.
   - Try your minimal inputs locally.
   - If local and remote behavior differ, note the difference; environment-related issues often show up here.

4. **Reduce to a minimal script**
   - Extract the failing part (e.g., model load + one inference) into a tiny script that does not depend on the UI.
   - This script is easier to understand and share on forums or in issues.

5. **Use an LLM as a log assistant**
   - Provide: minimal code, full traceback, what you are trying to do, and that you are running on a Hugging Face Space.
   - Ask for: 2–3 plausible root causes and 1–2 small experiments for each cause.

6. **Escalate with a good bug report**
   - When posting on the Hugging Face forum or opening an issue, include:
     - Space URL,
     - Logs,
     - Minimal repro (inputs + simplified script),
     - What you already tried and what changed recently (if anything).

Over time, this loop becomes “muscle memory” for debugging Spaces.


---

## 5. Common patterns and recipes (2024–2025)

### 5.1 Dependency and import problems

**Symptoms**

- Build fails quickly with `ModuleNotFoundError` or pip resolution errors.
- Runtime logs show imports failing at startup.

**Checklist**

- Verify every imported library is listed in `requirements.txt` or installed via Dockerfile.
- Test in a clean virtualenv:

  ```bash
  pip install -r requirements.txt
  python -c "import <your_module>"
  ```

- Avoid overly strict or obviously impossible version pins.
- For older code, check that APIs used by LLM-generated snippets still exist in the versions you install.

### 5.2 Memory and storage issues

**RAM / VRAM**

- Errors like `Memory limit exceeded (16Gi)` or CUDA out-of-memory.
- Crashes only for larger inputs or high batch sizes.

Mitigations:

- Use smaller models (e.g., 7B instead of 70B, SD 1.5 instead of SDXL).
- Reduce batch size and sequence length.
- Use half-precision (float16) on GPU when supported.
- Avoid loading multiple large models unnecessarily.

**Repository size / 1 GB limit**

- Git LFS errors and messages about repository storage limit reached.
- Space stuck on `Building` due to clone issues.

Mitigations:

- Move large weights or datasets into separate **model** or **dataset** repos.
- Use `hf_hub_download` or `from_pretrained` to fetch them at runtime.
- If history already contains big blobs, it may be easier to start a fresh Space with a clean repo.

### 5.3 ZeroGPU-specific pitfalls

ZeroGPU dynamically attaches NVIDIA H200 GPU slices only while a GPU-decorated function runs. Key rules:

- Do **not** call `model.to("cuda")` or any CUDA initialization at import time.
- Confine GPU-specific operations inside a function decorated with `@spaces.GPU()` (for Gradio) or equivalent supported patterns.
- As of 2025, ZeroGPU Spaces are officially supported with Gradio; other frameworks may not be compatible.

Safe pattern for a diffusion pipeline:

```python
import torch
import spaces
from diffusers import StableDiffusionPipeline

pipe = None

def _load_pipe_on_cpu():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
    return pipe

@spaces.GPU()
def generate(prompt: str):
    pipe_cpu = _load_pipe_on_cpu()
    if torch.cuda.is_available():
        pipe_gpu = pipe_cpu.to("cuda")
    else:
        pipe_gpu = pipe_cpu
    image = pipe_gpu(prompt, num_inference_steps=20).images[0]
    return image
# Docs: https://huggingface.co/docs/hub/spaces-zerogpu
```

If you port existing GPU code to ZeroGPU and immediately see CUDA / pickling / device errors, inspect whether any GPU work happens at import time or outside a GPU-decorated context.

### 5.4 HTTP / API usage for Gradio Spaces

#### 5.4.1 Correct host and URL

- Repo page (not API host): `https://huggingface.co/spaces/<owner>/<space>`  
- App/API host: `https://<owner>-<space>.hf.space`

Always point clients to the `.hf.space` host.

#### 5.4.2 View API page and `api_name`

Gradio Spaces expose a “Use via API” or “View API” page that documents:

- Endpoints (events and `gr.api` handlers).
- Parameter names, types, defaults.
- Example snippets for:
  - `gradio_client` (Python),
  - `@gradio/client` (JS),
  - curl / HTTP.

Use this page as the source of truth for:

- Which `api_name` to call (e.g., `"/generate_image"`).
- Parameter order and names.

#### 5.4.3 Python client example

```python
# Docs: https://www.gradio.app/docs/python-client/introduction
from gradio_client import Client

client = Client("https://<owner>-<space>.hf.space")
result = client.predict(
    "A cat playing guitar",  # prompt
    api_name="/generate_image",  # must match the View API page
)
print(result)
```

Common mistakes:

- Using the Hub URL instead of `.hf.space`.
- Passing the wrong `api_name`.
- Passing arguments in the wrong order for the server-side signature.

#### 5.4.4 Raw HTTP and SSE pattern

For streaming endpoints in Gradio 4–5, HTTP access typically uses:

1. `POST` to `/gradio_api/call/<api_name>` with JSON body `{"data":[...args...]}`.
2. `GET` (often SSE) on `/gradio_api/call/<api_name>/<event_id>` where `event_id` comes from the POST response.

Use the curl example from the Space’s API page as a reference. Small mistakes in the path or payload shape often cause 400/405/422 errors.

#### 5.4.5 `gr.api` and function signatures

`gr.api(fn=..., api_name=...)` exposes a function as an API endpoint. Gradio:

- Derives schema from the function’s signature and type hints.
- Calls the function as `fn(*inputs)` at runtime.

Handlers defined as `def fn(**kwargs):` cannot accept positional arguments and will fail with:

```text
TypeError: fn() takes 0 positional arguments but N were given
```

Safer pattern:

- Either declare explicit parameters (`def fn(prompt: str, seed: int, ...)`), or
- Use a wrapper that accepts `*args, **kwargs`, binds them with `inspect.signature`, and forwards to an internal pipeline.

This prevents Spaces that “work in the UI” from breaking when used via API clients or routers.

### 5.5 Guard messages like "Model is not initialized"

If your code raises custom errors like `gr.Error("Model is not initialized")`, remember:

- That message is a symptom, not the root cause.
- The real error likely occurred earlier during model loading and was swallowed by a broad `try/except`.

Temporarily remove or narrow such `try/except` blocks and log the full exception so you can fix the real problem (CUDA, model ID, permissions, etc.).


---

## 6. LLM-generated code and how to debug it

Code produced by ChatGPT, Gemini, Claude, and similar models tends to:

- Target an arbitrary library version that may not match what you install.
- Mix multiple APIs (old and new) from the same library.
- Add broad `try/except Exception:` blocks that hide root causes.
- Choose extremely large models or batch sizes for “impressive” demos.

To make LLM-generated Spaces debuggable:

1. **Normalize dependencies**
   - Choose specific versions of `gradio`, `transformers`, `diffusers`, etc.
   - Check official docs for those versions and confirm functions/arguments are valid.

2. **Tighten exception handling**
   - Replace `except Exception: pass` with logging and explicit, informative errors.
   - Print the full exception in model-loading paths.

3. **Scale down resources first**
   - Start with smaller models and batch sizes.
   - Only scale up once things work reliably on Spaces’ limits.

4. **Use LLMs as collaborators, not authors**
   - Ask for explanations of tracebacks and targeted patches, not full-file rewrites.
   - Keep architectural control in your own hands.

---

## 7. Minimum background knowledge that helps a lot

You do not need to be a deep expert to debug Spaces, but the following basics make everything easier:

- Reading Python tracebacks (knowing how to find the line in your code where it failed).
- Understanding the difference between build-time vs runtime errors.
- Familiarity with virtual environments (`python -m venv`, `pip install -r requirements.txt`).
- Awareness of Space types (Gradio, Streamlit, Docker, static) and the 1 GB repo limit.
- Basic understanding of GPUs vs CPUs, especially for ZeroGPU:
  - GPU-annotated functions,
  - Why you should not call CUDA at import time,
  - Memory constraints and how model size affects them.

Once you have this baseline plus the debugging loop in section 4, most Spaces problems become manageable.

---

## 8. Limitations, caveats, and future changes

- This guide focuses on 2024–2025 behavior for Spaces, ZeroGPU, Gradio 4–5, and common HF integrations.
- Hugging Face regularly updates base images, default library versions, and platform features.
- Always cross-check:
  - The Spaces overview and ZeroGPU docs,
  - Gradio’s latest documentation,
  - Recent forum threads if you hit unusual or new errors.

Treat this document as a practical starting point rather than a replacement for official docs. When in doubt, trust the latest official documentation and announcements.

---

## 9. References / links (non-exhaustive)

- Spaces overview: https://huggingface.co/docs/hub/spaces-overview  
- Spaces main docs hub: https://huggingface.co/docs/hub/spaces  
- Spaces ZeroGPU: https://huggingface.co/docs/hub/spaces-zerogpu  
- Spaces Dev Mode: https://huggingface.co/docs/hub/spaces-dev-mode  
- Inference Endpoints logs: https://huggingface.co/docs/inference-endpoints/guides/logs  
- Gradio documentation: https://www.gradio.app/docs/  
- Gradio Python client: https://www.gradio.app/docs/python-client/introduction  
- Hugging Face forum (Spaces category): https://discuss.huggingface.co/c/spaces/31  
