# Hugging Face Spaces Debugging Guide  
_Debugging ZeroGPU / GPU Spaces step by step (2024–2025)_

This document is a practical guide for debugging Hugging Face Spaces, especially Spaces that:

- Use **ZeroGPU** or GPU hardware.
- Host **Gradio / Streamlit** apps for LLMs, diffusion models, or other ML workloads.
- Are sometimes created or modified using **LLM-generated code**.

The goal is to give you a **repeatable debugging workflow**, plus a set of **common patterns and recipes** so that:

- You know where to look first (logs, status, hardware).
- You can quickly classify the problem type.
- You can iterate in a structured loop instead of randomly trying things.

This guide focuses on:

1. How Spaces actually run (mental model).
2. How to read logs and runtime state.
3. A simple, repeatable debugging loop.
4. Concrete recipes for common failures.
5. How to deal with LLM-generated code.
6. A learning roadmap and external resources.

---

## 0. Quick TL;DR

When a Space is “broken,” you can almost always classify it into one of these buckets:

1. **Build never finishes / fails**  
   - State: usually stuck in `Building...` or `Failed`.  
   - Fix by: checking build logs, reducing dependencies, fixing `requirements.txt` / `Dockerfile`, pinning versions.

2. **App crashes at runtime**  
   - State: `Running`, but the UI or API returns 500 or the app restarts.  
   - Fix by: reading runtime logs, fixing stack traces, handling edge cases, memory issues, or incorrect imports.

3. **HTTP / API errors (4xx / 5xx) when calling the Space**  
   - State: `Running`, but client calls fail.  
   - Fix by: inspecting response body, validating inputs, handling timeouts, checking authentication, aligning with expected schema.

4. **Platform-level / infrastructure issues**  
   - State: many Spaces failing similarly, logs look normal, sudden mass regressions.  
   - Fix by: checking Hugging Face status, forums, or trying a minimal Space on the same hardware.

For almost any bug:

1. **Observe** what the Space is doing (status + logs).
2. **Classify** the failure type (build / runtime / API / platform).
3. **Form a hypothesis** about the root cause.
4. **Design and run a small test** that isolates that hypothesis.
5. **Simplify the Space** until the minimal version works.
6. **Re-add complexity gradually**, confirming at each step.

---

## 1. Mental model: what you are debugging

### 1.1 What a Space really is

A Hugging Face Space is:

- A **Git repository** with code, config, and small assets.
- Deployed to a **managed container** (Docker-like environment).
- Running on **fixed hardware** (CPU, GPU, or ZeroGPU) with specific limits.
- Exposed as:
  - A web UI (Gradio/Streamlit).
  - A simple HTTP API for `/predict` or similar (depending on the framework and configuration).

The key point: you are debugging **your code in a controlled, repeatable environment**, not a mysterious black box.

### 1.2 ZeroGPU and GPU Spaces

ZeroGPU Spaces:

- Run on **CPU by default**.
- Temporarily spin up a **shared GPU** when requests arrive.
- Scale **back to CPU / idle** when inactive, to save costs and democratize GPU access.

Regular GPU Spaces:

- Have a **dedicated GPU** (e.g., T4, A10G, etc.) for the lifetime of the runtime.

In both cases:

- You get a **limited amount of RAM** and **ephemeral disk** (for caching models, temporary files, etc.).
- If your code tries to allocate more than available, you see OOM errors, crashes, or mysterious restarts.

### 1.3 What can break

In practice, most failures fall into:

- **Dependency problems**  
  (`ModuleNotFoundError`, version conflicts, incompatible CUDA, missing system packages).

- **Runtime logic errors**  
  (index errors, unexpected `None`, shape mismatches, incorrect paths or URLs).

- **Resource issues**  
  (out-of-memory, disk full, too many concurrent requests, timeouts).

- **Platform / configuration issues**  
  (Space stuck in `Building`, misconfigured `Dockerfile`, wrong `runtime.txt`, mis-specified entrypoint).

If you can decide which of these you are facing, the rest of the debugging becomes far more straightforward.

---

## 2. Quick symptom → section map

Use this as a quick navigator:

| Symptom                                                                 | Likely type         | Go to section |
|-------------------------------------------------------------------------|---------------------|---------------|
| Space stuck on “Building...” or instantly shows “Build failed”           | Build failure       | §5            |
| “Running” but UI is blank / crashes / restarts on every request        | Runtime crash       | §6            |
| HTTP 5xx when calling the Space API                                    | Runtime / API error | §7            |
| HTTP 4xx (400 / 403 / 404 / 429, etc.) when calling Space or endpoint  | Client/API misuse   | §7            |
| “Disk is full” or cannot download models / datasets                    | Resource issue      | §6.3          |
| GPU / CUDA / ZeroGPU-specific error messages                           | Resource / config   | §6.4          |
| Many Spaces fail similarly, suddenly, with no code change on your side | Platform issue      | §8            |
| LLM wrote most of the app and you don’t understand parts of it         | LLM code problem    | §9            |

---

## 3. Where to look first: status, hardware, logs

### 3.1 Runtime status

Each Space has a **runtime status** (“stage”) such as:

- `BUILDING` – the container image is being created and dependencies installed.
- `RUNNING` – the app process is running.
- `PAUSED` / `STOPPED` – the runtime is not active.
- `FAILED` – the runtime failed to start or crashed.

Confirm the current stage before doing anything else. It tells you whether you should focus on:

- Build logs (if `BUILDING` / `FAILED` early).
- Runtime logs (if `RUNNING` but misbehaving).
- Configuration (if it never reaches `RUNNING` at all).

### 3.2 Hardware and resource limits

Check which hardware type you are using:

- **CPU** only.
- **GPU** (e.g., T4, A10G, etc.).
- **ZeroGPU** (CPU with on-demand GPU).

Then remember:

- Ephemeral disk (for models, datasets, caches) has a fixed size per hardware tier.
- Your Space repo itself is also subject to storage limits; very large files belong in **Model** or **Dataset** repos, not in the Space repo.

If you push multiple GB of model weights directly into the Space repository, you are very likely to hit disk / storage problems.

### 3.3 Logs: build vs runtime vs app

Three key log streams:

1. **Build logs**  
   - Show dependency installation, `pip` output, `apt-get` output, etc.
   - Use these for:
     - `pip` failures.
     - System package installation failures.
     - Timeout during `pip install` or large downloads.

2. **Runtime logs**  
   - Show app startup and lifecycle after the container is built.
   - Use these for:
     - Import-time errors.
     - Model loading issues.
     - Unhandled exceptions at runtime.

3. **Application logs** (your own `print()`/`logging` outputs)  
   - Use them to:
     - Log every external call (HF Inference API, other Spaces, external services).
     - Log input shapes / sizes.
     - Log timings and memory usage (when possible).
     - Confirm which branch of the code is actually running.

If you cannot see the error in the UI, the logs almost always contain a stack trace, an exception, or at least a hint about what is wrong.

---

## 4. A standard debugging loop

Use this loop consistently. Start simple and repeat.

### Step 1: Identify the failure type

- Use the **status** (build vs runtime).
- Use the **symptom** (HTTP 4xx/5xx, crash, blank UI).
- Use the **logs** to confirm which path is failing.

### Step 2: Reproduce the problem in the simplest possible way

- Trigger the app exactly as the user would:
  - Send the same input via the UI.
  - Call the same HTTP endpoint with the same payload (using `curl` / `requests`).
- Try to isolate a **minimal input** that causes the failure.
  - If a long prompt breaks, test shorter prompts.
  - If a large image breaks, test with a tiny image.

### Step 3: Simplify the code until it works

- Comment out or remove:
  - Optional features.
  - Unnecessary logging.
  - Extra calls to other APIs.
- Replace the main function with a **minimal script**:
  - Load the model.
  - Run one simple inference.
  - Return a constant or very simple result.

If this minimal version fails, you know the problem is in the **environment, dependencies, or core model logic**, not in your UI wrappers.

### Step 4: Add logging to confirm assumptions

- Log:
  - Inputs (in a safe and privacy-preserving way).
  - Shapes and dtypes of tensors.
  - Device placement (CPU/GPU).
  - Start/end of each major step.
- Make logs **specific**:
  - “Loading model...” vs “Loading `meta-llama/Llama-3-8B-Instruct` using `AutoModelForCausalLM` on device `cuda`”.

### Step 5: Work locally and in Dev Mode when needed

- Reproduce issues locally using:
  - A small **conda** / **venv** environment with same `requirements.txt`.
  - Similar Python versions.
- Use **Dev Mode** to connect via SSH or VS Code to the live Space:
  - Inspect files and logs with your usual tools.
  - Run commands interactively in the same environment the Space uses.

### Step 6: Rebuild, test, and document

- Commit small changes and redeploy.
- Re-run:
  - Minimal test.
  - One or two realistic test cases.
- Keep a short note:
  - “Problem → Hypothesis → Experiment → Result → Fix”.
- This history helps when the same problem appears again months later.

---

## 5. Failure type A: Build never finishes or fails

Typical signs:

- Space stays in **“Building...”** for a long time.
- Space shows **“Build failed”** and never reaches “Running”.
- Build logs show errors but you never see the app.

### 5.1 Common root causes

1. **Dependency installation failure**
   - `pip install` cannot find a package.
   - Specific version conflicts (e.g., `torch==1.8.1` incompatible with other packages).
   - System library not available.

2. **Huge downloads during build**
   - Installing a package that downloads large assets at build time.
   - Using `pip install` from unstable or slow sources.

3. **Wrong or overly complex Dockerfile**
   - Custom `Dockerfile` that doesn’t match the Space runtime expectations.
   - Missing entrypoint or port.
   - Using base images that conflict with Hugging Face’s runtime.

4. **Repository too large or mis-organized**
   - Large model files or datasets pushed into the Space repo instead of a Model / Dataset repo.
   - Hitting storage or checkout limits.

### 5.2 Debugging steps

1. **Open build logs and scroll until the first error**  
   - Look for:
     - `ERROR: Could not find a version that satisfies the requirement`.
     - `Command "python setup.py build_ext" failed` or other C-extension issues.
     - Timeouts or repeated retries.

2. **Simplify `requirements.txt`**
   - Remove anything not needed to reproduce the bug.
   - Pin critical versions (especially `torch`, `transformers`, `diffusers`, `accelerate`, `gradio`).
   - If using nightly / dev versions, consider switching to stable releases.

3. **Avoid huge downloads in build phase**
   - Do not `wget` large files during build.
   - Host models in **Model repos** and load them at runtime via `from_pretrained` or HF Hub caching.

4. **If using Dockerfile**
   - Start from a **minimal working Dockerfile**.
   - Confirm:
     - You expose the correct port.
     - The entrypoint runs your app.
   - Avoid installing extra system packages unless necessary.

5. **If the repo is huge**
   - Move heavy assets into dedicated Model or Dataset repos.
   - Reference them by name from the Space code.
   - Ensure `.gitattributes` and `.gitignore` exclude unnecessary local artifacts.

---

## 6. Failure type B: App crashes at runtime

Typical signs:

- Space reaches **“Running”**, but:
  - The UI returns errors.
  - Logs show repeated exceptions.
  - The container restarts frequently.

### 6.1 Python exceptions and stack traces

First, look for:

- A Python traceback with the final line like:
  - `TypeError: ...`
  - `ValueError: ...`
  - `IndexError: ...`
  - `ModuleNotFoundError: ...`
- This indicates a **pure Python problem** in your app.

Checklist:

1. **Find the first stack trace after startup**.
2. **Identify the topmost line in your code** (not in external libraries).
3. **Check the input values** at that point:
   - Are shapes and data types as expected?
   - Are environment variables set?
   - Is the path or model ID correct?

### 6.2 Dependency mismatches at runtime

Sometimes dependencies install fine but fail at import time:

- Example: `ImportError` for a symbol that exists only in newer library versions.
- Example: `AttributeError` because an API changed.

Fix:

- Align versions in `requirements.txt` with libraries known to work together:
  - For example, pair `transformers` with a corresponding `torch` version as recommended in docs or examples.
- Use exactly the versions recommended in official HF example Spaces whenever possible.

### 6.3 Resource problems: memory and disk

Symptoms:

- Process killed with no obvious Python exception.
- Logs mention OOM or memory errors.
- Errors like “Disk full” or inability to cache models.

Consider:

- How big your model is.
- How many concurrent requests you allow.
- How much ephemeral disk your hardware tier provides.

Typical mitigations:

- Use a **smaller / quantized model**.
- Use **streaming generation** to reduce peak memory.
- Limit max input length, image resolution, batch size.
- Clean temporary files and only cache what you need.

### 6.4 GPU / ZeroGPU-specific issues

Common patterns:

1. **CUDA not available / GPU not found**
   - You requested `device="cuda"` but hardware is CPU-only.
   - ZeroGPU sometimes has a short delay before a GPU is available.
   - Fix by:
     - Detecting `torch.cuda.is_available()` before forcing GPU.
     - Allowing CPU fallback for small models.

2. **Device mismatch**
   - Tensors on CPU, model on GPU or vice versa.
   - Fix by ensuring consistent `.to(device)` for model and inputs.

3. **ZeroGPU cold-start delays and timeouts**
   - First request after idle can be slow while GPU spins up.
   - Use:
     - Warm-up requests.
     - Clear user-facing messaging about first-run latency.
   - Avoid heavy extra setup inside the request handler.

---

## 7. Failure type C: HTTP / API errors

You might call your Space (or an Inference Endpoint) via HTTP. Common status codes:

- **4xx** – problem with the client request:
  - `400` – malformed input (wrong schema, missing field).
  - `401` / `403` – authentication / permission issues.
  - `404` – wrong URL or Space name.
  - `429` – too many requests / rate limiting.

- **5xx** – problem on server side:
  - `500` – unhandled exception in your Space code or endpoint.
  - `502` / `503` – upstream or transient issues.
  - `504` – timeout.

### 7.1 Debugging 4xx errors

1. **Validate the request schema**
   - Compare your payload to the API docs or to the payload used in a working example.
   - Ensure:
     - Correct content type (`application/json`).
     - All required fields present.
     - Types match expected types (strings vs numbers vs arrays).

2. **Check auth and URL**
   - Confirm:
     - Correct Space / endpoint name and path.
     - Correct token with required permissions.

### 7.2 Debugging 5xx errors

1. **Check the response body for details**
   - Often contains an error message or stack trace fragment.
2. **Immediately check runtime logs**
   - Look for the same timestamp and trace.
3. **Reproduce using a minimal script**
   - A tiny `requests` or `curl` call that sends exactly the same payload.
4. **Treat as a runtime crash**
   - Use the loop from §4 and the runtime guidance from §6.

---

## 8. Failure type D: platform-level issues

Sometimes the problem is not your code. Typical signs:

- Multiple unrelated Spaces on the same hardware type fail similarly.
- A simple “Hello world” Space on the same hardware also fails.
- Logs look normal but runtimes go into `FAILED` or `RUNNING` → `FAILED` with minimal information.

In those cases:

1. **Try a minimal test Space**
   - Simple Gradio app with no model.
   - If even that fails, suspect platform issues or quota/problem with that hardware tier.

2. **Check public status / incident reports**
   - Look for notices about Spaces, hardware, or storage.

3. **Search community forums**
   - Many storage / hardware / runtime changes are discussed in HF forums.

4. **When you file an issue or ask for help**
   - Include:
     - Space URL and hardware type.
     - Exact error messages and timestamps.
     - Whether a minimal Space on the same tier works.

---

## 9. Debugging LLM-generated code

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

## 10. Using Dev Mode and VS Code for debugging

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

## 11. Cookbook: common problems and fixes

This section gives recipe-style answers for recurring issues.

### 11.1 `ModuleNotFoundError` / missing dependencies

**Symptom**

- Build succeeded, but Space crashes at startup with `ModuleNotFoundError`.

**Likely cause**

- Dependency missing in `requirements.txt`.
- Wrong import name (e.g., `import cv2` vs `pip install opencv-python`).

**Fix now**

- Add missing package to `requirements.txt` with a specific version.
- Rebuild the Space.

**Prevent later**

- When adding any import, also:
  - Add or update the requirement.
  - Keep a minimal dependency set.

---

### 11.2 Version conflicts and ABI issues

**Symptom**

- Errors like:
  - `undefined symbol` in C extensions.
  - `ImportError` mentioning incompatible versions.
- Often appears with `torch`, `xformers`, `bitsandbytes`, etc.

**Likely cause**

- Incompatible versions of `torch` and extension libraries.
- Using wheels not built for the platform or Python version.

**Fix now**

- Align versions exactly as recommended in:
  - Official HF examples or model cards.
- Remove extra low-level packages if you are not sure you need them.

**Prevent later**

- Prefer simple stacks:
  - Use `transformers`, `accelerate`, and `torch` in combinations seen in official docs or existing Spaces.

---

### 11.3 “Disk full” / cannot download models or datasets

**Symptom**

- Logs show:
  - “Disk is full”.
  - “No space left on device”.
- Model or dataset downloads fail.

**Likely cause**

- Repo and caches exceed the ephemeral disk capacity for the hardware tier.

**Fix now**

- Move heavy files into:
  - Separate Model or Dataset repos on the Hub.
- Use `from_pretrained` to fetch them instead of storing in the Space repo.
- Clean up:
  - Temporary files.
  - Old models that are no longer used.

**Prevent later**

- Design Spaces assuming:
  - Repo is for **code and light assets**.
  - Models and large data live in reusable Hub repos.

---

### 11.4 Timeouts and slow responses (esp. ZeroGPU)

**Symptom**

- First request after idle is very slow.
- API clients see timeouts.

**Likely cause**

- ZeroGPU cold-start: time to bring GPU online and load model.

**Fix now**

- Reduce model load time by:
  - Using smaller or quantized models.
  - Avoiding unnecessary heavy initialization in the request handler.
- Consider sending a **warm-up request** periodically if acceptable.

**Prevent later**

- Clearly document that the first request may be slower.
- Use benchmarks from HF docs / community discussions to calibrate expectations.

---

### 11.5 Authentication / private models

**Symptom**

- Works locally with your token but fails in the Space.
- Errors mentioning authentication or permissions.

**Likely cause**

- Private model or dataset requires an access token.
- Token not correctly configured in Space secrets.

**Fix now**

- Set an HF token as a **Space secret**.
- Use standard HF client methods to authenticate.

**Prevent later**

- Prefer public models / datasets when possible.
- When using private resources, treat auth configuration as part of deployment, not an afterthought.

---

## 12. Minimal example patterns

Having a simple, known-good Space is extremely useful.

### 12.1 Minimal Gradio “ping” Space

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

### 12.2 Minimal HF Transformers inference snippet

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

## 13. Learning roadmap: what to study and in what order

To debug Spaces effectively (and grow over time), it helps to build a layered skill set.

### 13.1 Immediate essentials (1–2 weeks)

Focus on:

1. **Basic Python**
   - Functions, modules, exceptions, virtual environments.
2. **Git + GitHub / HF Hub**
   - Cloning, committing, pushing, branches.
3. **Hugging Face Spaces basics**
   - Space types (Gradio, Streamlit, Static).
   - How building and runtime work.

Goal: you can read logs, understand stack traces, and modify simple apps.

### 13.2 Short-term: ML + HF ecosystem (1–2 months)

Add:

1. **Transformers and Diffusers basics**
   - How text and image models are loaded and called.
2. **Inference best practices**
   - Batching, streaming, max token limits.
3. **Resources and hardware**
   - Differences between hardware tiers.
   - How ephemeral disk and caching behave.

Goal: you can choose appropriate models, hardware, and trade-offs for your Spaces.

### 13.3 Medium-term: reliability and observability (3–6 months)

Then:

1. **Logging and monitoring**
   - Structured logs.
   - Request tracing and metrics.
2. **Performance and scaling**
   - Profiling Python code.
   - Measuring latency and memory.
3. **Robust API design**
   - Clear input/output schemas.
   - Safe defaults and validation.

Goal: you can build Spaces that are not only functional but **stable** and **predictable** under load.

---

## 14. External resources and further reading

A short, curated list of resources to deepen understanding.

### 14.1 Official documentation and guides

- **Spaces overview and tutorials** – how Spaces work, types, and examples.
- **Hub & storage / hardware docs** – hardware tiers, ephemeral disk, caching behavior.
- **Transformers documentation** – model loading, generation, tokenization.

These documents give you the **canonical behavior** of Spaces, hardware, and core libraries.

### 14.2 Community discussions and patterns

- **Spaces and storage / “disk full” issues** – community threads about caching, disk limits, and large models.
- **ZeroGPU experiences and tips** – discussions about cold starts, performance, and suitable models.

These help you see how other developers solve similar problems.

### 14.3 Troubleshooting and debugging patterns

- **HTTP API usage and Inference docs** – examples of calling endpoints, handling errors, and structuring requests.
- **Dev Mode and remote debugging notes** – how to connect via SSH / VS Code and inspect runtimes.

---

By combining:

- A clear **mental model** of Spaces and hardware.
- A structured **debugging loop**.
- A set of **common patterns and recipes**.
- A **learning roadmap** and curated references,

you can turn “my Space is broken and I have no idea why” into a systematic process: observe, classify, hypothesize, test, simplify, and iterate.
