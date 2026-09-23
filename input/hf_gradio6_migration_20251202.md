---
source: "chat+files+web"
topic: "Migrating to Gradio 6 (with Hugging Face Spaces)"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-12-02T10:09:20Z"
---

# Migrating to Gradio 6 (with Hugging Face Spaces)

This note consolidates your existing internal docs, Hugging Face forum / Spaces experience, and the official Gradio 6 migration material into a single, reuse‑able knowledge base focused on **Gradio 6 migration for Hugging Face Spaces and LLM apps**.

It assumes you are already comfortable with Gradio 5, Hugging Face Spaces, and RAG / agentic workflows.

---

## 1. Background and high‑level picture

### 1.1 Why Gradio 6 matters

From the official migration guide and changelog:

- **Gradio 6 becomes the mainline**: future development will happen on the 6.x series, while 5.x is in maintenance mode only.
- **Goal of the release**: more consistent Python API, better performance, and simpler customization (themes, CSS/JS, layouts).  
- **Backward compatibility strategy**: Gradio 5.50 was released first with deprecation warnings; 6.0.x then removes deprecated parameters and finalizes the new APIs.
- **Ecosystem alignment**: the changes also align Gradio’s APIs with:
  - **MCP-style tool definitions** (API names more descriptive and consistent),
  - **OpenAI-style chat message formats** (for `Chatbot` / `ChatInterface`),
  - and **Spaces / SSR** deployment model used by Hugging Face.

For Hugging Face Spaces, this means you eventually want most Python or Docker Spaces to run on **Gradio ≥ 6.0.x** and align your **`gradio_client`**, agents, and MCP tooling with that line.

### 1.2 Scope of this document

This KB focuses on:

- Migrating **Gradio 5.x → 6.x** for:
  - Regular **`Interface`** demos,
  - **`Blocks`** apps (including multi‑tab dashboards),
  - **`ChatInterface` / `Chatbot`**‑based LLM UIs.
- Keeping **Hugging Face Spaces** healthy when upgrading:
  - Avoiding or debugging **“Error: No API found”**,
  - Keeping the **“View API / Use via API”** behavior stable,
  - Avoiding common dependency pitfalls (Pydantic, ZeroGPU, etc.).
- Coordinating **Python clients** (`gradio_client`, external scripts) with the new API names.

It does **not** repeat every line of the official migration guide; instead it organizes the material specifically around your workloads (RAG, agents, Spaces).

---

## 2. Official Gradio 6 migration guidance (core ideas)

### 2.1 Recommended upgrade path

From the official **Gradio 6 Migration Guide**:

1. **Upgrade locally to 5.50 first**:

   ```bash
   pip install --upgrade "gradio==5.50"
   ```

   - Run your app and watch the console for **deprecation warnings**.  
   - Each warning corresponds to something that will break in 6.x (removed parameter, renamed behavior, etc.).

2. **Fix all deprecations under 5.50**:

   - Replace deprecated parameters with the new ones.
   - Update event listeners, chat history formats, etc.
   - Add tests if you don’t already have them (see checklist later).

3. **Then upgrade to 6.x** (ideally the latest patch, e.g. 6.0.2+):

   ```bash
   pip install --upgrade "gradio>=6,<7"
   ```

4. **Upgrade the Python client** if you use it:

   ```bash
   pip install --upgrade gradio_client
   ```

   - As of late 2025, Gradio 6.0.1+ goes with `gradio_client` 2.x, with fixes for backwards compatibility on Gradio 5 apps as well.

This “5.50 → fix warnings → 6.x” pattern is the safest way to migrate anything non‑trivial.

### 2.2 App‑level changes: from `Blocks(...)` to `launch(...)`

Gradio 5 allowed many **global app parameters** on the `gr.Blocks` constructor:

- `theme`, `css`, `css_paths`,
- `js`, `head`, `head_paths`, etc.

These are now **moved to `Blocks.launch()`** in 6.x, because `Blocks` objects can be nested and reused, while there is only one actual app launch per process.

**Before (5.x):**

```python
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".app-title { color: red; }",
) as demo:
    gr.Textbox(label="Input")

demo.launch()
```

**After (6.x):**

```python
with gr.Blocks() as demo:
    gr.Textbox(label="Input")

demo.launch(
    theme=gr.themes.Soft(),
    css=".app-title { color: red; }",
)
```

For most Spaces, this is a simple mechanical refactor:

- Strip those parameters off `Blocks(...)`,
- Add them to `launch(...)` instead.

### 2.3 Footer & API docs: `show_api` → `footer_links`

In 5.x you likely used `launch(show_api=False)` to hide the footer “API” link, or left it at default to show it.

In 6.x, that is replaced by:

```python
demo.launch(footer_links=["api", "gradio", "settings"])
```

- `footer_links` is a list controlling which footer links appear:
  - `"api"` → “View API / API docs” link,
  - `"gradio"` → “Built with Gradio” link,
  - `"settings"` → the settings cog.

**Mapping:**

| 5.x `show_api` | 6.x `footer_links`                     |
|----------------|----------------------------------------|
| `True` (default)  | `["api", "gradio", "settings"]` or omit |
| `False`         | `["gradio", "settings"]`              |

On **Spaces**, this directly influences whether the “View API / Use via API” UI is visible. In combination with `api_visibility` (next section), this controls what the **Gradio client** and HF Spaces expose as API endpoints.

### 2.4 Event listeners: `show_api`/`api_name=False` → `api_visibility`

In 5.x, event listeners (`.click()`, `.submit()`, etc.) had:

- `show_api=True/False` → show/hide endpoint in API docs, and
- `api_name=False` → completely disable the endpoint.

In 6.x these are replaced by **one parameter**:

```python
btn.click(
    fn=do_something,
    inputs=...,
    outputs=...,
    api_visibility="undocumented",  # or "public" / "private"
)
```

**Mapping:**

| 5.x pattern                 | 6.x `api_visibility`      | Meaning                                                |
|----------------------------|---------------------------|--------------------------------------------------------|
| `show_api=True` (default)  | `"public"` / omitted      | Visible in docs + usable by clients                    |
| `show_api=False`           | `"undocumented"`          | Hidden from docs, still callable by clients            |
| `api_name=False`           | `"private"`               | No API endpoint at all (client cannot call this event) |

**Practical tip for Spaces:**

- When you see **“Error: No API found”** from the frontend or from `gradio_client`, check:
  - Did the app actually finish starting (no tracebacks in logs)?
  - Are you accidentally using `api_visibility="private"` for the event you’re calling?
- For older clients that expect `show_api=False` semantics, use `"undocumented"` so the endpoint remains callable but hidden from the docs page.

### 2.5 Interface / ChatInterface endpoint names

Gradio 5 used **fixed default endpoint names**:

- `Interface` → `/predict`
- `ChatInterface` → `/chat`

Gradio 6 now **uses the function name** as the default endpoint:

- If your function is `generate_text`, then the endpoint becomes `/generate_text`.
- This makes the API self‑documenting and consistent with `Blocks` event names.

**If you rely on old endpoints in clients or Spaces:**

Set `api_name` explicitly:

```python
demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", api_name="predict")
# For ChatInterface
chat_demo = gr.ChatInterface(fn=chat_fn, api_name="chat")
```

This is critical when:

- You have **Python scripts** using `gradio_client.Client(...).predict("/predict", ...)`,
- You have other Spaces or services calling your Space via the HF Inference Endpoints or API.

After migrating to 6.x you should:
- Decide whether to keep legacy names for backward compatibility, or
- Update all callers to use the new function‑based names.

### 2.6 Chat & history formats (`Chatbot` / `ChatInterface`)

Gradio 6 adopts an **OpenAI‑style chat format** and removes older tuple‑based history formats.

Key changes:

1. **Tuple chat format removed:**

   - 5.x allowed `[[user, assistant], ...]` (“tuples”).
   - 6.x requires **message objects** with `role` and `content`.

   ```python
   # 6.x-style messages
   history = [
       {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
       {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
   ]
   ```

   You pass such histories to `gr.Chatbot` or `ChatInterface`, and your `fn` for `ChatInterface` should return them in this structured format.

2. **`type="tuples"` removed:**

   - `Chatbot` must use `type="messages"`.
   - Any stored examples / default values must be converted.

3. **Always‑structured content:**

   - Even plain text is now stored as `{"type": "text", "text": "..."}` inside a `content` list.
   - Files, images, etc. become their own content blocks (`{"type": "file", "file": {...}}`).

This is especially important for your **RAG / agentic chat UIs**:

- Middleware that inspects chat history must handle the new structure.
- Any manual history pre‑population must switch from plain strings to content blocks.

### 2.7 Examples caching (`cache_examples` / `cache_mode`)

In 5.x:

- `cache_examples` could be `True`, `False`, or `"lazy"`.

In 6.x:

- `cache_examples` is **boolean only**,
- A new `cache_mode` parameter chooses `"eager"` vs `"lazy"`.

**Example:**

```python
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    examples=["Hello", "World"],
    cache_examples=True,
    cache_mode="lazy",   # replaces cache_examples="lazy"
)
```

### 2.8 Component‑level highlights

A few component changes that may affect you:

- **`gr.Video`**: stop returning `(video_path, subtitle_path)` tuples; instead return a `gr.Video` instance with `subtitles=`.
- **`gr.HTML`**: `padding` default changed from `True` → `False`, to match `gr.Markdown`. If your layout relied on padding, set it explicitly.
- **`Chatbot.like_user_message`**: moved from `.like()` event to the `Chatbot` constructor.

---

## 3. Python client (`gradio_client`) and API semantics

### 3.1 Parameter rename: `hf_token` → `token`

The Python client now uses a simpler parameter name for authentication:

```python
from gradio_client import Client

client = Client("username/space-name", token="hf_your_token_here")
```

- Replace any use of `hf_token=` with `token=`.
- This is a small but breaking change for scripts and agents that connect to private Spaces.

### 3.2 Exception changes: `AppError`

`AppError` now derives directly from `Exception` instead of `ValueError`.

- If you previously had `except ValueError` blocks catching `AppError`, you must update them to catch `AppError` explicitly.
- This matters in tools / agents that distinguish input validation errors from true app failures.

### 3.3 Version alignment and Gradio 5 compatibility

From the changelog and issues:

- Gradio 6.0.x introduces **`gradio_client` 2.0.0**.
- Early versions had **regressions when calling 5.x apps**; later patches add backward compatibility.
- Practical advice for you:
  - For Spaces you control and migrate to 6.x, you can standardize on `gradio_client>=2.0.0`.
  - For **client code that must talk to both 5.x and 6.x Spaces**, test carefully and pin to a combination known to work in your environment, watching the Gradio changelog for fixes.

---

## 4. Hugging Face Spaces specifics (including “No API found”)

Your previous Spaces debugging surfaced a key pattern:

> “Error: No API found” almost always means the **backend crashed before Gradio finished building the API schema**, not a network problem.

### 4.1 What “No API found” means internally

From Gradio’s frontend/client code:

- The error is raised when the client cannot find any API info for an endpoint / component.
- This happens if:
  - Gradio never finished generating the configuration because the app crashed on startup, or
  - The endpoint was made **private** / removed (e.g. `api_visibility="private"`), or
  - There is a mismatch between what the client is trying to call and the app’s actual endpoints (e.g. calling `/predict` when the app only exposes `/generate_text`).

On Spaces this often appears as:

- Space shows as “Running”,
- UI renders, but the **first interaction fails** with “Error: No API found”.

### 4.2 Why it often works locally but fails on Spaces

Key differences on Hugging Face Spaces:

- **Fresh container per build**:
  - Dependencies are installed from `requirements.txt` and/or `sdk_version` in `README.md`.
- **Different dependency resolution**:
  - Even minor changes (adding `google-genai`, `anthropic`, etc.) can pull **newer Pydantic / FastAPI / Starlette** or other packages than your local venv.
- **Different hardware / ZeroGPU**:
  - Upgrading to ZeroGPU or different GPU tiers can force new versions of CUDA / PyTorch.

If any of these changes cause a Python exception during Gradio startup, the API schema is never built, and both the UI and `gradio_client` see “No API found”.

### 4.3 Pydantic + Gradio regressions (Gradio 5.x but still relevant)

Community threads and GitHub issues show a cluster of failures:

- Logs show `TypeError: argument of type 'bool' is not iterable` in `gradio_client.utils.get_type` while building the schema.
- Triggered by combinations of:
  - Newer **Pydantic 2.x** releases,
  - Certain **Gradio 5.x** versions,
  - Spaces that recently rebuilt or added new dependencies.

The practical, community‑tested workaround has been:

```text
pydantic==2.10.6
gradio==5.x.y   # pinned to a version you know works locally
```

This is still worth remembering when migrating to 6.x because:

- Your Spaces may still run 5.x during the “upgrade in place” phase.
- Some Gradio 6.x releases may internally still use similar schema code paths, so keeping Pydantic pinned to a version that works for you is a safe starting point.

**Recommended approach for you:**

1. For each important Space, create a **clean local venv**:
   - Install exactly what is in `requirements.txt`.
   - Confirm that the app launches locally with the same versions.
2. If you hit the `bool is not iterable` error:
   - First, **try pinning Pydantic to a known‑good version** (e.g. 2.10.6) and re‑testing.
   - Check Gradio’s issues for any explicit fixes in later 6.x releases.
3. Once stable, lock `gradio`, `pydantic`, `fastapi`, and `starlette` versions in `requirements.txt` so Spaces builds stay reproducible.

### 4.4 `api_visibility` and Spaces “View API / Use via API”

In Gradio 6.x on Spaces, the pieces fit together like this:

- **`footer_links`** controls whether the footer shows an **“API” link** that opens the docs page.
- The docs page shows only endpoints with `api_visibility="public"`.
- **Clients (Python/JS)** can call:
  - Endpoints marked `"public"` (visible + callable),
  - Endpoints marked `"undocumented"` (hidden but callable),
  - Not endpoints marked `"private"` (not callable).

**Typical policies for your Spaces:**

- For public demos where you want people to call the Space as an API:
  - `footer_links` includes `"api"`.
  - Important endpoints have `api_visibility="public"` and explicit `api_name` if you want stable URLs.
- For internal / tool‑only Spaces used by agents or MCP:
  - You can set `footer_links=["gradio"]` to hide the API link entirely,
  - Use `api_visibility="undocumented"` for endpoints that should be callable by tools but not advertised in docs,
  - Reserve `"private"` for internal plumbing endpoints that must not be callable from outside.

This mapping replaces the ad‑hoc mix of `show_api`/`api_name` you used in 5.x.

### 4.5 ZeroGPU and other environment‑specific issues

Forum threads show that moving a Space to **ZeroGPU** sometimes coincides with “No API found” errors. Common causes:

- Old `torch` or CUDA‑bound packages that do not work in the ZeroGPU environment.
- Large model or pipeline initialization at import time that fails due to resource limits.

Best practices when migrating to 6.x with ZeroGPU:

- Pin a **modern but compatible PyTorch** (for example `torch>=2.2` for many Spaces) and compatible Transformers / `huggingface_hub` versions.
- Move heavy initialization inside Gradio callbacks rather than module import time.
- Ensure your `README`’s `sdk` / `sdk_version` and `requirements.txt` are consistent with each other.

---

## 5. Migration patterns by app type

This section gives concrete patterns for the kinds of apps you run: simple demos, RAG / LLM chatbots, and MCP‑style tools.

### 5.1 Simple `Interface` / `Blocks` demo

**Before (5.x, typical Space):**

```python
import gradio as gr

def classify(text):
    return {"label": "positive", "score": 0.97}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    inp = gr.Textbox(label="Input")
    out = gr.JSON(label="Result")
    btn = gr.Button("Run")
    btn.click(classify, inp, out, show_api=True)
demo.launch(show_api=True)
```

**After (6.x):**

```python
import gradio as gr

def classify(text):
    return {"label": "positive", "score": 0.97}

with gr.Blocks() as demo:
    inp = gr.Textbox(label="Input")
    out = gr.JSON(label="Result")
    btn = gr.Button("Run")
    btn.click(
        classify,
        inp,
        out,
        api_visibility="public",  # or "undocumented" for hidden-but-callable
    )

demo.launch(
    theme=gr.themes.Soft(),
    footer_links=["api", "gradio", "settings"],
)
```

**Key changes:**

- App‑level params moved to `launch()`.
- Event listener uses `api_visibility` instead of `show_api`.

### 5.2 Chat‑based RAG / Agentic UI (`ChatInterface`)

**Before (5.x, sketch):**

```python
import gradio as gr

def chat_fn(message, history):
    # history is list of [user, assistant] tuples
    answer = run_rag(message, history)
    history = history + [[message, answer]]
    return "", history

demo = gr.ChatInterface(
    fn=chat_fn,
    title="RAG Chat",
    examples=[["Hello", "Hi there!"]],
    cache_examples="lazy",
)
demo.launch()
```

**After (6.x):**

```python
import gradio as gr

def chat_fn(message, history):
    # history is a list of message dicts with OpenAI-style content blocks
    # Convert to your RAG format if needed, then back to messages
    answer_text = run_rag_from_messages(message, history)
    history = history + [
        {"role": "user", "content": [{"type": "text", "text": message}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
    ]
    return "", history

demo = gr.ChatInterface(
    fn=chat_fn,
    title="RAG Chat",
    examples=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there!"}],
        },
    ],
    cache_examples=True,
    cache_mode="lazy",
    api_name="chat",  # keep old /chat endpoint if you have clients
)
demo.launch()
```

**Key migration points:**

- `history` and `examples` now use OpenAI‑style structured messages.
- `cache_examples="lazy"` becomes `cache_examples=True, cache_mode="lazy"`.
- `api_name="chat"` keeps compatibility for existing clients calling `/chat`.

### 5.3 MCP / tool‑like Spaces

You already have MCP and tool‑execution architecture; Gradio 6’s changes help here:

- Endpoint names reflect **function names**, which align nicely with MCP tool names.
- `api_visibility` maps to MCP‑style visibility semantics (public, internal, hidden).

Migration considerations:

1. **Decide on canonical function names**:

   - Use short, API‑like names (e.g. `answer_question`, `summarize_docs`, `extract_entities`).
   - With 6.x, these become your default endpoint names.

2. **Use `api_visibility` to match MCP expectations**:

   - Public tools: `api_visibility="public"`.
   - Internal helpers: `api_visibility="private"` (or keep them out of the MCP tool list altogether).

3. **Coordinate with your MCP server or HF MCP tooling**:

   - If the MCP layer expects `/predict` style endpoints, explicitly set `api_name="predict"` until you can migrate the MCP side as well.
   - In the longer term, prefer function‑named endpoints; they map more directly to MCP tool IDs.

---

## 6. Migration checklists

### 6.1 Per‑repo checklist (local before Spaces)

1. **Upgrade to 5.50 locally** and run the app:
   - Note all deprecation warnings and stack traces.
2. **Fix deprecations**:
   - Move app‑level params to `launch()`,
   - Replace `show_api` / `api_name=False` with `api_visibility`,
   - Convert chat history / examples to the new messages format,
   - Update `cache_examples` / `cache_mode`,
   - Update any `gr.Video` tuple returns.
3. **Add or update tests** for:
   - Main app load,
   - Critical endpoints / workflows,
   - Chat history in/out shapes for RAG / agents.
4. **Freeze dependencies**:
   - Pin `gradio` to the target 6.x version,
   - Pin other sensitive libs (`pydantic`, `fastapi`, `starlette`, `torch`, `transformers`, `huggingface_hub`) to combinations that work in your tests.

### 6.2 Space deployment checklist

For each Space:

1. **Align `README`’s `sdk`/`sdk_version` with `requirements.txt`**.
2. **Check logs on first deploy to 6.x**:
   - Any startup exception will likely surface as “No API found” in the UI.
3. **Verify the API page**:
   - Confirm the “API” footer link appears or does not appear as expected (via `footer_links`),
   - Ensure critical endpoints appear with the expected names (`/predict`, `/generate_text`, etc.).
4. **Test with `gradio_client`**:
   - From a separate script, call the Space using the expected endpoint names,
   - Make sure `token=` is used for private Spaces.
5. **Watch for known Gradio 6 issues**:
   - SSR quirks for Spaces,
   - `gr.State`+MCP bugs,
   - Widget‑specific regressions (DownloadButton, Image sliders, performance issues on large GUIs).  
   If you hit any of these, consider pinning to a specific 6.x patch or, in the worst case, temporarily staying on 5.49.1 for that particular app.

### 6.3 Agent / RAG / tool‑calling code checklist

1. Update **Python clients**:

   - Replace `hf_token=` with `token=`,
   - Catch `AppError` explicitly where needed.

2. Audit all **hard‑coded endpoint names**:

   - `/predict`, `/chat`, and any custom `api_name` values.
   - Decide case by case whether to:
     - Keep the old names via explicit `api_name=...`, or
     - Update code and use function‑named endpoints.

3. Normalize **chat histories**:

   - Add a utility layer that converts between:
     - Gradio 6 structured messages, and
     - Your internal conversation format for RAG / MCP tools.

4. For MCP‑aware systems:

   - Ensure the tools registry matches whatever endpoint names and visibility (`api_visibility`) you choose in Gradio,
   - If you rely on `gr.State` within MCP flows, track open Gradio 6 issues and test thoroughly.

---

## 7. Limitations, caveats, and open questions

As of early December 2025, there are still some open or evolving areas:

- **Gradio 6 is very new**:
  - GitHub shows active “Gradio 6 Cleanup” issues (SSR bugs, JS regressions, component quirks).
  - Expect minor breaking changes and bug fixes in 6.0.x series; plan for occasional patch upgrades.

- **`gradio_client` 2.x for mixed 5.x/6.x fleets**:
  - While backward compatibility has improved, any environment that must talk to **both** 5.x and 6.x Spaces needs careful testing.
  - You may end up pinning `gradio_client` at a specific version for stability.

- **Pydantic / schema interactions**:
  - The historical Pydantic 2.x regressions suggest keeping a close eye on schema‑generation code when updating dependencies.
  - If you hit the `bool is not iterable` pattern again under 6.x, start by checking Pydantic and Gradio versions and searching recent GitHub issues.

- **MCP integration and state handling**:
  - Gradio 6 adds and refines MCP integration, but some combinations (e.g., `gr.State` plus MCP) still have open bug reports.
  - For critical MCP tools, plan an isolated test harness and consider conservative version pinning.

- **Lite / static Spaces**:
  - Gradio‑Lite lives on a different track and brings its own set of issues (Pyodide, CDNs, etc.).
  - Migrating the Python backends to 6.x does not change Lite’s constraints; for serious workloads, continue preferring regular Python or Docker Spaces.

Overall, Gradio 6 is the direction of travel and aligns well with your plans (RAG, MCP, agentic stacks on Spaces). The core work is mechanical but you should budget time for:

- Dependency pinning,
- History and endpoint refactors,
- And regression testing, especially for Spaces that serve as **APIs** or **tools** for other systems.

---

## 8. Reference links

### 8.1 Official Gradio resources

- [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide)
- [Gradio Changelog](https://www.gradio.app/changelog)
- [Gradio GitHub Repository](https://github.com/gradio-app/gradio)
- [Getting Started With the Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client)

### 8.2 Hugging Face Spaces and forum threads

- [Spaces Configuration Reference](https://huggingface.co/docs/hub/en/spaces-config-reference)
- [Error: No API Found – Spaces forum thread](https://discuss.huggingface.co/t/error-no-api-found/146226)
- [TypeError: argument of type 'bool' is not iterable – Spaces crash thread](https://discuss.huggingface.co/t/gradio-space-crashing-on-startup-typeerror-argument-of-type-bool-is-not-iterable/154601)
- [Internal server error / bool not iterable – additional discussion](https://discuss.huggingface.co/t/internal-server-error-bool-not-iterable/149494)
- [If your Space stops working after restarting – stability tips](https://huggingface.co/posts/John6666/369491746519704)

### 8.3 GitHub issues and examples

- [Application does not launch (schema / Pydantic issue)](https://github.com/gradio-app/gradio/issues/10649)
- [Gradio showing Error: NO API Found](https://github.com/gradio-app/gradio/issues/8410)
- [Possible regression in gradio_client 2.0 for Gradio 5 API](https://github.com/gradio-app/gradio/issues/12484)
- [SSR Not Working Correctly 6.0](https://github.com/gradio-app/gradio/issues/12456)
- [HighlightedText does not work on Gradio 6](https://github.com/gradio-app/gradio/issues/12463)
- [hfs-florence-2 demo (example of pinning Pydantic to fix No API Found)](https://github.com/ohjho/hfs-florence-2)
