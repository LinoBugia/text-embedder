---
source: "huggingface+chat+local-specs"
topic: "Migration from the legacy Serverless Inference API to Inference Providers"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T00:00:00Z"
---

# Migrating from the legacy Serverless Inference API to Inference Providers

## 1. Background and overview

### 1.1 What the old Serverless Inference API was

For several years Hugging Face exposed a **Serverless Inference API** at:

- Base URL: `https://api-inference.huggingface.co`
- Pattern: `POST /models/{model_id}` with a JSON body like:

```json
{
  "inputs": "Hello world",
  "parameters": {
    "max_new_tokens": 32,
    "temperature": 0.7
  }
}
```

The same endpoint supported many different tasks:

- text generation (`text-generation`, `text2text-generation`, summarization, translation)
- token and sequence classification
- image and audio models

The **task** was inferred from the model or from server configuration, rather than from a task‑specific endpoint. Typical clients were:

- direct `curl` / `requests` calls,
- the legacy `InferenceApi` Python helper,
- older versions of `@huggingface/inference` in JavaScript.

The free tier was comparatively generous and was often used as a “default serverless backend” for small projects and prototypes.

### 1.2 Decommissioning of `api-inference.huggingface.co`

By late 2025 the legacy endpoint has been fully decommissioned:

- Requests to `https://api-inference.huggingface.co` now return **410 Gone** or **404 Not Found**.
- The error body explicitly states that `https://api-inference.huggingface.co is no longer supported` and instructs users to
  use `https://router.huggingface.co/hf-inference` instead.
- Recent Hugging Face forum threads confirm that this is a **permanent removal**, not an outage or temporary incident.

Any remaining code, SDK configuration, or infrastructure that still targets `api-inference.huggingface.co` must be treated as broken
and scheduled for migration.

### 1.3 What replaced it: Inference Providers and the router

Hugging Face now offers **Inference Providers**, a unifying abstraction over many serverless inference backends. Key ideas:

- A central **router** exposes an **OpenAI‑compatible API**, with base URL:

  - `https://router.huggingface.co/v1` for the Chat Completion / Responses APIs.

- Multiple **providers** sit behind the router:
  - third‑party inference platforms (Groq, Cerebras, SambaNova, Together, Fireworks, Novita, Scaleway, etc.),
  - Hugging Face’s own **HF Inference** provider.

- `HF Inference` is explicitly documented as:

  > “the serverless Inference API powered by Hugging Face. This service used to be called ‘Inference API (serverless)’ prior to Inference Providers.”

From the user’s perspective:

- The old “Serverless Inference API” has been folded into the **HF Inference provider**.
- Instead of calling `api-inference.huggingface.co` directly, you now either:
  - call the **router** using an OpenAI‑style client, or
  - call task‑specific helpers that route to a provider (including HF Inference).

### 1.4 Conceptual shift: from “one endpoint” to “providers + tasks”

Under the legacy API, a single endpoint tried to cover many models and tasks, with minimal distinction between backends. With
Inference Providers the model lineup and behavior become **provider‑aware** and **task‑aware**:

- A given model may be available on:
  - many providers (e.g. several vendors offering Llama 3.1),
  - a single provider,
  - or not via Providers at all (only as a plain Hub repo, Space, or self‑hosted deployment).
- The **Supported Models** catalog and per‑model **Inference Providers widget** show, for each model:
  - which providers host it,
  - which tasks (chat completion, image generation, embeddings, speech‑to‑text, etc.) are supported,
  - pricing and context limits for that combination.

This is the biggest conceptual change compared to the old Serverless Inference API. You no longer assume “any Hub model can be
called as a serverless endpoint”; instead, you check which **provider+task** combinations exist and design around them.

## 2. From official docs, blog posts, and guides

### 2.1 Inference Providers and the router

The Inference Providers documentation and “Getting Started” guides describe the new architecture:

- **Unified, OpenAI‑compatible endpoint.**  
  The router exposes a Chat Completion API that is compatible with OpenAI clients. You can usually migrate existing OpenAI code
  by changing only:
  - the `base_url` to `https://router.huggingface.co/v1`, and
  - the API key to a Hugging Face token with Providers permissions.
- **Provider selection.**  
  For many popular models, multiple providers are available. You can either:
  - pin a specific provider (e.g. `meta-llama/Llama-3.1-8B-Instruct:cerebras`), or
  - let the router auto‑select based on `:fastest` or `:cheapest` suffixes.
- **Chat‑focused router.**  
  The OpenAI‑style router is currently focused on **chat completion** (and related Responses API). Non‑chat tasks such as
  image generation, embeddings, or audio use the task‑specific helpers on the Hugging Face clients.

Blog posts and guides (for example, “Getting Started with Inference Providers” and introductions to GPT‑OSS) show end‑to‑end
examples of calling Llama and GPT‑OSS models through Providers with the OpenAI client.

### 2.2 HF Inference provider: the new home of the serverless API

The **HF Inference** provider documentation makes explicit the relationship between the old and new stacks:

- HF Inference is described as “the serverless Inference API powered by Hugging Face”.
- The docs and the pricing page both note that this service **used to be called “Inference API (serverless)”**.
- From the user’s point of view, HF Inference behaves like any other provider:
  - you can select it explicitly via `provider="hf-inference"` or `:hf-inference` in the model string,
  - it serves an allowlisted catalog of models (not arbitrary Hub repos),
  - its usage is billed via the same credit‑based mechanism as other providers.

Compared to the legacy API, this means:

- Some models that once worked via `api-inference.huggingface.co` may no longer be available from HF Inference.
- Newer, optimized deployments (for example, Hugging Face–maintained small models for text generation or embeddings) are exposed
  via HF Inference and advertised in the Providers catalog.

### 2.3 InferenceClient, task helpers, and non‑chat inference

The `huggingface_hub.InferenceClient` (Python) and `@huggingface/inference` (JS) libraries provide a higher‑level interface over
Inference Providers and HF Inference. Key points:

- **Task helpers.**  
  Instead of sending raw JSON to `/models/{id}`, you typically call methods such as:
  - `text_generation(...)`,
  - `summarization(...)`,
  - `token_classification(...)`,
  - `image_to_image(...)`,
  - `audio_to_text(...)`, and so on.
- **Flat keyword arguments.**  
  For generation‑style helpers (e.g. `text_generation`), you no longer pass a nested `parameters={...}` object. Instead, you pass
  flat keyword arguments like `max_new_tokens`, `temperature`, `top_p`, `do_sample`, and `return_full_text`. Using legacy
  `parameters` or names like `max_length` will raise `TypeError` in current client versions.
- **Routing and providers.**  
  By default, the client will auto‑route across Providers where possible. You can pin a provider by passing `provider="hf-inference"`
  or another provider name. For public “classic” models (such as `gpt2`), omitting `provider` typically results in HF Inference
  handling the request.

These helpers replace the ad‑hoc JSON contracts of the legacy API and make it easier to write code that stays compatible even
as backends evolve.

### 2.4 Auth: fine‑grained tokens and permissions

Inference Providers require **fine‑grained access tokens** with appropriate scopes:

- Tokens must be created in the Hugging Face web console.
- To use Providers (including the router), the token must have the **“Make calls to Inference Providers”** permission enabled.
- The recommended environment variables are:
  - `HF_TOKEN` or
  - `HUGGINGFACEHUB_API_TOKEN`.

In Python you typically verify configuration with a quick check like:

```python
from huggingface_hub import HfApi
import os

api = HfApi()
print(api.whoami(token=os.environ["HF_TOKEN"]))
```

Without the proper scope, router calls will fail with 401/403 errors or, in some edge cases, appear as routing failures
(e.g. empty provider mappings).

### 2.5 Quotas, pricing, and plans

The Inference Providers pricing documentation introduces a more explicit **credit‑based model** compared to the older free tier:

- HF accounts (including PRO) receive a pool of free credits that can be used with Providers.
- Each provider advertises prices per million tokens or per compute time unit for different hardware and models.
- HF Inference appears alongside third‑party providers in pricing tables, and is billed in the same way past the free quota.

In practice, this makes it more important to:

- choose models and providers with appropriate cost/performance characteristics,
- implement client‑side limits (`max_tokens`, rate limiting),
- and monitor usage via billing dashboards, especially after migration from legacy setups that relied heavily on free serverless calls.

### 2.6 Inference Endpoints vs Inference Providers

Official docs make a clear distinction between:

- **Inference Providers**
  - shared, serverless infrastructure,
  - many models, many providers,
  - convenient for evaluation, prototyping, and moderate production workloads.
- **Inference Endpoints**
  - dedicated, autoscaling infrastructure managed for your organization,
  - suitable for strict SLAs, private models, and VPC integration.

When migrating from the legacy Serverless Inference API you typically target **Inference Providers** first. Inference Endpoints
or self‑hosted Text Generation Inference (TGI) become relevant when:

- you rely on models that are not served by any Provider, or
- you need stronger guarantees and isolation than shared Providers can offer.

## 3. Changes in model lineup and discovery

### 3.1 Provider‑specific model catalogs

With Inference Providers, model availability depends on both **model** and **provider**:

- The **Inference Models** (Supported Models) catalog lists, for each model:
  - which providers host it,
  - which tasks are supported (chat completion, embeddings, image generation, etc.),
  - token costs, context length, and additional capabilities (tools, structured outputs, streaming).
- This catalog is dynamic: providers add and remove models over time, and pricing or metrics may change.

In contrast, the old Serverless Inference API had a more opaque, centrally managed list of models that were “deployable to
Inference API (serverless)”. Users often only learned that a model was not supported when calling it.

### 3.2 Model card “Inference Providers” widget

On each model’s Hub page, an **Inference Providers** widget now shows:

- available providers (e.g. `cerebras`, `groq`, `novita`, `fireworks-ai`, `hf-inference`),
- tasks supported by each provider (e.g. `chat-completion`, `image-generation`),
- quick access to a browser playground for that provider/model combination.

This widget is the fastest way to answer:

- “Is this model available via Inference Providers at all?”
- “For which tasks and on which providers can I call it?”

If the widget shows no providers, or only tasks that do not match your use case, you cannot call that model via Providers for
that use case and must either pick another model or self‑host.

### 3.3 Typical migration outcomes for existing models

When you migrate an application that previously called various models via `api-inference.huggingface.co`, you will often find that:

- **Mainstream, recent LLMs and vision models** (Llama 3.*, GPT‑OSS, Mistral and Qwen variants, popular diffusion models) are
  available across several Providers and sometimes HF Inference.
- **Older or niche models** (for example, small community chat models like TinyLlama variants) may have **no Provider support**,
  especially for chat completion. Attempting to route them via Providers results in:
  - 404 / “model not supported for this task” errors, or
  - Python exceptions such as `StopIteration` when provider mapping is empty.
- **Classic NLP models** (BERT, MiniLM, MPNet) are often available as embeddings or token‑classification models via HF Inference,
  but not necessarily as chat models on third‑party Providers.

This means that **model strategy** becomes part of the migration work: you may need to choose new default models (for example,
Llama 3.1 Instruct) that are well supported by Providers instead of carrying over legacy models that are not.

### 3.4 Spaces and other hosted experiences

Many models still have interactive demos via **Spaces**, even when they are not available through Providers. These are useful for:

- manual testing and comparison,
- debugging prompts and behaviors,
- validating that a local deployment matches the Hub experience.

However, Spaces‑based demos are **not** a drop‑in replacement for the old Serverless Inference API. For programmatic use you must
still rely on:

- Inference Providers (router or task helpers),
- Inference Endpoints,
- or a self‑hosted service such as TGI or vLLM with an OpenAI‑compatible API.

## 4. Changes in usage patterns and API shapes

### 4.1 Basic HTTP migration: from legacy POST to OpenAI‑style chat

A minimal conceptual migration for LLM‑style usage looks like this.

**Before (legacy serverless API):**

```python
import os
import requests

HF_TOKEN = os.environ["HF_TOKEN"]
BASE = "https://api-inference.huggingface.co"

payload = {
    "inputs": "Hello, who are you?",
    "parameters": {
        "max_new_tokens": 128,
        "temperature": 0.7,
    },
}
r = requests.post(
    f"{BASE}/models/gpt2",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json=payload,
)
print(r.json())
```

**After (router, OpenAI‑compatible Chat Completions):**

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ],
    max_tokens=128,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

Key changes:

- switch from `inputs` + `parameters` to `messages` + top‑level generation knobs,
- select a **provider‑backed model** instead of a random Hub model,
- ensure the token has Inference Providers permissions.

### 4.2 Non‑chat tasks via `InferenceClient` (Python)

For non‑chat uses (classic text generation, summarization, classification, etc.) you typically migrate to `InferenceClient`:

```python
from huggingface_hub import InferenceClient
import os

client = InferenceClient(api_key=os.environ["HF_TOKEN"])

# Text generation with flat kwargs (no `parameters={}`)
out = client.text_generation(
    "Hello, who are you?",
    model="HuggingFaceTB/SmolLM3-3B",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    return_full_text=False,
)
print(out)

# Token classification (e.g. keyphrase extraction)
spans = client.token_classification(
    "Transformers and sentence embeddings are widely used in NLP.",
    model="ml6team/keyphrase-extraction-kbir-inspec",
)
```

Compared to the legacy API:

- you no longer send arbitrary JSON payloads per task,
- each helper has a narrow, documented contract and a limited set of accepted kwargs,
- routing is handled by the client and Providers.

### 4.3 JavaScript and browser usage via `@huggingface/inference`

In JavaScript and TypeScript, `@huggingface/inference` is the recommended client for both HF Inference and third‑party Providers.

Typical usage patterns are:

```ts
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN!);

// Chat completion via Providers (router)
const chat = await client.chatCompletion({
  model: "meta-llama/Llama-3.1-8B-Instruct:novita",
  messages: [{ role: "user", content: "Explain Inference Providers in 2 sentences." }],
  max_tokens: 128,
});

// Text generation via HF Inference
const gen = await client.textGeneration({
  provider: "hf-inference",          // pin provider if needed
  model: "HuggingFaceTB/SmolLM3-1.7B-Instruct",
  inputs: "Once upon a time,",
  parameters: { max_new_tokens: 64 },
});
```

Important differences from the old world:

- the client supports both **chatCompletion** and **textGeneration**, each with their own routing rules,
- provider selection can be explicit (`provider: "hf-inference"` / `"fireworks-ai"`) or automatic,
- not all providers support all tasks for all models; provider‑model‑task mismatches will throw errors like
  “task `text-generation` not supported, available tasks: `conversational` or `chat-completion`”.

### 4.4 LangChain and other framework integrations

Libraries like **LangChain** have also adapted to the new Providers world:

- Python’s `ChatHuggingFace` wrapper routes chat through the router when using Hub/Providers. It expects **message lists** and
  fails with `StopIteration` if the selected model is not provider‑backed.
- Wrapping a `HuggingFaceEndpoint` inside `ChatHuggingFace` does not force text‑generation; it still uses chat routes.
- The recommended pattern is:
  - choose a provider‑backed chat model from the Inference Models catalog,
  - pass messages (`SystemMessage`, `HumanMessage`, etc.),
  - ensure tokens have the Inference Providers permission.

In JavaScript, the LangChain `HuggingFaceInference` LLM uses the HF JS client under the hood and always calls `textGeneration`,
which can fail if the chosen provider only exposes chat routes for that model. In such cases you often:

- bypass the LangChain wrapper and use `chatCompletion` directly, or
- write a tiny custom LangChain LLM wrapper around the HF client to control the task and provider explicitly.

### 4.5 Auth, environment, and error patterns

After migration, the most common runtime issues are:

- **410/404 from `api-inference.huggingface.co`.**  
  Indicates that legacy code is still using the old base URL; fix by switching to the router or client helpers.
- **401 Unauthorized / 403 Forbidden.**  
  Typically means the token is missing the **Inference Providers** permission, is not set in the environment, or is being
  stripped by a proxy.
- **“Model not supported for task …” / `StopIteration` / empty provider mapping.**  
  Indicates that the selected model has no Provider for the requested task; choose a different model, change task, or self‑host.
- **Provider‑specific errors.**  
  Each provider may enforce its own limits on max tokens, rate limits, or available features (tools, structured outputs, etc.).

Instrumenting basic health checks and surfacing error messages in logs is essential when performing the migration in production.

## 5. Implementation patterns and migration tips

### 5.1 Strategy overview

A pragmatic migration strategy from the legacy Serverless Inference API is:

1. **Inventory all usages of `api-inference.huggingface.co`.**
   - Search code, config, and infrastructure (Terraform, Helm, etc.).
   - Note which models and tasks are being used.
2. **Classify each workload:**
   - chat / conversational,
   - classic generation (prompt in, text out),
   - classification / embeddings,
   - multi‑modal (vision, audio, image generation).
3. **Decide a target integration path per workload:**
   - Chat‑style workloads → router (OpenAI‑compatible Chat Completions) or Responses API.
   - Classic text and simple tasks → `InferenceClient` or `@huggingface/inference` task helpers (HF Inference / Providers).
   - High‑throughput or special models → Inference Endpoints or self‑hosted TGI.
4. **Map each legacy model to a provider‑backed replacement, where necessary.**
   - Prefer models that appear in the Inference Models catalog and have multiple providers.
   - For non‑provider models, plan self‑hosting or replacements.

### 5.2 Concrete steps for a simple Python service

For a typical Python service that used the legacy API directly:

1. **Switch base URL and client.**  
   Replace raw `requests` calls with the OpenAI client or `InferenceClient`.
2. **Update auth.**  
   Create a fine‑grained token with Inference Providers scope and set it as `HF_TOKEN`.
3. **Choose a provider‑backed model.**  
   Check the Inference Models catalog and the model’s card widget; pick a well‑supported model like
   `meta-llama/Llama-3.1-8B-Instruct` on a suitable provider.
4. **Update payload shape.**  
   For chat, build `messages=[...]` and use `max_tokens` instead of nested `parameters`. For task helpers, use the documented
   kwargs (`max_new_tokens`, etc.).
5. **Add smoke tests.**  
   Add tests that call the router with a tiny prompt and assert a successful response and reasonable latency.

### 5.3 Migrating a LangChain‑based application

For a LangChain‑based app:

1. **Upgrade to recent LangChain and `langchain-huggingface` versions.**
2. **Switch to `ChatHuggingFace` for chat workloads**, using a provider‑backed chat model and proper message objects.
3. **For non‑chat workloads**, keep using `HuggingFaceEndpoint` (Python) or HF clients directly with HF Inference for supported
   models, or move to Inference Endpoints / TGI.
4. **Add explicit provider and model validation**, e.g. by checking the Inference Models catalog in a CI step.

### 5.4 Migrating JavaScript / browser front‑ends

For JS front‑ends that previously used the legacy API via fetch or earlier `@huggingface/inference` versions:

1. **Adopt `@huggingface/inference` and its `InferenceClient`.**
2. **Use `chatCompletion` for chat UIs** and `textGeneration` / other task helpers for non‑chat tasks.
3. **Pin providers where needed**, especially if “auto” routing picks a provider that does not support your desired task.
4. **Handle token configuration carefully**, avoiding direct exposure of the HF token in the browser by proxying through a backend
   or using short‑lived tokens.

### 5.5 Self‑hosting as a complement

When a required model is not available via Providers, or when high control and throughput are needed, self‑hosting becomes the
primary option:

- **Text Generation Inference (TGI)** exposes an OpenAI‑compatible Messages API (`/v1/chat/completions`), so you can reuse the
  same OpenAI client used for the router.
- Other backends such as **vLLM** also expose OpenAI‑compatible servers; again, the same client code can be reused.
- In practice, many teams adopt a hybrid strategy:
  - use Providers for mainstream, interchangeable models, and
  - self‑host specialty or internal models behind the same OpenAI interface.

## 6. Limitations, caveats, and open questions

Key caveats and limitations of the new setup:

1. **Legacy endpoint is gone.**  
   There is no supported workaround to keep using `api-inference.huggingface.co`; migration is mandatory.
2. **Provider‑specific model support.**  
   Not all models are available on all providers or tasks. Model availability can change over time, so CI checks and observability
   around provider mapping are important.
3. **Task‑specific routing.**  
   Some providers expose only chat routes for a model, not `text-generation`. Using the wrong task can cause hard errors even when
   the model itself is supported.
4. **Permissions and tokens.**  
   Fine‑grained tokens with proper scopes are required; misconfigured tokens lead to confusing 401/403 errors or empty provider
   mappings.
5. **Quotas and costs.**  
   Moving from an implicit free tier to explicit credits and per‑provider pricing may introduce new cost and rate‑limit failure
   modes. Budget monitoring becomes part of reliability.
6. **Implementation churn.**  
   As the Providers ecosystem evolves, provider lists, supported tasks, and recommended models can change. Keeping client
   libraries and documentation references up to date is an ongoing task.

Open questions you may still need to answer for a specific project include:

- Which exact models and providers best match your latency, quality, and cost targets?
- Where do you draw the line between using shared Providers and deploying your own Endpoints or TGI cluster?
- How do you design fallbacks (e.g. alternate providers or local TGI) when a provider is degraded or removes a model?

## 7. References and further reading

A non‑exhaustive list of useful resources:

- **Inference Providers overview**  
  [https://huggingface.co/docs/inference-providers/index](https://huggingface.co/docs/inference-providers/index)
- **Getting started with Inference Providers**  
  [https://huggingface.co/inference/get-started](https://huggingface.co/inference/get-started)
- **Chat Completion (OpenAI‑compatible router)**  
  [https://huggingface.co/docs/inference-providers/tasks/chat-completion](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
- **HF Inference provider (successor of “Inference API (serverless)”)**  
  [https://huggingface.co/docs/inference-providers/providers/hf-inference](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- **Inference Providers pricing and HF Inference cost details**  
  [https://huggingface.co/docs/inference-providers/pricing](https://huggingface.co/docs/inference-providers/pricing)
- **Run inference on servers with `InferenceClient`**  
  [https://huggingface.co/docs/huggingface_hub/guides/inference](https://huggingface.co/docs/huggingface_hub/guides/inference)
- **Hugging Face JS Inference client**  
  [https://huggingface.co/docs/huggingface.js/inference/README](https://huggingface.co/docs/huggingface.js/inference/README)
- **Fine‑grained access tokens and security**  
  [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)
- **Welcome to Inference Providers on the Hub (blog)**  
  [https://huggingface.co/blog/inference-providers](https://huggingface.co/blog/inference-providers)
- **“A love letter to the Open AI inference client” (blog)**  
  [https://huggingface.co/blog/burtenshaw/openai-client](https://huggingface.co/blog/burtenshaw/openai-client)
- **Text Generation Inference Messages API (for self‑hosting)**  
  [https://huggingface.co/docs/text-generation-inference/messages_api](https://huggingface.co/docs/text-generation-inference/messages_api)
