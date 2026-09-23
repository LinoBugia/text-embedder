---
source: "huggingface+chat+files+web"
topic: "Hugging Face Inference Providers"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T06:19:40.544256+00:00"
---

# Hugging Face Inference Providers

## 1. Background and overview

### 1.1 What Inference Providers are

Hugging Face Inference Providers are a shared, serverless inference layer that gives you access to hundreds of models across many third‑party providers (Groq, Cerebras, SambaNova, Together, Novita, Z.ai, and others) plus Hugging Face’s own HF Inference backend, all behind one API surface instead of many separate vendor APIs. They are exposed primarily through the Hugging Face router and integrated into the official Python and JavaScript client SDKs, so you can call models with minimal setup and without managing your own GPU infrastructure.  
The Inference Providers overview page emphasizes that the goal is to simplify access to cutting‑edge models and normalize differences in provider APIs and reliability under a single, consistent interface. [Hugging Face Inference Providers overview](https://huggingface.co/docs/inference-providers/index)

### 1.2 Router, providers, and tasks

At a high level, three concepts matter:

- **Router**  
  - Base URL: `https://router.huggingface.co/v1`  
  - Exposes an **OpenAI‑compatible Chat Completions / Responses API** for LLMs and VLMs.  
  - You send a `model` name and a `messages` array; the service chooses or contacts the configured provider and model deployment behind the scenes.

- **Providers**  
  - Each provider (Groq, Z.ai, HF Inference, etc.) offers a catalog of models and supported tasks such as `chat-completion`, `image-generation`, `embeddings`, `audio-to-text`, and more.  
  - Provider pages list capabilities and link back to the central pricing page.  
  - Example: the Z.ai provider page summarizes their GLM‑based models and points back at the pricing docs for current costs. [Z.ai provider](https://huggingface.co/docs/inference-providers/providers/zai-org)

- **Tasks**  
  - Inference Providers organize APIs by **task** (chat completion, zero‑shot classification, image generation, speech‑to‑text, etc.).  
  - Task pages document input/output schemas and examples; these are used by the client SDKs when you call helper methods like `text_generation`, `automatic_speech_recognition`, or `zero_shot_classification`. [Tasks index](https://huggingface.co/docs/inference-providers/tasks/index)

You can think of this stack as: **your app → router / task helpers → provider → model**.

### 1.3 Relationship to the legacy Serverless Inference API

Historically, Hugging Face exposed a **Serverless Inference API** at `https://api-inference.huggingface.co/models/{model_id}` that tried to support many tasks via a single endpoint and a generic JSON schema (`inputs` plus optional `parameters`). Your local migration notes capture that this endpoint is now decommissioned: requests to `api-inference.huggingface.co` return permanent errors and instruct you to use `https://router.huggingface.co` instead.   

In the new architecture:

- The old serverless infrastructure has been folded into the **HF Inference** provider, described in the docs as “the serverless Inference API powered by Hugging Face,” and explicitly marked as the successor to “Inference API (serverless).” [HF Inference provider](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- You no longer call `api-inference.huggingface.co` directly. Instead, you either:
  - use the **router** with OpenAI‑style chat completions for LLMs and VLMs, or  
  - use **task‑specific helpers** in `InferenceClient` / `@huggingface/inference` that internally route to a provider (including HF Inference).   

A key conceptual shift is that not every Hub model is automatically available as a serverless endpoint. You must now check **which provider–task combinations exist** for a given model and design around that reality.   

### 1.4 When to use Inference Providers vs other options

You typically reach for Inference Providers when:

- You want **zero infrastructure management** and are comfortable with shared, multi‑tenant deployments.
- You need to **try multiple providers or models quickly**, without writing new integration code for each vendor.
- You are building prototypes, side projects, or moderate production workloads where provider‑managed infrastructure is sufficient.

Other options in the HF ecosystem remain important:

- **Inference Endpoints**: dedicated, autoscaling deployments for a single organization; better for strict SLAs, private models, and VPC integration.   
- **Self‑hosted Text Generation Inference (TGI) / Text Embeddings Inference (TEI)**: when you need low‑level control over hardware, scheduling, and scaling or when models are not served by any provider.  
- **Local inference (Transformers, GGUF + Ollama, etc.)**: when you need full control and offline capability, usually combined with virtualized or bare‑metal GPUs.

Inference Providers complement these options rather than replacing them entirely.

---

## 2. Official docs, architecture, and provider catalog

### 2.1 Core documentation layout

The Inference Providers docs are organized around:

- **Overview page** — what Providers solve, partner list, high‑level benefits. [Overview](https://huggingface.co/docs/inference-providers/index)   
- **Pricing and Billing** — free credits, pay‑as‑you‑go model, and how routing vs custom provider keys affect who bills you. [Pricing](https://huggingface.co/docs/inference-providers/pricing)   
- **Tasks API reference** — per‑task guides (chat completion, zero‑shot classification, image editor, etc.). [Tasks index](https://huggingface.co/docs/inference-providers/tasks/index)   
- **Provider pages** — one page per infrastructure partner (Groq, Z.ai, Hyperbolic, etc.), listing supported tasks and pointing to their own docs.   
- **Guides** — practical tutorials like your first API call, building your first app, structured outputs, and function calling.   
- **Hub integration and usage insights** — how Providers appear in model cards, widgets, and billing dashboards.    

Your local spec `hf_router_langchain_vectorize_migration_consolidated_v1` already collects many of these URLs and positions Inference Providers alongside LangChain and Vectorize as core pieces of your stack.   

### 2.2 Provider‑specific model catalogs and Hub widgets

Key discovery tools:

- **Inference Models catalog** — a central table of provider‑backed models, tasks, context limits, and pricing signals; it is linked from the docs and the Hub.   
- **Model card “Inference Providers” widget** — each model’s page includes a widget showing:
  - which providers host that model,
  - which tasks each provider supports,
  - quick‑launch playgrounds for testing calls in the browser.

This is your fastest way to answer:

- “Is this model available via Inference Providers?”  
- “Which provider should I pick for this task?”  
- “Roughly what will it cost and what context length do I get?”   

### 2.3 HF Inference as a provider

The **HF Inference** provider is the direct successor of the old “Inference API (serverless)” service. The provider page underscores that HF Inference *is* the serverless Inference API, re‑exposed through the new Providers architecture. [HF Inference](https://huggingface.co/docs/inference-providers/providers/hf-inference)   

Important details:

- You use it like any other provider by setting `provider="hf-inference"` in `InferenceClient`, or by selecting an HF Inference‑backed model in the router.
- Supported tasks include ASR, text generation, translation, and others; each task section includes Python, JS, and cURL examples.  
- HF Inference is billed through the same credit‑based system as third‑party providers (see section 3).   

---

## 3. Pricing, billing, and usage breakdown

### 3.1 Credit model and plans

The Pricing and Billing docs describe how Inference Providers use a **credit‑based model**:

- Every Hugging Face account receives a monthly credit allocation (e.g., $0.10 for free users, $2 for PRO users, and per‑seat credits for Team/Enterprise plans).   
- Credits apply automatically to eligible Provider usage routed through Hugging Face.   
- Once credits are exhausted, you pay per‑usage according to each provider’s rates (tokens, images, or time). Hugging Face notes that it does **not add markup** beyond what providers charge.   

Your internal migration note emphasizes that this shift from a generous, opaque free tier in the legacy Serverless API makes it more important to:

- choose models and providers with cost/performance in mind,  
- enforce client‑side limits (`max_tokens`, rate limiting),  
- and monitor usage carefully after migration.   

### 3.2 Routed billing vs custom provider keys

The pricing guide distinguishes two billing modes:   

- **Routed by Hugging Face**  
  - You only need an HF account and token.  
  - Billing is consolidated on your HF account.  
  - HF credits apply where eligible.  
  - You do not need separate accounts with each provider.

- **Custom provider key**  
  - You configure your own provider‑specific API key in the HF settings.  
  - The provider bills you directly; HF credits do not apply.  
  - This is useful if you already have contracts with a provider but still want the routing and Hub integration benefits.

You can mix both approaches in one organization (for example, route some workloads through HF billing and others via direct provider billing).

### 3.3 Usage breakdown and dashboards

The **Inference Providers Usage Breakdown** changelog entry describes a settings UI where you can see:   

- your usage over the last month,  
- broken down by **model** and **provider**,  
- both for individual accounts and organizations (with an aggregated view over team members).   

In your own workflow, this is the main tool for verifying that migrations away from the legacy API have the expected spend profile.

---

## 4. API shapes and client SDKs

### 4.1 Router: OpenAI‑compatible Chat Completions

For chat and VLM use cases the recommended path is the **router’s OpenAI‑compatible Chat Completions API**:   

- Base URL: `https://router.huggingface.co/v1/chat/completions`
- POST body (simplified):

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct:novita",
  "messages": [
    {"role": "user", "content": "Say hi"}
  ]
}
```

- Response shape: matches OpenAI’s `chat/completions` schema; you typically read `choices[0].message.content` for the generated text.

Your Flutter troubleshooting note documents several common pitfalls:   

- **Wrong URL shape** — putting the model in the path (e.g., `.../<model>/v1/chat/completions`) yields 404; the model must live in the JSON body.  
- **Mixing payload formats** — sending `{"inputs": ...}` to the router is invalid because that schema belongs to the old Serverless API; router expects a `messages` array.  
- **Model availability assumptions** — some models are **router‑only** and not deployed on HF Inference serverless; calling them via `api-inference.huggingface.co/models/{repo}` legitimately returns 404 even though the Hub page exists.  
- **Missing token scopes** — router calls require a Hugging Face **personal access token** with “Inference Providers” permissions, passed as `Authorization: Bearer <HF_TOKEN>`.

A minimal `curl` smoke test from your note:

```bash
curl -s -X POST https://router.huggingface.co/v1/chat/completions   -H "Authorization: Bearer $HF_TOKEN"   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct:novita",
    "messages": [{"role":"user","content":"Say hi"}]
  }'
```

### 4.2 Python: `InferenceClient` and task helpers

The `huggingface_hub.InferenceClient` is the unified Python client for Inference Providers, HF Inference, Inference Endpoints, and some legacy paths.   

Key patterns:

- Instantiate with a token and, optionally, a pinned provider:

```python
from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="hf-inference",  # or "groq", "zai-org", etc., or omit to auto-route
)
```

- Use **task‑specific helpers** with **flat keyword arguments**, not nested `parameters`:

```python
resp = client.text_generation(
    "Tell me a dad joke.",
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)
print(resp)
```

- For non‑chat tasks you switch helpers:
  - `automatic_speech_recognition(...)`  
  - `image_to_image(...)` / `text_to_image(...)`  
  - `zero_shot_classification(...)`  
  - and others documented in the tasks pages. [HF Inference ASR example](https://huggingface.co/docs/inference-providers/providers/hf-inference)   

Your migration spec stresses that older patterns like `parameters={"max_new_tokens": 32}` or `max_length` must be removed; they cause runtime `TypeError` with the modern client.   

### 4.3 JavaScript: `@huggingface/inference` and other clients

In JS/TS, the `@huggingface/inference` SDK exposes a similar set of helpers to call Providers directly from browser or Node environments. Combined with the Hub API and provider metadata, it lets you dynamically list available models, pick a provider, and call tasks like image generation or zero‑shot classification without manually constructing HTTP requests.   

Additionally, there are focused integrations such as:

- **VS Code Copilot integration** — a guide shows how to add Hugging Face as a Copilot provider so that Copilot Chat can route requests through Inference Providers (for example to Kimi K2 or DeepSeek V3.1). [VS Code guide](https://huggingface.co/docs/inference-providers/guides/vscode)   

### 4.4 Hub API: listing models and providers

The **Hub API** section documents endpoints and SDK functions to:

- list models that are backed by Providers,  
- check the `providerId` and status (`staging` vs `live`) for each provider–model combination,  
- integrate this data into your own dashboards or internal catalogs. [Hub API](https://huggingface.co/docs/inference-providers/hub-api)   

Your consolidated spec uses this to reason about how LangChain, vector stores, and Inference Providers fit together in your platform.   

---

## 5. Migration from the legacy Serverless Inference API

### 5.1 What changed and why it matters

Your migration document lays out the situation clearly:   

- `https://api-inference.huggingface.co` has been **fully decommissioned**; it returns permanent errors such as 404/410 with a message telling you to use `https://router.huggingface.co/hf-inference` instead.
- Any code that still targets this base URL must be considered **broken** and scheduled for migration.
- Hugging Face encourages users to adopt Inference Providers (HF Inference plus third‑party providers) via the router and client SDKs.

### 5.2 Mapping old patterns to new ones

Concrete mappings:

- **Old: Serverless text generation**  
  - URL: `POST https://api-inference.huggingface.co/models/{model_id}`  
  - Body: `{"inputs": "text", "parameters": {...}}`  
  - Response: depends on task; for text generation often a list with `generated_text` keys.

- **New: Router chat completion**  
  - URL: `POST https://router.huggingface.co/v1/chat/completions`  
  - Body: `{"model": "your-model-id[:provider_or_suffix]", "messages": [...]}`  
  - Response: OpenAI‑style chat completion; read `choices[0].message.content`.

- **New: InferenceClient helpers**  
  - Use task‑specific methods; no `parameters` dict; pass flat kwargs.

For example, your Flutter note shows how to switch from a misconfigured router call to a correct one by:   

- fixing the URL to `/v1/chat/completions`,  
- moving the model into the JSON body,  
- using `messages` instead of `inputs`,  
- and checking `choices[0].message.content` on success.

### 5.3 Provider and model availability changes

Because Inference Providers make model availability **provider‑aware** and **task‑aware**, migrations sometimes reveal that models you previously used on Serverless are no longer available (or not available for the same tasks). Your notes recommend:   

- checking the Inference Providers widget on each critical model’s card,  
- identifying alternative providers or replacement models when necessary,  
- and documenting which combinations are “blessed” for your stack.

This is especially important when migrating RAG or agent systems that depend on specific reasoning or tool‑use capabilities.

---

## 6. Implementation patterns and recipes

### 6.1 Direct HTTP / cURL in small apps

For small scripts or language‑agnostic environments you can call the router directly with HTTP. The key steps are:

1. Create a Hugging Face access token with “Inference Providers” permission.  
2. Set `HF_TOKEN` in your environment.  
3. Use `curl` or your language’s HTTP client to POST to `/v1/chat/completions` with a `model` and `messages`.   

This pattern is useful for quick testing, non‑SDK environments (e.g., shell scripts), or introducing Providers into existing systems incrementally.

### 6.2 Mobile and Flutter clients

Your Flutter guidance shows how to build a Dart wrapper that:   

- held a `HuggingFaceAPI` class with an `apiKey` field,  
- used `http.post` to call the router chat endpoint,  
- serialized a `messages` array with a single user message,  
- and returned the `choices[0].message.content` string on success.

Common mistakes (now documented in your note) include using the wrong URL, mixing payloads from Serverless, and not handling non‑200 responses.

### 6.3 Python backends: Providers + LangChain

Your LangChain integration KB explains how to connect LangChain to Inference Providers via the `langchain_huggingface` partner package:   

- **`ChatHuggingFace`** uses the router under the hood for chat and can take a model ID that points to a provider‑backed model.  
- **`HuggingFaceEndpoint`** can route to Providers or Inference Endpoints for non‑chat generation tasks.  
- **`HuggingFaceEmbeddings` / TEI integrations** can complement Providers when building RAG systems (see your RAG KB).   

Combined with your router + vectorization spec, this creates a stack where:

- providers serve LLMs,
- TEI or other engines serve embeddings,
- a vector DB (Milvus, Redis, etc.) handles retrieval,
- and LangChain orchestrates prompts, retrieval, and tools.

### 6.4 Building end‑to‑end apps with multiple providers

Hugging Face’s guides on **“Your First Inference Provider Call”** and **“Building Your First AI App with Inference Providers”** illustrate multi‑step flows: transcribing audio with one provider, summarizing with another, and deploying the result as a Space.   

Your own plan is aligned with this pattern:

- treat Providers as a pool of capabilities (e.g., ASR, summarization, tool‑capable chat),  
- pick per‑task providers depending on latency, cost, and quality,  
- and keep application code mostly provider‑agnostic by using the router and `InferenceClient` abstractions.

### 6.5 Structured outputs and function calling

Guides on **structured outputs** and **function calling** show how to:   

- ask models to emit JSON that matches a given schema,  
- define functions or tools for models to call (with arguments),  
- and build simple agents on top of Providers, without needing a separate agent framework.

These capabilities are critical when you later connect Providers to RAG pipelines or to MCP‑style tool ecosystems (as summarized in your separate agents KB).   

---

## 7. Limitations, caveats, and open questions

### 7.1 Service limitations and gotchas

Key limitations from docs and your notes:

- **Not all Hub models are provider‑backed** — some repos have no Providers widget entries; they can’t be used via Inference Providers.   
- **Task coverage varies by provider** — a model may support `chat-completion` on one provider and only `text-generation` or no task at all on another. Always check the widget and model catalog.   
- **Rate limits and quotas** — both Hugging Face and providers enforce limits; you need retry logic and monitoring.  
- **Latency variance** — performance depends on provider hardware, queue length, and model size; provider selection (e.g., `:fastest` suffix) can help, but empirical testing is still necessary.   
- **Evolving APIs** — Providers and tasks are actively evolving; your own specs note that keeping SDK versions and integration libraries up to date is part of ongoing maintenance.   

### 7.2 Security and compliance

While not unique to Inference Providers, security considerations include:

- managing HF tokens securely (no hard‑coding in repos, using fine‑grained scopes, rotating tokens),  
- understanding provider‑side data retention and logging policies for compliance,  
- and, where necessary, preferring Inference Endpoints or self‑hosted deployments for stricter data control.   

### 7.3 Open questions in your stack

Your existing specs highlight some open design questions, for example:   

- when to pin explicit providers vs using routing hints like `:fastest` or `:cheapest`,  
- how aggressively to cache results or add client‑side guardrails to control cost,  
- how quickly ecosystem tools (LangChain, agent frameworks, MCP servers) will track new Provider features like function calling and structured outputs.

These are the levers you can tune as you gain more production experience with Inference Providers.

---

## 8. References / Links

### 8.1 Core docs

- Inference Providers overview:  
  [https://huggingface.co/docs/inference-providers/index](https://huggingface.co/docs/inference-providers/index)
- Pricing and Billing:  
  [https://huggingface.co/docs/inference-providers/pricing](https://huggingface.co/docs/inference-providers/pricing)
- HF Inference provider:  
  [https://huggingface.co/docs/inference-providers/providers/hf-inference](https://huggingface.co/docs/inference-providers/providers/hf-inference)
- Tasks and API reference:  
  [https://huggingface.co/docs/inference-providers/tasks/index](https://huggingface.co/docs/inference-providers/tasks/index)
- Your first Inference Provider call:  
  [https://huggingface.co/docs/inference-providers/guides/first-api-call](https://huggingface.co/docs/inference-providers/guides/first-api-call)
- Build your first AI app with Inference Providers:  
  [https://huggingface.co/docs/inference-providers/guides/building-first-app](https://huggingface.co/docs/inference-providers/guides/building-first-app)
- Function calling guide:  
  [https://huggingface.co/docs/inference-providers/guides/function-calling](https://huggingface.co/docs/inference-providers/guides/function-calling)
- Structured outputs guide:  
  [https://huggingface.co/docs/inference-providers/guides/structured-output](https://huggingface.co/docs/inference-providers/guides/structured-output)

### 8.2 Usage, hub integration, and partners

- Inference Providers usage breakdown:  
  [https://huggingface.co/changelog/inference-providers-usage-breakdown](https://huggingface.co/changelog/inference-providers-usage-breakdown)
- Hub integration & settings:  
  [https://huggingface.co/docs/inference-providers/hub-integration](https://huggingface.co/docs/inference-providers/hub-integration)
- Hub API (model/provider metadata):  
  [https://huggingface.co/docs/inference-providers/hub-api](https://huggingface.co/docs/inference-providers/hub-api)
- Register as an Inference Provider:  
  [https://huggingface.co/docs/inference-providers/register-as-a-provider](https://huggingface.co/docs/inference-providers/register-as-a-provider)
- Provider examples (Z.ai, Groq, Hyperbolic):  
  [https://huggingface.co/docs/inference-providers/providers/zai-org](https://huggingface.co/docs/inference-providers/providers/zai-org)  
  [https://huggingface.co/docs/inference-providers/providers/groq](https://huggingface.co/docs/inference-providers/providers/groq)  
  [https://huggingface.co/docs/inference-providers/providers/hyperbolic](https://huggingface.co/docs/inference-providers/providers/hyperbolic)

