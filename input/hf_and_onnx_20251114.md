---
source: "huggingface+chat+files+web"
topic: "Hugging Face and ONNX: export, optimization, and deployment patterns"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-13T19:00:37Z"
---

# Hugging Face and ONNX: export, optimization, and deployment patterns

## 1. Background and overview

ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models as computation graphs. It is a **model exchange format**, not a runtime. After export, you typically run the model using runtimes such as **ONNX Runtime**, **TensorRT**, **OpenVINO**, or other backends.

Hugging Face integrates with ONNX mainly through:

- **Optimum ONNX** – the recommended way to export and optimize models to ONNX, with tight integration to ONNX Runtime. See the Optimum ONNX overview: <https://huggingface.co/docs/optimum-onnx/en/onnx/overview>  
- **Transformers ONNX utilities** – the `transformers.onnx` package and `OnnxConfig` abstractions for model-architecture–aware export: <https://huggingface.co/docs/transformers/en/main_classes/onnx>  
- **Transformers.js + ONNX Runtime Web** – JavaScript bindings that run ONNX models in Node.js and browsers using `onnxruntime-node` and `onnxruntime-web`: <https://huggingface.co/docs/transformers.js/en/index>

A useful external companion is the **“Common AI Model Formats”** blog by ngxson, which compares ONNX to PyTorch, Safetensors, GGUF, and others, and gives a format-choice matrix: <https://huggingface.co/blog/ngxson/common-ai-model-formats>

Your own internal notes in this conversation treat ONNX as the “broad compatibility” route when ExecuTorch is not yet supported or when you require heterogeneous runtimes (CPU servers, GPU inference, edge devices, browser).

## 2. Official Hugging Face docs and tooling

### 2.1 Optimum ONNX: main export path

The **Optimum ONNX exporters** handle the export of PyTorch models to ONNX. They live under `optimum.exporters.onnx` and are documented at:

- Overview: <https://huggingface.co/docs/optimum-onnx/en/onnx/overview>  
- Export functions reference: <https://huggingface.co/docs/optimum-onnx/en/onnx/package_reference/export>  
- Usage guide “Export a model to ONNX”: <https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model>

Typical CLI usage looks like:

```bash
pip install "optimum[onnxruntime]" onnx onnxruntime

optimum-cli export onnx   --model distilbert-base-uncased-finetuned-sst-2-english   --task text-classification   distilbert_onnx/
```

Key points:

- Optimum supports many Transformers architectures and tasks (text classification, token classification, question answering, summarization, translation, text generation with past, vision2seq-lm, etc.).  
- For each architecture/task, the exporter knows how to structure graphs (encoder, decoder, decoder-with-past) and set dynamic axes.  
- Optimum provides `ORTModelFor*` classes to run ONNX models via ONNX Runtime with a Transformers-like API, often used together with `pipeline()`.

The Optimum ONNX GitHub repo shows live examples of CLI commands and optimization/quantization flows: <https://github.com/huggingface/optimum-onnx>

A classic external tutorial by Philipp Schmid, “Convert Transformers to ONNX with Hugging Face Optimum”, walks through BERT export and is still useful for conceptual understanding:  
<https://www.philschmid.de/convert-transformers-to-onnx>

### 2.2 Transformers ONNX utilities

Transformers provides a dedicated ONNX export section in the docs:

- Main ONNX page: <https://huggingface.co/docs/transformers/en/main_classes/onnx>  
- Serialization section with ONNX mention: <https://huggingface.co/docs/transformers/en/serialization>

These introduce the `transformers.onnx` package and three key configuration base classes:

- `OnnxConfig` – for encoder-only architectures.  
- `OnnxConfigWithPast` – for decoder-only architectures with KV cache.  
- `OnnxSeq2SeqConfigWithPast` – for encoder–decoder architectures.

Optimum ONNX builds on this configuration pattern; when you need to support a new architecture, you usually:

1. Implement a custom `OnnxConfig` subclass that declares input/output names and dynamic axes.  
2. Register it with Optimum’s `TasksManager`.  
3. Re-use the standard Optimum CLI or Python APIs.

### 2.3 Transformers.js and ONNX Runtime Web

**Transformers.js** uses ONNX Runtime internally to run models in JavaScript environments:

- Documentation index: <https://huggingface.co/docs/transformers.js/en/index>  
- ONNX backend details: <https://huggingface.co/docs/transformers.js/en/api/backends/onnx>

Key points:

- In **Node.js**, Transformers.js uses `onnxruntime-node`.  
- In **browsers**, it uses `onnxruntime-web` and supports WASM (CPU) and WebGPU backends.  
- You can configure ONNX backend options via `env.backends.onnx` in JS.

The docs recommend converting models to ONNX using Optimum (or supplied conversion scripts) and then loading them in Transformers.js, which hides most ONNX Runtime boilerplate.

The official ONNX Runtime docs also provide guidance on JavaScript and WebGPU usage:

- ONNX Runtime Web quickstart: <https://onnxruntime.ai/docs/get-started/with-javascript/web.html>  
- WebGPU execution provider tutorial: <https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>

### 2.4 Format-choice matrix: where ONNX fits

The Hugging Face blog post “Common AI Model Formats” by ngxson compares GGUF, PyTorch, Safetensors, and ONNX across CPU, GPU, mobile, and Apple silicon:  
<https://huggingface.co/blog/ngxson/common-ai-model-formats>

Roughly:

- ONNX is widely supported across CPU and GPU, with strong tooling via ONNX Runtime and vendor accelerators (TensorRT, OpenVINO).  
- GGUF is strong for quantized LLMs and CPU/edge targets (e.g., llama.cpp).  
- ExecuTorch `.pte` is specialized for Meta’s ExecuTorch runtime, especially for LLMs on mobile and embedded.  
- PyTorch / Safetensors remain the “authoring” formats but are not always optimal for deployment.

ONNX tends to be a good default for **cross-platform deployment**, especially when you need both server-side and on-device inference with a single exported model.

## 3. Core workflows with Hugging Face and ONNX

### 3.1 Python: Optimum ONNX + ONNX Runtime

A standard pipeline using Optimum ONNX and ONNX Runtime looks like:

1. **Install dependencies**:

```bash
pip install "optimum[onnxruntime]" onnx onnxruntime
```

2. **Export a model to ONNX**:

```bash
optimum-cli export onnx   --model distilbert-base-uncased-finetuned-sst-2-english   --task text-classification   distilbert_onnx/
```

3. **Run with ORTModel and Transformers pipeline**:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
onnx_dir = "distilbert_onnx"

model = ORTModelForSequenceClassification.from_pretrained(
    onnx_dir,
    from_transformers=False,  # already exported
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(clf("Could you assist me in finding my lost card?"))
```

4. **Optimize and quantize (optional but recommended)**:

- Use `ORTOptimizer` to apply graph optimizations.  
- Use `ORTQuantizer` for dynamic or static quantization.  
- Evaluate accuracy vs latency to find a good trade-off.

### 3.2 Transformers ONNX exporter + custom OnnxConfig

When Optimum does not support a model/task out of the box, you can fall back to:

- The Transformers ONNX docs and APIs: <https://huggingface.co/docs/transformers/en/main_classes/onnx>  
- The Optimum configuration-classes docs: <https://huggingface.co/docs/optimum-onnx/en/onnx/package_reference/configuration>

Pattern:

1. Implement a custom `OnnxConfig` that declares inputs, outputs, and dynamic axes.  
2. Register this config for your architecture.  
3. Use Optimum’s main export function or CLI with your custom task.

This allows most encoder-style and some decoder-style models to be exported without modifying the core libraries.

### 3.3 JavaScript: Transformers.js with ONNX

In JavaScript, you can use `@huggingface/transformers` with ONNX-backed pipelines:

```js
import { pipeline, env } from "@huggingface/transformers";

// Optional: configure local caching
env.localModelPath = "/models";
env.allowRemoteModels = true;

// Inspect ONNX backend options
console.log(env.backends.onnx);

// Load an ONNX-converted model (served from HF Hub or locally)
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

const embeddings = await extractor("Hello ONNX!");
console.log(embeddings);
```

For browsers:

- Use `device: "webgpu"` (where supported) to enable WebGPU acceleration.  
- For CPU-only, configure multi-threaded WASM in `env.backends.onnx` and ensure your site uses COOP/COEP headers so that `crossOriginIsolated` is `true`.

Community and HF articles (for example, guides on Transformers.js v3) show complete Next.js and SPA examples that run ONNX models fully client-side.

## 4. Architectures and task patterns

### 4.1 Supported architectures and how to check support

To check whether Optimum ONNX supports your model:

- Read the Optimum ONNX overview and “export a model” guide for the current list of supported tasks and architectures:  
  - <https://huggingface.co/docs/optimum-onnx/en/onnx/overview>  
  - <https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model>
- Check the Transformers ONNX docs for the list of architectures that already have `OnnxConfig` implementations:  
  - <https://huggingface.co/docs/transformers/en/main_classes/onnx>

If your model architecture is not listed, a common workflow is:

1. Search for the model name plus “ONNX export” in the HF docs, Hub, and forums.  
2. Look for existing Optimum ONNX examples, Spaces, or community repos.  
3. If none exist, plan for a custom `OnnxConfig` and possibly a component-wise export (vision, text, multimodal).

### 4.2 Decoder-only LLMs (`text-generation-with-past`)

For decoder-only LLMs (GPT-2 style, Llama-style, etc.), Optimum ONNX exposes the task `text-generation-with-past`. Export usually produces:

- `decoder_model.onnx`  
- `decoder_with_past_model.onnx`

The second graph accepts and returns KV cache tensors so that you can run token-by-token generation efficiently.

In many on-device LLM scenarios your internal guidance suggests:

- Use **Transformers ExecuTorch** or **Optimum ExecuTorch** first when targeting ExecuTorch runtimes.  
- Use **Optimum ONNX** (`text-generation-with-past`) when you need ONNX Runtime, TensorRT, or other ONNX-based runtimes, or when ExecuTorch is not yet ready for a given architecture.

### 4.3 Vision encoder–decoder models (TrOCR / Donut / Vision2Seq)

For models like TrOCR and Donut that implement `VisionEncoderDecoderModel`, Optimum ONNX provides the task `vision2seq-lm`. A typical command:

```bash
optimum-cli export onnx   --model microsoft/trocr-base-printed   --task vision2seq-lm   trocr_onnx/
```

This usually creates:

- `encoder_model.onnx`  
- `decoder_model.onnx`  
- `decoder_with_past_model.onnx`

At inference time, you can use `ORTModelForVision2Seq` (or similar Optimum classes) with the corresponding `Processor`/`FeatureExtractor` and `Tokenizer` to run OCR or document-understanding tasks.

### 4.4 Vision–language models (VLMs)

For VLMs such as LLaVA, BLIP-2, PaliGemma, Idefics2, Qwen2-VL, and Qwen2.5-VL, export support is more fragmented. In practice, two approaches appear:

- **Single-task export (easy path)**: when a VLM is close to `VisionEncoderDecoderModel`, `vision2seq-lm` sometimes works out of the box.  
- **Component-wise export (robust path)**:
  - Export the vision encoder and any projector/Q-Former as `vision_encoder.onnx`.  
  - Export the token embedding or text encoder as `embed_tokens.onnx` or `text_encoder.onnx`.  
  - Export the decoder with KV cache via `text-generation-with-past`.  
  - Write a small Python or JS wrapper to run:
    1. Vision encoder on the image.  
    2. Decoder in a token-by-token loop, seeded by image features and text prompt.  

Your internal notes show this pattern applied to Qwen2-VL and Qwen2.5-VL, and it generalizes to many modern VLMs that do not have first-class Optimum exporters yet.

### 4.5 Unsupported or custom architectures

When Optimum ONNX does not yet support an architecture, the official docs suggest:  

1. Creating a custom `OnnxConfig` that describes I/O and dynamic axes.  
2. Registering it with Optimum’s `TasksManager`.  
3. Using Optimum’s main export function or CLI with your custom task.

If this is still too complex, a more brute-force but sometimes effective approach is to:

- Export only parts of the model (e.g., encoder only) using `torch.onnx.export`.  
- Wrap those ONNX models in a higher-level Python or JS application that wires inputs/outputs manually.

## 5. Implementation patterns and practical tips

### 5.1 ONNX export best practices

Common best practices from HF docs, ONNX Runtime docs, and community tips:

- **Pin versions** of `transformers`, `optimum`, `onnx`, and `onnxruntime` when debugging issues; change one component at a time.  
- Prefer **Optimum CLI** or `ORTModelFor*` APIs over raw `torch.onnx.export` unless you absolutely need low-level control.  
- Use **dynamic axes** for batch size, sequence length, and image dimensions where appropriate, so you don’t need separate exports for each input shape.  
- After export, **compare outputs** between the original PyTorch model and the ONNX model on multiple test inputs (logits, log-probs, or task-level metrics).  
- If you quantize, run a **post-quantization evaluation** (e.g., accuracy, BLEU, Rouge, F1) before deploying to production.

### 5.2 Choosing between ExecuTorch and ONNX

Your internal guidance suggests this decision tree:

1. If the target runtime is **ExecuTorch** on mobile/embedded for decoder-only LLMs → start with Transformers’ ExecuTorch export or Optimum ExecuTorch.  
2. If the target is **general-purpose servers or edge devices** where ONNX Runtime or TensorRT is available → Optimum ONNX is often the best balance of maturity and portability.  
3. If you need **browser or Node.js** deployments → plan for **Transformers.js + ONNX Runtime Web** using ONNX models.  
4. If you need maximum portability across multiple vendors and platforms, default to ONNX and treat format-specific paths (GGUF, MLX, ExecuTorch) as specialized spin-offs.

### 5.3 Web / Transformers.js performance tips

Key points for ONNX in web environments:

- Prefer **WebGPU** where possible (`device: "webgpu"`), especially for embedding models and small LLMs.  
- For WASM CPU, enable **multi-threading** via correct HTTP headers (COOP/COEP) so `crossOriginIsolated` is true and ONNX Runtime Web can use multiple threads.  
- Use **smaller or quantized models** (e.g., `*-small-*`, bge-small, all-MiniLM variants) to keep inference time acceptable.  
- Re-use a single pipeline instance and **batch requests** instead of creating many concurrent sessions.  
- Cache models locally via `env.localModelPath` and `env.cacheDir` to avoid repeated downloads and initialization.

### 5.4 Debugging export and runtime issues

When something goes wrong with ONNX export or inference:

1. **Classify the failure**:
   - Export-time: Optimum / `torch.onnx.export` errors, unsupported ops, shape mismatches.  
   - Runtime: ONNX Runtime errors, invalid shapes, missing initializers, incompatible execution provider.  
2. **Simplify**:
   - Use a smaller model of the same architecture.  
   - Export encoder-only first, then add decoder.  
3. **Inspect configs**:
   - Double-check `OnnxConfig` or task settings (input names, dynamic axes, opset).  
4. **Version-bisect**:
   - Try upgrading to the latest Optimum/Transformers/ONNX Runtime.  
   - If the issue appeared after an upgrade, roll back one library at a time to find the minimal breaking change.  
5. **Search and ask**:
   - Use targeted queries like:  
     - `site:huggingface.co/docs/optimum-onnx "export a model to ONNX" "<error>"`  
     - `site:discuss.huggingface.co Optimum ONNX "<model name>" "<error>"`  
     - `site:github.com "huggingface/optimum-onnx" "<error>"`  
   - Many architectures already have community threads or issues documenting workarounds.

## 6. Limitations, caveats, and open questions

- ONNX is only as capable as the **runtime and execution provider** you choose; some backends lack support for newer operators or data types.  
- Dynamic shape support (long sequences, variable image sizes) can cause performance and memory surprises if not configured carefully.  
- VLMs and very new architectures often require **manual or component-wise export**; official exporters may lag behind HF Hub releases.  
- Quantization (especially aggressive 4-bit schemes) can hurt accuracy, especially on generative and multimodal tasks; always validate.  
- The ONNX + HF ecosystem evolves quickly (new opsets, new execution providers, new model families). Documentation and examples may lag; community posts and GitHub issues are often the fastest way to discover current best practices.

Open questions that often remain project-specific:

- How to standardize ONNX I/O conventions for vision–language models to reduce glue code.  
- When to favor ONNX vs GGUF vs ExecuTorch for each combination of hardware, latency target, and model family.  
- How much to invest in custom kernels or EP configuration vs simply picking smaller or more deployment-friendly models.

## 7. References and links

### 7.1 Official Hugging Face docs

- Optimum ONNX overview: <https://huggingface.co/docs/optimum-onnx/en/onnx/overview>  
- Optimum ONNX export usage guide: <https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model>  
- Optimum ONNX export function reference: <https://huggingface.co/docs/optimum-onnx/en/onnx/package_reference/export>  
- Optimum ONNX configuration classes: <https://huggingface.co/docs/optimum-onnx/en/onnx/package_reference/configuration>  
- Transformers ONNX main page: <https://huggingface.co/docs/transformers/en/main_classes/onnx>  
- Transformers serialization (ONNX section): <https://huggingface.co/docs/transformers/en/serialization>  
- Transformers.js docs: <https://huggingface.co/docs/transformers.js/en/index>  
- Transformers.js ONNX backend: <https://huggingface.co/docs/transformers.js/en/api/backends/onnx>

### 7.2 Blog posts and external tutorials

- Hugging Face blog – “Common AI Model Formats” (ONNX vs GGUF, PyTorch, Safetensors): <https://huggingface.co/blog/ngxson/common-ai-model-formats>  
- Philipp Schmid – “Convert Transformers to ONNX with Hugging Face Optimum”: <https://www.philschmid.de/convert-transformers-to-onnx>  
- ONNX Runtime + Hugging Face landing page: <https://onnxruntime.ai/huggingface>  
- ONNX Runtime Web JavaScript quickstart: <https://onnxruntime.ai/docs/get-started/with-javascript/web.html>  
- ONNX Runtime WebGPU tutorial: <https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>

### 7.3 Community and ecosystem links

- ngxson’s HF profile and posts (including the format-choice matrix): <https://huggingface.co/ngxson>  
- Example community articles and notes on Transformers.js v3 and WebGPU (searchable via `site:huggingface.co` and `site:zenn.dev` / `site:note.com` for local-language resources).

### 7.4 Internal notes (from this conversation)

- Internal spec on exporting LLM/VLM models into ExecuTorch or ONNX, including when to prefer each path.  
- Internal ONNX VLM conversion tips (vision2seq-lm vs component-wise export) for models like Qwen2-VL / Qwen2.5-VL.  
- Internal ONNX + Transformers.js + WebGPU tips (model choice, batching, caching, and threading) for browser deployments.

These internal documents should be kept alongside this KB file in your own knowledge base so that you can cross-reference project-specific conventions with the official Hugging Face and ONNX documentation.
