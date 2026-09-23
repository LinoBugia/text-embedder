---
source: "huggingface+chat"
topic: "Hugging Face and ComfyUI Integration"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T00:00:00Z"
---

# Hugging Face and ComfyUI Integration Knowledge Base

## 1. Background and overview

Hugging Face (HF) and ComfyUI solve complementary problems:

- **Hugging Face Hub** is a versioned registry for models, datasets, and Spaces, with first‑party libraries like **Diffusers**, **Transformers**, and **huggingface_hub** for loading, training, and serving models.
- **ComfyUI** is a node‑based visual workflow engine for diffusion and related models, optimized for flexible graph editing, custom nodes, and low‑level VRAM control, with a built‑in HTTP API for automation.

In practice you combine them in three main ways:

1. **HF as model + training hub, ComfyUI as inference UI.**  
   - Download model weights, LoRAs, VAEs, and control adapters from Hugging Face and place them into ComfyUI’s `models/` folders or pull them via custom nodes.
   - Use HF tutorials, Spaces, and papers to choose models and training setups; run the final graphs in ComfyUI.

2. **HF Diffusers pipelines and ComfyUI graphs side‑by‑side.**  
   - Use **Diffusers** in Python for scripted experiments, quantization and offloading, and training or LoRA finetuning.  
   - Mirror the best‑performing setups in ComfyUI graphs for interactive use or multi‑user servers.

3. **HF Spaces / services in front of ComfyUI.**  
   - A Space or web app (Gradio, FastAPI, etc.) provides the user interface or orchestration.  
   - ComfyUI runs as a backend (local or remote), accessed through its HTTP API (`/prompt`, `/history`, `/view`).  
   - HF resources (Spaces GPUs, Inference Endpoints, Jobs) can host Diffusers or auxiliary tooling around a ComfyUI backend.

This document consolidates:

- Official HF and ComfyUI docs on model formats, downloading, memory optimization, and hybrid/remote inference.
- Model cards and examples for **Qwen‑Image**, **Wan 2.2**, **Flux**, **SDXL**, and video tools like **SUPIR** and **SadTalker**.
- Community‑tested patterns from your uploaded notes: Qwen‑Image ComfyUI vs Diffusers, multi‑user ComfyUI servers, Wan 2.2 LoRA training and color issues, and SadTalker/SUPIR integration and operations.
- Concrete implementation recipes and pitfalls when wiring HF resources into ComfyUI workflows.

The rest of the KB is written to be reusable without this chat: it assumes you know basic Python, GPU hardware concepts (VRAM vs system RAM), and diffusion‑model terminology but not the details of each tool.

---

## 2. From official docs, blog posts, and papers

### 2.1 Diffusers model layouts and what they mean for ComfyUI

HF **Diffusers** distinguishes between two main model layouts:

- **Multi‑folder Diffusers layout.**  
  A repo stores weights and configs in multiple folders: for example `unet/`, `vae/`, `text_encoder/`, `scheduler/`, and configuration files. Diffusers pipelines (like `StableDiffusionXLPipeline` or `QwenImagePipeline`) expect this layout by default.

- **Single‑file checkpoints.**  
  A `.safetensors` (or older `.ckpt`) file contains UNet + VAE + text encoders combined. Diffusers can often import these via conversion utilities or `from_single_file`, but pipelines are not guaranteed to understand arbitrary single‑file layouts.

ComfyUI historically prefers **single‑file checkpoints** for classic Stable Diffusion models (put in `models/checkpoints` or `models/diffusion_models`), but newer “native” integrations (Flux, Qwen‑Image, Wan, HunyuanVideo, etc.) use more **Diffusers‑like decomposed layouts**:

- Multiple `.safetensors` files for transformer/UNet, text encoders, and VAE, placed under model‑type specific folders such as:
  - `models/diffusion_models/`
  - `models/text_encoders/`
  - `models/vae/`

Key consequences:

- A **ComfyUI‑optimized single‑file package is not automatically usable in Diffusers**, and vice versa. Qwen‑Image is the canonical example: the **Comfy‑Org/Qwen‑Image_ComfyUI** repo exposes FP8 split weights tailored to ComfyUI graphs, while Diffusers expects the official multi‑folder `Qwen/Qwen-Image` repo.
- When a model offers both layouts (e.g., SD1.5 / SDXL), HF docs and ComfyUI docs typically show **different example paths**. Always follow the instructions for the specific environment (Diffusers vs ComfyUI).

### 2.2 Downloading models from the Hub

HF’s Hub docs describe several patterns for downloading models:

- **Library‑integrated loading** (Diffusers, Transformers): call `from_pretrained()` with a model id, optionally using auth tokens and local caching.
- **Direct file access**: use `huggingface_hub` or `huggingface-cli` to pull specific files or entire repos.
- **Offline / local caches**: relying on `HF_HOME` or `TRANSFORMERS_CACHE` paths for reuse across tools.

For ComfyUI integration, these translate to:

- Manual download of `.safetensors` and config files, then moving them into ComfyUI’s `models/` subdirectories.
- Automatic download via ComfyUI custom nodes that wrap `huggingface_hub` and output local file paths into the graph (see §5.1.3).

Because many modern diffusion models are tens of GB, factor in:

- **Disk layout**: fast NVMe, enough capacity for multiple model families and quantized variants.
- **Cache behavior**: both Diffusers and ComfyUI will reuse local files; repeated downloads from HF are avoidable.

### 2.3 Memory optimization and hybrid/remote inference

HF Diffusers provides detailed guidance on:

- **Offloading:** `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` move parts of the model to CPU when not in use to fit large models into smaller VRAM, at the cost of speed.
- **VAE tiling and slicing:** enabling tiling or slicing reduces peak memory during VAE encode/decode, trading runtime for reduced VRAM use.
- **Quantization:** via **bitsandbytes** or **torchao**, applying 8‑bit, 4‑bit, or float8 “weight‑only” quantization especially on the transformer/UNet reduces VRAM with moderate quality impact.
- **Efficient attention:** leveraging PyTorch SDPA/FlashAttention kernels to reduce both runtime and peak memory usage.

Diffusers’ **Hybrid Inference** docs introduce **remote VAE decode/encode** helpers that offload expensive VAE steps to managed inference services. ComfyUI’s **HFRemoteVae** node plugs into this: the graph sends encoded latents to a remote HF‑powered VAE endpoint and receives decoded images, so the main ComfyUI server can run lighter workloads.

Takeaways for ComfyUI deployments:

- Small GPUs (6–8 GB) can still run SDXL, Flux, Wan, and some video models if you combine:
  - Lower precision (FP8 / GGUF).
  - Tiled VAE, both in Diffusers and in ComfyUI’s `VAE Encode/Decode (Tiled)` nodes.
  - CPU offload and smaller batch sizes.
- Remote/hybrid VAE or other offloaded components can shift the heaviest parts of the graph to HF infrastructure at the cost of latency and networking complexity.

### 2.4 Wan 2.2 and Qwen‑Image official materials

Official docs and model cards for **Wan 2.2** and **Qwen‑Image** highlight common themes:

- **Wan 2.2** (Alibaba) is a multimodal video generation model with MoE architecture and multiple variants (5B, 14B). ComfyUI’s official Wan tutorial notes that the 5B workflow is tuned to “fit well on 8 GB VRAM” using native offloading and VRAM settings, while larger variants need more VRAM or quantization.
- **Qwen‑Image** is a 20B image diffusion transformer optimized for multilingual text rendering and editing. HF model cards and ComfyUI docs stress:
  - Separate weights for diffusion transformer, text encoders, and VAE.
  - FP8 quantized weights for ComfyUI’s native integration, enabling use on 12 GB‑class GPUs.
  - Multi‑step schedulers and efficient attention for speed/quality trade‑offs.

Official ComfyUI example pages show:

- Exact files to download (e.g., `qwen_image_fp8_e4m3fn.safetensors`, FP8 text encoder, and VAE) and where to place them in `models/`.
- Template graphs you can drag‑and‑drop into ComfyUI to run Qwen‑Image “basic” or “edit” workflows.
- Wan 2.2 text‑to‑video (T2V) and image‑to‑video (I2V) templates and their VRAM expectations.

### 2.5 Hugging Face Spaces and resource tiers

HF Spaces documentation and pricing pages define several tiers:

- **CPU Basic** (2 vCPU, 16 GB RAM, constrained storage) – suitable for small demos, lightweight image pipelines, and prototypes.
- **GPU Spaces** (e.g., L4/A10G) – needed for practical SDXL/Flux/Wan/SadTalker workloads, especially at 1024² images or video.
- Higher‑tier options like **ZeroGPU** and **dedicated inference endpoints** for lower cold‑start latency and larger throughput.

Your own SadTalker spec correctly treats **CPU Basic as demo‑only** for video workloads: building, importing heavy dependencies (PyTorch + CUDA), and generating frames on CPU will hit performance and quota limits very quickly. For any HF+ComfyUI integration that touches video or 12B‑scale transformers, plan to use **GPU Spaces** or self‑hosted GPU boxes.

---

## 3. From model cards, dataset cards, and Spaces

### 3.1 Qwen‑Image family

Key repos and their roles:

- **Qwen/Qwen-Image** (HF model card): official Diffusers‑style multi‑folder pipeline used by `QwenImagePipeline`. Good for Python‑only experiments, quantization with TorchAO, and integration into HF‑native services.
- **Comfy-Org/Qwen-Image_ComfyUI**: FP8‑quantized split weights tailored for ComfyUI native nodes and templates. Includes:
  - FP8 diffusion weights (`qwen_image_fp8_e4m3fn.safetensors`).
  - FP8 text encoder weights (`qwen_2.5_vl_7b_fp8_scaled.safetensors`).
  - FP8 VAE (`qwen_image_vae.safetensors`).

Important compat note:

- The **ComfyUI layout is not directly loadable by Diffusers**, and the Diffusers pipeline expects the official Qwen repo. Keep these as **two distinct installs**:
  - Diffusers → `Qwen/Qwen-Image`.
  - ComfyUI → `Comfy-Org/Qwen-Image_ComfyUI` plus ComfyUI’s native Qwen graph.

Multiple example pages (ComfyUI examples, wiki, community blogs) walk through:

- Downloading the FP8 files.
- Placement into ComfyUI’s `models/` subfolders.
- Using native Qwen graphs for generation and edit workflows.
- VRAM expectations for 12 GB GPUs and options for GGUF quantized variants if available.

### 3.2 Wan 2.2 family

Important model card points for **Wan 2.2** (various HF repos like `Wan-AI/Wan2.2-Animate-14B`, `Wan-AI/Wan2.2-S2V-14B`, etc.) and their integration with ComfyUI:

- The official docs describe Wan as a large‑scale MoE video model, with multiple **experts** for different noise regimes (e.g., high‑noise vs low‑noise experts).
- Wan T2V / I2V variants at **5B** are considered the “sweet spot” for 8–12 GB GPUs when combined with tiling, offload, and sometimes GGUF quantization.
- HF model cards and open‑source repos show recommended resolutions and frame rates (e.g., 720p @ 24 fps) and mention expected compute needs for training and inference.

Some community GGUF repos bundle Wan 2.2 experts into `.gguf` files plus drag‑and‑drop ComfyUI workflows. These extend Wan support to smaller GPUs via **ComfyUI-GGUF** nodes.

From Wan model cards and tutorials:

- Read **license** and **use‑case restrictions** carefully – some variants have specific commercial usage rules.
- Observe **recommended prompts** and **style tokens**; Wan often responds well to descriptive, film‑like prompts with explicit camera and motion language.
- Capture **VRAM anchors**: even with offload, high resolutions and long clips push both GPU VRAM and system RAM.

### 3.3 LoRA training Spaces and configs

For Wan and similar video models, HF hosts multiple Spaces and repos that encapsulate training configs, dataset examples, and GUI frontends:

- **Musubi‑tuner** documentation and Spaces (e.g., Wan LoRA trainer GUIs) describe dataset formats via TOML configs, with:
  - Paired **image + caption** or **video + caption** blocks.
  - Fields like `num_repeats`, `target_frames`, `frame_extraction`, and resolution/FPS constraints.
- Example configs in Spaces use the same folder layouts you outlined in your Wan 2.2 LoRA doc:
  - `dataset/images/*.jpg` + same‑name `.txt` captions for image‑only LoRA.
  - `dataset/video/*.mp4` + same‑name `.txt` for video LoRA.

Training Spaces and guides emphasize:

- **Caching latents** and **text encoder outputs** once per dataset to reduce active VRAM during training.
- Using **FP8 and block‑swap features** to make 16–24 GB GPUs workable for Wan 2.2 LoRA training.
- Starting hyperparameters around:
  - `rank r = 32, alpha = 16` (or `r=64, alpha=32` for complex identities).
  - Learning rate around `2e-4`, adjusting upwards for motion‑only LoRAs.

These details align with your own LoRA training notes and are relevant when deciding how to move LoRAs into ComfyUI (see §5.3).

### 3.4 Spaces for orchestration and demos

HF Spaces often host:

- Web UIs for model demos (image, text, video).
- Orchestrators that call other APIs, including ComfyUI, Diffusers scripts, or external services.
- Training workflows (e.g., Wan LoRA training, SDXL finetuning).

Patterns relevant to ComfyUI:

- A Space can present a **Gradio UI** that calls a self‑hosted ComfyUI backend via HTTP. This decouples the UI from the heavy graph execution.
- Conversely, a ComfyUI server can call **HF inference endpoints** or Spaces via custom Python or HTTP nodes (e.g., for VAE, captioning, or LLM‑based prompt generation).

Because Spaces are ephemeral and resource‑limited, avoid running entire ComfyUI instances inside CPU‑only Spaces; instead, use Spaces to orchestrate or pre/post‑process workflows that ultimately run on dedicated ComfyUI hosts or GPU Spaces.

---

## 4. From community discussions, GitHub, and Q&A

### 4.1 ComfyUI Hugging Face downloaders and tooling

Several community projects bridge HF and ComfyUI more directly:

- **ComfyUI Hugging Face Downloader** repositories provide custom nodes that:
  - Accept Hugging Face model ids and optional file patterns.
  - Download files or entire folders via `huggingface_hub`.
  - Output the downloaded path(s) as strings you can feed into loader nodes.

- **Hugging Face Download Model** documentation (for ComfyUI‑focused websites) describes a similar node that:
  - Wraps model downloads into a single node with an optional auth token.
  - Integrates with `CheckpointLoader`, `UNetLoader`, or other nodes by passing the model name/path downstream.

Usage pattern (see §5.1.3):

1. Configure HF token (if needed) as an environment variable or in ComfyUI’s settings.
2. In the graph, use the HF download node with a repo id like `black-forest-labs/FLUX.1-dev` or `Qwen/Qwen-Image` (for compatible formats).
3. Feed the node’s output path into ComfyUI loader nodes.

These tools keep your graphs self‑contained: dragging them into a fresh ComfyUI install will automatically pull required weights from HF Hub.

### 4.2 Qwen‑Image: ComfyUI vs Diffusers, VRAM, and clients

Your Qwen‑Image notes synthesize multiple community sources:

- **Two distinct paths:**
  - ComfyUI uses **FP8 single‑file and split weights** that are pre‑quantized and tailored to Comfy graphs. This lets 12 GB GPUs handle a 20B model by combining FP8, tiling, and offload modes.
  - Diffusers uses the official `Qwen/Qwen-Image` repo with BF16/FP16 defaults and heavier VRAM usage, expecting a multi‑folder layout.
- **Do not cross wires:** the Comfy FP8 checkpoint layout does **not** load into `QwenImagePipeline`. If you want Diffusers, stay on the official repo; if you want ComfyUI, follow the Comfy instructions.

VRAM‑saving mechanisms you highlighted for ComfyUI’s Qwen graphs:

- FP8 weights for transformer, text encoder, and VAE.
- Explicit `lowvram` / `novram` / `normalvram` / `highvram` VRAM modes in the server config, which control module offload.
- Tiled VAE encode/decode nodes that cut peak memory for large resolutions.
- Memory‑efficient attention (PyTorch SDPA / xFormers) when available.

You also documented a **minimal Python client** that:

1. Starts ComfyUI with flags such as `--lowvram --listen 127.0.0.1 --port 8188`.
2. Downloads an official Qwen‑Image workflow JSON from GitHub.
3. Modifies CLIPTextEncode nodes for positive and negative prompts, as well as sampler parameters.
4. Submits the graph to `POST /prompt`, polls `/history/{prompt_id}`, and downloads images via `/view`.

This pattern generalizes to any ComfyUI workflow and is the recommended way to script ComfyUI from Python while still sourcing models from HF.

### 4.3 Multi‑user ComfyUI servers and VRAM sizing

Your multi‑user ComfyUI server doc consolidates key community wisdom:

- **Concurrency model:**
  - A single ComfyUI process manages a queue of prompts; it runs one heavy job at a time.
  - Real concurrency is achieved by running **one ComfyUI process per GPU**, each bound to a unique port, and routing requests to the least busy worker.

- **Hardware sizing:**
  - VRAM “anchor” values for inference:
    - SD1.5 512²: 4–6 GB workable with heavy offload; 8 GB is smoother.
    - SDXL 1024²: 8–12 GB workable with optimizations; 16–24 GB preferred for refiner/ControlNet.
    - Flux full‑precision (12B): ~22 GB VRAM just for weights; 24 GB cards recommended.
    - Flux/Wan GGUF quantized: 6.8–12.7 GB depending on quant level; enables 8–12 GB GPUs.
    - HunyuanVideo and Wan 2.2 video: can be forced onto 8 GB GPUs with tiling and offload, but at a latency and system‑RAM cost.
    - SUPIR upscaling: often ~10 GB VRAM plus ≥32 GB system RAM for comfortable 512→1024 upscales.

- **Scaling patterns:**
  - **Multi‑instance fan‑out:** one process per GPU, simple router in front.
  - **MultiGPU extension:** place text encoders and VAE on a second GPU, keep UNet/DiT on the main GPU.
  - **NetDist / Distributed:** offload parts of the graph to remote workers or split batches/tiles across multiple machines.

These patterns are important when planning how HF‑sourced models (Flux, Wan, Qwen) will behave under multiple users and which quantized variants you should choose.

### 4.4 Wan 2.2 color and VAE issues in ComfyUI

Your Wan color issue note aggregates multiple bug reports and community fixes:

- **Tiled VAE decode can cause color shifts or flicker**, especially at the beginning or end of Wan 2.2 video clips.
- Fixes:
  - Replace `VAE Decode (Tiled)` with standard `VAE Decode` for Wan output where possible.
  - Confirm that the exact Wan 2.2 VAE recommended in the official docs is loaded.
  - Keep VAE decode consistent across frames; avoid mixing different VAE nodes mid‑timeline.
  - If you still see a flash, trim 0–1 seconds at the head/tail of the clip—reports show glitches concentrated there.

Regarding **ColorMatch** nodes (e.g., KJNodes “Easy Use” ColorMatch):

- Certain methods like **MVGD** tend to darken the overall image.
- Alternatives like **Reinhard**, **MKL**, or compound passes such as **HM‑MVGD‑HM** often preserve luminance better.
- The practical A/B recipe you captured:
  1. Keep your current Wan workflow.
  2. Swap tiled VAE decode for non‑tiled decode (or lower resolution / quantize to fit).
  3. Verify correct VAE.
  4. Change ColorMatch method MVGD → Reinhard; if still dark, test MKL or compound passes.

This is directly relevant to any HF Wan 2.2 model card user migrating to ComfyUI: a naive “just load the model” approach can produce unexpected color artifacts unless VAE and ColorMatch details are aligned with community best practices.

### 4.5 Wan 2.2 LoRA training pipeline

Your Wan 2.2 LoRA document pulls together Musubi‑tuner docs, HF Spaces examples, and community walkthroughs. The core points:

- **Dataset layouts:**
  - Image LoRA:
    ```
    dataset/
      images/
        0001.jpg
        0001.txt
        0002.jpg
        0002.txt
        ...
    ```
  - Video LoRA:
    ```
    dataset/
      video/
        clip_0001.mp4
        clip_0001.txt
        clip_0002.mp4
        clip_0002.txt
        ...
    ```
  - Same‑name `.txt` captions accompany each image or clip; captions are long and descriptive, often >50 words for video.

- **`dataset.toml` blocks:** define `[datasets]` entries for images and videos, with fields for resolution, FPS, repeats, and caching paths. `num_repeats` is the main lever to weight certain clips or identities more strongly.

- **Preprocessing and caching:**
  - Run VAE latent caching and text‑encoder output caching before training, using Musubi‑tuner’s dedicated scripts.
  - This reduces active VRAM needs for training and allows 16–24 GB cards to train Wan LoRAs practically.

- **Hyperparameters that work in the field:**
  - Rank/alpha of (32, 16) or (64, 32).
  - Learning rate around `2e-4`, slightly higher for motion‑only LoRAs.
  - Batch size 1 on 16 GB, 1–2 on 24 GB with FP8 and block‑swap tricks.

- **VRAM expectations:**
  - Musubi‑tuner guidance: ≥12 GB VRAM for image LoRA training, ≥24 GB for video LoRA training at comfortable resolutions.
  - Community reports: Wan I2V LoRA training is possible on 16 GB with aggressive optimizations but is slower and more fragile.

Once trained, LoRA weights are typically stored as `.safetensors`. ComfyUI can then consume them via LoRA loader nodes in Wan workflows (see §5.3).

### 4.6 SadTalker and HF Spaces considerations

Your SadTalker spec covers both ComfyUI integration and HF runtime constraints:

- **SadTalker assets:**
  - Core checkpoints (`SadTalker_V0.0.2_256/512.safetensors`, `mapping_00109/00229-model.pth.tar`) plus GFPGAN and facexlib weights.
  - Strict expectations about folder layout (`ComfyUI/custom_nodes/Comfyui-SadTalker/SadTalker/checkpoints/`, `gfpgan/weights/`, etc.).

- **ComfyUI‑Manager security levels:**
  - Manager uses a `security_level` setting; Git URL installs may require temporarily switching from `normal` to `weak` security.
  - Best practice: change to `weak` only while installing, then revert to `normal` to reduce supply‑chain risk.

- **HF Spaces CPU Basic:**
  - 2 vCPU, 16 GB RAM, and limited storage make it unsuitable for production SadTalker workloads.
  - Video generation, heavy face restoration, and deep audio processing quickly exhaust CPU and memory budgets.
  - GPU Spaces are recommended for real use; CPU Spaces are for demos and testing only.

- **Python and library versions:**
  - Many legacy nodes (SadTalker, some video helpers) expect Python 3.11 and NumPy 1.x.
  - Moving to Python 3.12/3.13 requires careful auditing; some wheels and legacy import paths break.

These policies should be considered whenever you design a HF+ComfyUI architecture for talking‑head or video workflows.

### 4.7 General community tips for installing HF models into ComfyUI

From Reddit, GitHub discussions, and documentation:

- For classic SD checkpoints from HF:
  - Download `.safetensors` and put them in `ComfyUI/models/checkpoints` or `ComfyUI/models/diffusion_models`, depending on graph expectations.
  - Use simple base workflows from official ComfyUI examples as templates; swap the checkpoint in a `CheckpointLoader` node.

- For advanced models (Wan, Qwen, Flux, Hunyuan):
  - Follow ComfyUI example pages and official tutorials that list exact files and directories.
  - For GGUF models, install **ComfyUI-GGUF** and place `.gguf` files under the appropriate `models/` folder.

- Always cross‑check:
  - Repo README / model card on HF.
  - ComfyUI docs / wiki pages.
  - Issues and discussions for known pitfalls (e.g., VAE color shifts, missing node types, security settings).

---

## 5. Implementation patterns and practical recipes

This section distills the concepts above into practical patterns you can reuse.

### 5.1 Getting HF models into ComfyUI

#### 5.1.1 Manual download via CLI or browser

For any HF model that provides a single‑file checkpoint or clear file mapping:

1. Use `huggingface-cli` or the web UI to download the relevant `.safetensors` files.
2. Place files into ComfyUI’s `models/` structure, e.g.:

   ```text
   ComfyUI/
     models/
       checkpoints/          # classic SD1.5-style ckpt/safetensors
       diffusion_models/     # modern native models (Flux, Qwen, Wan)
       text_encoders/        # CLIP, T5, etc.
       vae/                  # VAEs
       loras/                # LoRA weights
       controlnet/           # ControlNet/ControlLoRA weights
   ```

3. Reload ComfyUI (or use “Rescan Models”) so the loaders see new files.
4. Start from an official or example workflow that expects the same model family; only then swap the checkpoint or LoRA.

Manual download is straightforward and transparent, but it does not encode dependencies inside the graph itself.

#### 5.1.2 Using HF‑aware custom nodes

Install a Hugging Face download node or extension, such as a “Hugging Face Downloader” for ComfyUI. These nodes typically:

- Accept a **model id** like `black-forest-labs/FLUX.1-dev` or `Wan-AI/Wan2.2-Animate-14B`.
- Optionally accept **file filters** or **subfolder paths**.
- Download the files to a configured directory (often under `models/` or a custom download cache).
- Output the path(s) into the graph for loader nodes.

Advantages:

- Graphs become more portable: when you share or re‑use them, the download nodes reproduce the environment.
- It is easier to update to newer model versions by changing model ids or revision tags.

Caveats:

- For large models, first run on a fast connection; downloads can be dozens of GB.
- Some gated or licensed models require HF auth tokens; you must configure those outside the graph (environment variables, config files).

#### 5.1.3 Respecting layout differences (Qwen‑Image as example)

When a model exists in multiple layouts, be explicit about which one you use where:

- **Diffusers path:**

  ```python
  from diffusers import QwenImagePipeline

  pipe = QwenImagePipeline.from_pretrained(
      "Qwen/Qwen-Image",
      torch_dtype="bfloat16"
  ).to("cuda")
  ```

- **ComfyUI path:**

  - Download `qwen_image_fp8_e4m3fn.safetensors` into `models/diffusion_models/`.
  - Download `qwen_2.5_vl_7b_fp8_scaled.safetensors` into `models/text_encoders/`.
  - Download `qwen_image_vae.safetensors` into `models/vae/`.
  - Use the official Qwen‑Image ComfyUI template graph and ensure the loaders point to these files.

Do **not** try to load the ComfyUI FP8 checkpoint in Diffusers, and do not assume Diffusers’ pipeline layout is understood by ComfyUI nodes that expect a single file.

### 5.2 Driving ComfyUI graphs from Python while sourcing models from HF

Assuming models are already in place (downloaded manually or via nodes), you can script ComfyUI using its HTTP API.

Minimal pattern (simplified Qwen‑Image example):

```python
# pip install requests
import json
import time
import pathlib
import requests

COMFY = "http://127.0.0.1:8188"
WF_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/image_qwen_image.json"

OUT = pathlib.Path("qwen_out")
OUT.mkdir(exist_ok=True)

# 1) Load workflow template
wf = requests.get(WF_URL, timeout=30).json()

# 2) Patch prompt and sampler settings
for n in wf["nodes"]:
    t = n.get("type")
    title = n.get("title", "")
    if t == "CLIPTextEncode" and "Positive" in title:
        n["widgets_values"][0] = "ultra-detailed poster, English and Japanese text"
    if t == "CLIPTextEncode" and "Negative" in title:
        n["widgets_values"][0] = "blurry, lowres, artifacts"

# 3) Submit prompt
resp = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": "qwen-script"}, timeout=30).json()
pid = resp["prompt_id"]

# 4) Poll history
for _ in range(180):
    hist = requests.get(f"{COMFY}/history/{pid}", timeout=30).json()
    if hist and list(hist.values())[0].get("outputs"):
        entry = next(iter(hist.values()))
        for node in entry["outputs"].values():
            for im in node.get("images", []):
                params = {
                    "filename": im["filename"],
                    "subfolder": im["subfolder"] or "",
                    "type": "output",
                }
                img_bytes = requests.get(f"{COMFY}/view", params=params, timeout=60).content
                (OUT / im["filename"]).write_bytes(img_bytes)
        break
    time.sleep(1)

print("done; outputs in", OUT)
```

This pattern applies to any HF‑sourced model once it is installed into ComfyUI: you use HF for **weights and documentation**, and ComfyUI for **graph execution**, with Python orchestrating via HTTP.

### 5.3 Training LoRAs on HF‑documented setups and using them in ComfyUI

A typical Wan 2.2 LoRA pipeline looks like this:

1. **Design dataset following HF examples and your LoRA notes:**
   - Images or videos with paired `.txt` captions.
   - Consistent resolution and FPS for video.
   - Clear naming and folder structure (`images/`, `video/`).

2. **Configure `dataset.toml` for Musubi‑tuner or similar tools:**
   - Add `[[datasets]]` blocks for each split (images vs video).
   - Set `num_repeats` to weight important clips/identities.
   - Define target resolution, FPS, and whether this block is image or video.

3. **Pre‑cache latents and text‑encoder outputs:**
   - Run the provided caching scripts once per dataset to generate latent and text‑encoder caches.
   - Confirm logs show `is_image_dataset` or `is_video_dataset` with correct resolutions and bucket info.

4. **Train LoRA:**
   - Start with moderate hyperparameters (rank 32, alpha 16, learning rate 2e‑4).
   - Monitor validation or quick previews every epoch.
   - Use FP8 and block‑swap options if VRAM is tight (e.g., 16 GB GPU).

5. **Move LoRA to ComfyUI:**
   - Save final LoRA weights as `.safetensors` if not already.
   - Place them in `ComfyUI/models/loras/` (or the path expected by LoRA loader nodes in your workflow).
   - In ComfyUI, add a LoRA loader node inline with your Wan UNet; set the LoRA weight and trigger tokens as described by your training metadata.

6. **Integrate into graphs:**
   - Start from an official Wan 2.2 ComfyUI template (T2V/I2V).
   - Insert the LoRA loader node where the UNet is connected.
   - Keep VAE, sampler, and VRAM settings as in the template to minimize moving parts.

HF model cards and community guides for Wan 2.1/2.2 LoRA training are useful to calibrate expectations and confirm hyperparameters. ComfyUI then becomes the main environment for applying, testing, and compositing LoRAs in larger workflows.

### 5.4 Combining HF Diffusers pipelines and ComfyUI graphs

You often want both:

- **Diffusers** for scripted experiments, training, and quantization.
- **ComfyUI** for interactive experimentation and multi‑user access.

A practical split:

- Use Diffusers to:
  - Prototype architectures and new schedulers (e.g., DPMSolver++, DPM‑Solver‑SDE, etc.).
  - Experiment with quantization strategies (bitsandbytes vs TorchAO).
  - Benchmark and log settings in a reproducible Python environment.

- Port stable configurations into ComfyUI graphs by:
  - Matching schedulers where available (e.g., DPM++ multi‑step analogues).
  - Ensuring the same VAE and text encoders are used; mis‑matched VAEs are a common cause of color/contrast drift.
  - Reproducing attention and memory options (e.g., enabling memory‑efficient attention in ComfyUI when your Diffusers experiments used SDPA/FlashAttention).

Where features diverge:

- Not all Diffusers pipelines have one‑to‑one ComfyUI nodes, and vice versa.
- For some models (e.g., Qwen‑Image, Wan), ComfyUI has first‑class native graph support earlier than Diffusers; for others, Diffusers gets features first.
- Hybrid inference (e.g., remote VAE) may be easier to prototype in Diffusers and then partially port to ComfyUI via dedicated nodes.

### 5.5 Architecture patterns with HF Spaces, Inference Endpoints, and ComfyUI

Several reference architectures combine HF’s managed services with ComfyUI:

1. **Space UI, ComfyUI backend (common pattern):**
   - HF Space hosts a Gradio or web UI.
   - UI calls a private ComfyUI server via HTTPS (e.g., self‑hosted on a GPU box or as another Space).
   - Pros: easy sharing, authentication via HF, decoupled UI and backend.
   - Cons: cross‑network latency, bandwidth considerations for large outputs (videos, high‑res images).

2. **ComfyUI in a GPU Space, HF as hosting:**
   - ComfyUI is installed inside a GPU Space container.
   - Users access the ComfyUI web UI directly or via a small wrapper.
   - Pros: simple single‑box pattern using HF’s GPUs.
   - Cons: ephemeral storage and resource limits; careful management of models and caches is required.

3. **Diffusers on Inference Endpoints, ComfyUI as orchestrator:**
   - Heavy Diffusers models are served on dedicated HF Inference Endpoints.
   - ComfyUI uses custom HTTP nodes or Python nodes to call the endpoints, treating them as black‑box “super nodes” (e.g., for captioning, LLM prompting, or specialized diffusion pipelines).

In all cases, pay attention to:

- **Spaces tier:** avoid CPU Basic for production workloads; prefer GPUs.
- **Data flow:** minimize large tensor transfers across networks; offload only components that benefit most.
- **Security and tokens:** handle HF tokens, API keys, and ComfyUI exposure carefully (reverse proxies, auth, allowed origins).

### 5.6 Troubleshooting checklist for HF + ComfyUI integrations

Common failure modes and checks:

1. **Model not found in ComfyUI loader:**
   - Confirm the filename and directory under `models/`.
   - Check for `.safetensors` vs `.ckpt` extension mismatch.
   - Rescan models or restart ComfyUI.

2. **Model loads but outputs look wrong (color, contrast, artifacts):**
   - Verify you use the **recommended VAE** from the model card / ComfyUI docs.
   - Disable tiled VAE decode for problematic models (like Wan) or adjust resolutions.
   - Check ColorMatch or other post‑processing nodes for aggressive methods (e.g., MVGD vs Reinhard).

3. **Out‑of‑memory errors:**
   - Lower resolution and batch size.
   - Switch ComfyUI to `lowvram` or similar modes.
   - Enable quantized variants (FP8, GGUF) where available.
   - Consider offloading (both in Diffusers and ComfyUI) and ensure system RAM is sufficient.

4. **Custom nodes missing or failing to import:**
   - Confirm ComfyUI‑Manager security settings.
   - Check Python version and NumPy / PyTorch compat according to node README.
   - Ensure external tools like `ffmpeg` are installed for video nodes (VHS, SadTalker).

5. **HF model download failures:**
   - Verify auth tokens for gated models.
   - Respect bandwidth and storage limits on Spaces.
   - Check for file path changes in the model card (e.g., new subfolder layouts).

6. **Multi‑user queue bottlenecks:**
   - Confirm you run one ComfyUI process per GPU.
   - Implement a simple router to distribute jobs across workers by queue depth.
   - Adjust VRAM and model variant choices so common jobs fit on your hardware class.

---

## 6. Limitations, caveats, and open questions

### 6.1 Layout and compatibility fragmentation

- Single‑file checkpoints, Diffusers multi‑folder repos, FP8 split weights, and GGUF bundles all coexist.
- Not every model provides all formats; there is no universal auto‑conversion tool for every combination, especially for advanced architectures like Wan and Qwen‑Image.
- ComfyUI might get native support for a model before Diffusers (or vice versa), leading to temporary feature gaps.

When in doubt, follow:

1. The model card’s “how to use” section.
2. The corresponding ComfyUI example or tutorial.
3. Community issues/threads for the latest working recipes.

### 6.2 Quantization and precision differences

- HF’s quantization guides (TorchAO, bitsandbytes) focus on Diffusers and Transformers APIs; ComfyUI supports quantized models via extensions like GGUF loaders and custom graph nodes.
- Float8, int8, and 4‑bit schemes have different quality/latency trade‑offs, and not every model supports all of them.
- Wan, Flux, and Qwen‑Image have fast‑moving ecosystems; VRAM and quality expectations change as new quants appear.

Plan to periodically re‑evaluate your quantization choices and check for updated quants or native low‑precision variants.

### 6.3 Dependency and runtime drift

- ComfyUI custom nodes sometimes lag behind current Python, NumPy, and PyTorch versions.
- HF libraries move faster, adding new CUDA and quantization features, dropping old dependencies, or raising minimum versions.
- Running everything in a single environment (e.g., one venv for ComfyUI, Diffusers, and training) can lead to library conflicts.

Best practices:

- Use **separate virtual environments** per major component (ComfyUI, training tools, other backends).
- Pin versions for legacy nodes (e.g., Python 3.11 and NumPy 1.26.x) when necessary.
- Follow PyTorch’s official installation matrix to ensure CUDA and driver compatibility.

### 6.4 Operational constraints on HF infrastructure

- HF Spaces CPU Basic is intentionally constrained; treat it as a demo tier.
- GPU Spaces offer more power but still have storage and runtime limitations; large ComfyUI deployments might be better on your own GPU servers.
- Inference Endpoints and Jobs provide more robust SLAs but require careful cost and capacity planning.

Open questions you may revisit:

- Long‑term: will ComfyUI add first‑class Hugging Face Hub integration (tokens, model lists) in core, beyond community nodes?
- Will more models standardize on a single, compatible layout across Diffusers and UIs?
- How far will hybrid inference go—will more of the graph (beyond VAE) run on remote HF services while ComfyUI orchestrates locally?

---

## 7. References and recommended reading

### 7.1 Hugging Face docs and blog

- [Diffusers main documentation](https://huggingface.co/docs/diffusers/index)
- [Downloading models from the Hub](https://huggingface.co/docs/hub/en/models-downloading)
- [Reduce memory usage in Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Quantization with TorchAO](https://huggingface.co/docs/diffusers/en/quantization/torchao)
- [xFormers and attention optimizations](https://huggingface.co/docs/diffusers/en/optimization/xformers)
- [Hybrid Inference overview](https://huggingface.co/docs/diffusers/v0.34.0/hybrid_inference/overview)
- [Hybrid Inference VAE decode example](https://huggingface.co/docs/diffusers/v0.33.0/hybrid_inference/vae_decode)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Hugging Face pricing](https://huggingface.co/pricing)

### 7.2 Model and dataset cards

- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
- [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf)

### 7.3 ComfyUI docs and example pages

- [ComfyUI server configuration](https://docs.comfy.org/interface/settings/server-config)
- [ComfyUI server routes and API](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [Wan 2.2 video generation (official ComfyUI tutorial)](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [Qwen-Image ComfyUI native workflow example](https://docs.comfy.org/tutorials/image/qwen/qwen-image)
- [Qwen-Image ComfyUI examples](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
- [VAE Decode (Tiled) community manual](https://blenderneko.github.io/ComfyUI-docs/Core%20Nodes/Experimental/VAEDecodeTiled/)

### 7.4 GitHub projects and custom nodes

- [ComfyUI Hugging Face Downloader (example)](https://github.com/jnxmx/ComfyUI_HuggingFace_Downloader)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
- [ComfyUI_NetDist](https://github.com/city96/ComfyUI_NetDist)
- [ComfyUI-Distributed](https://github.com/robertvoy/ComfyUI-Distributed)
- [Comfyui-SadTalker](https://github.com/haomole/Comfyui-SadTalker)
- [ComfyUI-VideoHelperSuite (VHS)](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [Musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)

### 7.5 Community guides and articles

- [Running Hunyuan with 8GB VRAM and PixArt Model Support](https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and)
- [Qwen-Image ComfyUI native, GGUF, and Nunchaku workflow guide](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image)
- [Wan2.2 ComfyUI workflow complete usage guide](https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2)
- [ColorMatch node documentation (KJNodes)](https://www.instasd.com/comfyui/custom-nodes/comfyui-kjnodes/colormatch)
- Selected discussions on Wan 2.2 color glitches, VAE tiling issues, and multi‑user ComfyUI setups (issue trackers and Reddit threads).

This knowledge base is intended as a living reference: as HF and ComfyUI evolve, update model versions, quantization options, and ops guidance accordingly.
