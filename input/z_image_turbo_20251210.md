---
source: "huggingface+chat"
topic: "Z-Image-Turbo: model, repos, and practical usage (Diffusers, ComfyUI, GGUF, Colab/Kaggle)"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-12-10T00:00:00Z"
---

# Z-Image-Turbo

## 1. Background and overview

### 1.1 Position in the Z-Image family

Z-Image is a family of text-to-image and image-editing foundation models released by Alibaba’s Tongyi-MAI team. The main variants are:

- **Z-Image-Base** – the base text-to-image foundation model for general use and fine-tuning.
- **Z-Image-Turbo** – a distilled, few-step inference variant for fast text-to-image.
- **Z-Image-Edit** – an image-editing model for instruction-based editing and inpainting.

Z-Image-Turbo is currently the primary publicly available checkpoint and is the focus of this note.

Key characteristics of Z-Image-Turbo, as described in the official model card and related material:

- About **6B parameters** in total.
- Built on **Scalable Single-Stream Diffusion Transformer (S3-DiT)**.
- Uses a **few-step distillation scheme** (Decoupled-DMD + RL-style “DMDR”) so that high quality is reached in around **8 function evaluations (NFEs)**.
- Designed to achieve **sub-second inference** on high-end H800-class GPUs and to **fit comfortably within 16 GB VRAM** consumer GPUs.
- Particularly strong at **photorealistic image generation**, **English and Chinese text rendering**, and **instruction adherence**.
- Released under **Apache-2.0 license**, which is permissive and typically suitable for commercial use, subject to standard open-source compliance and local law.

### 1.2 S3-DiT architecture in one paragraph

Traditional latent diffusion models (e.g. SDXL) often treat **text** and **image latents** with partially separate streams. Z-Image instead uses **S3-DiT (Scalable Single-Stream DiT)**:

- Text tokens, semantic tokens, and image VAE tokens are concatenated into a **single long sequence**.
- A diffusion transformer operates over this merged sequence.
- This design aims to improve **parameter efficiency** and **context fusion** while keeping the total parameter count moderate (~6B).
- The result is a model that can compete with larger dual-stream models while being easier to deploy on 16 GB GPUs.

### 1.3 Why Z-Image-Turbo matters

In practice, Z-Image-Turbo occupies a similar niche to SDXL or Flux-based models, but with a few distinct advantages:

- **Few-step inference** – quality is optimized for ~8–9 steps instead of 20–50.
- **Strong text rendering** – especially for Chinese and English text on signage, UI, posters, etc.
- **Balanced size** – large enough to be competitive, small enough to target 16 GB VRAM instead of >24 GB.
- **Open-source** – Apache-2.0, with public model weights, reference code, and community tooling.

This makes it attractive both as a **drop-in text-to-image model** and as a **base for LoRA finetuning** in the open ecosystem.

---

## 2. From official docs, model card, and paper

### 2.1 Official Hugging Face model card (Tongyi-MAI/Z-Image-Turbo)

The core source of truth is the Hugging Face model card:

- **Repository**: `Tongyi-MAI/Z-Image-Turbo`  
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>

Key information from the card and associated material:

- Z-Image-Turbo is a **distilled** version of Z-Image via a few-step distillation scheme plus reward-based post-training (DMDR).
- Recommended **inference settings**:
  - Resolution: **1024 × 1024** (default).
  - Steps: **9 inference steps** (corresponds to ~8 NFEs internally).
  - **Classifier-free guidance disabled**: `guidance_scale=0.0` and negative prompts are generally not used.
  - Long prompts: use `max_sequence_length=1024` if needed.
- The model is integrated directly into **Diffusers** via a dedicated `ZImagePipeline` and specialized components (transformer, VAE, scheduler).

### 2.2 Arxiv paper and technical summary

The Z-Image paper describes:

- A **single-stream diffusion transformer** that unifies text tokens, semantic tokens, and image tokens.
- An omni pre-training scheme that covers multiple downstream tasks (generation, editing).
- A **few-step distillation** and **reward post-training** procedure that yields Z-Image-Turbo with:
  - Latency suitable for **sub-second generation** on H800-class GPUs.
  - Compatibility with **<16 GB VRAM** consumer GPUs for single-image generation at 1024 × 1024.

The paper positions Z-Image-Turbo as a competitive alternative to other large text-to-image models while being more deployment-friendly on widely available hardware.

### 2.3 Official Z-Image collection and Spaces

Tongyi maintains a Hugging Face **collection** and **Spaces** for Z-Image:

- Collection: <https://huggingface.co/collections/Tongyi-MAI/z-image>
- Official Space for text-to-image:  
  <https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo>

These provide:

- Reference demos for prompting.
- Example prompts and typical output styles.
- A quick way to sanity-check your local installation against the official behavior.

---

## 3. Model cards, Spaces, and packaging repos

This section focuses on the three main repos the topic is about:

1. Diffusers model card: `Tongyi-MAI/Z-Image-Turbo`
2. ComfyUI packaging: `Comfy-Org/z_image_turbo`
3. GGUF quantized packaging: `jayn7/Z-Image-Turbo-GGUF`

### 3.1 Tongyi-MAI/Z-Image-Turbo (Diffusers-ready repo)

- URL: <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>

Contents:

- Standard Diffusers layout with `model_index.json`.
- Components:
  - S3-DiT backbone.
  - Qwen-based text encoder.
  - Flux-style VAE.
  - Scheduler / configuration files.
- Designed for use with **`diffusers.ZImagePipeline`**. Typical code pattern:

```python
from diffusers import ZImagePipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)
pipe.to(device)

image = pipe(
    prompt="a cinematic photo of a neon-lit street in Tokyo, wet asphalt, 4k, highly detailed",
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    max_sequence_length=1024,
).images[0]
```

Notes:

- `low_cpu_mem_usage=False` is explicitly used in the official example and tends to be more reliable for this model.
- `guidance_scale` stays at **0.0** for Turbo.
- It is recommended to install **Diffusers from GitHub** to ensure `ZImagePipeline` is available and up to date.

### 3.2 Comfy-Org/z_image_turbo (ComfyUI split checkpoints)

- URL (root): <https://huggingface.co/Comfy-Org/z_image_turbo>
- Diffusion model file:  
  <https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors>
- Text encoder file:  
  <https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors>
- VAE file:  
  <https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/vae>

Purpose:

- This repo is the **ComfyUI-ready packaging** of Z-Image-Turbo.
- It provides three safetensors that drop into the ComfyUI model folders.

Typical ComfyUI placement (based on official and community docs):

- `models/diffusion_models/z_image_turbo_bf16.safetensors`
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/vae/ae.safetensors` (often under a `flux1` or `zimage` subfolder depending on workflow).

ComfyUI offers an **official Z-Image-Turbo workflow template**:

- Docs: <https://docs.comfy.org/tutorials/image/z-image/z-image-turbo>
- Template JSON:  
  <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json>

This template expects the above files and wires them into a graph with:

- Text prompt node.
- Z-Image-Turbo diffusion node.
- VAE decode node.
- Output image node.

### 3.3 jayn7/Z-Image-Turbo-GGUF (quantized GGUF packaging)

- URL: <https://huggingface.co/jayn7/Z-Image-Turbo-GGUF>
- README: <https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/README.md>

Purpose:

- Provides **quantized GGUF versions of the Z-Image-Turbo backbone**, suitable for backends that understand GGUF (stable-diffusion.cpp, certain ComfyUI nodes, or Diffusers GGUF support).
- Lists available models and links to compatible Qwen3-4B text encoder GGUFs (e.g. from `unsloth/Qwen3-4B-GGUF`).

The README includes a **Diffusers GGUF example** where:

- The **transformer** is loaded from a `.gguf` file using `ZImageTransformer2DModel.from_single_file` with `GGUFQuantizationConfig`.
- The **pipeline** is still created from `Tongyi-MAI/Z-Image-Turbo` and receives the quantized transformer as an override.

Sketch (simplified):

```python
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
import torch

transformer = ZImageTransformer2DModel.from_single_file(
    "z_image_turbo-Q3_K_M.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")
```

Important implications:

- Only the **transformer backbone** is GGUF-quantized in this example.
- The **text encoder** and **VAE** remain full-precision Diffusers modules.
- GGUF reduces **weight storage size**, but activations still use BF16/FP16 and high resolution (e.g. 1024 × 1024) can still be VRAM-heavy.

### 3.4 Public Spaces and demos using Z-Image-Turbo

Beyond the official Tongyi Space, there are community Spaces such as:

- <https://huggingface.co/spaces/mrfakename/Z-Image-Turbo>

These provide:

- Ready-to-use browser demos.
- Example prompts for portraits, anime, logos, etc.
- A baseline for expected quality, especially if you suspect a local installation issue (e.g. wrong VAE or scheduler).

---

## 4. Community, forums, and ecosystem notes

### 4.1 Japanese technical blogs and notes

Several Japanese write-ups give practical feedback:

- Articles on WEEL, Zenn, and note.com describe:
  - Z-Image-Turbo as a **6B model** built on S3-DiT with **strong bilingual text rendering**.
  - Real-world attempts to run the **Diffusers checkpoint on Colab Free**, sometimes hitting OOM at 1024 × 1024 on a T4.
  - Using the official web demo or lighter variants (FP8, GGUF) when Colab Free is too tight.
- Community tutorials show **prompt examples** for kanji-heavy signage and typography and confirm high-quality text rendering.

These sources converge on:

- Z-Image-Turbo is technically impressive and highly capable for text rendering.
- 16 GB VRAM is realistic but not always comfortable on shared or heavily loaded environments like Colab Free.
- FP8 and GGUF packages can help but often require **more CPU RAM** and tuned workflows.

### 4.2 VRAM and performance reports from users

Reddit and GitHub discussions on Z-Image Turbo highlight:

- On desktop GPUs with **8–12 GB VRAM** and ample system RAM:
  - FP8 or GGUF variants can run 768×768 and even 1024×1024 images, but with slower generation and heavy offloading.
- On **4–6 GB VRAM** hardware:
  - Users report success with heavy offload (ComfyUI GGUF nodes, stable-diffusion.cpp), low resolutions, and long generation times.
- Reported timings (approximate, for low-VRAM laptops) for GGUF workflows at 768–896 resolutions are on the order of **1–3 minutes per image**.

The consistent message: **16 GB VRAM** is comfortable; below that, Z-Image-Turbo is still feasible but progressively slower and more fragile.

### 4.3 ComfyUI FP8 and GGUF workflows

Community guides describe:

- **FP8** versions of Z-Image-Turbo, where the DiT backbone is stored in FP8 to reduce VRAM for ComfyUI usage.
- **GGUF + text-encoder GGUF** setups, where both the DiT and Qwen3-4B are quantized and heavily offloaded to CPU.

Typical numbers reported in such guides:

- GGUF DiT Q4_K_M ~4–5 GB on disk.
- GGUF text encoder (imatrix IQ4_XS) ~2 GB, or Q8_0 ~4 GB.
- VAE ~0.3 GB.
- Effective VRAM usage around **<8 GB** for 1024 × 1024 when combined with aggressive offload and 32 GB+ system RAM.

These workflows are usually implemented in **ComfyUI**, not in standard Diffusers code, and rely on specialized nodes and offload behavior.

### 4.4 Due diligence and licensing concerns

Some due-diligence posts (e.g. on Qiita) note:

- Z-Image-Turbo is **Apache-2.0 licensed**, which allows commercial use, modification, and redistribution under broad terms.
- However, because it is developed by a Chinese company (Alibaba), organizations may need to consider potential **data governance or regulatory risks** related to Chinese law.
- These concerns are more about **organizational risk assessment** than about the open-source license itself.

For most individual or small-scale projects, Apache-2.0 is generally considered permissive and straightforward, but corporate users should still consult legal/compliance teams.

### 4.5 GitHub ecosystem and external tool support

Important code repositories include:

- Official Z-Image code: <https://github.com/Tongyi-MAI/Z-Image>
- ComfyUI example workflows:  
  <https://comfyanonymous.github.io/ComfyUI_examples/z_image/>
- Other tools (e.g. SwarmUI, krita-ai-diffusion) add support for Z-Image / Z-Image-Turbo using:
  - The Comfy-Org BF16 safetensors.
  - GGUF variants from jayn7.
  - Flux-compatible VAEs (shared with Flux 1).

This ecosystem makes Z-Image-Turbo accessible via:

- Diffusers Python APIs.
- GUI tools like ComfyUI or SwarmUI.
- Creative tools like Krita extensions.

---

## 5. Implementation patterns and tips

### 5.1 Recommended inference settings (generic)

Baseline settings that align with the official model card and common practice:

- **Prompt language**: English and Chinese are strongest, but other languages may work to some degree.
- **Resolution**:
  - Default: **1024 × 1024**.
  - Low-VRAM or experimentation: **768 × 768** or **832 × 832**.
- **Steps**:
  - `num_inference_steps = 9` (corresponds to 8 NFEs for Turbo).
- **Guidance**:
  - `guidance_scale = 0.0` (Turbo is distilled without CFG).
  - No negative prompt is required or particularly useful.
- **Max sequence length**:
  - For long prompts, set `max_sequence_length=1024` when calling the pipeline.
- **Batch size**:
  - Start with `num_images_per_prompt = 1` on 16 GB VRAM.

### 5.2 Running Z-Image-Turbo with Diffusers (full BF16/FP16)

#### 5.2.1 Environment setup

On Colab or Kaggle:

1. Enable GPU.
2. Install Diffusers from GitHub and required deps:

```bash
pip install -U "git+https://github.com/huggingface/diffusers"     transformers accelerate safetensors huggingface_hub sentencepiece
```

3. Optionally set Hugging Face cache to a writeable path:
   - Colab: `/content/hf_home`
   - Kaggle: `/kaggle/working/hf_home`

#### 5.2.2 Device and dtype helper

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda" and torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16
```

#### 5.2.3 Pipeline loading

```python
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False,
)
pipe.to(device)
```

Optional performance optimizations:

- `pipe.enable_attention_slicing("auto")`
- `pipe.enable_vae_slicing()`
- On supported GPUs: `pipe.transformer.set_attention_backend("flash")`
- For repeated runs on PyTorch 2.x: `pipe.transformer.compile()`

#### 5.2.4 Simple generation helper

```python
from PIL import Image
from datetime import datetime

def generate_zimage(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    steps: int = 9,
    guidance_scale: float = 0.0,
    seed: int | None = 42,
    max_sequence_length: int = 1024,
    save: bool = True,
) -> Image.Image:
    width = (width // 16) * 16
    height = (height // 16) * 16

    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        generator=generator,
    )
    img = result.images[0]

    if save:
        name = f"zimage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img.save(name)
        print("Saved:", name)

    return img
```

### 5.3 Running Z-Image-Turbo with Diffusers + GGUF (jayn7)

The jayn7 GGUF repo is most useful if:

- You want to reduce on-disk model size.
- You want to experiment with Diffusers’ GGUF integration.

#### 5.3.1 Conceptual model

- GGUF is applied to the **transformer** (DiT backbone).
- The **text encoder** and **VAE** from `Tongyi-MAI/Z-Image-Turbo` are still used as standard diffusers modules.
- GGUF reduces **weight memory** but not **activation memory**.
- At 1024 × 1024, activations dominate VRAM usage, so GGUF alone does not make this a “4 GB VRAM model”.

#### 5.3.2 Example loading pattern

```python
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

gguf_path = "z_image_turbo-Q3_K_M.gguf"

transformer = ZImageTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")
```

To save VRAM further:

- Enable attention and VAE slicing.
- If VRAM is very tight, use:

```python
pipe = pipe.to("cpu")
pipe.enable_model_cpu_offload()
```

Then generate at **512 × 512** or **768 × 768** first.

### 5.4 Running Z-Image-Turbo with ComfyUI (Comfy-Org/z_image_turbo)

#### 5.4.1 Basic setup

1. Install ComfyUI (local or on a cloud VM / Colab).
2. Download the three safetensors from `Comfy-Org/z_image_turbo`:
   - `split_files/diffusion_models/z_image_turbo_bf16.safetensors`
   - `split_files/text_encoders/qwen_3_4b.safetensors`
   - `split_files/vae/ae.safetensors`
3. Place them into:

   - `ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors`
   - `ComfyUI/models/text_encoders/qwen_3_4b.safetensors`
   - `ComfyUI/models/vae/ae.safetensors` (sometimes under `vae/flux1/` depending on guide).

4. Start ComfyUI.
5. Load the **Z-Image-Turbo template** from the workflow templates menu.

#### 5.4.2 FP8 and GGUF variants in ComfyUI

Community nodes and workflows provide:

- FP8 versions (DiT in FP8) to reduce VRAM at some quality cost.
- GGUF workflows where both DiT and text encoder are GGUF-quantized and can be heavily offloaded.

These workflows often require:

- At least **8 GB VRAM** for comfortable operation at 1024 × 1024.
- **32 GB system RAM** for heavy CPU offload.
- Patience, as generation can be significantly slower than full BF16 on a 16 GB GPU.

### 5.5 Colab and Kaggle patterns (Diffusers)

#### 5.5.1 Colab Free

For **Colab Free**:

- GPU: typically **T4 16 GB VRAM**.
- RAM: about **12–25 GB** depending on runtime.

Practical strategy:

- Use full **Diffusers ZImagePipeline** with BF16/FP16.
- Start with **1024 × 1024**, 9 steps, guidance 0.0.
- If you hit OOM:
  - Drop to **768 × 768** or **832 × 832**.
  - Enable attention and VAE slicing.
  - Optionally enable `enable_model_cpu_offload()`.
- Keep only one heavy pipeline loaded at a time; delete others and clear cache if switching models.

#### 5.5.2 Kaggle Free

For **Kaggle Free**:

- GPU: often **P100 16 GB** or **T4x2**.
- Recommended layout:
  - HF cache: `/kaggle/working/hf_home`
  - Outputs: `/kaggle/working/zimage_outputs`

Usage pattern:

- Same Diffusers setup as Colab.
- Download and cache the model once to `/kaggle/working` using `hf download` (with `--resume-download`) if needed.
- Use `ZImagePipeline.from_pretrained("./Z-Image-Turbo", local_files_only=True)` to avoid repeated large downloads.

### 5.6 Memory-management mental model

A useful simplified model for Z-Image-Turbo memory:

1. The model is **designed for 16 GB VRAM GPUs**.
2. Diffusers offload modes (`enable_model_cpu_offload`, `enable_sequential_cpu_offload`) mainly reduce **weight memory**, not **activation memory**.
3. At **1024 × 1024**, activation memory for a 6B DiT is inherently large.
4. GGUF in the Diffusers example only quantizes the **transformer**, leaving the text encoder and VAE full-precision.
5. On platforms like Colab Free, you are close to the hardware limit, so OOM errors are common and not necessarily bugs.

Practical levers with the biggest impact:

- **Resolution** (first lever).
- **Batch size** (keep at 1).
- **Offload and slicing** (help, but cannot beat physics).
- **Choice of backend** (ComfyUI, stable-diffusion.cpp, SDNQ, etc.).

### 5.7 Low-VRAM alternatives and complementary variants

If you cannot make the main Diffusers checkpoint fit comfortably, consider:

- **FP8 variants** (e.g. `T5B/Z-Image-Turbo-FP8`) for smaller weight footprints.
- **4-bit SDNQ variants** (e.g. `Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32`) with specialized loaders.
- **GGUF backends** such as stable-diffusion.cpp or SwarmUI with Z-Image-Turbo support.

These can significantly reduce VRAM requirements but involve **different code stacks**, and quality may degrade slightly depending on quantization level.

---

## 6. Limitations, caveats, and open questions

### 6.1 Hardware and memory limitations

- Z-Image-Turbo is **not** a tiny model:
  - ~6B parameters.
  - Comfortable target: **16 GB VRAM** + **32 GB RAM** for heavier workflows.
- On 12–16 GB GPUs with limited RAM (e.g. Colab Free):
  - 1024 × 1024 may OOM in naïve setups.
  - Careful use of slicing, offload, and lower resolutions is often required.
- Claims that it “runs on 4 GB VRAM” generally refer to **heavily offloaded, quantized workflows** (FP8, GGUF, SDNQ) at **low resolutions** and **low throughput**.

### 6.2 Diffusers GGUF integration is partial

- Current Diffusers GGUF examples quantize the **transformer backbone** only.
- Text encoder and VAE remain full-precision modules and can still consume multiple GB of VRAM.
- Full GGUF text-encoder support (e.g. Qwen3-4B) is still evolving, and plugging a GGUF text encoder directly into `ZImagePipeline` is not yet “drop-in” everywhere.

### 6.3 Ecosystem maturity

- Z-Image-Turbo is relatively new compared to SDXL.
- Tooling (ComfyUI nodes, LoRA training scripts, etc.) is rapidly evolving.
- Some workflows assume specific file layouts or node versions, so documentation can lag behind.

### 6.4 Licensing and jurisdictional considerations

- Apache-2.0 license is permissive and widely used, but:
  - Enterprise users should be aware that the model is developed in China and may subject data or usage to Chinese regulatory considerations depending on deployment context.
  - Compliance and governance teams may need to review usage scenarios.

### 6.5 Safety and content considerations

- As with other powerful text-to-image models, Z-Image-Turbo can generate sensitive or NSFW content depending on prompts and deployment settings.
- Many public demos implement filters or prompt sanitization; private deployments are responsible for their own safety and policy layers.

### 6.6 Open questions and expected evolution

- **Base and Edit variants**: broader public checkpoints for Z-Image-Base and Z-Image-Edit may expand customization and editing workflows.
- **LoRA training best practices**: community is still converging on robust pipelines and hyperparameters for Z-Image LoRAs.
- **Mobile / edge deployment**: while Turbo targets 16 GB GPUs, future compression and distillation could yield variants suitable for more constrained devices.

---

## 7. Practical quick-reference checklists

### 7.1 Diffusers (full pipeline) quick-start

1. Install from GitHub: `pip install "git+https://github.com/huggingface/diffusers"` plus `transformers`, `accelerate`, `safetensors`.
2. Choose dtype: `bfloat16` if supported, otherwise `float16`.
3. Load with `ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype, low_cpu_mem_usage=False)`.
4. Move to GPU: `pipe.to("cuda")`.
5. Generate:
   - 1024 × 1024
   - 9 steps
   - `guidance_scale = 0.0`
   - `max_sequence_length = 1024` for long prompts.
6. For OOM:
   - Reduce resolution to 768 × 768.
   - Enable attention/vae slicing.
   - Optionally enable `enable_model_cpu_offload()`.

### 7.2 Diffusers + GGUF quick-start

1. Download a small GGUF (e.g. Q3 or Q4) from `jayn7/Z-Image-Turbo-GGUF`.
2. Load the GGUF transformer with `ZImageTransformer2DModel.from_single_file(...)` using `GGUFQuantizationConfig`.
3. Create `ZImagePipeline` from `Tongyi-MAI/Z-Image-Turbo`, passing the GGUF transformer.
4. Optionally keep the pipeline on CPU and use `enable_model_cpu_offload()`.
5. Start at **512 × 512**, 9 steps, guidance 0.0, batch size 1.
6. Increase resolution only if this is stable.

### 7.3 ComfyUI quick-start (BF16 split model)

1. Install ComfyUI.
2. Download three safetensors from `Comfy-Org/z_image_turbo`.
3. Place them into `models/diffusion_models`, `models/text_encoders`, and `models/vae` as instructed.
4. Start ComfyUI and load the official Z-Image-Turbo workflow template.
5. Use 1024 × 1024, 9 steps; reduce resolution or enable low-VRAM options as needed.

### 7.4 Low-VRAM and debugging checklist

- **If you see CUDA OOM**:
  - Lower resolution.
  - Ensure batch size is 1.
  - Turn on attention and VAE slicing.
  - Consider CPU offload for Diffusers.
- **If outputs look wrong (blurry, distorted text)**:
  - Confirm VAE path and configuration (Flux-compatible VAE).
  - Compare outputs with the official HF Space using the same prompt and seed.
- **If downloads are slow or fail**:
  - Use `huggingface_hub` CLI (`hf download --resume-download`) to fetch the model once.
  - On cloud notebooks, store the downloaded model in a persistent path (`/content/drive` for Colab, `/kaggle/working` or a Kaggle Dataset for Kaggle).

---

## 8. References / links

### 8.1 Official and primary sources

- Z-Image-Turbo model card (Diffusers):  
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Z-Image paper (arxiv):  
  <https://arxiv.org/abs/2511.22699>
- Tongyi Z-Image collection:  
  <https://huggingface.co/collections/Tongyi-MAI/z-image>
- Official Z-Image-Turbo Space:  
  <https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo>
- Official Z-Image GitHub repo:  
  <https://github.com/Tongyi-MAI/Z-Image>

### 8.2 Packaging and ecosystem

- Comfy-Org Z-Image-Turbo split model (ComfyUI):  
  <https://huggingface.co/Comfy-Org/z_image_turbo>
- Z-Image-Turbo GGUF quantized models:  
  <https://huggingface.co/jayn7/Z-Image-Turbo-GGUF>
- ComfyUI Z-Image-Turbo tutorial:  
  <https://docs.comfy.org/tutorials/image/z-image/z-image-turbo>
- ComfyUI Z-Image workflow template JSON:  
  <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json>
- ComfyUI example workflows:  
  <https://comfyanonymous.github.io/ComfyUI_examples/z_image/>

### 8.3 Community write-ups and tutorials

- Japanese overview and how-to articles (Zenn, note.com, WEEL etc.):  
  <https://zenn.dev/kun432/scraps/c15de86463c670>  
  <https://note.com/aiaicreate/n/n3f5506008c24>  
  <https://weel.co.jp/media/tech/z-image-turbo/>
- Due diligence / risk discussion (Qiita):  
  <https://qiita.com/GeneLab_999/items/ea0f694a9749c41150e2>

### 8.4 Low-VRAM and related variants

- FP8 variant (example):  
  <https://huggingface.co/T5B/Z-Image-Turbo-FP8>
- Example SDNQ-based compressed variant:  
  <https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32>
- SwarmUI model support note mentioning Z-Image-Turbo GGUF:  
  <https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md>
