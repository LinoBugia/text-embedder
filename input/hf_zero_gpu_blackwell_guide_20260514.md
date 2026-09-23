# ZeroGPU Blackwell Runtime Drift Recovery Guide

This guide is for maintainers of Hugging Face Spaces whose ZeroGPU applications suddenly started failing after the ZeroGPU runtime moved onto Blackwell-class hardware.

The main failure mode is specific: a Space that previously worked under older ZeroGPU assumptions, H200-era examples, or older CUDA/PyTorch pins now runs on RTX PRO 6000 Blackwell / `sm_120` infrastructure and fails during build, import, model loading, CUDA kernel launch, or inference.

Treat this as a Blackwell-first recovery problem. General runtime drift still matters, especially for dormant Spaces that have not been rebuilt or restarted for weeks or months, but it is a secondary path. Start by checking the actual assigned hardware, PyTorch generation, CUDA wheel family, and attention backend. Then decide whether the failure is a Blackwell kernel mismatch, a dependency drift problem, a VRAM limit, a model-loading hang, or something model-specific.

The goal is practical recovery: keep the Space understandable, remove fragile acceleration paths first, return to a standard PyTorch/Transformers path, and only add architecture-specific acceleration after the Space is stable again.

---

## 0. Fast Calm-Down Summary

A sudden ZeroGPU failure is usually repairable. Do not start by rewriting the app or assuming the model is permanently broken.

Start with these checks:

1. Print the actual GPU name, compute capability, PyTorch version, and CUDA version.
2. If the device is RTX PRO 6000 Blackwell or capability `(12, 0)`, treat it as a Blackwell / `sm_120` target.
3. If the error says `no kernel image is available for execution on the device`, `invalid device function`, or `sm_120 is not compatible`, first suspect a CUDA kernel or wheel that does not support Blackwell.
4. Remove forced FlashAttention, FlashAttention-3, xFormers, custom Triton, custom CUDA, or hard-pinned attention paths before reducing model quality.
5. Try standard `sdpa` first where supported, then `eager` as a conservative fallback.
6. Treat VRAM as the cause only when the error is actually memory-related, such as `CUDA out of memory` or allocation failure.
7. In parallel, check runtime drift: loose dependency pins, rebuilds, cache loss, changed wheels, startup downloads, old requirements, and Spaces that have been dormant for a long time.

This matters because Blackwell migration failures and ordinary runtime drift often appear at the same time. A Space may wake up on new hardware, with a new runtime baseline, with dependencies re-resolved, and with caches missing. The repair should be calm and layered, not random.

---

## 1. What This Guide Covers

This guide focuses on one narrow but high-impact situation:

> A Hugging Face ZeroGPU Space worked before, then suddenly failed after ZeroGPU started assigning RTX PRO 6000 Blackwell-class GPUs instead of the older hardware assumptions around A100 or H200.

It covers:

- ZeroGPU hardware drift from A100-era public references to H200-era behavior and then Blackwell.
- Current ZeroGPU runtime assumptions from the [official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).
- Blackwell / `sm_120` failure signatures.
- Why H200-compatible CUDA kernels may fail on Blackwell.
- Attention backend migration: FlashAttention, FlashAttention-3, xFormers, Triton, custom CUDA, `sdpa`, and `eager`.
- VRAM triage for `large` 48GB and `xlarge` 96GB ZeroGPU tiers.
- Runtime drift as a secondary but common problem.
- Dormant Space restart failures.
- What information to collect before asking another maintainer or an AI assistant to patch the Space.

It does not try to be a complete Hugging Face Spaces guide, a complete CUDA guide, or a complete GPU architecture reference. It only gives the background needed to repair ZeroGPU Spaces affected by the Blackwell transition.

---

## 2. Who This Is For

Use this guide if one of these is true:

- Your ZeroGPU Space started failing around the Blackwell transition.
- Logs show `NVIDIA RTX PRO 6000 Blackwell`, `MIG 2g.48gb`, `capability (12, 0)`, `sm_120`, `CUDA 12.8`, or `torch 2.8.0+cu128`.
- The Space used to assume H200, A100, CUDA 11.8, CUDA 12.1, CUDA 12.4, old PyTorch, FlashAttention-2, FlashAttention-3, xFormers, or custom Triton kernels.
- The error mentions `no kernel image is available for execution on the device`.
- The app imports but fails at first GPU call.
- The Space was inactive for a long time and failed after restart or rebuild.
- You want to attach a compact but complete reference to another maintainer or code assistant and ask it to propose code edits.

Experienced maintainers can skim the tables and checklists. Less experienced maintainers should follow the recovery flow in order.

---

## 3. First Rule: Verify the Actual Runtime

Do not rely on old documentation, old blog posts, old README text, or old successful runs. Print the current runtime from inside the Space.

Add a temporary diagnostic block near startup or inside the GPU-decorated function:

```python
import os
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device capability:", torch.cuda.get_device_capability(0))
    print("Allocated GB:", round(torch.cuda.memory_allocated(0) / 1024**3, 3))
    print("Reserved GB:", round(torch.cuda.memory_reserved(0) / 1024**3, 3))

for key in [
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
]:
    print(f"{key}:", os.environ.get(key))
```

Why this matters: most bad fixes come from repairing the wrong environment. A Space that used to run on an H200-class target may now be executing on a Blackwell target. A requirement file that used to resolve a working PyTorch stack may now pull a third-party package that installs an older `torch`. A direct wheel URL that used to match the GPU may now be the wrong binary target.

---

## 4. ZeroGPU Hardware Drift Timeline

This table is intentionally short. It is not a full history of Hugging Face infrastructure. It is the minimum timeline needed to understand why older ZeroGPU examples, user expectations, and dependency pins may no longer match the current runtime.

| Date / period | Public signal | What it means for maintainers | Source |
|---|---|---|---|
| 2024-05 | ZeroGPU launched as shared GPU infrastructure for Hugging Face Spaces. External coverage and public announcements described ZeroGPU as using NVIDIA A100 devices. | Early examples and user expectations may assume A100-class behavior, 40GB-class VRAM, and older CUDA/PyTorch compatibility. | [The Register coverage](https://www.theregister.com/software/2024/05/17/hugging-face-plans-to-make-10m-in-gpus-available-to-public/1299644), [Clem Delangue launch post mirror on LinkedIn](https://www.linkedin.com/posts/clementdelangue_gpu-poor-no-more-super-excited-to-officially-activity-7196881557284868096-M96G) |
| 2024-09 | User-facing tutorials still described ZeroGPU as A100-backed with 40GB VRAM. | Even after launch, many community references kept the A100 mental model. | [Example community guide](https://dev.to/pi19404/hugging-face-zero-gpu-spaces-shieldgemma-application-12g5) |
| 2025-04 | In a Space discussion, `hysts` explained that ZeroGPU hardware was being migrated from half A100 to one-third H200 and half H200, while documentation had not yet been updated. | This is a public evidence point for A100-to-H200 drift and for documentation lag during hardware transitions. | [FoundHand discussion](https://huggingface.co/spaces/Chaerin5/FoundHand/discussions/4) |
| 2025-04 | The same discussion shows `sm_90` / H200-related PyTorch compatibility problems, including dependency resolution pulling an older `torch`. | H200 migration already created compatibility failures before Blackwell. This is a useful prior pattern: hardware migration plus dependency drift. | [FoundHand discussion](https://huggingface.co/spaces/Chaerin5/FoundHand/discussions/4) |
| H200 documentation period | The older ZeroGPU Explorers page described H200 devices under the hood, 70GB VRAM per workload, PyTorch `2.1.2` to `2.5.1`, and Python `3.10.13`. It now marks itself out of date and points to current documentation. | Many Spaces and maintainers reasonably built around H200-era assumptions. Those assumptions should now be treated as historical. | [ZeroGPU Explorers page](https://huggingface.co/zero-gpu-explorers) |
| 2026-05-13 | A forum thread reported ZeroGPU Spaces running on NVIDIA RTX PRO 6000 Blackwell Server Edition instead of H200. Logs showed capability `(12, 0)` and CUDA `12.8`. | This is the practical Blackwell incident point for maintainers: the runtime target changed and some Spaces broke. | [Forum thread: RTX PRO 6000 instead of H200](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960) |
| 2026-05-13 | The same forum thread includes a PyTorch compatibility warning for `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb` with `sm_120`, where the installed torch supported only up to `sm_90`. | If PyTorch or a third-party CUDA extension only supports older architectures, the Space may fail even when the repository code did not change. | [Forum thread](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960) |
| 2026-05-13 | `hysts` stated in the forum that ZeroGPU documentation was being updated to reflect the hardware change and linked the docs PR. | This is a public confirmation point that the docs were catching up with the hardware refresh. | [Forum thread](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960) |
| 2026-05-13 | Hub docs PR #2474 was merged. It explicitly described a hardware refresh from NVIDIA H200 to NVIDIA RTX Pro 6000 Blackwell and updated VRAM figures, `large`/`xlarge` sizing, and PyTorch support starting at `2.8.0`. | This is the strongest public documentation-change anchor for the Blackwell update. | [hub-docs PR #2474](https://github.com/huggingface/hub-docs/pull/2474) |
| Around the same period | User posts documented H200 expectations versus actual Blackwell runtime behavior. One public post showed `RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb`, capability `(12, 0)`, `torch 2.8.0+cu128`, CUDA `12.8`, and `no kernel image` failure in a Qwen3-TTS Space using `kernels-community/flash-attn3`. | This is a clean public case study for Blackwell architecture mismatch and attention kernel fragility. | [Imosu post](https://huggingface.co/posts/Imosu/801922682272974) |
| Current documentation | Current ZeroGPU docs describe dynamic allocation of NVIDIA RTX PRO 6000 Blackwell GPUs, `large` as half RTX PRO 6000 Blackwell with 48GB VRAM, `xlarge` as full RTX PRO 6000 Blackwell with 96GB VRAM, and PyTorch support from `2.8.0` upward. | New repairs should target Blackwell / `sm_120` / CUDA 12.8-class / PyTorch 2.8+ assumptions unless the docs change again. | [Current ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) |

Update this timeline when new public evidence appears. The table should stay factual: use dates, links, and cautious wording. Do not turn it into speculation.

---

## 5. Minimal GPU Generation Notes

A100, H200, and RTX PRO 6000 Blackwell are all NVIDIA GPUs, but they are not the same binary target.

For normal high-level PyTorch code, that difference is often hidden. For prebuilt CUDA extensions, attention kernels, xFormers, Triton kernels, direct wheel URLs, or model remote code that forces an acceleration backend, the exact GPU architecture matters.

| GPU generation / family | Typical architecture target | Why it matters |
|---|---|---|
| A100 | Ampere / `sm_80` class | Early ZeroGPU references and examples may have assumed this generation. Old wheels may work here but not on later targets. |
| H100 / H200 | Hopper / `sm_90` class | H200-era ZeroGPU examples and docs created a Hopper mental model. FlashAttention-3 and some kernels are often associated with Hopper-class assumptions. |
| RTX PRO 6000 Blackwell | Blackwell / observed `sm_120` class in ZeroGPU reports | Current ZeroGPU docs describe RTX PRO 6000 Blackwell. Kernels must support Blackwell or fail at runtime. |

Do not over-read the product names. The relevant operational fact is the compute capability and the binary support inside PyTorch or third-party kernels.

A Space can build successfully and import successfully but fail later when the first GPU kernel launches. That is why `no kernel image is available for execution on the device` is so important: it often points to a binary kernel that was not compiled for the active GPU architecture.

---

## 6. Current ZeroGPU Baseline

Current ZeroGPU documentation describes ZeroGPU as shared infrastructure that dynamically allocates and releases [NVIDIA RTX PRO 6000 Blackwell GPUs](https://huggingface.co/docs/hub/spaces-zerogpu) as needed.

The documented GPU sizes are:

| ZeroGPU size | Backing hardware | Documented VRAM | Practical note |
|---|---|---:|---|
| `large` | Half NVIDIA RTX Pro 6000 Blackwell | 48GB | Default ZeroGPU size. Do not assume H200 70GB behavior. |
| `xlarge` | Full NVIDIA RTX Pro 6000 Blackwell | 96GB | More appropriate for large memory-sensitive workloads, but still subject to ZeroGPU behavior and quota. |

The same documentation currently lists:

- Gradio SDK compatibility.
- Gradio 4+.
- PyTorch support starting at `2.8.0` and newer listed versions.
- Python `3.10.13` and `3.12.12`.

Why this matters: a Space that pins old PyTorch, installs a dependency that pulls old PyTorch, or uses a wheel built for older CUDA architectures can fail before your application logic is even relevant.

Do not assume Python 3.11 support unless current docs explicitly list it. If a Space needs Python 3.11 for some unrelated dependency, treat that as a separate compatibility investigation rather than a safe ZeroGPU baseline.

---

## 7. Blackwell-First Recovery Flow

Use this flow before making large edits.

### 7.1 Identify the failure phase

| Phase | Typical symptom | Meaning |
|---|---|---|
| Build | `pip` fails, wheel build fails, configuration error | Dependency, Python tag, CUDA wheel, or ZeroGPU supported-version issue. |
| Import | `ModuleNotFoundError`, `ImportError`, ABI error | A package is missing, wrong version, wrong Python wheel, or compiled extension mismatch. |
| Model loading | Startup hangs, timeout, safetensors load stalls | Large download, cache issue, FUSE/mmap path, or memory/offload issue. |
| First GPU call | `no kernel image`, `invalid device function`, `sm_120 not compatible` | Strong Blackwell kernel mismatch signal. |
| First generation | OOM, long stall, crash after model loads | Could be VRAM, attention backend, decoding length, resolution, frames, or model-specific code. |
| Large input only | Small prompt works; larger prompt/resolution/frames fail | More likely VRAM, KV cache, attention memory, VAE, video frames, or batch size. |
| After restart only | Same repo worked before; now fails | Runtime drift plus possible Blackwell transition. |

### 7.2 Choose the first action

| Error text or clue | First action |
|---|---|
| `no kernel image is available for execution on the device` | Remove forced FlashAttention/xFormers/Triton/custom CUDA paths. Check PyTorch and CUDA wheel target. Retry with `sdpa` or `eager`. |
| `sm_120 is not compatible with the current PyTorch installation` | Use a ZeroGPU-supported PyTorch version with CUDA 12.8-class support. Check whether another dependency installed old torch. |
| `CUDA out of memory` | Treat as VRAM pressure. Reduce batch/resolution/frames/context first; then consider offload, slicing, tiling, or `xlarge`. |
| Startup hang during model load | Check FUSE/mmap, cache, large model downloads, and `disable_mmap` behavior. |
| `ModuleNotFoundError: flash_attn` | Do not immediately install FlashAttention. First check whether the model remote code is forcing it unnecessarily. Patch import requirements if safe. |
| Build installs `torch==2.0.1` or old torch unexpectedly | Inspect dependency tree. A package such as old xFormers may be pinning torch indirectly. |
| Space worked for months and then failed after restart | Treat as Blackwell transition plus runtime drift. Compare logs and resolved package versions. |

### 7.3 First safe repair posture

Prefer this order:

1. Inspect logs and runtime.
2. Remove hard-forced architecture-specific kernels.
3. Use standard PyTorch/Transformers attention paths.
4. Restore a supported PyTorch/CUDA/Python baseline.
5. Reduce VRAM only if the error is actually memory-related.
6. Re-pin minimal dependencies after the Space is stable.
7. Reintroduce optional acceleration only after confirming Blackwell support.

This order avoids the common mistake of reducing model quality when the actual failure is a kernel compiled for the wrong architecture.

---

## 8. Blackwell Failure Signatures

### 8.1 Strong Blackwell / kernel mismatch signals

These are high-signal errors:

```text
CUDA error: no kernel image is available for execution on the device
```

```text
invalid device function
```

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

```text
current PyTorch install supports CUDA capabilities ... sm_90
```

```text
unsupported gpu architecture 'compute_120'
```

Why this matters: these messages point below your Python application logic. The loaded binary code may not contain a kernel image for the Blackwell GPU assigned by ZeroGPU. Reducing prompt length or image resolution will not fix a missing architecture target.

### 8.2 Medium signals

These need more context:

```text
CUDA error: invalid argument
```

```text
Triton compilation failed
```

```text
flash_attn import succeeded, but generation fails
```

```text
xformers memory_efficient_attention failed
```

```text
RuntimeError during first model.generate()
```

Why this matters: the package may import successfully but still fail on the first architecture-specific kernel path. Import success is not proof of Blackwell runtime safety.

### 8.3 Weak signals

These can be unrelated to Blackwell:

```text
ModuleNotFoundError
```

```text
pip resolution error
```

```text
download timeout
```

```text
file not found
```

```text
CUDA out of memory
```

Treat these as dependency, cache, model-loading, or VRAM issues unless paired with Blackwell-specific GPU or kernel messages.


### 8.6 Blackwell compatibility drift map

Use this map when a Space fails after the Blackwell transition but the first error message is ambiguous. Blackwell drift is not one bug. It is a family of compatibility boundaries that can all become visible after the assigned runtime changes to RTX PRO 6000 Blackwell / `sm_120`.

| Layer | Typical Space families | What can drift | Common symptom | First recovery move |
|---|---|---|---|---|
| PyTorch core | all torch Spaces | old torch wheel does not support `sm_120` | `sm_120 is not compatible`, `no kernel image`, CUDA op failure | move to current ZeroGPU-supported torch 2.8+ family |
| TorchVision / TorchAudio | image, video, audio Spaces | companion package mismatch or API removal | import error, missing operator, video API failure | align with torch matrix; remove old APIs |
| Triton / Inductor / `torch.compile` | optimized LLM, image, video Spaces | compiled kernel path does not match torch/CUDA/Blackwell | compile failure, runtime segfault, invalid device function | disable compile/Triton path during recovery |
| FlashAttention / FA2 | LLM, VLM, TTS | missing Blackwell kernel image | `no kernel image`, invalid device function | remove forced flash path; use SDPA/eager |
| FA3 | H200-era ZeroGPU optimization | Hopper/H200 assumption carried into Blackwell | startup or first inference failure | treat as optional, not baseline |
| FA4 | experimental/new attention stacks | uneven RTX PRO 6000 Blackwell support | compile/runtime failure | avoid as recovery default |
| xFormers | Diffusers, Stable Diffusion, ComfyUI-style Spaces | memory-efficient attention kernel mismatch or torch downgrade | NotImplementedError, kernel launch failure, dependency conflict | remove `enable_xformers_memory_efficient_attention()` first |
| SageAttention / `sage_hub` | Diffusers/video/image pipelines | backend-specific missing kernel | `no kernel image` only when backend is enabled | disable backend; test default/SDPA |
| bitsandbytes / quantization | LLM, VLM, TTS | quantization kernel does not support the runtime | load failure, first-token failure, invalid device function | test without quantization backend |
| ONNX Runtime GPU | ONNX export/inference Spaces | CUDA/cuDNN provider mismatch, `sm_120` PTX issue | CUDAExecutionProvider failure, CPU fallback, invalid PTX | verify provider; fallback to CPU for isolation |
| OpenCV / cv2 | image/video preprocessing Spaces | NumPy ABI, headless/GUI dependency, wheel drift | `import cv2` failure, libGL/highgui errors | use `opencv-python-headless`; check NumPy |
| Legacy `.pth` / `.pt` checkpoints | old research demos, copied Spaces | PyTorch 2.6+ `torch.load(weights_only=True)` default | `Weights only load failed` | use safetensors/state_dict or trusted load path |
| Custom CUDA extensions | research repos, ComfyUI nodes | extension built without `sm_120` | import or runtime kernel failure | disable extension; use PyTorch fallback |
| Serving stacks | TGI, vLLM, FlashInfer-style paths | bundled backend assumes older GPU targets | warmup failure, routing failure, no kernel image | update serving stack or disable backend-specific path |

Why this matters: the Space may correctly detect the Blackwell GPU and still fail later. Detection proves that the device is visible. It does not prove that every compiled extension, quantization backend, video decoder, attention kernel, or ONNX provider was built for the assigned runtime.

A useful recovery sentence is:

```text
The Space works only after the actual model path works without optional acceleration.
```

That means the order is:

```text
1. device visible
2. tiny CUDA op works
3. base PyTorch path works
4. minimal model forward works
5. full app works
6. optional acceleration works
```

Do not skip from step 1 to step 6.

---

## 9. Do Not Misdiagnose Kernel Mismatch as VRAM

Blackwell migration problems can look like performance or memory problems, especially when users remember that H200 had a different memory profile. But the recovery path is different.

| Symptom | More likely cause | First fix |
|---|---|---|
| `no kernel image` | Missing Blackwell kernel support | Remove or replace the failing CUDA extension. |
| `sm_120 not compatible` | PyTorch or extension lacks Blackwell support | Install supported torch/CUDA stack or remove old extension. |
| `CUDA out of memory` | VRAM pressure | Reduce memory use. |
| Small input works, large input fails | VRAM / KV cache / resolution / frame count | Reduce size or add offload. |
| Import succeeds, first CUDA call fails | Architecture-specific runtime kernel | Check FlashAttention/xFormers/Triton/custom CUDA. |
| Startup hangs before GPU call | Model loading / FUSE / download / cache | Inspect startup path and cache behavior. |

Rule: if the error is about a kernel image or architecture support, do not start with VRAM reduction. If the error is explicit OOM, do not start by replacing attention kernels unless attention memory is the likely source.

---

## 10. Attention Backend Migration

Attention is the most common place where high-level model code hides architecture-specific kernels.

The [Transformers attention interface](https://huggingface.co/docs/transformers/attention_interface) documents multiple attention backends, including `flash_attention_3`, `flash_attention_2`, `sdpa`, and `eager`, and shows that models can be loaded with `attn_implementation` or switched with `set_attn_implementation()`.

The Blackwell-safe migration posture is:

1. Do not force FlashAttention first.
2. Remove direct FlashAttention wheel assumptions.
3. Try `sdpa` where supported.
4. Fall back to `eager` if `sdpa` fails.
5. Reintroduce FlashAttention, xFormers, or Triton only after verifying Blackwell support for the exact runtime.

Why this matters: FlashAttention and similar packages may be excellent when correctly matched to the GPU, PyTorch, CUDA, Python, and ABI. They are also fragile when the target GPU generation changes.

### 10.1 Preferred initial pattern

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your/model"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
```

### 10.2 Conservative fallback pattern

```python
from transformers import AutoModelForCausalLM

model_id = "your/model"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
except Exception as sdpa_error:
    print("SDPA load failed; falling back to eager attention")
    print(repr(sdpa_error))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
```

Use this as a diagnostic recovery step, not necessarily as final performance tuning.

### 10.3 Runtime switching where supported

```python
try:
    model.set_attn_implementation("sdpa")
except Exception as e:
    print("Could not switch to SDPA:", repr(e))
    try:
        model.set_attn_implementation("eager")
    except Exception as e2:
        print("Could not switch to eager:", repr(e2))
```

This is useful only if the model supports the Transformers attention interface. For older remote-code models, you may need to inspect the model code.

### 10.4 Dangerous patterns to search for

Search the repo for:

```text
flash_attn
flash_attention_2
flash_attention_3
kernels-community/flash-attn
kernels-community/flash-attn2
kernels-community/flash-attn3
xformers.ops.memory_efficient_attention
xformers
triton.jit
@triton.jit
load_inline
CUDAExtension
cpp_extension
setup.py
TORCH_CUDA_ARCH_LIST
attn_implementation="flash_attention_2"
attn_implementation="flash_attention_3"
```

Why this matters: these strings often mark the line where the app stops being generic PyTorch and starts relying on a specific binary kernel path.

### 10.5 Remote code forcing FlashAttention

Some model repositories use `trust_remote_code=True` and import FlashAttention at module import time. That can fail even if the actual inference path could run without FlashAttention.

Bad pattern:

```python
from flash_attn import flash_attn_func
```

Better pattern:

```python
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    HAS_FLASH_ATTN = False
```

Then select a safe path when FlashAttention is absent or incompatible.

Be careful: patching remote code should be minimal and documented. The goal is not to rewrite the model. The goal is to stop an optional accelerator from becoming a mandatory import or mandatory runtime path.

---

## 11. Acceleration Stack Risk: FlashAttention, FA3, FA4, xFormers, SageAttention, Triton, and Custom CUDA

Blackwell migration failures often appear inside an acceleration layer, not inside the model architecture itself. The repository may still import. The model may still load. A tiny CPU path may still work. The failure appears when a compiled GPU kernel is finally launched.

This is why the first recovery posture is conservative:

```text
Baseline first:
  standard PyTorch / Transformers / Diffusers behavior
  SDPA where supported
  eager fallback where needed
  no forced third-party CUDA attention path during initial recovery

Acceleration later:
  FlashAttention / FA3 / FA4
  xFormers
  SageAttention
  FlashInfer
  Triton kernels
  custom CUDA extensions
  quantization kernels
```

Why this matters: a package can install successfully and still be the wrong binary target for the assigned GPU. A Python import only proves that Python can import the package. It does not prove that the package contains a working Blackwell `sm_120` kernel for the exact operation that your Space will execute.

### 11.1 FlashAttention, FA2, FA3, and FA4

FlashAttention can be correct and fast, but it is not a safe first assumption during a ZeroGPU hardware migration.

Treat all FlashAttention paths as optional until verified:

- `attn_implementation="flash_attention_2"`;
- `attn_implementation="flash_attention_3"`;
- direct imports such as `from flash_attn import ...`;
- direct wheel URLs;
- model remote code that checks for FlashAttention at import time;
- container images or serving stacks that bundle FlashAttention inside the runtime image.

The older ZeroGPU FA3 material was written for the H200 / Hopper period, not the current Blackwell baseline. The [ZeroGPU AOTI / FA3 blog material](https://huggingface.co/blog/zerogpu-aoti) discusses an H200-era optimization path, while current ZeroGPU documentation describes [RTX PRO 6000 Blackwell allocation](https://huggingface.co/docs/hub/spaces-zerogpu). That does not mean FA3 can never work on Blackwell. It means FA3 should not be treated as the default recovery path.

The practical rule is simple:

```text
FA3 in H200-era ZeroGPU material:
  valid historical optimization context

FA3 on Blackwell ZeroGPU:
  optional acceleration path
  must be explicitly verified
  remove it during first recovery
```

Blackwell reports around FlashAttention, FA3, and FA4 are still uneven. Issues such as [flash-attention #1987](https://github.com/Dao-AILab/flash-attention/issues/1987), [flash-attention #2307](https://github.com/Dao-AILab/flash-attention/issues/2307), and related Blackwell threads are useful warning signs: support may depend on exact GPU, CUDA, PyTorch, Python, build flags, wheel source, and ABI. A Space repair guide should not rely on these paths unless the exact runtime has been tested.

First repair:

```text
1. Remove mandatory FlashAttention import paths.
2. Remove hard-coded flash attention backend selection.
3. Try SDPA.
4. If SDPA fails, try eager.
5. Only after the Space works, test FlashAttention as an optional acceleration layer.
```

Search patterns:

```text
flash_attn
flash-attn
flash_attention_2
flash_attention_3
attn_implementation="flash_attention_2"
attn_implementation="flash_attention_3"
use_flash_attn
use_flash_attention
```

Do not interpret `pip install flash-attn` success as runtime confirmation. The real confirmation is a small model forward pass or generation path that actually enters the attention kernel.

### 11.2 xFormers, especially in Diffusers and image Spaces

xFormers deserves special handling because many Diffusers, Stable Diffusion, ComfyUI-style, and image generation Spaces copied old optimization snippets that enable xFormers unconditionally.

The most important pattern to search for is:

```python
pipe.enable_xformers_memory_efficient_attention()
```

Also search for:

```text
xformers
memory_efficient_attention
xformers.ops.memory_efficient_attention
```

In a Blackwell recovery pass, treat xFormers as a suspect before treating the whole model as broken. xFormers is an acceleration layer, not the baseline. A Space should first work without it.

Diffusers-style repair order:

```text
1. Remove or guard `enable_xformers_memory_efficient_attention()`.
2. Use the default PyTorch / Diffusers attention path.
3. Prefer SDPA where the stack supports it.
4. Use attention slicing, VAE slicing, VAE tiling, or lower resolution for memory pressure.
5. Reintroduce xFormers only after checking exact compatibility.
```

Example guarded pattern:

```python
USE_XFORMERS = os.environ.get("USE_XFORMERS", "0") == "1"

if USE_XFORMERS:
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers attention enabled")
    except Exception as exc:
        print("xFormers unavailable; continuing without it:", repr(exc))
```

For emergency recovery, the default should be `USE_XFORMERS=0`.

Why this matters: image Spaces often fail during first generation, not at import time. The pipeline loads, the UI appears, and then the attention kernel fails when the user submits a prompt. That timing can make the problem look like a model, VRAM, or prompt issue. It may be only the xFormers path.

External Blackwell reports such as [diffusers #13043](https://github.com/huggingface/diffusers/issues/13043), [xformers #1329](https://github.com/facebookresearch/xformers/issues/1329), and related Diffusers/ComfyUI reports are not ZeroGPU-specific proof by themselves, but they are strong analogues for the same class of failure: a Blackwell GPU is visible, but a specific acceleration kernel is not safe.

### 11.3 SageAttention and `sage_hub`

SageAttention-like paths should be treated like FlashAttention or xFormers: optional acceleration, not the initial recovery baseline.

Search for:

```text
sageattention
SageAttention
sage_hub
set_attention_backend("sage")
```

Diffusers reports such as [diffusers #13043](https://github.com/huggingface/diffusers/issues/13043) show the useful diagnostic pattern: the failure can appear only when a specific attention backend is selected. If the Space works with SDPA or eager but fails with `sage_hub`, that is not a general model failure. It is an acceleration backend failure.

Recovery posture:

```text
1. Disable SageAttention / sage_hub.
2. Restore default attention or SDPA.
3. Confirm a real inference pass.
4. Re-enable only if the exact backend supports Blackwell in the deployed environment.
```

### 11.4 Triton kernels

Triton is not one thing. PyTorch itself depends on Triton versions for parts of its stack, and model/application code can also include custom Triton kernels.

Search for:

```text
import triton
import triton.language as tl
@triton.jit
triton.jit
```

A custom Triton kernel is architecture-sensitive until proven otherwise. It can compile and still fail or crash at runtime. PyTorch issue threads such as [pytorch #176426](https://github.com/pytorch/pytorch/issues/176426) are useful reminders that Blackwell compatibility is not only about detecting the device. Runtime kernel behavior still matters.

Recovery posture:

```text
If Triton is optional:
  disable it first
  use a PyTorch fallback

If Triton is required:
  isolate the smallest failing call
  check torch / triton version alignment
  check whether the kernel assumes older architectures
  avoid rebuilding large kernels during every Space startup
```

Do not solve a Triton failure by blindly pinning an older Triton version. PyTorch 2.8-era stacks may expect a matching Triton version. A mismatched Triton pin can create a second failure.

### 11.5 FlashInfer, serving backends, and bundled kernels

A Space may not import FlashAttention directly but still fail inside a serving stack that bundles or selects a compiled attention kernel. Text Generation Inference, vLLM, FlashInfer, and quantization-serving stacks can all carry their own kernel compatibility assumptions.

Use this rule:

```text
If the stack hides kernel selection, test the stack's own Blackwell support.
Do not assume that PyTorch support automatically means every serving backend is safe.
```

Relevant analogue reports include [text-generation-inference #3342](https://github.com/huggingface/text-generation-inference/issues/3342), [vLLM #16901](https://github.com/vllm-project/vllm/issues/16901), and [FlashInfer #2555](https://github.com/flashinfer-ai/flashinfer/issues/2555). The exact fix differs by stack. The shared lesson is the same: bundled kernels can lag behind the visible PyTorch runtime.

### 11.6 Quantization backends: bitsandbytes, FP8, AWQ, GPTQ, and friends

Do not limit the investigation to attention. Quantization layers can launch their own CUDA kernels.

Search for:

```text
bitsandbytes
bnb
load_in_4bit
load_in_8bit
BitsAndBytesConfig
awq
gptq
exllama
marlin
fp8
```

If the error appears during model load, first token generation, or first matrix multiplication after a quantized model loads, suspect the quantization backend.

Recovery posture:

```text
1. Test the same model without the quantization backend if possible.
2. Test a smaller non-quantized model to confirm the runtime.
3. Check whether the quantization package has Blackwell support.
4. Avoid adding multiple quantization packages during first recovery.
```

A `no kernel image` error from quantization should not be repaired by lowering image resolution or sequence length. It is usually not a capacity error. It is a binary compatibility error.

### 11.7 Custom CUDA extensions and ComfyUI-style nodes

Custom CUDA extensions are high risk during Blackwell migration.

Search for:

```text
CUDAExtension
cpp_extension
setup.py
load_inline
.cu
TORCH_CUDA_ARCH_LIST
custom_ops
custom_nodes
```

If the extension was compiled without Blackwell architecture support, it may fail at runtime. Rebuilding inside a Space may also fail due to build tools, CUDA toolkit availability, time limits, or storage constraints.

External ComfyUI-style reports such as [ComfyUI-Impact-Pack #1179](https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/1179) are useful analogues: precompiled kernels can exclude `sm_120`, and a PyTorch-only fallback may be the practical recovery path.

Repair posture:

```text
1. Disable the custom CUDA extension.
2. Use the slower Python/PyTorch path if available.
3. Confirm the Space runs.
4. Only then rebuild or replace the extension.
```

### 11.8 Blackwell acceleration risk matrix

| Component | Common Space family | Failure style | First recovery action |
|---|---|---|---|
| FlashAttention / FA2 | LLM, VLM, TTS, remote-code models | import failure, `no kernel image`, invalid device function | Remove mandatory flash path; use SDPA/eager |
| FA3 | H200-era optimized Spaces, video/LLM demos | may be Hopper/H200-oriented; unsafe as default on Blackwell | Treat as optional; disable first |
| FA4 | newer attention experiments | still uneven on RTX PRO 6000 Blackwell reports | Do not use as recovery baseline |
| xFormers | Diffusers, SD, ComfyUI-style image Spaces | generation-time failure, NotImplementedError, kernel mismatch | Remove `enable_xformers_memory_efficient_attention()` |
| SageAttention / `sage_hub` | Diffusers and video/image pipelines | backend-specific `no kernel image` | Disable backend; test SDPA/default |
| Triton custom kernels | video, image, custom research demos | compile/runtime mismatch, segfault, invalid device function | Disable optional kernel; check torch/triton alignment |
| FlashInfer / vLLM / TGI | serving-style LLM Spaces | bundled kernel mismatch | check backend Blackwell support separately |
| bitsandbytes / quantization | LLM, VLM, TTS | load or first-generation kernel failure | test without quantization backend |
| Custom CUDA extension | ComfyUI nodes, research repos | build or runtime architecture mismatch | PyTorch fallback first |

The key distinction is not whether an accelerator is good or bad. The key distinction is whether it has been verified for the exact runtime assigned to the Space.

---

## 12. PyTorch 2.8+, CUDA 12.8-Class Wheels, and Dependency Drift

Current ZeroGPU documentation lists PyTorch support from `2.8.0` upward. The Blackwell documentation update also moved the ZeroGPU baseline to a PyTorch 2.8+ generation. This makes PyTorch drift a first-class recovery issue, not a secondary detail.

For general PyTorch outside ZeroGPU, PyTorch forum maintainers have stated that Blackwell GPUs are supported by CUDA 12.8-enabled PyTorch builds, and PyTorch's own install matrix is the normal source for selecting a supported binary. For ZeroGPU, follow the [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) first, because the platform may enforce its own supported versions and Python combinations.

Practical rule:

```text
ZeroGPU Blackwell repair target:
  Python: 3.10.13 or 3.12.12
  PyTorch: ZeroGPU-supported 2.8.0+ version
  CUDA wheel family: CUDA 12.8-class where applicable
  Attention: SDPA first, eager fallback, architecture-specific kernels last
```

### 12.1 Why PyTorch drift is heavier after the Blackwell switch

Before Blackwell, an old Space might have survived with older torch stacks because its assigned GPU target matched what those wheels knew how to run. After Blackwell, the same old stack can become invalid.

The important failure is not just:

```text
torch is old
```

It is more precise:

```text
torch / torchvision / torchaudio / triton / CUDA wheel family
no longer match the assigned Blackwell sm_120 runtime
```

This can happen even if the repository code did not change.

Common drift paths:

- an old `requirements.txt` pins `torch==2.0.1`, `2.1.*`, `2.4.*`, or `2.5.*`;
- `xformers` pulls a torch version older than the current ZeroGPU baseline;
- a direct PyTorch wheel index still points to `cu118`, `cu121`, or `cu124`;
- `torchvision` or `torchaudio` is pinned to a version from a different torch generation;
- `triton` is pinned separately and conflicts with the torch version;
- a Space was dormant, then rebuilt and resolved a different dependency set;
- a package install script runs at startup and mutates the runtime;
- the Space duplicates an old lockfile into a new Blackwell runtime.

### 12.2 Device detection is not enough

Add this to the guide as a hard rule:

```text
CUDA detection is not proof of runtime compatibility.
```

A Space can print a Blackwell GPU name, report `torch.cuda.is_available() == True`, and still fail when a compiled kernel launches.

Use a tiered check:

```text
Level 0: Python imports torch
Level 1: torch.cuda.is_available() is True
Level 2: device name and capability show Blackwell / sm_120
Level 3: a tiny tensor operation runs on CUDA
Level 4: a small model forward pass runs
Level 5: the real model performs one minimal inference
Level 6: the intended acceleration backend also works
```

Levels 0 to 2 are useful diagnostics. They are not final confirmation.

Example runtime check:

```python
import torch

print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    x = torch.ones((8, 8), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print("tiny cuda matmul ok", y[0, 0].item())
```

Then test the real path. If the tiny tensor test passes but the model fails, the failure may be in attention, quantization, custom CUDA, or model-specific kernels.

### 12.3 Check torch generation and companion packages together

Do not inspect `torch` alone.

Check at least:

```python
import importlib.metadata as im

for pkg in [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "transformers",
    "diffusers",
    "accelerate",
    "xformers",
    "flash-attn",
    "bitsandbytes",
    "gradio",
    "spaces",
]:
    try:
        print(pkg, im.version(pkg))
    except im.PackageNotFoundError:
        print(pkg, "not installed")
```

Why this matters: PyTorch is part of a small compatibility cluster. If torch is updated but torchvision is from an older generation, a different error may appear. If Triton is pinned independently, a kernel path may break. If xFormers forces an older torch, the repair may regress silently.

### 12.4 Suspect old torch and old CUDA wheel indexes

Dangerous patterns during current ZeroGPU Blackwell recovery:

```text
torch==2.0.1
torch==2.1.*
torch==2.4.*
torch==2.5.*
torchvision==0.15.*
torchvision==0.16.*
torchvision==0.17.*
torchvision==0.18.*
torchvision==0.19.*
xformers==0.0.20
--index-url https://download.pytorch.org/whl/cu118
--extra-index-url https://download.pytorch.org/whl/cu118
--index-url https://download.pytorch.org/whl/cu121
--extra-index-url https://download.pytorch.org/whl/cu121
--index-url https://download.pytorch.org/whl/cu124
--extra-index-url https://download.pytorch.org/whl/cu124
```

Not every old package is automatically wrong in every environment. But if the target is current ZeroGPU Blackwell, old torch/CUDA wheel families are suspects until proven safe.

### 12.5 Avoid accidental downgrades from transitive dependencies

A maintainer may not pin torch directly. Another package can still pull it.

Common sources:

- old xFormers pins;
- old Gradio demo requirements copied from an older Space;
- old Diffusers example requirements;
- ComfyUI custom node requirements;
- packages that list strict `torch<...` constraints;
- direct `pip install` commands in `app.py`, `startup.sh`, or notebooks copied into a Space.

Search:

```text
torch==
torch<=
torch<
torchvision==
torchaudio==
xformers==
--extra-index-url
--index-url
download.pytorch.org/whl/cu
pip install torch
pip install xformers
pip install flash-attn
```

If a package pulls an old torch generation, remove the package first if possible. Do not fight the resolver by adding more and more constraints before establishing a working baseline.

### 12.6 Triton is part of the torch drift story

Torch and Triton should not be treated as unrelated.

A real-world Hunyuan-GameCraft-style requirements pattern showed an important class of failure: a requested Triton version can conflict with the Triton version expected by torch. In one public Space requirements file, the comment states that torch 2.8.0 depends on `triton==3.4.0`, and the file aligns to that version.

That does not mean every ZeroGPU Space must manually pin Triton. It means this:

```text
If you pin torch, do not independently pin an incompatible Triton.
If you pin Triton, verify that torch expects that Triton generation.
If custom Triton kernels exist, test them separately.
```

Search for:

```text
triton==
triton>=
triton<=
```

During first recovery, prefer letting the supported torch stack bring the matching Triton unless you have a known reason to pin it.

### 12.7 Python version drift: cp310, cp312, and cp311 uncertainty

Current ZeroGPU documentation lists Python `3.10.13` and `3.12.12`. Treat those as the safe documented targets.

Why this matters: compiled wheels are tagged by Python ABI. A wheel built for `cp310` is not the same as a wheel built for `cp312`. A package that has a cp310 wheel may not have a cp312 wheel, or the cp312 wheel may be newer and less tested in the exact Space pattern.

Recovery posture:

```text
If the Space is old or wheel-heavy:
  try Python 3.10.13 first

If the Space is newer and dependency-light:
  Python 3.12.12 may be acceptable

If Python 3.11 appears:
  treat it as undocumented/uncertain for current ZeroGPU unless verified
```

Do not mix a Python-version migration with too many other changes. If the Space is already broken from Blackwell migration, keep Python stable unless Python itself is part of the failure.

### 12.8 Requirements recovery: remove first, then pin

A common mistake is to solve a Blackwell migration by adding more packages immediately. That can make the resolver state worse.

Better first-pass repair:

```text
1. Remove old torch pins.
2. Remove old xFormers pins.
3. Remove direct old PyTorch wheel indexes.
4. Remove mandatory FlashAttention pins.
5. Let the platform-supported torch generation install.
6. Confirm a tiny CUDA op.
7. Confirm a minimal model path.
8. Then pin the working versions.
```

After recovery, pin the important pieces. Before recovery, avoid over-pinning the broken state.

### 12.9 Minimal temporary requirements pattern

A temporary recovery requirements file should be boring.

Example shape:

```text
# Prefer the current ZeroGPU supported runtime.
# Do not add flash-attn, xformers, or custom CUDA packages during first recovery.

transformers>=4.51
diffusers>=0.34
accelerate
gradio
spaces
```

This is not a universal lock. It is a recovery posture. The exact versions should be chosen for the model family and then pinned after the Space works.

For a model that requires a specific Transformers or Diffusers version, pin that package. But avoid pinning torch-era packages until the current ZeroGPU baseline is confirmed.

### 12.10 When to pin torch explicitly

Pin torch explicitly when:

- the platform does not provide the expected torch generation;
- a dependency tries to downgrade torch;
- a reproducibility requirement needs exact versions;
- the working state has been confirmed and should be preserved.

Do not pin torch explicitly just because the Space is broken. First determine whether the existing pin is the reason it is broken.

If you do pin, pin the compatibility cluster, not torch alone:

```text
torch
torchvision
torchaudio
triton
Python version
CUDA wheel family
```

### 12.11 Typical PyTorch drift symptoms

PyTorch drift can appear as:

```text
NVIDIA GeForce RTX ... with CUDA capability sm_120 is not compatible with the current PyTorch installation
no kernel image is available for execution on the device
invalid device function
CUDA error: invalid argument
undefined symbol in a torch extension
torchvision operator does not exist
Triton compile/runtime failure
xFormers requires a different torch version
flash-attn was compiled against a different torch/CUDA/Python combination
```

The exact message matters. A `torchvision` operator error is not repaired the same way as a FlashAttention kernel error. But both can be caused by the same underlying event: old dependency assumptions meeting a new runtime.

### 12.12 Good logs for PyTorch drift

Ask for or print:

```python
import sys, platform, torch

print("python", sys.version)
print("platform", platform.platform())
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
```

Then print package versions as shown above.

Why this matters: when this guide is used as context for another maintainer or an AI assistant, those few lines prevent guesswork. They reveal whether the problem is likely torch generation, CUDA wheel family, Python ABI, or a higher-level acceleration backend.

### 12.13 Safe interpretation of PyTorch sources

Use the source hierarchy carefully:

1. Current ZeroGPU documentation for the platform baseline.
2. PyTorch install matrix for general supported PyTorch binaries.
3. PyTorch forum maintainer comments for current support state and edge cases.
4. GitHub issues for real-world failure patterns.
5. Random blog posts and forum fragments only as secondary clues.

Do not override ZeroGPU's documented baseline with a random local workstation workaround. A local RTX 5090 fix may not apply to a managed ZeroGPU Space.


### 12.14 Torch family version matrix for current Blackwell recovery

In many Spaces, installing `torch==2.8` together with an unpinned `torchvision` may appear to resolve to a compatible pair. Do not depend on that during recovery. Make the torch-family matrix explicit, because ZeroGPU Blackwell failures often come from a mismatched torch cluster rather than from torch alone.

Use the official [PyTorch previous versions install matrix](https://pytorch.org/get-started/previous-versions/) as the source for exact install commands. For the current ZeroGPU Blackwell generation, the most important family to preserve is the CUDA 12.8 torch family.

| Torch | TorchVision | TorchAudio | CUDA wheel index | Practical use |
|---|---|---|---|---|
| `2.8.0` | `0.23.0` | `2.8.0` | `https://download.pytorch.org/whl/cu128` | documented PyTorch 2.8 CUDA 12.8 family; important baseline for ZeroGPU Blackwell recovery |

Do not copy this table blindly into every Space. Use it as a recovery guardrail. The current ZeroGPU documentation also lists newer supported PyTorch versions, including `2.9.1`, `2.10.0`, and `2.11.0`; when using those versions, check the current PyTorch install matrix and pin the matching `torchvision`, `torchaudio`, CUDA wheel index, and any explicitly pinned `triton` version as a set. Do not infer companion package versions from memory.

Bad pattern:

```text
torch==2.8.0
torchvision
```

Better explicit pattern, when manual pinning is needed:

```text
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
--index-url https://download.pytorch.org/whl/cu128
```

If the platform already supplies a working torch stack, you may not need to pin torch explicitly. But if a dependency downgrades torch, or if a build log shows the resolver choosing an unexpected package, pin the family as a set.

### 12.15 Legacy `.pth` / `.pt` checkpoint loading after PyTorch 2.6+

Many old Spaces load `.pth` or `.pt` checkpoints with `torch.load(path)`. That can now fail even when the file exists and the GPU is fine.

PyTorch changed the default behavior of `torch.load()` so that `weights_only=True` is the default in PyTorch 2.6+. The change is a security hardening measure, because pickle-based model loading can execute arbitrary code. The PyTorch developer discussion explains the default flip and the security reason behind it, and downstream issues show real packages hitting `Weights only load failed` after upgrading.

This matters for ZeroGPU Blackwell because the hardware migration also moves old Spaces into a newer PyTorch generation. A Space can fail during checkpoint loading before it reaches any CUDA kernel.

Common old patterns:

```python
ckpt = torch.load("model.pth")
model = torch.load("model.pt")
state = torch.load(path, map_location="cuda")
```

Failure clues:

```text
Weights only load failed
Unsupported global
WeightsUnpickler error
This file can still be loaded, but only if you trust the source
```

Recovery order:

```text
1. Prefer safetensors if available.
2. Prefer a plain state_dict checkpoint.
3. Recreate the model class in code, then load the state_dict.
4. Use safe globals only for trusted classes and trusted files.
5. Use weights_only=False only for checkpoints from a trusted source.
```

Safer state-dict style:

```python
import torch

model = build_model_somehow()
state = torch.load("model_state_dict.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

Trusted legacy fallback:

```python
# Only for checkpoints from a trusted source.
ckpt = torch.load("legacy_model.pt", map_location="cpu", weights_only=False)
```

Do not treat this as a Blackwell CUDA problem. If the error occurs before model weights move to GPU, it is a checkpoint serialization problem.

### 12.16 TorchVision and PyTorch API drift

A Space can break on a newer PyTorch-family runtime without ever launching a CUDA kernel. TorchVision and torch APIs change over time, and old warnings can turn into hard errors after a runtime refresh.

Important examples:

- `pretrained=True` in `torchvision.models` is the old style. Use `weights=...` with the appropriate weights enum. TorchVision's model docs explain the weights API.
- TorchVision video decoding and encoding APIs such as `read_video`, `write_video`, and `VideoReader` have been deprecated in favor of TorchCodec. TorchVision 0.24 documentation explicitly warns that video decoding/encoding functionality is deprecated and moving out of TorchVision.
- `torch.load()` changed default behavior in PyTorch 2.6+, as described above.
- Some transforms, model constructors, and enum names can differ across TorchVision generations.

Old pattern:

```python
from torchvision import models
model = models.resnet50(pretrained=True)
```

Newer pattern:

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
```

Old video pattern:

```python
from torchvision.io import read_video
frames, audio, info = read_video("input.mp4")
```

Recovery direction:

```text
If a Space uses TorchVision video IO, treat it as API drift.
Move video decode/encode responsibility to a supported library such as TorchCodec, imageio, decord, OpenCV, or ffmpeg-based code depending on the app.
```

Do not overfit these examples. The broader rule is:

```text
Deprecation warnings are not cosmetic during runtime migration.
A warning that was harmless on the old runtime may become an import error, missing API, or changed behavior after the Space wakes up on a newer stack.
```

### 12.17 OpenCV / `cv2` drift in image and video Spaces

OpenCV failures are common in image/video Spaces, and they are usually dependency or wheel problems rather than Blackwell kernel problems.

Use `opencv-python-headless` for server-side Spaces unless the app truly needs GUI windows. The PyPI page for `opencv-python-headless` keeps the import name as `cv2`, so the package name and import name are intentionally different.

Common bad patterns:

```text
opencv-python
opencv-python-headless
opencv-contrib-python
opencv-contrib-python-headless
```

Do not install multiple OpenCV wheel families together. Pick one.

Preferred Spaces pattern:

```text
opencv-python-headless
```

Common failure classes:

```text
import cv2 fails
libGL.so.1 not found
Qt platform plugin error
highgui / GUI function failure
NumPy ABI mismatch
opencv wheel constrains numpy differently from the rest of the app
```

NumPy 2.x drift matters. OpenCV Python had a visible transition period where wheels built against older NumPy ABIs could fail with NumPy 2.x, and newer OpenCV Python wheels moved toward NumPy 2-compatible builds. If a Space updates OpenCV or NumPy after a rebuild, `import cv2` may be the first thing to fail.

Recovery direction:

```text
1. Use opencv-python-headless.
2. Remove other OpenCV wheel families.
3. Check NumPy version.
4. Avoid GUI functions such as cv2.imshow in Spaces.
5. Pin OpenCV and NumPy together if the resolver keeps changing them.
```

Diagnostic:

```python
import cv2, numpy as np
print("cv2", cv2.__version__)
print("numpy", np.__version__)
```

### 12.18 ONNX Runtime GPU drift

`onnxruntime-gpu` is another binary stack boundary. It is not repaired by changing PyTorch attention settings.

The official ONNX Runtime CUDA Execution Provider documentation states that ONNX Runtime GPU packages require compatible CUDA and cuDNN versions, and that CUDA 12.x packages are compatible across CUDA 12.x minor versions but cuDNN 8.x and 9.x are not mutually compatible. The install docs also point users to CUDA and cuDNN requirements for GPU packages.

Common failure classes:

```text
Failed to create CUDAExecutionProvider
CUDAExecutionProvider silently unavailable
provider falls back to CPU
libonnxruntime_providers_cuda.so load failure
missing libcublasLt.so.12
Require cuDNN 9.* and CUDA 12.*
invalid PTX on sm_120
```

Blackwell adds one more layer: even if the CUDA/cuDNN provider loads, the runtime or provider may still need to support the Blackwell target. ONNX Runtime issue reports include `sm_120` / invalid PTX failures, so treat ONNX Runtime GPU as a separately verified path.

Diagnostic:

```python
import onnxruntime as ort

print("onnxruntime", ort.__version__)
print("available providers", ort.get_available_providers())
```

Recovery direction:

```text
1. Confirm whether CUDAExecutionProvider is available.
2. Confirm whether the session actually uses CUDAExecutionProvider.
3. Check ORT, CUDA, and cuDNN major versions together.
4. If GPU provider fails, test CPUExecutionProvider to separate model validity from provider validity.
5. If Blackwell PTX/kernel errors remain, update ORT or keep CPU fallback until a compatible GPU path is available.
```

Example safe isolation:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)
print(session.get_providers())
```

If CPU works and CUDA does not, the model may be valid. The failure is likely provider/runtime compatibility.

### 12.19 Blackwell drift triage by phase

Use this phase table to avoid mixing unrelated fixes.

| Failing phase | Likely class | First check | First repair direction |
|---|---|---|---|
| Build | resolver, wheel, Python ABI | `pip` log, wheel tags, torch index URL | simplify requirements; pin compatible family |
| Import | API drift or binary import failure | full traceback before CUDA init | update API use; remove conflicting binary packages |
| Checkpoint load | `.pth` / pickle / `weights_only` | `Weights only load failed` | safetensors/state_dict/trusted load path |
| Device detection | torch/CUDA visibility | torch version and CUDA availability | correct torch CUDA wheel |
| Tiny CUDA op | core torch / `sm_120` support | matrix multiplication on CUDA | torch 2.8+ / cu128 family |
| Model forward | model code / attention / quantization | disable acceleration | SDPA/eager, no quantization first |
| First generation | attention or custom kernel | `no kernel image`, invalid function | remove flash/xFormers/Triton/custom CUDA |
| Video decode | TorchVision/OpenCV/ffmpeg drift | import and decode path | TorchCodec/headless OpenCV/ffmpeg path |
| ONNX inference | ORT provider | available providers | match ORT/CUDA/cuDNN or CPU fallback |

This table is intentionally redundant. Redundancy helps when the guide is attached to another troubleshooting conversation or handed to a maintainer who has only partial logs.

---

### 12.20 CUDA 12.8 boundary failures

On Blackwell ZeroGPU, CUDA 12.8 is not just another version number. It is a compatibility boundary between the assigned GPU architecture, the PyTorch wheel family, CUDA extension wheels, cuDNN, Triton, FlashAttention, ONNX Runtime, quantization backends, and any package that ships compiled kernels.

NVIDIA describes CUDA Toolkit 12.8 as the first CUDA Toolkit release with Blackwell support across the developer toolchain. PyTorch 2.7 introduced prototype Blackwell support and CUDA 12.8 wheels, while the current ZeroGPU documentation points users toward a newer PyTorch generation. For this guide, the practical recovery target is the current ZeroGPU baseline, not the oldest Blackwell-capable torch version.

Common CUDA 12.8 boundary symptoms:

```text
sm_120 is not compatible with the current PyTorch installation
no kernel image is available for execution on the device
device kernel image is invalid
invalid device function
invalid PTX
CUDA available is True, but the first real CUDA operation fails
a package imports successfully, but fails when its CUDA extension is first called
```

First repair direction:

```text
1. Print torch, CUDA, Python, torchvision, torchaudio, and triton versions.
2. Confirm the torch wheel is a CUDA 12.8-class wheel, not CPU-only and not cu121/cu124/cu126.
3. Confirm companion packages match the torch family.
4. Disable optional compiled acceleration.
5. Run a tiny CUDA tensor operation.
6. Run a tiny model forward pass.
7. Only then re-enable attention, quantization, or serving acceleration.
```

A useful rule: CUDA device detection is not proof that all compiled kernels are valid. Detection proves that PyTorch can see the GPU. It does not prove that FlashAttention, xFormers, ONNX Runtime, bitsandbytes, Triton, custom CUDA extensions, or quantized matrix multiplication kernels were built for Blackwell.

### 12.21 Blackwell dependency rollback and install-order drift

A Space can start with a compatible PyTorch stack and lose it later in the same build. This happens when a later package install pulls an older torch version, switches the environment to a CPU-only torch wheel, or installs a CUDA wheel built for a different CUDA generation.

This is especially likely when requirements include broad or old dependencies such as:

```text
xformers
flash-attn
bitsandbytes
vllm
tensorrt-llm
mmcv
onnxruntime-gpu
custom GitHub installs
old ComfyUI nodes
old Stable Diffusion WebUI extensions
```

Danger signs in build logs:

```text
Collecting torch<2.8
Collecting torch==2.4.*
Collecting torch==2.5.*
Installing collected packages: torch
Attempting uninstall: torch
Successfully uninstalled torch-2.8.0+cu128
Successfully installed torch-...+cpu
Successfully installed torch-...+cu121
Successfully installed torch-...+cu124
```

Recovery posture:

```text
1. Install the torch family deliberately.
2. Install fragile optional libraries only after the torch family is correct.
3. Use `--no-deps` only when you understand the dependency graph and want to prevent torch rollback.
4. Re-run `pip freeze` after the final install step, not before it.
5. Do not assume that the first torch version printed during build is the final torch version used at runtime.
```

Example inspection snippet:

```python
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
```

If the build log shows a compatible torch installation followed by another package replacing it, do not debug attention first. Fix the install order or dependency pins first.

### 12.22 Common Spaces library drift map

Blackwell drift often appears through helper libraries rather than through the top-level app code. A Space may contain only a few lines of model-loading code, while the actual failure comes from an attention backend, quantization backend, provider backend, or dispatch helper selected by a library.

Use this map during triage:

| Library / area | Why it can drift on Blackwell | First recovery move |
|---|---|---|
| `torch` | old wheel lacks `sm_120` support | use current ZeroGPU torch baseline |
| `torchvision` | companion version mismatch, deprecated APIs, video IO changes | align with torch matrix; update API use |
| `torchaudio` | missing companion wheel, import-time dependency failures | align with torch matrix or remove if unused |
| `triton` | custom kernels, `torch.compile`, Inductor path changes | disable compile/custom Triton first |
| `transformers` | `attn_implementation`, remote code, `device_map`, quantization config | use SDPA/eager and plain placement first |
| `accelerate` | device dispatch and `device_map="auto"` can fail before inference | test explicit device placement or CPU load path |
| `diffusers` | attention backend, xFormers, SageAttention, quantized transformer blocks | disable optional acceleration first |
| `xformers` | architecture-specific attention kernels and torch coupling | remove or rebuild only after baseline works |
| `flash-attn` / FA3 / FA4 | kernel support may lag Blackwell | treat as optional, verified acceleration |
| `bitsandbytes` | 4-bit/8-bit kernels and backend routing | test non-quantized load first |
| `torchao` | FP8/FP4 paths are promising but hardware/backend-specific | use after recovery, not during first repair |
| `onnxruntime-gpu` | CUDA/cuDNN/provider compatibility | test CPU provider first, then CUDA provider |
| `vLLM` / TGI | bundled kernels, FlashAttention, CUTLASS, quantization paths | verify Blackwell support for that release |
| `llama.cpp` / GGUF runners | quantized matmul kernels may be architecture-sensitive | use CPU/cuBLAS fallback or updated build |
| `opencv-python` / `cv2` | NumPy ABI, GUI libraries, wheel constraints | prefer headless and align NumPy |
| legacy `.pth` / `.pt` | PyTorch 2.6+ `weights_only=True` default | use safetensors/state_dict/trusted load path |

Do not treat this as a list of packages to install. It is a list of packages to suspect when a Space fails after the hardware/runtime boundary moved.

### 12.23 Quantization drift: bitsandbytes, torchao, FP8, and serving backends

Quantization is not a neutral memory-saving option during Blackwell recovery. It often adds compiled kernels, backend routing, compute capability checks, and torch/CUDA version assumptions.

Common quantization-related failure classes:

```text
bitsandbytes 4-bit or 8-bit load fails with no kernel image
model loads in full precision but fails when quantization_config is enabled
vLLM or TGI fails only for quantized models
GGUF runner sees the GPU but quantized matmul crashes
torchao FP8/FP4 path imports but fails during first forward
inference works but training or backward crashes
```

First recovery direction:

```text
1. Disable quantization and confirm the model can load.
2. If full precision is too large, test a smaller model with the same code path.
3. If CPU load works and GPU quantized load fails, treat the quantization backend as the suspect.
4. Check whether the backend explicitly supports Blackwell / `sm_120`.
5. Reintroduce quantization only after the standard path works.
```

For Diffusers and image/video Spaces, this means testing without `bitsandbytes`, experimental FP8/FP4 settings, custom quantized transformer blocks, or backend-specific attention first. For LLM Spaces, it means separating the model loader from `BitsAndBytesConfig`, AWQ/GPTQ loaders, vLLM quantization, TGI quantization, GGUF MMQ kernels, and torchao experiments.

### 12.24 Accelerate, `device_map`, and dispatch drift

`device_map="auto"` is convenient, but it is not a neutral baseline. It delegates placement and weight materialization through helper code, and failures may occur during loading rather than during inference.

Symptoms:

```text
segmentation fault during from_pretrained
exit code 139
crash during weight dispatch
no Python traceback
failure only when device_map="auto" is enabled
failure only with quantization_config + device_map="auto"
```

First recovery direction:

```text
1. Try CPU load or explicit device placement for a smaller model.
2. Remove quantization while testing dispatch.
3. Avoid combining `device_map="auto"`, quantization, and custom attention during first recovery.
4. Print package versions for transformers, accelerate, torch, and bitsandbytes.
5. Restore automatic dispatch only after the plain model path works.
```

This matters because a crash during weight materialization can look like a CUDA kernel problem, an OOM problem, or a model corruption problem. The failure phase is the clue. If it crashes before the first forward pass, attention kernels may not be the first suspect.

### 12.25 Community wheels and binary artifact policy

During a hardware transition, community wheels can be useful. They can also make a Space harder to reproduce, audit, or repair.

Use a community wheel only after the official path and standard fallback path are understood. A wheel is not just a package name. It encodes assumptions about:

```text
operating system
CPU architecture
Python ABI tag: cp310, cp311, cp312, ...
PyTorch version
CUDA version
glibc / manylinux tag
GPU architecture target: sm_90, sm_100, sm_120, ...
C++ ABI and compiler assumptions
```

Rules:

```text
1. Prefer official wheels when they exist.
2. Prefer standard PyTorch/Transformers fallback when official acceleration wheels do not exist.
3. Treat H200/Hopper wheels as H200/Hopper wheels, not as Blackwell wheels.
4. Use Linux x86_64 wheels for Spaces, not Windows wheels.
5. Match Python tag and CUDA tag explicitly.
6. Pin exact URLs only when necessary.
7. Prefer hashes when the installation method supports them.
8. Record why the wheel is used and what it replaces.
```

A community wheel built for a previous ZeroGPU H200/Hopper path is not automatically safe on current Blackwell ZeroGPU.

### 12.26 GGUF, llama.cpp, Ollama-style runners, and quantized matmul

Some Spaces wrap external inference runners rather than using raw PyTorch. These may still fail at the Blackwell boundary because they ship CUDA kernels or compile CUDA backends outside the PyTorch wheel system.

Examples of risky areas:

```text
GGUF quantized matrix multiplication
llama.cpp CUDA backends
Ollama-style CUDA backends
vLLM CUTLASS paths
TGI bundled FlashAttention paths
custom CUDA graph or matmul kernels
```

The failure may not mention PyTorch. It may instead mention:

```text
device kernel image is invalid
invalid PTX
CUDA backend failed to load
MMQ kernel crash
CUTLASS / FlashAttention / cuBLAS path failure
```

Recovery direction:

```text
1. Confirm whether the runner uses its own CUDA backend.
2. Check whether the runner release explicitly supports Blackwell / `sm_120`.
3. Try a CPU fallback or cuBLAS fallback if available.
4. Disable custom quantized matmul kernels where possible.
5. Update the runner before changing the model.
```

This is a separate branch from PyTorch recovery. A torch-only test can pass while an external runner still fails on its bundled CUDA kernels.


### 12.27 CUDA 12.8 boundary recipes

Use this subsection when the log contains a CUDA architecture, PTX, kernel-image, or provider-load error and the Space is now running on Blackwell.

| Symptom | Most likely boundary | First recovery action | Why |
|---|---|---|---|
| `sm_120 is not compatible` | PyTorch wheel too old or wrong CUDA family | install a current ZeroGPU-supported torch family | the wheel does not contain support for the assigned architecture |
| `no kernel image is available for execution on the device` | compiled extension missing Blackwell kernels | remove optional CUDA extensions first | the app may be fine; one backend kernel may be invalid |
| `invalid PTX` | JIT or provider backend compiled for the wrong target | update provider or disable provider acceleration | PTX compatibility is not a full substitute for a correct Blackwell build |
| import works, first inference fails | runtime kernel path is selected lazily | run a tiny CUDA op and minimal forward pass | many libraries load compiled kernels only when first used |
| CPU fallback appears silently | CUDA provider failed or torch wheel is CPU-only | print provider list and wheel tags | a Space can look alive while not using the intended GPU path |
| build log uninstalls torch | dependency rollback | install fragile packages after torch, or use controlled `--no-deps` | install order can replace a working stack |

Use the current [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) as the platform baseline. Use the [PyTorch install matrix](https://pytorch.org/get-started/previous-versions/) to check the torch / torchvision / torchaudio / CUDA wheel family, and use NVIDIA's Blackwell migration notes such as the [CUDA 12.8 Blackwell guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330) to understand why CUDA 12.8-class rebuilds matter.

### 12.28 Frequent Spaces library drift matrix

This is the practical version of the drift map for repositories that use common Spaces libraries.

| Area | Typical old assumption | Blackwell-era failure | Conservative recovery |
|---|---|---|---|
| `transformers` | remote code chooses fast attention | FlashAttention / custom attention fails | pass `attn_implementation="sdpa"`, then try `eager` |
| `diffusers` | xFormers or custom attention is always safe | image/video pipeline fails inside attention | remove `enable_xformers_memory_efficient_attention()` first |
| `accelerate` | `device_map="auto"` always improves placement | dispatch or load-time crash | load on CPU or explicit CUDA first, then reintroduce dispatch |
| `bitsandbytes` | 4-bit/8-bit is a safe memory fix | quantized load fails before model runs | test full precision or CPU load first |
| `torchao` | FP8/FP4 is a drop-in speedup | backend gate, dtype, or compile issue | use only after standard inference works |
| `onnxruntime-gpu` | installing ORT GPU is enough | provider fails, CPU fallback, invalid PTX | print providers and test CPU provider first |
| `opencv-python` | desktop OpenCV wheel is fine | GUI library or NumPy ABI import failure | use `opencv-python-headless` and align NumPy |
| `vLLM` / TGI | serving image bundles correct kernels | warmup failure or FlashAttention crash | update serving stack or disable unsupported backend |
| `llama.cpp` / GGUF | old CUDA build is still adequate | quantized matmul path fails or is slow | rebuild/update for Blackwell or use fallback backend |
| custom CUDA | previous wheel worked on H200 | missing `sm_120` kernel | rebuild for Blackwell or provide PyTorch fallback |

Do not read this table as a recommendation to install all of these packages. Read it as a suspicion map. The first recovery path is usually to remove optional acceleration, prove that the standard path works, then reintroduce one optimization at a time.

### 12.29 Dependency rollback and install-order recipes

Blackwell recovery can fail even when the correct package appears earlier in the build log. The final environment is what matters.

Search build logs for these patterns:

```text
Attempting uninstall: torch
Successfully uninstalled torch-2.8.0+cu128
Successfully installed torch-2.5.*
Successfully installed torch-...+cpu
Collecting torch<2.8
Collecting torch==2.4.*
Collecting nvidia-cublas-cu12==...
```

Safer install posture:

```text
1. Avoid old torch pins during recovery.
2. Avoid broad optional accelerators in the first repair pass.
3. Install the torch family from the intended wheel index.
4. Install application libraries.
5. Install fragile CUDA extensions last, if still needed.
6. Re-print the final torch family after all installs finish.
```

A temporary recovery file can intentionally be boring:

```text
# requirements.txt recovery pass
# Keep this small until the Space boots and runs one minimal inference.

transformers
diffusers
accelerate
safetensors
huggingface_hub
gradio
spaces
```

Then add project-specific packages back one at a time. If a later package downgrades torch, either remove it, choose a newer version, or install it with carefully reviewed dependency control. Do not use `--no-deps` blindly; use it only when you know the package's dependencies are already satisfied and you are deliberately preventing a rollback.

### 12.30 Community wheel decision checklist

Community wheels can be useful during a fast hardware transition, but they are not neutral. A wheel is a binary artifact tied to a precise environment.

Before using one, check:

```text
OS: Linux, not Windows
CPU architecture: x86_64
Python tag: cp310 or cp312 for current ZeroGPU docs
Torch version: compatible with the installed torch
CUDA tag: CUDA 12.8-class when relevant
GPU target: includes Blackwell / sm_120, not only H100/H200 / sm_90
manylinux / glibc tag: compatible with the Space runtime
source: trusted enough for this application
hash: pinned if the install method supports it
fallback: standard path still available if the wheel fails
```

A wheel built for an H200-era ZeroGPU workaround should be treated as an H200-era artifact unless its source explicitly documents Blackwell / `sm_120` support. A wheel that happens to install is not proof that its kernels are valid for the assigned GPU.

### 12.31 Quantization recovery notes

Quantization should not be the first repair path after a Blackwell migration. It may reduce memory, but it also adds backend complexity.

Use this order:

```text
1. Load the model without quantization if VRAM allows.
2. If not, test CPU loading or smaller model variant.
3. Confirm normal PyTorch inference.
4. Add quantization only after the standard path works.
5. Test inference and training separately.
6. Record the exact quantization backend and version.
```

Common traps:

- `bitsandbytes` 4-bit or 8-bit paths can fail because the selected CUDA kernel is not built for Blackwell.
- `torchao` FP8 / FP4 paths can be promising on Blackwell, but they are still backend-specific optimization paths.
- vLLM, TGI, and other serving stacks may route quantized models through separate kernels from full-precision models.
- A quantized load error may look like a model error even when the base checkpoint is fine.

Practical rule: during recovery, quantization is suspect until a non-quantized or conservative path proves that the model and application logic still work.

### 12.32 Optional Blackwell optimization after recovery

Blackwell-specific optimization belongs after recovery, not before it.

Useful candidates may include:

- torchao MXFP8 or NVFP4 quantization for supported diffusion paths;
- verified Blackwell attention kernels;
- CUDA Graphs where the app shape is stable;
- updated TensorRT or TensorRT-LLM paths where appropriate;
- updated llama.cpp / GGUF CUDA builds where the runner explicitly supports Blackwell.

The [PyTorch / Hugging Face torchao Blackwell diffusion article](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/) shows that Blackwell-specific FP8/FP4 optimization can be valuable. It should still be treated as an optimization layer. A Space that cannot complete a standard inference path should not be debugged through Blackwell-specific FP4/FP8 acceleration first.

Safe optimization loop:

```text
1. Save the known-good standard path.
2. Enable exactly one optimization.
3. Run the same smoke test.
4. Compare output quality, latency, and VRAM.
5. Keep the optimization only if it is reproducible.
6. Document how to disable it.
```

---

## 13. VRAM Triage on Blackwell ZeroGPU

Current docs list:

- `large`: half RTX PRO 6000 Blackwell, 48GB VRAM.
- `xlarge`: full RTX PRO 6000 Blackwell, 96GB VRAM.

This is not the same memory profile as the older H200 Explorers page, which described 70GB per workload. A model that fit under H200-era assumptions may need changes under `large` 48GB. However, do not treat every failure as VRAM.

### 13.1 Strong VRAM signals

```text
CUDA out of memory
Tried to allocate ...
OutOfMemoryError
CUDA OOM
```

Also suspect VRAM if:

- small input works but large input fails;
- low resolution works but high resolution fails;
- fewer frames work but more frames fail;
- batch 1 works but batch 2 fails;
- short context works but long context fails.

### 13.2 First VRAM reductions

For image/video/diffusion Spaces:

- set batch size to 1;
- reduce width and height;
- reduce frame count;
- reduce number of steps for testing;
- use VAE slicing;
- use VAE tiling;
- avoid preloading multiple large pipelines on GPU;
- move inactive models back to CPU;
- unload after inference if practical;
- use `torch.float16` or `torch.bfloat16` where safe.

For LLM/text Spaces:

- reduce max new tokens;
- reduce context length;
- reduce batch size;
- reduce number of beams;
- disable large speculative decoding experiments during recovery;
- prefer quantized weights only if the quantization stack itself is compatible;
- use CPU offload or `device_map="auto"` carefully.

### 13.3 When to consider `xlarge`

Consider `xlarge` only after confirming that the error is actually memory pressure or that the model requires a memory tier closer to full 96GB.

Do not use `xlarge` to hide a `no kernel image` error. A full GPU still needs compatible kernels.

---

## 14. FUSE, mmap, and Model Loading Hangs

Some failures occur before attention or generation. The Space may hang during startup, model download, or safetensors loading.

The [Transformers model loading documentation](https://huggingface.co/docs/transformers/en/main_classes/model) includes a `disable_mmap` parameter. It explains that when `disable_mmap=None`, Transformers auto-detects and disables mmap when a checkpoint lives on an `hf-mount` FUSE filesystem used by HF Spaces/Endpoints, because mmap plus parallel page faults can deadlock.

Why this matters: a startup hang may not be Blackwell, VRAM, or PyTorch. It may be a model-loading path problem.

### 14.1 Symptoms

- `Building` succeeds but the Space never becomes usable.
- Logs stop during `from_pretrained()`.
- Logs stop during safetensors loading.
- No CUDA error is printed.
- Repeated restarts appear to hang at the same model load line.

### 14.2 Recovery clues

Check:

- whether weights are read from a mounted path;
- whether a large model is downloaded at startup;
- whether multiple workers load the same large model concurrently;
- whether cache directories are empty after restart;
- whether the app copies models to local ephemeral storage before loading;
- whether `disable_mmap` can be passed through the loading path.

Example:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
    disable_mmap=True,
    trust_remote_code=True,
)
```

Do not add `disable_mmap=True` blindly everywhere. Use it when the model loading path and Transformers version support it, and when logs point toward a load-time stall.

---

## 15. Runtime Drift as a Secondary Cause

Blackwell is the main topic of this guide. Runtime drift is still important because it often appears at the same moment.

A dormant Space may wake up into a different environment even if the Git repository did not change. The build image, hardware assignment, package resolver output, model cache, external wheel URLs, and default dependencies may differ from the last successful run.

The [Hugging Face Hub Space runtime API](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime) exposes operations and fields around runtime state, including current hardware, requested hardware, logs, restart, pause, and sleep time. That does not mean every restart changes everything, but it does mean Space runtime state is something to inspect.

### 15.1 Drift clues

Check runtime drift when:

- the Space was inactive for weeks or months;
- it failed after restart, rebuild, duplication, or hardware change;
- the failure is during build or import rather than first CUDA call;
- `pip` resolves a newer or older version than expected;
- requirements are loosely pinned;
- requirements are too tightly pinned to old CUDA/PyTorch packages;
- a direct wheel URL disappeared or now points to a different build;
- a GitHub install changed because it used a branch instead of a commit;
- model weights download again at startup;
- cache directories are empty;
- duplicated Spaces behave differently;
- the same repo has different logs before and after rebuild.

### 15.2 Compare these files and outputs

Collect:

```text
README.md Space metadata
requirements.txt
pyproject.toml
packages.txt
Dockerfile if present
app.py or main entrypoint
build logs
runtime logs
pip freeze output
python version
torch version
torch cuda version
GPU name
GPU capability
```

Why this matters: Blackwell may be the visible trigger, but an old `requirements.txt` can be the actual reason the Space installs a PyTorch version that cannot use Blackwell.

### 15.3 Runtime drift should not hide Blackwell

Do not let drift become a vague explanation.

Use this split:

| If the log says... | Treat first as... |
|---|---|
| `sm_120`, `no kernel image`, `invalid device function` | Blackwell kernel or CUDA binary mismatch |
| `ModuleNotFoundError`, pip conflict, old package pulled | Dependency drift |
| startup timeout, download loop, cache miss | Runtime/cache/model-loading drift |
| `CUDA out of memory` | VRAM pressure |
| older torch installed unexpectedly | Dependency drift causing Blackwell failure |

The best diagnosis may be combined: Blackwell exposed a dependency drift that was harmless before.

---

## 16. Long-Dormant Spaces

A Space that has not been used for months should be treated as a migration candidate even if the repository is unchanged.

Reasons:

- ZeroGPU hardware assumptions may have changed.
- The documented supported PyTorch range may have changed.
- A dependency resolver may pick a newer transitive package.
- A transitive package may pick an older torch.
- A cache may be gone.
- A model may need to be re-downloaded.
- A third-party wheel URL may have disappeared.
- Remote model code may have changed if loaded from a floating revision.
- A Space duplicate may resolve metadata differently.

### 16.1 Dormant Space recovery checklist

1. Add the runtime probe from Section 3.
2. Read the build log before editing code.
3. Read the runtime log before editing dependencies.
4. Identify whether the first failure is build, import, model load, first GPU call, or inference.
5. Confirm whether the assigned GPU is Blackwell / `sm_120`.
6. Remove old forced attention kernels.
7. Move to a ZeroGPU-supported PyTorch baseline.
8. Test a tiny input first.
9. Test a normal input second.
10. Only then optimize performance.

---

## 17. Practical Code Edits

### 17.1 Add a small hardware banner

Keep this during recovery:

```python
def print_runtime_banner():
    import sys
    import torch

    print("=== Runtime banner ===")
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
    print("======================")

print_runtime_banner()
```

Remove it later if logs are too noisy.

### 17.2 Make FlashAttention optional

```python
USE_FLASH_ATTN = False

try:
    import flash_attn  # noqa: F401
    USE_FLASH_ATTN = True
except Exception as e:
    print("FlashAttention is not available or not safe in this runtime:", repr(e))
```

Then ensure the model has a non-FlashAttention path.

### 17.3 Explicit attention fallback

```python
def load_model_with_attention_fallback(model_id):
    from transformers import AutoModelForCausalLM

    common_kwargs = dict(
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    for attn_impl in ["sdpa", "eager"]:
        try:
            print(f"Trying attention implementation: {attn_impl}")
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                **common_kwargs,
            )
        except Exception as e:
            print(f"Failed with {attn_impl}:", repr(e))

    raise RuntimeError("Could not load model with sdpa or eager attention")
```

### 17.4 Test the first CUDA operation separately

```python
import torch

if torch.cuda.is_available():
    x = torch.randn((1, 16), device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("Basic CUDA test OK", y.shape)
```

If this fails, the issue is below the model. If this succeeds but model generation fails, inspect model-specific kernels.

### 17.5 Disable startup GPU loading when possible

For Gradio apps, avoid loading the full model onto GPU at import time if that makes debugging impossible. Load lazily inside the `@spaces.GPU` function or initialize on CPU first, depending on the model.

A simplified pattern:

```python
import spaces

pipe = None

@spaces.GPU(duration=120)
def generate(prompt):
    global pipe
    if pipe is None:
        pipe = load_pipeline_safely()
    return pipe(prompt)
```

This is not always best for performance, but it can help separate app import failure from GPU runtime failure.

---

## 18. `requirements.txt` Recovery Patterns

### 18.1 Start by removing old hard pins

Temporarily remove or update lines like:

```text
torch==2.0.1
torch==2.1.0
torch==2.4.0
torch==2.5.0
torchvision==0.15.2
xformers==0.0.20
flash-attn==...
```

Also inspect direct wheel URLs:

```text
https://...
cu118
cu121
cu124
cp310
cp311
cp312
cxx11abi
```

A direct wheel URL is not automatically wrong, but it must match Python, torch, CUDA, ABI, and GPU architecture expectations.

### 18.2 Do not solve every problem by installing more packages

If remote model code imports FlashAttention unnecessarily, installing FlashAttention may make the build harder and still fail at runtime. It may be better to patch the import path so the model can use SDPA or eager attention.

### 18.3 Keep the recovery lock small

A recovery requirements file should be boring:

```text
transformers>=4.56
diffusers>=0.35
accelerate
safetensors
gradio>=4
spaces
```

Do not copy this blindly. Use versions appropriate to the Space. The point is to avoid pinning a large old stack until the app boots on the new baseline.

### 18.4 Pin after recovery

Once the Space works, record the working versions:

```bash
python -m pip freeze | sort > working-freeze.txt
```

Then convert that into a maintainable `requirements.txt`. Do not necessarily pin every transitive package forever; pin the packages whose drift would break the app.

---

## 19. README Space Metadata

The Space `README.md` YAML front matter controls important runtime behavior such as SDK and Python version.

Example shape:

```yaml
---
title: My ZeroGPU Space
emoji: rocket
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
python_version: 3.12.12
---
```

For current ZeroGPU recovery, prefer documented Python versions: `3.10.13` or `3.12.12`. If the app depends on packages that do not support Python 3.12, use `3.10.13` first. If the app depends on newer packages that expect Python 3.12, test `3.12.12`.

Do not rely on `suggested_hardware` as if it assigns actual runtime hardware. It is not the same as selecting hardware in Space settings.

---

## 20. Case Study: H200 Expectation, Blackwell Reality, Qwen3-TTS Failure

A public Hugging Face Post by Imosu documents a clean Blackwell migration symptom:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb
Capability: (12, 0)
Torch: 2.8.0+cu128
CUDA: 12.8
```

The Space was running Qwen3-TTS and failed with:

```text
CUDA error:
no kernel image is available for execution on the device
```

The post identifies `kernels-community/flash-attn3` as a likely source of the issue because the expected H200/Hopper-class environment differed from the actual Blackwell / compute capability 12.0 runtime.

Operational lesson:

- Do not treat this as a generic model failure.
- Do not assume that the GPU name is cosmetic.
- Do not reduce resolution or batch size first unless there is OOM evidence.
- Remove or replace the architecture-specific attention kernel first.
- Try `sdpa` or `eager` if the model supports it.

Source: [Imosu ZeroGPU Hardware Mismatch Post](https://huggingface.co/posts/Imosu/801922682272974).

---

## 21. Case Study: Forum Blackwell Transition Report

The forum thread titled [NVIDIA RTX PRO 6000 instead of H200 for ZeroGPU](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960) reported that Spaces restarted onto RTX PRO 6000 Blackwell instead of H200. The report included:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Capability: (12, 0)
CUDA: 12.8
```

A later comment showed a PyTorch warning for:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb
CUDA capability sm_120
```

The installed PyTorch build supported older capabilities up to `sm_90`, not `sm_120`. The thread also included a ZeroGPU configuration error listing supported torch versions starting at `2.8.0`.

Operational lesson:

- Check whether old torch is installed directly or indirectly.
- If the installed torch only supports up to `sm_90`, it is not enough for Blackwell `sm_120`.
- If another dependency pulls old torch, updating only your app code will not fix it.
- Use the ZeroGPU supported PyTorch matrix, not old H200 assumptions.

---

## 22. Case Study: A100 to H200 Migration Pattern

The [FoundHand discussion](https://huggingface.co/spaces/Chaerin5/FoundHand/discussions/4) is useful because it shows that this pattern existed before Blackwell.

In April 2025, `hysts` explained that ZeroGPU hardware was being migrated from half A100 to one-third H200 and half H200, and that documentation had not yet been updated. The same thread shows a Space failing with `sm_90` compatibility complaints and a dependency path where pinned `xformers==0.0.20` appeared to pull `torch==2.0.1`.

Operational lesson:

- Hardware migration plus documentation lag is not new.
- A dependency that was harmless on one target can break on another.
- Do not assume that unpinned torch means a current torch will be installed.
- Check the resolved dependency tree.

---

## 23. Case Study: LTX / Video / FA3 / Memory Ambiguity

Video generation Spaces are easy to misdiagnose because they are both kernel-sensitive and memory-sensitive.

A Blackwell migration can expose attention kernel issues. At the same time, video models may also hit VRAM limits due to resolution, frame count, latent size, VAE usage, or decoding memory.

Operational lesson:

- If the log says `no kernel image`, start with kernel compatibility.
- If the log says OOM, start with VRAM reduction.
- If both appear in different tests, fix the kernel path first, then tune memory.
- For video models, reduce frames and resolution for diagnostics, not as the final assumption.

---

## 24. Florence-2 / Remote Code Pattern

Some older remote-code models import optional acceleration packages as if they were mandatory.

A typical pattern:

```python
from flash_attn import flash_attn_func
```

If this happens at import time, the Space may fail before you can select `sdpa` or `eager`.

Repair posture:

- inspect the remote-code model files;
- determine whether FlashAttention is required or optional;
- make the import optional if the model has a safe path;
- pass `attn_implementation="sdpa"` where supported;
- use `eager` fallback if `sdpa` still hits a backend issue.

Do not blindly fake imports unless you understand the path. A fake import can move the failure later and make debugging harder.

---

## 25. Helper Context Block

When asking another maintainer or an AI assistant to patch a Space, provide this information.

```text
Task:
Repair a Hugging Face ZeroGPU Space after Blackwell migration.

Space URL:
<url>

Failure phase:
build / import / model loading / first GPU call / first inference / large inference

Exact error:
<paste full error>

Observed runtime:
GPU name:
GPU capability:
Python version:
Torch version:
Torch CUDA version:
Transformers version:
Diffusers version:
Gradio version:
spaces package version:

Files to inspect:
README.md YAML metadata
requirements.txt
packages.txt
app.py or main entrypoint
model loading code
attention backend selection code
any custom CUDA / Triton / xFormers / flash-attn code

Important recovery preference:
Blackwell / sm_120 compatibility first.
Remove forced FlashAttention / xFormers / Triton / custom CUDA before reducing model quality.
Try SDPA first where supported, then eager fallback.
Treat runtime drift as secondary but collect dependency and cache clues.
```

Why this matters: an assistant given only an error line may guess VRAM or reinstall FlashAttention. An assistant given hardware, torch, CUDA, dependency, and phase information is much more likely to patch the right layer.

---

## 26. Experienced Maintainer Quick Table

| Situation | Do first | Do not do first |
|---|---|---|
| Blackwell `sm_120` warning | Fix torch/CUDA support or remove old extension | Rewrite model logic |
| `no kernel image` | Remove forced architecture-specific kernels | Reduce batch/resolution as primary fix |
| FlashAttention import failure | Make it optional or use SDPA/eager | Install random wheel URL |
| xFormers pin pulls old torch | Remove/upgrade xFormers pin | Keep torch uninspected |
| Startup hang | Check model loading, cache, `disable_mmap` path | Blame attention immediately |
| Explicit OOM | Reduce memory use | Replace torch first |
| Long-dormant Space | Compare resolved deps and runtime | Assume repo code changed |
| H200-era README | Treat as historical | Assume H200 is still assigned |

---

## 27. Non-Specialist Step-by-Step Repair

If you are not comfortable with CUDA details, follow this order.

### Step 1: Copy the full error

Do not paraphrase. Copy the full build log or runtime log around the first failure.

### Step 2: Add the runtime banner

Add the code from Section 17.1 and restart.

### Step 3: Look for Blackwell markers

Look for:

```text
RTX PRO 6000
Blackwell
MIG 2g.48gb
sm_120
Capability: (12, 0)
CUDA 12.8
torch 2.8.0+cu128
```

If present, this guide is relevant.

### Step 4: Search your repository

Search for:

```text
flash_attn
flash_attention_2
flash_attention_3
xformers
triton
CUDAExtension
attn_implementation
```

### Step 5: Disable forced acceleration

If the model can run with `sdpa` or `eager`, use that first.

### Step 6: Check PyTorch version

If torch is older than the current ZeroGPU supported baseline, fix that before debugging the model.

### Step 7: Try a tiny test

Use a small prompt, small image, low resolution, or few frames. This separates basic runtime success from size-related failure.

### Step 8: Only then tune VRAM

If tiny inputs work and larger ones fail with OOM, reduce memory use or consider `xlarge`.

### Step 9: Re-pin the working state

Once it works, record the versions and remove unnecessary debug prints.

---


## 28. Decision Tree

Use this decision tree when the logs are noisy.

```text
1. Does the runtime show RTX PRO 6000 Blackwell, MIG 2g.48gb, sm_120, or capability (12, 0)?
   yes -> continue with Blackwell-first recovery.
   no  -> this guide may still help, but verify the actual assigned hardware.

2. Does the first hard error mention no kernel image, invalid device function, sm_120, or unsupported architecture?
   yes -> inspect PyTorch/CUDA/third-party kernels before VRAM.
   no  -> continue.

3. Does the error happen during build or import?
   yes -> inspect requirements, wheel tags, Python version, old torch, xFormers, flash-attn, and direct URLs.
   no  -> continue.

4. Does the error happen during model loading before generation?
   yes -> inspect cache, model download, hf-mount/FUSE, disable_mmap, and startup initialization.
   no  -> continue.

5. Does a tiny inference work but a normal inference fails with OOM?
   yes -> VRAM recovery path.
   no  -> continue.

6. Does the Space depend on old remote code, custom CUDA, Triton, or fixed attention backends?
   yes -> patch toward sdpa/eager or safe PyTorch path.
   no  -> continue.

7. Did the Space sleep, restart, rebuild, or duplicate after a long time?
   yes -> inspect runtime drift in parallel.
   no  -> investigate model-specific behavior.
```

Do not skip step 1. The whole repair depends on knowing what GPU target the Space actually received.

---

## 29. Model Family Notes

### 29.1 LLM and text generation Spaces

Common risks:

- FlashAttention forced through model config or remote code.
- Long context causing KV-cache memory growth.
- Quantization packages with architecture-specific kernels.
- Old `bitsandbytes` or custom CUDA wheels.
- `torch.compile` or Triton paths that were tested on another GPU.

First recovery:

- use `attn_implementation="sdpa"`;
- try `eager` if SDPA fails;
- reduce context length only after kernel compatibility is resolved;
- disable speculative or paged attention experiments during recovery;
- test a minimal prompt before normal prompts.

Why this matters: LLM Spaces can fail for both architecture and memory reasons. The exact error text determines which path comes first.

### 29.2 Diffusion image Spaces

Common risks:

- xFormers memory-efficient attention forced by older examples;
- FlashAttention or custom attention processors;
- VAE memory spikes;
- high resolution default settings inherited from H200-era behavior;
- multiple pipelines loaded onto GPU at startup.

First recovery:

- disable xFormers first if it produces kernel errors;
- use standard attention processors;
- test at low resolution;
- enable VAE slicing or tiling for OOM;
- load one pipeline at a time.

Do not confuse an xFormers kernel failure with a resolution problem. If the first failure is an architecture error, resolution is not the primary fix.

### 29.3 Video generation Spaces

Common risks:

- frame count;
- latent tensor size;
- attention memory;
- VAE decode memory;
- temporal attention kernels;
- FlashAttention-3 or custom Triton paths.

First recovery:

- force safe attention first;
- test very few frames;
- reduce resolution;
- decode in chunks if supported;
- avoid loading multiple video pipelines simultaneously;
- consider `xlarge` only after confirming memory pressure.

Video Spaces are where Blackwell kernel mismatch and VRAM pressure most often overlap. Separate them with tiny tests.

### 29.4 TTS / audio / multimodal Spaces

Common risks:

- remote code forcing a specific attention backend;
- generated sequence length;
- custom kernels in codec or vocoder components;
- multiple models loaded together;
- audio chunking or batching behavior.

First recovery:

- inspect model remote code;
- make acceleration imports optional;
- test a short utterance;
- use SDPA/eager if possible;
- separate text model failure from vocoder failure.

The Imosu Qwen3-TTS report is a useful example: the apparent model failure was strongly tied to the actual Blackwell runtime and a suspected FlashAttention-3 path.

### 29.5 ComfyUI-style or node-based Spaces

Common risks:

- many dependencies installed indirectly;
- custom nodes with compiled extensions;
- old xFormers pins;
- startup installs;
- model downloads during launch;
- several models loaded before the first request.

First recovery:

- disable custom nodes that install CUDA extensions;
- remove old xFormers pins;
- boot with a minimal workflow;
- inspect which node triggers the first CUDA call;
- add the runtime banner to the earliest visible log point.

Node-based systems can hide the real failure behind a generic workflow error. Find the first underlying CUDA or import error.

---

## 30. Patch Patterns by Failure Type

### 30.1 Replace forced FlashAttention with configurable attention

Before:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_3",
    trust_remote_code=True,
)
```

After:

```python
attention_backend = os.environ.get("ATTN_IMPL", "sdpa")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation=attention_backend,
    trust_remote_code=True,
)
```

Then set `ATTN_IMPL=eager` for a conservative fallback test.

Why this matters: configuration lets the Space recover without editing code again if one backend fails.

### 30.2 Remove direct GPU placement at import time

Before:

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
```

After:

```python
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
    return pipe
```

This is not always the fastest design, but it makes the failure phase easier to see.

### 30.3 Add a safe tiny inference test

```python
def smoke_test_model(model, tokenizer):
    import torch

    prompt = "test"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=4)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
```

If this fails with a kernel error, do not spend time tuning long prompts.

### 30.4 Add a safe image smoke test

```python
def smoke_test_pipe(pipe):
    image = pipe(
        "test image",
        width=512,
        height=512,
        num_inference_steps=2,
        guidance_scale=1.0,
    ).images[0]
    print("Smoke test image size:", image.size)
```

If this works but production settings fail with OOM, move to VRAM tuning.

### 30.5 Make environment selection explicit


### 30.6 Make acceleration opt-in during recovery

For a Blackwell recovery branch, make optional accelerators opt-in rather than opt-out.

```python
import os

USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "0") == "1"
USE_XFORMERS = os.environ.get("USE_XFORMERS", "0") == "1"
USE_SAGE = os.environ.get("USE_SAGE", "0") == "1"
```

The exact flags depend on the application. The principle is stable: a broken Space should boot on the conservative path first. Acceleration can return after the baseline path works.

### 30.7 Add a dependency drift banner

A dependency banner is cheap and often saves hours.

```python
import importlib.metadata as im

PKGS = [
    "torch", "torchvision", "torchaudio", "triton",
    "transformers", "diffusers", "accelerate",
    "xformers", "flash-attn", "bitsandbytes",
    "gradio", "spaces",
]

for pkg in PKGS:
    try:
        print(f"{pkg}: {im.version(pkg)}")
    except im.PackageNotFoundError:
        print(f"{pkg}: not installed")
```

This is especially useful when a Space has not been restarted for months. The repository may be unchanged while the resolver, cache state, and runtime image changed underneath it.

```python
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")
LOW_VRAM = os.environ.get("LOW_VRAM", "1") == "1"

print("ATTN_IMPL:", ATTN_IMPL)
print("LOW_VRAM:", LOW_VRAM)
```

Why this matters: Spaces are easier to recover when behavior is visible in logs and controlled by environment variables.

---


### 30.8 Patch OpenCV import drift

If a Space uses OpenCV only for image decoding, resizing, drawing, or video frame handling, prefer the headless package.

```text
# usually better for Spaces
opencv-python-headless

# avoid installing both unless you know why
# opencv-python
# opencv-python-headless
```

Then add a simple import check:

```python
import cv2
import numpy as np

print("cv2", cv2.__version__)
print("numpy", np.__version__)
```

Why this matters: `cv2` failures are often wheel, GUI-library, or NumPy ABI problems. They are usually not Blackwell kernel problems.

### 30.9 Patch legacy `.pth` / `.pt` loading

Prefer safetensors or state-dict-only checkpoints. If the project must load an older pickle-backed checkpoint, make the trust boundary explicit.

```python
import torch

# Safer for plain tensor/state_dict checkpoints.
state = torch.load("model.pth", map_location="cpu")
```

If PyTorch reports a `weights_only` failure, do not immediately set `weights_only=False` for arbitrary files. First check whether the checkpoint source is trusted and whether a safetensors or state_dict version exists.

```python
# Last-resort pattern for trusted checkpoints only.
state = torch.load(
    "trusted_legacy_checkpoint.pth",
    map_location="cpu",
    weights_only=False,
)
```

Why this matters: PyTorch 2.6+ changed the default loading posture. A checkpoint that loaded under an older runtime may fail under the current runtime even before CUDA is involved.

### 30.10 Patch ONNX Runtime provider drift

Print the available providers and test CPU before assuming the ONNX model is broken.

```python
import onnxruntime as ort

print("onnxruntime", ort.__version__)
print("available providers", ort.get_available_providers())

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("active providers", session.get_providers())
```

If CUDA provider loading fails, test:

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)
```

Why this matters: ONNX Runtime GPU failures often come from CUDA/cuDNN/provider mismatch, not from the ONNX model itself.

### 30.11 Patch quantization drift

Make quantization optional during recovery.

```python
import os

USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"
USE_8BIT = os.environ.get("USE_8BIT", "0") == "1"
USE_TORCHAO = os.environ.get("USE_TORCHAO", "0") == "1"
```

Then load a conservative baseline first. Add `bitsandbytes`, torchao, FP8, FP4, AWQ, GPTQ, or serving-specific quantization only after the baseline path works.

Why this matters: quantization can introduce compiled kernels and architecture gates. It can fix memory pressure, but it can also be the only reason the Space fails.

### 30.12 Patch dependency rollback

At the end of the build or startup, print final versions. Do not rely on the first installation line in the log.

```python
import importlib.metadata as im

for name in [
    "torch", "torchvision", "torchaudio", "triton",
    "transformers", "diffusers", "accelerate",
    "xformers", "flash-attn", "bitsandbytes",
    "onnxruntime-gpu", "opencv-python-headless",
]:
    try:
        print(name, im.version(name))
    except im.PackageNotFoundError:
        print(name, "not installed")
```

Why this matters: a later install can silently downgrade torch, switch CUDA wheel families, or replace a working GPU build with a CPU build.

## 31. Known False Fixes

### 31.1 Reinstalling FlashAttention immediately

Do not install FlashAttention just because the app imports it. First decide whether the app truly requires it.

A safer question:

```text
Can this model run with sdpa or eager attention on Blackwell?
```

### 31.2 Reducing quality for a kernel error

Reducing resolution, frame count, or batch size does not fix a missing `sm_120` kernel image.

### 31.3 Pinning every dependency before understanding the failure

A lock file can preserve a broken state. During recovery, keep the lock narrow enough to test the correct runtime baseline.

### 31.4 Trusting old H200 documentation

The old ZeroGPU Explorers page is useful history, but it marks itself out of date and points to current documentation. Use it to understand old assumptions, not to set current expectations.

### 31.5 Assuming import success means runtime success

A CUDA package can import successfully and still fail on the first kernel launch. Always test a tiny GPU operation.

### 31.6 Assuming Blackwell is the only cause

Blackwell is the main topic of this guide, but rebuilds can also change dependencies, cache state, startup downloads, and package resolution.

---

## 32. After the Space Works Again

Do not stop at "it works once." Stabilize the recovery.

### 32.1 Record the working runtime

Add a short comment in the repository or a maintenance note:

```text
Verified on ZeroGPU Blackwell runtime:
GPU: RTX PRO 6000 Blackwell / sm_120
Python: ...
Torch: ...
CUDA: ...
Transformers: ...
Attention backend: sdpa/eager/other
Date verified: ...
```

This is operational evidence for future maintainers.

### 32.2 Keep a smoke test

Keep a minimal inference path that can be run after rebuilds. It should be cheaper than a full demo request and should fail close to the real root cause.

### 32.3 Reintroduce acceleration carefully

Order:

1. stable SDPA/eager baseline;
2. memory tuning;
3. xlarge decision if needed;
4. optional FlashAttention/xFormers/Triton;
5. benchmark;
6. fallback switch retained.

Do not remove the fallback after adding acceleration. Future hardware changes may happen again.

### 32.4 Optional Blackwell optimization after recovery

Optimization belongs after recovery. A Space that cannot complete a standard inference path should not be debugged through Blackwell-specific FP4/FP8 acceleration first.

Potential post-recovery experiments:

```text
torchao MXFP8
torchao NVFP4
verified Blackwell FP8/FP4 Diffusers paths
CUDA Graphs when the app has stable shapes
verified serving backend releases with Blackwell support
```

Use this order:

```text
1. Standard path works.
2. Smoke test passes.
3. Output quality is acceptable.
4. Add one optimization.
5. Measure latency and VRAM.
6. Keep a rollback path.
```

PyTorch and torchao materials show that Blackwell-specific formats such as MXFP8 and NVFP4 can be valuable for diffusion inference, but this guide treats them as optional acceleration, not a baseline recovery method.

### 32.5 Pin the important pieces

Pin:

- model code revision if remote code is unstable;
- major libraries that changed behavior;
- attention backend selection;
- Python version in Space metadata;
- package versions known to pull old torch if left unconstrained.

Avoid pinning huge transitive stacks without reason. A maintainable pin is better than a frozen accident.

---


## 33. Public Release Hygiene

Before publishing a guide revision, remove anything that is not useful to an outside maintainer.

Check for:

- private repository names that are not part of a public case study;
- local file paths from the author's machine;
- unpublished package names;
- private tokens, usernames, emails, or organization details;
- conversation notes or planning notes;
- temporary authoring comments such as "fix later" or "temporary note";
- claims that are not backed by public links or clearly marked as cautious guidance.

Keep:

- public source links;
- public issue links;
- public Space discussions;
- public forum posts;
- reproducible commands;
- minimal code patches;
- cautious uncertainty where the evidence is incomplete.

A release-ready version should read like an operational guide, not like a project log. The reader should not need to know how the guide was prepared.

## 34. Future Update Policy for This Guide

This guide should be updated when public evidence changes.

Update when:

- ZeroGPU docs change hardware, VRAM tiers, Python versions, or PyTorch versions;
- Hugging Face publishes a new announcement or docs PR;
- maintainers clarify Blackwell behavior in public threads;
- common FlashAttention/xFormers/Triton compatibility changes;
- PyTorch changes stable Blackwell support assumptions;
- new recurring failure patterns appear in public Spaces discussions.

Keep updates source-linked. Prefer official documentation and public maintainer comments. Use user posts and Space discussions as case studies, not as universal facts.


## 35. References and Source Links

Primary ZeroGPU sources:

- [Current ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
- [ZeroGPU Blackwell update PR #2474](https://github.com/huggingface/hub-docs/pull/2474)
- [Forum thread: NVIDIA RTX PRO 6000 instead of H200 for ZeroGPU](https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960)
- [Older ZeroGPU Explorers page](https://huggingface.co/zero-gpu-explorers)
- [FoundHand discussion: A100 to H200 migration and PyTorch mismatch](https://huggingface.co/spaces/Chaerin5/FoundHand/discussions/4)
- [Imosu Post: ZeroGPU Hardware Mismatch](https://huggingface.co/posts/Imosu/801922682272974)

Attention and model loading:

- [Transformers attention interface](https://huggingface.co/docs/transformers/attention_interface)
- [Transformers model loading docs](https://huggingface.co/docs/transformers/en/main_classes/model)

Runtime and Spaces operations:

- [Hugging Face Hub: Managing your Space runtime](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime)

Blackwell / PyTorch context:

- [PyTorch forum: when will sm120 support be available?](https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621)
- [PyTorch forum: PyTorch support for sm120](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)

Historical launch context:

- [The Register coverage of ZeroGPU launch](https://www.theregister.com/software/2024/05/17/hugging-face-plans-to-make-10m-in-gpus-available-to-public/1299644)
- [Clem Delangue launch post mirror on LinkedIn](https://www.linkedin.com/posts/clementdelangue_gpu-poor-no-more-super-excited-to-officially-activity-7196881557284868096-M96G)

### Additional Blackwell / PyTorch Drift Sources

- [PyTorch support for sm120 forum thread](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [PyTorch support for sm_120 / RTX 5060 forum thread](https://discuss.pytorch.org/t/pytorch-support-for-sm-120-nvidia-geforce-rtx-5060/220941)
- [PyTorch 2.8 release blog](https://pytorch.org/blog/pytorch-2-8/)
- [PyTorch local install matrix](https://pytorch.org/get-started/locally/)
- [Text Generation Inference issue: FlashAttention and sm_120](https://github.com/huggingface/text-generation-inference/issues/3342)
- [Diffusers issue: Blackwell attention backend no kernel image](https://github.com/huggingface/diffusers/issues/13043)
- [FlashAttention issue: RTX PRO 6000 Blackwell](https://github.com/Dao-AILab/flash-attention/issues/1987)
- [FlashAttention issue: FA4 / RTX PRO 6000 Blackwell](https://github.com/Dao-AILab/flash-attention/issues/2307)
- [xFormers issue: Blackwell no kernel image](https://github.com/facebookresearch/xformers/issues/1329)
- [vLLM issue: Blackwell / sm_120 compatibility](https://github.com/vllm-project/vllm/issues/16901)
- [FlashInfer issue: Blackwell backend routing](https://github.com/flashinfer-ai/flashinfer/issues/2555)
- [ComfyUI Impact Pack issue: sm_120 custom CUDA fallback](https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/1179)


Additional PyTorch-family and binary-stack drift sources:

- [PyTorch previous versions install matrix](https://pytorch.org/get-started/previous-versions/)
- [PyTorch developer discussion: torch.load default changing to weights_only=True](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573)
- [nnUNet issue showing PyTorch 2.6 weights_only load failure](https://github.com/MIC-DKFZ/nnUNet/issues/2681)
- [TorchVision model weights API documentation](https://docs.pytorch.org/vision/main/models.html)
- [TorchVision 0.24 video IO deprecation documentation](https://docs.pytorch.org/vision/0.24/io.html)
- [TorchVision releases](https://github.com/pytorch/vision/releases)
- [opencv-python-headless PyPI page](https://pypi.org/project/opencv-python-headless/)
- [opencv-python issue: NumPy 2.0 support](https://github.com/opencv/opencv-python/issues/997)
- [ONNX Runtime CUDA Execution Provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime install docs](https://onnxruntime.ai/docs/install/)
- [ONNX Runtime issue: CUDA Blackwell sm_120 support / invalid PTX](https://github.com/microsoft/onnxruntime/issues/26177)
- [PyTorch issue: RTX 5090 sm_120 CUDA kernel execution fails after device detection](https://github.com/pytorch/pytorch/issues/173237)


Additional CUDA 12.8 / Blackwell drift and optimization sources:

- [NVIDIA: CUDA Toolkit 12.8 delivers Blackwell support](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
- [PyTorch 2.7 release blog: prototype NVIDIA Blackwell support and CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch blog: Faster Diffusion on Blackwell with Diffusers and torchao](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)
- [torchao repository](https://github.com/pytorch/ao)
- [bitsandbytes issue: Blackwell sm_120 4-bit/8-bit quantization failure](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937)
- [Accelerate issue: device_map auto weight dispatch segfault on Blackwell](https://github.com/huggingface/accelerate/issues/3933)
- [PyTorch issue: device_map auto segfault on Blackwell](https://github.com/pytorch/pytorch/issues/175614)
- [vLLM issue: sm120/cu128 Blackwell deployment friction](https://github.com/vllm-project/vllm/issues/16515)
- [Ollama issue: Blackwell quantized MMQ CUDA kernel crash](https://github.com/ollama/ollama/issues/14374)
- [NVIDIA developer forum: Blackwell migration guide for CUDA 12.8, PyTorch, TensorRT, llama.cpp](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [Hugging Face Hub: example prebuilt wheel collection for CUDA 12.8 / Python 3.12](https://huggingface.co/yo9otatara/prebuilt_wheels)
- [Hugging Face Hub: sm_120 verification-style repo](https://huggingface.co/pirola/triattention-sm120-verification)
- [ComfyUI-QwenVL code path with SM120 SageAttention branch and SDPA fallback](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/AILab_QwenVL.py)

Additional CUDA 12.8 / library drift sources:

- [NVIDIA Blackwell RTX migration guide: CUDA 12.8, PyTorch, TensorRT, llama.cpp](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [NVIDIA CUDA Toolkit 12.8 Blackwell support announcement](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
- [PyTorch previous versions install matrix](https://pytorch.org/get-started/previous-versions/)
- [PyTorch forum: Blackwell / sm_120 support](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [PyTorch blog: Faster Diffusion on Blackwell with Diffusers and TorchAO](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)
- [sayakpaul/diffusers-blackwell-quants](https://github.com/sayakpaul/diffusers-blackwell-quants)
- [ONNX Runtime CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [NVIDIA TensorRT-LLM issue: torch upper-bound conflict risk](https://github.com/NVIDIA/TensorRT-LLM/issues/4275)
- [bitsandbytes issue: Blackwell / sm_120 kernel failure](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937)
- [xFormers issue: Blackwell and FlashAttention routing](https://github.com/facebookresearch/xformers/issues/1342)
- [Accelerate issue: Blackwell dispatch / device map failure](https://github.com/huggingface/accelerate/issues/3933)
- [ONNX Runtime issue: Blackwell / sm_120 invalid PTX](https://github.com/microsoft/onnxruntime/issues/26177)
- [OpenCV Python headless package](https://pypi.org/project/opencv-python-headless/)

