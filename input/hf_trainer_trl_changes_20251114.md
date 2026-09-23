---
source: "huggingface+chat+files+web"
topic: "Transformers & TRL Trainer usage changes and Trackio integration (2024–2025)"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-13T19:11:32Z"
---

# Transformers & TRL Trainer usage changes and Trackio integration (2024–2025)

## 1. Background and overview

This knowledge base summarises recent and ongoing changes in how to use:

- The `transformers.Trainer` API and `TrainingArguments`
- TRL trainers such as `SFTTrainer` and their config classes (`SFTConfig`, `PPOConfig`, `GRPOConfig`, etc.)
- Trackio as Hugging Face’s recommended lightweight experiment tracker

The focus is on **practical usage differences** that affect real training code in 2024–2025:

- New or preferred arguments (`eval_strategy`, `use_cpu`, `dtype`, Trackio-related fields)
- The `compute_loss_func` hook for custom losses in `Trainer`
- TRL SFT API changes: constructor reshape, `SFTConfig` taking over data arguments
- Updated loss‑masking behaviour (`completion_only_loss`, `assistant_only_loss`) and packing/FlashAttention‑2 (FA2)
- Trackio‑first logging in both Transformers and TRL

The content integrates:

- Official docs and blogs for Transformers, TRL, Trackio, and FA2 (for example the [Transformers Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer), [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer), [Trackio docs](https://huggingface.co/docs/trackio/en/index), and the [Packing with FA2 blog](https://huggingface.co/blog/packing-with-FA2)).
- Your internal specs on TRL + FA2, masking rules, argument naming, and environment pinning.
- A previous version of this KB, now cleaned up and slightly restructured.

Use this file as a reference when writing new training scripts or migrating older ones.

---

## 2. Transformers Trainer / TrainingArguments changes

### 2.1 `eval_strategy` and evaluation configuration

Recent Transformers versions document `eval_strategy` as the main field for controlling evaluation frequency in `TrainingArguments` (replacing the older `evaluation_strategy` name, which is treated as legacy in new examples):

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    eval_strategy="steps",   # "no" | "steps" | "epoch"
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
)
```

Guidelines:

- Always set `eval_strategy` explicitly for non‑toy runs.
- Align `save_strategy` with `eval_strategy` when using `load_best_model_at_end=True`, so that the evaluated checkpoint is actually saved.
- Make sure `do_eval=True` and that an eval dataset is provided, otherwise you get “training without metrics” even though logging is enabled.

For migration, change:

```diff
- evaluation_strategy="steps"
+ eval_strategy="steps"
```

and avoid mixing both names in the same config.

### 2.2 `use_cpu` vs `no_cuda`

`TrainingArguments` now prefers `use_cpu=True` for CPU‑only runs. `no_cuda=True` still exists but is associated with deprecation‑style warnings in newer stacks and is considered legacy:

```diff
- TrainingArguments(..., no_cuda=True, bf16=True)
+ TrainingArguments(..., use_cpu=True)  # and drop bf16/fp16 on CPU
```

Rules of thumb:

- For GPU training, leave `use_cpu=False` and control devices via `CUDA_VISIBLE_DEVICES` or launchers (Accelerate, DeepSpeed, etc.).
- For CPU‑only training, set `use_cpu=True` and **do not** enable `fp16`/`bf16` – mixed precision on CPU tends to fail or silently fall back to fp32.

### 2.3 Dtype configuration and FA2 compatibility

Across Transformers and TRL, configs are converging on a single `dtype` knob (string like `"bf16"` or `"fp16"`) instead of exposing `torch_dtype` directly in high‑level user configs.

Practical policy:

- Use `dtype="bf16"` or `dtype="fp16"` in your config classes where FA2 or other fused kernels are enabled and the hardware supports it.
- Continue to use `torch.float16` / `torch.bfloat16` at lower levels (manual model loading, patching) when needed, but treat `dtype` as the external switch in configs.
- Expect FA2 on NVIDIA to require fp16/bf16 tensors, a compatible CUDA version, and matching `flash-attn` wheels; fp32 or mixed dtypes often force a fallback to SDPA or produce unstable behaviour.

Pin and test the entire stack (PyTorch, Transformers, TRL, flash‑attn) per project; do not upgrade any piece in isolation.

### 2.4 `compute_loss_func`: official custom loss hook

Instead of subclassing `Trainer` and overriding `compute_loss`, Transformers now recommend passing a `compute_loss_func` to the Trainer constructor. The function receives model outputs, labels, and the logical batch size after gradient accumulation:

```python
def my_loss(outputs, labels, num_items_in_batch):
    # Example: standard token‑level cross‑entropy
    import torch
    logits = outputs.logits  # [B, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_loss_func=my_loss,
)
```

Best practices:

- Implement losses on **logits + masks**, not on decoded text or argmaxed predictions.
- Respect `ignore_index` (usually `-100`) so that padding and masked tokens do not contribute to loss.
- Use `num_items_in_batch` when you need custom normalisation that stays compatible with gradient accumulation.

The same pattern is mirrored in TRL’s trainers for supervised fine‑tuning and RL‑style objectives.

### 2.5 Gradient checkpointing and `use_cache`

For decoder‑only LLMs (Llama, Mistral, Gemma, etc.) you generally want:

- Gradient checkpointing enabled during training to save memory.
- `model.config.use_cache=False` during training to avoid warnings and potential incompatibilities with checkpointing.
- `model.config.use_cache=True` again at inference time.

Example:

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False

trainer.train()

model.config.use_cache = True   # after training, for fast generation
```

This pattern avoids the common “past key values are not used” warnings and keeps training behaviour predictable.

---

## 3. TRL SFTTrainer and config changes

### 3.1 SFTTrainer constructor reshape and `processing_class`

In TRL 0.24 the `SFTTrainer` constructor was reshaped. Two important consequences:

1. `SFTTrainer` no longer accepts `tokenizer=` or `dataset_text_field=` as top‑level arguments.
2. Data‑related arguments belong in `SFTConfig`, and the tokenizer/processor is passed via `processing_class` when needed.

Conceptual modern signature (simplified):

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="out",
    max_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # instead of tokenizer=...
)
```

Migration from older code:

```diff
- trainer = SFTTrainer(model, tokenizer=tok,
-                      dataset_text_field="text", args=cfg)
+ cfg = SFTConfig(..., dataset_text_field="text")
+ trainer = SFTTrainer(model=model,
+                      args=cfg,
+                      processing_class=tok)
```

Dataset shapes are explicit: `"text"` (single text field), `"prompt"/"completion"` pairs, or `"messages"` (chat format using a template).

### 3.2 `max_seq_length` → `max_length` and truncation behaviour

`SFTConfig` now exposes `max_length` for truncation. Passing `max_seq_length` raises a `TypeError` in recent TRL versions.

Example:

```diff
- args = SFTConfig(max_seq_length=2048, output_dir="out")
+ args = SFTConfig(max_length=2048, output_dir="out")
```

Related points:

- Truncation happens inside TRL’s preprocessing (e.g. `truncate_dataset`) before collation.
- Packing and device‑cache optimisation options live alongside `max_length` in SFT and iterative SFT configs.
- Internal policy: consistently use `max_length` for all sequence‑length settings across Transformers, TRL, and any wrapper configs to reduce confusion.

### 3.3 Dataset formats, `dataset_text_field`, and `formatting_func`

TRL’s SFTTrainer supports several dataset formats:

- Plain text: a single column like `"text"` that already contains prompt + completion.
- Prompt/completion: two‑column data where masking can be restricted to the completion.
- Message‑based: `"messages"` lists using the tokenizer’s chat template.

Practical guidelines:

- If you use `dataset_text_field`, it should point at a **single field** containing the final text string that will be tokenised.
- For multi‑column data, prefer a `formatting_func` that takes a row and returns one or more strings (for example, assembling prompt + completion), then let TRL handle tokenisation.
- In chat mode, rely on `tokenizer.apply_chat_template` and make sure the template includes a generation block (for example with a `{% generation %}` section) so that `assistant_only_loss` knows which tokens are supervised.

Validate masking after preprocessing: inspect a couple of batches and check that prompt tokens are labelled with `-100` where expected.

### 3.4 Completion‑only and assistant‑only loss

Modern TRL masking behaviour is governed mainly by two flags in `SFTConfig`:

- `completion_only_loss=True` (default) – for prompt/completion style data, loss is applied only on completion tokens.
- `assistant_only_loss=True` – for chat data, loss is applied only on tokens belonging to the assistant role, as indicated by masks derived from the chat template.

Caveats:

- Assistant masks depend on the chat template emitting the correct “generation” region. If the template is missing or mis‑configured, `assistant_only_loss` may silently behave like full‑sequence loss.
- When using truncation or packing, it is possible for a batch to end up with **zero supervised tokens** after masking (for example, if everything was truncated away). Those batches should be skipped or handled specially to avoid NaN losses.
- Some experimental kernels (e.g. certain Liger/FA2 combinations) have historically had bugs where masks are dropped. Treat NaNs or too‑good‑to‑be‑true loss curves as signals to audit masks and collators.

### 3.5 FlashAttention‑2, packing, and collators

For padding‑free FA2 training, Hugging Face’s FA2 packing blog and your own spec agree on a key contract:

- Use a collator such as `DataCollatorWithFlattening` that:
  - Flattens variable‑length sequences into a single 1×T tensor per batch.
  - Emits FA2 metadata (`seq_idx`, `cu_seqlens_q`, `cu_seqlens_k`, `max_length_*`).
  - Drops dense `attention_mask` when FA2 is active.
  - Sets labels to `-100` at all boundaries between packed sequences (seams).

In TRL SFTTrainer, when packing is enabled you must ensure:

- The collator obeys the same FA2 invariants.
- Completion/assistant masks are aligned with packing seams (no supervision across sample boundaries).
- FA2 is only used on hardware and stacks that pass your pinned‑version smoke tests.

If you see NaNs, “!!!!”‑like degenerate generations, or unexpected throughput drops, treat that as a strong hint that either FA2 configuration or packing/masking is broken and temporarily fall back to SDPA or single‑GPU FA2 while debugging.

---

## 4. Trackio: experiment tracking for Transformers and TRL

### 4.1 What Trackio is

Trackio is a lightweight, free, Hugging Face–native experiment tracker:

- Local‑first: it runs a local dashboard (Gradio app) by default.
- Built on top of Hugging Face Datasets and Spaces.
- ML‑friendly: integrates with Transformers, TRL, and Accelerate via simple hooks.
- Capable of syncing dashboards to Spaces for sharing.

Core CLI / Python usage:

```bash
pip install trackio
trackio show  # launch local dashboard
```

```python
import trackio

trackio.show(project="my-sft-project")
```

### 4.2 Using Trackio from Transformers Trainer

Transformers integrate Trackio in `TrainingArguments` via `report_to` and a couple of extra fields:

- `report_to="trackio"` – enable Trackio logging.
- `project` – Trackio project name (groups runs).
- `trackio_space_id` – optional Space ID for hosting the dashboard.

Example:

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

training_args = TrainingArguments(
    output_dir="outputs/qwen-sft",
    eval_strategy="steps",
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=50,
    report_to="trackio",
    project="my-sft-project",
    trackio_space_id="org/trackio-space",  # optional
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
```

This avoids `wandb`‑style shims and makes Trackio a first‑class logging backend.

### 4.3 Using Trackio from TRL trainers

TRL uses the same idea: config classes such as `SFTConfig`, `PPOConfig`, `GRPOConfig`, etc., accept `report_to` and (depending on version) Trackio‑related fields.

Typical supervised fine‑tuning example:

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

ds = load_dataset("trl-lib/Capybara", split="train")

training_args = SFTConfig(
    output_dir="outputs/qwen-sft",
    max_length=2048,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=20,
    report_to="trackio",
    run_name="sft_qwen2-5_capybara",
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=ds,
    args=training_args,
    dataset_text_field="text",
)

trainer.train()
```

Environment variables can also influence Trackio’s behaviour, for example:

```bash
export TRACKIO_PROJECT="my_project"
export TRACKIO_SPACE_ID="username/space_id"
```

This lets you unify logging across:

- Plain Transformers Trainer
- TRL SFT / DPO / PPO / GRPO trainers
- Custom loops that call Trackio’s Python API directly

### 4.4 Mixed stacks and comparison workflows

With Trackio used everywhere, you can:

- Compare SFT vs. DPO vs. GRPO runs side by side.
- Track FA2 vs. SDPA experiments with the same config template.
- Store environment and dependency info as metadata, linked to your pinned stack.

For multi‑library projects (Transformers, TRL, Sentence‑Transformers, custom RL), Trackio becomes the common logging layer, even when each stack has its own trainer class.

---

## 5. Implementation patterns and migration tips

### 5.1 Greenfield (new) projects

For new work, a modern pattern that plays well with current HF tooling is:

- Use Transformers Trainer for “classic” supervised tasks; use TRL’s trainers for LLM post‑training (SFT, DPO, PPO, GRPO).
- Configure:
  - `eval_strategy` instead of `evaluation_strategy`
  - `max_length` instead of `max_seq_length`
  - `dtype` rather than user‑visible `torch_dtype`
  - `use_cpu=True` for CPU‑only jobs
- Implement custom objectives via `compute_loss_func` in Trainer/TRL.
- Use `report_to="trackio"` everywhere, with consistent `project`, `run_name`, and (optionally) `trackio_space_id` / `TRACKIO_*` env vars.
- Pin versions for PyTorch, Transformers, Datasets, TRL, flash‑attn, and Trackio; validate them with small “tiny runs” before scaling.

### 5.2 Migrating older Transformers Trainer code

Common updates:

1. Evaluation:

   ```diff
   - evaluation_strategy="epoch"
   + eval_strategy="epoch"
   ```

2. CPU‑only training:

   ```diff
   - no_cuda=True
   + use_cpu=True
   ```

3. Dtype:

   ```diff
   - torch_dtype=torch.bfloat16
   + dtype="bf16"
   ```

4. Custom loss:

   ```diff
   - class MyTrainer(Trainer):
   -     def compute_loss(self, model, inputs, return_outputs=False):
   -         ...
   - trainer = MyTrainer(...)
   + trainer = Trainer(..., compute_loss_func=my_loss_fn)
   ```

5. Logging backend:

   ```diff
   - report_to=["wandb"]
   + report_to="trackio"
   ```

Apply each change incrementally and run a short training job to compare logs and metrics.

### 5.3 Migrating older TRL SFTTrainer code

Checklist for moving to the reshaped SFTTrainer API:

- Remove `tokenizer=` and `dataset_text_field=` from the SFTTrainer constructor; move `dataset_text_field`, `max_length`, `add_eos_token`, packing flags, and loss‑masking options into `SFTConfig`.
- Pass tokenisers or processors via `processing_class` if you need custom preprocessing.
- Update `max_seq_length` → `max_length` and verify truncation by running tiny tests and inspecting batch lengths.
- Audit chat templates if you use `assistant_only_loss=True` to ensure a proper generation region is defined.
- Validate masks and labels after collation, especially in packed/FA2 mode, and skip batches with zero supervised tokens.

### 5.4 FA2, kernels, and hardware‑specific considerations

From both official docs and your own spec:

- FA2 and related kernels are sensitive to the combination of GPU type, CUDA version, and compiled wheel versions.
- Treat the following as signals of misconfiguration:
  - NaN training or eval losses
  - Exclamation‑only or otherwise degenerate generations
  - Abrupt throughput changes when nothing else changed
- In such cases, fall back to SDPA or single‑GPU FA2, re‑check collator behaviour (`cu_seqlens`, seams, masks, padding), and confirm that dtypes and versions match your compatibility matrix.

Trackio is newer than WandB/MLflow; core logging is stable, but Spaces integration and UI details can change, so always confirm that tutorials match your installed Trackio version.

---

## 6. Limitations, caveats, and open questions

### 6.1 Library and ecosystem maturity

- Trackio is relatively young; its docs and blog posts evolve quickly as new features land.
- TRL’s SFTTrainer has seen breaking changes across 0.2x versions; pin TRL per project and plan upgrades explicitly.
- FA2, Liger, and multi‑GPU attention stacks are still sharp‑edged; treat CI “tiny runs” and alerting on NaNs as mandatory.

### 6.2 Project‑specific policies vs. official behaviour

Your internal specs define stricter rules than the official docs in some areas, for example:

- Forbid `device_map="auto"` during training; treat it as inference‑only.
- Require warning‑free logs for production or “baseline” runs.
- Enforce “evaluation present and working” (no more silent training‑only runs for important jobs).

When docs and specs differ, treat:

- Official HF docs as the source of truth for **API shape and supported behaviour**.
- Internal specs as the source of truth for **what is allowed in your organisation**.

### 6.3 Open questions

A few areas remain open design or research questions rather than settled policy:

- Exact version matrices (per GPU family) that are considered “green” for FA2 + TRL + Trackio.
- Whether to prefer fused CE kernels or safe PyTorch CE in all FA2 modes by default.
- Official support (if any) for multi‑GPU FA2 beyond per‑rank attention.
- How much to centralise custom loss functions into shared libraries versus keeping them local to each project.

Document these decisions as they are made so future migrations are easier.

---

## 7. References and links

### 7.1 Official docs

- [Transformers Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)  
- [Transformers training guide](https://huggingface.co/docs/transformers/training)  
- [Chat templating guide](https://huggingface.co/docs/transformers/chat_templating)  
- [TRL documentation index](https://huggingface.co/docs/trl)  
- [TRL SFTTrainer and SFTConfig](https://huggingface.co/docs/trl/en/sft_trainer)  
- [Trackio docs](https://huggingface.co/docs/trackio/en/index)  

### 7.2 Blog posts and examples

- [Packing variable‑length sequences with FlashAttention‑2](https://huggingface.co/blog/packing-with-FA2)  
- [Introducing Trackio](https://huggingface.co/blog/trackio)  
- [Unsloth + TRL integration](https://huggingface.co/blog/unsloth-trl)  
- [Distributed SFT with TRL and DeepSpeed](https://huggingface.co/blog/jlzhou/distributed-sft-with-trl-and-deepspeed-part1)  

### 7.3 Community and GitHub

- [Transformers‑Tutorials GitHub repo](https://github.com/NielsRogge/Transformers-Tutorials)  
- [FlashAttention repo](https://github.com/Dao-AILab/flash-attention)  
- [Trackio GitHub repo](https://github.com/gradio-app/trackio)  

### 7.4 Internal specs and KBs (this conversation)

- TRL SFT + FA2 padding, length, and custom‑loss spec (Markdown + JSON versions).  
- TRL API and argument‑rename spec (`SC-TRL-API-001`, `SC-TRL-ARG-002`, etc.).  
- HF stack / NumPy / Sentence‑Transformers environment‑pinning spec.  
- Previous KB drafts on this same topic that this file replaces and cleans up.

Together, these references give you a single, updated view of how to configure and use Transformers Trainer, TRL trainers, and Trackio in late‑2024/2025‑era stacks.
