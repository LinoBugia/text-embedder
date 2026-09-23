---
source: "huggingface+chat+files+web"
topic: "Multi-GPU Environments and LLM Fine-Tuning"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-21T00:00:00Z"
---

# Multi-GPU Environments and LLM Fine-Tuning

This knowledge base summarizes how to combine **multi-GPU training** with **LLM fine-tuning**, focusing on the Hugging Face ecosystem (Transformers, Accelerate, TRL, PEFT, DeepSpeed, FSDP) and the practical pitfalls you encountered in your own notes (dataset caching, KTO multi-GPU errors, Accelerate configs, DeepSpeed learning path). It is intentionally redundant: key ideas are repeated with slightly different angles so that each section can be read independently.

---

## 1. Conceptual map: how parallelism fits into fine-tuning

### 1.1 Why multi-GPU matters for fine-tuning

For LLMs, multi-GPU is usually about one of two goals:

1. **Speed**: the model fits on a single GPU, but training is slow → use **data parallelism** (e.g. PyTorch DDP, Accelerate DDP) to split batches across GPUs and sync gradients.
2. **Capacity**: the model *does not* fit on a single GPU → use **sharded / model parallel methods** (FSDP, ZeRO, tensor/pipeline parallelism, ND-parallel) to spread weights, gradients, and optimizer state across devices.

Fine-tuning stacks these concerns on top of the usual design questions (base model, data, optimizer, schedule). Multi-GPU design is therefore not independent: it interacts with

- **Model size and architecture** (7B vs 70B, dense vs MoE, context length).
- **Fine-tuning style** (full fine-tune vs LoRA/QLoRA vs continued pre-training vs DPO/KTO).
- **Tooling** (Transformers `Trainer`, custom Accelerate loop, TRL trainers, DeepSpeed/FSDP).

A robust mental model: *first* decide what you are training (task, base model, PEFT vs full); *then* choose a multi-GPU strategy that matches your hardware and model size.

### 1.2 Main parallelism families

In practice you only need to remember a small taxonomy; everything else is a variation:

- **Single-GPU, single process**  
  - Baseline for debugging and first working runs.
  - Almost all libraries (Transformers `Trainer`, TRL trainers, Unsloth, etc.) default to this when you don’t use multi-GPU launchers.

- **Data parallelism (DDP)**  
  - Each GPU has a full copy of the model; batches are split across GPUs.  
  - PyTorch’s `DistributedDataParallel` (DDP) is the canonical implementation.  
  - Best when the model fits on one GPU but you want to speed up training.

- **Sharded data parallelism**  
  - Also called “model sharding” or “Zero Redundancy”:
    - **FSDP** (Fully Sharded Data Parallel) in PyTorch, integrated deeply with Accelerate and Transformers.
    - **DeepSpeed ZeRO-1/2/3**, exposed via Transformers, Accelerate, and native DeepSpeed.
  - Weights, gradients, and optimizer states are split across GPUs to reduce memory.

- **Model parallelism and pipeline parallelism**
  - Tensor/model parallelism: split individual layers across devices (e.g. Megatron-LM style).
  - Pipeline parallelism: split layers by depth into pipeline stages.  
  - Often used in pre-training or very large inference; fine-tuning usually prefers FSDP/ZeRO unless you need exact Megatron-like setups.

- **ND-parallel (combinations)**  
  - Modern libraries (Accelerate ND-Parallel, various research systems) combine data, tensor, and pipeline parallelism; in practice many fine-tuning jobs can stay with “1D” data parallel or sharded data parallel on a single node.

### 1.3 When to choose which (rule-of-thumb)

- **Model easily fits in one GPU (with optimizations)** → Start with **single-GPU**. Once stable, move to **DDP** for speed.
- **Model barely fits, but fits** → Single GPU plus memory tricks: FP16/BF16, QLoRA, gradient checkpointing, smaller batch size.
- **Model does not fit even with tricks** → Move to **FSDP or DeepSpeed ZeRO-2/3**.
- **Multi-node cluster with very large models** → Consider FSDP/ZeRO-3, ND-parallel or Megatron-style tensor parallel, but expect more engineering and debugging.

Always prototype on one GPU first, then scale out. Multi-GPU magnifies mistakes in data loading, caching, device placement, and configs.

---

## 2. Core building block: PyTorch DDP and samplers

Even when you use higher-level tools, they wrap the same primitives, so understanding DDP clarifies what Accelerate, TRL, and DeepSpeed are doing under the hood.

### 2.1 One process per GPU

**DistributedDataParallel (DDP)** follows a simple but critical pattern:

- One **OS process per GPU** (often launched with `torchrun`, `accelerate launch`, or `deepspeed`).
- Each process:
  - Creates its own model instance and wraps it with `DistributedDataParallel`.
  - Creates its own `DataLoader` with a `DistributedSampler`.
  - Runs the forward/backward/optimizer step on its local batch.
- DDP uses collective communication (all-reduce) to average gradients across ranks after each backward pass.

Implications:

- There is **no shared Python object** for the model or dataset across GPUs; each process holds its own copy.
- Anything you load into RAM inside the dataset object is **multiplied by the number of ranks**.
- If you create extra caches (e.g. a Manager-based global cache), they are also multiplied and can explode CPU memory usage.

### 2.2 DistributedSampler and epoch semantics

With DDP you should almost always use a `DistributedSampler` (or equivalent) so that each process sees a disjoint subset of the dataset:

```python
from torch.utils.data import DataLoader, DistributedSampler

train_sampler = DistributedSampler(
    dataset=train_ds,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
)
```

Key points:

- The effective batch size is `per_gpu_batch_size × world_size`.
- You must call `train_sampler.set_epoch(epoch)` every epoch so that each rank shuffles differently but still covers the full dataset without overlap.
- When using `Transformers` `Trainer` or Accelerate, this is handled for you, but it is useful to know what’s happening.

### 2.3 Dataset design in DDP: avoid global Python caches

From your own experiments with HDF5 datasets and a `CachedDataset` wrapper, one central lesson emerges:

- **DDP creates separate dataset instances per rank**. If each dataset holds the entire data in a `multiprocessing.Manager().dict()` (or similar), you effectively try to cache your full dataset *once per rank*, which is catastrophic for 100+ GB corpora.
- Manager-based cross-process caches serialize large Python objects and store them in a separate server process; this adds more copies and overhead.

Good patterns for DDP datasets:

- Implement the dataset as a **thin index-to-example view** over storage (HDF5, Arrow, Parquet, JSONL on disk, etc.).
- Load samples **on demand** in `__getitem__` rather than pre-loading everything.
- Avoid long-lived Python dictionaries or lists of full samples.
- Use library backends that support memory mapping and shared OS page cache (HF Datasets + Arrow/Parquet is a good example).

This also applies when you go beyond “raw” DDP to higher-level tools like Accelerate, FSDP, and DeepSpeed: data loading still uses these underlying mechanics.

---

## 3. HF Accelerate and Transformers Trainer in multi-GPU setups

### 3.1 Trainer + Accelerate basics

The Hugging Face `Trainer` is already multi-GPU aware:

- Internally it uses **Accelerate** for device placement and distributed training.
- You do not need to write DDP boilerplate; distribution is controlled via:
  - `TrainingArguments` (e.g. `per_device_train_batch_size`, `gradient_accumulation_steps`, `ddp_find_unused_parameters`), and
  - the **launcher** (`python`, `torchrun`, or `accelerate launch`).

For single-node, multi-GPU fine-tuning, a minimal workflow can look like:

```bash
accelerate config  # choose multi-GPU, no DeepSpeed/FSDP initially
accelerate launch train.py
```

and inside `train.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

training_args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=bsz,
    gradient_accumulation_steps=grad_acc,
    num_train_epochs=epochs,
    fp16=True,                  # or bf16=True where supported
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
```

Under the hood:

- Accelerate creates one process per GPU and wraps the model in DDP.
- Trainer uses `DistributedSampler` equivalents for datasets.
- Gradients are synchronized across GPUs each step.

### 3.2 Accelerate as a low-level multi-GPU engine

If you write your own training loop, you can call Accelerate directly:

```python
from accelerate import Accelerator

accelerator = Accelerator()  # multi-GPU config is read from `accelerate config`

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

for batch in train_loader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

This abstracts:

- Device placement (GPU, TPU, CPU).
- Mixed precision (fp16, bf16, etc.).
- Distributed training (DDP, FSDP, DeepSpeed).

### 3.3 Common Accelerate/FSDP pitfalls (from your FSDP OOM notes)

Key lessons from your experiments:

- An `accelerate` YAML config can **turn on extra features** like TorchDynamo / `torch.compile` (Inductor) or FSDP even if your script itself looks minimal.
- Enabling `dynamo_backend: INDUCTOR` can increase **peak GPU memory** compared to plain eager mode, sometimes enough to turn a borderline configuration into an OOM.
- If you are already near the memory limit on a single GPU:
  - Prefer **plain eager mode** (no TorchDynamo) until you have comfortable headroom.
  - Keep `distributed_type: "NO"` and `num_processes: 1` until you explicitly need multi-GPU.
  - Use FSDP/DeepSpeed only when you really need sharding; they introduce additional complexity.

A safe pattern:

- For initial single-GPU fine-tuning with `Trainer`, do **not** use `accelerate launch`. Just run

  ```bash
  python train.py
  ```

  and let Trainer’s internal Accelerate handle device placement.
- Introduce `accelerate launch` only when:
  - You need multi-GPU, or
  - You need to experiment with FSDP/DeepSpeed configurations.

When you do move to `accelerate launch`, explicitly control:

- `dynamo_backend` (start with `"NO"`).
- `distributed_type` (e.g. `MULTI_GPU`, `FSDP`, or `DEEPSPEED` only when needed).
- Unused FSDP/DeepSpeed config blocks (delete or minimize them until required).

---

## 4. Data pipelines for multi-GPU fine-tuning

### 4.1 From HDF5 to Arrow/Parquet and HF Datasets

Your HDF5 + `CachedDataset` experiment demonstrated a classic anti-pattern: caching full samples in a cross-process `Manager().dict()` can consume hundreds of GB of CPU RAM when combined with DDP.

Safer alternatives:

- **Plain HDF5Dataset**:
  - Keep the dataset as a thin `__getitem__` view over HDF5 groups.
  - Open one HDF5 file handle per process (`self.f = h5.File(..., "r")`), which is the typical SWMR/multi-process pattern.
  - Read only the sample you need, on demand.
- **Hugging Face Datasets (Arrow/Parquet)**:
  - Store your data in Arrow-backed datasets on disk.
  - Use `load_dataset` with `with_format("torch")` to get PyTorch tensors on the fly.
  - Data is memory-mapped and OS page cache is shared across processes, so many ranks can efficiently read from the same files without each holding a copy in Python RAM.

Example:

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})
ds = ds["train"].with_format("torch")

# In a DDP rank:
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
```

This approach scales better than custom Python caches for large corpora (100+ GB).

### 4.2 Worker count, prefetching, and memory

Even with lightweight datasets, DataLoader **workers** multiply memory usage:

- Each worker holds one or more pre-fetched batches in RAM.
- Memory scales roughly with `num_workers × prefetch_factor × batch_size`.

Guidelines in multi-GPU setups:

- Start with `num_workers=2`–`4` per process and a moderate `prefetch_factor` (default is often fine).
- Monitor RAM usage with tools like `htop` or `psutil`; reduce workers if CPU RAM climbs too high.
- Beware that DDP multiplies workers: `world_size × num_workers` total worker processes.

### 4.3 Dataset schemas for SFT and preference tuning

Schema design is independent of parallelism, but it interacts with I/O and tokenization costs. Common patterns for LLM fine-tuning:

- **Instruction SFT**:

  ```json
  {
    "instruction": "Explain what LoRA is.",
    "input": "",
    "output": "LoRA is a parameter-efficient fine-tuning technique that ..."
  }
  ```

- **Chat-style SFT**:

  ```json
  {
    "messages": [
      {"role": "user", "content": "Teach me QLoRA briefly."},
      {"role": "assistant", "content": "QLoRA is a way to ..."}
    ]
  }
  ```

- **Preference data (DPO/ORPO/KTO)**:

  ```json
  {
    "prompt": "Write a secure password reset email.",
    "chosen": "Dear user, ... (good email)",
    "rejected": "Hey, here is your password in plain text: ..."
  }
  ```

For multi-GPU runs, aim for schemas that:

- Are easy to tokenize in a streaming/map fashion.
- Avoid large nested Python objects that are expensive to pickle.
- Can be compressed and stored in a columnar format (Arrow/Parquet) if the corpus is huge.

---

## 5. TRL trainers (SFT, DPO, KTO, etc.) on multiple GPUs

### 5.1 How TRL uses Accelerate

The TRL trainers (e.g. `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `PPOTrainer`, GRPO variants) all rely on **Accelerate** for distributed training. The pattern is:

1. You configure multi-GPU / DeepSpeed / FSDP with `accelerate config`.
2. You launch your script with `accelerate launch train.py` (or `torchrun`, depending on the example).
3. TRL’s trainer takes a model, tokenizer, dataset, and training args; it does `accelerator.prepare(...)` inside and handles device placement.

So the same rules from Sections 2 and 3 apply:

- One process per GPU.
- Datasets must be CPU-based or device-agnostic; Accelerate moves batches to the right device.
- You should not manually shard the model with `device_map="auto"` inside a single process.

### 5.2 KTOTrainer and device-mismatch errors

Your KTO notes highlight a very specific but important pitfall:

- If you load a model with `device_map="auto"` across multiple GPUs in a single process, KTO’s loss computation may see tensors from `cuda:0` and `cuda:1` in the same operation and crash with

  > Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0 ...

- If you call `dataset.set_format("torch", device="cuda:1")`, you can hit a similar problem: batches are pre-placed on one GPU while the model or internal buffers live on another.

Safer practice for KTO (and other TRL trainers):

- Load the model **without** `device_map="auto"` in the training script:

  ```python
  model = AutoModelForCausalLM.from_pretrained("your-model", torch_dtype=torch.bfloat16)
  ```

- Keep the dataset on CPU:

  ```python
  dataset.set_format("torch")  # no device argument
  ```

- Let **Accelerate** handle multi-GPU via `accelerate launch` and the configuration file.

KTO and other preference-based trainers are multi-GPU capable, but they assume "one device per process" managed by Accelerate, not manual model parallelism inside a single process.

### 5.3 TRL distributed training pattern

A typical multi-GPU TRL run looks like:

```bash
accelerate config  # choose MULTI_GPU, optionally DeepSpeed or FSDP
accelerate launch train_kto.py  # or train_sft.py, train_dpo.py, etc.
```

Inside `train_kto.py` (simplified):

```python
from trl import KTOTrainer, KTOConfig

kto_config = KTOConfig(
    per_device_train_batch_size=bsz,
    gradient_accumulation_steps=grad_acc,
    num_train_epochs=epochs,
    learning_rate=lr,
)

trainer = KTOTrainer(
    model=model,
    args=kto_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
```

Guidelines:

- Avoid any manual `.to("cuda:x")` calls on the model or batches; let TRL/Accelerate do it.
- Do not combine TRL’s trainers with `device_map="auto"`; if the model is too big to fit on one GPU, consider FSDP/DeepSpeed via Accelerate instead.

---

## 6. FSDP and DeepSpeed for sharded fine-tuning

### 6.1 When do you truly need FSDP or DeepSpeed?

FSDP and DeepSpeed ZeRO are designed for models that are **too large to fit on a single GPU**, or when you want to:

- Increase batch size significantly without OOM.
- Train or fine-tune very large models (tens of billions of parameters) or long-context variants.

If your model plus optimizer states comfortably fit on one GPU with fp16/bf16 and modest batch sizes, plain DDP with data parallelism is often simpler and more robust.

### 6.2 FSDP in Transformers and Accelerate

There are two main entry points:

1. **Transformers + Accelerate FSDP guides**: show how to use FSDP in custom loops or with `Trainer` to shard models across GPUs.
2. **PEFT + FSDP**: for LoRA/QLoRA fine-tuning, some official examples demonstrate how to combine PEFT with FSDP so that adapters are sharded as well.

Key ideas:

- FSDP shards parameters, gradients, and optimizer states across GPUs; each rank only holds a subset at any given time.
- There are different sharding strategies (full, shard-grad-only, etc.) and AutoWrap policies (e.g. transformer-layer-based), which influence memory use vs communication cost.
- Accelerate’s FSDP integration requires a config that specifies sharding strategy, mixed precision, and checkpointing options.

From your experiments:

- It is easy to accidentally enable FSDP in an Accelerate config while still running only a single GPU; this usually does not help memory, and can complicate saving/loading or inference.
- Start with simple, official example configs and avoid editing many keys at once. Change only a few parameters (`zero_stage` or `fsdp_sharding_strategy`, precision, offload) initially.

### 6.3 DeepSpeed ZeRO and its HF integrations

DeepSpeed provides **ZeRO-1/2/3** optimizer states partitioning and optional CPU/NVMe offload. Hugging Face exposes this in three main ways:

- Native **DeepSpeed launcher** (`deepspeed train.py --deepspeed ds_config.json`).
- **Transformers Trainer** with `deepspeed="ds_config.json"` argument.
- **Accelerate + DeepSpeed**: configure DeepSpeed in `accelerate config` and run `accelerate launch`.

Common patterns for fine-tuning:

- Use ZeRO-1 or ZeRO-2 for moderate models when you want larger batch sizes.
- Use ZeRO-3 with offload for very large models that cannot fit otherwise, accepting slower step times.
- Combine DeepSpeed with LoRA/QLoRA for parameter-efficient fine-tuning on fewer GPUs.

Your DeepSpeed learning notes emphasize an effective learning path:

1. Start with **Transformers Trainer + DeepSpeed** using an official example and minimal config.
2. Read the **DeepSpeed config JSON reference** to understand essential keys (ZeRO stage, fp16/bf16, gradients, offload).
3. Only then branch into native DeepSpeed scripts or complex multi-node setups.

### 6.4 Accelerate concept guide: FSDP vs DeepSpeed

Recent Accelerate concept guides explicitly compare FSDP and DeepSpeed:

- FSDP is the torch-native approach; DeepSpeed ZeRO inspired its sharding strategies.
- Accelerate aligns configuration so that you can swap between FSDP and DeepSpeed with similar semantics (e.g. mapping `FULL_SHARD` to `zero_stage=3`).
- Low-precision training (fp16/bf16/fp8) and optimizer behavior can differ between FSDP and DeepSpeed; you need to pay attention to default upcasting and optimizer precision.

Takeaway for practical fine-tuning:

- For **single-node multi-GPU**, FSDP is increasingly the “default” sharding option in the PyTorch/HF ecosystem.
- DeepSpeed is still a strong choice for mature ZeRO-3 setups, offloading, and some specialized features.
- Use Accelerate’s examples and concept guides rather than building configs entirely from scratch.

---

## 7. Putting it together: design recipes for multi-GPU fine-tuning

### 7.1 Simple recipe: multi-GPU SFT with Trainer + DDP

**Goal**: Fine-tune a 7B–14B model that fits in single-GPU memory, but train faster using multiple GPUs.

1. Prepare data in a simple JSONL schema (instruction or chat style) and load it with HF Datasets.
2. Start with single-GPU `Trainer` run to verify loss decreases and samples look good.
3. Create an Accelerate config for multi-GPU without DeepSpeed/FSDP and with `dynamo_backend: "NO"`.
4. Launch with `accelerate launch train.py`.
5. Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` so that total batch = `per_device × world_size × grad_acc` matches or slightly exceeds the single-GPU baseline.

Outcome:

- Training time per epoch drops roughly proportional to number of GPUs (minus communication overhead).
- Model behavior is identical to single-GPU; you just get results sooner.

### 7.2 Sharded recipe: LoRA/QLoRA on 34B+ models

**Goal**: Fine-tune a larger model (e.g. 34B) with LoRA/QLoRA across multiple GPUs because a single GPU is not enough.

High-level steps:

1. Choose a PEFT framework (PEFT + Transformers, TRL `SFTTrainer` or Unsloth-like tooling).
2. Enable FSDP or DeepSpeed ZeRO in Accelerate/Trainer:
   - For FSDP: use an official example config and keep other hyperparameters simple.
   - For DeepSpeed: start from a minimal ZeRO-2 or ZeRO-3 config from docs.
3. Run `accelerate config` and select the chosen backend, number of processes, precision, and offload options.
4. Use modest per-device batch sizes and rely on gradient accumulation to get a reasonable global batch.

Key checks:

- Test **saving and loading** checkpoints early; sharded training changes checkpoint formats.
- Test **inference** on a single GPU (or CPU) from a checkpoint to ensure your save/load logic is correct.

### 7.3 TRL preference tuning recipe (KTO/DPO) on 2–4 GPUs

**Goal**: Run KTO or DPO on 2–4 GPUs for alignment without running into device mismatch errors.

1. Prepare preference data (`prompt`, `chosen`, `rejected` or KTO-specific schema) as HF Dataset.
2. Load model without `device_map="auto"` and without manually calling `.to("cuda:x")`.
3. Keep dataset on CPU; do not set `device=...` in `set_format`.
4. Configure Accelerate for multi-GPU (DDP) with no DeepSpeed/FSDP initially.
5. Launch `accelerate launch train_kto.py` (or corresponding script).

If you later need sharding, transition to FSDP/DeepSpeed via official TRL examples and keep the same “no manual device_map” rule.

---

## 8. Limitations, caveats, and open questions

### 8.1 Limits of this knowledge base

- Focuses on **single-node multi-GPU**; multi-node introduces networking, rendezvous, and cluster-specific issues that are only lightly touched upon.
- Assumes **PyTorch + HF ecosystem**; JAX/Flax, TensorFlow, or other stacks have analogous but different tools.
- Does not cover exotic ND-parallel strategies in depth; it highlights them but focuses on the more common DDP/FSDP/ZeRO setups for fine-tuning.

### 8.2 Common failure modes to watch for

- **CPU RAM blow-up**:
  - Caused by dataset-level caching (e.g. Manager-based global caches) replicated across ranks.
  - Mitigate with on-demand reads and Arrow/Parquet-based datasets.
- **CUDA OOM in multi-GPU despite fitting on single-GPU eager mode**:
  - Often caused by extra features like TorchDynamo/Inductor, or by FSDP/DeepSpeed configs that increase peak memory.
- **Device mismatch errors (`Expected all tensors to be on the same device`)**:
  - Usually due to manual model sharding (`device_map="auto"`) or putting datasets directly on a GPU while Accelerate expects to handle data movement.

### 8.3 Open questions and areas for further exploration

- How to best combine **vLLM**-based inference with TRL-based training in a distributed setting (co-locating fine-tuning and teacher models for distillation).
- When to move from single-node sharding to **multi-node ND-parallel** for very large models, and how to keep configs manageable.
- Robust patterns for **long-context fine-tuning** (e.g. 128k tokens) across multiple GPUs, where sequence parallelism and context parallelism become important.

---

## 9. References and further reading

This section lists a curated set of high-signal resources you can revisit while designing or debugging multi-GPU fine-tuning pipelines.

### 9.1 Hugging Face docs and blogs

- PyTorch distributed and FSDP guides integrated with Accelerate and Transformers:  
  - [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)  
  - [Accelerate basic launch tutorial](https://huggingface.co/docs/accelerate/basic_tutorials/launch)  
  - [FSDP usage guide in Accelerate](https://huggingface.co/docs/accelerate/usage_guides/fsdp)  
  - [FSDP vs DeepSpeed concept guide](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed)  
- DeepSpeed integrations:  
  - [Accelerate + DeepSpeed usage guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)  
  - [Transformers DeepSpeed integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)  
- Multi-GPU and sharded training blog posts:  
  - [Accelerate large model training with DeepSpeed](https://huggingface.co/blog/accelerate-deepspeed)  
  - [Accelerate large model training with PyTorch FSDP](https://huggingface.co/blog/pytorch-fsdp)  
  - [From DeepSpeed to FSDP and back again](https://huggingface.co/blog/deepspeed-to-fsdp-and-back)  
  - [TRL distributed training docs](https://huggingface.co/docs/trl/distributing_training)

### 9.2 PyTorch distributed docs

- [Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)  
- [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)  
- [Multi-GPU DDP tutorial](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)  
- [DDP notes and reference](https://pytorch.org/docs/stable/notes/ddp.html)

### 9.3 DeepSpeed and FSDP focused guides

- DeepSpeed official site:  
  - [Getting Started](https://www.deepspeed.ai/getting-started/)  
  - [ZeRO, ZeRO-Offload, ZeRO-Infinity tutorials](https://www.deepspeed.ai/tutorials/zero/)  
  - [Training overview](https://www.deepspeed.ai/training/)  
- Community explainers:  
  - W&B report on ZeRO with HF Trainer:  
    [A Guide to DeepSpeed ZeRO With the HuggingFace Trainer](https://wandb.ai/byyoung3/ml-news/reports/A-Guide-to-DeepSpeed-Zero-With-the-HuggingFace-Trainer--Vmlldzo2ODkwMDc4)

### 9.4 Data and dataset resources

- Hugging Face Dataset formats and best practices:  
  - [LLM Dataset Formats 101](https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face)  
  - [HF Datasets documentation](https://huggingface.co/docs/datasets)

### 9.5 Troubleshooting and community knowledge

- TRL / KTO multi-GPU issues and discussions (device mismatch, dataset device placement):  
  - [TRL GitHub issues (KTO and device maps)](https://github.com/huggingface/trl/issues)  
- FSDP and DeepSpeed forum threads and Q&A:  
  - [Hugging Face Forums - FSDP and DeepSpeed](https://discuss.huggingface.co/tag/fsdp)  
  - [Stack Overflow "deepspeed" tag](https://stackoverflow.com/questions/tagged/deepspeed)  

These links, combined with your own local notes on datasets, Accelerate configs, KTO behavior, and DeepSpeed, provide a solid foundation for designing and debugging multi-GPU fine-tuning pipelines in the Hugging Face ecosystem.
