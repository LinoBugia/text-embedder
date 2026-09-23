---
source: "huggingface+chat+files+web"
topic: "LLM fine-tuning: data, tooling, evaluation, and ecosystem"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T07:12:48.899731+00:00"
---


# LLM Fine-Tuning: From Foundations to Practical Pipelines

This knowledge base collects core concepts, patterns, and practical recipes for fine-tuning large language models (LLMs), with an emphasis on the Hugging Face ecosystem (Transformers, Datasets, TRL, Spaces, Jobs), Unsloth, and current community practices around reasoning and code models.

It is intentionally redundant and dense: key ideas are restated from multiple angles so you can dip into any section independently.

---

## 1. Background and mental model

### 1.1 What fine-tuning a LLM means

A large language model is a big neural network (usually a Transformer) trained to predict the next token in a sequence. Pre-training teaches it broad world knowledge and language patterns using trillions of tokens and huge compute budgets.

Fine-tuning means you:

- Start from an already pre-trained model (e.g. Llama 3.x, Qwen, DeepSeek, Gemma, GLM, GPT-OSS, etc.).
- Continue training it on a **smaller, specialized dataset** that encodes the behavior, domain, or style you want.
- Update either all parameters or a small adapter subset (LoRA/QLoRA) using gradient descent.

You are sculpting behavior into an existing model rather than training from scratch.

### 1.2 Fine-tuning vs prompting vs RAG vs pre-training

Think of four levers you can pull:

- **Prompting / system prompts / tool calling**  
  - No weight updates.  
  - You change behavior by changing instructions, examples, or tools.  
  - Fast, flexible, but sometimes brittle and hard to keep consistent.

- **RAG (retrieval-augmented generation)**  
  - No weight updates.  
  - You attach an external knowledge base and feed retrieved chunks into the prompt.  
  - Excellent for large, frequently changing corpora; not great for changing core behavior.

- **Fine-tuning**  
  - You permanently change the model’s weights.  
  - Good for domain style, safety tone, chain-of-thought habits, tool-use protocol, or compressing a huge prompt into the model itself.

- **Pre-training / continued pre-training**  
  - Pre-training from scratch is rarely done by individuals.  
  - Continued pre-training on domain text is a middle ground: you push the model’s distribution toward your domain before doing supervised instruction fine-tuning.

In modern systems you often combine all of them. Example stack:

- Use RAG for fresh domain documents.  
- Use prompts and tools for orchestration.  
- Use fine-tuning for stable behavior (formatting, tone, reasoning style, safety).

### 1.3 Main fine-tuning paradigms

In current open-source practice you see a small set of recurring paradigms:

- **Supervised fine-tuning (SFT)**  
  - Train on (input, output) or (prompt, response) pairs.  
  - Typical for instruction/chat tuning, code assistants, tool-use patterns, OCR post-processing, etc.

- **Continued pre-training / domain adaptation**  
  - Train on raw domain text or code with a language modeling loss.  
  - Shifts the base distribution without explicit instructions.

- **PEFT (parameter-efficient fine-tuning)**  
  - Methods like **LoRA** and **QLoRA** add tiny trainable matrices and keep base weights frozen.  
  - Makes it possible to fine-tune 7B–14B models on a single 8–24 GB GPU.

- **Preference-based tuning / alignment**  
  - **DPO, ORPO, KTO**: train on pairs (prompt, chosen, rejected).  
  - **PPO / GRPO**: RL-style updates using a learned reward model or environment score.  
  - Used after SFT to align with human or task preferences.

- **Distillation**  
  - A smaller student model learns from a larger teacher (often via chain-of-thought traces).  
  - Many reasoning models are trained this way from models like DeepSeek-R1 or similar teachers.

Real projects often chain these: continued pre-training → SFT → preference/RL tuning → lightweight updates after deployment.

---

## 2. Data and dataset formats

### 2.1 Why data quality dominates

Across official blogs, Unsloth docs, and arXiv surveys, everyone converges on the same point:

> **Data quality matters more than model or optimizer tweaks once you pick a reasonable base model.**

Fine-tuning largely teaches the model whatever patterns are in your dataset:

- Style (formal vs casual, verbose vs concise).  
- Safety and refusal patterns.  
- Domain vocabulary and reasoning style (finance, law, math, code).  
- Formatting (JSON schemas, Markdown formats, custom DSLs).

If your data is noisy, inconsistent, or misaligned, fine-tuning will faithfully reproduce those issues.

### 2.2 Practical dataset schemas in the Hugging Face world

The Hugging Face community article “LLM Dataset Formats 101: A No‑BS Guide for Hugging Face Devs” is a concise reference for common schemas.  
Key patterns (written here in simplified form):

- **Plain text (for LM / continued pre-training)**  

  ```json
  { "text": "... arbitrary text ..." }
  ```

- **Instruction SFT**  

  ```json
  {
    "instruction": "Explain what LoRA is.",
    "input": "",
    "output": "LoRA is a parameter-efficient fine-tuning technique that ..."
  }
  ```

- **Chat-style SFT**  

  ```json
  {
    "messages": [
      {"role": "user", "content": "Teach me QLoRA briefly."},
      {"role": "assistant", "content": "QLoRA is a way to ..."}
    ]
  }
  ```

- **Preference data (for DPO / ORPO)**  

  ```json
  {
    "prompt": "Write a secure password reset email.",
    "chosen": "Dear user, ... (good email)",
    "rejected": "Hey, here is your password in plain text: ..."
  }
  ```

Guidelines:

- Choose schema based on training objective (SFT vs preference vs RL).  
- Stick to common column names when possible (`instruction`, `input`, `output`, `messages`, `prompt`, `chosen`, `rejected`).  
- Keep raw text so you can retokenize for different base models without regenerating data.

### 2.3 File formats: JSONL, CSV, Parquet, raw text

In practice you will mostly see:

- **JSONL** for SFT and preference data (one JSON object per line).  
- **CSV/TSV** for simple small datasets.  
- **Parquet** for large scaled pipelines (columnar, compressed).  
- **Plain text** for continued pre-training.

The Hugging Face Datasets library lets you treat them uniformly via `load_dataset` and stream, map, and split them easily.

Example: load JSONL train/validation splits:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "val.jsonl",
    },
)

print(ds["train"][0])
```

### 2.4 Synthetic data generation

Modern open-source recipes frequently use synthetic data generated from a stronger teacher model:

- Use a high-quality model (hosted via HF Inference, Unsloth, local vLLM, etc.) to create instruction/answer or chain-of-thought pairs.  
- Use helpers or CLIs (e.g. community tools similar to “Completionist”) to iterate through base datasets and augment them with completions.  
- Filter, de-duplicate, and quality-check synthetic data before training.

For reasoning and code, many datasets now come from distilling traces from deep reasoning models (e.g. DeepSeek-R1, Open-R1-style models).

---

## 3. Seven-stage pipeline for LLM fine-tuning

A robust fine-tuning project can be described as a seven-stage loop:

1. **Define objectives and constraints**  
   - What skills or behaviors do you want? (domain coverage, reasoning depth, coding quality, style, safety).  
   - What constraints? (latency, memory, on-device vs cloud, licensing, privacy).

2. **Collect and prepare data**  
   - Gather domain text, code, logs, chat transcripts, or user interactions.  
   - Clean (remove PII, profanity where appropriate), normalize formats, deduplicate near-duplicates.  
   - Convert to JSONL/Parquet with clear schemas.

3. **Choose base model and adaptation method**  
   - Model family (Llama, Qwen, DeepSeek, Gemma, GLM, GPT-OSS, etc.).  
   - Size vs hardware (1–8B for light setups, 14–34B for stronger performance, MoE if you have more resources).  
   - Adaptation method: SFT, LoRA/QLoRA, continued pre-training, DPO/ORPO, GRPO, distillation.

4. **Design training configuration**  
   - Framework: Transformers Trainer, TRL, Unsloth, or custom Accelerate loops.  
   - Hyperparameters: learning rate, batch size, gradient accumulation, epochs, context length, warmup, weight decay.  
   - Memory tricks: 4‑bit quantization (QLoRA), gradient checkpointing, FlashAttention, fused ops.

5. **Run training**  
   - Monitor training and validation loss.  
   - Watch for divergence or overfitting.  
   - Log sample generations regularly for qualitative inspection.

6. **Evaluate and iterate**  
   - Evaluate on held-out dev/test sets and benchmarks (e.g., HF leaderboards, code/agent benchmarks, internal unit tests).  
   - Check both task performance and general abilities (MMLU/GSM8K-like tasks).  
   - Iterate on data (often the biggest gains come from better data, not bigger models).

7. **Deploy and monitor**  
   - Export to your target inference stack (Transformers, GGUF for llama.cpp/Ollama, vLLM, Inference Endpoints).  
   - Monitor usage, drift, and failure cases.  
   - Periodically re-fine-tune or distill into smaller models as requirements evolve.

This high-level loop is stable across tools: whether you use HF Trainer, TRL, Unsloth, or custom code, the lifecycle stays the same.

---

## 4. Hugging Face tooling stack

### 4.1 Transformers + Trainer

The `transformers` library plus the `Trainer` API is the classic fine-tuning workhorse:

- Supports causal LM, seq2seq, masked LM, and more.  
- Integrates with Datasets for data loading.  
- Handles checkpointing, logging, and evaluation loops.

A minimal SFT-like fine-tune (without LoRA) looks like:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def format_example(example):
    prompt = f"Instruction: {example['instruction']}\n\nAnswer:"
    return {"text": prompt + example["output"]}

ds = ds.map(format_example)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048)

tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir="./llm_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

trainer.train()
```

For real projects you nearly always pair this with PEFT (LoRA/QLoRA) and memory optimizations.

### 4.2 TRL (Transformers Reinforcement Learning)

[TRL](https://github.com/huggingface/trl) is the Hugging Face library for post-training and alignment:

- Supports SFT on chat/instruction data.  
- Implements DPO/ORPO/KTO for preference-based training.  
- Implements PPO/GRPO for RL with reward models or task environments.

The official “Supervised Fine-Tuning (SFT) with LoRA/QLoRA using TRL” notebook and community posts show that, thanks to quantization and LoRA, you can fine-tune **14B models on a free Colab T4 GPU** with careful configuration and small batch sizes.

Patterns to note:

- Use `SFTTrainer` with a chat template to apply the correct prompt format.  
- Use 4‑bit quantization (BitsAndBytes) and LoRA adapters to fit models into memory.  
- Combine with Unsloth or other efficiency tweaks when necessary.

### 4.3 Unsloth: efficient and opinionated fine-tuning

[Unsloth](https://github.com/unslothai/unsloth) is a fast-growing project focused on **efficient fine-tuning and RL for LLMs**. Their docs emphasize that:

- Unsloth = speed + memory efficiency + simplicity for fine-tuning.  
- You can get ~2× speed and ~70% less VRAM usage for many models compared to naive baselines.  
- It provides curated notebooks for many popular models (Llama, Qwen3, DeepSeek, Gemma, GLM, Kimi K2, etc.).

Important ideas:

- **QLoRA-first mindset**: 4‑bit quantization of base weights + LoRA adapters as the default path.  
- **Model collections**: ready-made models like `unsloth/llama-3.1-8b-unsloth-bnb-4bit` designed as starting points.  
- **End-to-end recipes**: guides for SFT, GRPO reasoning training, DPO/ORPO, and vision/OCR fine-tuning, often with built-in support for exporting to GGUF or to local runners like Ollama.

If you want a practical, batteries-included UX for fine-tuning, Unsloth is often simpler than writing everything from scratch in Trainer/TRL, while still working on top of Hugging Face models and tokenizers.

### 4.4 Other runtime and deployment tools

Fine-tuning is only half the story; you must deploy the result:

- **vLLM**: high-throughput transformer inference server, widely used for serving fine-tuned models.  
- **Text Generation Inference (TGI)**: HF’s own optimized serving stack.  
- **llama.cpp / GGUF + Ollama**: local/offline inference of quantized models; HF has an official “Use Ollama with any GGUF model on Hugging Face Hub” guide.  
- **HF Inference Providers / Inference Endpoints**: managed serving options when you do not want to manage GPUs yourself.

Most training stacks aim to export weights in a way that is compatible with at least one of these deployment options.

---

## 5. Task types and strategies

### 5.1 Instruction and chat fine-tuning

For instruction/chat models:

- Use high-quality, diverse instruction datasets (open collections plus your own domain data).  
- Apply the **correct chat template** for the base model (tokenizers now often expose `apply_chat_template`).  
- Start with SFT to make the model obey your desired format, tone, and protocol.  
- Optionally run DPO/ORPO afterwards to refine preferences (e.g., more helpful, less verbose, safer).

Checklist:

- Normalize roles (`user`, `assistant`, optional `system`).  
- Ensure answers are complete and consistent.  
- Avoid mixing too many conflicting styles in the same fine-tune unless you explicitly want that diversity.

### 5.2 Code and tool-use models

Specialized code models (Qwen Coder, DeepSeek-Coder, NextCoder, Devstral, etc.) and reasoning models like OlympicCoder show some common patterns:

- Start from a **code-focused base model** rather than a generic chat model.  
- Use realistic multi-turn logs: problem → attempts → compiler/runtime errors → incremental fixes.  
- Evaluate with serious code benchmarks (SWE-bench, LiveCodeBench, IOI-style tasks).

Fine-tuning strategies for code:

- SFT on high-quality QA/code pairs, including test-driven development examples.  
- Distillation from strong closed or open models to capture reasoning traces.  
- RL or GRPO on environment-based tasks (run tests, keep if pass).

### 5.3 Reasoning and chain-of-thought

Reasoning-focused efforts (e.g. Open-R1-style models, OlympicCoder, math reasoning models) emphasize:

- Training on full **reasoning traces**, not just final answers.  
- Mixing math, logic, code, and natural language reasoning data.  
- Being careful with **sequence packing** and truncation; breaking traces harms performance.

A common pattern:

1. Use a strong teacher model to generate step-by-step solutions.  
2. Fine-tune a student model on those traces (SFT).  
3. Optionally run GRPO-like RL to further improve reasoning quality and robustness.

### 5.4 Fine-tuning vs RAG and agents

Often you can fix a problem without fine-tuning at all:

- If failures are about **missing knowledge**, improve RAG (indexing, chunking, retrieval).  
- If failures are about **multi-step workflows**, consider agent frameworks and tool calling.  
- If failures are about **style, persona, safety, or protocol adherence**, fine-tuning is usually appropriate.

A good strategy is:

1. Build a robust RAG/agent stack first.  
2. Log user sessions and identify true model behavior failures.  
3. Convert those logs into SFT or preference datasets.  
4. Fine-tune to harden those behaviors.

---

## 6. Evaluation and leaderboards

### 6.1 Hugging Face leaderboards and the Big Benchmarks Collection

Hugging Face maintains:

- **Leaderboards** (e.g. Open LLM Leaderboard) and associated docs explaining metrics and datasets.  
- The **Big Benchmarks Collection**, a curated bundle of standard tasks (MMLU, GSM8K, ARC, etc.).

These are useful for:

- Comparing your fine-tuned model with public baselines.  
- Ensuring you do not catastrophically regress general capabilities while specializing.  
- Getting a quick sense of where your model stands on broad abilities.

### 6.2 Task-specific benchmarks

For certain domains your evaluation should mirror real workloads:

- **Code**: SWE-bench (including Bash-only), LiveCodeBench, Aider leaderboards.  
- **Agents**: frameworks like OpenEnv or specialized evaluation Spaces.  
- **RAG**: retrieval and answer quality benchmarks, often built as HF Spaces or custom harnesses.  
- **Domain-specific**: your own unit tests, QA sets, or simulation environments.

Where possible, automate these evaluations and make them part of your training pipeline (e.g., run them after each fine-tune and compare to baseline).

### 6.3 Practical evaluation loop

A simple practical loop for small teams:

1. Create a small but representative **golden set** of prompts (50–200) with good reference answers.  
2. Evaluate the base model on this set and record scores (automatic and human).  
3. Fine-tune your model.  
4. Re-run evaluation and compare.  
5. Periodically expand and improve the golden set based on real user interactions and new edge cases.

This gives you a focused, evolving view of whether fine-tuning actually helps.

---

## 7. Hardware, runtimes, and practical constraints

### 7.1 Memory and compute constraints

Key practical facts:

- Full fine-tuning 7B+ models in full precision can require **tens of GB of GPU memory** just for optimizer states (AdamW keeps 3 copies of parameters).  
- PEFT (LoRA/QLoRA) with quantization can reduce memory dramatically and make 7B–14B models feasible on 8–24 GB GPUs.

Unsloth, together with quantization and clever scheduling, makes it routine to fine-tune mid-sized models on:

- Free or low-cost cloud instances (T4/L4/A10G).  
- Consumer GPUs (e.g. 3090/4090 class).

### 7.2 Typical single-GPU recipe (QLoRA + SFT)

A robust starting recipe for many tasks:

1. Choose a 7B–8B instruction model compatible with your license and domain.  
2. Quantize to 4‑bit using BitsAndBytes (QLoRA) or use a pre-quantized Unsloth model.  
3. Apply LoRA adapters on attention and MLP layers (rank 8–64, depending on capacity).  
4. Train 1–3 epochs on your dataset with a moderate learning rate (e.g. 2e‑4 for adapters).  
5. Regularly inspect sample outputs and evaluation metrics.

Extensions:

- For very small datasets, rely heavily on strong base models and small learning rates to avoid overfitting.  
- For very large datasets, pay attention to training stability and schedule (warmup, cosine decays).

### 7.3 Deployment targets

After fine-tuning you usually export to one or more of:

- **Transformers + vLLM** for high-throughput server inference.  
- **GGUF + llama.cpp/Ollama** for local/offline use.  
- **HF Inference Endpoints / Inference Providers** for managed serving.  
- **Custom backends** wrapped around OpenAI-compatible APIs (e.g. router-style endpoints).

Choosing the deployment format early helps you ensure compatibility (e.g. paying attention to maximum context length, tokenizer choice, and architecture support).

---

## 8. Practical checklists

### 8.1 Minimal SFT checklist

- [ ] Decide on objective (style, domain, reasoning depth, safety).  
- [ ] Choose a base model and verify license.  
- [ ] Design dataset schema (`messages` or `instruction`/`output`).  
- [ ] Clean and deduplicate data; remove obviously harmful or low-quality samples.  
- [ ] Implement data loading via `datasets`.  
- [ ] Choose a PEFT strategy (LoRA/QLoRA) and framework (Transformers + PEFT, TRL, Unsloth).  
- [ ] Configure conservative hyperparameters and verify that loss decreases.  
- [ ] Evaluate before/after on golden sets and at least one public benchmark.  
- [ ] Push model and dataset to Hugging Face Hub with clear cards and metadata.

### 8.2 When to consider preference/RL methods

Consider DPO/ORPO/GRPO or PPO when:

- You care about nuanced preferences (e.g. “helpful but concise”, “strictly follow JSON schema”, “never reveal certain secrets”).  
- You can obtain pairwise feedback (chosen vs rejected answers) or reward signals from tests/environments.  
- SFT alone leads to models that are capable but inconsistent in following your desired preferences.

Be aware that preference and RL methods are more sensitive to configuration and reward design; start small and monitor carefully.

---

## 9. Curated resources and links

This section lists a small, opinionated set of external references you can follow up with. All URLs are standard web links (no internal IDs).

### 9.1 Data and dataset formats

- LLM Dataset Formats 101 (Hugging Face community article):  
  [https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face](https://huggingface.co/blog/tegridydev/llm-dataset-formats-101-hugging-face)
- The 1 Billion Token Challenge: Finding the Perfect Pre-training Mix:  
  [https://huggingface.co/blog/codelion/optimal-dataset-mixing](https://huggingface.co/blog/codelion/optimal-dataset-mixing)

### 9.2 Fine-tuning guides (HF + Unsloth)

- Fine-tuning LLMs Guide (Unsloth docs):  
  [https://docs.unsloth.ai/get-started/fine-tuning-llms-guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- Tutorials: How to Fine-tune & Run LLMs (Unsloth):  
  [https://docs.unsloth.ai/models/tutorials-how-to-fine-tune-and-run-llms](https://docs.unsloth.ai/models/tutorials-how-to-fine-tune-and-run-llms)
- Unsloth GitHub repository:  
  [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- Hugging Face blog – general LLM training and fine-tuning topics:  
  [https://huggingface.co/blog](https://huggingface.co/blog)

### 9.3 TRL and SFT examples

- TRL GitHub (SFT/DPO/GRPO, notebooks):  
  [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- Supervised Fine-Tuning (SFT) with LoRA/QLoRA using TRL (Colab notebook):  
  [https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb)
- Community post: fine-tuning a 14B model with TRL + SFT on a free Colab T4 GPU:  
  [https://huggingface.co/posts/sergiopaniego/990279445625588](https://huggingface.co/posts/sergiopaniego/990279445625588)

### 9.4 Courses and conceptual foundations

- Hugging Face LLM Course:  
  [https://huggingface.co/blog/mlabonne/llm-course](https://huggingface.co/blog/mlabonne/llm-course)
- Hugging Face Learn – LLMs, RAG, agents, diffusion, etc.:  
  [https://huggingface.co/learn](https://huggingface.co/learn)
- 3Blue1Brown – “But what is a neural network?” (intuition for training):  
  [https://www.3blue1brown.com/lessons/neural-networks](https://www.3blue1brown.com/lessons/neural-networks)
- “So You Want to Learn LLMs? Here's the Roadmap”:  
  [https://ahmadosman.com/blog/learn-llms-roadmap/](https://ahmadosman.com/blog/learn-llms-roadmap/)

### 9.5 Evaluation and leaderboards

- Hugging Face Leaderboards and Evaluations docs:  
  [https://huggingface.co/docs/leaderboards/index](https://huggingface.co/docs/leaderboards/index)
- The Big Benchmarks Collection (Hugging Face):  
  [https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a)
- SWE-bench Bash Only (code benchmark):  
  [https://www.swebench.com/bash-only.html](https://www.swebench.com/bash-only.html)
- Aider LLM Leaderboards (code copilots):  
  [https://aider.chat/docs/leaderboards/](https://aider.chat/docs/leaderboards/)
- OpenEnv (agent environments):  
  [https://huggingface.co/posts/sergiopaniego/527159223589050](https://huggingface.co/posts/sergiopaniego/527159223589050)

### 9.6 Deployment and runtimes

- Common AI model formats (GGUF, Safetensors, ONNX, etc.):  
  [https://huggingface.co/blog/ngxson/common-ai-model-formats](https://huggingface.co/blog/ngxson/common-ai-model-formats)
- Use Ollama with any GGUF model on Hugging Face Hub:  
  [https://huggingface.co/docs/hub/ollama](https://huggingface.co/docs/hub/ollama)
- llama-cpp-python (efficient local inference for GGUF):  
  [https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
