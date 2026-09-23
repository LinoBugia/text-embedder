---
source: "huggingface+chat+files+web"
topic: "LLM Fine-Tuning"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-22T19:03:56Z"
---

# LLM Fine-Tuning

This knowledge base summarizes how to fine-tune large language models (LLMs) in practice, with a focus on:

- The conceptual landscape of LLM fine-tuning (SFT, PEFT, RLHF/DPO-style alignment, continued pre-training).  
- A stage-by-stage pipeline grounded in the *Ultimate Guide to Fine-Tuning LLMs* and related research.  
- Concrete tools and recipes in the Hugging Face and Unsloth ecosystems.  

It is intentionally redundant: key ideas are repeated from several angles so each section can be read on its own.

---

## 1. Conceptual overview

### 1.1 What is fine-tuning?

**Fine-tuning** adapts a pretrained LLM to a specific task, domain, or style by continuing training on a smaller, targeted dataset. Compared to training from scratch, fine-tuning:

- Starts from a model that already has strong language understanding and some reasoning ability.  
- Requires **far less data and compute** than full pre-training.  
- Is usually framed as supervised learning (SFT) or preference optimization (RLHF-like training, DPO, GRPO, KTO, etc.).

Fine-tuning changes model parameters so that the model:

- Follows instructions better.  
- Speaks in a given tone or persona.  
- Specializes in a domain (e.g., law, medicine, finance, coding).  
- Obeys safety and alignment requirements defined by human or synthetic preference data.

### 1.2 Key fine-tuning regimes

You can think of LLM fine-tuning along three main axes:

1. **What is updated?**
   - Full-model fine-tuning (all weights).  
   - Parameter-efficient fine-tuning (PEFT) where only a small number of extra parameters are trained (LoRA, QLoRA, prefix-tuning, etc.).  

2. **What objective is used?**
   - **Supervised Fine-Tuning (SFT)** – minimize cross-entropy loss on desired outputs (answers, completions, messages).  
   - **Preference-based alignment** – optimize a preference or reward signal using DPO, ORPO, GRPO, PPO, KTO, etc.  
   - **Continued pre-training** – train on unlabeled or weakly labeled text to adapt to new domains, formats, or languages.

3. **What data structure is used?**
   - Plain text (for continued pre-training).  
   - Instruction/response pairs or chat-style messages (for SFT).  
   - Triples or pairs `(prompt, chosen, rejected)` (for DPO/ORPO-style methods).  
   - Trajectories or environment interactions (for RL-style reasoning training, such as GRPO with OpenEnv).

### 1.3 RAG vs fine-tuning

Retrieval-Augmented Generation (**RAG**) and fine-tuning solve different problems:

- **RAG**: Keep the base LLM fixed and augment it with a retrieval system (vector store, document index).  
  - Good when knowledge changes frequently (docs, FAQs, codebases).  
  - Avoids overwriting generic capabilities; easier to update by refreshing the index.  

- **Fine-tuning**: Change the model’s parameters.  
  - Good when you need new behavior built-in (style, reasoning patterns, structured outputs, domain expertise).  
  - Better for scenarios without reliable retrieval or when latency/edge deployment makes RAG harder.

In practice, many systems use **both**: RAG for up-to-date facts and fine-tuning for behavior, reasoning, and domain style.

---

## 2. A stage-by-stage pipeline (from the Ultimate Guide and practice)

The *Ultimate Guide to Fine-Tuning LLMs* proposes a multi-stage pipeline that closely matches what practitioners do in the field. We paraphrase it here into eight recurring stages you can apply to most projects:

1. **Problem framing and requirements**  
   - Clarify user stories: who will use the model, for what tasks, under which constraints.  
   - Decide whether you need a domain expert, a coding assistant, a RAG-aware agent, a reasoning model, etc.  
   - Identify latency, cost, and deployment constraints (cloud vs on-device, GPU vs CPU).

2. **Model and tokenizer selection**  
   - Choose a base model family: Llama, Qwen, Mistral, Phi, DeepSeek, etc.  
   - Match the model to the task:  
     - Code assistant → use a coder model (e.g., Qwen Coder, DeepSeek Coder, NextCoder).  
     - Reasoning-heavy tasks → use reasoning-optimized or GRPO-trained variants (Open R1, reasoning-focused Qwen, DeepSeek-R1-like).  
     - Multilingual or multimodal → use multilingual or vision-language models.  
   - Consider hardware fit (VRAM, quantization options). Guides like Unsloth’s “What model should I use?” help formalize this decision.

3. **Data strategy and curation**  
   - Decide which data types you need: instructions, chat logs, code, domain-specific documents, preference pairs, etc.  
   - Design schema: for example `{"messages": [...]}`, `{"instruction": ...,"output": ...}`, or preference triplets.  
   - Collect data (real + synthetic) and apply cleaning, deduplication, and quality filters.  
   - For classification or specialized tasks, check if synthetic data can efficiently augment limited human labels.

4. **Training strategy and infrastructure**  
   - Decide between full fine-tuning vs PEFT (LoRA/QLoRA) based on model size and budget.  
   - Choose training stack:  
     - Hugging Face **Transformers + Trainer** for general tasks.  
     - **TRL** (SFTTrainer, DPOTrainer, GRPO, etc.) for post-training and reasoning.  
     - **Unsloth** for optimized PEFT / quantized fine-tuning workflows.  
     - Accelerate, FSDP, DeepSpeed for multi-GPU or large models.  
   - Configure hyperparameters: learning rate, batch size, max sequence length, warmup, weight decay, number of steps/epochs.

5. **Experiment design and monitoring**  
   - Define baselines: base model, simple prompts, perhaps a smaller tuned model.  
   - Log metrics: training loss, evaluation loss, task-specific scores (accuracy, ROUGE, BLEU, etc.).  
   - Track experiments using tools like Weights & Biases, MLflow, or the HF Hub’s model versions and metadata.

6. **Evaluation and safety**  
   - Evaluate on:  
     - Task-specific benchmarks (classification, QA, code benchmarks like SWE-bench).  
     - General LLM benchmarks (e.g., HF leaderboards and associated datasets).  
     - Custom internal test sets representing your real use cases.  
   - Check safety: refusal behavior, harmful outputs, bias and fairness in sensitive domains.

7. **Deployment and inference optimization**  
   - Choose an inference strategy:  
     - Hosted endpoints (HF Inference Endpoints, other providers).  
     - Self-hosted servers running vLLM/text-generation-inference, Unsloth, or llama.cpp.  
   - Apply quantization and compilation: 4-bit/8-bit quantization, 1.58-bit schemes for extreme efficiency, model pruning, KV cache optimization.  
   - Integrate with applications (APIs, agents, web UIs).

8. **Monitoring, iteration, and editing**  
   - Monitor user feedback, error cases, and drift.  
   - Add new data (success and failure examples) to the fine-tuning dataset.  
   - Address model editing and retention: how long do edits persist? Do fine-tuned behaviors decay over time or across layers?  
   - Periodically re-run fine-tuning or targeted editing when requirements or base models change.

This loop is continuous: fine-tuning is not a single event but an **ongoing process** of aligning the model to evolving tasks and data.

---

## 3. Main families of fine-tuning methods

### 3.1 Supervised Fine-Tuning (SFT)

**SFT** is the most common starting point:

- Objective: minimize cross-entropy between the model’s output distribution and the desired target text.  
- Data: instruction/response pairs or chat transcripts labeled with correct answers, completions, or behaviors.  
- Use cases:  
  - Instruction-following models.  
  - Domain-specific chatbots.  
  - Code assistants and summarization models.  
  - “Teaching” a model specific formatting or tool-use patterns.

Implementation patterns:

- Use **Transformers Trainer** with causal language modeling (for decoder-only LLMs) and an appropriate data collator.  
- Use **TRL SFTTrainer** when you want a high-level SFT wrapper with better defaults for chat-style datasets and PEFT.

### 3.2 Parameter-Efficient Fine-Tuning (PEFT, LoRA, QLoRA)

Full fine-tuning is often too expensive for large models. **PEFT** methods update only a small fraction of parameters (or add small trainable modules) while keeping most weights frozen.

Key methods:

- **LoRA (Low-Rank Adaptation)** – inserts low-rank matrices into attention or MLP layers; only these new matrices are trained.  
- **QLoRA** – quantizes base weights (e.g., to 4-bit) and still trains low-rank adapters, enabling fine-tuning larger models on consumer GPUs.  
- Other PEFT variants: prefix-tuning, prompt-tuning, IA3, adapter layers, etc.

Benefits:

- Much lower VRAM and compute requirements.  
- Easy model distribution: only LoRA weights must be shared; users apply them on top of a base model.  
- Multiple adapters can be combined to support different tasks/domains with one base model.

The Hugging Face **PEFT library** provides unified APIs to apply these methods with Transformers and TRL.

### 3.3 Preference optimization and RLHF-style methods

After SFT, most high-quality assistants are further aligned using **preference optimization** (often grouped under the RLHF umbrella). The goal is to train models to **prefer helpful, harmless, and honest outputs** according to human or synthetic preferences.

Common methods:

- **PPO-style RLHF** – uses a reward model and reinforcement learning to optimize the policy model.  
- **DPO (Direct Preference Optimization)** – optimizes directly on preference pairs `(prompt, chosen, rejected)` using a loss derived from policy and reference log-probabilities.  
- **ORPO, KTO and related algorithms** – alternative preference objectives designed to simplify training or improve stability.  
- **GRPO** – group-based RL method used in Open R1-style reasoning training, where the model solves tasks in environments like OpenEnv.

Patterns:

- Start from an SFT model.  
- Build a dataset of preference pairs or rated outputs (human or synthetic).  
- Use TRL’s DPOTrainer, GRPOTrainer, or PPO-based trainers to optimize alignment objectives.  
- Carefully control KL-divergence to avoid drifting too far from the reference model and losing base capabilities.

### 3.4 Continued pre-training and domain adaptation

For some applications, you want the model to **ingest large amounts of domain text** (e.g., legal cases, scientific literature, internal knowledge) before SFT or alignment:

- Objective: language modeling (predicting next tokens) on domain-specific corpora.  
- Use cases:  
  - Domain adaptation (finance, medicine, law, enterprise docs).  
  - Language adaptation to low-resource languages.  
  - Style adaptation for writing or code formatting.

Typical workflow:

1. Start from a strong base model.  
2. Run continued pre-training on domain text with a causal language modeling objective and a moderate learning rate.  
3. Follow with SFT on domain-specific instructions, tasks, or QA pairs.  
4. Optionally apply preference optimization for safety and style.  
5. Evaluate both **domain tasks** and **general benchmarks** to watch for catastrophic forgetting.

Continued pre-training interacts strongly with **data quality and diversity**: poor or narrow corpora can hurt general performance.

---

## 4. Tooling in the Hugging Face and Unsloth ecosystems

### 4.1 Transformers + Trainer

The **Transformers** library provides:

- Model classes for major architectures (Llama, Mistral, Qwen, Phi, etc.).  
- Tokenizers and preprocessors.  
- The **Trainer** API for fine-tuning: handles training loops, gradient accumulation, mixed precision, logging, and checkpointing.

Useful patterns:

- `Trainer` or `Seq2SeqTrainer` for supervised tasks (classification, QA, summarization, translation).  
- Custom collators and `DataCollatorForLanguageModeling` for causal LM fine-tuning.  
- Integration with `datasets` for streaming and processing large corpora.

### 4.2 TRL: SFT, DPO, GRPO, GRLHF

**TRL (Transformers Reinforcement Learning)** builds on Transformers to offer post-training algorithms:

- **SFTTrainer** – high-level helper for chat SFT, supporting PEFT and HF Datasets.  
- **DPOTrainer, IPO, ORPO, KTO** – preference optimization algorithms based on pairwise comparisons or reward-like signals.  
- **GRPO and GRLHF examples** – used in the Open R1 ecosystem to fine-tune reasoning models via environment interactions.

TRL is central when moving beyond pure SFT and into alignment, reasoning, and RL.

### 4.3 PEFT library

The **PEFT** library provides unified interfaces to apply LoRA, QLoRA, and other PEFT methods to Transformers models:

- Wrap a base model with PEFT configuration (e.g., LoRA rank, target modules).  
- Automatically freeze base weights and only optimize PEFT parameters.  
- Integrate with Trainer and TRL trainers using a few lines of code.

This enables fine-tuning larger models on smaller GPUs and sharing compact adapter weights.

### 4.4 Unsloth

**Unsloth** focuses on **efficient, low-friction fine-tuning and RL**:

- Beginner-friendly **Fine-Tuning LLMs Guide** and **Datasets Guide**.  
- Pre-configured notebooks and Docker images that solve many environment and dependency issues.  
- Tutorials for running and fine-tuning popular open models (Qwen, DeepSeek, Gemma, gpt-oss, etc.).  
- Hardware-specific guidance (e.g., Blackwell / RTX 50 series GPUs) and quantized training recipes.

Unsloth is a good choice when you want to:

- Avoid manual configuration of PEFT and quantization.  
- Run full pipelines locally or on cloud GPUs with minimal setup.  
- Follow curated best practices for hyperparameters, batch sizes, and optimizers.

### 4.5 Courses, examples, and notebooks

A few particularly useful resources for hands-on learning:

- Hugging Face **LLM Course** – covers the full workflow from tokenization and pre-training to fine-tuning and reasoning.  
- Hugging Face blog tutorials on fine-tuning LLMs on Kaggle notebooks and other environments.  
- Community posts demonstrating fine-tuning large models (e.g., 14B parameters) on limited hardware like free Colab T4 with TRL and PEFT.

---

## 5. Data for fine-tuning: design, sources, and synthetic augmentation

### 5.1 Data types and schemas

Common data types for LLM fine-tuning:

- **Instruction or chat datasets**  
  - Schema: `{"instruction": ..., "input": ..., "output": ...}` or `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.  
  - Used for assistants and task-following models.

- **Code datasets**  
  - Schema: `{"prompt": ..., "code": ...}`, or chat-like messages with system/user/assistant roles.  
  - Used for coder models.

- **Classification and tagging**  
  - Schema: `{"text": ..., "label": ...}` with label mappings.  
  - Good for adapting encoders or LLM-based classifiers to new label sets.

- **Preference datasets**  
  - DPO-style: `{"prompt": ..., "chosen": ..., "rejected": ...}`.  
  - RLHF-style: `{"prompt": ..., "response": ..., "reward": ...}` or pairwise comparisons.

- **Domain corpora for continued pre-training**  
  - Simple: `{"text": ...}` or raw text files.

It is helpful to design schemas up front so they stay consistent across dataset versions and projects.

### 5.2 Synthetic data for fine-tuning

Synthetic data plays several roles in fine-tuning:

- **Instruction and chat SFT** – synthetic instructions and answers generated by strong teacher models.  
- **Classification augmentation** – synthetic labeled examples when human labels are scarce.  
- **Preference data** – synthetic comparisons obtained from LLM-as-a-judge.  
- **Domain QA** – turning proprietary documents into QA pairs using chunk-and-generate pipelines.

Key advice from research and practice:

- Mix synthetic data with smaller, high-quality human data when possible.  
- Anchor synthetic generations in real documents or prompts (via retrieval or templates) to avoid free-floating hallucinations.  
- Use filtering and LLM-as-judge to improve quality.

For more detail, cross-reference a dedicated knowledge base on synthetic datasets for LLMs.

### 5.3 Dataset creation and curation patterns

Useful patterns from real-world LLM projects:

- Start with open-source instruction datasets (e.g., UltraChat-style, domain-specific SFT sets) and then add **project-specific data**.  
- Use internal logs and user feedback (appropriately anonymized) to build realistic training and evaluation sets.  
- For reasoning tasks, build datasets that **reward full chain-of-thought reasoning** and also include failures to teach the model where it struggles.  
- For code and software engineering tasks, use benchmarks like **SWE-bench** and real code repositories to test the model’s editing and debugging capabilities.

---

## 6. Practical fine-tuning patterns by scenario

### 6.1 Small/medium model SFT with LoRA/QLoRA (single GPU)

Target: adapt a 7B–14B model for a domain-specific assistant or code helper using one GPU (e.g., 24–48 GB, or even 16 GB with QLoRA).

Recipe (high level):

1. Pick a base instruct or chat model (e.g., Llama-3.1-instruct-small, Mistral 7B Instruct, Qwen 2.5 or 3).  
2. Prepare an instruction or chat dataset in JSONL or HF Datasets format.  
3. Tokenize with the matching tokenizer and apply a data collator that builds chat-style sequences.  
4. Configure PEFT (LoRA/QLoRA) with appropriate rank and target modules.  
5. Use TRL SFTTrainer or Transformers Trainer with:  
   - Learning rate in the `1e-5`–`5e-5` range for LoRA (often smaller for larger models).  
   - Batch size determined by VRAM (use gradient accumulation).  
   - 1–3 epochs (or equivalent steps) as a starting point.  
6. Evaluate on held-out data and small internal benchmarks.  
7. Export and share the LoRA adapter; keep the base model separate.

### 6.2 Domain adaptation with continued pre-training + SFT

Target: make a model more fluent and accurate in a niche domain (e.g., legal, biotech).

Recipe:

1. Build a domain corpus (papers, docs, FAQs, transcripts).  
2. Run continued pre-training on this corpus with a causal language modeling objective and a moderate learning rate.  
3. Follow with SFT on domain-specific instructions, tasks, or QA pairs.  
4. Optionally apply preference optimization for safety and style.  
5. Evaluate both **domain tasks** and **general benchmarks** to watch for catastrophic forgetting.

### 6.3 Preference optimization with DPO/ORPO/KTO

Target: align an SFT model to preferred behaviors, such as being more helpful, safe, or reasoning-focused.

Recipe:

1. Collect prompts and associated candidate responses (from your SFT model and/or other models).  
2. Obtain preferences:  
   - Human annotators choose the best response.  
   - Synthetic judges (LLM-as-a-judge) provide rankings or scores.  
3. Build a preference dataset (pairs `(prompt, chosen, rejected)` or other schemas).  
4. Use TRL DPOTrainer or other preference trainers, with:  
   - A reference model (often the SFT model or the base model).  
   - A KL control parameter to limit divergence.  
5. Evaluate on alignment and safety benchmarks, plus human evaluation where possible.

### 6.4 Reasoning fine-tuning with GRPO and environments

Target: improve reasoning and problem-solving ability (math, coding, multi-step tasks).

Patterns from Open R1 and GRPO-style training:

1. Use environments like **OpenEnv**: games, tasks, or custom environments.  
2. Collect trajectories: prompts, reasoning steps, actions, rewards.  
3. Train with GRPO or related RL algorithms to optimize long-horizon rewards.  
4. Combine with supervised data that encourages explicit reasoning (chain-of-thought, tree-of-thought).  
5. Evaluate on reasoning benchmarks and environment-specific tasks.

Reasoning training is hardware- and data-intensive but can yield dramatic gains on structured tasks.

---

## 7. Risks, limitations, and common pitfalls

### 7.1 Catastrophic forgetting and edit decay

Fine-tuning can cause models to:

- Forget general capabilities or languages not present in the fine-tuning data.  
- Overfit to a narrow domain or style.  
- Exhibit “edit decay,” where local fine-tuning edits fade as prompts move away from training examples.

Mitigations:

- Mix in a small amount of general-purpose data or keep KL divergence to a reference model under control.  
- Use regular evaluation on broad benchmarks and tasks.  
- Use localized editing methods or low learning rates for delicate changes.

### 7.2 Overfitting and data contamination

Overfitting is particularly risky when:

- The fine-tuning dataset is small and repetitive.  
- The model is large relative to your data size.  
- Evaluation sets inadvertently overlap with training data (data leakage).

Mitigations:

- Use validation splits and early stopping.  
- Deduplicate at the document and sample levels.  
- Carefully track data provenance and avoid using benchmarks as training data.

### 7.3 Safety, bias, and misuse

Fine-tuning can introduce new safety risks even if the base model was moderated:

- Domain-specific data may contain harmful patterns or biased language.  
- Synthetic data may encode subtle biases from teacher models or judges.  
- Preference optimization may inadvertently reward unsafe shortcuts.

Mitigations:

- Include safety-specific training and evaluation sets.  
- Use constrained decoding (e.g., content filters) at inference time.  
- Audit datasets and model outputs for high-risk use cases.

### 7.4 Hardware and cost pitfalls

Common issues include:

- Underestimating VRAM requirements, especially for long context or full fine-tuning.  
- Ignoring I/O and CPU bottlenecks, causing poor GPU utilization.  
- Training for too many epochs or steps, wasting compute with minimal gains.

Mitigations:

- Start with PEFT and modest context lengths.  
- Benchmark small subsets and scale gradually.  
- Use cost-effective environments (Kaggle, Colab, HF Spaces Jobs, etc.) for prototyping.

---

## 8. Checklists and quick-start recipes

### 8.1 Checklist: before you fine-tune

1. **Clarify goal** – What user experience do you want? Chatbot, code assistant, domain expert, reasoning agent?  
2. **Choose base model** – Match size and specialization to your hardware and task.  
3. **Inspect data** – Do you have enough data? Is it high quality, safe, and representative?  
4. **Select method** – SFT only, SFT + PEFT, preference optimization, continued pre-training, or a combination.  
5. **Plan evaluation** – At least one automatic metric and one human-like evaluation method.  
6. **Plan deployment** – Where will this model live? What latency and cost constraints apply?

### 8.2 Quick-start: SFT with Transformers + PEFT

1. Load a base model and tokenizer from the Hugging Face Hub.  
2. Load your dataset with `datasets` (JSONL/Parquet/HF dataset).  
3. Preprocess to build `(input_ids, labels)` for causal LM.  
4. Wrap model with LoRA/QLoRA using PEFT.  
5. Configure `TrainingArguments` and instantiate `Trainer`.  
6. Train, evaluate, and push the adapter to the Hub.

### 8.3 Quick-start: Chat SFT with TRL SFTTrainer

1. Prepare a dataset of chat messages (`{"messages": [{"role": "user" ...}, ...]}`).  
2. Use `SFTTrainer` with your base model and PEFT config.  
3. Specify packing/sequence length, batch size, and logging.  
4. Train and evaluate on held-out chat interactions.  
5. Save and reuse the fine-tuned model in inference pipelines.

### 8.4 Quick-start: Preference optimization with DPO

1. Start from an SFT model and dataset of `(prompt, chosen, rejected)` pairs.  
2. Use TRL `DPOTrainer` with a reference model and appropriate beta (KL weight).  
3. Train for a modest number of steps, monitoring losses and example outputs.  
4. Evaluate on safety and helpfulness metrics.  
5. Use the aligned model in your application, keeping the reference model for rollback.

---

## 9. References and curated links

This section lists representative, non-exhaustive resources you can consult for deeper dives. Many more links exist in your local link collections.

### 9.1 Official and semi-official docs

- Hugging Face Transformers training and fine-tuning guides:  
  - [Fine-tuning with Trainer](https://huggingface.co/docs/transformers/training)  
  - [Fine-tuning on custom datasets](https://huggingface.co/docs/transformers/v4.17.0/custom_datasets)  
- Hugging Face LLM Course (includes fine-tuning chapters):  
  - [LLM Course index](https://huggingface.co/learn/llm-course)  
  - [Training a causal LM from scratch](https://huggingface.co/learn/llm-course/chapter7/6)  
- Hugging Face PEFT library:  
  - [PEFT documentation](https://huggingface.co/docs/peft/index)  
- TRL documentation:  
  - [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)  
  - [TRL index](https://huggingface.co/docs/trl/index)  
- Unsloth docs (fine-tuning, datasets, model choice, tutorials):  
  - [Fine-Tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)  
  - [Datasets Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide)  
  - [What Model Should I Use?](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use)  
  - [Tutorials: How to Fine-tune & Run LLMs](https://docs.unsloth.ai/models/tutorials-how-to-fine-tune-and-run-llms)  
  - [Fine-tune LLMs with Unsloth & Docker](https://docs.unsloth.ai/new/how-to-fine-tune-llms-with-unsloth-and-docker)

### 9.2 Hugging Face blogs and posts

- Fine-tuning walkthroughs:  
  - [How to train a new language model from scratch](https://huggingface.co/blog/how-to-train)  
  - [Fine-tuning LLMs on Kaggle notebooks](https://huggingface.co/blog/lmassaron/fine-tuning-llms-on-kaggle-notebooks)  
  - [Fine-tuning Falcon 7B with DeepSpeed](https://huggingface.co/blog/Neo111x/falcon-7b-finetuning-deepspeed)  
- Reasoning-focused fine-tuning:  
  - [Open R1 update (GRPO reasoning)](https://huggingface.co/blog/open-r1/update-3)  
  - [Making any model better at reasoning](https://huggingface.co/blog/Metal3d/making-any-model-reasoning)  
- Quantization and extreme efficiency:  
  - [Fine-tuning LLMs to 1.58 bits](https://huggingface.co/blog/1_58_llm_extreme_quantization)

### 9.3 Research papers and surveys

- Pipeline and best practices:  
  - [The Ultimate Guide to Fine-Tuning LLMs](https://arxiv.org/abs/2408.13296)  
- Alignment and preference optimization:  
  - Kseniase’s overview of preference algorithms (community post):  
    [Ten recent preference optimization algorithms](https://huggingface.co/posts/Kseniase/304021452230579)  
- Fine-tuning behavior and editing:  
  - Papers on edit decay and catastrophic forgetting (e.g., *Quantifying Edits Decay in Fine-tuned LLMs*).

### 9.4 Evaluation and leaderboards

- Hugging Face leaderboards and benchmarks:  
  - [Leaderboards documentation](https://huggingface.co/docs/leaderboards/index)  
  - [The Big Benchmarks Collection](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a)  
- SWE-bench and code evaluation:  
  - [SWE-bench Bash-only](https://www.swebench.com/bash-only.html)  
  - [Aider leaderboards](https://aider.chat/docs/leaderboards/)

### 9.5 Reasoning courses and environments

- Hugging Face Reasoning Course and related resources:  
  - [Reasoning Course](https://huggingface.co/reasoning-course)  
  - [OpenEnv and GRPO environments post](https://huggingface.co/posts/sergiopaniego/527159223589050)  
  - [GRPO deep-dive forum thread](https://discuss.huggingface.co/t/offering-a-technical-deep-dive-on-grpo-dapo-dr-grpo-algorithms/154480)

### 9.6 Hardware, deployment, and misc

- Hardware selection and VRAM–model-size rules of thumb:  
  - [Hardware recommendations for LLM inference and fine-tuning](https://huggingface.co/posts/mitkox/956503968449347)  
- Model formats and deployment:  
  - [Common AI model formats](https://huggingface.co/blog/ngxson/common-ai-model-formats)  
  - [Ollama integration with the Hub](https://huggingface.co/docs/hub/ollama)  
- Running models locally:  
  - [llama.cpp Python bindings](https://github.com/abetlen/llama-cpp-python)

This document is designed to be a **standalone knowledge base**. You can adapt and extend it with project-specific notes (data schemas, hyperparameters, evaluation sets) to form your own fine-tuning playbook.

