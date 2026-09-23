---
source: "huggingface+chat+web"
topic: "Muon optimizer, Hugging Face ecosystem, and Moonshot/Kimi-K2 models"
generated_by: "LLM assistant (web + user-provided URLs)"
generated_at: "2025-11-14T05:42:21Z"
---

# Muon optimizer, Hugging Face ecosystem, and Moonshot/Kimi-K2 models

## 1. Background and high-level picture

Muon is a relatively new optimizer that targets **hidden-layer weight matrices** in neural networks rather than individual scalar parameters. It was proposed by Keller Jordan in late 2024 as an alternative to AdamW that:

- Treats each **2D parameter tensor as a matrix** (a linear transformation) instead of a bag of independent scalars.
- Uses **SGD with momentum** to build a matrix-valued update.
- Then **orthogonalizes** that update matrix using a fast **Newton–Schulz iteration**, so that all singular directions are balanced.
- Stores only **two copies** of the parameters for these matrices (weights + momentum), instead of three copies (weights + first moment + second moment) as in AdamW.

The core idea can be summarized twice, from two complementary angles:

1. **Geometry-aware angle.**  
   - Muon treats each weight matrix as a whole linear map and rebalances the directions in which it moves.  
   - The Newton–Schulz step forces the update matrix to have singular values close to 1, so that no direction dominates and all directions are explored more evenly.

2. **Systems / memory angle.**  
   - For each matrix, AdamW keeps parameters, first moment, and second moment. Muon keeps only parameters + momentum.  
   - For large transformer-style models dominated by big matrices, this reduces optimizer state memory by about **33%** at the cost of roughly **1% extra FLOPs**.

Muon is not a full replacement for AdamW. Instead, practical implementations apply Muon only to matrices in the hidden layers and still use AdamW (or a similar optimizer) for embeddings, heads, biases, and layernorm parameters.

Moonshot AI’s **Moonlight** models and the **Kimi** family of thinking/agentic models (including **Kimi-K2-Thinking**) sit on top of this ecosystem: Moonshot’s technical report “Muon is Scalable for LLM Training” shows that Muon can be used to train very large models efficiently, and K2-Thinking is a 1T-parameter MoE “thinking” model from the same line of work.

This document focuses on:

- What Muon is and how it works.
- How Hugging Face resources explain and implement it.
- How Moonshot uses Muon for large-scale LLM training (Moonlight, Kimi/K2 line).
- Practical guidance and pitfalls when adopting Muon in your own projects.

---

## 2. From official-style sources: writeups, papers, and blogs

### 2.1 Keller Jordan’s original Muon writeup

Keller Jordan’s long-form post “Muon: An optimizer for hidden layers in neural networks” gives the canonical definition of Muon and its motivation.

Key points from that writeup:

- **Scope.** Muon is an optimizer specifically for **hidden-layer weight matrices** in neural networks such as transformers and convnets. It is not intended for scalar or 1D parameters.  
- **Algorithm sketch.** For each matrix weight \(W\):  
  1. Compute the gradient \(G_t\).  
  2. Update a momentum matrix \(M_t = \beta M_{t-1} + (1 - \beta) G_t\).  
  3. Apply an **orthogonalization step** using Newton–Schulz iterations, obtaining an approximately orthogonal matrix \(\tilde{M}_t\) whose singular values are near 1.  
  4. Use \(\tilde{M}_t\) as the direction of the parameter update, with an appropriate scaling and learning rate.
- **Performance claims.** The post reports:  
  - Speed records for **CIFAR-10** speedrunning.  
  - Better wall-clock to a target validation loss in **NanoGPT-like language models**.  
  - Competitive or better performance than tuned AdamW at similar or lower compute budgets.
- **Implementation details.**  
  - Newton–Schulz is implemented via a small odd polynomial (e.g., quintic) applied iteratively to \(X X^\top\).  
  - The orthogonalization is run in bfloat16 for speed, with a small fixed number of iterations (often 5).  
  - Convolution kernels are flattened into matrices before applying Muon, while 1D/bias parameters stay in a non-Muon optimizer.

This writeup is the conceptual starting point for almost all later work on Muon.

### 2.2 Moonshot’s “Muon is Scalable for LLM Training” (Moonlight paper)

Moonshot AI’s technical report “Muon is Scalable for LLM Training” and the associated **Moonlight** code base show that Muon scales beyond small benchmarks into full-scale LLM training.

Key contributions of this work:

- **Scaling to large LLMs.** The paper shows that Muon can train multi-billion and trillion-parameter models without losing stability, matching or beating AdamW in both loss and downstream tasks.  
- **Two critical techniques.**  
  1. **Weight decay:** decoupled weight decay is emphasized as crucial for stability and generalization in very large models.  
  2. **Per-parameter update scaling / RMS matching:** the paper introduces a strategy to keep root-mean-square (RMS) update norms consistent with AdamW baselines, simplifying hyperparameter transfer.
- **Compute efficiency.** Scaling-law experiments suggest that Muon can achieve up to about **2× compute efficiency** over AdamW in certain LLM training setups, if you measure tokens to reach a given loss under compute-optimal schedules.
- **Practical implementation.** Moonshot open-sources a **distributed Muon implementation** that is:  
  - Memory optimal (keeping only the necessary per-matrix momentum state).  
  - Communication efficient in FSDP/ZeRO settings.  
  - Integrated into their Moonlight training stack.

This paper is the main justification that Muon is not only a “fun small-model optimizer,” but a **serious candidate for frontier-scale LLM training**.

### 2.3 Hugging Face community article: “Muon Optimizer: The Power of Collective Momentum”

Yi Cui’s community article on Hugging Face, “Muon Optimizer: The Power of Collective Momentum”, is the best high-level exposition oriented toward practitioners.

Highlights:

- **Framing.**  
  - Adam’s memory issue is framed as “three copies per parameter” becoming dominant as model sizes hit tens or hundreds of billions of parameters.  
  - Muon is introduced as a way to **switch from element-wise accounting to group-level dynamics** at the matrix level.

- **Before / after orthogonalization.**  
  - The article uses a simple 2×2 example to show how orthogonalization takes a badly conditioned matrix (one long vector, one short) and turns it into orthogonal unit vectors.  
  - Intuitively, this:  
    - Downscales directions that dominate momentum.  
    - Upscales directions that have been under-explored.  
    - Leads to more balanced exploration of the parameter space.

- **Memory vs compute tradeoff.**  
  - For a 7B model, the article estimates that:  
    - AdamW’s optimizer states require approximately **84 GB** (3 copies).  
    - Muon’s require approximately **56 GB** (2 copies).  
    - This is a **33% reduction in optimizer memory** for matrix parameters.  
  - The cost is approximately **1% extra FLOPs**, mainly from a small number of matrix multiplications per layer for Newton–Schulz.

- **Implementation rules.**  
  - Muon is applied only to **2D matrices** (transformer linear layers).  
  - Embeddings, output heads, biases, and layernorm parameters still use AdamW or similar.  
  - In distributed settings, each pipeline stage or shard runs Muon on its local matrices, so the orthogonalization is local to each GPU.

This article is particularly useful for explaining Muon to engineers and for motivating why it is a good fit for transformer-heavy architectures.

### 2.4 Supporting tutorials and datasets

Multiple Hugging Face forum posts and datasets expand on Muon:

- **Tutorial threads.**  
  - “[Tutorial] Understanding and Implementing the Muon Optimizer” and “First instalment the Muon Optimizer tutorial series” walk through Muon’s intuition, pseudocode, and PyTorch implementation, including how to split parameters into Muon and non-Muon groups.  
  - A later “Muon replication journey” series explains distributed optimizers, ZeRO/FSDP, and how Muon fits into that picture.

- **Tutorial dataset.**  
  - The `bird-of-paradise/muon-tutorial` dataset hosts a structured Muon tutorial (notebook-like text) summarizing Muon’s origins, math, and implementation, ideal for study or for building educational material.

Together with the official writeup and Moonshot paper, these resources form a coherent ecosystem of **concept → math → code → distributed training** for Muon.

---

## 3. From model cards and Moonshot’s Kimi / Moonlight line

### 3.1 Kimi-K2-Thinking model card (Moonshot AI)

The Hugging Face model card for **`moonshotai/Kimi-K2-Thinking`** describes K2 Thinking as Moonshot’s latest open-source “thinking model”: a Mixture-of-Experts LLM optimized for long-horizon reasoning and tool use.

Key information from the card:

- **Model scale and architecture.**  
  - Total parameters: **1T**.  
  - Activated parameters per token: **32B**.  
  - Mixture-of-Experts architecture with **384 experts**, **8 experts selected per token**, plus shared experts.  
  - 61 layers (including a dense layer), 7168 attention hidden dimension, 2048 MoE hidden dimension per expert.

- **Capabilities and design goals.**  
  - Built as a **thinking agent** that performs multi-step reasoning and tool calls over hundreds of steps (200–300 sequential tool calls in benchmarks).  
  - Focused on agentic tasks: search, coding, data analysis, and long-form reasoning with a 256k context window.  
  - Exposes native tool-calling and agent frameworks in the usage examples.

- **Quantization.**  
  - Uses **native INT4 quantization** with Quantization-Aware Training (QAT) applied post-training to the MoE components.  
  - Checkpoints are saved in `compressed-tensors` format, which can be unpacked to higher-precision formats (e.g., FP8, BF16).

- **Benchmarks.**  
  - Reports strong performance on reasoning benchmarks such as HLE, AIME, HMMT, IMO-AnswerBench, and on coding/agent benchmarks like SWE-bench, Terminal-Bench, BrowseComp, etc., often competitive with or surpassing GPT-5 and other frontier models.

The card itself does not spell out the optimizer used during pretraining, but it lives in the same Moonshot ecosystem where **Moonlight models and Kimi-2** are explicitly associated with Muon in technical reports and community posts. In practice, K2-Thinking should be read as a **successor model built on top of the Muon-powered training stack** introduced by Moonlight, even if the card does not repeat all optimizer details.

### 3.2 Moonlight-16B and related model cards

Moonshot’s **Moonlight-16B** and related models on Hugging Face provide an interface to the models introduced in the “Muon is Scalable for LLM Training” report.

From those cards and abstracts you can extract that:

- Moonlight models are **Mixture-of-Experts LLMs** (e.g., 3B/16B-parameter configurations) trained with Muon.  
- The abstracts reiterate that scaling Muon required:  
  - Strong **decoupled weight decay**, and  
  - A **consistent RMS update scheme** (often called “consistent RMS updates” in the abstract) so that step magnitudes across parameters remain well-behaved as width and depth grow.

These models serve as concrete **“Muonic” LLMs** whose weights are available on Hugging Face, making them natural baselines or references if you want to see Muon’s impact on real LLMs.

### 3.3 Relation between Muon, Moonlight, and Kimi-K2-Thinking

Putting the pieces together:

- The **Muon optimizer** provides a way to reduce optimizer memory and improve training efficiency for large transformer-like models by operating on hidden-layer matrices.  
- **Moonlight** demonstrates that Muon can be scaled to multi-billion-parameter MoE LLMs and that, with proper weight decay and RMS scaling, it can be significantly more compute-efficient than AdamW on LLM pretraining workloads.  
- **Kimi-K2-Thinking** is a **1T-parameter MoE “thinking model”** in this same family, purpose-built for deep reasoning and tool use.  
- While the K2 Thinking model card focuses on architecture, benchmarks, and deployment, it should be viewed as a **downstream product of the Muon-based training stack** that Moonshot describes in the Moonlight paper and associated repos.

For someone investigating Muon, these model cards provide real-world anchor points showing **what kinds of models Muon can train** and **what performance levels are achievable**.

---

## 4. From community, forums, and GitHub

### 4.1 Hugging Face forum tutorials and study groups

Several forum threads deepen the practical understanding of Muon:

- **Step-by-step tutorial series.**  
  - Threads like “First instalment the Muon Optimizer tutorial series” and “[Tutorial] Understanding and Implementing the Muon Optimizer” walk through:  
    - Intuition and motivation (why group momentum and orthogonalization).  
    - Pseudocode based on Keller Jordan’s definitions.  
    - A clean PyTorch implementation with parameter grouping.  
    - Comparisons of loss curves vs AdamW on medium-sized language models.

- **Distributed Muon study group.**  
  - The “Implementing a Scalable, FSDP-Compatible Muon Optimizer” study group focuses on replicating and simplifying Moonshot’s distributed Muon implementation.  
  - Goals include:  
    - Understanding how FSDP/ZeRO sharding interacts with matrix-level orthogonalization.  
    - Designing communication patterns that keep Newton–Schulz efficient at scale.  
    - Producing an open, FSDP-compatible Muon implementation suitable for Hugging Face Trainer, Accelerate, or custom loops.

- **Distributed glossary / reverse-engineering series.**  
  - A later “Muon replication journey” thread acts as a kind of no-nonsense glossary of distributed optimizer concepts, documenting aha-moments around:  
    - Sharding optimizer states vs parameters.  
    - How Muon’s 2-copy vs 3-copy design interacts with ZeRO stage choices.  
    - Practical debugging tips when loss or memory usage does not match expectations.

These threads collectively translate the Moonshot and Keller Jordan materials into **hands-on engineering guidance** for the HF ecosystem.

### 4.2 GitHub: KellerJordan/Muon and issues

The `KellerJordan/Muon` GitHub repo is the reference PyTorch implementation. It provides:

- A `MuonWithAuxAdam` class that:  
  - Applies Muon to 2D hidden weights.  
  - Applies AdamW to all remaining parameters in an auxiliary fashion.
- Example parameter-group configs for common architectures (e.g., NanoGPT variants, vision transformers).  
- Default hyperparameters such as:  
  - `momentum ≈ 0.95`  
  - `nesterov = True`  
  - `ns_steps = 5` for Newton–Schulz

The issues section is also informative:

- Some users report **no observable GPU memory savings** under specific mixed-precision or accelerator setups, indicating that your framework’s optimizer and offloading logic must actually store states as intended for Muon’s benefits to materialize.  
- Others ask about **learning-rate ratios** between Muon and AdamW, reinforcing the pattern that Muon’s LR for matrices is often much larger than AdamW’s LR for non-matrix parameters.

Reading the issues is helpful for spotting **real-world edge cases and gotchas** beyond the idealized benchmarks.

### 4.3 Other community resources

Additional resources include:

- Short blog posts and notes (e.g., in Japanese) summarizing Muon’s characteristic as “SGD with momentum followed by Newton–Schulz-based orthogonalization” and explaining why it only applies to rank-2-or-higher tensors.  
- Reddit and HF Papers entries that summarize the Moonlight paper’s main findings, such as “about half the FLOPs compared to AdamW” for some 1.5B LLM benchmarks at a specific token count or loss target.

These provide sanity checks and multiple independent perspectives on Muon’s properties.

---

## 5. Implementation patterns and tips

### 5.1 Parameter grouping: what runs under Muon vs AdamW

Virtually all serious implementations agree on a strict parameter grouping rule:

- **Muon group (2D matrices):**  
  - Transformer attention projection matrices (Q, K, V, and output).  
  - MLP / feed-forward weight matrices.  
  - Other internal 2D linear mappings.  
  - Convolution filters, once reshaped into matrices.

- **Non-Muon group (AdamW or similar):**  
  - Token embeddings.  
  - Output / LM heads.  
  - Biases and scalar parameters.  
  - LayerNorm and RMSNorm parameters (1D).

Example pattern in PyTorch pseudocode:

```python
muon_params = []
adamw_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 2 and "embed" not in name and "lm_head" not in name:
        muon_params.append(p)   # hidden matrices
    else:
        adamw_params.append(p)  # embeddings, heads, norms, biases
```

Getting this grouping right is critical. If you accidentally:

- Put 1D parameters into the Muon group, the orthogonalization assumptions can break.  
- Put large matrices into AdamW, you lose much of Muon’s intended memory/computation structure.

### 5.2 Example: MuonWithAuxAdam-style configuration

A typical combined optimizer configuration inspired by the KellerJordan repo looks like:

```python
# pip install git+https://github.com/KellerJordan/Muon  # see repo for latest
from muon import MuonWithAuxAdam

muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
aux_params  = (
    [p for p in model.body.parameters() if p.ndim < 2]
    + list(model.embed.parameters())
    + list(model.head.parameters())
)

param_groups = [
    dict(
        params=muon_params,
        use_muon=True,
        lr=0.02,          # Muon LR, often much larger
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ),
    dict(
        params=aux_params,
        use_muon=False,
        lr=3e-4,          # AdamW LR for non-matrix params
        betas=(0.9, 0.95),
        weight_decay=0.01,
    ),
]

optimizer = MuonWithAuxAdam(param_groups)
```

Pattern to note:

- All Muon-specific options (`momentum`, `nesterov`, `ns_steps`) live in the Muon param group.  
- AdamW-specific options (`betas`) live in the non-Muon group.  
- Learning-rate **ratios** between the two groups matter; Muon LR is typically much higher than AdamW LR.

### 5.3 Hyperparameter guidance from papers and tutorials

Pulling together recommendations from Keller Jordan, Moonshot, and HF tutorials:

- **Momentum and NS steps.**  
  - `momentum ≈ 0.95` with Nesterov is a robust default.  
  - `ns_steps ≈ 5` is a good compromise between accuracy and overhead.

- **Learning rate.**  
  - Muon’s LR can be **orders of magnitude larger** than AdamW’s LR for the same model (e.g., 0.02 vs 3e-4).  
  - Some groups use **muP-style scaling** so that Muon learning rates transfer more easily across widths.

- **Weight decay.**  
  - Decoupled weight decay (à la AdamW) is important, especially at scale.  
  - Moonshot emphasizes weight decay as one of the two crucial ingredients for scaling Muon to very large models.

- **Scale modes and RMS matching.**  
  - Emerging implementations (e.g., NVIDIA’s Emerging Optimizers, Moonshot’s distributed Muon) provide options for:  
    - **Shape-based scaling**: scale updates based on matrix dimensions.  
    - **Spectral scaling**: scale based on largest singular values or related spectral properties.  
    - **Unit RMS norm scaling**: scale updates so that their RMS norm matches a target value, often aligned with an AdamW baseline.

When migrating a production LLM from AdamW to Muon, it is common to:

1. Start from a well-tuned AdamW config.  
2. Swap hidden matrices to Muon with RMS-matched scaling.  
3. Transfer or lightly adjust learning rates and weight decay as recommended by the Moonlight work.

### 5.4 Distributed training patterns (FSDP, ZeRO)

Muon’s orthogonalization is naturally matrix-local, but distributed training introduces complexity:

- **Parameter sharding.**  
  - In FSDP/ZeRO setups, weight matrices may be sharded across devices. Implementations must decide whether to:  
    - Orthogonalize only local shards and rely on symmetry, or  
    - Communicate shards to reconstruct full matrices (more expensive, but closer to the conceptual Muon update).

- **Optimizer-state sharding.**  
  - Since Muon keeps fewer optimizer states, there is potential for extra memory savings in sharded setups.  
  - However, you must ensure that your framework’s ZeRO/FSDP logic does not internally create redundant buffers that negate these savings.

- **Study-group objectives.**  
  - HF study groups are currently working toward **FSDP-compatible Muon** that:  
    - Preserves theoretical properties.  
    - Stays communication efficient.  
    - Plays nicely with HF Accelerate and Trainer abstractions.

For now, if you are not working on the bleeding edge of distributed systems, a pragmatic approach is:

- Start with single-node or modest multi-GPU runs, using a reference implementation.  
- Move carefully into FSDP/ZeRO, re-validating memory usage and loss curves at each step.

### 5.5 Using Muon in the context of Kimi / Moonshot-style models

If your goal is to train Kimi-like or Moonlight-like models (large MoE LLMs with long context and agentic tool use), Muon fits into the stack roughly as follows:

- **Architectures.**  
  - Multi-expert transformers with large projection matrices are prime candidates for Muon’s matrix-wise updates.  
  - You will still rely on standard optimizers for embeddings, output heads, and normalization parameters.

- **Agentic / tool-using behavior.**  
  - Muon itself does not “know” about tools or agents, but better optimization and stability at high depth and long context make it easier to train models that support hundreds of reasoning/tool steps without collapse.

- **Quantization-friendly training.**  
  - Kimi-K2-Thinking’s QAT + INT4 pipeline shows that Muon-trained models can still be post-trained and quantized aggressively, as long as quantization-aware methods are used.

This is more about **matching the training infrastructure** of the Kimi/Moonlight line than about Muon changing the agentic design directly.

---

## 6. Limitations, pitfalls, and open questions

Despite its promise, Muon has several practical caveats.

### 6.1 Implementation and integration complexity

Compared to AdamW, Muon is more complex to integrate because:

- You must implement a correct and efficient **Newton–Schulz kernel**.  
- Parameter grouping must be carefully curated to apply Muon only where it is mathematically intended.  
- Distributed configurations require non-trivial engineering for correctness and performance.

Poor implementations can silently degrade performance, so starting from vetted codebases (KellerJordan/Muon, Moonlight, NVIDIA Emerging Optimizers) is strongly recommended.

### 6.2 Real vs theoretical memory savings

Some users report that in real training runs, especially with certain accelerator or mixed-precision setups, they do **not** see the expected 33% optimizer memory reduction.

Possible reasons include:

- Frameworks keeping internal buffers or extra optimizer states for bookkeeping.  
- Offloading strategies that move states to CPU or NVMe, masking differences in GPU memory.  
- Mis-grouped parameters leading to larger-than-necessary AdamW/auxiliary groups.

Verifying memory savings yourself requires careful instrumentation of **actual optimizer state allocations**, not just model parameter sizes.

### 6.3 Numerical stability and coefficient choices

Muon’s orthogonalization uses polynomial iterations whose behavior depends on:

- Coefficient values \((a, b, c)\).  
- Number of iterations.  
- Normalization strategy (e.g., dividing by Frobenius norm).  
- Precision (bfloat16 vs float16 vs float32).

Most public implementations ship with robust defaults, but if you alter these, you should:

- Monitor singular-value distributions (if feasible for small models).  
- Check for exploding gradients or update norms.  
- Compare against a known-good configuration before scaling up.

### 6.4 Domain coverage and generalization

Public evidence for Muon is strongest on:

- Language modeling (NanoGPT-style, LLM pretraining).  
- Some computer vision tasks (e.g., CIFAR-10).  
- MoE LLMs (Moonlight, Kimi family).

Less is known empirically about:

- Large-scale multimodal models beyond those in the Moonshot ecosystem.  
- Fine-tuning regimes like instruction tuning, RLHF, or domain adaptation on smaller datasets.  
- Non-transformer architectures such as state-space models, although community experimentation is ongoing.

You should treat Muon as **well supported but still evolving**; in some settings, a well-tuned AdamW or Shampoo baseline may still be preferable.

### 6.5 Tooling and ecosystem maturity

Muon is newer than AdamW, and ecosystem support is still catching up:

- Framework-native implementations are emerging but not as standardized.  
- Debugging tools, logging templates, and best-practice configs are mostly community-driven.  
- Some widely used training frameworks may not yet expose Muon as a first-class optimizer option.

For production settings, this means you need a certain tolerance for **lower-level optimization work**.

---

## 7. Practical checklists for using Muon

### 7.1 Minimal checklist for a first Muon experiment

1. **Start from a strong AdamW baseline.**  
   - Same architecture, same dataset, same batch size.

2. **Integrate a known-good Muon implementation.**  
   - For example, `MuonWithAuxAdam` from `KellerJordan/Muon` or a Muon module from a trusted library.

3. **Define parameter groups carefully.**  
   - Double-check that only 2D hidden-layer weights are included in the Muon group.

4. **Adopt recommended defaults.**  
   - `momentum ≈ 0.95`, `nesterov = True`, `ns_steps ≈ 5`, decoupled weight decay ≈ 0.01.

5. **Set learning rates.**  
   - Use a higher LR for Muon matrices and keep AdamW LR in a typical range.  
   - Optionally, match RMS update norms to an AdamW run.

6. **Monitor training closely.**  
   - Compare loss curves, update norms, and GPU memory usage against the AdamW baseline.  
   - Verify that training is stable and that memory savings are real.

### 7.2 Checklist for a Moonshot/Kimi-like large-scale setup

1. **Architecture design.**  
   - Design a MoE or dense transformer architecture whose hidden layers are dominated by large matrices.  
   - Decide context length, expert counts, and activation dimensions.

2. **Optimizer design.**  
   - Use Muon for hidden-layer matrices; AdamW (or similar) for embeddings, heads, and norms.  
   - Integrate weight decay and RMS update scaling as per Moonlight-like configurations.

3. **Distributed training.**  
   - Choose FSDP/ZeRO settings compatible with Muon’s matrix-wise orthogonalization.  
   - Validate that optimizer state and communication patterns behave as expected.

4. **Post-training adaptation.**  
   - For thinking/agentic models, add instruction tuning and tool-use training on top of the pretraining run.  
   - If needed, integrate QAT or other quantization schemes (as Kimi-K2-Thinking does with INT4).

5. **Evaluation and iteration.**  
   - Evaluate on reasoning, coding, and agentic benchmarks.  
   - Iterate on scale, data mix, and optimizer hyperparameters to refine performance.

This mirrors the path taken by Moonshot: from Muon concept → Moonlight LLMs → Kimi/K2 “thinking” models.

---

## 8. References and links

### 8.1 Core Muon resources

- Muon original writeup (Keller Jordan):  
  [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
- KellerJordan/Muon GitHub repository:  
  [https://github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)
- Muon Optimizer: The Power of Collective Momentum (Hugging Face community article):  
  [https://huggingface.co/blog/onekq/muon-optimizer](https://huggingface.co/blog/onekq/muon-optimizer)
- Muon is Scalable for LLM Training (arXiv):  
  [https://arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)
- Muon is Scalable for LLM Training (GitHub / Moonlight):  
  [https://github.com/MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)

### 8.2 Hugging Face ecosystem

- Kimi-K2-Thinking model card (Moonshot AI):  
  [https://huggingface.co/moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- Moonlight-16B model card:  
  [https://huggingface.co/moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
- Muon tutorials dataset:  
  [https://huggingface.co/datasets/bird-of-paradise/muon-tutorial](https://huggingface.co/datasets/bird-of-paradise/muon-tutorial)
- HF Papers search for Muon:  
  [https://huggingface.co/papers?q=Muon+optimizer](https://huggingface.co/papers?q=Muon+optimizer)

### 8.3 Hugging Face forum discussions

- [Tutorial] Understanding and Implementing the Muon Optimizer:  
  https://discuss.huggingface.co/t/tutorial-understanding-and-implementing-the-muon-optimizer/167717
- First instalment the Muon Optimizer tutorial series:  
  https://discuss.huggingface.co/t/first-instalment-the-muon-optimizer-tutorial-series/167227
- Implementing a Scalable, FSDP-Compatible Muon Optimizer (study group):  
  https://discuss.huggingface.co/t/study-group-implementing-a-scalable-fsdp-compatible-muon-optimizer/168626
- Muon replication journey / distributed glossary:  
  https://discuss.huggingface.co/t/my-muon-replication-journey-from-distributed-optimizers-to-a-no-bs-training-glossary/168982
- Reverse-engineering / blueprint follow-up:  
  https://discuss.huggingface.co/t/tutorial-update-reverse-engineering-breakdown-released-the-muon-is-scalable-cpu-friendly-blueprint/170078

### 8.4 Additional community references

- Issues in KellerJordan/Muon repo (learning rate, memory questions):  
  [https://github.com/KellerJordan/Muon/issues](https://github.com/KellerJordan/Muon/issues)
- Reddit: “[R] Muon is Scalable for LLM Training”:  
  [https://www.reddit.com/r/MachineLearning/comments/1ixzj26/r_muon_is_scalable_for_llm_training/](https://www.reddit.com/r/MachineLearning/comments/1ixzj26/r_muon_is_scalable_for_llm_training/)
- Example concise Muon notes in Japanese (Zenn scrap):  
  [https://zenn.dev/colum2131/scraps/5971d43e710718](https://zenn.dev/colum2131/scraps/5971d43e710718)

These references collectively give you the conceptual foundations, math details, implementation guidance, distributed training insights, and concrete LLM examples needed to understand and apply Muon in real projects, including Moonshot-style Kimi/K2 thinking models.
