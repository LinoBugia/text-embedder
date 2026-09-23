---
source: "huggingface+chat+web+files"
topic: "Muon optimizer and ecosystem"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-30T00:00:00Z"
---

# Muon optimizer: matrix-orthogonal updates for large neural networks

## 1. Background and overview

Muon is a neural network optimizer that operates on **entire weight matrices** in hidden layers instead of treating each scalar parameter independently. It was introduced by Keller Jordan in late 2024 under the title *“Muon: An optimizer for hidden layers in neural networks”*, and quickly attracted attention because it combines:

- **SGD with momentum on matrices**, not scalars.
- **Orthogonalization of the update** via a fast Newton–Schulz iteration, so that updates have roughly unit singular values in all directions.
- **Lower optimizer-state memory** than AdamW for matrix-heavy models, while adding only a small amount of extra compute.

At a high level, Muon is designed to address two persistent themes in large-scale optimization:

1. **Skewed update directions.** In large linear layers, most of the gradient energy tends to concentrate in a few singular directions. This means we effectively “learn” in a small subspace while many directions barely move at all, wasting model capacity and sometimes hurting stability.

2. **Optimizer-state memory explosion.** AdamW stores first and second moments for each parameter. For models dominated by large matrices (transformers, CNN backbones), these extra tensors become a major memory cost.

Muon addresses both:

- It **groups parameters as matrices**, maintains a **matrix-valued momentum**, and then orthogonalizes this momentum before applying the update. The orthogonalization flattens the singular value spectrum, so each singular direction sees a similar step magnitude.
- It stores just **two copies of each matrix** (parameters + momentum) rather than three copies (parameters + first moment + second moment). For matrix-heavy models this cuts optimizer memory by roughly one third compared to AdamW, at the cost of ~1% extra FLOPs for orthogonalization.

In practice, Muon is **not a full replacement** for AdamW. Most real-world systems use a **hybrid setup**:

- Use **Muon for 2D weight matrices** (hidden-layer linear and conv weights).
- Use **AdamW (or similar) for 1D/other parameters**: embeddings, layernorms, biases, output heads, etc.

The subsequent ecosystem of papers, tutorials, and engineering work (including Moonshot AI’s *“Muon is Scalable for LLM Training”* and the Kimi K2 MoE models) shows that this idea is not just a curiosity: Muon can train multi-billion to trillion-parameter models with strong stability and competitive or better compute efficiency.

This knowledge base consolidates:

- The **core algorithm and math** from the original Muon writeup.
- Scaling results and design decisions from **Moonshot’s LLM work** (Moonlight, Kimi K2).
- **Practical patterns** from Hugging Face community tutorials and reproducibility projects.
- **Recent extensions** such as AuON and adaptive variants that build on the Muon idea.

Throughout, the focus is on how Muon actually works and what you should pay attention to if you want to use it in practice.


## 2. From official-style sources: writeups, papers, and blogs

### 2.1 Keller Jordan’s original Muon writeup

The canonical description of Muon comes from Keller Jordan’s blog post *“Muon: An optimizer for hidden layers in neural networks”* and the associated `KellerJordan/Muon` GitHub repository.

#### 2.1.1 Intended scope

Muon is explicitly **an optimizer for hidden-layer weight matrices**:

- It assumes parameters are organized into **2D tensors** representing linear maps.
- It can be used on convolution kernels by reshaping them into matrices.
- It is *not* intended for scalar parameters, layernorm scales, biases, or embeddings.

Practically, you split parameters into two groups:

- **Muon parameters:** all 2D matrices in attention, MLP, convolution blocks, etc.
- **Non-Muon parameters:** everything else, handled by AdamW or another optimizer.

This division is the basis of the hybrid setups used in Moonshot AI’s Moonlight and Kimi K2 training and in the distributed tutorials on Hugging Face.


#### 2.1.2 Algorithm sketch

For a single weight matrix \(W\), Muon maintains a momentum matrix \(M_t\) and applies an orthogonalized update at each step. A simplified sketch is:

1. **Gradient and momentum update**

   \[
   G_t = \nabla_W L_t, \quad
   M_t = \beta M_{t-1} + (1 - \beta) G_t
   \]

   where \(\beta\) is the momentum coefficient (e.g. 0.9–0.97). Optionally, Nesterov-style momentum can be used.

2. **Reshape and center**

   - Ensure \(M_t\) is in matrix form (\(m \times n\)). For convs, flatten `(out_channels, in_channels * kernel_h * kernel_w)`.
   - Optionally apply simple conditioning (mean-centering or RMS normalization) as a pre-step.

3. **Orthogonalization via Newton–Schulz**

   The core of Muon is a small number of Newton–Schulz iterations that transform \(M_t\) into a matrix \(\tilde M_t\) whose singular values are approximately 1.

   Conceptually, you apply a function \(f\) to \(M_t\) such that:

   - \(f(M_t)\) preserves the singular vectors of \(M_t\),
   - but rescales the singular values so that \(\sigma_i(f(M_t)) \approx 1\) for all \(i\).

   In practice, this is implemented with a low-order polynomial iteration in BF16 or FP16, typically in **5 or fewer steps**. The key properties:

   - **Spectral balancing:** large singular values are reduced, small ones are increased.
   - **Orientation preserved:** the “direction” of the update in matrix space is preserved up to rotation, but the scale per singular direction is equalized.

4. **Apply the update**

   After orthogonalization you treat \(\tilde M_t\) as the step direction:

   \[
   W_{t+1} = W_t - \eta \, s(W_t) \, \tilde M_t
   \]

   where \(\eta\) is the learning rate and \(s(W_t)\) is a shape-dependent scaling factor (discussed more in the Moonshot paper).


#### 2.1.3 Intuition: why orthogonalization helps

Jordan’s writeup and subsequent tutorials emphasize a geometric picture:

- For a typical gradient matrix \(G\), the singular values may have a **long-tailed distribution**: a few large values dominate, and many are tiny.
- If you repeatedly apply such updates, you mostly move along those dominant directions, while many degrees of freedom barely change.
- The **orthogonalization step** rebalances these directions so that each one has a similar step size.

Results:

- The update moves **more evenly across the parameter space**, which can reduce pathological curvature effects and help avoid getting stuck in subspaces.
- The optimizer can tolerate **larger learning rates** without divergent behavior, because the spectral norm of the update is controlled by construction.
- For speedrun-style tasks (NanoGPT, CIFAR-10), Muon can reach target losses in fewer steps or less wall-clock time compared to tuned AdamW baselines.


#### 2.1.4 Memory vs compute tradeoff

Compared to AdamW:

- **AdamW** keeps parameters \(W\), first moment \(m\), and second moment \(v\) for each scalar parameter. This is effectively **3× parameter memory** for matrix parameters.
- **Muon** keeps parameters \(W\) and momentum matrix \(M\) for each matrix, for **2× parameter memory**.

For transformer models where most parameters are in 2D weight matrices, this yields approximately **33% reduction in optimizer-state memory**. The cost is:

- A small number of additional matrix multiplications per layer, resulting in roughly **1% extra FLOPs** in end-to-end training.
- Practically negligible impact on step time compared to the attention and MLP compute.


### 2.2 Moonshot AI: “Muon is Scalable for LLM Training”

The Moonshot AI technical report *“Muon is Scalable for LLM Training”* and the associated `MoonshotAI/Moonlight` code base extend Muon to **compute-optimal LLM training at scale**.

#### 2.2.1 Two key tricks for scaling

The paper identifies two crucial techniques to make Muon “just work” for large LLMs without extensive hyperparameter retuning:

1. **Decoupled weight decay**

   - Similar to AdamW, they use **decoupled weight decay** instead of L2 regularization integrated into the gradient.
   - This is critical for controlling norm growth in big models and for maintaining generalization performance when using large learning rates.

2. **Per-parameter update scale matching (RMS matching)**

   - When switching from AdamW to Muon, the per-parameter RMS of the update can change dramatically because Muon’s orthogonalization normalizes singular values.
   - The authors introduce a strategy to **match RMS update norms** to an AdamW baseline layer-by-layer, allowing **learning-rate schedules from AdamW to transfer** with minimal tuning.
   - This “RMS matching” helps prevent under- or over-updating specific layers, especially in wide/deep transformers and MoE blocks.

With these techniques, the paper reports that Muon can be largely **plugged into existing LLM training recipes** with minimal hyperparameter changes.


#### 2.2.2 Scaling laws and compute efficiency

Moonshot’s experiments measure Muon’s performance under **compute-optimal training** conditions:

- They compare AdamW vs. Muon across multiple model sizes and token budgets.
- Under equal compute budgets, Muon often reaches **lower cross-entropy loss** than AdamW.
- They report up to about **2× compute efficiency** in some regimes when you ask: “How many FLOPs are needed to reach a target loss?”

The key takeaway is not that Muon dominates AdamW in all settings, but that:

- There exists a large region of LLM training configurations where **Muon is at least competitive and often more efficient**, especially when memory bandwidth or optimizer state is the bottleneck.
- Carefully engineered **distributed implementations** (DP × TP) can keep Muon’s extra matrix operations from becoming a communication bottleneck.


#### 2.2.3 Moonlight models

The Moonlight models are Mixture-of-Experts LLMs (e.g. 3B and 16B variants) trained with Muon as the primary optimizer for hidden matrices. From their code and model cards you can extract that:

- They apply Muon to **dense and MoE expert weights**, while keeping a standard optimizer (AdamW-like) for the rest.
- Training uses **5.7T tokens** and strong regularization (including weight decay) across a wide range of tasks.
- The resulting models update the **Pareto frontier** of performance vs training FLOPs, demonstrating that Muon is viable for “frontier-adjacent” LLMs, not just toy models.


### 2.3 Hugging Face community article: “Muon Optimizer: The Power of Collective Momentum”

Yi Cui’s community article on Hugging Face, *“Muon Optimizer: The Power of Collective Momentum”*, provides an accessible, practitioner-friendly description of Muon.

Key contributions of the article:

- **2D geometric explanation.**

  - It uses small examples (e.g., 2×2 matrices) to visualize what orthogonalization does to a gradient matrix.
  - Before orthogonalization: one long vector and one short vector (highly skewed singular values).
  - After orthogonalization: two orthogonal unit vectors, representing **balanced update directions**.

- **“Collective momentum” framing.**

  - Rather than tracking per-parameter statistics, Muon tracks a **collective momentum for the entire matrix**.
  - The orthogonalization step enforces that this collective momentum does not become dominated by a single direction.
  - This “group view” clarifies why Muon can both accelerate optimization and stabilize training in high dimensions.

- **Memory vs compute tradeoff clarified with concrete numbers.**

  - For a 7B transformer, AdamW can require tens of gigabytes of optimizer state (three copies of matrix parameters).
  - Muon can cut this by about one third while adding only ~1% extra FLOPs, thanks to efficient mixed-precision Newton–Schulz implementations.

The article is a good starting point for engineers who want an intuitive picture before reading the more technical papers.


### 2.4 Related theoretical and algorithmic developments

The Muon idea has already inspired follow-up work on **orthogonalized momentum and spectral-normalized updates**:

- **AuON (A Linear-time Alternative to Semi-Orthogonal Momentum Updates).**

  - AuON proposes a **linear-time approximation** to Muon-like semi-orthogonal momentum updates.
  - It uses trust-region and spectral-norm bounds to provide **convergence guarantees** for non-convex optimization under certain assumptions.
  - AuON can be seen as a lightweight counterpart to Muon: aiming for similar spectral control with less compute overhead, while preserving good empirical performance on vision and language tasks.

- **“Adagrad meets Muon” (adaptive stepsizes for orthogonal updates).**

  - This work combines ideas from **Adagrad** (adaptive learning rates) with Muon’s orthogonal update structure.
  - It introduces **adaptive step-size rules** for orthogonalized updates, aiming to improve robustness across different scales and tasks.

- **Gradient orthogonalization as trust-region methods.**

  - Other work interprets Muon’s orthogonalization as solving a **non-Euclidean trust-region problem under a spectral norm constraint**.
  - This provides a more principled lens: Muon approximately solves a constrained optimization problem where updates are restricted to a spectral ball, explaining its stability with large learning rates.

These developments suggest that **orthogonalized momentum** is a fruitful direction for both theory and practice, and that Muon is one point in a broader design space.


## 3. From model cards and large-scale deployments

### 3.1 Kimi K2 and MuonClip

The **Kimi K2** models from Moonshot AI are large-scale Mixture-of-Experts LLMs (1T parameters total, ~32B active per token) designed for **agentic, tool-using behavior**. Public summaries and model-card style descriptions highlight that:

- Kimi K2 is **trained with the Muon optimizer** at unprecedented scale.
- Training uses **15.5T tokens** with **zero training instability**, a point they emphasize to show Muon’s reliability in long-horizon training.
- They introduce a **“MuonClip” optimizer variant**, which combines Muon with additional clipping and stabilization tricks to handle extreme scales.

From the model ecosystem and related datasets, you can infer several practical points:

- Muon (and MuonClip) are used for **hidden-layer matrices** in both dense and MoE experts.
- The hybrid optimizer setup remains: embeddings and other non-matrix parameters still rely on an AdamW-like optimizer.
- The training process is **deeply intertwined with agentic objectives**: long context windows, heavy tool use, and reasoning benchmarks like SWE-Bench, BrowseComp, etc.

Kimi K2 demonstrates that Muon’s principles extend to **1T-parameter MoE models** and **agent-style training curricula**, not just standard next-token prediction.


### 3.2 Moonshot AI’s Moonlight models

Moonlight models (e.g., `Moonlight-16B`) are **MoE LLMs** trained as part of the Muon scaling study. At a high level:

- They provide concrete open-source examples of **LLMs trained with Muon** whose checkpoints are available on Hugging Face.
- They showcase **compute-efficiency improvements**: better loss at similar or lower FLOPs than AdamW baselines.
- Their training logs and hyperparameters exemplify **practical Muon recipes**: learning rates, weight decay, RMS matching settings, and the exact parameter group splits.

If you want model-scale examples of Muon in practice, Moonlight provides some of the best currently available references.


### 3.3 Relationship between Muon, Moonlight, and Kimi K2

The ecosystem can be summarized as:

- **Muon**: the core optimizer idea (SGD-like momentum + spectral orthogonalization on matrices).
- **Moonlight**: proof that Muon can scale to multi-billion-parameter LLMs with improved compute efficiency.
- **Kimi K2**: a 1T-parameter MoE “agentic” LLM family, trained with Muon-based variants (including MuonClip) for real-world reasoning and tool-use workloads.

For an engineer or researcher, this means:

- You can study the **algorithm and intuition** via Keller Jordan’s and community tutorials.
- You can see **mid-scale deployments** via Moonlight models and code.
- You can see **frontier-scale deployments** via Kimi K2’s training descriptions and associated research.


## 4. From community, forums, GitHub, and reproducibility projects

### 4.1 “Understanding and implementing the Muon optimizer” (HF tutorial dataset)

The Hugging Face dataset `bird-of-paradise/muon-tutorial` is a **notebook-style, educational deep dive** into Muon that complements the original blog:

- It explains the **problem of skewed singular value distributions** and shows concrete SVD examples on real gradients.
- It walks through a **clean PyTorch implementation** of Muon, including:
  - Parameter grouping into Muon vs non-Muon tensors.
  - Newton–Schulz orthogonalization in BF16/FP16.
  - A simple NanoGPT-style language model experiment comparing Muon vs AdamW.
- It emphasizes **debugging techniques**, including logging:
  - Singular value histograms for gradient and momentum.
  - Update RMS before and after orthogonalization.
  - Per-layer learning rate and update norms.

For learners, this tutorial is effectively a “Muon textbook chapter” that bridges theory and code.


### 4.2 Distributed Muon: CPU-friendly blueprint

The `bird-of-paradise/muon-distributed` dataset is an **expert-level systems engineering breakdown** of Moonshot’s distributed Muon implementation:

- It re-implements Muon in a **CPU-only, Gloo-based environment**, so you can run a full DP × TP Muon optimizer on a single machine or Colab-like environment.
- It provides heavily annotated code and notebooks that explain step by step:
  - How data parallel (ZeRO-1) and tensor parallel sharding interact with matrix updates.
  - The sequence of collective operations required for a Muon step:
    - Data-parallel gather of shards.
    - Tensor-parallel gather of matrix slices.
    - Newton–Schulz orthogonalization on the full matrix.
    - Redistributing shards back to TP then DP layouts.
  - The design of a **“dist_meta”** data structure that encodes the virtual layout of shards for different parallelism dimensions.

The goal is not to simplify the algorithm but to **make the “distributed nightmare” explorable**, by exposing every step of the pipeline explicitly in well-commented code.


### 4.3 Distributed Muon reproducibility artifacts

The dataset `bird-of-paradise/muon-distributed-reproducibility` and the associated Hugging Face forum posts provide:

- **Chrome trace files** (PyTorch Profiler) comparing AdamW vs Muon in various DP × TP configurations on a 4-GPU cluster.
- **Analysis scripts** that parse these traces and compute metrics like:
  - Communication vs compute time ratios.
  - Optimizer latency as a fraction of total step time.
- **Figures and logs** used to validate claims such as:
  - Muon’s extra orthogonalization logic contributes only ~1.1% of total step time in realistic LLM settings.
  - Certain hybrid DP=2/TP=2 configurations achieve **0.57× communication cost** compared to comparable AdamW setups.

Together, `muon-distributed` and `muon-distributed-reproducibility` turn Moonshot’s high-level claims into **hands-on, reproducible experiments**, which is particularly valuable if you plan to adopt Muon in production or research infrastructure.


### 4.4 GitHub implementations and research repos

Several GitHub repositories deepen the ecosystem:

- **KellerJordan/Muon**

  - Reference implementation of Muon for smaller models and research experiments.
  - Includes code for NanoGPT-style language models and CIFAR-10 benchmarks.

- **MoonshotAI/Moonlight**

  - Contains code for training Moonlight models with Muon in large-scale distributed settings.
  - Exposes how Muon is integrated into a full LLM training stack: parameter groups, finetuning recipes, etc.

- **Muon vs AdamW research repos (e.g., `vukrosic/muon-optimizer-research`)**

  - Explore **learning-rate scaling laws** and small-LLM behavior when switching from AdamW to Muon.
  - Provide additional benchmarks and ablations complementing the main Moonshot paper.

- **AuON reference implementation (`ryyzn9/AuON`)**

  - Demonstrates how linear-time approximations to semi-orthogonal momentum updates can be implemented and integrated with Muon-like optimizers.
  - Offers a useful comparison point for engineers wanting to trade off orthogonalization fidelity vs compute.


### 4.5 Community discussion themes

Hugging Face forum threads and community posts highlight several recurring themes when practitioners try Muon:

- **Learning rate transfer and tuning**

  - Naively reusing AdamW learning rates sometimes fails because Muon’s normalized updates have different RMS behavior.
  - RMS matching (as in the Moonshot paper) or **per-layer LR scaling** often fixes this.

- **Stability at very large scale**

  - On very large transformers, edge cases like **exploding attention logits** can appear when using more aggressive orthogonalization or insufficient clipping.
  - Moonshot’s MuonClip variants and QK-clipping style techniques are examples of stabilization tricks that may be necessary at frontier scales.

- **Implementation pitfalls**

  - Incorrect reshaping of conv weights or transposed linear layers can silently break the orthogonalization logic.
  - Failing to exclude 1D parameters from Muon, or accidentally mixing Muon and AdamW on the same tensor, can lead to subtle bugs.

These discussions make it clear that while Muon is powerful, it also demands **careful engineering**.


## 5. Implementation patterns and practical tips

This section summarizes common patterns that emerge from official sources, community tutorials, and empirical studies. It is written from the perspective of someone implementing Muon in a PyTorch + Hugging Face ecosystem.

### 5.1 Parameter grouping

A typical parameter grouping strategy is:

1. **Muon group (matrices only)**

   - All 2D weight tensors from:
     - Attention projections (Q, K, V, O).
     - MLP layers (input, output projections).
     - Convolution kernels reshaped to 2D.

2. **Non-Muon group (everything else)**

   - Token and position embeddings.
   - LayerNorm / RMSNorm weights and biases.
   - Output heads, bias vectors, and any 1D or scalar parameters.

This yields two optimizers (or one optimizer with two parameter groups):

- **Muon optimizer** on the matrix group.
- **AdamW** (or equivalent) on the non-matrix group.

In large systems, additional grouping by parallelism (per-shard groups) or by module type is common, but the matrix vs non-matrix split is the key semantic boundary.


### 5.2 Core Muon hyperparameters

While exact values depend on the task and architecture, the following patterns are common in public implementations and tutorials:

- **Momentum \(\beta\)**

  - Typically in the range **0.9–0.97**.
  - Higher momentum can improve stability but may require more careful RMS matching.

- **Learning rate \(\eta\)**

  - Often **similar to or slightly higher than AdamW** for the same model, thanks to the spectral normalization of updates.
  - When transferring schedules from AdamW, it is safer to start conservative and adjust after logging update RMS statistics.

- **Weight decay**

  - Decoupled weight decay as in AdamW is considered **mandatory** for large-scale Muon training.

- **Orthogonalization iterations**

  - 3–5 Newton–Schulz iterations in BF16/FP16 are typical.
  - More iterations improve orthogonalization fidelity but incur slightly more compute.

- **Shape-dependent scaling**

  - Implementations usually include a factor like \(\sqrt{\max(m, n) / \min(m, n)}\) or related heuristics to normalize step size by matrix shape.
  - When combined with RMS matching, this helps keep per-layer update RMS within a comfortable range.


### 5.3 Diagnostics and monitoring

When first integrating Muon, robust diagnostics are essential. Recommended metrics include:

- **Per-layer update RMS**

  - Log RMS of the Muon update per layer and compare to AdamW baselines.
  - Adjust LR or shape-dependent scaling if some layers are under- or over-updating.

- **Singular value distributions**

  - Periodically compute SVD for a few representative gradients/momenta.
  - Compare singular values before vs after orthogonalization to ensure the transform is doing what you expect.

- **Optimizer time and communication**

  - Use PyTorch Profiler or similar tools to measure time spent in Muon step vs total step time.
  - Especially in distributed setups, verify that matrix orthogonalization is not dominating communication. Datasets like `muon-distributed-reproducibility` provide reference numbers for realistic systems.

- **Stability indicators**

  - Track signs of divergence: exploding loss, NaNs in attention logits, sudden spikes in gradient norms.
  - If these appear only with Muon (not AdamW), re-check clipping, weight decay, and orthogonalization fidelity.


### 5.4 Integration with Hugging Face Transformers and training stacks

In a Hugging Face-style stack (Transformers + Accelerate/TRL/PEFT), some common integration patterns are:

- **Custom optimizer factories**

  - Define a function that inspects model modules and returns two parameter lists: `muon_params`, `adamw_params`.
  - Instantiate a Muon optimizer for `muon_params` and an AdamW (or RMSProp, etc.) optimizer for `adamw_params`.

- **FSDP / ZeRO compatibility**

  - Ensure that Muon’s orthogonalization step sees **full matrices**, not just shards.
  - This often requires:
    - Gathering shards within a TP group, applying orthogonalization, then re-sharding.
    - Coordinating with FSDP hooks so that parameters are in the correct state when Muon runs.

- **Mixed precision**

  - Running orthogonalization in BF16/FP16 is standard to keep overhead low.
  - Care must be taken to avoid numerical issues; sometimes a small number of ops are forced into FP32 for safety (e.g., some parts of Newton–Schulz).

Studying open-source implementations (Moonlight, Muon tutorials, AuON reference code) is strongly recommended before attempting a production integration.


### 5.5 When to consider Muon

Muon makes the most sense when:

- Your model is dominated by **large weight matrices**, e.g. transformers or convnets.
- Optimizer-state memory is a limiting factor, or you want to squeeze more useful parameters into the same memory budget.
- You are willing to invest some engineering time to integrate and validate a more complex optimizer.

It may be less beneficial when:

- The model is small or dominated by embeddings/1D parameters.
- The training regime is extremely sensitive and already tuned around a specific optimizer.
- Simplicity and mature tooling (AdamW, Lion, etc.) outweigh the plausibly modest efficiency gains.


## 6. Limitations, caveats, and open questions

Even with strong empirical results, Muon and orthogonalized momentum methods come with important caveats.

### 6.1 Complexity and engineering cost

Muon is **significantly more complex** than AdamW:

- It requires matrix-aware parameter grouping.
- The orthogonalization logic must be implemented carefully for different tensor shapes and parallelism strategies.
- Debugging distributed Muon often means understanding **DP × TP communication patterns**, which is non-trivial.

For many teams, this engineering cost may be higher than the marginal gains over a well-tuned AdamW setup.


### 6.2 Stability at frontier scales

While Moonshot’s results and Kimi K2 training suggest that Muon can be stable at 1T-parameter scale, open questions remain:

- How robust is Muon (or MuonClip) across **diverse architectures** (non-MoE, vision transformers, multimodal models)?
- How sensitive is performance to small changes in orthogonalization hyperparameters (number of iterations, precision, scaling rules)?
- How do Muon variants behave with extreme long-context or streaming setups where gradient statistics change rapidly?

Recent work like AuON highlights that even small changes to orthogonalization schemes can affect stability, especially in very large transformers.


### 6.3 Theoretical understanding and alternatives

Although there is growing theoretical work interpreting gradient orthogonalization as a **spectral trust-region method**, the theory for full-scale LLMs is still incomplete. Questions include:

- What are the exact conditions under which orthogonalized momentum guarantees faster convergence than AdamW-like methods?
- How should we design **adaptive** orthogonalized optimizers that remain stable across a wide range of tasks and model sizes?
- Are there simpler algorithms (like AuON or other normalized SGD variants) that capture most of Muon’s benefits with less complexity?

These questions are beginning to be explored but are far from settled.


### 6.4 Ecosystem maturity

Compared to AdamW and other classic optimizers, Muon’s ecosystem is still young:

- Fewer libraries have **battle-tested implementations**.
- Educational material is rapidly improving (e.g., HF tutorials and distributed blueprints) but still limited compared to mainstream optimizers.
- Many integrations are currently **research prototypes** rather than fully supported features in popular training frameworks.

This means that early adopters should be prepared to:

- Read and understand research code and tutorial datasets.
- Contribute bug fixes and improvements upstream.
- Maintain their own forks or wrappers around Muon implementations for some time.


## 7. Summary and recommended reading

Muon is a matrix-aware optimizer that uses **collective momentum and spectral orthogonalization** to:

- Balance update directions across singular vectors.
- Reduce optimizer-state memory for matrix-heavy models by about one third.
- Achieve competitive or better training efficiency than AdamW in a variety of settings, including large LLMs.

Moonshot AI’s work (Moonlight, Kimi K2), together with community tutorials and reproducibility datasets, demonstrate that Muon is not just a theoretical curiosity: it can train **multi-billion to trillion-parameter models** with strong stability and performance.

At the same time, Muon introduces substantial algorithmic and systems complexity. Its adoption is best suited for teams that:

- Operate at scales where optimizer memory and compute efficiency are critical.
- Are comfortable engineering and debugging custom optimizers in distributed environments.
- Are willing to follow evolving best practices from the community as new variants (AuON, adaptive Muon, MuonClip) are developed.

### Recommended reading and resources

If you want to go deeper, the following resources provide a good path through the Muon ecosystem:

1. **Core idea and algorithm**
   - [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
   - [KellerJordan/Muon (GitHub)](https://github.com/KellerJordan/Muon)

2. **Scaling to LLMs**
   - [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
   - [MoonshotAI/Moonlight (GitHub)](https://github.com/MoonshotAI/Moonlight)

3. **Practitioner-friendly explanations**
   - [Muon Optimizer: The Power of Collective Momentum](https://huggingface.co/blog/onekq/muon-optimizer)
   - [Understanding and implementing the Muon optimizer](https://huggingface.co/datasets/bird-of-paradise/muon-tutorial)

4. **Distributed engineering and reproducibility**
   - [The "Muon is Scalable" Blueprint: A Distributed Muon Engineering Breakdown](https://huggingface.co/datasets/bird-of-paradise/muon-distributed)
   - [Distributed Muon: Field Notes & Reproducibility Artifacts](https://huggingface.co/datasets/bird-of-paradise/muon-distributed-reproducibility)
   - [Reproducing & Validating Distributed Muon (MoonshotAI)](https://discuss.huggingface.co/t/reproducing-validating-distributed-muon-moonshotai-performance-communication-results/170969)

5. **Large-scale deployments and variants**
   - [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
   - [AI ecosystem with model cards (Kimi K2 entry)](https://huggingface.co/datasets/modelbiome/ai_ecosystem_withmodelcards)
   - [AuON: A Linear-time Alternative to Semi-Orthogonal Momentum Updates](https://arxiv.org/abs/2509.24320)
   - [Adagrad meets Muon: Adaptive stepsizes for orthogonal updates](https://arxiv.org/abs/2509.02981)

Taken together, these resources provide a complete learning path from **basic Muon intuition** through **full distributed implementations** and **frontier-scale deployments**.
