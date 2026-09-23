---
source: "huggingface+chat+web+repo"
topic: "Vertex AI with Hugging Face ecosystem (KB v2)"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-12-05T00:00:00Z"
---

# Vertex AI with Hugging Face ecosystem (KB v2)

This knowledge base focuses on **how to use Google Cloud Vertex AI together with the Hugging Face ecosystem**, with a strong emphasis on the officially supported **Hugging Face Deep Learning Containers (DLCs)**, the **Google-Cloud-Containers** repository, and lessons from recent community discussions.

The content assumes you have:
- Basic familiarity with Google Cloud (projects, service accounts, IAM, GCS).
- Some experience with Hugging Face libraries (`transformers`, `datasets`, `trl`).
- Access to a GCP project with Vertex AI enabled and GPU/TPU quotas.

It is structured so you can either read it linearly or jump directly to the implementation patterns and best practices.

## 1. Background and overview

### 1.1 What is Vertex AI?

Vertex AI is Google Cloud’s managed platform for building, training, deploying, and operating ML and generative AI workloads on a unified stack. It provides:  

- **Custom training**: serverless training jobs on CPU/GPU/TPU, or user-managed training on GKE.
- **Model deployment**: managed online and batch prediction endpoints.
- **Generative AI / Model Garden**: a catalog of Google and partner/open models, including some Hugging Face models, with consistent governance, monitoring, and billing.
- **MLOps tooling**: experiment tracking, model registry, pipelines, evaluation, monitoring.

Official overview and docs:  

- Vertex AI docs index: [https://docs.cloud.google.com/vertex-ai/docs](https://docs.cloud.google.com/vertex-ai/docs)  
- Product page: [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)

For Hugging Face users, Vertex AI is best viewed as:

> A managed execution and serving environment where you run Hugging Face training and inference containers, with GCS, Artifact Registry, and other GCP services providing storage and infrastructure around them.

### 1.2 Hugging Face + Google Cloud partnership

Hugging Face and Google Cloud maintain an official integration that centers on **Deep Learning Containers (DLCs)** for GCP. These DLCs are published in Google Cloud’s Artifact Registry and described in both the Hugging Face docs and the `huggingface/Google-Cloud-Containers` GitHub repository.

Key entry points:

- Hugging Face on Google Cloud docs:  
  [https://huggingface.co/docs/google-cloud/en/index](https://huggingface.co/docs/google-cloud/en/index)
- Google-Cloud-Containers repo (containers + examples):  
  [https://github.com/huggingface/Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers)
- Google Cloud partnership organization on Hugging Face Hub:  
  [https://huggingface.co/google-cloud-partnership](https://huggingface.co/google-cloud-partnership)

From the repository README:

- DLCs are Docker images for **training** and **inference** of Transformers, Sentence Transformers, and Diffusers models on Vertex AI, GKE, and Cloud Run.
- Containers are publicly maintained and periodically updated by Hugging Face and Google Cloud, and are available to all GCP customers in Artifact Registry.
- The repo contains **examples** on how to train and deploy models with Vertex AI using these containers, including TRL SFT, LoRA, and TGI/TEI deployments.  
  See the README “Examples” tables: Vertex AI sections for training and inference.  

The Hugging Face docs for Google Cloud (linked from the repo) further document how to:

- Launch **custom training jobs** on Vertex using the PyTorch Training DLC.
- Deploy **TGI** (Text Generation Inference) and **TEI** (Text Embeddings Inference) DLCs as Vertex AI endpoints.
- Combine DLCs with Hugging Face libraries and Model Garden open models.

### 1.3 Why this KB (and how it relates to Google-Cloud-Containers)

A recent Hugging Face forum thread explicitly calls for an extended knowledge base for using the Hugging Face toolbox on Vertex AI Training. The official response from Hugging Face confirms that:

- The **Google-Cloud-Containers repo is intended to be the single source of truth** for GCP users coming from the Hugging Face ecosystem.
- Users are encouraged to **open issues and contribute examples** to that repo to expand coverage and document best practices.

Forum thread:  

- “Need to build extended knowledge base about Hugging Face toolbox best practices and recipes in the VertexAI Training environment”  
  [https://discuss.huggingface.co/t/need-to-build-extended-knowledge-base-about-hugging-face-toolbox-best-practices-and-recipes-in-the-vertexai-training-environment/171168](https://discuss.huggingface.co/t/need-to-build-extended-knowledge-base-about-hugging-face-toolbox-best-practices-and-recipes-in-the-vertexai-training-environment/171168)

This document is designed to be compatible with that goal:

- It **summarizes and organizes** patterns already present in docs, blog posts, examples, and the repo.
- It **integrates community learnings** (including tricky edge cases around TRL CLI, datasets on GCS, and gated models).
- It can be used as a **draft knowledge base** whose content can later be turned into additional examples or docs inside `Google-Cloud-Containers` and the Hugging Face Google Cloud docs.

## 2. Official docs and reference material

### 2.1 Vertex AI + Hugging Face in official docs

Google Cloud provides a dedicated guide on using **Hugging Face models as open models** in Vertex AI’s Model Garden:

- “Use Hugging Face Models” (Generative AI on Vertex AI):  
  [https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models)

Key points from that guide:

- Vertex AI can directly deploy certain classes of Hugging Face models (text generation, text embeddings, text-to-image, image-text) via the Model Garden UI and API.
- These models are **managed open models** inside Vertex AI, benefiting from:
  - Centralized governance,
  - Integrated monitoring,
  - Uniform billing and access control.
- You can deploy such models either in **Vertex AI** or **GKE**, depending on how much control you want over infrastructure vs how much you want to rely on a serverless managed stack.

This “Model Garden” route is complementary to DLC-based custom training/serving:

- Use **Model Garden** when you want managed open models with minimal container management.
- Use **Hugging Face DLCs** when you need **full control** over training/inference code, custom datasets, custom weights, or unsupported architectures.

### 2.2 Hugging Face on Google Cloud docs and examples

The Hugging Face “Google Cloud” docs consolidate DLC information, examples, and links to Model Garden:

- Main index:  
  [https://huggingface.co/docs/google-cloud/en/index](https://huggingface.co/docs/google-cloud/en/index)
- “Other Resources” page listing training and inference examples:  
  [https://huggingface.co/docs/google-cloud/en/resources](https://huggingface.co/docs/google-cloud/en/resources)

Representative Vertex AI examples (all implemented using Google-Cloud-Containers DLCs):

- **Training**:
  - Fine-tune Gemma 2B with PyTorch Training DLC using SFT + LoRA on Vertex AI.  
  - Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT on Vertex AI.  
- **Inference**:
  - Deploy BERT models with PyTorch Inference DLC on Vertex AI.  
  - Deploy embedding models with TEI DLC on Vertex AI.  
  - Deploy FLUX with PyTorch Inference DLC on Vertex AI.  
  - Deploy Gemma 7B with TGI DLC on Vertex AI (from Hub or from GCS).  
  - Deploy Llama 3 / Llama Vision models with TGI DLC on Vertex AI.

These examples use the **Vertex AI Python SDK** (`google-cloud-aiplatform`) and cloud-native constructs like:

- `CustomContainerTrainingJob` for training.
- `aiplatform.Model.upload` with `serving_container_image_uri` (DLC) + environment variables for inference.
- `aiplatform.Endpoint` for deployment.

### 2.3 Google-Cloud-Containers repository structure and role

The attached `Google-Cloud-Containers` repository (and its public GitHub mirror) is the central place where:

- Container definitions for training/inference DLCs live (`containers/`).
- Example notebooks and scripts are maintained (`examples/`).
- Docs and feature descriptions are authored (`docs/`).

From the README’s **“Examples”** section (Vertex AI rows):

- **Training**:
  - `examples/vertex-ai/notebooks/trl-lora-sft-fine-tuning-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/trl-full-sft-fine-tuning-on-vertex-ai`
- **Inference**:
  - `examples/vertex-ai/notebooks/deploy-bert-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-embedding-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-flux-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-gemma-from-gcs-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-llama-vision-on-vertex-ai`  
  - `examples/vertex-ai/notebooks/deploy-llama-3-1-405b-on-vertex-ai`

The README also lists **latest DLC container URIs** (PyTorch training/inference, TGI, TEI), updated as new versions are released. For full and latest listings, you are expected to:

- Check Deep Learning Containers documentation.
- Use `gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-"`.

This repo is explicitly pointed to in the VertexAI KB forum thread as the **place to contribute examples and best practices**, which is why this document is written to map naturally into its directory structure and documentation style.

### 2.4 Custom prediction routines and self-deployed models

Hugging Face also provides a blog post that explains how to deploy Hub models on Vertex AI using **Custom Prediction Routines (CPR)** instead of DLCs:

- “Deploying 🤗 Hub models in Vertex AI” (HF blog).  
  [https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai)

Key ideas:

- Build a **custom prediction container** that wraps `transformers.pipeline` with pre/post-processing logic.
- Register the model with Vertex AI’s Model Registry and deploy to an endpoint.
- Use the **Vertex AI Python SDK** to programmatically build and push the image, then deploy.

On the Vertex side, this fits into the **self-deployed models** framework described in Google Cloud docs (“deploy models with custom weights”, “self-deployed models in Model Garden”), where you bring your own container image and model weights, and Vertex handles deployment and scaling.

## 3. Community insights (TRL CLI, datasets, gated models, KB plans)

### 3.1 The knowledge-base initiative for Vertex AI + HF

The forum thread that you referenced is both a question and a roadmap:

- A user describes the need for an extended KB for Vertex AI Training + Hugging Face toolbox, because real projects quickly hit scenarios not fully covered by existing docs and notebooks.
- Hugging Face staff reply that:
  - This is “exciting” and clearly a useful initiative.
  - The right place to **accumulate examples and best practices is the Google-Cloud-Containers repo**, which they want to ramp up as the **single source of truth** for HF-on-GCP users.
  - Users are invited to open issues or contribute examples, and to share which specific recipes they need (datasets on GCS, multi-node training, RAG, etc.).

This implies a “three-layer” knowledge structure:

1. **Vertex AI docs** – authoritative on platform semantics and guarantees.  
2. **Hugging Face GCP docs + Google-Cloud-Containers** – authoritative on HF-specific containers and patterns for Vertex AI/GKE/Cloud Run.  
3. **Forum / GitHub Issues** – high-signal discussion of edge cases, failure modes, and missing docs.

This KB is meant to sit between layers 2 and 3, summarizing and normalizing the patterns that appear repeatedly in forum threads and GitHub issues.

### 3.2 TRL CLI, local datasets, and “three moving targets”

A detailed answer in the “TRL CLI parameter for local dataset” thread explains why users often struggle when combining:

1. **TRL CLI** and its evolving parameters / YAML schema.  
2. **Hugging Face `datasets`** and its support for local files, GCS (`gs://`) paths, and FUSE-mounted paths.  
3. **Vertex AI custom training** job wiring (commands, env vars, Cloud Storage FUSE).  

Common symptoms:

- `unrecognized arguments` from TRL CLI when config keys do not match the installed TRL version.
- Confusing “dataset not found” errors caused by mixing `/gcs/...` paths (FUSE) with `gs://...` URIs (`gcsfs`), or using the wrong prefix inside the container.
- YAML config keys for TRL not matching the dataclasses in the installed version (e.g. `TrlParser` + `HfArgumentParser` changes).

The answer highlights two groups of users who *do* have stable setups:

- Those who copy the official Vertex AI examples **almost verbatim**, including versions and path conventions.
- Those who **skip the TRL CLI entirely**, and instead use a plain Python script with `SFTTrainer` (treating Vertex AI as “just a remote Linux box with GPUs”).

This leads to the first major best-practice recommendation of this KB:

> **If you already have a working local script using `SFTTrainer`, prefer running that script directly in a PyTorch Training DLC on Vertex AI instead of trying to shoehorn everything into the TRL CLI.**

We will revisit this in the implementation patterns section.

### 3.3 Gated models and environment variables (`HF_TOKEN`)

Another recurring topic is how to access **gated models** (e.g. Gemma/TxGemma) from within Vertex AI training jobs and TGI/TEI deployments.

Community and docs converge on this pattern:

- Set a **Hugging Face Hub token** as an environment variable in the Vertex job or model deployment:
  - For training jobs: `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in `environment_variables` on `job.run(...)`.
  - For TGI/TEI DLC deployments: `HUGGING_FACE_HUB_TOKEN` in `serving_container_environment_variables` when calling `aiplatform.Model.upload`.
- Let `transformers` and `huggingface_hub` pick up that token automatically; you typically do not need to pass it explicitly to `from_pretrained` unless you want to override behaviour.

Security and ops best practice:

- Store the token in **Google Secret Manager** and inject it at runtime into Vertex jobs/endpoints via environment variables.
- Avoid hard-coding tokens in YAML config files, notebooks, or source control.

## 4. Implementation patterns for Vertex AI + Hugging Face

This section converts the above concepts into concrete, copy-pastable patterns.

### 4.1 Core building blocks

#### 4.1.1 Hugging Face DLCs on Vertex AI

From the Google-Cloud-Containers README and HF docs, you get a consistent set of DLCs:

- **Training**:
  - PyTorch Training DLC (GPU, TPU) – includes `transformers`, `datasets`, `trl`, and typical ML stack.
- **Inference**:
  - PyTorch Inference DLC (CPU/GPU) – general-purpose inference.
  - TGI DLC (GPU/TPU) – optimized text generation serving.
  - TEI DLC (CPU/GPU) – optimized text-embedding serving.

The README lists example URIs such as (example only, always check the repo for latest versions):

- `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-...`
- `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-...`
- `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-...`

These containers are published and updated in **Google Cloud’s Artifact Registry**, and you reference them from Vertex AI jobs and deployments via `container_uri` / `serving_container_image_uri`.

#### 4.1.2 Vertex AI SDK and GCS

For Python-based orchestration, you will typically use:

- `google-cloud-aiplatform` for training and deployment.
- GCS buckets for:
  - Dataset storage (`gs://...` or `/gcs/...`).
  - Training outputs and checkpoints (`base_output_dir`, `output_dir`).

In serverless training jobs, Vertex AI automatically **mounts your GCS bucket under `/gcs/<BUCKET>`** using Cloud Storage FUSE. This is important when referencing paths inside containers.

### 4.2 Pattern A – TRL CLI–based SFT training on Vertex AI

This pattern matches the official Vertex AI + TRL examples (Mistral SFT, Gemma LoRA SFT). It is suitable if:

- You want a CLI-driven workflow, and
- You are comfortable aligning your TRL version with example docs and YAML schemas.

**High-level steps:**

1. Put your YAML configuration (TRL SFT config) and dataset files in GCS.  
2. Launch a **CustomContainerTrainingJob** that uses the **PyTorch Training DLC** as `container_uri`.  
3. Set the job `command` to a shell wrapper that runs `trl sft "$@"`.  
4. Pass `--config=/gcs/<BUCKET>/configs/sft_config.yaml` and other CLI args via `args`.  
5. Pass `HF_TOKEN` and other environment variables in `environment_variables`.  

Skeleton (job launcher):

```python
from google.cloud import aiplatform
import os

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ["LOCATION"]
BUCKET_URI = os.environ["BUCKET_URI"]
CONTAINER_URI = os.environ["CONTAINER_URI"]  # HF PyTorch Training DLC
HF_TOKEN = os.environ["HF_TOKEN"]

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET_URI,
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="mistral-sft-vertex",
    container_uri=CONTAINER_URI,
    command=[
        "sh",
        "-c",
        'exec trl sft "$@"',
        "--",
    ],
)

args = [
    "--config=/gcs/my-bucket/configs/sft_config.yaml",
]

job.run(
    args=args,
    replica_count=1,
    machine_type="g2-standard-24",
    accelerator_type="NVIDIA_L4",
    accelerator_count=2,
    base_output_dir=f"{BUCKET_URI}/outputs/mistral-sft",
    environment_variables={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/root/.cache/huggingface",
        "TRL_USE_RICH": "0",
    },
)
```

**Best-practice notes:**

- Use the **same TRL version** as the official example you follow (see the DLC tag and example notebook). This avoids YAML/key mismatches.
- In YAML, pick **one dataset specification method**:
  - Either `dataset_name` for Hub datasets, or
  - `datasets:` with `data_files` for local/GCS JSON/Parquet files.
- When using FUSE, always use `/gcs/<BUCKET>/...` paths inside the container (not `gs://`), and ensure those paths match what you see when debugging inside a test job.

### 4.3 Pattern B – Direct Python script using `SFTTrainer`

This pattern implements the “treat Vertex AI as a remote Linux box” advice from the TRL CLI discussion. It trades some declarative convenience for simpler debugging and version control.

**High-level idea:**

- Write a `train.py` script that does everything (arg parsing, dataset loading, model/tokenizer, `SFTTrainer`).
- Use the **same PyTorch Training DLC** as in Pattern A, but set `command=["python", "train.py"]` and pass simple CLI args.

Example `train.py` skeleton:

```python
# train.py
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/txgemma-2b-predict")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Dataset from GCS (either gs://... with gcsfs or /gcs/... via FUSE)
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.eval_file,
        },
    )

    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=1024,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
```

Vertex AI launcher:

```python
from google.cloud import aiplatform
import os

aiplatform.init(
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION"),
    staging_bucket=os.getenv("BUCKET_URI"),
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="txgemma-sft-direct",
    container_uri=os.getenv("CONTAINER_URI"),  # HF PyTorch Training DLC
    command=["python", "train.py"],
)

job.run(
    args=[
        "--model_id=google/txgemma-2b-predict",
        "--train_file=gs://my-bucket/data/train.jsonl",
        "--eval_file=gs://my-bucket/data/eval.jsonl",
        "--output_dir=/gcs/my-bucket/outputs/txgemma-sft",
    ],
    environment_variables={
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    },
    machine_type="g2-standard-12",
    accelerator_type="NVIDIA_L4",
    accelerator_count=1,
)
```

**When to prefer this pattern:**

- You already have a working local training script.
- You want to minimize the number of abstraction layers (no TRL CLI parser, no YAML intricacies).
- You need complex logic that is easier to express in Python than in CLI flags.

### 4.4 Data handling patterns (Hub, GCS + FUSE, GCS + `gcsfs`)

When using Hugging Face `datasets` in Vertex AI, you generally have three options:

1. **Datasets from Hugging Face Hub**

   ```python
   dataset = load_dataset("some/dataset")
   ```

   - Easiest path, especially for examples.
   - Works in both TRL CLI and Python-script setups.

2. **Datasets in GCS with Cloud Storage FUSE (`/gcs/...`)**

   Inside serverless training containers, Vertex AI mounts buckets at `/gcs/<BUCKET>`. Use those paths as if they were local files:

   ```python
   dataset = load_dataset(
       "json",
       data_files={
           "train": "/gcs/my-bucket/data/train.jsonl",
           "validation": "/gcs/my-bucket/data/eval.jsonl",
       },
   )
   ```

   This is what the official Vertex + TRL examples typically assume.

3. **Datasets in GCS via `gs://...` and `gcsfs`**

   Install `gcsfs` in the DLC (e.g. via `pip install gcsfs` in a startup script or custom Dockerfile), then use `gs://` URIs directly:

   ```python
   dataset = load_dataset(
       "json",
       data_files={
           "train": "gs://my-bucket/data/train.jsonl",
           "validation": "gs://my-bucket/data/eval.jsonl",
       },
   )
   ```

   `datasets` will use fsspec + `gcsfs` to stream data from GCS without going through FUSE.

**KB recommendation:**

- For **TRL CLI + YAML** flows, stick to `/gcs/...` paths to stay close to official examples.
- For **Python script** flows, choose whichever is simplest for your team:
  - FUSE (`/gcs`) for fewer dependencies.
  - `gcsfs` if you want direct `gs://` semantics and potentially better behavior with many small files.

### 4.5 Tokens and gated models

Standard pattern:

- Put your HF token (`hf_...`) in an environment variable accessible to the container:
  - For training jobs: `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`).
  - For TGI/TEI endpoints: `HUGGING_FACE_HUB_TOKEN` in serving env.
- Avoid embedding tokens in code, notebooks, or YAML configs; prefer **Secret Manager + env var injection**.

Example for TGI DLC deployment:

```python
from google.cloud import aiplatform
from huggingface_hub import get_token
import os

aiplatform.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))

model = aiplatform.Model.upload(
    display_name="google--gemma-7b-it",
    serving_container_image_uri=os.getenv("CONTAINER_URI"),  # TGI DLC URI
    serving_container_environment_variables={
        "MODEL_ID": "google/gemma-7b-it",
        "NUM_SHARD": "1",
        "MAX_INPUT_TOKENS": "512",
        "MAX_TOTAL_TOKENS": "1024",
        "MAX_BATCH_PREFILL_TOKENS": "1512",
        "HUGGING_FACE_HUB_TOKEN": get_token(),
    },
    serving_container_ports=[8080],
)
```

### 4.6 Inference patterns with TGI / TEI DLCs

The TGI and TEI DLCs provide production-grade serving for LLMs and embedding models, respectively.

**Typical deployment flow on Vertex AI:**

1. Choose the **appropriate DLC URI** from the README or docs (GPU or CPU, cuDNN/CUDA versions, etc.).  
2. Call `aiplatform.Model.upload` with:
   - `serving_container_image_uri` = DLC URI.
   - `serving_container_environment_variables` containing `MODEL_ID` and other runtime config.
   - Optional `HUGGING_FACE_HUB_TOKEN` if the model is gated.
3. Create an endpoint and `deploy` the model with a chosen machine type and accelerators.

For TEI, the pattern is similar but with environment variables and models appropriate for embeddings (e.g. BGE, GTE, etc.).

### 4.7 Model Garden integration

If you are using **Hugging Face open models exposed in Model Garden**, you can:

- Discover them via the Vertex AI Studio UI or `vertexai.model_garden` in Python.
- Deploy them directly via the Model Garden API, sometimes with custom weights (self-deployed models).
- Combine them with Gemini models and other Google services (e.g., for RAG, multi-modal flows, or tool-calling agents).

The choice between:

- **Model Garden** (managed open models) and  
- **DLC-based custom deployments**

depends on:

- How much control you need (custom training, custom inference logic).  
- Whether the target model is already available as a managed open model.  
- Organizational constraints around governance and infrastructure control.

## 5. Operational advice, limitations, and common pitfalls

### 5.1 Path confusion (`/gcs` vs `gs://`)

- Inside serverless training containers, GCS is mounted under `/gcs/<BUCKET>`.  
- `gs://` URIs are supported by libraries that understand fsspec/`gcsfs` (like `datasets`) if `gcsfs` is installed.  
- Mixing these two conventions in TRL YAML or scripts is a common source of errors; pick **one convention per project** and stick to it.

### 5.2 TRL version drift and YAML schemas

- TRL’s CLI and YAML mapping (`TrlParser`, `HfArgumentParser`) evolve; parameter names can change across versions.
- DLC images pin specific TRL versions. If you read docs for a different version, you can get `unrecognized arguments` or missing fields.
- To avoid this:
  - Check the TRL version inside the container (`python -c "import trl; print(trl.__version__)"`).  
  - Align your YAML and CLI flags with the docs for that exact version.
  - Prefer smaller YAMLs and explicitly documented keys; avoid deep nesting unless examples show it.

### 5.3 Complexity of multi-layer stacks

A typical Vertex+HF+TRL stack can look like:

- Vertex AI → HF DLC → TRL CLI → YAML → `datasets` → GCS (FUSE or `gcsfs`).

Every extra layer introduces its own configuration model and error messages. The more layers you combine, the more ambiguous debugging becomes.

**KB recommendation:**

- For **complex projects**, favor the **direct Python script (Pattern B)** over stacking TRL CLI + YAML + custom dataset wiring.
- Use TRL CLI mainly when copying official examples or when you need a quick, simple run and are okay staying on the “happy path”.

### 5.4 Performance considerations (GCS and FUSE)

- Cloud Storage FUSE is convenient but can be slower for many small random reads.
- Where possible:
  - Use fewer, larger files (e.g. sharded JSON/Parquet).  
  - Consider pre-processing/packing data before training.  
  - If you see I/O bottlenecks, experiment with `gs://` + `gcsfs` streaming instead of FUSE.

### 5.5 Cost and resource selection

- Match model size and training regime to appropriate accelerator types:
  - L4 / T4 for smaller models or experimentation.
  - A100 / H100 / TPU for large models or heavy workloads.
- Start with minimal GPU counts and scale up after profiling; serverless training makes it easy to adjust and rerun.
- For inference, size your endpoints based on latency and throughput requirements; use Vertex AI autoscaling and monitoring.

## 6. How to extend this knowledge base (and where to contribute)

Given the explicit intent to build an extended KB for **Vertex AI Training + Hugging Face**, the natural places to extend and formalize this document are:

1. **Google-Cloud-Containers repository**

   - Add new examples under `examples/vertex-ai` (notebooks or Python scripts) that demonstrate:
     - More TRL / SFT patterns (e.g. QLoRA, multi-dataset training).  
     - Evaluation workflows on Vertex AI (possibly using Gemini for eval).  
     - RAG or multi-modal pipelines tying together Vertex AI, BigQuery, GCS, and HF models.
   - Improve docs under `docs/` to reflect new patterns and community findings.

2. **Hugging Face Google Cloud docs**

   - Propose new sections or updates that:
     - Clarify TRL CLI vs SFTTrainer patterns.  
     - Document data-handling patterns (`/gcs` vs `gs://`, FUSE vs `gcsfs`).  
     - Give “decision trees” for when to use Model Garden vs DLCs.

3. **Hugging Face forums and GitHub Issues**

   - Use forum threads to:
     - Share new recipes or pitfall post-mortems.  
     - Ask focused questions about new features (e.g. new DLC tags, TPU support).
   - Use GitHub issues on `Google-Cloud-Containers` when:
     - Containers have bugs or version mismatches.  
     - Examples are outdated or break with new TRL/Transformers releases.

This KB can serve as a **living outline** for that work: sections 4–5 can be mapped fairly directly to new examples and doc pages.

## 7. References / links

**Core docs and product pages**  

- Vertex AI docs index:  
  [https://docs.cloud.google.com/vertex-ai/docs](https://docs.cloud.google.com/vertex-ai/docs)
- Vertex AI product page:  
  [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)
- Use Hugging Face models in Vertex AI (Model Garden):  
  [https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models)

**Hugging Face on Google Cloud**  

- Hugging Face on Google Cloud docs:  
  [https://huggingface.co/docs/google-cloud/en/index](https://huggingface.co/docs/google-cloud/en/index)
- Other resources (examples index):  
  [https://huggingface.co/docs/google-cloud/en/resources](https://huggingface.co/docs/google-cloud/en/resources)
- Google-Cloud-Containers repo:  
  [https://github.com/huggingface/Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers)
- Google Cloud partnership org on HF Hub:  
  [https://huggingface.co/google-cloud-partnership](https://huggingface.co/google-cloud-partnership)

**Vertex AI + HF examples**  

- Fine-tune Gemma 2B with PyTorch Training DLC using SFT + LoRA on Vertex AI (TRL CLI):  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-trl-lora-sft-fine-tuning-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-trl-lora-sft-fine-tuning-on-vertex-ai)
- Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT on Vertex AI:  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-trl-full-sft-fine-tuning-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-trl-full-sft-fine-tuning-on-vertex-ai)
- Deploy BERT models with PyTorch Inference DLC on Vertex AI:  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-bert-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-bert-on-vertex-ai)
- Deploy embedding models with TEI DLC on Vertex AI:  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-embedding-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-embedding-on-vertex-ai)
- Deploy Gemma 7B with TGI DLC on Vertex AI (from Hub and from GCS):  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-gemma-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-gemma-on-vertex-ai)  
  [https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-gemma-from-gcs-on-vertex-ai](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-deploy-gemma-from-gcs-on-vertex-ai)

**Blog and community**  

- Deploying Hub models in Vertex AI (HF blog):  
  [https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai)
- KB initiative: Need to build extended knowledge base for VertexAI Training + HF toolbox:  
  [https://discuss.huggingface.co/t/need-to-build-extended-knowledge-base-about-hugging-face-toolbox-best-practices-and-recipes-in-the-vertexai-training-environment/171168](https://discuss.huggingface.co/t/need-to-build-extended-knowledge-base-about-hugging-face-toolbox-best-practices-and-recipes-in-the-vertexai-training-environment/171168)
- TRL CLI parameter for local dataset (Vertex AI context):  
  [https://discuss.huggingface.co/t/trl-cli-parameter-for-local-dataset/171058](https://discuss.huggingface.co/t/trl-cli-parameter-for-local-dataset/171058)
