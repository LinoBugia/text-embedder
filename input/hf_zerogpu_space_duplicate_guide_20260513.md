# Hugging Face ZeroGPU Space Duplication and GPU Migration Guide

## 0. How to Use This Guide

This guide is for one practical situation:

> You have an existing **Hugging Face ZeroGPU Space**, and you want your own working copy.

Most of the time, you are choosing one of two routes:

1. **ZeroGPU Space → standard GPU Space**
2. **ZeroGPU Space → ZeroGPU Space**

The first route is the main focus. It is more than a duplicate button plus a hardware change. It is a small runtime migration: you move from ZeroGPU's request-based GPU model to a standard Space where one selected accelerator belongs to the running Space.

The second route is usually simpler. You keep the ZeroGPU model, but you still need to recreate the parts that make the copy behave like the source: hardware selection, secrets, variables, README configuration, dependency versions, and the path you use to test the app.

You don't need to read this guide linearly. A practical route is:

1. use the route map below
2. capture the source evidence
3. duplicate or rebuild the target Space
4. restore the runtime settings
5. run the smallest hosted smoke test
6. only then tune hardware, dependencies, and API paths

This guide is not a general Hugging Face Spaces tutorial. It assumes you already have a ZeroGPU Space in mind and want a safe way to copy or migrate it.

## 0A. Using This Guide with an LLM or Agent

This guide is written for humans, but it should also be useful as context for ChatGPT, coding agents, and other assistants.

If you use it that way, the assistant should slow down just enough to avoid guessing. The important move is not to produce a clever fix first. The important move is to identify the route and collect the evidence.

A good assistant should:

1. **Classify the route first.**
   Is this ZeroGPU → standard GPU, ZeroGPU → ZeroGPU, a clean rebuild, a local Docker preflight, or still unknown?

2. **Build an evidence bundle.**
   The minimum useful bundle is: source Space URL, target route, README YAML, dependency files, logs, hardware/runtime state, secrets/variables inventory, and the smallest failing action.

3. **Keep facts and guesses separate.**
   Don't infer that the source Space used A100, H200, or RTX Pro 6000 Blackwell unless the evidence says so. If the hardware generation is unclear, mark it as **unknown** and test compatibility directly.

4. **Treat secrets as missing until proven otherwise.**
   `HF_TOKEN`, model access tokens, API keys, OAuth secrets, and private dataset/model access often explain why a duplicate behaves differently.

5. **Prefer reversible changes.**
   Keep the source Space intact. Prefer private duplicates, snapshots, small hardware/config changes, and clean rollback points.

6. **Return a phased plan.**
   Good answers usually separate:
   - source evidence capture
   - duplicate or clean target creation
   - runtime contract reconstruction
   - smallest hosted test
   - optional local Docker preflight
   - expansion to larger workloads

The guide repeats a few ideas on purpose. That is useful when a reader, LLM, or agent lands in the middle of the file and only sees one section.

### Good answer shape

For most troubleshooting or migration questions, an assistant answer should look like this:

```text
Route:
- ZeroGPU → standard GPU / ZeroGPU → ZeroGPU / clean rebuild / local Docker preflight / unknown

Known facts:
- ...

Missing evidence:
- ...

Likely failure class:
- duplicate/runtime contract/hardware/dependency/secrets/API/local-vs-hosted

Next safest actions:
1. ...
2. ...
3. ...

Do not do yet:
- ...
```

### When evidence is missing

If the logs, README YAML, dependency files, or hardware/runtime state are missing, don't invent them. Give the conservative path and label the unknowns.

## 0B. Quick Route Map

Use this table when you enter the guide from a concrete question.

| User goal or symptom | Start with | Then use |
|---|---|---|
| “I want my own copy of this ZeroGPU Space” | 4. What the Duplicate Workflow Copies — and What It Does Not | 5. Source Evidence Capture, 15. Migration Checklists |
| “I want to move this ZeroGPU Space to a normal GPU” | 6. Target A: ZeroGPU → Standard GPU Space | 10. Dependencies and Torch/CUDA Compatibility, 14. Symptom-First Playbooks |
| “I only want another ZeroGPU copy” | 7. Target B: ZeroGPU → ZeroGPU Space | 11. Secrets, Variables, Tokens, and Private Assets, 13. API, Client, and Frontend Paths |
| “The duplicate built but runs differently” | 4.5 Why “same files” is not enough | 14.7 Output differs from the source |
| “It starts on CPU or says no GPU” | 14.1 Duplicate lands on CPU | 12. Programmatic Duplication and Runtime Management |
| “It fails with CUDA or torch errors” | 10. Dependencies and Torch/CUDA Compatibility | 14.3 `CUDA error: no kernel image is available for execution on the device` |
| “Browser works but API/client fails” | 13. API, Client, and Frontend Paths | 14.6 Browser works, API fails |
| “I want to test locally first” | 12B. Local Docker Preflight Checks | 17.6 Local Docker smoke test commands |
| “I need a checklist” | 15. Migration Checklists | 17B. More Checklists for Thick-but-Safe Migration |
| “I need snippets” | 17. Practical Snippets | 12. Programmatic Duplication and Runtime Management |

---

## 0C. Terms Used in This Guide

These terms keep the guide readable. They also help an assistant avoid mixing up the source Space, the target Space, and the runtime you are trying to recreate.

| Term | Meaning in this guide |
|---|---|
| Source Space | The original ZeroGPU Space you are copying from |
| Target Space | The new Space you own or control |
| Duplicate | A repo-level copy operation; not a full runtime clone |
| Runtime contract | Hardware, SDK, Python, dependency pins, secrets, variables, storage, request path, startup behavior, and API/frontend assumptions |
| ZeroGPU → standard GPU | Migration from ZeroGPU's request-based GPU model to an assigned GPU Space hardware model |
| ZeroGPU → ZeroGPU | A copy that keeps the ZeroGPU runtime model |
| Standard GPU | A normal paid/assigned GPU Space tier, not ZeroGPU |
| Hardware generation unknown | The safe default when logs don't prove A100, H200, Blackwell, or another GPU generation |
| Hosted smoke test | The smallest meaningful test inside the Hugging Face hosted Space |
| Local Docker preflight | A local container test that can catch dependency/startup mistakes but cannot fully emulate ZeroGPU scheduling, quota, or Hugging Face-hosted runtime behavior |
| Evidence bundle | The minimum facts needed before changing code or hardware |

---

## 1. The Short Version

If you only remember one thing, remember this: duplicating a Space copies files, but it does not automatically recreate the runtime that made the original work.

If you want the safest path, do this:

1. Capture evidence from the source Space before changing anything.
2. Decide the target:
   - **standard GPU** if you want a more conventional always-assigned GPU runtime
   - **ZeroGPU** if you want to keep ZeroGPU's shared, quota-based runtime model
3. Duplicate the Space privately.
4. Recreate secrets manually.
5. Confirm the actual assigned hardware, not only the requested hardware.
6. Pin or update the dependency stack before testing large workloads.
7. If the target is a standard GPU Space, use a local Docker smoke test when practical.
8. Run the smallest useful hosted test first.
9. Only then adjust hardware size, startup behavior, API access, or frontend paths.

The most important mental model is:

> A duplicate copies a repo. It does not automatically recreate the runtime contract.

---

## 2. Why ZeroGPU Duplication Is Special

A normal Space can often be duplicated and then debugged as a familiar app deployment problem. A ZeroGPU Space adds another layer.

ZeroGPU is not simply “a free version of a normal GPU Space.” It is a separate runtime model for Gradio Spaces. Current Hugging Face documentation describes ZeroGPU as a dynamic GPU allocation system with two sizes:

| ZeroGPU size | Backing hardware | VRAM | Quota cost |
|---|---:|---:|---:|
| `large` | Half NVIDIA RTX Pro 6000 Blackwell | 48GB | 1× |
| `xlarge` | Full NVIDIA RTX Pro 6000 Blackwell | 96GB | 2× |

ZeroGPU is currently Gradio-only. It is designed to be compatible with most PyTorch-based GPU Spaces, especially high-level Hugging Face libraries such as `transformers` and `diffusers`, but it can still have limited compatibility compared with standard GPU Spaces.

That matters because a Space that works on ZeroGPU may depend on assumptions such as:

- GPU work runs inside functions decorated with `@spaces.GPU`.
- The GPU appears only when the decorated work is called.
- Request identity and quota may matter.
- Supported PyTorch versions are bounded by the ZeroGPU validator.
- Older code or older documentation may have assumed a different backing GPU generation.

When you migrate from ZeroGPU to a standard GPU Space, you are changing those assumptions.

---

## 3. Decide the Target First

Start here before changing code. Many failed migrations come from treating two different routes as the same task.

Before you duplicate, decide what you are actually trying to achieve.

### Target A — ZeroGPU → standard GPU Space

Choose this path when:

- you want a conventional GPU assigned to the Space runtime
- you want to avoid ZeroGPU quota behavior
- the app needs longer-running jobs
- the app is sensitive to ZeroGPU scheduling, request identity, or compatibility limits
- you want to migrate toward a paid or more predictable deployment model

This path usually requires more work because you are not only duplicating the repo. You are changing the runtime model.

### Target B — ZeroGPU → ZeroGPU Space

Choose this path when:

- you want your own private or modified copy but still want ZeroGPU behavior
- the source app is already ZeroGPU-shaped
- the workload fits ZeroGPU duration and quota limits
- you want to preserve the `@spaces.GPU` execution pattern

This path is closer to a faithful duplication, but it can still fail because secrets, variables, dependency resolution, hardware size, and request identity may not match the source.

### Don't mix both targets in one first test

Don't duplicate and migrate at the same time unless you explicitly accept the ambiguity.

Bad first test:

- duplicate the Space
- change hardware
- change dependencies
- change Gradio version
- change frontend path
- test with a large workload

Good first test:

- duplicate privately
- recreate secrets
- set one intended hardware target
- keep the app code mostly unchanged
- run the smallest useful workload

---

## 4. What the Duplicate Workflow Copies — and What It Does Not

This section is deliberately detailed because many failed migrations start with a wrong mental model of **Duplicate this Space**.

The duplicate workflow is good at copying the repository. It is not a guarantee that the new Space will run under the same runtime contract. A duplicated Space can have the same files and still differ in hardware, secrets, environment variables, package resolution, startup state, storage expectations, and request path.

### 4.1 What duplication usually copies well

Duplication is primarily a repository-level operation. It tends to preserve:

- the Space repository files
- the app code
- `README.md` metadata
- dependency files committed to the repo
- examples and assets stored in Git or LFS
- for Spaces, repository history can come along as well

That means a duplicate is usually a good starting point. It is not an environment snapshot.

Useful references:

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [More ways to create Spaces](https://huggingface.co/docs/hub/spaces-more-ways-to-create)

### 4.2 What duplication does not automatically recreate

Treat these as separate runtime contract items:

| Runtime item | Why it matters | What to do after duplication |
|---|---|---|
| Hardware | UI duplicates can fall back to free CPU unless another hardware option is chosen | explicitly select ZeroGPU or request the standard GPU target |
| Secrets | secrets are not simply safe-to-copy ambient state | recreate `HF_TOKEN`, API keys, OAuth secrets, and private service credentials |
| Public variables | public variables may be auto-populated, but still need review | compare names and values against source expectations |
| Storage | cached files or persistent storage may not match the source | recreate downloads, use `preload_from_hub`, or attach storage deliberately |
| Build timing | a duplicate rebuilds now, under current package resolution | pin important versions before assuming app logic is broken |
| Visibility | duplicates are commonly private by default | test browser and API paths with the right token model |
| Request identity | ZeroGPU quota can depend on request headers | test the normal HF page first before custom frontends or app-to-app calls |

Useful references:

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [User access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Using ZeroGPU Spaces with the Clients](https://www.gradio.app/docs/python-client/using-zero-gpu-spaces)

### 4.3 UI duplication: what to check in the dialog

When using the Hugging Face web UI, use the duplicate dialog as a runtime checklist, not only as a copy button. Check:

- owner and destination name
- visibility
- hardware choice
- storage choice if offered
- variables
- warning about secrets
- whether the source repo is allowed to be duplicated

For the first attempt, prefer a conservative private duplicate. Don't change five things at once. If the source is ZeroGPU and your goal is standard GPU migration, still capture the source behavior first, then move hardware deliberately. If the source is ZeroGPU and your goal is a faithful copy, make sure the duplicate is actually on ZeroGPU before interpreting any failure.

### 4.4 Programmatic duplication: when it helps

Programmatic duplication is useful when you want reproducibility, repeated migration tests, or a clean audit trail. `huggingface_hub` exposes Space runtime controls including duplication, hardware requests, secrets, restart, and runtime inspection.

A safe programmatic flow is:

1. duplicate the Space privately
2. add secrets and variables
3. request target hardware
4. inspect actual runtime state
5. restart only when needed
6. fetch logs and test the smallest workload

Useful references:

- [Managing your Space runtime](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime)
- [Manage your Space](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces)
- [HfApi reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)

### 4.5 Why “same files” is not enough

A duplicate can differ from the source even when every visible Python file matches. Typical reasons:

- the source container was warm, while the duplicate starts cold
- the source had cached models or generated assets
- the source had secrets that are missing from the duplicate
- the source resolved older dependency versions
- the source was running under a different GPU generation
- the source was tested through the normal HF page, while the duplicate is tested through a custom frontend or API path

The practical test is not “do the files match?” The practical test is:

> Can this new Space rebuild from a clean state and run the smallest meaningful workload under the target runtime?

---

## 4B. Before You Duplicate: Can This Space Be Copied Cleanly?

Do this check before spending time on app code. Some failures are not caused by ZeroGPU, CUDA, or Gradio. They happen because the source repository or destination account cannot support the duplicate you are trying to create.

### 4B.1 Duplication blockers and friction points

Check these first:

- whether the source Space allows duplication
- whether the source depends on gated models or datasets
- whether the destination account has access to every private or gated dependency
- whether the source repo has unusually large Git/LFS history
- whether you are moving between personal and organization ownership
- whether the destination org has token policies or storage restrictions

Useful references:

- [Repositories next steps](https://huggingface.co/docs/hub/repositories-next-steps)
- [Create and manage a repository](https://huggingface.co/docs/huggingface_hub/en/guides/repository)
- [Storage limits](https://huggingface.co/docs/hub/storage-limits)
- [Repository settings](https://huggingface.co/docs/hub/repositories-settings)

### 4B.2 Repository history can be part of the problem

A Space can fail to duplicate, build, or update cleanly because old repository history carries too much baggage. This is separate from runtime storage. Deleting a large file from the current tree does not necessarily remove it from Git history. If the Space has accumulated large artifacts, generated outputs, checkpoints, or repeated binary commits, treat repository health as part of the migration.

Practical checks:

```bash
git clone https://huggingface.co/spaces/OWNER/SPACE
cd SPACE
git count-objects -vH
git lfs ls-files
```

If the repository itself is heavy, don't solve it by adding more runtime fixes. Move large model artifacts to a model repo, datasets to a dataset repo, or persistent/object storage where appropriate.

### 4B.3 When a fresh Space is cleaner than a duplicate

Sometimes the safest migration is not a literal duplicate. A clean new Space may be better when:

- the old Space has messy Git history
- the old Space includes generated files that should not be source files
- the old Space mixes app code, model weights, outputs, and temporary assets
- the goal is standard GPU migration rather than faithful ZeroGPU preservation
- you need to split private assets from public app code

In that case, use the source Space as evidence and rebuild the target Space deliberately. The goal is not to preserve every accident of the source. The goal is to preserve the working behavior and encode the new runtime contract clearly.

### 4B.4 Clean target layout for migration

A good migrated Space usually separates concerns like this:

```text
Space repo:
- app.py
- README.md with Space YAML
- requirements.txt
- packages.txt if OS packages are needed
- small static assets and examples

Model repo or dataset repo:
- large model weights
- LoRA/adapters/checkpoints
- large example data
- reusable artifacts

Secrets / variables:
- HF_TOKEN
- private model identifiers
- external API keys
- deployment mode flags
```

That layout makes future rebuilds, duplicates, and hardware changes easier to reason about.

---

## 5. Source Evidence Capture

Before you click Duplicate, capture evidence from the source Space.

Don't rely on memory. You will need this information when the duplicate behaves differently.

### 5.1 Space metadata

Record:

- source Space ID
- visibility
- SDK type
- whether it is a Gradio Space
- whether it is actually using ZeroGPU hardware
- whether the app uses OAuth or private/gated resources

### 5.2 README YAML

Read the YAML block at the top of `README.md`. Important fields include:

- `sdk`
- `python_version`
- `sdk_version`
- `app_file`
- `suggested_hardware`
- `startup_duration_timeout`
- `preload_from_hub`
- `models`
- `datasets`
- OAuth-related fields if present

The README YAML is not decorative. It is part of the runtime contract.

### 5.3 Dependency files

Capture:

- `requirements.txt`
- `pre-requirements.txt`
- `packages.txt`
- Dockerfile, if present
- any lock files or install scripts
- any runtime downloads in `app.py`

### 5.4 ZeroGPU-specific code

Search for:

- `import spaces`
- `@spaces.GPU`
- `duration=`
- `size="xlarge"` or `size='xlarge'`
- `torch.cuda` at import time
- `model.to("cuda")` at import time
- client calls to another Space
- custom frontends or proxy layers

### 5.5 Hardware and CUDA evidence

If you can see logs or run diagnostics, capture:

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
```

This is especially useful now that older ZeroGPU assumptions may have referred to A100 or H200, while current ZeroGPU documentation points to RTX Pro 6000 Blackwell-backed sizes.

### 5.6 Secrets and variables inventory

You usually cannot read secret values from the source, but you can identify what the app expects.

Search for environment names such as:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `OPENAI_API_KEY`
- model provider keys
- private endpoint URLs
- OAuth secrets
- private model or dataset identifiers

A missing secret can make the duplicate look like a model failure even when the app code is fine.

---

## 5A. A Practical Source Snapshot Template

Before changing hardware, copy these into a scratch note. This is especially important when the source Space has been stable for a long time and nobody remembers which GPU generation or package set it originally assumed.

### 5A.1 Minimum human-readable snapshot

```text
source_space: owner/name
source_visibility: public / private / org
source_sdk: gradio / docker / static
source_hardware_observed: ZeroGPU large / ZeroGPU xlarge / unknown
source_target_goal: standard GPU / ZeroGPU copy
first_known_good_date: unknown / date
main_app_file: app.py / other
uses_spaces_gpu_decorator: yes / no / unknown
uses_custom_frontend: yes / no / unknown
uses_space_to_space_calls: yes / no / unknown
requires_private_models_or_datasets: yes / no / unknown
requires_external_api_keys: yes / no / unknown
```

### 5A.2 Runtime evidence snippet

Use something like this inside a temporary diagnostic path or a local reproduction when safe:

```python
import os
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

for name in ["SPACE_ID", "SPACE_HOST", "SPACE_AUTHOR_NAME", "SPACE_REPO_NAME"]:
    print(name, os.environ.get(name))
```

Don't publish secrets, tokens, private URLs, or full environment dumps. Capture only what helps classify the migration.

### 5A.3 README YAML snapshot

Copy the top YAML block from `README.md`, especially:

- `sdk`
- `sdk_version`
- `python_version`
- `app_file`
- `suggested_hardware`
- `suggested_storage`
- `startup_duration_timeout`
- `preload_from_hub`

Useful reference: [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference).

---

## 5B. Evidence Bundle Schema for Agents

If a human or agent is trying to diagnose a duplicated Space, collect the following bundle before changing code. This schema is intentionally plain Markdown/YAML-like text so it can be pasted into an issue, chat, or runbook.

```yaml
source_space:
  url: ""
  visibility: "public/private/unknown"
  source_goal: "copy as ZeroGPU / migrate to standard GPU / unknown"
  observed_working_path: "browser / API / custom frontend / app-to-app / unknown"

target_space:
  url: ""
  visibility: "public/private/unknown"
  intended_runtime: "standard GPU / ZeroGPU / CPU smoke test / unknown"
  selected_hardware: ""
  actual_runtime_state: ""
  storage: "none / persistent / unknown"

readme_yaml:
  sdk: ""
  sdk_version: ""
  python_version: ""
  app_file: ""
  suggested_hardware: ""
  startup_duration_timeout: ""
  preload_from_hub: ""

dependencies:
  requirements_txt: "present/missing"
  pyproject_toml: "present/missing"
  dockerfile: "present/missing"
  torch_version: ""
  cuda_wheel_or_index: ""
  gradio_version: ""
  spaces_package_version: ""

zerogpu_shape:
  uses_spaces_gpu_decorator: true
  gpu_work_inside_decorated_function: "yes/no/unknown"
  request_identity_needed: "yes/no/unknown"
  custom_frontend_or_client: "yes/no/unknown"

secrets_and_access:
  hf_token_present_in_target: "yes/no/unknown"
  private_or_gated_models: []
  external_api_keys_needed: []
  public_variables_checked: "yes/no"

failure:
  first_failing_action: ""
  error_message: ""
  build_logs_available: "yes/no"
  runtime_logs_available: "yes/no"
  smallest_reproduction: ""
```

### How to use the evidence bundle

- If `selected_hardware` and `actual_runtime_state` disagree, debug hardware selection before dependency code.
- If secrets are unknown, debug access before model code.
- If `sdk`, `sdk_version`, or `python_version` are missing, inspect README YAML before rewriting the app.
- If the app uses `@spaces.GPU`, decide whether the target keeps ZeroGPU semantics or moves to standard GPU semantics.
- If `torch_version` is old and Blackwell-backed ZeroGPU appears in logs, check CUDA wheel compatibility before assuming the model is broken.
- If local Docker passes but hosted Space fails, compare secrets, hardware, storage, startup timeout, and request path.

---

## 6. Target A: ZeroGPU → Standard GPU Space

This is the main migration path.

The key idea is:

> You are not trying to preserve ZeroGPU behavior. You are trying to preserve the app result while changing the runtime model.

### 6.1 What changes

When moving from ZeroGPU to a standard GPU Space, these assumptions change:

| Area | ZeroGPU Space | Standard GPU Space |
|---|---|---|
| GPU availability | Requested around decorated calls | Assigned to the Space runtime |
| Runtime model | Shared, dynamic, quota-based | Conventional assigned hardware |
| Main SDK constraint | Gradio-only | Depends on selected Space SDK |
| GPU wrapper | `@spaces.GPU` controls GPU allocation | Not the main scheduling mechanism |
| Quota behavior | Daily quota and queue priority matter | Billing and hardware assignment matter |
| Startup | Often designed to avoid eager GPU use | Eager model load may be possible but can slow startup |
| Request identity | `X-IP-Token` may matter | Usually not relevant unless calling ZeroGPU downstream |

### 6.2 First migration strategy

Start conservative:

1. Duplicate privately.
2. Request the standard GPU target.
3. Recreate secrets.
4. Keep the code as close as possible for the first build.
5. Run the smallest test.
6. Only then simplify ZeroGPU-specific code.

Don't remove all ZeroGPU code before the first baseline unless it clearly blocks startup. Current documentation says the `@spaces.GPU` decorator is designed to be effect-free in non-ZeroGPU environments, so it can often be left temporarily while you prove the duplicate works. After the app runs, you can clean up the code to match the standard GPU runtime.

### 6.3 Choosing standard GPU hardware

Don't choose hardware only by name. Choose it by workload shape.

A practical order:

1. Start with the smallest standard GPU that can load the model.
2. If the model fails at load time with out-of-memory errors, move up.
3. If the model loads but jobs are too slow, consider a faster tier.
4. If startup is too slow, separate model download from app health.
5. If the app needs multiple concurrent users, test concurrency instead of only single prompts.

README `suggested_hardware` can describe a recommended flavor for users duplicating the Space, but it does not automatically assign hardware.


### 6.3A How many standard GPU examples should this guide name?

Don't try to memorize every GPU flavor.

Hugging Face hardware choices change over time, and the list is large enough that a guide can become stale if it tries to explain every accelerator. For migration work, examples are useful only when they teach a decision pattern.

This guide therefore uses two preferred example families:

1. **Near-equivalent capacity or generation examples**
   These are useful when the source ZeroGPU Space probably relied on a particular VRAM class or GPU-generation behavior.

2. **Low-cost smoke-test examples**
   These are useful when you only need to prove that the duplicate builds, starts, authenticates, and reaches CUDA before you pay for a larger target.

Treat named hardware below as examples, not as permanent recommendations. Always check the current hardware list and pricing before choosing a target.

Useful references:

- [Spaces GPU hardware](https://huggingface.co/docs/hub/spaces-gpus)
- [Hugging Face pricing](https://huggingface.co/pricing)
- [Spaces Overview — hardware resources](https://huggingface.co/docs/hub/spaces-overview)

### 6.3B Practical hardware-selection buckets

Use buckets before names.

| Migration need | What you are trying to preserve | Useful example bucket | Why this bucket helps | What not to assume |
|---|---|---|---|---|
| Cheapest hosted GPU smoke test | Build, launch, CUDA visibility, auth path | T4 / L4 / small standard GPU tier | Cheap way to prove the duplicate is structurally alive | It may not match ZeroGPU memory or speed |
| 24 GB-class first real test | Many diffusion and medium model demos | L4 or A10G-class target | Often enough for a practical first GPU migration test | It is not equivalent to ZeroGPU `large` |
| Around 48 GB VRAM | Similar memory class to ZeroGPU `large` | 48 GB-class standard GPU or multi-GPU equivalent | Useful when source likely depended on larger memory | Architecture and CUDA capability may still differ |
| Around 80–96 GB VRAM | Large model or source used high ZeroGPU size | A100 80 GB or 96 GB-class aggregate target | Useful for capacity-preserving migration | Aggregate multi-GPU memory is not the same as one large GPU |
| Blackwell-specific compatibility concern | Source or current ZeroGPU shows Blackwell / `sm_120` behavior | Keep torch/CUDA compatibility check first | Captures the main failure class before hardware overfitting | A non-Blackwell normal GPU may hide this class of bug |

A good migration note is not “use GPU X.”

A good migration note is:

```text
The source looked like ZeroGPU large or xlarge.
The target test started on a cheap standard GPU only for build/auth validation.
The real migration target was chosen after checking source VRAM need, torch/CUDA compatibility, and expected concurrency.
```

### 6.3C Cheap first test vs real migration target

Separate these two decisions.

A cheap first test answers:

- does the repository build?
- does the app start?
- are secrets present?
- does the target have any CUDA device?
- does the smallest request complete?

A real migration target answers:

- does the actual model fit?
- is latency acceptable?
- does concurrency work?
- does the GPU architecture match the dependency stack?
- does the hourly cost fit the use case?

Don't skip the cheap first test when you are unsure whether the failure is even GPU-related. Don't stop at the cheap first test if the source Space probably depended on larger ZeroGPU memory.

### 6.4 What to do with `@spaces.GPU`

For the first migration test:

- keep it if it does not break anything
- don't rely on it for standard GPU scheduling
- remove or simplify it after the app is stable

A common cleanup is to move from:

```python
import spaces

@spaces.GPU(duration=120)
def generate(prompt):
    return pipe(prompt).images
```

to a standard Gradio function:

```python
def generate(prompt):
    return pipe(prompt).images
```

But do this after you know the duplicate works. Removing wrappers and changing hardware at the same time can hide the real failure.

### 6.5 CUDA initialization changes

ZeroGPU often rewards delayed GPU work. Standard GPU Spaces may tolerate earlier CUDA initialization, but that does not mean eager loading is always best.

For standard GPU migration, decide intentionally:

- **Eager load at startup** if predictable warm performance matters and startup time is acceptable.
- **Lazy load on first request** if startup health is fragile or model downloads are large.
- **Preload model files at build time** when download time is the problem rather than import time.

A standard GPU Space can still fail startup if the app takes too long to become healthy.

### 6.5A Example: converting eager model load safely

A ZeroGPU app may put GPU work inside a decorated function. A standard GPU Space can often load the model once at startup, but only if the selected hardware has enough memory and startup time.

A simple migration pattern is:

```python
# ZeroGPU-style shape
import spaces

@spaces.GPU(duration=120)
def generate(prompt):
    pipe.to("cuda")
    return pipe(prompt).images[0]
```

For a standard GPU target, you may prefer:

```python
# Standard GPU-style shape
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = load_pipeline()
pipe.to(device)

def generate(prompt):
    return pipe(prompt).images[0]
```

Don't make this change blindly. If model loading is large or slow, first verify startup health, memory, and whether `startup_duration_timeout` or `preload_from_hub` should be adjusted.

Useful references:

- [Using GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)

### 6.6 Remove ZeroGPU quota assumptions

After migrating to standard GPU:

- don't debug normal GPU failures as ZeroGPU quota failures
- don't expect `X-IP-Token` to solve ordinary authentication
- don't use ZeroGPU duration settings as the central performance control
- do track hardware billing, sleep behavior, and assigned accelerator

If the migrated app still calls another ZeroGPU Space, then ZeroGPU request identity may still matter for that downstream call.

---

## 7. Target B: ZeroGPU → ZeroGPU Space

This path keeps the ZeroGPU runtime model.

It is shorter, but it still needs checks.

### 7.1 Safe ZeroGPU-to-ZeroGPU workflow

1. Confirm the source is a Gradio ZeroGPU Space.
2. Duplicate privately.
3. Select or request ZeroGPU hardware.
4. Recreate secrets.
5. Match `python_version` and `sdk_version` where possible.
6. Use a ZeroGPU-supported PyTorch version.
7. Test from the normal Hugging Face Space page while logged in.
8. Start with a small workload.

### 7.1A What to keep for a faithful ZeroGPU copy

For ZeroGPU → ZeroGPU, keep these stable until the duplicate works once:

- Gradio SDK
- `@spaces.GPU` boundaries
- declared `duration` values
- dependency pins around `torch`, `gradio`, `spaces`, `diffusers`, `transformers`, and accelerator-sensitive packages
- normal HF Space page as the first test path
- same ZeroGPU size when possible

Only optimize after the duplicate proves it can run the smallest meaningful workload.

Useful reference: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

### 7.2 Keep the ZeroGPU execution shape

For ZeroGPU-to-ZeroGPU, keep:

- `import spaces`
- `@spaces.GPU`
- duration and size declarations, if they were deliberate
- Gradio SDK assumptions

Then check whether they are still valid under current ZeroGPU docs.

### 7.3 Check ZeroGPU size

Current ZeroGPU sizes are:

- `large`: Half NVIDIA RTX Pro 6000 Blackwell, 48GB, 1× quota cost
- `xlarge`: Full NVIDIA RTX Pro 6000 Blackwell, 96GB, 2× quota cost

Use `xlarge` only if the workload really needs it. It costs more quota and can make the duplicate feel worse for no benefit if `large` is enough.

### 7.4 Test from the standard Space page first

For ZeroGPU-to-ZeroGPU, first test from the normal Hugging Face Space page while logged in.

Avoid starting with:

- direct `*.hf.space` links
- embeds
- custom frontends
- another Space calling this one
- API clients

The reason is request identity. ZeroGPU rate limiting uses a request header called `X-IP-Token`. If the identity path is missing, the request may be treated as unauthenticated and the duplicate can look broken even when the app is fine.

---

## 8. Hardware Generation Uncertainty: A100, H200, Blackwell

Many old ZeroGPU discussions, examples, and mental models refer to different hardware generations.

You may see or suspect:

- **A100-era assumptions** in older ZeroGPU examples or client documentation
- **H200-era assumptions** in earlier public ZeroGPU documentation
- **RTX Pro 6000 Blackwell assumptions** in current ZeroGPU documentation

Don't try to infer too much from the age of the Space alone. A Space can keep old code while being rebuilt under a newer runtime.

### 8.1 Treat unknown as unknown

If you cannot prove the source runtime generation, write down:

```text
source_gpu_generation = unknown
```

Then verify behavior instead of guessing.

### 8.2 What to capture

Use logs or diagnostics:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```

For Blackwell-class devices, the capability may expose a newer CUDA architecture than older PyTorch wheels were built to support. In current ZeroGPU Blackwell observations, this can show up as CUDA capability `sm_120`. A common symptom is:

```text
CUDA error: no kernel image is available for execution on the device
```

That usually means the installed binary does not contain a kernel compatible with the device architecture.

### 8.3 Practical rule

When in doubt, don't preserve a historical torch pin blindly.

For current ZeroGPU, use a supported PyTorch version and a CUDA-capable wheel appropriate for the current runtime. For standard GPU migration, choose a PyTorch/CUDA combination that matches the selected GPU tier and the packages used by the app.

---

## 8A. Hardware Generation Decision Aid

When a source ZeroGPU Space is old, you may not know whether its working assumptions came from A100-era examples, H200-era documentation, or the current RTX PRO 6000 Blackwell-backed ZeroGPU runtime. Treat that uncertainty as evidence to collect, not as trivia.

### 8A.1 Practical hardware clues

| Clue | What it suggests | Caveat |
|---|---|---|
| logs mention `sm_80` or A100 | old CUDA/Ampere expectation | may be from a dependency, not actual hardware |
| docs/comments mention H200 | H200-era ZeroGPU expectation | may be stale documentation |
| logs mention `sm_120` | Blackwell device exposure | requires CUDA/wheel support that understands that capability |
| `no kernel image is available` appears | CUDA wheel cannot run kernel for device | often torch/CUDA build mismatch, not app logic |
| app only fails after duplicate/rebuild | package resolver or hardware assignment changed | capture both old and new runtime evidence |

### 8A.2 Don't infer too much from the source UI

The source Space might still appear healthy because it is warm, cached, pinned, or rarely rebuilt. A duplicate is rebuilt now. That means the duplicate can reveal a compatibility problem the source has not yet hit.

### 8A.3 Current ZeroGPU baseline

Use the current ZeroGPU documentation as the baseline for new work. At the same time, when migrating an older Space, preserve enough evidence to understand whether the source depended on older assumptions.

Useful references:

- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Using GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)

---

## 9. README YAML and Runtime Configuration

The README YAML block is part of the runtime contract.

### 9.0 Example YAML block

A Space can encode important runtime expectations in the YAML header of `README.md`. A simplified example:

```yaml
---
title: My Migrated Space
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10
app_file: app.py
suggested_hardware: a10g-small
startup_duration_timeout: 1h
preload_from_hub:
  - black-forest-labs/FLUX.1-schnell
---
```

This header is not decorative. It is part of the operational contract.

### 9.1 Fields to inspect

Check:

```yaml
---
sdk: gradio
python_version: "3.12"
sdk_version: "6.0.0"
app_file: app.py
suggested_hardware: a10g-large
startup_duration_timeout: 1h
preload_from_hub:
  - owner/model-name
---
```

Don't copy these fields blindly. Decide whether they still match the target runtime.

### 9.2 `sdk`

ZeroGPU requires Gradio. Standard GPU Spaces can use other SDKs, but if the source is a Gradio ZeroGPU Space, the first migration should usually remain Gradio until the GPU migration works.

### 9.3 `sdk_version`

Pinning `sdk_version` can stabilize the duplicate. It is especially useful when a Gradio major version changed event behavior, frontend behavior, or API behavior.

### 9.4 `python_version`

Match the source when the goal is faithful reproduction. Change it only when a dependency or platform requirement demands it.

### 9.5 `suggested_hardware`

This is a recommendation for duplicate users. It does not assign hardware. Use it to document the intended standard GPU target after you know the migration works.

### 9.6 `startup_duration_timeout`

Use this when startup legitimately needs more time. Don't use it to hide a broken startup loop.

### 9.7 `preload_from_hub`

Use this when startup is slow because the app downloads large Hub artifacts. Moving downloads into build time can make the app become healthy faster.

---

## 10. Dependencies and Torch/CUDA Compatibility

Dependency drift is one of the most common reasons a duplicate behaves differently from the source.

### 10.1 Start with the smallest critical set

Capture and compare:

- `torch`
- `torchvision`
- `torchaudio`
- `gradio`
- `spaces`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- model-specific packages

### 10.2 Don't pin randomly

A good pin explains a boundary:

```text
torch==2.9.1+cu128 because current ZeroGPU Blackwell runtime needs a CUDA 12.8-capable wheel
```

A weak pin only says:

```text
torch==old.version because it used to work
```

### 10.2A Dependency comparison worksheet

Use a small worksheet instead of guessing:

| Package | Source value | Duplicate value | Why it matters |
|---|---|---|---|
| Python |  |  | package support and image behavior |
| `gradio` |  |  | Space SDK/runtime behavior |
| `torch` |  |  | CUDA wheel and GPU capability support |
| `spaces` |  |  | ZeroGPU decorator behavior |
| `transformers` |  |  | model loading and generation APIs |
| `diffusers` |  |  | pipeline loading and scheduler behavior |
| `accelerate` |  |  | device placement and memory behavior |

For a standard GPU migration, don't keep ZeroGPU-specific pins just because they worked once. Keep them only if they are still compatible with the target GPU and the app.

Useful references:

- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)

### 10.3 Standard GPU migration rule

For ZeroGPU → standard GPU migration, the PyTorch version does not need to be chosen only for ZeroGPU. It needs to fit:

- the selected standard GPU
- the CUDA runtime available in the Space image
- the model libraries
- any compiled extensions
- the desired Python version

### 10.4 When `no kernel image` appears

If you see:

```text
CUDA error: no kernel image is available for execution on the device
```

check:

1. GPU name
2. CUDA capability
3. `torch.__version__`
4. `torch.version.cuda`
5. whether any extension package includes device-specific CUDA kernels

Don't start by rewriting the model pipeline. Verify the binary compatibility layer first.

---

## 11. Secrets, Variables, Tokens, and Private Assets

Duplicated Spaces often fail because secrets were not recreated.

### 11.0 Inventory first

Before the first serious test, list the names of required environment values. Don't paste the secret values into notes or public logs.

```text
Required variables:
- MODEL_REPO_ID
- APP_MODE

Required secrets:
- HF_TOKEN
- OPENAI_API_KEY
- PRIVATE_BACKEND_URL
```

Common symptoms of missing secrets include blank outputs, private model load failures, `401`, `403`, silent fallback to public assets, or browser success while API paths fail.

Useful references:

- [User access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Manage your Space](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces)

### 11.1 Variables vs secrets

Use:

- **Variables** for non-sensitive configuration
- **Secrets** for tokens, API keys, private credentials, OAuth secrets, or sensitive URLs

Public Variables may be surfaced during duplication. Secrets must be explicitly set.

### 11.2 Private and gated models

If the source app loads private or gated models, the duplicate needs a token with access to those resources.

Check:

- whether the model is gated
- whether the account behind the token accepted the terms
- whether the token scope is enough
- whether organization token policies require fine-grained tokens

### 11.3 Browser success does not prove API success

A private Space can work in your browser but fail through an API client if the client token is missing or has the wrong scope.

Test both paths separately:

1. browser path
2. `gradio_client` path
3. direct HTTP path, if used
4. Space-to-Space path, if used

---

## 12. Programmatic Duplication and Runtime Management

You can manage Spaces with `huggingface_hub`.

Useful operations include:

- duplicate a Space
- inspect runtime
- request hardware
- fetch logs
- add secrets
- restart or pause the Space

A minimal shape using the current general duplication API:

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_...")

# Duplicate the source Space.
api.duplicate_repo(
    from_id="source-owner/source-space",
    to_id="your-name/your-copy",
    repo_type="space",
    visibility="private",
)

# Add required secrets.
api.add_space_secret("your-name/your-copy", key="HF_TOKEN", value="hf_...")

# Request standard GPU hardware for migration.
api.request_space_hardware("your-name/your-copy", hardware="a10g-large")

# Inspect actual state.
runtime = api.get_space_runtime("your-name/your-copy")
print(runtime.stage)
print(runtime.hardware)
print(runtime.requested_hardware)
```

The older `duplicate_space()` helper may still appear in examples, but the current `huggingface_hub` reference marks it as deprecated in favor of `duplicate_repo()` for future compatibility.

Always compare `hardware` and `requested_hardware`. They can differ while a request is still being applied.

---

## 12A. UI, Python, and CLI Paths

There are three practical ways to create or manage a duplicate. Use the one that matches the job.

### 12A.1 UI path

Use the UI when you are doing a one-off private duplicate and want to inspect settings manually.

Good for:

- first exploratory copy
- visually checking hardware/storage choices
- noticing warnings about secrets

Bad for:

- repeated migration tests
- exact audit trails
- multi-Space migration work

Reference: [More ways to create Spaces](https://huggingface.co/docs/hub/spaces-more-ways-to-create).

### 12A.2 Python API path

Use Python when you want repeatability:

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_...")
repo_id = "your-name/private-copy"

api.duplicate_repo(
    from_id="source-owner/source-space",
    to_id=repo_id,
    repo_type="space",
    private=True,
)

api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_...")
api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="owner/model")
api.request_space_hardware(repo_id=repo_id, hardware="a10g-small")

runtime = api.get_space_runtime(repo_id)
print(runtime.stage, runtime.hardware, runtime.requested_hardware)
```

Depending on your installed `huggingface_hub` version, exact argument names may differ. Check your installed version and the current API docs before automating a large migration.

References:

- [Managing your Space runtime](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime)
- [HfApi reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)

### 12A.3 CLI path

Use the CLI for simple repo operations when you prefer shell workflows. For runtime management, Python usually gives clearer inspection and error handling.

Reference: [Hugging Face CLI reference](https://huggingface.co/docs/huggingface_hub/package_reference/cli).

---

## 12B. Local Docker Preflight Checks

Local Docker is optional, but it is useful when you are migrating a ZeroGPU Space to a standard GPU Space.

Use it to answer a narrow question:

> Can this repo start as a normal container, see the expected environment variables, load the model, and reach its HTTP port before I pay for or debug hosted GPU hardware?

Don't use local Docker as proof that the hosted Space will behave exactly the same. Hugging Face Spaces add their own build, proxy, hardware, secrets, storage, and account-layer behavior. ZeroGPU adds an even more specific request-based GPU runtime. Local Docker is a **preflight**, not a perfect emulator.

Useful references:

- [Run with Docker](https://huggingface.co/docs/hub/spaces-run-with-docker)
- [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Desktop GPU support for Windows](https://docs.docker.com/desktop/features/gpu/)

### 12B.1 When local Docker helps

Local Docker is especially useful for the **ZeroGPU → standard GPU** path.

It can help you catch:

- missing system packages
- broken import paths
- missing environment variables
- wrong launch command
- wrong exposed port
- model download or cache assumptions
- CUDA wheel mismatch
- `torch.cuda.is_available()` returning false inside the container
- model load happening too early or too late
- startup time that will probably exceed hosted Space health checks

It is less useful for proving ZeroGPU-specific behavior.

Local Docker will not reliably reproduce:

- ZeroGPU quota behavior
- `@spaces.GPU` dynamic allocation timing
- `X-IP-Token` request identity behavior
- Hugging Face's hosted proxy behavior
- the exact hosted GPU generation
- the exact container image Hugging Face will build for non-Docker SDK Spaces
- account, org, or token policy behavior

### 12B.2 Three useful local Docker paths

There are three different things people mean by “run the Space locally.” Keep them separate.

| Local path | What it tests | Best for | What it does not prove |
|---|---|---|---|
| Clone repo and run `python app.py` | Python app startup without container | quick code sanity check | Dockerfile, system packages, hosted proxy |
| Build your own Docker image from the repo | container startup and GPU access | standard GPU migration | hosted Space settings and account behavior |
| Use the Space page's **Run with Docker** command | the published Space image when available | checking an existing hosted image | source-code migration choices |

Hugging Face documents a **Run with Docker** button for running most Spaces locally, and notes that some Spaces require login to the Hugging Face Docker registry using your username and a User Access Token.

### 12B.3 Minimal local Docker smoke test

For a Docker-based target, the simplest local test is:

```bash
# From the Space repository root.
docker build -t local-space-test .

docker run --rm \
  -p 7860:7860 \
  --env-file .env \
  local-space-test
```

Then open:

```text
http://localhost:7860
```

This checks four basic things:

1. the image builds
2. the app starts
3. the app binds to the right host and port
4. the required non-secret and secret values are available locally

For Docker Spaces, Hugging Face's default exposed app port is `7860`, and the README YAML can set `sdk: docker` and `app_port: 7860`.

### 12B.4 GPU local Docker smoke test

If your local machine has an NVIDIA GPU and Docker can see it, test with `--gpus all`:

```bash
docker run --rm \
  --gpus all \
  -p 7860:7860 \
  --env-file .env \
  local-space-test
```

Inside the container, this small check is often enough:

```bash
python - <<'LOCALCHECK'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
LOCALCHECK
```

If this fails locally, don't immediately blame Hugging Face. First check whether Docker itself has GPU access. On Linux, NVIDIA's official path is the NVIDIA Container Toolkit. On Windows, Docker Desktop GPU support depends on WSL2 and NVIDIA GPU support.

A host-level sanity check is:

```bash
nvidia-smi
```

A Docker-level sanity check is:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` works on the host but not inside Docker, the problem is usually the local Docker GPU setup, not the Space code.

### 12B.5 Minimal Dockerfile shape for a standard GPU migration

A migrated Space does not always need to become a Docker SDK Space. However, a Dockerfile-shaped local test often clarifies what the app really needs.

A simple Gradio-style shape looks like this:

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 7860
CMD ["python", "app.py"]
```

For a Gradio app, the Python side should bind to `0.0.0.0` and the expected port:

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("Local Docker smoke test")

demo.launch(server_name="0.0.0.0", server_port=7860)
```

This is not meant to be a universal production Dockerfile. It is a diagnostic baseline. Add CUDA images, system packages, model cache handling, or a non-root user only after the simple version proves the app shape.

### 12B.6 CUDA image vs Python image

For a real standard GPU migration, choose the base image deliberately.

| Base image direction | Good for | Risk |
|---|---|---|
| `python:3.11-slim` | fast CPU/startup debugging | no CUDA libraries by default |
| NVIDIA CUDA runtime image | GPU inference runtime | larger image and more version decisions |
| Framework image such as PyTorch CUDA | matching torch/CUDA stack | can hide version assumptions |
| Existing project image | preserving upstream behavior | can carry old or unnecessary baggage |

A practical sequence is:

1. use a small Python image to prove non-GPU startup
2. use a CUDA/PyTorch image to prove GPU visibility
3. only then tune for size, caching, and startup time

### 12B.7 Local Docker and ZeroGPU-specific code

Be careful with code written specifically for ZeroGPU.

For ZeroGPU, the app may rely on:

- `import spaces`
- `@spaces.GPU`
- GPU initialization happening only inside a decorated function
- request identity headers
- quota behavior

Local Docker does not reproduce the ZeroGPU scheduler. If you run the same repo locally on a normal GPU, treat it as a **standard GPU migration test**, not as a faithful ZeroGPU test.

The useful question is not:

> Does local Docker perfectly emulate ZeroGPU?

The useful question is:

> If I remove or bypass the ZeroGPU runtime assumptions, can this app run as a normal GPU service?

### 12B.8 Local `.env` file pattern

Don't hard-code secrets in the Dockerfile.

Use a local `.env` file that mirrors the target Space settings:

```bash
HF_TOKEN=hf_...
MODEL_ID=owner/model
CACHE_DIR=/data/.cache/huggingface
```

Run with:

```bash
docker run --rm \
  --gpus all \
  -p 7860:7860 \
  --env-file .env \
  -v "$PWD/.cache:/data/.cache" \
  local-space-test
```

The cache mount is optional. It is useful when model downloads are large and you don't want every container run to redownload the same files.

Don't commit `.env`.

### 12B.9 Local Docker decision table

| Result | What it probably means | Next move |
|---|---|---|
| Builds and runs locally on CPU | app shape is probably valid | test hosted CPU or requested GPU |
| Fails to build locally | dependency or Dockerfile issue | fix before hosted GPU testing |
| CPU works, GPU Docker cannot see CUDA | local Docker GPU setup issue or driver/toolkit issue | fix NVIDIA Container Toolkit / Docker Desktop GPU path |
| GPU visible, model fails with CUDA error | torch/CUDA/model compatibility issue | check torch wheel, CUDA version, compute capability |
| Local GPU works, hosted standard GPU fails | hosted hardware/config difference | compare hardware, logs, environment, startup |
| Hosted works, local fails | local Docker environment mismatch | don't overfit local setup |

### 12B.10 Local Docker should not delay the first hosted smoke test forever

Local Docker is valuable, but it can become a distraction.

Use it when it answers a specific question:

- Does the Dockerfile build?
- Does the app bind to port `7860`?
- Are secrets and variables named correctly?
- Can torch see CUDA in a container?
- Does the model load before the Space health timeout?

After those are answered, move back to the hosted Space. The hosted environment is the target.



### 12B.11 Local Docker preflight examples

These examples are deliberately small. The goal is not to reproduce Hugging Face infrastructure. The goal is to separate local app errors from hosted runtime errors.

#### Example A — CPU-only build sanity check

Use this when you want to know whether the repo and dependencies are basically coherent.

```bash
docker build -t space-preflight:cpu .
docker run --rm -p 7860:7860 space-preflight:cpu
```

Good signal:

- imports succeed
- the Gradio app starts
- the port opens
- obvious dependency errors appear locally before you spend hosted GPU time

Bad conclusion:

- “CPU Docker works, so GPU migration will work.”

CPU Docker says almost nothing about CUDA wheel compatibility, GPU memory, or ZeroGPU request behavior.

#### Example B — GPU visibility sanity check

Use this when the host has NVIDIA Docker support and you want to confirm CUDA visibility.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Then test your own image:

```bash
docker run --rm --gpus all -p 7860:7860 space-preflight:gpu
```

Inside the container, check:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda build', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
PY
```

Good signal:

- Docker can see the GPU
- torch can see CUDA
- the app can import and start

Bad conclusion:

- “Local GPU works, so ZeroGPU behavior is reproduced.”

Local GPU Docker is closer to a standard GPU Space than to ZeroGPU.

#### Example C — secret and gated-model preflight

Use an `.env` file for local smoke tests, but don't commit it.

```bash
cat > .env <<'EOF'
HF_TOKEN=hf_xxx
MODEL_ID=owner/private-or-gated-model
EOF

docker run --rm --env-file .env -p 7860:7860 space-preflight:cpu
```

If the hosted duplicate fails but local Docker succeeds only when `.env` is present, check Space secrets first.

#### Example D — cache mount preflight

Large models can make local tests slow. Use a cache mount so repeated tests are not dominated by downloads.

```bash
mkdir -p .hf-cache

docker run --rm \
  --env-file .env \
  -e HF_HOME=/data/.cache/huggingface \
  -v "$PWD/.hf-cache:/data/.cache/huggingface" \
  -p 7860:7860 \
  space-preflight:cpu
```

This resembles a persistent cache, not a guarantee that the hosted Space will have the same cache state. Hosted rebuilds can start colder than your local test.

### 12B.12 Local Docker failure map

| Local Docker symptom | Usually means | First local fix | Hosted follow-up |
|---|---|---|---|
| `ModuleNotFoundError` | Missing dependency | Add or pin in `requirements.txt` | Rebuild Space and inspect logs |
| `ImportError` from CUDA package on CPU image | GPU package imported in CPU-only test | Use GPU image or delay import | Decide if standard GPU target is required |
| `torch.cuda.is_available() == False` with `--gpus all` | NVIDIA container runtime not available | Install/configure NVIDIA Container Toolkit | Don't treat as HF Space bug |
| Model download fails with 401/403 | Missing token or gated access | Add local env token | Add Space secret with correct token |
| App runs locally but not on Space | Hosted config or hardware mismatch | Compare README YAML and Space settings | Check `app_file`, hardware, secrets, logs |
| Space runs but local Docker fails | Local image differs from Space build | Align base image and dependencies | Don't overfit to local environment |

### 12B.13 When to stop local Docker debugging

Stop local Docker debugging when:

- the app imports locally
- the app starts locally
- secrets and model access are understood
- CUDA visibility has been checked if local GPU exists
- the remaining failure depends on Space hardware, ZeroGPU quota, or hosted build logs

At that point, move back to the hosted duplicate and test the smallest request there.

Local Docker is a preflight tool. It is not the release environment unless you are deliberately using a Docker SDK Space.

### 12B.14 Docker SDK Space vs local Docker

Don't confuse these two ideas.

| Topic | Local Docker preflight | Docker SDK Space |
|---|---|---|
| Where it runs | Your machine or workstation | Hugging Face Space infrastructure |
| Goal | Debug build/start/runtime assumptions before hosted test | Define the Space runtime with a custom Dockerfile |
| GPU behavior | Depends on local NVIDIA Docker setup | Depends on selected Space hardware |
| Secrets | Local `.env` or environment variables | Space secrets and variables |
| Best use | Preflight and diagnosis | Production-ish Space packaging when Gradio/Streamlit defaults are not enough |

If a Space is already a simple Gradio Space, don't switch to Docker SDK just because duplication failed. First prove whether the failure is secrets, hardware, dependency drift, or ZeroGPU-specific behavior.

## 13. API, Client, and Frontend Paths

Don't assume every access path behaves the same.

### 13.1 Browser path

Use the normal Hugging Face Space page for the first validation.

This is especially important for ZeroGPU because the standard page is the least surprising path for request identity.

### 13.1A Public browser success is not enough

A Space can work from the browser and still fail from an API client. The browser path, private API path, custom frontend path, and Space-to-Space path can carry different authentication and request-identity context.

For private Gradio Spaces, start with the official API endpoint workflow and a token with appropriate read access.

Reference: [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints).

### 13.2 Private Space API path

For private Spaces, use a token with read access.

Example:

```python
from gradio_client import Client

client = Client("your-name/private-space", token="hf_...")
result = client.predict("hello", api_name="/predict")
print(result)
```

### 13.3 Space-to-Space ZeroGPU path

If a Gradio app calls a ZeroGPU Space, forward the user identity token:

```python
import gradio as gr
from gradio_client import Client

def process(prompt, request: gr.Request):
    x_ip_token = request.headers.get("x-ip-token", "")
    client = Client("owner/zerogpu-space", headers={"x-ip-token": x_ip_token})
    return client.predict(prompt, api_name="/predict")
```

This matters only when the target or downstream call is ZeroGPU. For a pure standard GPU migration, don't confuse ordinary API authentication with ZeroGPU quota identity.

### 13.4 Custom frontends

Custom frontends can break assumptions that the normal Hub page satisfies. If quota or authentication behavior looks strange, test the standard page first before debugging the model.

---

## 14. Symptom-First Playbooks

Use this section after duplication or migration.

---


### 14.0 Fast triage map

Start here before editing code.

| First visible symptom | Most likely layer | First evidence to collect | Don't do this first |
|---|---|---|---|
| Stuck on `Building` | Build queue, dependency, storage, or hardware scheduling | Build log, dependency file, hardware request | Rewrite model code |
| `Running` but wrong output | Runtime drift | README YAML, package versions, GPU name/capability, source settings | Change hardware and dependencies together |
| Quota or PRO behavior wrong | ZeroGPU identity path | Standard HF page test, login state, `X-IP-Token` path | Debug only from custom frontend |
| Browser works, API fails | Auth/API path | token scope, Space privacy, `api_name`, client version | Assume model failure |
| CUDA kernel image error | CUDA wheel / GPU architecture | torch version, CUDA build, device capability | Downgrade randomly |
| `No CUDA GPUs are available` | No assigned GPU or wrong execution path | runtime hardware, `torch.cuda.is_available()` | Assume ZeroGPU bug immediately |
| OOM after migration | capacity or loading strategy | VRAM, batch size, eager loads, source ZeroGPU size | Keep scaling inputs during debugging |

### 14.0A Branch decision tree

Use this compact decision tree when the failure is unclear.

```text
START
|
|-- Did the duplicate ever become Running?
|   |-- no  -> inspect build logs, dependency files, hardware request, storage, app_file
|   |-- yes -> continue
|
|-- Does the standard HF browser page work?
|   |-- no  -> debug hosted app path before API/custom frontend
|   |-- yes -> continue
|
|-- Does the API/custom/client path fail?
|   |-- yes -> check token, api_name, privacy, X-IP-Token if ZeroGPU is involved
|   |-- no  -> continue
|
|-- Does it fail only on GPU work?
|   |-- yes -> check hardware, CUDA, torch, VRAM, @spaces.GPU assumptions
|   |-- no  -> compare config, dependencies, secrets, source assets
```

### 14.0B When the problem is probably not your model code

Suspect platform/config/runtime before model code when:

- the exact same commit behaves differently after duplication
- the failure appears before the model function is called
- logs show build queue, storage, or scheduling messages
- a private duplicate fails only through API calls
- browser UI works but programmatic callers fail
- the duplicate defaulted to CPU
- the failure disappears after restoring secrets
- the failure changes when hardware is changed

In these cases, make a runtime snapshot before touching the model pipeline.

### 14.0C When the problem is probably your migration edit

Suspect your edit when:

- the original duplicate worked before cleanup
- failure began after removing `@spaces.GPU`, changing torch, or changing model load timing
- local Docker and hosted Space both fail the same import
- the traceback points to your app code, not build/setup/runtime assignment
- a smaller input works but real input fails after changing batch/resolution defaults

Rollback one change at a time. Don't change dependency pins, hardware tier, and model-loading strategy in one pass.

---

### 14.1 Duplicate lands on CPU

#### Symptom

The duplicate builds, but no GPU is available.

#### Likely cause

The duplicate defaulted to CPU or the hardware request was not applied yet.

#### First checks

- Did you explicitly choose hardware during duplication?
- What does `get_space_runtime()` report?
- Are `hardware` and `requested_hardware` different?
- Is the Space still building?

#### First fixes

- request the intended hardware
- restart after the hardware request settles
- confirm actual runtime again
- don't debug model CUDA code until the Space really has a GPU

---

### 14.2 Standard GPU migration fails at startup

#### Symptom

The app never becomes healthy, or the Space returns a startup failure.

#### Likely causes

- model loads too early and startup exceeds the health window
- large files download during startup
- wrong `app_file`
- missing secret
- dependency import failure
- out-of-memory during eager load

#### First fixes

- inspect logs
- reduce to a minimal startup
- lazy-load the model
- use `preload_from_hub` for large Hub files
- increase `startup_duration_timeout` only if startup is genuinely slow and otherwise healthy

---

### 14.3 `CUDA error: no kernel image is available for execution on the device`

#### Symptom

The app starts or imports, but CUDA execution fails with a kernel image error.

#### Likely cause

The installed PyTorch wheel or compiled CUDA extension does not support the GPU architecture.

#### First checks

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```

#### First fixes

- use a current supported torch/CUDA wheel
- rebuild or remove incompatible compiled extensions
- avoid preserving old torch pins without checking GPU architecture

---

### 14.4 `CUDA has been initialized before importing the spaces package`

#### Symptom

A ZeroGPU copy fails with an error about CUDA being initialized before `spaces`.

#### Likely cause

The app touches CUDA too early for ZeroGPU's runtime model.

#### First fixes for ZeroGPU target

- import `spaces` early
- keep GPU work inside `@spaces.GPU`
- avoid `torch.cuda` checks at module import time
- avoid `model.to("cuda")` before the ZeroGPU-managed path

#### First fixes for standard GPU target

- decide whether you are still trying to run as ZeroGPU
- if migrating to standard GPU, remove the dependency on ZeroGPU allocation semantics after the baseline runs
- don't debug this as a standard GPU quota issue

---

### 14.5 Quota exceeded or PRO ignored

#### Symptom

A ZeroGPU copy appears to have too little quota or treats a logged-in user like an anonymous user.

#### Likely causes

- missing `X-IP-Token` path
- direct or custom frontend path
- old Gradio stack
- app-to-app calls without forwarding user identity
- real quota exhaustion

#### First fixes

- test from the standard Hugging Face Space page while logged in
- avoid custom frontends for the first validation
- forward `x-ip-token` for Space-to-Space ZeroGPU calls
- check Gradio version
- reduce `duration` or avoid `xlarge` if unnecessary

---

### 14.6 Browser works, API fails

#### Symptom

The app works in the browser but fails through `gradio_client`, curl, or another Space.

#### Likely causes

- missing token
- wrong token scope
- private Space access issue
- endpoint or API name mismatch
- ZeroGPU identity not forwarded for downstream calls

#### First fixes

- test with a read-capable token
- confirm the API endpoint name
- retry with exponential backoff for transient failures
- if calling ZeroGPU from another Space, forward `x-ip-token`

---

### 14.7 Output differs from the source

#### Symptom

The duplicate runs but produces different results or fails on inputs that worked in the source.

#### Likely causes

- dependency drift
- different GPU generation
- different torch/CUDA stack
- missing secret or private model access
- different startup cache state
- different hardware size

#### First fixes

- compare package versions
- compare README YAML
- compare hardware and GPU capability
- test the smallest shared input
- pin only the version boundaries you can explain

---

### 14.8 Out of memory on standard GPU

#### Symptom

The app migrates from ZeroGPU but fails with CUDA OOM on the selected standard GPU.

#### Likely causes

- standard GPU has less usable memory than the ZeroGPU size used by the source
- model loads too many components eagerly
- batch size or image size is too high
- multiple pipelines are loaded at once

#### First fixes

- lower resolution, batch size, steps, or concurrent work
- unload unused components
- use a larger standard GPU tier
- verify whether the source needed `xlarge` ZeroGPU
- avoid assuming a standard small GPU matches ZeroGPU `large` or `xlarge`

---

### 14.9 Duplicate is private and API calls return auth errors

#### Symptom

The browser works while logged in, but `gradio_client`, curl, or another app receives `401`, `403`, or endpoint errors.

#### Likely causes

- private Space API call lacks a bearer token
- token has wrong scope
- org token policy requires fine-grained tokens
- API route or `api_name` is wrong
- custom frontend path is not equivalent to the normal Space API path

#### First fixes

1. test with the official `gradio_client` pattern
2. use a token with read access to the private Space
3. confirm org policy if the Space is under an organization
4. inspect the Space API page or generated client usage

Useful references:

- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [User access tokens](https://huggingface.co/docs/hub/security-tokens)

### 14.10 Build succeeds, launch times out

#### Symptom

The duplicate builds, but never becomes healthy, or fails after a long startup window.

#### Likely causes

- large model download at startup
- slow conversion or cache generation at startup
- wrong `app_file`
- dependency import stall
- normal GPU target too small or not yet assigned

#### First fixes

1. check logs before editing app logic
2. verify `app_file`
3. consider `preload_from_hub` for large Hub assets
4. consider `startup_duration_timeout` only after you know startup is legitimately slow, not stuck
5. inspect actual runtime hardware

Useful reference: [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference).

### 14.11 Migrated Space works once, then fails after sleep or rebuild

#### Symptom

The migrated standard GPU Space works after manual fixes, then fails after restart, sleep, or rebuild.

#### Likely causes

- fix was made only in a running container
- cache or generated file was not committed or recreated
- dependency version was not pinned
- secret or variable was added to one Space but not the new target

#### First fixes

1. factory rebuild once
2. check whether the fix is actually in Git
3. make startup recreate required generated state
4. move large persistent assets to the correct Hub repo or storage path

---

## 15. Migration Checklists

### 15.1 Pre-duplicate checklist

- [ ] Source Space ID recorded
- [ ] README YAML captured
- [ ] dependency files captured
- [ ] ZeroGPU-specific code located
- [ ] source secrets/variables inventory created
- [ ] source hardware evidence captured if possible
- [ ] target chosen: standard GPU or ZeroGPU

### 15.2 ZeroGPU → standard GPU checklist

- [ ] Duplicate created privately
- [ ] standard GPU requested
- [ ] `hardware` and `requested_hardware` checked
- [ ] secrets recreated
- [ ] first test kept small
- [ ] ZeroGPU quota/debug assumptions not used for standard GPU failures
- [ ] `@spaces.GPU` kept temporarily or removed intentionally
- [ ] torch/CUDA compatibility verified
- [ ] startup behavior checked
- [ ] final README YAML documents intended hardware

### 15.3 ZeroGPU → ZeroGPU checklist

- [ ] Duplicate created privately
- [ ] ZeroGPU selected/requested
- [ ] secrets recreated
- [ ] Gradio SDK confirmed
- [ ] ZeroGPU-supported torch version used
- [ ] `@spaces.GPU` path intact
- [ ] standard HF page tested first
- [ ] quota and `xlarge` usage checked
- [ ] API or app-to-app path tested separately

### 15.4 Release checklist for your migrated Space

- [ ] Minimal input works
- [ ] realistic input works
- [ ] private/gated model access works
- [ ] browser path works
- [ ] API path works if needed
- [ ] logs don't hide dependency warnings
- [ ] README YAML matches actual runtime
- [ ] secrets are not hard-coded
- [ ] hardware cost/quota implications are understood

---

## 16. Common Error Messages

| Error or symptom | Usually means | First place to look |
|---|---|---|
| `No CUDA GPUs are available` | hardware not assigned, still CPU, or ZeroGPU allocation issue | runtime state, hardware request, logs |
| `CUDA has been initialized before importing the spaces package` | CUDA touched too early for ZeroGPU | import order, `@spaces.GPU`, eager CUDA checks |
| `no kernel image is available for execution on the device` | torch/CUDA wheel or extension does not support device architecture | torch version, CUDA version, GPU capability |
| quota exceeded too early | real quota use or missing ZeroGPU request identity | standard HF page, `X-IP-Token`, duration, size |
| browser works but API fails | auth/token/API path issue | read token, API name, private Space access |
| duplicate builds on CPU | default duplicate hardware or pending request | hardware, requested_hardware |
| output differs from source | dependency or runtime drift | versions, hardware, secrets, YAML |
| startup timeout | slow startup, large downloads, health check failure | logs, `startup_duration_timeout`, preload, lazy loading |

---

## 17. Practical Snippets

### 17.1 Runtime snapshot

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_...")
rt = api.get_space_runtime("your-name/your-space")
print("stage:", rt.stage)
print("hardware:", rt.hardware)
print("requested_hardware:", rt.requested_hardware)
print("sleep_time:", rt.sleep_time)
print("raw:", rt.raw)
```

### 17.2 Torch/CUDA snapshot

```python
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
```

### 17.3 Minimal authenticated Space API call

```python
from gradio_client import Client

client = Client("your-name/your-space", token="hf_...")
print(client.view_api())
result = client.predict("test", api_name="/predict")
print(result)
```

### 17.4 Retry wrapper for client calls

```python
import time
from gradio_client import Client

def predict_with_retry(client, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.predict(*args, **kwargs)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

client = Client("your-name/your-space", token="hf_...")
result = predict_with_retry(client, "hello", api_name="/predict")
```

### 17.5 Forward ZeroGPU request identity in Space-to-Space calls

```python
import gradio as gr
from gradio_client import Client

def process(prompt, request: gr.Request):
    x_ip_token = request.headers.get("x-ip-token", "")
    client = Client("owner/zerogpu-space", headers={"x-ip-token": x_ip_token})
    return client.predict(prompt, api_name="/predict")
```

---

### 17.6 Local Docker smoke test commands

```bash
# Build.
docker build -t local-space-test .

# CPU or non-GPU launch.
docker run --rm -p 7860:7860 --env-file .env local-space-test

# GPU launch.
docker run --rm --gpus all -p 7860:7860 --env-file .env local-space-test

# CUDA visibility check inside a CUDA base image.
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 17.7 Local torch/CUDA check inside your own container

```bash
docker run --rm --gpus all local-space-test python - <<'LOCALCHECK'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
LOCALCHECK
```



## 17A. Small Worked Scenarios

These scenarios are intentionally generic. They show how to reason, not which GPU to buy.

### 17A.1 Source works on ZeroGPU; duplicate starts on CPU and fails

**Situation.** You duplicate a public ZeroGPU Space. The duplicate builds but fails with CUDA unavailable.

**Interpretation.** The duplicate did not recreate the source runtime. This is expected if the duplicate started on free CPU or the requested hardware did not settle.

**First action.** Check actual runtime, not impressions.

```python
from huggingface_hub import HfApi
api = HfApi()
rt = api.get_space_runtime("your-name/duplicate-space")
print(rt.stage)
print(rt.hardware)
print(rt.requested_hardware)
```

**Fix path.** Request the intended hardware, restart after assignment, then rerun the smallest test.

### 17A.2 Source works on ZeroGPU; standard GPU migration fails with OOM

**Situation.** The source probably used a large ZeroGPU allocation. The standard GPU target is cheaper but smaller.

**Interpretation.** This is not meant to be a duplication failure. It is a capacity mismatch or loading-strategy problem.

**First action.** Reduce the workload before changing libraries.

- lower image size
- lower batch size
- lower concurrency
- load one pipeline instead of several
- confirm whether the source needed `large` or `xlarge`

**Fix path.** If the app still OOMs on the smallest meaningful input, choose a larger target or redesign memory use.

### 17A.3 Source works in browser; private duplicate fails through API

**Situation.** The duplicate is private. It works while you are logged in, but `gradio_client` fails.

**Interpretation.** This is usually auth path mismatch, not model failure.

**First action.** Test with a token that can read the private Space.

```python
from gradio_client import Client
client = Client("your-name/private-space", hf_token="hf_xxx")
print(client.view_api())
```

**Fix path.** Confirm token scope, organization access policy, `api_name`, and whether the Space is private or gated.

### 17A.4 ZeroGPU-to-ZeroGPU copy works in browser but app-to-app calls burn wrong quota

**Situation.** A frontend Space calls a ZeroGPU backend Space. Browser testing looks different from app-to-app testing.

**Interpretation.** The request identity path may not be forwarded.

**First action.** Preserve the `X-IP-Token` header when calling the downstream ZeroGPU Space.

```python
import gradio as gr
from gradio_client import Client

def call_backend(prompt, request: gr.Request):
    headers = {}
    token = request.headers.get("x-ip-token")
    if token:
        headers["x-ip-token"] = token
    client = Client("owner/backend-zerogpu-space", headers=headers)
    return client.predict(prompt, api_name="/predict")
```

**Fix path.** Test through the normal HF page first, then through the custom path.

### 17A.5 Local Docker passes but hosted Space fails

**Situation.** Local Docker builds and runs, but the hosted duplicate fails.

**Interpretation.** The remaining difference is likely hosted config: secrets, hardware, README YAML, cache, or Space-specific startup.

**First action.** Compare the hosted Space settings to the local assumptions.

- local `.env` vs Space secrets
- local cache vs fresh hosted build
- local GPU vs requested Space hardware
- local port vs `app_port` or Gradio default
- local branch vs Space branch/revision

**Fix path.** Reproduce the hosted environment assumption locally only if it helps. Otherwise, debug from hosted logs.

### 17A.6 Kernel image error appears only after Blackwell-backed ZeroGPU run

**Situation.** A Space that once ran on older assumptions now sees a CUDA kernel image error under current ZeroGPU.

**Interpretation.** The torch/CUDA wheel or compiled extension may not support the current GPU architecture.

**First action.** Capture the actual device and CUDA build.

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
```

**Fix path.** Move to a wheel and dependency stack that supports the observed GPU. Don't assume that an old H200-era or A100-era pin remains safe.

## 17B. More Checklists for Thick-but-Safe Migration

### 17B.1 Source Space reading checklist

- [ ] What SDK is used?
- [ ] Is it Gradio?
- [ ] Does the code import `spaces`?
- [ ] Is `@spaces.GPU` used?
- [ ] Is GPU work inside decorated functions?
- [ ] Does the Space depend on custom frontend behavior?
- [ ] Does it call another Space?
- [ ] Does it call private or gated models?
- [ ] Does it require secrets?
- [ ] Does it use persistent storage?
- [ ] Does it rely on large cache state?
- [ ] Does the README YAML specify `suggested_hardware`?
- [ ] Does it mention ZeroGPU size, duration, or quota?

### 17B.2 Target Space setup checklist

- [ ] Start private unless public release is intended immediately
- [ ] Confirm duplicate created under the correct user or organization
- [ ] Confirm default hardware did not remain CPU by accident
- [ ] Recreate variables
- [ ] Recreate secrets
- [ ] Confirm storage needs
- [ ] Confirm `app_file`
- [ ] Confirm `sdk_version`
- [ ] Confirm `python_version`
- [ ] Confirm dependency pins
- [ ] Run smallest hosted smoke test
- [ ] Only then test real workload

### 17B.3 Dependency-change checklist

Before changing `torch`, `diffusers`, `transformers`, `gradio`, or `huggingface_hub`, write down:

- current source version
- current duplicate version
- target version
- reason for change
- expected failure fixed by the change
- rollback plan

A dependency update without a specific expected fix is a guess. Guesses are sometimes necessary, but they should be labeled as guesses.

### 17B.4 Hardware-change checklist

Before changing hardware, write down:

- current hardware
- requested hardware
- reason for change
- whether the failure is build-time, startup-time, load-time, or inference-time
- whether cheaper smoke-test hardware is only temporary
- whether the target is meant to approximate source VRAM/generation

Hardware changes are expensive debugging tools. Use them deliberately.

### 17B.5 Local Docker checklist

- [ ] Can the repo build locally?
- [ ] Can the app start locally?
- [ ] Are local secrets provided without committing them?
- [ ] Is model download working?
- [ ] Does local Docker use CPU or GPU?
- [ ] If GPU, does `nvidia-smi` work inside container?
- [ ] Does torch see CUDA?
- [ ] Did you avoid treating local Docker as ZeroGPU reproduction?
- [ ] Did you return to hosted smoke testing once local preflight was useful enough?

## 17C. Compact Error-to-Action Map

| Error or message | Most useful first action | Common wrong reaction |
|---|---|---|
| `Build queued` | Wait briefly, check status/logs, confirm no capacity issue | Edit model code immediately |
| `Repository storage limit reached` | Inspect repo/storage size and large files | Change GPU hardware |
| `ModuleNotFoundError` | Fix dependency file | Change Space privacy |
| `No CUDA GPUs are available` | Check actual hardware and requested hardware | Assume torch is broken |
| `CUDA initialized before importing spaces` | Fix ZeroGPU import/init order or remove ZeroGPU semantics after migration | Randomly upgrade CUDA |
| `CUDA error: no kernel image is available` | Check torch CUDA build and GPU capability | Retry the same wheel repeatedly |
| `ZeroGPU illegal duration` | Reduce requested duration or size | Move immediately to larger normal GPU |
| `Quota exceeded` | Test standard HF page and identity path | Only debug API client |
| `401` / `403` | Check token, privacy, gated assets, org policy | Tune model parameters |
| `503` after build | Check startup logs and health timing | Assume permanent platform outage |
| OOM | Reduce memory use or choose suitable target | Change auth token |

## 17D. A Conservative Expansion Strategy

Use this order when the first duplicate works but the real workload is not stable yet:

1. smallest input
2. smallest model path
3. one request at a time
4. same dependency stack
5. same hardware setting
6. restore one feature
7. increase input size
8. increase concurrency
9. change dependency only with a reason
10. change hardware only with a reason

The goal is not to be slow. The goal is to keep each failure interpretable.

## 17E. LLM and Agent Prompt Templates

These templates are optional, but they make the guide easier to use with ChatGPT-like assistants or automation agents.

### 17E.1 Ask for a migration plan

```text
I have a Hugging Face ZeroGPU Space and want to migrate it to a standard GPU Space.

Source Space:
- URL:
- public/private:
- known working path: browser/API/custom frontend/app-to-app

Target goal:
- standard GPU / ZeroGPU / not sure
- preferred cost constraint:
- preferred VRAM constraint:

Known files:
- README YAML:
- requirements.txt / pyproject.toml / Dockerfile:
- uses @spaces.GPU: yes/no/unknown

Observed failure:
- error:
- build log:
- runtime log:

Please classify the route, list missing evidence, and give the next safest actions without assuming secrets or hardware generation.
```

### 17E.2 Ask for a symptom-first diagnosis

```text
Diagnose this duplicated Space as a runtime-contract problem.

Symptom:
- duplicate starts on CPU / CUDA error / browser works but API fails / quota issue / build timeout / output differs

Evidence:
- source runtime:
- target runtime:
- README YAML:
- dependency pins:
- torch and CUDA:
- logs:
- secrets/variables status:

Return:
1. likely failure class
2. facts vs unknowns
3. first three checks
4. changes not to make yet
```

### 17E.3 Ask for a local Docker preflight

```text
I want a local Docker preflight before changing the hosted Space.

Repo shape:
- app file:
- requirements:
- Dockerfile present: yes/no
- model download path:
- secrets needed:

Machine:
- OS:
- GPU:
- NVIDIA Container Toolkit installed: yes/no/unknown

Goal:
- catch dependency/startup issues
- confirm torch sees CUDA
- compare with hosted Space

Give a minimal Dockerfile or docker commands, and list what local Docker cannot prove about Hugging Face ZeroGPU or hosted Spaces.
```

## 17F. Agent Output Quality Checklist

Before trusting an automated answer, check whether it does these things.

| Good agent behavior | Bad agent behavior |
|---|---|
| Classifies the route first | Treats every duplicate as a generic repo copy |
| Separates source evidence from target changes | Changes hardware, dependencies, and code at once |
| Marks unknown GPU generation as unknown | Assumes A100, H200, or Blackwell without logs |
| Checks README YAML and runtime state | Only reads `app.py` |
| Treats secrets as missing until confirmed | Assumes secrets copied |
| Tests the smallest hosted workload | Starts with the largest model path |
| Explains local Docker limits | Claims local Docker fully reproduces ZeroGPU |
| Gives reversible steps | Suggests deleting or rewriting source history first |

### A strong answer should mention

For ZeroGPU → standard GPU migration:

- duplicate is not the same as runtime reproduction
- `@spaces.GPU` may need to be kept, removed, or isolated depending on target
- torch/CUDA compatibility must match the target GPU and wheel
- secrets and private model access must be recreated
- local Docker can catch dependency and startup problems but not hosted quota/scheduling behavior

For ZeroGPU → ZeroGPU duplication:

- the target must actually be ZeroGPU
- secrets and variables still need review
- request identity and `X-IP-Token` can matter for API/client paths
- current ZeroGPU hardware generation and supported PyTorch versions should be checked against logs and docs

## 17G. Search Phrases for Humans and Agents

Use these phrases when looking up current behavior.

- `Hugging Face Spaces duplicate Space secrets variables`
- `Hugging Face Spaces free CPU default duplicate`
- `Hugging Face Spaces ZeroGPU Blackwell PyTorch supported versions`
- `Hugging Face Spaces configuration reference README YAML sdk_version python_version`
- `huggingface_hub duplicate_repo repo_type space`
- `huggingface_hub request_space_hardware get_space_runtime add_space_secret`
- `Hugging Face Spaces Run with Docker`
- `NVIDIA Container Toolkit Docker --gpus all torch cuda`
- `Gradio ZeroGPU X-IP-Token client`
- `CUDA error no kernel image available execution on device sm_120`

---

## 18. What Not to Do

Do not:

- assume duplicate means runtime match
- assume the duplicate is on GPU without checking runtime state
- assume public Variables and Secrets behave the same
- start with a maximum-size workload
- preserve old torch pins without checking the current GPU generation
- treat every standard GPU failure as a ZeroGPU quota issue
- treat every ZeroGPU failure as a normal CUDA issue
- change hardware, dependencies, frontend, and API path in one pass
- debug custom frontend quota behavior before testing the standard Space page

---

## 19. Final Takeaway

A ZeroGPU source Space has two separable parts:

1. the repo
2. the runtime contract

Duplication copies the first. You must recreate or intentionally change the second.

For **ZeroGPU → standard GPU**, the job is a migration: preserve the app while changing the execution model.

For **ZeroGPU → ZeroGPU**, the job is a faithful runtime copy: preserve the ZeroGPU execution model and verify the duplicate did not silently become CPU, lose secrets, resolve new dependencies, or lose request identity.

If you classify the target first and capture evidence before changing anything, most duplication failures become ordinary runtime-contract problems instead of mysterious Space failures.

---

## References

### Official Hugging Face docs

- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [More ways to create Spaces](https://huggingface.co/docs/hub/spaces-more-ways-to-create)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Run with Docker](https://huggingface.co/docs/hub/spaces-run-with-docker)
- [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Docker Spaces first demo](https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo)
- [Docker Spaces examples](https://huggingface.co/docs/hub/spaces-sdks-docker-examples)
- [Spaces Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode)
- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Using GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [User access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Managing your Space runtime](https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime)
- [Manage your Space](https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces)
- [HfApi reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [Hugging Face CLI reference](https://huggingface.co/docs/huggingface_hub/package_reference/cli)
- [Repositories next steps](https://huggingface.co/docs/hub/repositories-next-steps)
- [Create and manage a repository](https://huggingface.co/docs/huggingface_hub/en/guides/repository)
- [Repository settings](https://huggingface.co/docs/hub/repositories-settings)
- [Storage limits](https://huggingface.co/docs/hub/storage-limits)

### Gradio and client behavior

- [Using ZeroGPU Spaces with the Clients](https://www.gradio.app/docs/python-client/using-zero-gpu-spaces)
- [Gradio Python Client](https://www.gradio.app/docs/python-client/client)
- [`gr.Server` custom frontends and ZeroGPU quota issue](https://github.com/gradio-app/gradio/issues/13209)

### Local Docker and GPU containers

- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NVIDIA Container Toolkit overview](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker Desktop GPU support for Windows](https://docs.docker.com/desktop/features/gpu/)
- [Docker blog: Build Machine Learning Apps with Hugging Face Docker Spaces](https://www.docker.com/blog/build-machine-learning-apps-with-hugging-faces-docker-spaces/)
- [Runpod blog: Run Hugging Face Spaces on Runpod](https://www.runpod.io/blog/run-hugging-face-spaces-on-runpod)

### Runtime compatibility

- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [NVIDIA CUDA GPUs compute capability table](https://developer.nvidia.com/cuda-gpus)
