---
source: "huggingface+chat+local-specs"
topic: "Hugging Face Datasets 3.6.0+ usage changes and end of dataset builder scripts"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T00:00:00Z"
---

# Hugging Face Datasets 3.6.0+ usage changes and end of dataset builder scripts

## 1. Background and overview

Hugging Face's `datasets` library went through a significant transition between the 3.6.x line and the 4.x series. The changes are not only about new features: they also change how you are supposed to _use_ the library and how datasets are hosted on the Hugging Face Hub.

At a high level:

- **3.6.0 is a “bridge” release.** Script‑based datasets stored on the Hub still work, and the CLI tools that rely on `trust_remote_code` are available. This makes 3.6.0 a convenient version for converting old script‑based datasets into Parquet/Arrow.  
- **4.0.0 is the breaking release.** It introduces new backends and new types, and it removes dataset _builder scripts_ from the loading path. Any dataset that still relies on a `dataset script` (such as `superb.py` or `librispeech_asr.py`) fails to load with `RuntimeError: Dataset scripts are no longer supported, but found <script>.py`.   
- **4.x strongly encourages Parquet‑backed and streaming‑friendly datasets.** Instead of running arbitrary Python on load, `datasets` now prefers pure data (Parquet/Arrow, JSON, CSV, image/audio/video files) plus a static `dataset_infos.json` and optional config metadata.   
- **Media handling changed substantially.** Audio and video decoding in 4.0.0 are delegated to TorchCodec + FFmpeg, which changes how you interact with `Audio` and `Video` features and adds new runtime requirements.   

These changes together mean that upgrading from 3.6.0 to 4.x is not just a version bump: it is a migration from “Hub‑hosted Python builder scripts” to “Hub‑hosted data + metadata”.

## 2. From official docs, releases, and guides

### 2.1 What changed in 3.6.0?

The 3.6.0 release itself did **not** introduce the breaking script removal; instead, it delivered smaller improvements and served as the last “fully script‑compatible” line.   

Key points from the 3.6.0 release notes and docs:

- Script‑based datasets (with Python builder files on the Hub) continue to work as they did in 3.5.x.   
- New features include:
  - Improvements to `DatasetDict.map`.
  - Documentation updates (for example, documenting `HF_DATASETS_CACHE`).
  - Performance and bug‑fixes for image and Spark interop.   
- There is no mention of script removal; at this point, Hub‑hosted dataset scripts are still a normal, supported pattern.

In practice, 3.6.0 is the **recommended version** for running the conversion CLI (`datasets-cli convert_to_parquet`) on script‑based datasets before moving production workloads to 4.x.   

### 2.2 What changed in 4.0.0?

The 4.0.0 release is the inflection point where both the runtime behavior and the Hub contract change. Important items from the release notes:   

- **New media backend (TorchCodec + FFmpeg).**
  - TorchCodec replaces SoundFile for audio and Decord for video.
  - Accessing `Audio` or `Video` columns now returns decoder objects (e.g. `AudioDecoder`, `VideoDecoder`) that expose methods such as `get_all_samples()` or `get_frames_in_range(...)`.   
  - TorchCodec requires **`torch >= 2.7.0`** and **FFmpeg ≥ 4**, and at 4.0.0 launch it is explicitly **not supported on Windows**; Windows users are told to keep `datasets<4.0`.   
- **New `Column` object and `List` type.**
  - `ds["text"]` can yield a lazy `Column` object, letting you iterate over a single column without materializing the full dataset in memory.   
  - The `List` feature type replaces `Sequence` as the canonical way to represent lists; `Sequence` becomes a helper that returns `List` (or dicts of lists) under the hood.   
- **New streaming and Hub integration.**
  - `IterableDataset.push_to_hub()` is added, letting you build streaming pipelines like `load_dataset(..., streaming=True).map(...).filter(...).push_to_hub(...)`.   

Most importantly for day‑to‑day use:

- **Dataset scripts are removed from the loading path.**
  - The “Breaking changes” section references “Remove scripts altogether”.   
  - Corresponding changes in the loader raise `RuntimeError: Dataset scripts are no longer supported, but found <script>.py` when `load_dataset` detects a script file in a Hub dataset repo.   
  - `trust_remote_code` is no longer supported, even for conversion.   

### 2.3 Official audio/media docs (TorchCodec behavior)

The updated `Audio` / media docs explain how decoding works in 4.x:   

- Audio decoding uses **TorchCodec**, which itself uses FFmpeg. You must install the `audio` extra (`pip install "datasets[audio]"`) and have FFmpeg available.   
- Accessing an `Audio` column with `decode=True` (the default) decodes on access, so even `print(dataset[0]["audio"])` can fail if TorchCodec/FFmpeg are not correctly installed.   
- You can opt out by using `dataset.decode(False)` or by casting the column with `Audio(decode=False)`, which returns a path or raw bytes that you decode yourself.   

Taken together with the release notes, this means that upgrading to 4.0.0 often requires both **environment changes** (Torch + TorchCodec + FFmpeg) and **code changes** (using `decode=False` in more places).

### 2.4 Streaming vs map‑style behavior

The docs emphasize the distinction between **map‑style** datasets and **streaming**/`IterableDataset`:   

- `load_dataset(..., streaming=True)` returns an `IterableDataset` that does **not** create a local cache: iteration recomputes transformations each time unless you explicitly remux to Parquet.  
- `load_dataset(...)` (without `streaming`) returns the classic map‑style `Dataset`, which can be cached locally and used with random access.  
- For large datasets, the recommended pattern is:
  1. Stream the raw data.
  2. Apply light transformations in streaming.
  3. Remux to Parquet once (using Arrow’s `write_dataset`).
  4. Load the Parquet‑backed dataset in subsequent runs.

These expectations existed before 4.0.0, but they become more central once script‑based datasets are removed and more workloads move to Parquet.

## 3. From model/dataset cards and Hub examples

### 3.1 `openslr/librispeech_asr` as a migration example

The `openslr/librispeech_asr` dataset is a canonical example of how script deprecation affects real users and how maintainers are expected to respond:   

- Originally, the dataset used a builder script on the Hub, so `load_dataset("openslr/librispeech_asr", ...)` worked with `datasets<4.0`.  
- After 4.0.0, loading it failed with `RuntimeError: Dataset scripts are no longer supported`.  
- The recommended workflow was:
  1. Create an environment with `datasets==3.6.0`.  
  2. Run `datasets-cli convert_to_parquet openslr/librispeech_asr --trust_remote_code` to execute the script once and write Parquet artifacts.  
  3. Push the Parquet files and updated `dataset_infos.json` to a dedicated branch such as `refs/convert/parquet`.  
  4. Test loading the converted dataset with `datasets>=4.0` and `revision="refs/convert/parquet"`.  
  5. Merge the converted branch into `main` so users can load it with 4.x without special parameters.

Once this conversion is done, **end users no longer depend on builder scripts or `trust_remote_code`**; they just read Parquet/Arrow plus metadata.

### 3.2 Tutorial datasets like `superb`

Many tutorials (for example, older ASR pipeline notebooks) use datasets such as `superb` that were originally script‑backed.   

- With `datasets==4.0.0`, calling

  ```python
  from datasets import load_dataset
  ds = load_dataset("superb", name="asr", split="test")
  ```

  now raises `RuntimeError: Dataset scripts are no longer supported, but found superb.py`.   

- Workarounds for reproducing such tutorials include:
  - Pinning `datasets<4.0` in a dedicated environment just for tutorial replication.  
  - Switching to a Parquet‑backed dataset (if the maintainers or community have already converted and updated the Hub repo).   
  - Contributing a conversion yourself (following the 3.6.0→4.x workflow above) if you maintain or can contribute to the dataset repo.

### 3.3 Hub datasets with TorchCodec‑related notes

Several audio datasets hosted on the Hub now explicitly mention TorchCodec, FFmpeg, and platform limitations in their READMEs, reflecting the 4.x behavior:   

- They warn that accessing an `Audio` column in 4.x _always_ triggers TorchCodec + FFmpeg, and that misconfigured environments will see errors such as `ImportError: To support decoding audio data, please install 'torchcodec'` or `RuntimeError: Could not load libtorchcodec`.   
- Some dataset cards suggest using `Audio(decode=False)` or `dataset.decode(False)` for inspection and preprocessing, then decoding in the collator or downstream code.   

These cards reinforce the pattern that **audio/video decoding is now a first‑class part of Datasets 4.x** rather than an incidental, SoundFile‑based detail.

## 4. From community, forums, and GitHub issues

### 4.1 GitHub issue: “Dataset scripts are no longer supported, but found superb.py”

The GitHub issue #7693 provides a concrete example of post‑4.0 script behavior:    

- A user follows an older Pipelines tutorial, calls `load_dataset("superb", name="asr", split="test")` with `datasets==4.0.0`, and gets

  ```text
  RuntimeError: Dataset scripts are no longer supported, but found superb.py
  ```

- The stack trace shows that `dataset_module_factory` detects the script file on the Hub and **deliberately aborts** instead of downloading and executing it.   
- Discussion around the issue confirms that:
  - This is _expected_ behavior in 4.0.0, not a bug.
  - The intended fix is to use a Parquet‑backed revision or pin `datasets<4.0` if conversion is not yet available.

### 4.2 Other script‑related reports

Multiple issues on GitHub show the same error for different datasets (`librispeech_asr.py`, `hest.py`, etc.), often with advice to downgrade to 3.6.0 until the dataset is converted:   

- Users attempting to load Hub datasets that still rely on scripts hit the same `RuntimeError` mentioning that scripts are no longer supported.  
- In several threads, maintainers or HF engineers recommend:
  - Downgrading `datasets` (e.g., `pip install datasets==3.6.0`) as a temporary workaround.  
  - Converting the dataset to Parquet using `datasets-cli convert_to_parquet` in the 3.6.0 environment.   

### 4.3 TorchCodec and audio decoding issues

Community threads and issues highlight common problems after the TorchCodec switch:    

- Missing FFmpeg on Colab/CI leads to immediate failures when printing or iterating over audio examples.  
- Mismatched Torch vs TorchCodec versions produce `RuntimeError: Could not load libtorchcodec` with a “probe chain” in the logs.  
- Recommended mitigations include:
  - Installing FFmpeg before importing `datasets` (e.g., `apt-get install ffmpeg` on Linux, `conda install ffmpeg` on Windows/WSL).  
  - Pinning Torch and TorchCodec to compatible versions listed in the TorchCodec README.   
  - Using `Audio(decode=False)` and deferring decoding to a collator or domain‑specific library (e.g., `soundfile`, `torchaudio`) when environment constraints make TorchCodec unreliable.   

### 4.4 Map‑heavy audio pipelines and RAM explosions

HF forums and internal specs warn about using `.map()` with batched audio feature extraction:   

- Precomputing large tensors via `.map()` over long audio clips can materialize huge Arrow tables and intermediate Python lists.  
- This is especially dangerous with multi‑process mapping (`num_proc>1`) and large `writer_batch_size`.  
- Recommended pattern:
  - Keep datasets relatively raw (paths, short metadata).
  - Do feature extraction inside the data collator at batch time.
  - Use streaming + Parquet remux for very large corpora.

These community reports translate into concrete usage guidelines for Datasets 4.x.

## 5. Implementation patterns and migration tips

### 5.1 Choosing a version strategy

A pragmatic strategy in 2025:

- **For migration and conversion work:**  
  - Use **`datasets==3.6.0`** as the “conversion environment” for any dataset that still depends on a builder script.  
  - Run `datasets-cli convert_to_parquet` with `--trust_remote_code` once, then push results to the Hub.   
- **For production training and evaluation:**  
  - Prefer **`datasets>=4.0.0`** once the datasets you rely on have Parquet/Arrow revisions.  
  - Pin compatible `torch`, `torchcodec`, and `datasets` versions; treat the TorchCodec README’s matrix as the source of truth.   
- **For tutorial or legacy code reproduction:**  
  - Pin `datasets<4.0` (for example, 3.6.0) in dedicated environments rather than downgrading globally.

This “two‑environment” approach isolates risky script execution to a controlled 3.6.0 context while keeping day‑to‑day training on the modern, script‑free 4.x stack.

### 5.2 Migrating script‑based datasets from the Hub

A generic migration recipe that follows HF community discussions and internal specs:   

1. **Inventory your datasets.**
   - Identify which Hub repos still contain Python builder scripts (e.g., `dataset_name.py`) and which ones already ship Parquet/Arrow.  
2. **Set up a 3.6.0 conversion environment.**

   ```bash
   pip install "datasets==3.6.0"
   ```

3. **Convert each script‑based dataset to Parquet/Arrow.**

   ```bash
   datasets-cli convert_to_parquet <repo_id> --trust_remote_code
   ```

   - This uses the old builder script one last time to produce static Parquet shards.   
4. **Push converted data to the Hub.**
   - Commit Parquet files and updated `dataset_infos.json` either to a dedicated branch (`refs/convert/parquet`) or to `main` if you are confident.   
5. **Test loading with 4.x.**

   ```python
   from datasets import load_dataset
   ds = load_dataset("<repo_id>", "<config>", split="<split>")  # plus revision=... if needed
   ```

   - Verify that no script‑related error appears and that features and splits match expectations.   
6. **Update downstream code and docs.**
   - Replace references to “builder scripts” and `trust_remote_code` with Parquet/Arrow terminology.
   - Update examples to use the Parquet‑backed revision and Datasets 4.x semantics.   

Once this workflow is complete, the dataset behaves like any other Parquet‑backed Hub dataset and remains compatible with future Datasets releases.

### 5.3 Working with audio in Datasets 4.x

Best‑practice patterns for audio workloads after the TorchCodec switch:    

1. **Environment bootstrap**
   - Install the audio extra and compatible media stack:

     ```bash
     # Example (Linux/WSL)
     conda install -c conda-forge "ffmpeg<8"
     pip install "torch==2.8.*" "torchcodec==0.7.*" "datasets[audio]"
     ```

   - Avoid nightly or vendor‑suffixed Torch builds unless they are explicitly supported by TorchCodec.   
2. **Preflight media check**
   - Before running training or notebooks, print version information and decode a single short audio sample to verify that TorchCodec + FFmpeg work.   
3. **Use `Audio(decode=False)` during inspection and mapping**
   - When exploring or mapping over a dataset, disable decoding:

     ```python
     from datasets import load_dataset, Audio

     ds = load_dataset("some/audio_dataset", split="train")
     ds = ds.cast_column("audio", Audio(decode=False))
     raw = ds[0]["audio"]  # path or bytes
     ```

   - Decode in the data collator or a separate preprocessing script, using `soundfile`, `torchaudio`, or TorchCodec directly.    
4. **Handle platform constraints explicitly**
   - On Windows, either:
     - Pin `datasets<4.0` until TorchCodec support is complete, or
     - Use WSL/containers that provide Linux‑style FFmpeg installation.   

These patterns minimize surprise decoding failures and keep memory usage predictable.

### 5.4 Streaming pipelines and Parquet remux

To build scalable pipelines aligned with the post‑4.0 expectations:   

- **Ingestion layer:**

  ```python
  from datasets import load_dataset

  ds = load_dataset("repo_or_url", split="train", streaming=True)
  ds = ds.map(light_transform).filter(predicate)
  ```

- **Remux layer (optional but recommended for reuse):**

  Use PyArrow’s `write_dataset` with an explicit schema and sensible row‑group/file limits to write Parquet shards.   

- **Serving/training layer:**

  Load the Parquet‑backed dataset (map‑style) with `load_dataset("path_or_repo", data_files=..., split=...)` or by pointing directly to the Hub repo.

This pattern gives you constant‑RAM streaming for ingestion plus fast local or Hub‑backed Parquet for training and evaluation.

### 5.5 Interaction with the rest of the HF stack

Within a full Hugging Face stack (Transformers, TRL, etc.), the Datasets changes have downstream implications:   

- **Transformers Trainer**
  - Use `processing_class` instead of `tokenizer` so that the processor (including feature extractors for audio) is saved with the model.  
  - Set `remove_unused_columns=False` when your collator needs raw `"audio"` or other non‑model inputs, especially when using `Audio(decode=False)`.   
- **TRL SFT pipelines**
  - Keep feature extraction in the collator; avoid precomputing large tensors via `.map()` on audio.  
- **PyTorch DataLoader**
  - On small VMs, use `num_workers=0` and consider setting `prefetch_factor=1` if you increase workers, to avoid multiplying RAM usage.   

These patterns match the direction of Datasets 4.x: keep raw data in the dataset, defer heavy computation to the collator or model, and rely on Parquet/Arrow for persistent storage.

## 6. Limitations, caveats, and open questions

Even with the migration plan in place, several caveats remain:

- **Script‑only datasets may remain broken for some time.**
  - If a dataset owner does not convert their repo to Parquet, end users must either pin `datasets<4.0` or create a fork where they perform the conversion themselves.   
- **Security vs flexibility trade‑off.**
  - Removing script execution improves security and reproducibility, but it also removes a flexible mechanism for complex data preparation. Some advanced use cases may now require out‑of‑band preprocessing pipelines (e.g., custom ETL code that writes Parquet to the Hub).  
- **TorchCodec ecosystem maturity.**
  - TorchCodec is still evolving; compatibility with different Torch builds and platforms can be brittle, especially on Windows or on GPU‑specific builds. Careful pinning and preflight checks remain necessary.   
- **Long‑term status of 3.6.0 as a bridge.**
  - The longer 3.6.0 is kept alive as a conversion environment, the more divergence accumulates between conversion scripts and current data. At some point, organizations may want to snapshot all critical script‑based datasets and retire 3.6.0 from active use.

Open questions you may want to track over time:

- Will future Datasets releases provide additional tooling for automatic script‑to‑Parquet conversion?  
- How quickly will the community convert widely used script‑based datasets, and which ones will remain legacy?  
- Will alternative media backends or configurations (e.g., different codecs, CPU‑only builds) emerge for environments where TorchCodec is hard to deploy?

## 7. References / Links

**Official docs and releases**

- Datasets releases page: [https://github.com/huggingface/datasets/releases](https://github.com/huggingface/datasets/releases)  
- Datasets 3.6.0 release: [https://github.com/huggingface/datasets/releases/tag/3.6.0](https://github.com/huggingface/datasets/releases/tag/3.6.0)  
- Datasets 4.0.0 release: [https://github.com/huggingface/datasets/releases/tag/4.0.0](https://github.com/huggingface/datasets/releases/tag/4.0.0)  
- Audio loading docs: [https://huggingface.co/docs/datasets/audio_load](https://huggingface.co/docs/datasets/audio_load)  
- Streaming docs: [https://huggingface.co/docs/datasets/stream](https://huggingface.co/docs/datasets/stream)  
- Map‑style vs iterable docs: [https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable](https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable)  

**Hub examples and community**

- `openslr/librispeech_asr` dataset card and discussions: [https://huggingface.co/datasets/openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)  
- GitHub issue #7693 “Dataset scripts are no longer supported, but found superb.py”: [https://github.com/huggingface/datasets/issues/7693](https://github.com/huggingface/datasets/issues/7693)  
- TorchCodec‑related issues:  
  - [https://github.com/huggingface/datasets/issues/7678](https://github.com/huggingface/datasets/issues/7678)  
  - [https://github.com/huggingface/datasets/issues/7707](https://github.com/huggingface/datasets/issues/7707)  

**TorchCodec and PyArrow**

- TorchCodec repo and version matrix: [https://github.com/meta-pytorch/torchcodec](https://github.com/meta-pytorch/torchcodec)  
- PyArrow `write_dataset` docs: [https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html)  
