---
source: "huggingface+chat+files+web"
topic: "Handling Large Datasets: streaming, sharding, and resource-aware pipelines"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T10:23:46.022777+00:00"
---

# Handling Large Datasets: Streaming, Sharding, and Resource-Aware Pipelines

This knowledge base summarizes practical patterns for working with *large* datasets (tens to hundreds of gigabytes and beyond) in machine learning workflows, with emphasis on:

- Hugging Face Datasets (map-style vs iterable, streaming).
- PyTorch and TensorFlow data pipelines.
- Parquet/Arrow and sharded storage layouts.
- Multi-process / multi-GPU usage.
- Audio/image-heavy datasets and their RAM/IO behavior.
- Platform constraints such as Kaggle’s storage layout.

It intentionally repeats key ideas from multiple angles so each section can be read in isolation.

---

## 1. Background: what “large” really means

“Large” is relative to your resources. A dataset becomes *large* when:

- It cannot be fully loaded into **RAM** (e.g., 32 GB RAM vs 200 GB dataset).
- It cannot be duplicated in **disk** (e.g., 150 GB raw + 150 GB preprocessed on a 256 GB machine).
- It saturates **I/O or network** if you try to read it naively each epoch.
- Preprocessing it *all at once* takes many hours and blocks experimentation.

Typical constraints:

- **RAM**: 16–64 GB on workstations, sometimes less on shared servers or Kaggle.
- **GPU memory**: 8–24 GB (consumer) to 80 GB (A100/H100), but must hold model + activations, leaving limited space for large batches.
- **Disk**: Kaggle-style setups may allow 100+ GB of read-only inputs but only ~20 GB of writable, persistent storage.
- **Network**: remote datasets accessed via HTTP, S3, GCS, or HF Hub, where re-reading the entire dataset each epoch is too slow or costly.

For such setups, the classical workflow

> load everything → preprocess everything → save a full preprocessed copy → train

breaks down. The solution is a different mental model:

> **stream and transform** instead of **preprocess and save everything**.

---

## 2. New mental model: streaming instead of full materialization

### 2.1 Old model (that fails at scale)

The “small data” model:

1. Load full dataset into RAM or a single giant DataFrame.
2. Run heavy preprocessing (featurization, tokenization, augmentations).
3. Save a full preprocessed copy to disk.
4. Train models using only the preprocessed copy.

Problems at large scale:

- You need **2× disk** (raw + preprocessed).
- Any change to preprocessing requires paying the full ETL cost again.
- You pay big one-time costs before even running a single training step.

On constrained platforms (like Kaggle’s 20 GB `/kaggle/working` limit for persistent output), this model is simply impossible if raw data is already >20 GB.

### 2.2 New model (that scales)

The “large data” model:

- Treat the dataset as a **stream of samples**, not a single in-memory object.
- Use **lazy iteration**: only a small window of data is in RAM at a time.
- Do most preprocessing **on the fly in the data pipeline**, not as a separate ETL job.
- Store only compact, shard-based representations (Parquet, WebDataset tars, LMDB) if you must persist something.

Key consequences:

- You never create a full second copy of the dataset.
- You can train on larger-than-disk or larger-than-memory datasets (using streaming from remote sources).
- You can adjust preprocessing logic without rewriting 100+ GB of preprocessed outputs.

Modern libraries (Hugging Face Datasets, PyTorch `IterableDataset`, TensorFlow `tf.data`) are explicitly designed for this streaming model.

---

## 3. Storage layouts and platform constraints

### 3.1 Kaggle as a concrete example

Kaggle notebooks expose three main filesystem roots:

- `/kaggle/input`  
  - Read-only.  
  - Where attached datasets live (can be ≫100 GB total).

- `/kaggle/working`  
  - Read–write, **persisted** between runs.  
  - Hard limit of about **20 GB** for all outputs (checkpoints, small artifacts, etc.).

- `/kaggle/tmp` (or `/kaggle/temp`)  
  - Read–write, **ephemeral**; not saved between sessions.  
  - Allows significantly more usage than `/kaggle/working`, but is wiped after the session ends.

Implications:

- You cannot keep a full 150 GB preprocessed copy in `/kaggle/working` alongside 150 GB of raw data in `/kaggle/input`.
- Feasible patterns are:
  - Stream from `/kaggle/input` and preprocess **on the fly**.
  - Optionally write **temporary shards** to `/kaggle/tmp` during a single long run.
  - Or move heavy preprocessing off Kaggle (local machine/cloud), then bring back a **smaller, compact dataset** (e.g., filtered Parquet shards).

This pattern generalizes: whenever writable disk is much smaller than the dataset, avoid full persistent copies and instead rely on streaming + compact shards.

### 3.2 Columnar storage for big data: Parquet + Arrow

For large tabular or feature datasets, **Parquet** (with **Apache Arrow** in memory) is a strong default:

- **Columnar layout**: store columns contiguously to enable vectorized reads and better compression.
- **Row groups** and **column chunks**: allow skipping entire blocks based on metadata (e.g., statistics). This is crucial when filtering queries and reduces I/O.
- **Compression and encoding**: per-column choice of codecs (ZSTD, Snappy, etc.) and encodings (dictionary, run-length encoding) to keep on-disk size manageable.
- **Immutable files**: write-once files that can be partitioned and versioned.

Practical best practices from Arrow/Parquet docs and data-engineering guides:

- Target **row group sizes** around 128–512 MB for analytics workloads; this balances I/O efficiency and parallelism.
- Partition directories by high-level keys, such as `year=2024/`, `split=train/`, or `lang=en/`, so that entire directories can be skipped when filtering.
- For ML pipelines, aim for **shard sizes** of roughly 10k–100k examples per Parquet file (or chosen by byte size), so that you can easily parallelize over files without massive overhead.

---

## 4. Hugging Face Datasets for large-scale data

### 4.1 Map-style vs iterable-style

Hugging Face Datasets offers two core dataset types:

- **`Dataset` (map-style)**  
  - Has a fixed `len(dataset)` and supports random access: `dataset[i]`.  
  - Ideal when the dataset fits comfortably on disk and in cached Arrow format.  
  - Enables many powerful table-style operations (`map`, `filter`, `shuffle`, `train_test_split`).

- **`IterableDataset` (iterable-style)**  
  - Acts as a stream: you iterate with `for example in dataset`.  
  - Lazily loads data; ideal for **hundreds of GBs** or streaming from remote sources.  
  - Operations like `map` and `filter` are applied **on the fly** during iteration.  
  - Often faster and more memory-friendly for heavy, sequential training loops.

High-level rule:

- If your data **comfortably fits** in local storage and your transformations are lightweight → use **map-style `Dataset`**.
- If your data is **very large** or remote, or you want to avoid building huge caches → use **`IterableDataset`** via **streaming** or `to_iterable_dataset`.

### 4.2 Streaming mode (`streaming=True`)

`load_dataset(..., streaming=True)` returns an `IterableDataset` that streams rows as you iterate; it never downloads or materializes the entire dataset up front.

Two common cases:

1. **Remote datasets on HF Hub**  
   - Data is fetched progressively over HTTP.  
   - Great when the dataset is larger than your local disk or you only need a subset.

2. **Local files with streaming**  
   - Data is read lazily from local storage without converting everything to Arrow.  
   - Useful when you already have large CSV/JSONL/Parquet files and want zero-copy iteration.

Key operations on streaming datasets:

- `map` and `filter` still exist, but they operate lazily at iteration time.
- `shuffle` has special semantics (uses a buffer/window rather than full-deck shuffling).
- `take(n)` lets you quickly peek at the first `n` examples without reading the whole dataset.
- `to_iterable_dataset(num_shards=k)` (for map-style → iterable conversion) gives you fine control over sharding and works well with multi-worker setups.

### 4.3 `.map` vs collator for heavy transforms

A common failure mode with large datasets is putting heavy, tensor-producing transforms inside `dataset.map(...)` and asking it to keep those results:

- Each `.map` can materialize new columns and intermediate states in memory and in the Arrow cache.  
- For audio/image data, storing arrays for every example quickly exhausts RAM and disk.  
- For streaming datasets, heavy `.map` operations can blow up per-process memory when combined with many DataLoader workers and prefetching.

A better pattern for large datasets, especially for audio and vision:

- Keep raw data in the dataset (`audio`/`image` columns).  
- Move heavy operations (feature extraction, tokenization, augmentations) into a **data collator** that runs **per batch** just before feeding the model.  
- Configure `Trainer` or custom loops with `remove_unused_columns=False` so the collator still sees the raw columns.  
- For audio models, drop unnecessary tensors like `attention_mask` when the architecture does not use them; this can halve per-batch memory usage.

This pattern makes RAM usage scale roughly with **batch size**, not dataset size.

### 4.4 Audio-specific tips (TorchCodec, `Audio` feature)

When working with large audio datasets via HF Datasets:

- The `Audio` feature in recent versions returns a **TorchCodec `AudioDecoder`** object when accessed.  
- Call the decoder (e.g., `.get_all_samples()`) inside the collator to obtain waveform tensors and sampling rates.  
- If you need bytes or paths instead, cast the column to `Audio(decode=False)` and decode manually (e.g., with `soundfile` or TorchCodec).  
- Decode only the **head** of long clips (e.g., first 5–10 s) and enforce a consistent maximum length.  
- Avoid unnecessary attention masks when the model architecture does not require them; padding-only approaches often suffice for group-norm based models.

These practices reduce both memory spikes and CPU time when handling large, many-hours-long audio corpora.

### 4.5 Remuxing / re-sharding datasets

For some projects, you want to convert a public dataset into your own sharded layout (e.g., augmenting images, changing label schema, or compressing to JPEG). A robust pattern is:

1. Load the source dataset in **streaming mode**.  
2. Apply augmentations and metadata enrichments per example or per batch.  
3. Write out examples using a **streaming Parquet writer** (or WebDataset tars) with an explicit schema.  
4. Aim for shard sizes that balance I/O and parallelism (e.g., 25k–100k rows per Parquet file).  
5. Train from these shards later via map-style `Dataset` or `to_iterable_dataset`.

This pattern keeps RAM usage nearly constant while still allowing you to move heavy augmentations out of the training loop and into a one-time preprocessing step.

---

## 5. PyTorch and TensorFlow pipelines for massive data

### 5.1 PyTorch: custom `Dataset` / `IterableDataset`

For datasets larger than memory, PyTorch forum discussions and best-practice threads converge on similar recommendations:

- Implement a custom `Dataset` where:
  - `__init__` stores only **paths/indices**, not raw data.  
  - `__getitem__` opens and decodes a single sample from disk (image/audio/text), applies transforms, and returns tensors.  
- Alternatively, implement an `IterableDataset` that yields samples from a generator, especially for stream-like sources (e.g., endless logs, remote stream).  
- Avoid loading entire CSVs or HDF5 tables into memory when they have hundreds of GB; instead, memory-map or chunk reads.

For HDF5/LMDB/Parquet data:

- LMDB or memory-mapped formats generally outperform “load entire HDF5 into RAM” for huge datasets.  
- A pattern is: per-file DataLoader that reads shards of Parquet files, plus a higher-level loader or sampler that cycles through files.

Important knobs for `DataLoader`:

- `num_workers`: more workers = more parallel reads, but each worker holds its own batch in memory.  
- `prefetch_factor`: per-worker prefetch; reduce it (e.g., to 1) for huge batches to avoid multiplicative RAM usage.  
- `persistent_workers`: usually `False` if you want workers to release memory cleanly between epochs on constrained systems.

### 5.2 TensorFlow: `tf.data` and TFDS

The `tf.data` API is built specifically for efficient, scalable pipelines:

- `tf.data.Dataset.from_tensor_slices` for small arrays (not recommended for huge data).  
- `TextLineDataset`, `TFRecordDataset`, `FixedLengthRecordDataset`, and custom readers for large binary formats.  
- Use `map`, `batch`, `shuffle`, `prefetch`, and `interleave` to build a pipeline that overlaps I/O and compute.

Performance guidance from TensorFlow docs and TFDS tips:

- Always end pipelines with `.prefetch(tf.data.AUTOTUNE)` to overlap CPU and GPU work.  
- Use `.cache()` only when the dataset fits in RAM or on fast local SSD; avoid caching 100+ GB on slow disks.  
- Apply `num_parallel_calls=tf.data.AUTOTUNE` in `map` to parallelize transformations.  
- Measure with the TensorFlow Profiler to find I/O bottlenecks and tune buffer sizes.

Combined with sharded TFRecord or Parquet files, `tf.data` can handle multi-terabyte datasets as long as the underlying storage and network are adequate.

---

## 6. Multi-process, multi-GPU, and distributed scenarios

### 6.1 Splitting work across processes and nodes

When training in distributed setups (DDP, DeepSpeed, HF `accelerate`), you need each rank to see a distinct slice of data.

With Hugging Face Datasets:

- Use `datasets.distributed.split_dataset_by_node(ds, rank, world_size)` to divide map-style datasets across ranks for both PyTorch and streaming.  
- For `IterableDataset`, you can also rely on built-in rank-aware behavior (especially with `Accelerate`), but avoid combining both splitting methods accidentally.  
- Ensure shard counts are divisible by `world_size` to avoid some ranks doing disproportionate amounts of work or wasting I/O on skipped records.

### 6.2 Concatenating datasets and per-rank outputs

Common pattern for heavy preprocessing or distillation:

1. Each rank/process maps over a subset of the dataset and writes results to its own folder via `save_to_disk` or `write_dataset`.  
2. After all ranks finish, load each processed dataset and combine them with `concatenate_datasets`.  
3. Optionally run a final `shuffle` or `sort` on the concatenated dataset.

Important details:

- All shards must share the same schema (same columns and compatible types).  
- If they differ, cast/normalize columns before concatenation.  
- For iterative data (logs, multi-epoch distillation), you may want `interleave_datasets` instead of pure concatenation to mix sources more uniformly.

### 6.3 Sharding iterable datasets

Large iterable datasets are often composed of many files (shards). Practical rules:

- If an `IterableDataset` has **only one shard**, `.shard(num_shards, index)` provides limited flexibility; use `skip` / `take` or `split_dataset_by_node`, but you lose randomization and fine-grained control.  
- If you control storage, **pre-split** large files into many smaller files and pass them as multiple data files; each file becomes a shard.  
- Or load a local map-style dataset first, then call `to_iterable_dataset(num_shards=k)` to create a virtually sharded iterable with your chosen `k`.

Shards enable:

- Better worker utilization (`num_workers` > number of shards becomes counterproductive).  
- Cleaner separation of data across nodes/ranks.  
- More stable and reproducible shuffling when combined with per-epoch seed management.

---

## 7. Subsetting, partial downloads, and dataset portions

You rarely need every row of a huge dataset. Techniques to limit scope:

- **Dataset slicing**: HF Datasets support split syntax like `"train[:10%]"` or `"train[10%:20%]"` for quick subsets.  
- **Filtering on metadata**: keep only rows where `pass_rate == 1.0`, remove specific source datasets, or filter by language/length/quality flags.  
- **Streaming + filter**: on remote datasets, stream and filter lazily so you only download and retain matching rows.  
- **Column projection**: select only the columns you truly need (e.g., `["id", "question", "solution"]`) to reduce transfer and memory.

For extremely large multi-split datasets (e.g., separate `cpp` and `python` splits), load only the split you actually need instead of all splits at once; ensure you pass the correct `split` name rather than trying to treat a split folder as a separate config.

---

## 8. Large multimedia datasets: audio, images, and beyond

### 8.1 Audio datasets at scale

Key pitfalls with large audio datasets:

- **Decoding overhead**: naive decoding of full clips can create massive temporary arrays; decode only the portion you need and truncate early.  
- **Attention masks**: for some models (e.g., certain Wav2Vec2 checkpoints), masks are optional; dropping them reduces per-batch tensor count and memory cost.  
- **`map` vs collator**: heavy feature extraction via `.map` multiplies memory usage, especially with `return_tensors="pt"`; prefer collator-based transforms.  
- **DataLoader workers**: multiple workers with large `prefetch_factor` multiply memory usage by `workers × prefetch_factor`; for large arrays, start with `num_workers=0` and `prefetch_factor=1`, then scale carefully.

Recommended pattern:

- Keep audio as raw files or `Audio` features in the dataset.  
- Decode and featurize inside a collator that:  
  - Converts multi-channel to mono if needed.  
  - Truncates or pads to a fixed maximum length.  
  - Builds model inputs (e.g., `input_values`) and labels only for the current batch.  
- Tune batch size, gradient accumulation, and gradient checkpointing to trade runtime for RAM.

### 8.2 Image datasets and diffusion / LoRA workflows

For high-resolution image datasets (e.g., for LoRA or diffusion training):

- **Curate before scaling**: 500–3000 images can be enough for style or concept LoRAs if chosen carefully; excessive, low-quality data can slow convergence or hurt quality.  
- **Resolution vs VRAM**: doubling resolution roughly quadruples pixel count; expect to reduce batch size or rely on gradient accumulation at 1024×1024.  
- **Sharding and remuxing**: convert large image datasets to sharded Parquet or WebDataset tar files with embedded JPEG/PNG bytes and labels; this improves I/O and enables streaming.  
- **Augmentations**: implement augmentations in the data pipeline or in a one-time remux script; avoid baking dozens of variations per image into the persisted dataset unless you know you need them.

As with audio, streaming and on-the-fly transforms generally beat precomputing huge, static augmented datasets.

### 8.3 Video and sequence datasets (briefly)

For video or extremely long sequences:

- Favor **frame-level or clip-level shards** (e.g., 1–10 seconds per clip) rather than full movies.  
- Leverage specialized formats (e.g., chunked MP4 with indexing, WebDataset shards) so that you only read the needed segments.  
- Expect to rely more heavily on multi-worker I/O and asynchronous prefetch.

---

## 9. Practical recipes

### 9.1 Kaggle: 150 GB dataset with 20 GB persistent storage

Scenario: 150 GB of raw data in `/kaggle/input`, 20 GB limit in `/kaggle/working`.

Viable recipe:

1. **Do not** attempt to write a full preprocessed 150 GB copy to `/kaggle/working`.  
2. Implement a data loader that reads **directly** from `/kaggle/input` and preprocesses **on the fly**.  
3. Optionally, during a long run:
   - Write temporary Parquet shards or feature caches to `/kaggle/tmp`.  
   - Train from these shards within the same session.  
4. Persist only:
   - Model checkpoints and logs.  
   - Small derived artifacts (e.g., global statistics, splits).

If you need a reusable preprocessed version of the full dataset:

- Download the raw data to a local or cloud machine with sufficient disk.  
- Run a dedicated ETL job to create compact Parquet/WebDataset shards.  
- Upload the resulting dataset to Hugging Face Hub or Kaggle (in ≤20 GB shards).  
- Back in Kaggle, attach the processed dataset as `/kaggle/input/...` and train from that.

### 9.2 HF Datasets + PyTorch: streaming training loop

A typical pattern for huge text/image/audio datasets:

1. Use `load_dataset(..., streaming=True)` or `to_iterable_dataset(num_shards=k)` to obtain an `IterableDataset`.  
2. Wrap it in a PyTorch `DataLoader` with appropriate `num_workers` and `prefetch_factor`.  
3. Use a custom collator that:  
   - Decodes raw features (audio/image/text).  
   - Applies tokenization, feature extraction, or augmentations.  
   - Builds batch tensors (`input_ids`, `attention_mask`, labels, etc.).  
4. Train in a standard loop or via HF `Trainer`, setting `remove_unused_columns=False` to preserve raw columns.

This pattern gives constant RAM usage and allows training directly from datasets that do not fit into disk or memory in their fully decoded form.

### 9.3 TensorFlow: sharded TFRecord/Parquet + `tf.data`

For TF-based projects:

1. Store data in **sharded TFRecord or Parquet files**, each with 10k–100k examples.  
2. Create a `tf.data.Dataset` from the list of files.  
3. Use `interleave` to read from multiple files in parallel and `map` to parse/transform examples.  
4. Apply `shuffle`, `batch`, and `prefetch` to overlap I/O and training.  
5. Use the TF Profiler to confirm that your input pipeline keeps GPUs/TPUs busy.

This is the TensorFlow analogue of the HF Datasets + PyTorch streaming pattern.

---

## 10. Checklists and common pitfalls

### 10.1 Design checklist for large datasets

- [ ] Clarify which resource is the bottleneck: RAM, disk, network, or GPU memory.  
- [ ] Decide whether you *must* persist a preprocessed dataset, or whether on-the-fly transforms are acceptable.  
- [ ] Choose a storage format and layout (Parquet, WebDataset, LMDB, TFRecord) and shard sizes.  
- [ ] Decide between map-style `Dataset` and `IterableDataset` / streaming.  
- [ ] Implement a safe data collator that handles decoding, truncation, and batching.  
- [ ] Configure DataLoader / `tf.data` workers, prefetching, and caching conservatively at first; then tune.  
- [ ] For distributed training, plan how data is split across ranks and nodes to avoid duplication or missed samples.  
- [ ] Add unit tests / smoke tests for the pipeline on a tiny subset (e.g., 1k examples) before scaling up.

### 10.2 Pitfalls to avoid

- **Trying to “just load everything”** into a DataFrame when the dataset is 10× bigger than RAM.  
- **Duplicating data**: raw + preprocessed copies of multi-hundred-GB datasets on the same disk.  
- **Running heavy `.map` operations** on huge datasets without considering memory or disk cache usage.  
- **Using too many DataLoader workers** with large batches, causing RAM explosion due to prefetching.  
- **Assuming one format is universally optimal**: CSV is convenient but slow; Parquet/WebDataset/LMDB or TFRecord are more appropriate at scale.  
- **Ignoring platform constraints** (like Kaggle’s 20 GB limit) until late in the project.  
- **Forgetting to measure pipeline performance**; use profilers and logs to see whether the model or the data pipeline is the bottleneck.

---

## 11. References and further reading

Below is a short, curated list of high-signal resources on large-dataset handling, streaming, and pipelines.

### 11.1 Hugging Face Datasets and streaming

- Datasets: map-style vs iterable-style – when to use each.  
  - https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable  
- Streaming datasets (remote and local) – usage and caveats.  
  - https://huggingface.co/docs/datasets/en/stream  
- Using Datasets with PyTorch and distributed setups.  
  - https://huggingface.co/docs/datasets/en/use_with_pytorch  
- Main classes and `to_iterable_dataset` API.  
  - https://huggingface.co/docs/datasets/en/package_reference/main_classes  

### 11.2 PyTorch and large-data patterns

- “How to use dataset larger than memory?” – forum discussion on lazy loading.  
  - https://discuss.pytorch.org/t/how-to-use-dataset-larger-than-memory/37785  
- “Loading big dataset (bigger than memory) using pytorch” – HDF5 and custom Dataset patterns.  
  - https://discuss.pytorch.org/t/loading-big-dataset-bigger-than-memory-using-pytorch/123832  
- StackOverflow: memory-mapped datasets, LMDB vs HDF5.  
  - https://stackoverflow.com/questions/53576113/most-efficient-way-to-use-a-large-data-set-for-pytorch  

### 11.3 TensorFlow / tf.data

- `tf.data` guide – building efficient pipelines.  
  - https://www.tensorflow.org/guide/data  
- `tf.data` performance tips and profiler usage.  
  - https://www.tensorflow.org/guide/data_performance  
  - https://www.tensorflow.org/guide/data_performance_analysis  
- TensorFlow Datasets performance tips.  
  - https://www.tensorflow.org/datasets/performances  

### 11.4 Parquet and Arrow

- PyArrow Parquet reading/writing docs.  
  - https://arrow.apache.org/docs/python/parquet.html  
- Parquet file structure (row groups, column chunks).  
  - https://medium.com/data-engineering-with-dremio/all-about-parquet-part-03-parquet-file-structure-pages-row-groups-and-columns-d7c7e54a8311  
- Parquet performance tuning and best practices.  
  - https://dev.to/alexmercedcoder/all-about-parquet-part-10-performance-tuning-and-best-practices-with-parquet-1ib1  
- R4DS: Arrow chapter – larger-than-memory analytics.  
  - https://r4ds.hadley.nz/arrow  

### 11.5 General pipeline and performance resources

- HF forums: best practices for large datasets (streaming + DataLoader workers).  
  - https://discuss.huggingface.co/t/best-practices-for-a-large-dataset/137632  
- PyTorch Lightning and other discussions on custom loaders for large Parquet datasets.  
  - https://github.com/Lightning-AI/pytorch-lightning/discussions/17317  
- TensorFlow blog: high-performance data pipelines with new formats (e.g., Grain, ArrayRecord).  
  - https://developers.googleblog.com/en/building-high-performance-data-pipelines-with-grain-and-arrayrecord/  

These references reinforce the same themes: stream, shard, and design your input pipeline as carefully as your model when working with truly large datasets.
