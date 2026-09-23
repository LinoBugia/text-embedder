---
source: "huggingface+chat+local-specs"
topic: "Open-source OCR for PDFs and documents with open models"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T00:00:00Z"
---

# Open-source OCR for PDFs and documents with open models

## 1. Background and overview

Optical Character Recognition (OCR) for PDFs and complex documents has moved far beyond classical engines like Tesseract. Modern open-source stacks combine:

- **Vision–language models (VLMs) and specialized OCR models** (e.g., GOT-OCR2, DeepSeek-OCR, dots.ocr, Florence-2, Surya).
- **PDF- and image-processing libraries** (e.g., `pypdfium2`, `pdfminer.six`, `PyMuPDF`).
- **Document-structure and table models** (e.g., Table Transformer / TATR, Granite-Docling).
- **Key-value extraction and layout models** (e.g., LayoutLMv3, Donut).

This document summarizes how to use open models for OCR on PDFs and related document types, combining:

- Internal specs and notes for **invoice OCR**, **CAD/engineering PDFs**, and **handwritten exam OCR**.
- The latest Hugging Face documentation, blog posts, model cards, and community resources (2024–2025).
- GitHub toolkits for modular OCR pipelines (e.g., Surya, MMOCR).

The goal is a practical, HF-centered knowledge base that lets you:

1. Choose a **baseline OCR model**.
2. Decide when to use an **all-in-one OCR/VLM** vs. a **modular stack**.
3. Design pipelines for **general documents**, **invoices**, **CAD PDFs**, and **handwritten or code-like text**.
4. Understand **deployment, performance, and environment pinning** considerations.

---

## 2. From official docs, blogs, and papers

### 2.1 Hugging Face blog: landscape and pipelines

Recent Hugging Face blog posts give a high-level map of the current OCR ecosystem and recommended practices:

- **“Supercharge your OCR pipelines with open models” (2025)** — explains how to:
  - Compare the current generation of OCR/VLM models by capabilities.
  - Decide when to fine-tune vs. use models out-of-the-box.
  - Integrate OCR with retrieval, document QA, and production serving (e.g., Inference Endpoints, vLLM/SGLang).  
  - Emphasizes metrics, latency, hardware constraints, and end-to-end pipeline design.  
  See: [Supercharge your OCR Pipelines with Open Models](https://huggingface.co/blog/ocr-open-models).

- **“Hall of Multimodal OCR VLMs and Demonstrations” (2025 community article)** — a curated “hall of fame” for modern multimodal OCR models such as DeepSeek-OCR, dots.ocr, OlmOCR-2, Nanonets-OCR2 and others, with demo Spaces and example prompts. Useful for quickly testing multiple models on the same page and understanding trade-offs in accuracy vs. cost vs. latency.  
  See: [Hall of Multimodal OCR VLMs and Demonstrations](https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms).

Key takeaways from these articles:

- OCR is now largely **multimodal**, often built on top of general VLMs with OCR-tuned prompts.
- **All-in-one VLMs** are excellent baselines but may struggle with extreme layout, tables, or tiny rotated text.
- **Hybrid stacks** (specialized detectors, recognizers, table models) remain best for demanding domains like invoices and CAD PDFs.
- Production deployments require early decisions about **quantization, batching, attention backends (SDPA vs. FlashAttention)**, and **version pinning**.

### 2.2 GOT-OCR2: general-purpose OCR-2.0

The Transformers docs describe **GOT-OCR2** as a general OCR model that covers:

- Plain document OCR.
- Scene-text OCR.
- Formatted documents (tables, charts, mathematical formulas, sheet music, etc.).
- In the reference implementation, the model outputs **plain text**, which you can further post-process to reconstruct structure (tables, formulas, music) with external tools.  
See: [GOT-OCR2 model doc](https://huggingface.co/docs/transformers/model_doc/got_ocr2).

Highlights:

- Supports a wide variety of document types with **one API**.
- Good default for **“OCR as text”** where exact layout or geometry is not critical.
- Easy integration in Python via `AutoProcessor` + `AutoModelForVision2Seq` or similar, depending on the implementation in your Transformers version.

### 2.3 DeepSeek-OCR: vision-text compression and PDF focus

**DeepSeek-OCR** is an open-source OCR model (≈3B MoE decoder) designed around the idea of **vision-text compression**:

- Large amounts of text are compressed into **high-resolution images** using a “DeepEncoder”.
- A decoder (`DeepSeek3B-MoE-A570M`) recovers the text from those images.
- This approach can cut token usage by up to ~20× at the cost of some decoding accuracy, especially at extreme compression ratios (≈60% accuracy at 20×, ≈97% at less aggressive ratios).  
See:
- Hugging Face model card: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR).
- Technical overviews and coverage articles summarizing the vision-text compression method and benchmarks.

Practical points (from the model card and environment guides):

- Official environment examples pin **Python 3.12**, **Torch 2.6**, specific **Transformers** and **FlashAttention** builds to ensure stable performance and avoid compilation errors.
- The model is now supported in **vLLM**, which simplifies scalable serving.
- When FlashAttention is unavailable or fragile (e.g., on Windows), you can run the model with **scaled dot-product attention (SDPA)** via `attn_implementation="sdpa"` or the default attention backend.

DeepSeek-OCR is particularly attractive when:

- You want to **compress long documents** cheaply for later decoding.
- You need efficient OCR in **long-context pipelines** (e.g., retrieval over many pages).
- You are willing to manage somewhat more complex dependencies (FlashAttention, CUDA versions).

### 2.4 Surya: open toolkit for OCR + layout + tables

**Surya** is an open-source toolkit focused on document OCR and layout analysis:

- OCR in **90+ languages**, with benchmarks that compare favorably to cloud services.
- **Line-level text detection** (any language).
- **Layout analysis** (tables, images, headers, etc.), reading-order detection.
- **Table recognition** support.  
See: [Surya GitHub](https://github.com/datalab-to/surya).

Surya is valuable when you want:

- A **single toolkit** instead of assembling detection + recognition + layout yourself.
- Strong multilingual performance.
- CLI and Python APIs for production use.

### 2.5 Table Transformer (TATR) and document-structure models

For tables and structured layout, the ecosystem includes:

- **Table Transformer (TATR)** — DETR-style models for both **table region detection** and **table structure recognition** (rows/columns/merged cells). The official repo ships training/eval scripts and GriTS / TEDS metrics.  
  See: [microsoft/table-transformer](https://github.com/microsoft/table-transformer).

- **Granite-Docling-258M** — a small vision–language model tightly integrated with the **Docling** toolkit for converting PDFs into structured formats (HTML/Markdown). It is useful as a **document-to-structure** baseline, especially when you want Docling’s unified data model.  
  See: [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M).

- **PaddleOCR-VL** — a compact multilingual VLM specialized for document parsing, with a collection of models covering layout and table tasks.  
  See: [PaddleOCR-VL collection](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl).

### 2.6 PDF and image processing building blocks

Foundational libraries used across pipelines:

- **`pypdfium2`** — Python bindings to PDFium; offers fast, deterministic PDF rasterization.
  - Typical usage for OCR: render pages at **300–400 DPI** with `scale = dpi / 72`, per-page `PdfPage.render()` to avoid API changes and to keep long-edge normalization and background handling under your control.
- **`pdfminer.six`** — extracts **vector text and coordinates** from PDFs.
  - For CAD PDFs and “born-digital” invoices, vector text extraction often recovers tiny rotated text and layout without rasterization.
- **`PyMuPDF` (fitz)** — another PDF toolkit with APIs like `Page.get_drawings()` and `Page.get_text("words")` to access vectors and word-level bounding boxes, plus rendering.

These libraries are the backbone for hybrid vector + raster pipelines: you exploit vector information where available and fall back to OCR only when necessary.

---

## 3. From model cards and dataset cards

This section focuses on open models and datasets you can use directly, especially those documented in Hugging Face model/dataset cards.

### 3.1 All-in-one OCR / VLM baselines

**Good starting points when you want fast results and minimal wiring:**

- **GOT-OCR2 (Transformers integration)**  
  - Broad coverage: regular documents, scene text, tables, formulas, and more.
  - Simple usage via the Transformers API, returning plain text that you can post-process.
  - Strong baseline if you do not need full layout reconstruction.

- **DeepSeek-OCR (≈3B MoE)**  
  - Vision-text compression for token-efficient OCR, especially valuable in document QA or retrieval pipelines.
  - Hugging Face model card includes **environment pins** and guidance for running with vLLM or standard Transformers.
  - Good choice for large-scale PDF processing if you manage CUDA and attention backends properly.

- **dots.ocr (≈3B)**  
  - Multilingual OCR model with on-device and Core ML–friendly workflows.
  - Focuses on producing both text and layout (JSON-style outputs), often with strong results on complex documents.

- **Granite-Docling-258M**  
  - Lightweight doc-conversion model integrated with Docling; produces Docling-friendly structured outputs (e.g., Markdown/HTML with layout annotations).
  - Excellent for pipeline stages where you want structured text with minimal custom post-processing.

- **PaddleOCR-VL**  
  - Compact multilingual document VLM for layout and OCR; often used as a baseline for non-English documents or multilingual corpora.

- **Surya**  
  - Toolkit rather than a single model; includes line detection, OCR, layout, and reading order, plus table recognition.
  - Particularly attractive if you want ready-made components and a multilingual focus.

### 3.2 Modular building blocks for CAD and technical documents

For CAD/engineering PDFs and similar technical drawings, internal specs and curated docs highlight the following components:

- **Vector-first text extraction:**  
  - Use `pdfminer.six` or PyMuPDF to extract all vector text first, preserving coordinates and rotations.
  - Only rasterize regions where vector text is missing; this preserves fidelity for dimensions, GD&T, and small callouts.

- **Text detection and recognition:**  
  - **MMOCR** for detection (e.g., DBNet, FCENet) and classical OCR pipelines.
  - **TrOCR** (Transformers) as a recognition head for cropped snippets (printed or handwritten checkpoints depending on the content).  
  - **DeepSeek-OCR / GOT-OCR2 / Florence-2** as page-level or region-level OCR fallback when you want generative OCR or need to handle complex layouts quickly.

- **Tables and bill-of-materials (BOM) structures:**  
  - **Table Transformer (TATR)** models for table detection and structure recognition.
  - Trained and evaluated on datasets like **PubTables-1M**, but applicable to engineering BOMs and schedules as long as you fine-tune.

- **Key–value extraction and title blocks:**  
  - **LayoutLMv3** when you already have OCR text and bounding boxes; good for key–value extraction on forms, title blocks, and structured layouts.
  - **Donut** when you prefer OCR-free “image → JSON” mapping for specific regions (e.g., cropped title blocks).

### 3.3 Handwriting and code-like OCR

Internal handwritten OCR docs and HF resources highlight:

- **TrOCR**:
  - Purpose-built for **handwritten text recognition (HTR)** and integrated into Transformers.
  - Printed and handwritten checkpoints are available; you can fine-tune them on your own domains (e.g., handwritten Python code in exam sheets).
- **Datasets:**
  - **IAM Handwriting Database / IAM lines** — standard dataset for handwriting recognition.
  - **HASYv2** — handwritten symbols, including punctuation and operators.
  - **CROHME** — math expressions, useful for complex layouts and symbol robustness.
  - **`gopika13/answer_scripts`** — a Hugging Face dataset of handwritten exam answers (including code), suitable for fine-tuning TrOCR on code-like text.

The recommended approach for code-like handwriting:

- Align student sheets to teacher templates.
- Extract only **handwritten ink** via differencing.
- OCR short crops per blank with TrOCR.
- Re-insert text into the template and evaluate via AST parsing and sandboxed tests.

---

## 4. Community, forums, and GitHub insights

Community threads and GitHub issues fill important gaps that official docs do not always cover. Key categories:

### 4.1 Attention backends and CUDA / FlashAttention

For DeepSeek-OCR and similar heavy VLMs:

- PyTorch wheels encode a specific **CUDA runtime** (e.g., cu118, cu121, cu124).
- FlashAttention builds must match both your **CUDA toolkit** and **Torch** version; mismatches often cause compilation errors.
- Community Windows guides recommend:
  - Installing a matching **Torch + CUDA** wheel from the official PyTorch index.
  - Installing a prebuilt FlashAttention wheel (e.g., from Hugging Face community repos) that matches the Torch/CUDA version.
  - Falling back to **SDPA** (`attn_implementation="sdpa"` or `"flash_attention_2"` where available) when FlashAttention is unstable or unavailable.

These details matter for reliable long-context OCR with DeepSeek-OCR or GOT-OCR2.

### 4.2 pypdfium2 and rendering policies

Internal invoice OCR specs codify best practices that align with external docs:

- Use **per-page `PdfPage.render()`** rather than deprecated batch APIs.
- Normalize by **long edge** (e.g., ≈1024 px for CPU, ≈1536 px for GPU), preserving aspect ratio and using an opaque white background.
- Apply a strict **resize policy** in TorchVision:
  - Prefer `InterpolationMode.NEAREST_EXACT` when available.
  - Use a feature probe and only fall back to `NEAREST` with a clear warning on older TorchVision versions.

This ensures deterministic PDF → image behavior across machines and CI.

### 4.3 Date, money, and semantic post-processing

For invoice OCR and other structured documents, robust parsing requires more than raw text:

- **Date parsing**:
  - Use `dateparser` with strict settings:
    - `REQUIRE_PARTS` for day/month/year.
    - `PREFER_DATES_FROM="past"`.
    - Explicit year bounds (e.g., [1990, today]).
  - Keep looser parsing only for non-critical fields.

- **Monetary amounts**:
  - Use **price-parser** plus **Babel/CLDR** currency metadata.
  - Infer decimal separators at the document level (dot vs. comma).
  - Filter out percentages and spurious tokens.

- **Semantic de-duplication and vendor detection**:
  - Use **SentenceTransformers (SBERT)** for embeddings.
  - Normalize vectors and tune thresholds for section scoping and duplicate suppression.

These practices come from an internal **Invoice OCR spec** that consolidates multiple iterations into a single, consistent policy.

### 4.4 Fine-tuning TrOCR and handwriting pitfalls

HF forum threads and internal notes highlight:

- Start from **handwritten** or **stage-1** pretraining checkpoints when adapting TrOCR to handwriting; they converge faster and reach better CER/WER.
- Pay attention to **vocabulary and tokenization**, especially for code-like text:
  - Ensure that characters like `:`, `_`, `()`, `[]`, `{}`, `#`, `=` are tokenized in a stable way.
- Use realistic augmentations:
  - Slight blur, noise, affine transforms, and illumination changes.
  - Avoid distortions that change the aspect ratio along the text baseline.

### 4.5 CAD / engineering drawing pipelines

Internal CAD OCR notes and external papers converge on a hybrid, multi-stage approach:

- **Vector-first**: parse vector text and geometry wherever present (CAD exports often preserve text and dimensions as vectors).
- **Local orientation**: treat each text region separately and avoid global deskew on multi-angle drawings.
- **Table structure**: use TATR (or similar) for BOMs and schedules and only then OCR each cell.
- **Symbol detection**: treat GD&T, weld symbols, and similar icons as objects with their own detectors; link them to nearby text for semantics.
- **Scale inference**: determine scale from dimension lines rather than title-block scale alone, using regression over multiple dimension pairs.

---

## 5. Implementation patterns and tips

This section describes concrete pipeline patterns for different document types, focusing on open-source models and HF-compatible components.

### 5.1 General PDF documents (reports, forms, mixed content)

**Goal:** robust text extraction with reasonable layout awareness.

**Recommended baseline:**

1. **Detect type of PDF per page:**
   - If **programmatic** (vector text):
     - Extract text with **`pdfminer.six`** or **PyMuPDF**.
     - Keep word-level boxes and reading order.
   - If **scanned**:
     - Render with **`pypdfium2`** at 300–400 DPI, per page or region.

2. **OCR engine choice:**
   - Start with **GOT-OCR2** or **DeepSeek-OCR** as the main OCR engine.
   - For multilingual documents or when you want turn-key CLI tools, test **Surya** or **PaddleOCR-VL**.

3. **Layout and structure recovery:**
   - For lightweight structure and HTML/Markdown outputs, run the page through **Granite-Docling-258M** or Docling’s converters.
   - For tables, run **TATR** on rasterized pages or on Docling’s structured representation to recover row/column structure.

4. **Post-processing:**
   - Normalise whitespace, join lines, and fix hyphenation.
   - Optionally build a **section hierarchy** using SBERT embeddings and heuristics over headings vs. body text.

This pipeline is sufficient for many generic OCR needs (search indices, basic extraction for LLM pipelines, etc.).

### 5.2 Invoices and semi-structured business documents

Internal specs recommend a **geometry-first, VLM-based pipeline**:

1. **PDF to image:**
   - Use **`pypdfium2`** per page with long-edge normalization (e.g., 1024–1536 px) and opaque white background.
   - Resize with TorchVision using `InterpolationMode.NEAREST_EXACT` or a feature-probed fallback.

2. **OCR engine stack:**
   - Primary: **Florence-2** with `<OCR>` and `<OCR_WITH_REGION>`:
     - `<OCR>` for plain text.
     - `<OCR_WITH_REGION>` to obtain lines or regions with bounding boxes.
   - Fallback: a lighter VLM such as **InternVL** or a compact OCR model.
   - Legacy: Tesseract/PaddleOCR behind a feature flag for edge cases.

3. **Parsing and normalization:**
   - Map geometry and text to fields: vendor, dates, totals, line items.
   - Use **strict date parsing** (`dateparser`) and **currency-aware amount parsing** (price-parser + Babel).
   - Aggregate and deduplicate using SBERT embeddings + geometry.

4. **Environment and rollout:**
   - Pin Transformers, Torch, TorchVision, and pypdfium2 versions.
   - Use **dtype policies**:
     - GPUs: fp16/bf16 with SDPA.
     - CPUs: fp32 with eager attention.
   - Include golden-page tests to catch regressions when dependencies change.

This design prioritizes deterministic behavior, geometry-aware extraction, and robust normalization, which are essential for financial documents.

### 5.3 CAD / engineering PDFs and drawings

For CAD PDFs and engineering drawings, a **hybrid vector + raster stack** is recommended.

**Stage 0 — Vector text and geometry:**

- Extract vector text and coordinates via **`pdfminer.six`** or **PyMuPDF**.
- Only rasterize regions lacking vector text; this preserves font size, rotation, and precise geometry for dimensions and notes.

**Stage 1 — Raster gaps:**

- Render regions at 300–400 DPI with **`pypdfium2`**, keeping transforms to map pixels back to page coordinates.

**Stage 2 — Text detection:**

- Use **MMOCR** (DBNet/FCENet) for small, rotated text.
- Keep per-box orientation; avoid global deskew.

**Stage 3 — OCR recognition:**

- Use **TrOCR** on cropped, rotated snippets (printed checkpoint for dimensions, handwritten for annotations).
- Optionally use **DeepSeek-OCR / GOT-OCR2 / Florence-2** as page-level or region-level OCR baselines.

**Stage 4 — Tables and BOMs:**

- Detect and recover table structure with **Table Transformer (TATR)**; fine-tune if needed.
- Fill each cell with vector text or OCR output, then export CSV/HTML.

**Stage 5 — Key–value and metadata:**

- Apply **LayoutLMv3** or **Donut** to title blocks, legends, and summary tables for structured JSON outputs.

**Stage 6 — Symbols and GD&T:**

- Train a small oriented detector (e.g., via MMRotate) for GD&T and special symbols.
- Link symbols to nearest text and geometry to reconstruct semantics.

This pipeline is more complex than general OCR but necessary for high-fidelity CAD digitization and quantity take-off workflows.

### 5.4 Handwritten exams and code-like OCR

A reliable design for handwritten programming exams with templated blanks:

1. **Alignment and differencing:**
   - Align each student sheet to the teacher template using feature-based homography.
   - Difference aligned images to isolate only handwritten ink.

2. **Segmentation:**
   - Identify connected components, cluster by y-coordinate, and crop per blank/line.
   - Avoid full-page OCR; process short code snippets instead.

3. **OCR model:**
   - Start with `microsoft/trocr-base-handwritten`, then fine-tune on your dataset (e.g., `gopika13/answer_scripts`).
   - Ensure tokenizer and vocab handle common programming punctuation.

4. **Program reconstruction and grading:**
   - Insert OCR’d strings back into the template code.
   - Parse via `ast.parse`, reject dangerous nodes (`import`, `eval`, `exec`, suspicious dunder calls).
   - Run tests in a sandbox (e.g., `isolate` or containerized pytest).

This approach leverages known template structure to simplify OCR and focuses training on short, high-value snippets.

### 5.5 Command-line and end-to-end tools

Beyond low-level libraries and models, there are high-level tools focused on PDF conversion:

- **Marker** (open-source CLI + Python library) — optimized for converting PDFs, including low-quality scans, into Markdown/JSON, using a combination of OCR and layout heuristics.
- **Docling** — IBM’s open framework for document conversion using Granite-Docling and other models; integrates well with HF models.

Such tools can serve as baselines or infrastructure components in larger pipelines.

---

## 6. Limitations, caveats, and open questions

### 6.1 Model-related limitations

- **General OCR VLMs vs. domain-specific needs:**
  - GOT-OCR2, DeepSeek-OCR, dots.ocr, etc., perform well across many documents but may struggle with:
    - Extremely small or rotated text (common in CAD).
    - Highly structured tables with complex spanning headers.
    - Handwritten or mixed printed/handwritten text.
  - Hybrid stacks with dedicated detectors and recognizers still outperform one-shot models in these areas.

- **Vision-text compression trade-offs (DeepSeek-OCR):**
  - High compression reduces token count but introduces decoding errors; accuracy can drop significantly at very high compression ratios.
  - You must tune compression levels and validate performance against your requirements.

- **Multilingual coverage:**
  - Many models support multiple languages, but quality varies; evaluate Surya, PaddleOCR-VL, or language-specific checkpoints for non-English workloads.

### 6.2 Infrastructure and dependency pitfalls

- **CUDA and FlashAttention mismatches:**
  - Installing FlashAttention without matching Torch and CUDA versions leads to compilation errors, especially on Windows.
  - Prefer prebuilt wheels that match your Torch/CUDA tags, or stick to SDPA where possible.

- **TorchVision interpolation behavior:**
  - Different TorchVision versions have different interpolation enums and behaviors.
  - Using `InterpolationMode.NEAREST_EXACT` with a feature probe prevents subtle raster differences between environments.

- **PDF library changes:**
  - pypdfium2 and PyMuPDF periodically change APIs; pin versions and monitor changelogs.
  - Use per-page rendering APIs rather than bulk helpers tied to older versions.

### 6.3 Evaluation challenges

- **Metrics beyond CER/WER:**
  - For invoices, you care about **field-level accuracy** (e.g., correct amounts and dates), not just character-level CER.
  - For tables, you need structural metrics (TEDS, GriTS) and semantic checks (e.g., totals, units).
  - For CAD, you may need specialized metrics like **dimension-parse success rate** or **scale-fit residuals**.

- **Human-in-the-loop needs:**
  - Even strong models produce occasional catastrophic errors (e.g., wrong currency, misinterpreted minus signs).
  - For compliance-sensitive domains (finance, engineering, legal), plan for review UIs and provenance tracking.

### 6.4 Open questions and future directions

Some directions to watch or decide on for your own stack:

- How aggressively to adopt **vision-text compression** for large-scale document indexing.
- Whether to standardize on a single general OCR VLM (e.g., GOT-OCR2 or DeepSeek-OCR) with a modular fallback stack.
- How far to push **document-to-JSON** models (e.g., Donut, newer OCR-free architectures) vs. a layered approach (OCR → layout → KIE).
- How to integrate OCR outputs with retrieval and RAG pipelines:
  - Chunking strategy for long documents.
  - Embedding models and indexing structures.
  - Metadata and provenance (page, bbox, source model, confidence).

---

## 7. Quick-start recipes

### 7.1 GOT-OCR2 baseline on PDFs

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import pypdfium2 as pdfium

# 1) Render first page of a PDF
doc = pdfium.PdfDocument("doc.pdf")
page = doc[0]
pil_image = page.render(scale=300/72).to_pil().convert("RGB")

# 2) Run GOT-OCR2
model_id = "gaianet/got-ocr2-base"  # or official ID from HF docs/model card
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id)

inputs = processor(images=pil_image, return_tensors="pt")
out_ids = model.generate(**inputs, max_new_tokens=1024)
text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
print(text)
```

### 7.2 DeepSeek-OCR with SDPA (no FlashAttention)

```python
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pypdfium2 as pdfium

doc = pdfium.PdfDocument("doc.pdf")
page = doc[0]
pil_image = page.render(scale=300/72).to_pil().convert("RGB")

model_id = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="sdpa",  # stable on CPU/GPU when FlashAttention is unavailable
)

prompt = "You are an OCR engine. Read all text in this page and output plain text."
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

out = model.generate(
    **inputs,
    images=[pil_image],
    max_new_tokens=1024,
)
decoded = tokenizer.decode(out[0], skip_special_tokens=True)
print(decoded)
```

### 7.3 Hybrid vector-first pipeline (CAD-friendly skeleton)

```python
from pdfminer.high_level import extract_pages
import pypdfium2 as pdfium
from PIL import Image

# Vector text first
for page_layout in extract_pages("drawing.pdf"):
    for element in page_layout:
        # Inspect element types; TextContainer elements carry character-level text
        pass

# Rasterize only if needed
doc = pdfium.PdfDocument("drawing.pdf")
page = doc[0]
pil = page.render(scale=350/72).to_pil().convert("RGB")

# Now feed `pil` into:
# - text detector (MMOCR)
# - OCR recognizer (TrOCR)
# - or a VLM like DeepSeek-OCR / GOT-OCR2 / Florence-2
```

---

## 8. References / links

Below is a non-exhaustive but curated list of resources mentioned above.

### 8.1 Hugging Face docs and blog

- GOT-OCR2 Transformers docs: <https://huggingface.co/docs/transformers/model_doc/got_ocr2>
- DeepSeek-OCR model card: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>
- Hall of Multimodal OCR VLMs and Demonstrations: <https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms>
- Supercharge your OCR Pipelines with Open Models: <https://huggingface.co/blog/ocr-open-models>
- Granite-Docling-258M model card: <https://huggingface.co/ibm-granite/granite-docling-258M>
- PaddleOCR-VL collection: <https://huggingface.co/collections/PaddlePaddle/paddleocr-vl>
- TrOCR docs: <https://huggingface.co/docs/transformers/model_doc/trocr>
- LayoutLMv3 docs: <https://huggingface.co/docs/transformers/model_doc/layoutlmv3>
- Donut docs: <https://huggingface.co/docs/transformers/model_doc/donut>

### 8.2 Toolkits, libraries, and GitHub repos

- Surya OCR toolkit: <https://github.com/datalab-to/surya>
- Table Transformer (TATR): <https://github.com/microsoft/table-transformer>
- pdfminer.six: <https://pdfminersix.readthedocs.io/en/latest/tutorial/highlevel.html>
- pypdfium2: <https://pypi.org/project/pypdfium2/>
- PyMuPDF docs: <https://pymupdf.readthedocs.io/en/latest/page.html>
- MMOCR: <https://github.com/open-mmlab/mmocr>
- MMRotate: <https://github.com/open-mmlab/mmrotate>

### 8.3 Datasets

- PubTables-1M: <https://huggingface.co/datasets/bsmock/pubtables-1m>
- DocLayNet v1.1: <https://huggingface.co/datasets/docling-project/DocLayNet-v1.1>
- Gopika13 Answer Scripts: <https://huggingface.co/datasets/gopika13/answer_scripts>

### 8.4 Other helpful articles and tools

- Marker (high-performance PDF→Markdown/JSON tool): see e.g. “High-Performance OCR Applications for Low-Quality PDF” (community write-up).
- OpenCV feature-based homography tutorial: <https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html>
- PyImageSearch image alignment and registration tutorial: <https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/>
- IAM Handwriting (via various mirrors) and HASYv2 (handwritten symbols): <https://github.com/MartinThoma/HASY>
