---
source: "huggingface+chat+local-specs"
topic: "Open-source structured document OCR"
generated_by: "LLM assistant (web + attachments)"
generated_at: "2025-11-14T20:05:37Z"
---

# Open-source structured document OCR (images & PDFs)

## 1. Background and overview

**Structured document OCR** means extracting not only text but also layout, structure, and semantics from document images or PDFs. Instead of returning a flat string, a structured OCR system typically outputs:

- Reading-order text (headings, paragraphs, footers)
- Logical layout (blocks, sections, tables, figures, lists)
- Geometric information (bounding boxes, page coordinates)
- Structured formats such as JSON, XML, Markdown, or key–value maps
- Domain-specific fields (e.g. invoice numbers, totals, tax IDs, line items)

Typical document types:

- Business documents: invoices, receipts, purchase orders, contracts, forms
- Enterprise content: reports, manuals, scientific papers, policies
- Operational documents: shipping labels, tickets, medical forms
- Visually rich layouts: engineering drawings, CAD PDFs, brochures

From an engineering point of view, there are three broad families of open-source approaches:

1. **Classical modular pipelines**  
   - Separate **text detection**, **text recognition**, **layout analysis**, and **key information extraction (KIE)**.  
   - Example stacks: PaddleOCR + PP-Structure, docTR + LayoutParser + LayoutLM.

2. **OCR-free / sequence-to-sequence models**  
   - Single Transformer model maps an image directly to structured text (often JSON) without running a separate OCR engine.  
   - Representative models: Donut, some Florence-2 style pipelines, and newer document VLMs.

3. **Multimodal document transformers with external OCR**  
   - External OCR produces tokens + bounding boxes; a multimodal model (LayoutLM-like) performs classification, QA, or relation extraction.  
   - Representative models: LayoutLM, LayoutLMv2, LayoutLMv3 and derivatives.

Modern systems often mix these families with **vision–language models (VLMs)** and **LLMs** for schema mapping, normalization, and validation.

---

## 2. From official docs, blogs, and papers

### 2.1 PaddleOCR and PP-StructureV3

**PaddleOCR** is an Apache-licensed OCR toolkit that supports over 100 languages and provides text detection and recognition models for images and PDFs. The official documentation highlights **PP-StructureV3** as a complex document parsing solution that combines:

- Layout analysis (title, text, table, figure, list, header/footer, etc.)
- Table detection and structure recognition
- End-to-end document conversion into **Markdown and JSON** while preserving hierarchy and layout

The PaddleOCR 3.0 technical report describes PP-StructureV3 as an end-to-end framework that integrates layout analysis, table recognition, and structure extraction, achieving strong performance on benchmarks such as OmniDocBench and targeting forms, invoices, and scientific literature.  
Key links:

- PaddleOCR documentation: <https://paddlepaddle.github.io/PaddleOCR/main/en/index.html>  
- PP-StructureV3 pipeline usage: <https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/PP-StructureV3.html>  
- PaddleOCR GitHub: <https://github.com/PaddlePaddle/PaddleOCR>  
- PaddleOCR 3.0 technical report: <https://arxiv.org/abs/2507.05595>

### 2.2 docTR (Document Text Recognition)

**docTR** (by Mindee) is an Apache-2.0 licensed library focused on document OCR. Official docs and GitHub emphasize:

- A two-stage view of OCR: **text detection** and **text recognition**
- A unified `ocr_predictor` that chains detection and recognition
- Multiple backbones (DBNet-style detectors, CRNN/Transformer recognizers)
- Support for both PyTorch and TensorFlow backends

The docs describe the full OCR task as two consecutive steps, each backed by a dedicated architecture and accessible through high-level predictors. Community analyses and technical blogs repeatedly note that docTR performs strongly on structured documents (forms, reports) where layout fidelity matters, making it a good building block in structured OCR pipelines.

Key links:

- docTR docs: <https://mindee.github.io/doctr/latest/>
- docTR GitHub: <https://github.com/mindee/doctr>

### 2.3 Donut: OCR-free Document Understanding Transformer

**Donut (Document Understanding Transformer)** is an OCR-free model: it takes a document image plus a prompt and directly generates text (often JSON) without a separate OCR engine. The Transformers model documentation and original paper describe:

- A visual encoder + text decoder Transformer architecture
- Training on synthetic and real documents for tasks like:

  - Document classification  
  - Key information extraction (document parsing)  
  - Visual document QA

- Advantages over OCR-based pipelines:

  - No OCR error propagation
  - Fewer language/script constraints when trained appropriately
  - Simple end-to-end training objective (cross-entropy over tokens)

Common usage pattern:

- Use base Donut checkpoint (`naver-clova-ix/donut-base`) or a fine-tuned variant.
- Define a JSON schema for your document type (e.g. `{{"invoice_id": ..., "date": ..., "total": ...}}`).
- Fine-tune Donut on (image, JSON) pairs.
- At inference time, parse the output JSON string and validate.

Key links:

- Transformers Donut model docs: <https://huggingface.co/docs/transformers/en/model_doc/donut>  
- Donut GitHub: <https://github.com/clovaai/donut>  
- Donut paper: <https://arxiv.org/abs/2111.15664>

### 2.4 LayoutLM and LayoutLMv3

The **LayoutLM** family combines text, 2D positions, and visual features in a multimodal Transformer, making it a core component for structured document understanding based on OCR outputs.

- **LayoutLM (v1)** – originally released under an MIT-style license; widely used for forms and receipts.  
- **LayoutLMv2 / LayoutLMv3** – newer models adding more advanced image-text fusion and unified masking objectives.

The LayoutLMv3 model docs describe it as a general-purpose document pre-training model that can be fine-tuned on:

- Form understanding and key-value extraction
- Receipt understanding
- Document visual question answering (DocVQA)
- Document image classification and layout analysis

Recent work (2024+) shows LayoutLMv3-based models achieving strong performance on relation extraction and structured information extraction tasks in visually rich documents.

Key links:

- LayoutLMv3 model docs: <https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3>  
- LayoutLMv3 base model card: <https://huggingface.co/microsoft/layoutlmv3-base>  
- Recent LayoutLMv3-based relation extraction paper: <https://arxiv.org/abs/2404.10848>

### 2.5 Document AI overviews (Hugging Face blog and others)

Recent articles and blogs on Document AI emphasize that:

- Enterprises have large volumes of semi-structured documents (invoices, forms, receipts, letters) that can now be processed with **open models**, often with no per-document cost.  
- Building robust pipelines requires **combining OCR, layout models, VLMs, and LLMs**, not just one model.  
- Structured extraction often hinges on:

  - Handling tables robustly
  - Managing multi-page documents
  - Normalizing values (currency, tax, dates) and validating constraints

Notable resources:

- Hugging Face Document AI blog: <https://huggingface.co/blog/document-ai>  
- Comparative analyses of open-source OCR tools (e.g., Tesseract, PaddleOCR, docTR, Surya, GOT-OCR2, DeepSeek-OCR) and discussions of when to choose modular vs VLM-based stacks.

---

## 3. From model cards, dataset cards, and Spaces

This section lists key Hugging Face models and datasets relevant to structured document OCR.

### 3.1 General-purpose OCR and document VLMs

These models serve as strong baselines for reading documents and sometimes preserving structure:

- **GOT-OCR2** – General OCR model available in Transformers; supports plain documents, scene text, tables, charts, formulas, etc. It typically outputs text (often structured with Markdown or tokens that hint at layout).  
  - Model doc: <https://huggingface.co/docs/transformers/en/model_doc/got_ocr2>

- **DeepSeek-OCR** – Around 3B parameters; aims for fast, promptable image-to-Markdown OCR across documents and natural scenes.  
  - Example model: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>

- **dots.ocr** – A model that outputs JSON with text and layout in a single pass, useful as a building block for structured OCR pipelines that want explicit geometry.  
  - Example: <https://huggingface.co/rednote-hilab/dots.ocr>

- **Granite-Docling-258M** – A lightweight document conversion model focused on PDF/docx → structured formats, often used for layout-aware conversion and as a pre-processor for LLMs.  
  - Example: <https://huggingface.co/ibm-granite/granite-docling-258m>

- **PaddleOCR-VL** – A compact multilingual VLM for document QA and OCR-style tasks, bridging PaddleOCR’s detection/recognition stack with multimodal understanding.  
  - Example: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL>

These models are useful when you want **“OCR + some layout”** quickly, and can add custom logic or LLM post-processing for full structure.

### 3.2 Donut checkpoints

Key Donut checkpoints on Hugging Face include:

- Base pre-trained model: <https://huggingface.co/naver-clova-ix/donut-base>  
- Fine-tuned on CORD (receipt dataset): <https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2>  
- Community fine-tunes for DocVQA and other tasks (e.g., jinhybr’s models).

These models typically output JSON-like strings describing fields such as `date`, `total`, `items`, etc., making them well-suited for **invoice and receipt parsing** once fine-tuned.

### 3.3 LayoutLM-based models and datasets

Important components for modular structured OCR:

- **LayoutLMv3 base/large models** for token classification and relation extraction.  
  - Example: <https://huggingface.co/microsoft/layoutlmv3-base>

- **Datasets** commonly used for documents:

  - FUNSD – form understanding dataset with token-level annotations.  
  - CORD – receipt dataset for KIE (often used with Donut and LayoutLM).  
  - SROIE – scanned receipts with field annotations.  
  - DocVQA – visual question answering over documents.  
  - DocLayNet – large document layout dataset (e.g., docling-project/DocLayNet-v1.1 on Hub).  
  - PubTables-1M – large table dataset used by Table Transformer (TATR).

These datasets can be accessed via Hugging Face Datasets or original repositories and serve as starting points for fine-tuning document models.

### 3.4 Table and layout models

For structured document OCR, table and layout components are critical:

- **Table Transformer (TATR)** – Detects tables and reconstructs table structure using Transformer-based detection.  
  - Example: <https://huggingface.co/microsoft/table-transformer-detection>

- **DocLayNet & Docling tools** – Provide high-quality layout annotations and conversion tools for PDF documents; strongly related to table and region extraction.

- **LayoutParser** – Python library for layout analysis with pre-trained detectors.  
  - <https://layout-parser.github.io/>

- **MMOCR** – OpenMMLab OCR framework with text detectors and recognizers, including support for rotated text and small text elements.  
  - <https://github.com/open-mmlab/mmocr>

- **Deepdoctection** – Document analysis toolkit that wraps detectors, layout models, and some OCR engines under a unified interface.  
  - <https://github.com/deepdoctection/deepdoctection>

---

## 4. Community, forums, and comparative analyses

### 4.1 SOTA discussions on document understanding

Hugging Face forum threads and GitHub issues discuss the state-of-the-art in open document understanding:

- Community threads compare LayoutLMv3, Donut, Pix2Struct, GOT-OCR2, and other models for tasks like invoice parsing and document QA. Many practitioners report that:

  - **LayoutLMv3** is still very strong for token-level KIE when you already have OCR outputs.  
  - **Donut and Pix2Struct** are attractive for OCR-free pipelines but require substantial fine-tuning and careful schema design.  
  - Newer document VLMs (e.g., Qwen2.5-VL, Florence-family models, DeepSeek-OCR) offer strong baselines with fewer components, at the cost of heavy GPU requirements.

- For relation extraction and complex graphs (e.g., linking line items to totals), recent LayoutLMv3-based research shows that adding relation heads on top of token encoders can outperform simpler models on visually rich documents.

### 4.2 Non-LLM OCR technology reviews

Technical articles analyzing non-LLM OCR tools note that:

- **PaddleOCR** and **docTR** are among the strongest open-source engines for structured documents (forms, papers, business docs), with good accuracy and layout preservation.  
- Tesseract remains relevant but often lags behind modern deep-learning OCR engines for complex layouts.  
- The best results for document-heavy workflows usually come from **hybrid stacks** combining strong OCR, table models, and custom post-processing rather than plain text-only OCR.

### 4.3 Industry comparisons and invoice-specific tools

Industry blogs comparing invoice-processing solutions often highlight that:

- Open-source OCR engines (PaddleOCR, docTR, Tesseract) can achieve high quality if combined with domain-specific KIE.  
- Commercial APIs add convenience, hosting, and integrated validation, but their internal models are often similar to open research.  
- For teams with ML capacity, building in-house pipelines using open-source models gives more control over privacy, customization, and cost.

---

## 5. Implementation patterns and tips

This section synthesizes patterns from official docs, community practice, and local specs for invoices, PDFs, and CAD-like documents.

### 5.1 Pattern A: PaddleOCR + PP-StructureV3 for general documents

**Goal:** parse diverse business documents (reports, manuals, invoices) into Markdown/JSON with preserved layout.

**High-level pipeline:**

1. **Layout analysis**  
   - Run PP-StructureV3 on each page to detect regions: titles, paragraphs, tables, figures, lists, headers/footers.

2. **Text recognition**  
   - For text regions, run PP-OCRv5 (or later) recognition to extract multilingual text.

3. **Table recognition**  
   - For table regions, run the integrated table recognizer to reconstruct rows/columns and cell content.

4. **Export**  
   - Use built-in utilities to export Markdown and/or JSON with block-level structure.

Example sketch (simplified; refer to official PaddleOCR docs for exact APIs):

```python
from paddleocr import PPStructure, save_structure_res

engine = PPStructure(
    layout=True,   # enable layout analysis
    ocr=True,      # enable OCR
    show_log=True,
)

result = engine("invoice_sample.jpg")
save_structure_res(result, save_folder="out/")

for block in result:
    print(block["type"], block.get("text", ""), block.get("res", ""))
```

**When to use:**

- Baseline for end-to-end conversion of PDFs/images to Markdown/JSON.  
- Input for downstream LLMs (summarization, QA, RAG) that benefit from structured context.  
- Multilingual documents, especially where CJK scripts are important.

### 5.2 Pattern B: docTR + custom layout + LayoutLM for KIE

**Goal:** precise control over layout, fields, and post-processing.

**Pipeline:**

1. **OCR with docTR**  
   - Use `ocr_predictor` to get pages → blocks → lines → words with bounding boxes.

2. **Custom layout clustering**  
   - Cluster words or lines into columns and blocks based on coordinates.  
   - Detect tables with heuristics or a dedicated detector (e.g., Table Transformer).

3. **Token-level KIE with LayoutLM/LayoutLMv3**  
   - Convert tokens and boxes to model inputs (including page size normalization).  
   - Fine-tune LayoutLMv3 for token classification or QA (e.g., `"What is the invoice total?"`).

4. **Post-processing and validation**  
   - Map labels to a schema (`invoice_id`, `vendor`, `date`, `total`, `line_items`, …).  
   - Normalize values, run consistency checks (sum of line items ≈ total).

Example skeleton for docTR:

```python
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_images(["page1.png", "page2.png"])
result = model(doc)

tokens = []
for page_idx, page in enumerate(result.pages):
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                tokens.append({
                    "page": page_idx,
                    "text": word.value,
                    "bbox": word.geometry,  # normalized (x0, y0, x1, y1)
                })

# Next steps: layout clustering, KIE model, schema mapping...
```

**When to use:**

- You have or plan to collect bounding-box annotations for KIE.  
- You want explicit, explainable geometry and token labels.  
- You need to swap OCR engines or layout algorithms independently.

### 5.3 Pattern C: Donut / OCR-free models for image→JSON

**Goal:** direct generation of JSON from document images, especially for template-like documents (receipts, invoices, ID cards, business cards).

**Pipeline:**

1. **Schema design**  
   - Decide the JSON schema (field names, types, and nested structure).

2. **Dataset preparation**  
   - For each training example, pair a document image with a JSON string following the schema.

3. **Fine-tuning**  
   - Fine-tune Donut on this dataset with a maximum sequence length adequate for your JSON.

4. **Inference**  
   - At runtime, run the model with a prompt (e.g., `"Extract fields as JSON"`), decode, and parse the JSON string.

Simplified usage example:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json, torch

model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

image = Image.open("receipt.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=512)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
data = json.loads(sequence)
print(data)
```

**Pros:**

- Simple pipeline (one model call).  
- Naturally emits structured JSON.  
- Handles many layout variations if fine-tuned well.

**Cons:**

- Requires label-rich (image, JSON) training data.  
- Can hallucinate fields or produce malformed JSON; requires robust parsing and validation.  
- Larger models can be slow or GPU-intensive.

### 5.4 Pattern D: LayoutLMv3 + OCR for forms and receipts

**Goal:** high-accuracy token-level KIE for forms, receipts, and contracts where layout is critical.

**Pipeline:**

1. **OCR step** – PaddleOCR or docTR for tokens + bounding boxes.  
2. **Feature building** – Convert tokens, bounding boxes, and (optionally) image patches into LayoutLMv3 inputs.  
3. **Fine-tuning task:**

   - Token classification (`B-INVOICE_NO`, `I-INVOICE_NO`, `B-TOTAL`, `O`, …).  
   - Span-level classification or QA (ask questions like `"What is the total?"`).  
   - Relation extraction (linking amounts to labels, linking line items to totals).

4. **Post-processing** – Assemble labels into a JSON schema and validate consistency.

**When to use:**

- You have annotated datasets with bounding boxes and field labels (FUNSD-, CORD-, SROIE-style).  
- You need fine-grained control and interpretability.  
- You are comfortable training multimodal Transformers.

### 5.5 Pattern E: Open VLM-based invoice pipeline (Florence-2-style)

Local specs for an invoice pipeline built around **Florence-2** and similar open VLMs provide a concrete pattern that can be generalized to other models (GOT-OCR2, DeepSeek-OCR, Qwen2.5-VL, etc.). Key ideas:

- Use a **geometry-aware OCR prompt** (e.g., `<OCR>`, `<OCR_WITH_REGION>`) to read entire pages and/or specific crops (header, footer, line-item region).  
- Implement region-specific crops (e.g., `HEADER_CROP`, `FOOTER_CROP`) to recover:

  - Vendor and customer blocks  
  - Invoice metadata (ID, dates, currency, purchase order numbers)  
  - Totals and tax sections

- Combine model output with deterministic parsing:

  - Currency detection using ISO 4217 metadata and symbol maps.  
  - Regex-based anchors to locate “Subtotal”, “Tax”, “Total”, etc. in extracted text.  
  - Line-item grouping logic based on amounts, description lines, and heuristics about columns.

- Wrap the VLM calls behind a simple **OCREngine** interface so that you can later swap Florence-2 for another open model without rewriting the pipeline.

From the engineering notes:

- Track **Transformers version changes**, especially around image processors (`use_fast=True/False`), which can introduce small numerical differences and change borderline OCR outputs.  
- Maintain regression tests (golden invoices, strict JSON comparison) and CI checks that detect unexpected behavior changes when upgrading Torch, Transformers, or PDF libraries.  
- Pin versions in production and only relax pins after re-baselining metrics.

Although Florence-2 itself is not a “classic” OCR engine, this pattern demonstrates how an open VLM can be embedded as the core recognizer in a structured invoice stack, with domain-specific logic providing reliability and auditability.

### 5.6 Pattern F: Hybrid pipeline for CAD PDFs and engineering drawings

CAD PDFs and engineering drawings are a challenging but important special case for structured OCR:

- Text is small, sparse, and often rotated.  
- Many semantic elements (dimensions, callouts, legends) are graphics, not text.  
- PDFs frequently contain a mix of vector text and rasterized regions.

A practical open-source hybrid pipeline usually looks like this:

1. **Stage 0 – Vector text extraction**  
   - Use a PDF library such as `pdfminer.six` or `PyMuPDF` to extract embedded text and coordinates from the PDF directly.  
   - This often recovers a large fraction of the text with perfect fidelity, especially for dimensions and annotations saved as vector text.

2. **Stage 1 – Raster gaps**  
   - For regions with no vector text (e.g., scanned layers, images, legacy drawings), rasterize pages or selected areas at high DPI using `pypdfium2`.  
   - Maintain a mapping between raster coordinates and PDF coordinates.

3. **Stage 2 – Text detection for small/rotated text**  
   - Use an OCR detector tuned for arbitrary orientations and tiny text, such as MMOCR’s DBNet/FCENet or similar models.  
   - Optionally incorporate rotated detection (e.g., MMRotate) for dimension arrows and oblique tags.

4. **Stage 3 – Text recognition**  
   - Feed cropped patches to a strong recognizer like TrOCR or PaddleOCR’s recognizer; account for rotation and scaling.

5. **Stage 4 – Tables, BOMs, and title blocks**  
   - Apply **Table Transformer (TATR)** to detect tables (e.g., bills of materials, schedules).  
   - Use LayoutLMv3 or similar models to link quantities, part numbers, and descriptions.

6. **Stage 5 – Graph / semantic extraction**  
   - Build domain-specific logic to interpret dimension chains, tolerance notations, and callouts.  
   - Optionally train a document transformer (LayoutLM-like or DocTr-style SIE) to predict relations between text boxes and drawing elements.

Datasets like **DocLayNet** and **PubTables-1M** provide layouts and tables that can help with the table and layout pieces, while specialized GD&T/drawing studies can inform the approach for dimension semantics.

### 5.7 Combining OCR with LLMs: schema mapping and validation

Across domains (invoices, forms, CAD, reports), a recurring pattern is:

1. Use open-source OCR + layout models to produce **structured but generic JSON/Markdown** with text and geometry.  
2. Use an LLM (open or hosted) to:

   - Map generic structures to a domain-specific schema.  
   - Normalize values (dates, currencies, IDs).  
   - Validate constraints (e.g., sum of line items ≈ total; tax rates plausible; CAD dimensions within expected ranges).  
   - Provide explanations and flags for ambiguous or low-confidence fields.

This hybrid approach can significantly reduce the amount of hand-written rules or annotated KIE data, at the cost of managing LLM latency and hallucination risks.

### 5.8 Engineering practices: versioning, testing, and deployment

Based on local invoice specs and broader community experience, recommended practices include:

- **Version pinning and change management**  
  - Pin `transformers`, `torch`, `pypdfium2`, `pdfminer.six`, and core OCR libraries.  
  - Treat upgrades as mini-projects, with golden tests and metric checks.

- **Image processor behavior**  
  - Be aware of `use_fast` defaults for image processors in Transformers. Upgrades can silently switch from slow to fast processors, causing small but important differences in OCR tokens.  
  - Explicitly set `use_fast=True/False` where deterministic behavior matters (e.g., invoice totals, regression tests).

- **Reproducible pipelines**  
  - Fix random seeds, especially for training (Donut, LayoutLM).  
  - Keep configuration files (YAML/TOML) that fully describe model versions, prompts, crop boxes, and thresholds.

- **Monitoring and alerts**  
  - Log model version, prompt, and input hash for each processed document.  
  - Watch for sudden changes in field-level error rates or anomaly scores.

---

## 6. Limitations, caveats, and open questions

### 6.1 Licensing and usage constraints

Not all models and datasets have the same license:

- **PaddleOCR**, **docTR**, and many HF models use permissive licenses (Apache-2.0, MIT), suitable for commercial use, but always verify the specific repo.  
- **LayoutLMv1** is MIT-licensed; **LayoutLMv2/v3** weights are sometimes released under research or non-commercial terms—check model cards and papers before deploying commercially.  
- Synthetic or proprietary datasets may restrict redistribution or commercial usage.

Always check:

- Repository license (Apache-2.0, MIT, custom).  
- Model card notes for usage restrictions.  
- Dataset licenses, especially for fine-tuning and redistribution.

### 6.2 Data and annotation cost

High-quality structured OCR systems require data, especially for KIE and relation extraction:

- Donut-style models need many (image, JSON) examples per domain.  
- LayoutLM-style KIE requires bounding-box labels and token-level tagging.  
- Domain-specific templates (local invoice formats, custom forms) often behave differently from public benchmarks.

Mitigations:

- Start from strong pretrained models and fine-tune incrementally.  
- Use active learning to focus annotation effort on uncertain or novel cases.  
- Let LLMs pre-fill labels or candidates that humans verify, reducing manual workload.

### 6.3 Robustness and domain shift

Common challenges:

- Template drift: new invoice or form layouts, different tax layouts, multi-language vendors.  
- Low-quality scans: blur, skew, compression artifacts, partial pages.  
- Mixed content: hybrid languages/scripts (e.g., Japanese + English), vertical text, handwritten notes.

Possible mitigations:

- Regular re-training / fine-tuning with fresh data and strong augmentations.  
- Combining vector PDF parsing with raster OCR to reduce scan artifacts.  
- Using ensembles or fallback models for low-confidence cases.  
- Implementing business-rule validation and LLM-based sanity checks.

### 6.4 Evaluation and business metrics

Evaluation should cover both ML metrics and business outcomes:

- OCR quality: character/word error rate (CER/WER).  
- KIE: field-level precision/recall/F1, exact match for critical fields.  
- Tables: structure accuracy (row/column correctness), cell content accuracy, and downstream impact (e.g., correct line-item totals).  
- Business metrics: percentage of documents processed without manual correction, average handling time, error rates on critical amounts.

Having an explicit evaluation plan helps avoid over-optimizing on generic benchmarks that do not reflect real, noisy documents.

---

## 7. Practical selection guide

### 7.1 If you want a fast, general-purpose baseline

- Start with **PaddleOCR + PP-StructureV3** to convert PDFs/images to Markdown and JSON.  
- For many workloads, this alone provides sufficiently good structure to feed a downstream LLM or search index.  
- Add LLM post-processing for schema mapping and validation.

### 7.2 If OCR quality is the main challenge

- Use **docTR** or PaddleOCR as the OCR engine; tune preprocessing (DPI, binarization, deskewing).  
- Use simple coordinate heuristics and possibly LayoutParser for layout.  
- Only add complex models (LayoutLM, Donut) once OCR text is strong.

### 7.3 If you have a well-defined schema and can label examples

- Choose **Donut** or a similar OCR-free model:  
  - Define schema; generate or annotate examples.  
  - Fine-tune for your document type.  
  - Wrap inference in robust JSON parsing and validation.

### 7.4 If you have many forms and receipts with complex layouts

- Choose **LayoutLMv3 + OCR**:  
  - Label tokens and relations on representative forms.  
  - Train a KIE model with token classification and relation heads.  
  - Keep OCR and LayoutLM modular so you can swap OCR engines later.

### 7.5 If you face very unusual layouts (CAD, engineering, schematics)

- Use a **hybrid stack**:  
  - Extract vector text wherever possible.  
  - Use specialized detectors for small/rotated text and tables.  
  - Use LayoutLMv3 or a custom Transformer for relationships between text and drawing elements.  
  - Employ domain-specific rules and LLM validation for semantic interpretation.

### 7.6 If you are building a modern invoice-processing engine

- Combine ideas from **Pattern A–E**:  
  - General document parsing (PaddleOCR/PP-Structure or docTR + layout).  
  - Invoice-specific VLM pipeline (Florence-2-style or GOT-OCR2/DeepSeek-OCR with well-designed prompts).  
  - Token-level KIE for tricky fields using LayoutLMv3.  
  - LLM-based normalization and business-rule validation.

---

## 8. References and links

**Core toolkits and docs**

- PaddleOCR docs: <https://paddlepaddle.github.io/PaddleOCR/main/en/index.html>  
- PaddleOCR GitHub: <https://github.com/PaddlePaddle/PaddleOCR>  
- PaddleOCR 3.0 technical report: <https://arxiv.org/abs/2507.05595>  
- docTR docs: <https://mindee.github.io/doctr/latest/>  
- docTR GitHub: <https://github.com/mindee/doctr>

**Models and model docs**

- Donut model docs: <https://huggingface.co/docs/transformers/en/model_doc/donut>  
- Donut GitHub: <https://github.com/clovaai/donut>  
- Donut paper: <https://arxiv.org/abs/2111.15664>  
- LayoutLMv3 model docs: <https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3>  
- LayoutLMv3 base card: <https://huggingface.co/microsoft/layoutlmv3-base>  
- GOT-OCR2 model docs: <https://huggingface.co/docs/transformers/en/model_doc/got_ocr2>  
- DeepSeek-OCR: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>  
- dots.ocr: <https://huggingface.co/rednote-hilab/dots.ocr>  
- Granite-Docling-258M: <https://huggingface.co/ibm-granite/granite-docling-258m>  
- PaddleOCR-VL: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL>  
- Table Transformer detection: <https://huggingface.co/microsoft/table-transformer-detection>

**Blogs and overviews**

- Hugging Face Document AI blog: <https://huggingface.co/blog/document-ai>  
- Example Donut tutorial in Japanese: <https://zenn.dev/mutex_inc/articles/demonstrate-donut>  
- Donut vs Pix2Struct comparison: <https://medium.com/data-science/ocr-free-document-data-extraction-with-transformers-2-2-38ce26f41951>  
- Technical analysis of non-LLM OCR engines: <https://intuitionlabs.ai/articles/non-llm-ocr-technologies>  
- Recent comparisons of open-source OCR tools: search for 2024–2025 reviews by vendors like Affinda, Unstract, etc.

**Datasets**

- FUNSD, CORD, SROIE, DocVQA – accessible via Hugging Face Datasets or original repositories.  
- DocLayNet v1.1 – e.g., `docling-project/DocLayNet-v1.1` on Hugging Face.  
- PubTables-1M – e.g., `bsmock/pubtables-1m` on Hugging Face.

This knowledge base consolidates official documentation, model cards, community experience, and local specs into a practical overview of open-source structured document OCR, with an emphasis on pipelines that you can adapt for invoices, forms, PDFs, and CAD-like documents.
