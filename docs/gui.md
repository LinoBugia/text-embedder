# GUI

```bash
pip install -r requirements.txt   # once (customtkinter)
python app.py
```

A split window — transform on the left, browse on the right. The right side
works without an embedding backend, so you can always inspect past runs.

## Header

Provider (`ollama`/`gemini`) and model as dropdowns. For Ollama the **actually
installed models** are listed, embedding models sorted to the top; for Gemini
the known embedding models. The backend status — Ollama reachable, API key
set — is checked at startup and shown in green or red, with a "Re-check"
button next to it. Changing either dropdown writes the choice to
`config.json` and re-checks.

## Left — input workspace

- Input folder chosen via a dialog and remembered in `config.json`
- All `.md`/`.txt` files are listed **recursively including subfolders**, shown
  with their path relative to the input folder (e.g. `notes/models/bge.md`),
  each with a checkbox
- **Chunking mode per file** (dropdown: `marker`, `token`, `markdown`); the
  selected files are processed together in a single run
- **Mode (all)** sets the mode for every file at once; `max_tokens`,
  `title_depth` and `marker` are editable inline. Everything is written back
  to `config.json`
- Three checkboxes, explained in [chunking.md](chunking.md#text-transforms):
  **Redact markdown (# and \*)**, **Strip YAML header**,
  **Hide links in embedding**
- Two actions: **Chunk only (dry run)** — always available, costs nothing and
  is the fastest way to inspect a chunking — and **Chunk + embed**, enabled
  only when the backend is ready

## Right — output workspace

- Output folder chosen via a dialog; all run directories (detected by their
  `specs.json`) appear in the dropdown, newest first
- For the selected run: chunk count, embedding dimension, modes used
- A list of all chunks with **ID, mode, token count, source file, heading path
  and part number**. Clicking an ID shows that chunk exactly as it came out of
  the pipeline, including its metadata
- If links were hidden for the embedding, the text that was actually embedded
  is shown below the chunk, so you can compare what the model saw
- **Show whole document with chunk boundaries** prints all chunks of the
  selected source document in order, separated by marker lines with ID, token
  count and part — the full chunking of a document at a glance

After a transform run the right-hand side jumps to the newly created run.

## Window size

The window starts at 1440×880 and can be shrunk to 1280×720. Below that,
controls would be cut off, hence the enforced minimum.

## Building an app bundle

```bash
pip install pyinstaller
pyinstaller --windowed --name TextEmbedder --collect-all customtkinter app.py
# Result: dist/TextEmbedder.app (macOS) or dist/TextEmbedder/
```
