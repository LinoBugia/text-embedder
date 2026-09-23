"""Shared chunk+embed pipeline for the CLI (main.py) and the GUI (app.py)."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path

from chunkers import chunk_text
from embedders import get_embedder

_ID_ALPHABET = string.ascii_letters + string.digits

SUFFIXES = (".md", ".txt")


def find_input_files(input_dir: Path) -> list[Path]:
    """All .md/.txt files, recursively including subfolders (no hidden ones)."""
    files = []
    for path in input_dir.rglob("*"):
        rel_parts = path.relative_to(input_dir).parts
        if (path.is_file() and path.suffix.lower() in SUFFIXES
                and not any(part.startswith(".") for part in rel_parts)):
            files.append(path)
    return sorted(files)


def rel_name(path: Path, input_dir: Path | None) -> str:
    """Path relative to the input folder (unique across subfolders too)."""
    if input_dir is not None:
        try:
            return path.relative_to(input_dir).as_posix()
        except ValueError:
            pass
    return path.name


def make_chunk_id(existing: set[str]) -> str:
    """16-character alphanumeric ID (~95 bits), collision-free within a run."""
    while True:
        cid = "".join(secrets.choice(_ID_ALPHABET) for _ in range(16))
        if cid not in existing:
            existing.add(cid)
            return cid


def reembed_run(source_run: Path, config: dict, *,
                output_dir: str | Path | None = None,
                run_name: str | None = None,
                progress=lambda msg: None) -> Path:
    """Embed the chunks of an existing run with a different model.

    Chunks, IDs and their order are carried over unchanged, which makes the
    vectors of different models comparable chunk by chunk via the same ID
    (benchmark matrix). Returns the new run directory.
    """
    with open(source_run / "chunks.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if not records:
        raise RuntimeError(f"{source_run}/chunks.jsonl is empty")
    source_specs = json.loads((source_run / "specs.json").read_text(encoding="utf-8"))

    progress(f"Re-embedding {len(records)} chunks from {source_run.name} ...")
    embedder = get_embedder(config["embedding"])
    vectors = embedder.embed([r.get("embed_text", r["text"]) for r in records])

    base = Path(output_dir or config.get("output_dir", "./output"))
    model_slug = re.sub(r"[^a-zA-Z0-9._-]", "-", config["embedding"].get("model", "model"))
    run_dir = base / (run_name or model_slug)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Carry chunks.jsonl over verbatim - same IDs, same order
    shutil.copyfile(source_run / "chunks.jsonl", run_dir / "chunks.jsonl")
    with open(run_dir / "embeddings.jsonl", "w", encoding="utf-8") as f:
        for r, vec in zip(records, vectors):
            f.write(json.dumps({"id": r["id"], "embedding": vec}) + "\n")

    specs = dict(source_specs)
    specs.update({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": {k: v for k, v in config["embedding"].items() if k != "api_key"},
        "embedding_dim": len(vectors[0]),
        "files": {"chunks": "chunks.jsonl", "embeddings": "embeddings.jsonl"},
        "reembedded_from": source_run.name,
    })
    with open(run_dir / "specs.json", "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    progress(f"Done: {len(records)} vectors ({len(vectors[0])} dim) -> {run_dir}")
    return run_dir


def run_pipeline(files: list[Path], config: dict, *,
                 mode_overrides: dict[str, str] | None = None,
                 dry_run: bool = False,
                 output_dir: str | Path | None = None,
                 input_dir: Path | None = None,
                 progress=lambda msg: None) -> Path:
    """Chunk (and embed) the files and write a run directory.

    Files are identified by their path relative to input_dir (source,
    mode_overrides, ids_by_file) - unique across subfolders too.
    mode_overrides: {relpath: mode} overrides config['chunking']['mode'] per
    file. Returns the path of the created run directory.
    """
    mode_overrides = mode_overrides or {}
    seen_ids: set[str] = set()
    records: list[dict] = []
    per_file: dict[str, list[str]] = {}
    modes_by_file: dict[str, str] = {}

    for path in files:
        name = rel_name(path, input_dir)
        cfg = dict(config["chunking"])
        cfg["mode"] = mode_overrides.get(name, cfg.get("mode", "markdown"))
        modes_by_file[name] = cfg["mode"]
        # utf-8-sig: strips a BOM at the start of the file if present
        chunks = chunk_text(path.read_text(encoding="utf-8-sig"), cfg)
        ids = []
        for chunk in chunks:
            cid = make_chunk_id(seen_ids)
            ids.append(cid)
            record = {"id": cid, "source": name,
                      "text": chunk.text, "metadata": chunk.meta}
            if chunk.embed_text is not None:
                # embed_text is what gets embedded, text is what is served
                record["embed_text"] = chunk.embed_text
            records.append(record)
        per_file[name] = ids
        progress(f"{name}: {len(ids)} chunks")

    if not records:
        raise RuntimeError("No chunks produced - is the input empty?")

    vectors: list[list[float]] | None = None
    if not dry_run:
        progress(f"Embedding {len(records)} chunks ...")
        embedder = get_embedder(config["embedding"])
        vectors = embedder.embed([r.get("embed_text", r["text"])
                                  for r in records])

    # Run directory: <file-names>_<mode>_<timestamp>
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    modes = sorted(set(modes_by_file.values()))
    mode_part = modes[0] if len(modes) == 1 else "mixed"
    stems = [p.stem for p in files]
    name_part = "+".join(stems[:3]) + (f"+{len(stems) - 3}more" if len(stems) > 3 else "")
    base = Path(output_dir or config.get("output_dir", "./output"))
    run_dir = base / f"{name_part}_{mode_part}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if vectors is not None:
        with open(run_dir / "embeddings.jsonl", "w", encoding="utf-8") as f:
            for r, vec in zip(records, vectors):
                f.write(json.dumps({"id": r["id"], "embedding": vec}) + "\n")

    specs = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunking": config["chunking"],
        "modes_by_file": modes_by_file,
        # Never write api_key into specs.json (it would end up in every run
        # directory) - the key belongs in the environment or the local config
        "embedding": None if dry_run else {
            k: v for k, v in config["embedding"].items() if k != "api_key"},
        "embedding_dim": len(vectors[0]) if vectors else None,
        "input_files": [rel_name(p, input_dir) for p in files],
        "chunk_count": len(records),
        "files": {"chunks": "chunks.jsonl",
                  "embeddings": "embeddings.jsonl" if vectors else None},
        "ids_by_file": per_file,
        "ids": [r["id"] for r in records],
    }
    with open(run_dir / "specs.json", "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    progress(f"Done: {len(records)} chunks -> {run_dir}")
    return run_dir
