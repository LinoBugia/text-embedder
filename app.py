#!/usr/bin/env python3
"""GUI: chunk/embed the input workspace on the left, browse runs on the right.

Run:    python app.py        (requires: pip install customtkinter)
Build:  pyinstaller --windowed --name TextEmbedder --collect-all customtkinter app.py
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from chunkers import CHUNKERS
from embedders import PROVIDERS, check_provider, list_models
from pipeline import find_input_files, run_pipeline

MODES = list(CHUNKERS)
MONO = ("Menlo", 12)

DEFAULT_CFG = {
    "input_dir": "./input",
    "output_dir": "./output",
    "chunking": {"mode": "markdown", "marker": "---", "max_tokens": 350,
                 "min_tokens": 30, "title_depth": 2, "redact_markdown": False,
                 "strip_yaml_header": True, "strip_links_for_embedding": False,
                 "tokenizer": "heuristic"},
    "embedding": {"provider": "ollama", "model": "bge-m3",
                  "base_url": "http://localhost:11434", "batch_size": 16},
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Text Embedder - Chunker & Viewer")
        self.geometry("1440x880")
        self.minsize(1280, 720)  # below this, controls would be cut off
        ctk.set_appearance_mode("dark")

        self.config_path = Path(__file__).parent / "config.json"
        self.cfg = self._load_config()
        self.input_dir = self._existing_dir(self.cfg.get("input_dir"))
        self.output_dir = self._existing_dir(self.cfg.get("output_dir"))
        self.file_rows: dict[str, tuple[tk.BooleanVar, tk.StringVar]] = {}
        self.run_dirs: dict[str, Path] = {}
        self.chunks: list[dict] = []
        self.specs: dict = {}
        self.backend_ok = False

        self._build_ui()
        self.refresh_files()
        self.refresh_runs()
        self.check_backend()

    # ------------------------------------------------------------- config

    def _load_config(self) -> dict:
        cfg = json.loads(json.dumps(DEFAULT_CFG))  # deep copy
        if self.config_path.exists():
            loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
            for key, val in loaded.items():
                if isinstance(val, dict):
                    cfg.setdefault(key, {}).update(val)
                else:
                    cfg[key] = val
        return cfg

    def _save_config(self):
        self.config_path.write_text(
            json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    @staticmethod
    def _existing_dir(path: str | None) -> Path | None:
        if path and Path(path).is_dir():
            return Path(path)
        return None

    # ------------------------------------------------------------- ui

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1, uniform="half")
        self.grid_columnconfigure(1, weight=1, uniform="half")
        self.grid_rowconfigure(1, weight=1)

        # Header: embedding backend (provider + model) and its status
        head = ctk.CTkFrame(self)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        emb = self.cfg["embedding"]
        ctk.CTkLabel(head, text="Provider:").pack(side="left", padx=(10, 2), pady=6)
        self.provider_var = tk.StringVar(value=emb.get("provider", "ollama"))
        ctk.CTkOptionMenu(head, variable=self.provider_var, width=110,
                          values=list(PROVIDERS),
                          command=lambda _: self._backend_changed()
                          ).pack(side="left", pady=6)
        ctk.CTkLabel(head, text="Model:").pack(side="left", padx=(10, 2))
        self.model_var = tk.StringVar(value=emb.get("model", "bge-m3"))
        self.model_menu = ctk.CTkOptionMenu(
            head, variable=self.model_var, width=240,
            values=[self.model_var.get()],
            command=lambda _: self._backend_changed())
        self.model_menu.pack(side="left", pady=6)
        self.backend_label = ctk.CTkLabel(head, text="checking ...",
                                          text_color="gray")
        self.backend_label.pack(side="left", padx=10, pady=6)
        ctk.CTkButton(head, text="Re-check", width=110,
                      command=self.check_backend).pack(side="left", pady=6)
        self.status_label = ctk.CTkLabel(head, text="", anchor="e")
        self.status_label.pack(side="right", padx=10)

        self._build_left()
        self._build_right()

    def _build_left(self):
        left = ctk.CTkFrame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        row = ctk.CTkFrame(left, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ctk.CTkButton(row, text="Choose input workspace", width=190,
                      command=self.choose_input).pack(side="left")
        self.input_label = ctk.CTkLabel(row, text="-", anchor="w")
        self.input_label.pack(side="left", padx=8, fill="x", expand=True)

        ctk.CTkLabel(left, text="Files & chunking mode per file:",
                     anchor="w").grid(row=1, column=0, sticky="ew", padx=12)
        self.file_frame = ctk.CTkScrollableFrame(left)
        self.file_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self.file_frame.grid_columnconfigure(0, weight=1)

        ch = self.cfg["chunking"]
        params = ctk.CTkFrame(left, fg_color="transparent")
        params.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 0))
        ctk.CTkLabel(params, text="Mode (all):").pack(side="left", padx=(8, 2))
        self.mode_var = tk.StringVar(value=ch.get("mode", "markdown"))
        ctk.CTkOptionMenu(params, variable=self.mode_var, values=MODES,
                          width=112, command=lambda m: self._set_all_modes(m)
                          ).pack(side="left")
        self.max_tokens_var = tk.StringVar(value=str(ch.get("max_tokens", 350)))
        self.title_depth_var = tk.StringVar(value=str(ch.get("title_depth", 2)))
        self.marker_var = tk.StringVar(value=ch.get("marker", "---"))
        for label, var, width in (("max_tokens", self.max_tokens_var, 62),
                                  ("title_depth", self.title_depth_var, 46),
                                  ("marker", self.marker_var, 62)):
            ctk.CTkLabel(params, text=label).pack(side="left", padx=(6, 2))
            ctk.CTkEntry(params, textvariable=var, width=width).pack(side="left")

        # Two columns so that nothing gets cut off
        options = ctk.CTkFrame(left, fg_color="transparent")
        options.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 0))
        options.grid_columnconfigure((0, 1), weight=1, uniform="opt")
        self.redact_var = tk.BooleanVar(value=bool(ch.get("redact_markdown")))
        self.yaml_var = tk.BooleanVar(value=bool(ch.get("strip_yaml_header", True)))
        self.links_var = tk.BooleanVar(
            value=bool(ch.get("strip_links_for_embedding")))
        for i, (var, label) in enumerate((
                (self.redact_var, "Redact markdown (# and *)"),
                (self.yaml_var, "Strip YAML header"),
                (self.links_var, "Hide links in embedding"))):
            ctk.CTkCheckBox(options, variable=var, text=label).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=2)

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.grid(row=5, column=0, sticky="ew", padx=8, pady=(4, 8))
        self.chunk_btn = ctk.CTkButton(btns, text="Chunk only (dry run)",
                                       command=lambda: self.transform(True))
        self.chunk_btn.pack(side="left", padx=(0, 6))
        self.embed_btn = ctk.CTkButton(btns, text="Chunk + embed",
                                       fg_color="#2c7a3f", hover_color="#1f5c2e",
                                       state="disabled",
                                       command=lambda: self.transform(False))
        self.embed_btn.pack(side="left")

    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=2)
        right.grid_rowconfigure(5, weight=3)

        row = ctk.CTkFrame(right, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ctk.CTkButton(row, text="Choose output workspace", width=190,
                      command=self.choose_output).pack(side="left")
        self.output_label = ctk.CTkLabel(row, text="-", anchor="w")
        self.output_label.pack(side="left", padx=8, fill="x", expand=True)

        row2 = ctk.CTkFrame(right, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        self.run_var = tk.StringVar(value="")
        self.run_menu = ctk.CTkOptionMenu(row2, variable=self.run_var,
                                          values=[""], width=420,
                                          command=lambda _: self.load_run())
        self.run_menu.pack(side="left")
        ctk.CTkButton(row2, text="Refresh", width=100,
                      command=self.refresh_runs).pack(side="left", padx=6)

        self.run_info = ctk.CTkLabel(right, text="", anchor="w", justify="left")
        self.run_info.grid(row=2, column=0, sticky="ew", padx=12)

        list_frame = ctk.CTkFrame(right)
        list_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        self.chunk_list = tk.Listbox(list_frame, font=MONO, bg="#1d1e1e",
                                     fg="#e5e5e5", selectbackground="#2c7a3f",
                                     highlightthickness=0, borderwidth=0)
        self.chunk_list.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        sb = tk.Scrollbar(list_frame, command=self.chunk_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.chunk_list.configure(yscrollcommand=sb.set)
        self.chunk_list.bind("<<ListboxSelect>>", lambda _: self.show_chunk())

        ctk.CTkButton(right, text="Show whole document with chunk boundaries",
                      command=self.show_document
                      ).grid(row=4, column=0, sticky="w", padx=8, pady=2)

        self.detail = ctk.CTkTextbox(right, font=MONO, wrap="word")
        self.detail.grid(row=5, column=0, sticky="nsew", padx=8, pady=(4, 8))

    # ------------------------------------------------------------- left side

    def choose_input(self):
        path = filedialog.askdirectory(title="Choose input workspace")
        if path:
            self.input_dir = Path(path)
            self.cfg["input_dir"] = path
            self._save_config()
            self.refresh_files()

    def refresh_files(self):
        for widget in self.file_frame.winfo_children():
            widget.destroy()
        self.file_rows.clear()
        self.input_label.configure(text=str(self.input_dir or "-"))
        if not self.input_dir:
            return
        files = find_input_files(self.input_dir)
        default_mode = self.cfg["chunking"].get("mode", "markdown")
        for i, path in enumerate(files):
            rel = path.relative_to(self.input_dir).as_posix()
            selected = tk.BooleanVar(value=True)
            mode = tk.StringVar(value=default_mode)
            ctk.CTkCheckBox(self.file_frame, text=rel, variable=selected
                            ).grid(row=i, column=0, sticky="w", pady=2)
            ctk.CTkOptionMenu(self.file_frame, variable=mode, values=MODES,
                              width=130).grid(row=i, column=1, padx=4, pady=2)
            self.file_rows[rel] = (selected, mode)

    def _set_all_modes(self, mode: str):
        """Global mode dropdown: sets the mode for every file row."""
        for _, mode_var in self.file_rows.values():
            mode_var.set(mode)
        self.cfg["chunking"]["mode"] = mode
        self._save_config()

    def transform(self, dry_run: bool):
        if not self.input_dir:
            messagebox.showwarning("No input", "Please choose an input workspace.")
            return
        files = [self.input_dir / name
                 for name, (sel, _) in self.file_rows.items() if sel.get()]
        if not files:
            messagebox.showwarning("No files", "No file is selected.")
            return
        try:
            self.cfg["chunking"]["max_tokens"] = int(self.max_tokens_var.get())
            self.cfg["chunking"]["title_depth"] = int(self.title_depth_var.get())
        except ValueError:
            messagebox.showerror("Invalid value",
                                 "max_tokens/title_depth must be numbers.")
            return
        self.cfg["chunking"]["marker"] = self.marker_var.get() or "---"
        self.cfg["chunking"]["mode"] = self.mode_var.get()
        self.cfg["chunking"]["redact_markdown"] = self.redact_var.get()
        self.cfg["chunking"]["strip_yaml_header"] = self.yaml_var.get()
        self.cfg["chunking"]["strip_links_for_embedding"] = self.links_var.get()
        self.cfg["embedding"]["provider"] = self.provider_var.get()
        self.cfg["embedding"]["model"] = self.model_var.get().strip()
        self._save_config()
        overrides = {name: mode.get()
                     for name, (sel, mode) in self.file_rows.items() if sel.get()}
        out_dir = self.output_dir or Path(self.cfg.get("output_dir", "./output"))

        self.chunk_btn.configure(state="disabled")
        self.embed_btn.configure(state="disabled")
        self._status("Processing ...")

        def worker():
            try:
                run_dir = run_pipeline(files, self.cfg, mode_overrides=overrides,
                                       dry_run=dry_run, output_dir=out_dir,
                                       input_dir=self.input_dir,
                                       progress=self._status)
                self.after(0, lambda: self._transform_done(run_dir))
            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: self._transform_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _transform_done(self, run_dir: Path):
        self.output_dir = self.output_dir or run_dir.parent
        self.chunk_btn.configure(state="normal")
        self.embed_btn.configure(state="normal" if self.backend_ok else "disabled")
        self.refresh_runs(select=run_dir.name)

    def _transform_failed(self, err: Exception):
        self.chunk_btn.configure(state="normal")
        self.embed_btn.configure(state="normal" if self.backend_ok else "disabled")
        self._status("Failed.")
        messagebox.showerror("Error", str(err))

    # ------------------------------------------------------------- right side

    def choose_output(self):
        path = filedialog.askdirectory(title="Choose output workspace")
        if path:
            self.output_dir = Path(path)
            self.cfg["output_dir"] = path
            self._save_config()
            self.refresh_runs()

    def refresh_runs(self, select: str | None = None):
        self.output_label.configure(text=str(self.output_dir or "-"))
        self.run_dirs.clear()
        if self.output_dir and self.output_dir.is_dir():
            runs = sorted((d for d in self.output_dir.iterdir()
                           if (d / "specs.json").exists()),
                          key=lambda d: d.stat().st_mtime, reverse=True)
            self.run_dirs = {d.name: d for d in runs}
        names = list(self.run_dirs) or [""]
        self.run_menu.configure(values=names)
        self.run_var.set(select if select in self.run_dirs else names[0])
        self.load_run()

    def load_run(self):
        self.chunk_list.delete(0, "end")
        self.chunks, self.specs = [], {}
        self._set_detail("")
        run_dir = self.run_dirs.get(self.run_var.get())
        if not run_dir:
            self.run_info.configure(text="")
            return
        self.specs = json.loads((run_dir / "specs.json").read_text(encoding="utf-8"))
        with open(run_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        dim = self.specs.get("embedding_dim")
        modes = self.specs.get("modes_by_file", {})
        self.run_info.configure(text=(
            f"{self.specs.get('chunk_count', len(self.chunks))} chunks · "
            f"embeddings: {f'{dim}-dim' if dim else 'none (dry run)'} · "
            f"modes: {', '.join(sorted(set(modes.values()))) or '?'}"))
        for c in self.chunks:
            m = c["metadata"]
            part = (f" · part {m['part']}/{m['total_parts']}"
                    if m.get("part") else "")
            heading = " > ".join(m.get("heading_path", [])) or "-"
            tokens = m.get("token_count", "?")
            self.chunk_list.insert(
                "end",
                f"{c['id']}  [{m.get('mode', '?'):8}] {tokens:>5} tok  "
                f"{c['source']}  ·  {heading}{part}")

    def _selected_chunk(self) -> dict | None:
        sel = self.chunk_list.curselection()
        return self.chunks[sel[0]] if sel and sel[0] < len(self.chunks) else None

    def show_chunk(self):
        chunk = self._selected_chunk()
        if not chunk:
            return
        meta = json.dumps(chunk["metadata"], ensure_ascii=False, indent=2)
        detail = (f"id: {chunk['id']}   source: {chunk['source']}\n"
                  f"{meta}\n{'-' * 60}\n{chunk['text']}")
        if chunk.get("embed_text"):
            detail += (f"\n\n{'=' * 20} embedded (links removed) {'=' * 20}\n"
                       f"{chunk['embed_text']}")
        self._set_detail(detail)

    def show_document(self):
        if not self.chunks:
            return
        chunk = self._selected_chunk() or self.chunks[0]
        source = chunk["source"]
        parts = []
        doc_chunks = [c for c in self.chunks if c["source"] == source]
        for i, c in enumerate(doc_chunks, start=1):
            m = c["metadata"]
            part = (f" · part {m['part']}/{m['total_parts']}"
                    if m.get("part") else "")
            parts.append(f"{'=' * 24} chunk {i}/{len(doc_chunks)} · {c['id']}"
                         f" · {m.get('token_count', '?')} tok{part} {'=' * 24}\n"
                         f"{c['text']}")
        self._set_detail(f"Document: {source}\n\n" + "\n\n".join(parts))

    # ------------------------------------------------------------- misc

    def _set_detail(self, text: str):
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)

    def _status(self, msg: str):
        self.after(0, lambda: self.status_label.configure(text=msg))

    def _backend_changed(self):
        self.cfg["embedding"]["provider"] = self.provider_var.get()
        self.cfg["embedding"]["model"] = self.model_var.get().strip()
        self._save_config()
        self.check_backend()

    def check_backend(self):
        emb = dict(self.cfg["embedding"])
        emb["provider"] = self.provider_var.get()

        def worker():
            ok, msg = check_provider(emb)
            models = list_models(emb)
            self.after(0, lambda: self._set_backend(ok, msg, models))

        self.backend_label.configure(text="checking ...", text_color="gray")
        threading.Thread(target=worker, daemon=True).start()

    def _set_backend(self, ok: bool, msg: str, models: list[str]):
        self.backend_ok = ok
        if models:
            self.model_menu.configure(values=models)
            if self.model_var.get() not in models:
                self.model_var.set(models[0])
                self.cfg["embedding"]["model"] = models[0]
                self._save_config()
        if ok:
            self.backend_label.configure(text=msg, text_color="#4cc36a")
            self.embed_btn.configure(state="normal")
        else:
            self.backend_label.configure(
                text=f"{msg} (dry run only)", text_color="#e05c5c")
            self.embed_btn.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
