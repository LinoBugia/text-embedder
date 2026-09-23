"""Chunking strategies: marker, token, markdown."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------- tokenizer

def make_token_counter(name: str):
    """Return a function text -> int (number of tokens)."""
    if name == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        except ImportError:
            print("Warning: tiktoken not installed, falling back to heuristic.",
                  file=sys.stderr)
    # Heuristic: ~4 characters per token (good enough for English/German prose)
    return lambda text: max(1, len(text) // 4)


# ---------------------------------------------------------------- datamodel

@dataclass
class Chunk:
    text: str
    meta: dict = field(default_factory=dict)
    embed_text: str | None = None  # if set, this is embedded instead of text

    @property
    def text_to_embed(self) -> str:
        return self.embed_text if self.embed_text is not None else self.text


# ---------------------------------------------------------------- blocks

_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Sentence boundary: punctuation followed by an opening word/quote. The
# character class covers German umlauts and quotes so that German source
# documents split correctly as well.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-ZÄÖÜ0-9(\"'„])")


def _split_blocks(lines: list[str]) -> list[tuple[str, str]]:
    """Split lines into blocks: (kind, text).

    kind: 'prose' (may be split at sentence boundaries) or 'atomic' (table,
    list, code block - never cut in the middle).
    """
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    kind: str | None = None

    def flush():
        nonlocal buf, kind
        if buf:
            blocks.append((kind or "prose", "\n".join(buf).strip()))
        buf, kind = [], None

    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                if kind != "atomic":
                    flush()
                kind = "atomic"
                in_fence = True
                buf.append(line)
            else:
                buf.append(line)
                in_fence = False
                flush()
            continue
        if in_fence:
            buf.append(line)
            continue

        if not stripped:
            flush()
            continue

        if _TABLE_RE.match(line) or _LIST_RE.match(line):
            line_kind = "atomic"
        elif kind == "atomic" and line.startswith((" ", "\t")):
            # indented continuation line of a list item
            line_kind = "atomic"
        else:
            line_kind = "prose"

        if kind is not None and line_kind != kind:
            flush()
        kind = line_kind
        buf.append(line)
    flush()
    return [b for b in blocks if b[1]]


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------- packer

def _split_hard(text: str, max_tokens: int, count_tokens) -> list[str]:
    """Emergency split at line boundaries (table rows and list items stay
    intact); single overlong lines are cut by character count."""
    lines: list[str] = []
    approx_chars = max(1, max_tokens) * 4
    for line in text.splitlines():
        while count_tokens(line) > max_tokens and len(line) > approx_chars:
            lines.append(line[:approx_chars])
            line = line[approx_chars:]
        lines.append(line)
    pieces, current, current_len = [], [], 0
    for line in lines:
        llen = count_tokens(line)
        if current and current_len + llen > max_tokens:
            pieces.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += llen
    if current:
        pieces.append("\n".join(current))
    return pieces


def _pack_blocks(blocks: list[tuple[str, str]], max_tokens: int,
                 count_tokens, hard_max_tokens: int = 6000) -> list[str]:
    """Greedily pack blocks into chunks of <= max_tokens.

    Prose blocks are split at sentence boundaries when needed; atomic blocks
    (tables, lists, code) are kept together - unless they exceed
    hard_max_tokens (context-window guard), in which case they are split at
    line boundaries.
    """
    pieces: list[str] = []
    for kind, text in blocks:
        if count_tokens(text) <= max_tokens:
            pieces.append(text)
        elif kind == "atomic":
            print(f"Warning: atomic block ({count_tokens(text)} tokens) "
                  f"exceeds max_tokens={max_tokens}, left unsplit.",
                  file=sys.stderr)
            pieces.append(text)
        else:
            # split prose into max_tokens pieces at sentence boundaries
            current: list[str] = []
            for sent in _split_sentences(text):
                joined = " ".join(current + [sent])
                if current and count_tokens(joined) > max_tokens:
                    pieces.append(" ".join(current))
                    current = [sent]
                else:
                    current.append(sent)
            if current:
                pieces.append(" ".join(current))

    # Emergency brake: anything above hard_max_tokens (context-window guard)
    # is split at line boundaries - prose, table or code block alike.
    if hard_max_tokens:
        capped: list[str] = []
        for piece in pieces:
            plen = count_tokens(piece)
            if plen > hard_max_tokens:
                print(f"Warning: piece of {plen} tokens exceeds "
                      f"hard_max_tokens={hard_max_tokens} - "
                      f"emergency split at line boundaries.", file=sys.stderr)
                capped.extend(_split_hard(piece, max_tokens, count_tokens))
            else:
                capped.append(piece)
        pieces = capped

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for piece in pieces:
        plen = count_tokens(piece)
        if current_parts and current_len + plen > max_tokens:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_len = [], 0
        current_parts.append(piece)
        current_len += plen
    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


# ---------------------------------------------------------------- mode 1: marker

def chunk_marker(text: str, cfg: dict, count_tokens) -> list[Chunk]:
    """Split at a user-defined marker (a line equal to marker)."""
    marker = cfg.get("marker", "---")
    parts, buf = [], []
    for line in text.splitlines():
        if line.strip() == marker:
            parts.append("\n".join(buf))
            buf = []
        else:
            buf.append(line)
    parts.append("\n".join(buf))
    return [Chunk(p.strip(), {"mode": "marker", "chunk_index": i})
            for i, p in enumerate(parts) if p.strip()]


# ---------------------------------------------------------------- mode 2: token

def chunk_token(text: str, cfg: dict, count_tokens) -> list[Chunk]:
    """Token-based at sentence boundaries; tables/lists/code stay intact."""
    max_tokens = cfg.get("max_tokens", 350)
    blocks = _split_blocks(text.splitlines())
    return [Chunk(c, {"mode": "token", "chunk_index": i,
                      "token_count": count_tokens(c)})
            for i, c in enumerate(_pack_blocks(
                blocks, max_tokens, count_tokens,
                cfg.get("hard_max_tokens", 6000)))]


# ---------------------------------------------------------------- mode 3: markdown

@dataclass
class _Section:
    path: list[tuple[int, str]]  # [(level, title), ...] from top to bottom
    lines: list[str] = field(default_factory=list)


def _parse_sections(text: str) -> list[_Section]:
    sections: list[_Section] = [_Section(path=[])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section(path=list(stack)))
        else:
            sections[-1].lines.append(line)
    return sections


def _breadcrumb(path: list[tuple[int, str]], title_depth: int,
                part: int | None, total_parts: int) -> str:
    """Heading header: title_depth levels upwards from the lowest one.

    title_depth=0 -> lowest level only; =1 -> lowest plus the one above, etc.
    The part numbering is attached to the lowest heading.
    """
    if not path:
        return ""
    shown = path[-(title_depth + 1):]
    lines = []
    for i, (level, title) in enumerate(shown):
        suffix = ""
        if i == len(shown) - 1 and total_parts > 1:
            suffix = f" (Part {part} of {total_parts})"
        lines.append(f"{'#' * level} {title}{suffix}")
    return "\n".join(lines)


def chunk_markdown(text: str, cfg: dict, count_tokens) -> list[Chunk]:
    """Chapter-wise chunks: every heading starts a section whose body is split
    token-based (preserving sentences and tables). Each chunk is prefixed with
    its heading path; split sections are numbered Part 1..n."""
    max_tokens = cfg.get("max_tokens", 350)
    min_tokens = cfg.get("min_tokens", 0)
    title_depth = cfg.get("title_depth", 2)

    chunks: list[Chunk] = []
    for section in _parse_sections(text):
        body = "\n".join(section.lines).strip()
        if not body:
            continue
        blocks = _split_blocks(section.lines)
        bodies = _pack_blocks(blocks, max_tokens, count_tokens,
                              cfg.get("hard_max_tokens", 6000))
        total = len(bodies)
        for part_idx, part_body in enumerate(bodies, start=1):
            if min_tokens and total == 1 and count_tokens(part_body) < min_tokens \
                    and not section.path:
                continue  # tiny preamble without a heading
            head = _breadcrumb(section.path, title_depth, part_idx, total)
            full = f"{head}\n\n{part_body}" if head else part_body
            chunks.append(Chunk(full, {
                "mode": "markdown",
                "chunk_index": len(chunks),
                "heading_path": [t for _, t in section.path],
                "part": part_idx if total > 1 else None,
                "total_parts": total,
                "token_count": count_tokens(full),
            }))
    return chunks


# YAML front matter: ---\n...\n--- at the very start of the file
_YAML_HEADER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.DOTALL)


def strip_yaml_header(text: str) -> tuple[str, bool]:
    """Remove a YAML header at the start of the file, if present.

    Returns (text, removed?); without a header the text stays unchanged.
    """
    m = _YAML_HEADER_RE.match(text.lstrip("﻿"))
    if m:
        return text.lstrip("﻿")[m.end():], True
    return text, False


# Markdown link forms
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")        # ![alt](url)
_MD_INLINE_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")   # [text](url)
_MD_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")     # [text][ref]
_MD_LINK_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*\S+")      # [ref]: url
_AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>")            # <https://...>
_BARE_URL_RE = re.compile(r"(?<![\w(])(?:https?://|www\.)\S+")
_SPACE_CLEANUP_RE = re.compile(r"[ \t]{2,}")


def strip_links(text: str) -> str:
    """Remove URLs while keeping the link text.

    `[ColBERT](https://arxiv.org/...)` -> `ColBERT`; link definition lines
    (`[ref]: https://...`) and bare URLs are dropped entirely. Code blocks are
    left untouched. Intended for the embedding text - the chunk itself keeps
    its links.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if _MD_LINK_DEF_RE.match(line):
            continue  # pure link definition line
        original = line
        line = _MD_IMAGE_RE.sub(r"\1", line)
        line = _MD_INLINE_LINK_RE.sub(r"\1", line)
        line = _MD_REF_LINK_RE.sub(r"\1", line)
        line = _AUTOLINK_RE.sub("", line)
        line = _BARE_URL_RE.sub("", line)
        if line == original:
            out.append(line)
            continue
        # only where a link was actually removed: close the resulting gaps
        # (table rows keep their alignment)
        if not line.lstrip().startswith("|"):
            line = _SPACE_CLEANUP_RE.sub(" ", line)
        out.append(line.rstrip())
    # collapse triple blank lines left behind by removed definition lines
    result = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def redact_markdown(text: str) -> str:
    """Remove markdown characters (# before headings, all *), rest verbatim.

    Optional, because markdown syntax can slightly bias the embedding.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^(\s*)#{1,6}\s*", r"\1", line)  # heading marks
        lines.append(line.replace("*", ""))
    return "\n".join(lines)


CHUNKERS = {
    "marker": chunk_marker,
    "token": chunk_token,
    "markdown": chunk_markdown,
}


def chunk_text(text: str, cfg: dict) -> list[Chunk]:
    mode = cfg.get("mode", "markdown")
    if mode not in CHUNKERS:
        raise ValueError(f"Unknown chunking mode: {mode!r} "
                         f"(available: {', '.join(CHUNKERS)})")
    count_tokens = make_token_counter(cfg.get("tokenizer", "heuristic"))
    yaml_stripped = False
    if cfg.get("strip_yaml_header", True):
        text, yaml_stripped = strip_yaml_header(text)
    chunks = CHUNKERS[mode](text, cfg, count_tokens)
    if yaml_stripped:
        for chunk in chunks:
            chunk.meta["yaml_header_stripped"] = True
    if cfg.get("redact_markdown"):
        for chunk in chunks:
            chunk.text = redact_markdown(chunk.text)
            chunk.meta["redacted"] = True
    if cfg.get("strip_links_for_embedding"):
        # remove links from the embedding text only - the chunk keeps them
        for chunk in chunks:
            stripped = strip_links(chunk.text)
            if stripped != chunk.text:
                chunk.embed_text = stripped
                chunk.meta["links_stripped_for_embedding"] = True
    return chunks
