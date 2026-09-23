"""Embedding providers. To add one: subclass Embedder and register it in
PROVIDERS (see README)."""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class Embedder(ABC):
    """Base class for all embedding providers."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.batch_size = cfg.get("batch_size", 16)

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (at most batch_size of them)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self.embed_batch(texts[i:i + self.batch_size]))
        return vectors


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OllamaEmbedder(Embedder):
    """Local models via the Ollama API (POST /api/embed).

    num_ctx raises the context window (bge-m3: up to 8192); truncate=True
    clips inputs that exceed the window instead of failing with HTTP 400.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.truncate = cfg.get("truncate", True)
        self.num_ctx = cfg.get("num_ctx")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts,
                   "truncate": self.truncate}
        if self.num_ctx:
            payload["options"] = {"num_ctx": self.num_ctx}
        try:
            data = _post_json(f"{self.base_url}/api/embed", payload, {})
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama error {e.code}: {e.read().decode(errors='replace')[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} "
                f"(is `ollama serve` running? is the model pulled?): {e}"
            ) from e
        return data["embeddings"]


class GeminiEmbedder(Embedder):
    """Google Gemini API (AI Studio). API key via environment variable
    (default: GEMINI_API_KEY) or embedding.api_key in config.json.

    Note on the free tier: Google trains on the inputs there - for sensitive
    data use the paid tier or stay local (Ollama).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.base_url = cfg.get("gemini_base_url", GEMINI_BASE_URL).rstrip("/")
        self.api_key = _gemini_key(cfg)
        self.output_dim = cfg.get("output_dimensionality")
        if not self.api_key:
            raise RuntimeError(
                "Gemini API key missing: set GEMINI_API_KEY "
                "(or embedding.api_key in config.json).")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self.model if self.model.startswith("models/") \
            else f"models/{self.model}"
        requests = []
        for text in texts:
            r = {"model": model, "content": {"parts": [{"text": text}]}}
            if self.output_dim:
                r["outputDimensionality"] = self.output_dim
            requests.append(r)
        try:
            data = _post_json(f"{self.base_url}/{model}:batchEmbedContents",
                              {"requests": requests},
                              {"x-goog-api-key": self.api_key})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini API error {e.code}: "
                               f"{e.read().decode(errors='replace')[:300]}") from e
        return [e["values"] for e in data["embeddings"]]


def _gemini_key(cfg: dict) -> str:
    return cfg.get("api_key") or os.environ.get(
        cfg.get("api_key_env", "GEMINI_API_KEY"), "")


PROVIDERS: dict[str, type[Embedder]] = {
    "ollama": OllamaEmbedder,
    "gemini": GeminiEmbedder,
}


def get_embedder(cfg: dict) -> Embedder:
    provider = cfg.get("provider", "ollama")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r} "
                         f"(available: {', '.join(PROVIDERS)})")
    return PROVIDERS[provider](cfg)


KNOWN_GEMINI_MODELS = ["gemini-embedding-001", "text-embedding-004"]

# Name patterns of common embedding models - used to sort them to the top
_EMBED_HINTS = ("embed", "bge", "e5", "arctic", "gte", "minilm", "nomic")


def list_models(cfg: dict) -> list[str]:
    """Models available for the provider (Ollama: installed models, embedding
    models first; Gemini: the known embedding models)."""
    provider = cfg.get("provider", "ollama")
    if provider == "gemini":
        return list(KNOWN_GEMINI_MODELS)
    if provider == "ollama":
        base = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
        except Exception:
            return []
        names = [m["name"] for m in data.get("models", [])]
        names = [n[:-len(":latest")] if n.endswith(":latest") else n
                 for n in names]
        return sorted(names, key=lambda n: (
            not any(h in n.lower() for h in _EMBED_HINTS), n))
    return []


def check_provider(cfg: dict) -> tuple[bool, str]:
    """Check whether the configured provider is ready (used by the GUI)."""
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        base = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=2):
                return True, "Ollama running"
        except Exception:
            return False, "Ollama unreachable"
    if provider == "gemini":
        if _gemini_key(cfg):
            return True, "Gemini: API key set"
        return False, "Gemini: GEMINI_API_KEY missing"
    if provider in PROVIDERS:
        return True, f"Provider: {provider}"
    return False, f"Unknown provider: {provider}"
