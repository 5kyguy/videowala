from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..config import settings


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    """
    Minimal Ollama HTTP wrapper used by Videowala.

    Notes:
    - Ollama "unload" is approximated by sending `keep_alive=0` so the model is evicted from Ollama memory.
    - Because Ollama keep/unload is global to the Ollama server, we serialize calls with a lock to
      keep the current "one stage at a time" assumption intact.
    """

    _LOCK = threading.Lock()

    def __init__(self, base_url: str | None = None, *, timeout_s: float = 180.0) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout_s = timeout_s

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise OllamaClientError(f"Ollama returned non-JSON response: {body[:500]}") from exc
                if not isinstance(parsed, dict):
                    raise OllamaClientError(f"Ollama returned unexpected response type: {type(parsed)}")
                return parsed
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = "(unable to read error body)"
            raise OllamaClientError(f"Ollama HTTP {exc.code} for {path}: {body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise OllamaClientError(f"Ollama request failed for {path}: {exc}") from exc

    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        images: list[str] | None = None,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        format: Any | None = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model, "stream": stream}
        if prompt is not None:
            payload["prompt"] = prompt
        if images:
            payload["images"] = images
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options
        if format is not None:
            payload["format"] = format

        with self._LOCK:
            resp = self._post_json("/api/generate", payload)

        response_text = str(resp.get("response") or "")
        return response_text

    def embed(
        self,
        *,
        model: str,
        input_texts: list[str],
        dimensions: int,
        truncate: bool = True,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_texts if len(input_texts) != 1 else input_texts[0],
            "dimensions": dimensions,
            "truncate": truncate,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options

        with self._LOCK:
            resp = self._post_json("/api/embed", payload)

        embeddings = resp.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaClientError("Ollama /api/embed returned missing/invalid 'embeddings' field.")

        out: list[list[float]] = []
        for v in embeddings:
            if not isinstance(v, list):
                raise OllamaClientError("Ollama /api/embed returned embedding vectors with invalid shape.")
            out.append([float(x) for x in v])
        return out

    def unload(self, *, model: str) -> None:
        # Ollama documents model unload as a generate call with keep_alive=0.
        # We avoid sending prompt to reduce accidental generation requirements.
        with self._LOCK:
            _ = self._post_json("/api/generate", {"model": model, "keep_alive": 0, "stream": False})


ollama_client = OllamaClient()

