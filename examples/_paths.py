"""Resolve a model config or steering vector to an absolute path.

A demo asks for a name relative to ``model_configs/`` or ``steering_vectors/``, not a
directory, so it runs the same from the repo root or from anywhere else.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(top: str, rel: str) -> str:
    path = REPO_ROOT / top / rel.lstrip("/")
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"{rel!r} is not under {top}/. Add it there, or pass an explicit path.")


def config_path(rel: str) -> str:
    """Absolute path for ``model_configs/<rel>``."""
    return _resolve("model_configs", rel)


def vector_path(rel: str) -> str:
    """Absolute path for ``steering_vectors/<rel>``."""
    return _resolve("steering_vectors", rel)


__all__ = ["REPO_ROOT", "config_path", "vector_path"]
