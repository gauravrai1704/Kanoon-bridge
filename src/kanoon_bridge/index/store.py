"""Save and load indexes and other artefacts under data/processed/.  [owner: A — working]

Everything built offline (indexes, graph, authority scores) is saved here once,
so the app and the evaluator start instantly.

    from kanoon_bridge.index.store import save, load, exists
    save(index, "positional")          # -> data/processed/index/positional.pkl
    index = load("positional")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from kanoon_bridge.config import load_config, project_path


def index_dir() -> Path:
    path = project_path(load_config().paths.index_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(name: str, fmt: str) -> Path:
    return index_dir() / f"{name}.{fmt}"


def save(obj: Any, name: str, fmt: str = "pkl") -> Path:
    """Save `obj` as data/processed/index/<name>.<fmt>; fmt is 'pkl' or 'json'."""
    if fmt not in {"pkl", "json"}:
        raise ValueError(f"unknown format: {fmt}")

    path = _path(name, fmt)

    if fmt == "pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    return path


def load(name: str, fmt: str = "pkl") -> Any:
    """Load an index or artefact from the processed index directory."""
    if fmt not in {"pkl", "json"}:
        raise ValueError(f"unknown format: {fmt}")

    path = _path(name, fmt)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/02_build_index.py first"
        )

    if fmt == "pkl":
        with open(path, "rb") as f:
            return pickle.load(f)

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exists(name: str, fmt: str = "pkl") -> bool:
    """Return True if the requested stored artefact exists."""
    if fmt not in {"pkl", "json"}:
        raise ValueError(f"unknown format: {fmt}")

    return _path(name, fmt).exists()