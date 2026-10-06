"""Load configs/*.yaml into an object with attribute access.  [owner: D — working]

Usage:
    from kanoon_bridge.config import load_config, project_path
    cfg = load_config()                 # configs/default.yaml
    cfg.bm25.k1                         # 1.6
    cfg.zones["ratio"]                  # dict-style access also works
    project_path(cfg.paths.docs)        # absolute Path inside the repo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Config(dict):
    """A dict whose keys are also attributes; nested dicts become Config too."""

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(name: str = "default.yaml", overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML file from configs/ and apply optional overrides (nested dicts merged)."""
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


def project_path(relative: str | Path) -> Path:
    """Resolve a path from the config (relative to the repo root) to an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
