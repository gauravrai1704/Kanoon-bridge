"""The ONE text pipeline used for both documents (at index time) and queries.  [shared — working]

Team rule 2: documents and queries must go through the same steps, or normalisation breaks
(e.g. "IPC 302" in a judgment and "BNS 103" in a query must both become "off:murder").

    tokens = analyze_text("Convicted u/s 302 IPC", resources, date=decision_date)

Steps (each can be switched off for ablations via TextOptions):
    tokenize        text/tokenize.py        [A]  working
    stop words      lexicons/*.txt          [A]  working
    stemming        text/stem.py            [C]  English (Porter) + Hindi (light suffix stripper)
    collision       text/collision.py       [B]  resolves "sec:?:302" to a code (date + context)
    version norm    text/version_norm.py    [B]  adds "off:<offence>" tokens

Unfinished steps are skipped (with a one-time warning) so A and D can test end to end
before B and C land their parts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.text.tokenize import is_section_token, tokenize

log = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(step: str, err: Exception) -> None:
    if step not in _warned:
        log.warning("text pipeline: skipping '%s' (%s)", step, err)
        _warned.add(step)


@dataclass
class TextOptions:
    remove_stopwords: bool = True
    stem: bool = True
    resolve_collisions: bool = True
    version_normalise: bool = True


@dataclass
class TextResources:
    """Loaded once and shared: stop words, normaliser, collision resolver."""

    cfg: Config
    stopwords: set[str] = field(default_factory=set)
    normalizer: object | None = None       # text.version_norm.VersionNormalizer
    resolver: object | None = None         # text.collision.CollisionResolver

    @classmethod
    def load(cls, cfg: Config | None = None) -> "TextResources":
        cfg = cfg or load_config()
        res = cls(cfg=cfg, stopwords=_load_stopwords(cfg))
        try:
            from kanoon_bridge.text.version_norm import VersionNormalizer

            res.normalizer = VersionNormalizer.load(cfg)
        except (NotImplementedError, FileNotFoundError) as err:
            _warn_once("version_norm", err)
        try:
            from kanoon_bridge.text.collision import CollisionResolver

            res.resolver = CollisionResolver.load(cfg)
        except (NotImplementedError, FileNotFoundError) as err:
            _warn_once("collision", err)
        return res


def _load_stopwords(cfg: Config) -> set[str]:
    words: set[str] = set()
    for key in ("stopwords_english", "stopwords_legal", "stopwords_hindi"):
        if key not in cfg.paths:
            continue
        path = project_path(cfg.paths[key])
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                words.update(w.strip().lower() for w in f if w.strip() and not w.startswith("#"))
    return words


@lru_cache(maxsize=1)
def default_resources() -> TextResources:
    return TextResources.load()


def analyze_text(
    text: str,
    resources: TextResources | None = None,
    *,
    lang: str = "en",
    date: date | None = None,
    options: TextOptions | None = None,
) -> list[str]:
    """Text -> index terms. `date` is the judgment or incident date (used to resolve bare sections)."""
    res = resources or default_resources()
    opt = options or TextOptions()
    tokens = tokenize(text)

    if opt.remove_stopwords and res.stopwords:
        tokens = [t for t in tokens if is_section_token(t) or t not in res.stopwords]

    if opt.stem:
        try:
            from kanoon_bridge.text.stem import stem_tokens

            tokens = stem_tokens(tokens, lang=lang)
        except NotImplementedError as err:
            _warn_once("stem", err)

    if opt.resolve_collisions and res.resolver is not None:
        try:
            tokens = res.resolver.resolve_tokens(tokens, date=date)
        except NotImplementedError as err:
            _warn_once("collision", err)

    if opt.version_normalise and res.normalizer is not None:
        try:
            tokens = res.normalizer.normalize_tokens(tokens)
        except NotImplementedError as err:
            _warn_once("version_norm", err)

    return tokens
