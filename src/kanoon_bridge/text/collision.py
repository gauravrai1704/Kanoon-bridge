"""Resolve bare or colliding section numbers.  [owner: B — trickiest part of the project]

The problem
-----------
"Section 302" means murder under IPC but wounding religious feelings under BNS. When the
text names no code ("sec:?:302"), we must guess which code is meant, from:

  1. a code word nearby (handled by the tokenizer when present)
  2. the date: incident or judgment date before 2024-07-01 -> IPC, on/after -> BNS
  3. offence words nearby ("murder", "killed" vs "religious", "feelings") matched against
     the keywords column of data/crosswalk/offence_ids.csv
  4. otherwise keep BOTH readings with confidence weights, and tell the user

Output: one or more resolved section refs with confidence, e.g.
    resolve("302", context=["murder", "knife"], date=None)
      -> [Reading("ipc:302", 0.9, "context: murder"), Reading("bns:302", 0.1, "fallback")]

Evaluation warning (see proposal): write the E2 collision queries BEFORE finalising these
rules, or the test only measures the rules against themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from kanoon_bridge.config import Config, load_config


@dataclass
class Reading:
    section_ref: str          # "ipc:302"
    confidence: float         # readings for one mention sum to 1.0
    reason: str               # shown in --debug and the video


@dataclass
class CollisionResolver:
    bns_in_force: date
    normalizer: object | None = None   # VersionNormalizer, for offence keywords

    @classmethod
    def load(cls, cfg: Config | None = None) -> "CollisionResolver":
        cfg = cfg or load_config()
        from kanoon_bridge.text.version_norm import VersionNormalizer

        try:
            norm = VersionNormalizer.load(cfg)
        except FileNotFoundError:
            norm = None
        return cls(bns_in_force=date.fromisoformat(cfg.legal.bns_in_force), normalizer=norm)

    def code_for_date(self, d: date | None) -> str | None:
        """'ipc' before 1 July 2024, 'bns' on or after it, None if no date. (working)"""
        if d is None:
            return None
        return "bns" if d >= self.bns_in_force else "ipc"

    def resolve(self, section: str, context: list[str], date: date | None = None) -> list[Reading]:
        """Readings for a bare section number, most likely first.

        TODO(B): implement steps 2-4 above. Start simple (date rule + keyword overlap),
        measure on E2, then refine. Keep `reason` short and human-readable.
        """
        raise NotImplementedError("TODO(B): resolve bare section numbers")

    def resolve_tokens(self, tokens: list[str], date: date | None = None, window: int = 10) -> list[str]:
        """Replace each 'sec:?:<n>' token with its most likely reading(s).

        TODO(B): use `resolve` with the `window` tokens either side as context. Emit the top
        reading; if confidence < 0.7 also emit the second (both become index terms).
        """
        raise NotImplementedError("TODO(B): resolve bare section tokens")
