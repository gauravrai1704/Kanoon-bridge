"""Resolve bare or colliding section numbers.  [owner: Gaurav — working]

The problem
-----------
"Section 302" means punishment for murder under IPC but wounding religious feelings under BNS.
When the text names no code ("sec:?:302"), we decide which code is meant:

  1. a code word nearby — already handled by the tokenizer ("302 IPC" -> sec:ipc:302)
  2. the date: incident or judgment date before 2024-07-01 favours IPC, on/after favours BNS
     (prior = collision.date_prior; no date -> 50/50)
  3. offence words nearby: each context token matching a keyword of that reading's offence
     (data/crosswalk/offence_ids.csv, stemmed like the text) multiplies the reading's score
     by (1 + collision.context_weight)
  4. readings are normalised to sum to 1; if the best is below collision.keep_second_below,
     both readings are emitted as index terms, and the reasons go to the trace

    resolve("302", context=["murder", "stab"], date=None)
      -> [Reading("ipc:302", 0.9, "context: murder"), Reading("bns:302", 0.1, ...)]

Evaluation warning (see proposal): the E2 collision queries must be written BEFORE tuning these
weights, or the test measures the rules against themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.text.tokenize import SECTION_PREFIX

UNKNOWN_PREFIX = f"{SECTION_PREFIX}?:"


@dataclass
class Reading:
    section_ref: str          # "ipc:302"
    confidence: float         # readings for one mention sum to 1.0
    reason: str               # shown in --debug and the video


@dataclass
class CollisionResolver:
    bns_in_force: date
    normalizer: object | None = None   # VersionNormalizer, for offence keywords and valid sections
    date_prior: float = 0.75
    context_weight: float = 4.0
    keep_second_below: float = 0.7
    _keywords: dict[str, set[str]] = field(default_factory=dict)   # offence id -> stemmed keywords
    last_readings: list[tuple[str, list[Reading]]] = field(default_factory=list)

    @classmethod
    def load(cls, cfg: Config | None = None) -> "CollisionResolver":
        cfg = cfg or load_config()
        from kanoon_bridge.text.version_norm import VersionNormalizer

        try:
            norm = VersionNormalizer.load(cfg)
        except FileNotFoundError:
            norm = None
        c = cfg.get("collision", {})
        res = cls(bns_in_force=date.fromisoformat(cfg.legal.bns_in_force), normalizer=norm,
                  date_prior=c.get("date_prior", 0.75), context_weight=c.get("context_weight", 4.0),
                  keep_second_below=c.get("keep_second_below", 0.7))
        res._prepare_keywords()
        return res

    def _prepare_keywords(self) -> None:
        if self.normalizer is None:
            return
        from kanoon_bridge.text.stem import stem_english

        for oid, info in self.normalizer.offences.items():
            words = set(info.get("keywords", []))
            for w in info.get("label", "").lower().split():
                if len(w) > 3:
                    words.add(w.strip(",.()"))
            self._keywords[oid] = {stem_english(w) for w in words if w}

    def code_for_date(self, d: date | None) -> str | None:
        """'ipc' before 1 July 2024, 'bns' on or after it, None if no date."""
        if d is None:
            return None
        return "bns" if d >= self.bns_in_force else "ipc"

    def _exists(self, ref: str) -> bool:
        if self.normalizer is None:
            return True
        return bool(self.normalizer.to_offences(ref))

    def resolve(self, section: str, context: list[str], date: date | None = None) -> list[Reading]:
        """Readings for a bare section number, most likely first (confidences sum to 1)."""
        section = section.lower()
        candidates = [f"{code}:{section}" for code in ("ipc", "bns") if self._exists(f"{code}:{section}")]
        if not candidates:
            return [Reading(f"?:{section}", 1.0, "number not in either code")]
        if len(candidates) == 1:
            return [Reading(candidates[0], 1.0, "only valid in one code")]

        in_force = self.code_for_date(date)
        context_set = set(context)
        scored: list[tuple[str, float, str]] = []
        for ref in candidates:
            code = ref.partition(":")[0]
            prior = 0.5 if in_force is None else (self.date_prior if code == in_force else 1 - self.date_prior)
            reasons = [f"date -> {in_force}" if in_force else "no date"]
            hits: set[str] = set()
            for off in (self.normalizer.offences_for(ref) if self.normalizer else []):
                hits |= self._keywords.get(off, set()) & context_set
            score = prior * (1 + self.context_weight) ** len(hits)
            if hits:
                reasons.append("context: " + ", ".join(sorted(hits)))
            scored.append((ref, score, "; ".join(reasons)))
        total = sum(s for _, s, _ in scored) or 1.0
        readings = [Reading(r, s / total, why) for r, s, why in scored]
        return sorted(readings, key=lambda x: -x.confidence)

    def resolve_tokens(self, tokens: list[str], date: date | None = None, window: int = 10) -> list[str]:
        """Replace each 'sec:?:<n>' with its most likely reading; keep the runner-up too when unsure.

        The readings of the last call are kept in `last_readings` for the query trace.
        """
        out: list[str] = []
        self.last_readings = []
        for i, tok in enumerate(tokens):
            if not tok.startswith(UNKNOWN_PREFIX):
                out.append(tok)
                continue
            section = tok[len(UNKNOWN_PREFIX):]
            context = tokens[max(0, i - window):i] + tokens[i + 1:i + 1 + window]
            readings = self.resolve(section, context, date)
            self.last_readings.append((tok, readings))
            out.append(SECTION_PREFIX + readings[0].section_ref)
            if len(readings) > 1 and readings[0].confidence < self.keep_second_below:
                out.append(SECTION_PREFIX + readings[1].section_ref)
        return out
