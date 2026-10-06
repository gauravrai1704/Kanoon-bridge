"""Language detection, transliteration and the Hinglish legal lexicon.  [owner: C]

Goal: "mere bhai ko chaku maara", "मेरे भाई को चाकू मारा" and "my brother was stabbed"
should reach the same statutes. Three pieces:

  detect_lang(text)          -> "en" | "hi" (Devanagari) | "hinglish" (Roman Hindi)
  devanagari_to_roman(text)  -> Roman form, so both scripts share one lexicon lookup
  normalize_roman(word)      -> spelling-normalised Roman Hindi ("maaraa" -> "mara")
  LegalLexicon.lookup(word)  -> English legal terms with weights ("hatya" -> murder)

Lexicon file: data/lexicons/hinglish_legal_terms.csv
    roman, devanagari, english, weight
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config, project_path

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Small seed list of very common Roman-Hindi function words, used only for language detection.
# TODO(C): extend, or train a tiny char n-gram classifier on the COMI-LINGUA LID data.
_HINGLISH_MARKERS = {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "mera", "mere", "meri", "aur",
    "kya", "nahi", "tha", "thi", "kiya", "kar", "diya", "liya", "wala", "bhai", "maara", "mara",
}


def detect_lang(text: str) -> str:
    """Rough language guess. Working baseline; TODO(C): improve and evaluate on E4 queries."""
    if _DEVANAGARI.search(text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "en"
    share = sum(w in _HINGLISH_MARKERS for w in words) / len(words)
    return "hinglish" if share >= 0.2 else "en"


def devanagari_to_roman(text: str) -> str:
    """Transliterate Devanagari to a simple Roman scheme.

    TODO(C): use a character table (or the `indic-transliteration` package, declared in the
    report), then pass the result through normalize_roman so both scripts meet.
    """
    raise NotImplementedError("TODO(C): Devanagari -> Roman")


def normalize_roman(word: str) -> str:
    """Normalise Roman Hindi spellings so variants meet.

    TODO(C): rules such as aa->a, ee->i, oo->u, w->v, ph->f, merge aspirated forms
    (th->t, dh->d, kh->k, bh->b, chh->ch), collapse doubled consonants (tt->t). Keep a table
    of the rules in the report; they are part of the IR story (normalisation).
    """
    raise NotImplementedError("TODO(C): Roman Hindi spelling normalisation")


@dataclass
class LegalLexicon:
    """Roman/Devanagari legal terms -> English terms used in judgments and statutes."""

    entries: dict[str, list[tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def load(cls, cfg: Config | None = None) -> "LegalLexicon":
        """Load the CSV (working). Keys are stored exactly as written; lookup normalises."""
        cfg = cfg or load_config()
        lex = cls()
        path = project_path(cfg.paths.lexicon_hinglish)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(line for line in f if not line.startswith("#")):
                weight = float(row.get("weight") or 1.0)
                for key in (row.get("roman", ""), row.get("devanagari", "")):
                    key = key.strip().lower()
                    if key:
                        lex.entries[key].append((row["english"].strip().lower(), weight))
        return lex

    def lookup(self, word: str) -> list[tuple[str, float]]:
        """English terms for a Hindi/Hinglish word.

        TODO(C): try the exact key, then normalize_roman(word), then the phonetic code
        (text/phonetic.py) against normalised keys. Return [] if nothing matches.
        """
        raise NotImplementedError("TODO(C): lexicon lookup with normalisation")
