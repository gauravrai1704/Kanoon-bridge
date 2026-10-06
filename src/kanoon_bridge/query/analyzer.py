"""Query analyzer: raw query -> AnalyzedQuery.  [owner: C, with B for sections]

Steps (each one appends to `trace` so --debug and the video can show the rewrites):

    1. parse facet filters + Boolean syntax     query/parser.py              (working)
    2. detect language                          text/transliterate.py        (baseline working)
    3. Devanagari -> Roman, spelling normalise  text/transliterate.py        (working)
    4. lexicon + phonetic expansion             LegalLexicon, PhoneticIndex  (working)
       e.g. "chaku maara" -> knife, stab, hurt, attempt to murder
    5. same text pipeline as documents          text/pipeline.analyze_text   (working; B/C steps plug in)
    6. code in force from incident date         CollisionResolver            (working)
    7. cross-code expansion of sections         VersionNormalizer.equivalents (working)

Each step is recorded in the trace (--debug, the app's "How the query was understood").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.query.parser import parse
from kanoon_bridge.schema import AnalyzedQuery, Code, Query
from kanoon_bridge.text.pipeline import TextResources, analyze_text
from kanoon_bridge.text.tokenize import is_section_token
from kanoon_bridge.text.transliterate import FUNCTION_WORDS


@dataclass
class QueryAnalyzer:
    cfg: Config
    text_res: TextResources
    lexicon: object | None = None          # text.transliterate.LegalLexicon
    phonetic: object | None = None         # text.phonetic.PhoneticIndex (built from index vocabulary)
    steps: dict = field(default_factory=lambda: {"transliterate": True, "lexicon": True, "phonetic": True})
    # ^ switches for the E4 language ablation (eval/ablation.py); all on in normal use

    @classmethod
    def load(cls, cfg: Config | None = None, vocabulary: list[str] | None = None) -> "QueryAnalyzer":
        cfg = cfg or load_config()
        an = cls(cfg=cfg, text_res=TextResources.load(cfg))
        try:
            from kanoon_bridge.text.transliterate import LegalLexicon

            an.lexicon = LegalLexicon.load(cfg)
        except FileNotFoundError:
            pass
        if vocabulary and cfg.text.phonetic_matching:
            from kanoon_bridge.text.phonetic import PhoneticIndex

            an.phonetic = PhoneticIndex.build(vocabulary)
        return an

    def analyze(self, query: Query) -> AnalyzedQuery:
        from kanoon_bridge.text.transliterate import detect_lang, devanagari_to_roman, normalize_roman

        aq = AnalyzedQuery(query=query)

        # 1. filters
        parsed = parse(query.text)
        query.filters = {**parsed.filters, **query.filters}
        if "state" in query.filters and not query.state:
            query.state = query.filters["state"]
        if "date" in query.filters and not query.incident_date:
            from datetime import date

            query.incident_date = date.fromisoformat(query.filters["date"])
        text = parsed.text_for_ranking
        aq.trace.append(("filters", str(query.filters)))
        if parsed.tree is not None:
            from kanoon_bridge.query.parser import show

            aq.boolean = parsed.tree
            aq.trace.append(("boolean", f"{show(parsed.tree)}  (ranked on: {text})"))

        # 2. language
        aq.detected_lang = query.lang or detect_lang(text)
        aq.trace.append(("language", aq.detected_lang))

        # 3. transliteration / spelling normalisation (C)
        normalised = text
        if aq.detected_lang in ("hi", "hinglish") and self.steps.get("transliterate", True):
            roman = devanagari_to_roman(text) if aq.detected_lang == "hi" else text
            # section mentions ("BNS 103") are kept as typed; other words get the spelling rules
            from kanoon_bridge.text.tokenize import extract_sections

            keep = {i for m in extract_sections(roman) for i in range(m.start, m.end)}
            normalised = re.sub(r"[A-Za-z]+", lambda m: m.group(0) if m.start() in keep
                                else (normalize_roman(m.group(0)) or m.group(0)), roman)
        aq.transliterated = normalised
        aq.trace.append(("normalised", normalised))

        # 4. lexicon + phonetic expansion (C)
        #    lexicon: exact -> normalised spelling -> (phonetic step on) same Hindi phonetic code
        #    phonetic vs index vocabulary: a Roman word the index has never seen (5+ letters)
        #    picks up at most 3 same-sounding index terms at weight 0.3 (names, misspellings)
        if aq.detected_lang in ("hi", "hinglish"):
            use_phonetic = self.steps.get("phonetic", True)
            looked: list[str] = []
            if self.lexicon is not None and self.steps.get("lexicon", True):
                for word in re.findall(r"[^\W\d_]+", normalised):
                    if word.lower() in FUNCTION_WORDS:            # function words: never legal terms
                        continue
                    hits = self.lexicon.lookup(word, use_phonetic=use_phonetic)
                    if hits:
                        looked.append(f"{word}->{'/'.join(t for t, _ in hits)}")
                    for term, weight in hits:
                        for t in analyze_text(term, self.text_res):
                            aq.expanded_terms[t] = max(aq.expanded_terms.get(t, 0.0), weight)
                aq.trace.append(("lexicon", ", ".join(looked) or "no lexicon matches"))
            if use_phonetic and self.phonetic is not None:
                sounds: list[str] = []
                for word in re.findall(r"[^\W\d_]+", normalised):
                    w = word.lower()
                    if (len(w) < 5 or not w.isalpha() or not w.isascii() or w in self.text_res.stopwords
                            or w in FUNCTION_WORDS):
                        continue
                    if any(w in l.split("->")[0] for l in looked):
                        continue
                    matches = sorted(self.phonetic.matches(w))
                    if 0 < len(matches) <= 3:
                        for m in matches:
                            aq.expanded_terms[m] = max(aq.expanded_terms.get(m, 0.0), 0.3)
                        sounds.append(f"{w}~{'/'.join(matches)}")
                if sounds:
                    aq.trace.append(("phonetic", ", ".join(sounds)))
        aq.trace.append(("expanded", str(aq.expanded_terms)))

        # 5. shared pipeline (tokenise, stop words, stem, collision, version norm)
        aq.tokens = analyze_text(normalised, self.text_res, lang="en", date=query.incident_date)
        aq.sections = [t[len("sec:"):] for t in aq.tokens if is_section_token(t)]
        aq.offence_ids = [t for t in aq.tokens if t.startswith("off:")]
        aq.trace.append(("tokens", " ".join(aq.tokens)))
        resolver = self.text_res.resolver
        for tok, readings in getattr(resolver, "last_readings", []) or []:
            aq.trace.append(("collision", f"{tok} -> " + ", ".join(
                f"{r.section_ref} {r.confidence:.2f} ({r.reason})" for r in readings)))
        if aq.offence_ids and self.text_res.normalizer is not None:
            aq.trace.append(("offences", "; ".join(f"{o} = {self.text_res.normalizer.label(o)}"
                                                  for o in dict.fromkeys(aq.offence_ids))))

        # 6. code in force
        resolver = self.text_res.resolver
        if resolver is not None and query.incident_date is not None:
            code = resolver.code_for_date(query.incident_date)
            aq.code_in_force = Code(code) if code else Code.UNKNOWN
        aq.trace.append(("code_in_force", aq.code_in_force.value))

        # 7. cross-code expansion (B)
        norm = self.text_res.normalizer
        if norm is not None:
            try:
                for ref in aq.sections:
                    for eq in norm.equivalents(ref):
                        aq.expanded_terms[f"sec:{eq}"] = max(aq.expanded_terms.get(f"sec:{eq}", 0.0), 0.8)
                aq.trace.append(("cross_code", str([k for k in aq.expanded_terms if k.startswith("sec:")])))
            except NotImplementedError:
                aq.trace.append(("cross_code", "skipped: not implemented"))
        return aq
