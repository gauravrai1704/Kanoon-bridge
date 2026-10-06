"""Query analyzer: raw query -> AnalyzedQuery.  [owner: C, with B for sections]

Steps (each one appends to `trace` so --debug and the video can show the rewrites):

    1. parse facet filters + Boolean syntax     query/parser.py              (working)
    2. detect language                          text/transliterate.py        (baseline working)
    3. Devanagari -> Roman, spelling normalise  text/transliterate.py        (working)
    4. lexicon + phonetic expansion             LegalLexicon, PhoneticIndex  (working)
       e.g. "chaku maara" -> knife, stab, hurt, attempt to murder
    1b. wildcard terms ("extort*")              text/spell.py k-gram index   (working)
    5. same text pipeline as documents          text/pipeline.analyze_text   (working)
    5b. spelling correction / did you mean      text/spell.Speller           (working)
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


_WILDCARD_RE = re.compile(r"[A-Za-z]*\*[A-Za-z*]*")


@dataclass
class QueryAnalyzer:
    cfg: Config
    text_res: TextResources
    lexicon: object | None = None          # text.transliterate.LegalLexicon
    phonetic: object | None = None         # text.phonetic.PhoneticIndex (built from index vocabulary)
    speller: object | None = None          # text.spell.Speller (built from both indexes' vocabulary)
    steps: dict = field(default_factory=lambda: {"transliterate": True, "lexicon": True, "phonetic": True,
                                                 "spelling": True, "wildcard": True})
    # ^ switches for the E4 language ablation and the typo experiment (eval/); all on in normal use

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

    def _correct(self, aq: AnalyzedQuery, text: str) -> None:
        from kanoon_bridge.text.stem import stem_english
        from kanoon_bridge.text.tokenize import extract_sections

        min_len = 4 if aq.detected_lang == "en" else 6            # Hinglish: the phonetic step leads
        weight = self.cfg.text.get("spelling_weight", 0.9)
        in_section = {i for m in extract_sections(text) for i in range(m.start, m.end)}
        seen: set[str] = set()
        for m in re.finditer(r"[A-Za-z]+", text):
            w = m.group(0).lower()
            if (m.start() in in_section or w in seen or len(w) < min_len or w in self.text_res.stopwords
                    or w in FUNCTION_WORDS or (self.lexicon is not None and aq.detected_lang != "en"
                                               and self.lexicon.lookup(w, use_phonetic=False))):
                continue
            seen.add(w)
            c = self.speller.check(w, stem_english(w))
            if c is None:
                continue
            aq.corrections.append(c)
            if c.applied:
                aq.expanded_terms[c.term] = max(aq.expanded_terms.get(c.term, 0.0), weight)
        if aq.corrections:
            fixed = aq.query.text
            for c in aq.corrections:
                fixed = re.sub(rf"\b{re.escape(c.word)}\b", c.display, fixed, flags=re.IGNORECASE)
            aq.suggestion = fixed if fixed != aq.query.text else None
            aq.trace.append(("spelling", "; ".join(
                f"{c.word} -> {c.display} ({c.edits} edit{'s' if c.edits > 1 else ''}, df {c.df}"
                f"{'' if c.applied else ', suggestion only'})" for c in aq.corrections)))

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
        for key, edge, default in (("after", "date_from", "-01-01"), ("before", "date_to", "-12-31")):
            if key in query.filters and edge not in query.filters:
                v = query.filters[key]
                query.filters[edge] = v + default if re.fullmatch(r"\d{4}", v) else v
        text = parsed.text_for_ranking
        aq.trace.append(("filters", str(query.filters)))

        # 1b. wildcard terms ("extort*", "*bail"): expanded through the k-gram index, removed
        #     from the text before tokenising
        patterns = [p for p in _WILDCARD_RE.findall(text) if len(p.replace("*", "")) >= 2]
        if patterns:
            text = _WILDCARD_RE.sub(" ", text)
            if self.speller is not None and self.steps.get("wildcard", True):
                w_weight = self.cfg.text.get("wildcard_weight", 0.8)
                for pat in patterns:
                    terms = self.speller.expand_wildcard(pat.lower(), self.cfg.text.get("wildcard_max_terms", 30))
                    aq.wildcards[pat.lower()] = terms
                    for t in terms:
                        aq.expanded_terms[t] = max(aq.expanded_terms.get(t, 0.0), w_weight)
                aq.trace.append(("wildcard", "; ".join(f"{p} -> {' '.join(t[:8]) or 'no match'}"
                                                      + (" ..." if len(t) > 8 else "") for p, t in aq.wildcards.items())))
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

        # 5b. spelling correction (IIR ch. 3): unseen words get their closest frequent index term
        # (skipped for long queries such as whole judgments in E1: nobody types those, and every
        #  party name would be "corrected")
        if (self.speller is not None and self.steps.get("spelling", True)
                and len(normalised.split()) <= self.cfg.text.get("spelling_max_query_words", 40)):
            self._correct(aq, normalised)
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
