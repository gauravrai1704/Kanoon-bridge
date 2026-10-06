"""Query analyzer: raw query -> AnalyzedQuery.  [owner: C, with B for sections]

Steps (each one appends to `trace` so --debug and the video can show the rewrites):

    1. parse facet filters                      query/parser.py              (working)
    2. detect language                          text/transliterate.py        (baseline working)
    3. Devanagari -> Roman, spelling normalise  text/transliterate.py        TODO(C)
    4. lexicon + phonetic expansion             LegalLexicon, PhoneticIndex  TODO(C)
       e.g. "chaku maara" -> knife, stab, hurt, attempt to murder
    5. same text pipeline as documents          text/pipeline.analyze_text   (working; B/C steps plug in)
    6. code in force from incident date         CollisionResolver            (working)
    7. cross-code expansion of sections         VersionNormalizer.equivalents TODO(B)

Unfinished steps are skipped and noted in the trace, so the pipeline runs end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.query.parser import parse
from kanoon_bridge.schema import AnalyzedQuery, Code, Query
from kanoon_bridge.text.pipeline import TextResources, analyze_text
from kanoon_bridge.text.tokenize import is_section_token


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
        except (FileNotFoundError, NotImplementedError):
            pass
        if vocabulary and cfg.text.phonetic_matching:
            try:
                from kanoon_bridge.text.phonetic import PhoneticIndex

                an.phonetic = PhoneticIndex.build(vocabulary)
            except NotImplementedError:
                pass
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
        text = parsed.free_text
        aq.trace.append(("filters", str(query.filters)))

        # 2. language
        aq.detected_lang = query.lang or detect_lang(text)
        aq.trace.append(("language", aq.detected_lang))

        # 3. transliteration / spelling normalisation (C)
        normalised = text
        if aq.detected_lang in ("hi", "hinglish") and self.steps.get("transliterate", True):
            try:
                roman = devanagari_to_roman(text) if aq.detected_lang == "hi" else text
                normalised = " ".join(normalize_roman(w) for w in roman.split())
            except NotImplementedError:
                aq.trace.append(("transliterate", "skipped: not implemented"))
        aq.transliterated = normalised
        aq.trace.append(("normalised", normalised))

        # 4. lexicon + phonetic expansion (C)
        if aq.detected_lang in ("hi", "hinglish") and self.lexicon is not None and self.steps.get("lexicon", True):
            try:
                for word in normalised.split():
                    for term, weight in self.lexicon.lookup(word):
                        for t in analyze_text(term, self.text_res):
                            aq.expanded_terms[t] = max(aq.expanded_terms.get(t, 0.0), weight)
            except NotImplementedError:
                aq.trace.append(("lexicon", "skipped: not implemented"))
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
