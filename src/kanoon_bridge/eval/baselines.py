"""Fair baselines for the version-dependent sets: BM25 that knows the crosswalk.  [owner: Gaurav — working]

Plain BM25 cannot know that BNS 103 is IPC 302, so comparing the core against it on E2, E3, E7–E10
mostly shows that a crosswalk beats no crosswalk. These baselines give BM25 the same crosswalk
the core uses, as simple query expansion, so what remains of the core's gain is what the rest of
the system adds (offence-id matching inside the index, collision resolution, the date-based code
filter, zones, bridge, authority, LTR):

    BM25 + crosswalk         every section the query names is followed by its counterpart(s) in
                             the other code ("cases under section 103 BNS" -> "... or IPC 302");
                             a bare number ("section 84") gets the counterparts of both readings
    BM25 + crosswalk + date  the same, plus the code filter: statutes restricted to the code in
                             force on the incident date (the date rule, without the rest of the core)

Both run plain BM25 underneath (SearchOptions.baseline()), with code_filter switched on for the
second. Use them like an agent: `evaluate_set(..., agent=CrosswalkBM25(engine))`.
"""

from __future__ import annotations

import copy

# "<code> <number>" after "or": the section tokenizer reads "... 103 BNS section 302 IPC" as BNS 302
_FORM = {"ipc": "IPC {}", "bns": "BNS {}", "crpc": "CrPC {}", "bnss": "BNSS {}", "iea": "Evidence Act section {}",
         "bsa": "BSA {}"}
_READINGS = ("ipc", "bns")            # a bare number is read in both penal codes


def expand(text: str, normalizer) -> str:
    """The query with each named section's crosswalk counterparts appended."""
    from kanoon_bridge.text.tokenize import extract_sections

    extra = []
    for m in extract_sections(text):
        codes = _READINGS if m.code.value == "?" else (m.code.value,)
        for code in codes:
            for eq in normalizer.equivalents(f"{code}:{m.section}"):
                c, _, sec = eq.partition(":")
                form = _FORM.get(c)
                if form and form.format(sec.upper()) not in extra:
                    extra.append(form.format(sec.upper()))
    return text if not extra else f"{text} or " + " or ".join(extra)


class CrosswalkBM25:
    """BM25 with crosswalk query expansion (and optionally the date-based code filter)."""

    def __init__(self, engine, date_filter: bool = False):
        from kanoon_bridge.search import SearchOptions

        self.engine = engine
        self.normalizer = engine.analyzer.text_res.normalizer
        self.options = SearchOptions.baseline()
        self.options.code_filter = date_filter

    def run(self, query, options=None):
        q = copy.deepcopy(query)
        q.text = expand(q.text, self.normalizer)
        opt = copy.deepcopy(self.options)
        if options is not None:
            opt.top_k = options.top_k
            opt.max_query_terms = getattr(options, "max_query_terms", None)
        return self.engine.search(q, opt)


__all__ = ["expand", "CrosswalkBM25"]
