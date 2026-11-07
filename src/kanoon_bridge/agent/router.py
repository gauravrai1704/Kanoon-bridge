"""Query routing for the research agent v2: what kind of question is this, about which code?  [layer 2 — working]

The v1 agent ran the same five sub-queries for every question. On IL-PCSR's pasted judgments
that HURT (E1 MAP 0.414 core -> 0.306 agent with RRF, results/tables/agent_e1_ilpcsr.csv): a
whole judgment is already a rich query, and Boolean / facet variants of it only add noise.
The router decides per question:

    kind      case_text      a pasted judgment or brief (>= ngram.min_query_words words)
              section_lookup names a section and little else ("cases under section 482 BNSS")
              fact_question  everything else: facts, offence names, Hinglish narratives
    family    penal | procedure | evidence: which pair of codes the question is about. Named
              sections decide it. Otherwise `refine_family` decides it AFTER the first round,
              from retrieval: the code families of the core's top statutes (1/rank each) and of
              the sections the top precedents cite, plus a weaker prior from cue words ("stab",
              "cheat" -> penal; "bail", "FIR" -> procedure; "confession", "presumption" ->
              evidence) and from penal offence ids. Cue words alone misroute
              penal offences that mention evidence ("false evidence", "destruction of an
              electronic record").
    plan      which sub-query kinds to run (RulePlanner rules), and whether to decompose the
              question into legal issues (agent/issues.py)

Routing is rule-based and logged in the trace, so every decision can be shown and audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.schema import AnalyzedQuery, Query

PROCEDURE_CUES = re.compile(
    r"\b(bail|anticipatory|arrest\w*|fir|first information|investigat\w*|custody|remand|magistrate|warrant|"
    r"summons?|charge[- ]?sheet|cognizable|non-cognizable|trial procedure|appeal|revision|quash\w*|"
    r"search and seizure|proclaimed|maintenance|zero fir|police report|inquest)\b", re.I)
EVIDENCE_CUES = re.compile(
    r"\b(evidence|confession\w*|presum\w*|admissib\w*|witness\w*|electronic record|certificate|burden of proof|"
    r"dying declaration|expert opinion|hearsay|relevan(?:t|cy) fact|estoppel|accomplice|privilege\w*)\b", re.I)
PENAL_CUES = re.compile(
    r"\b(stab\w*|kill\w*|murder\w*|beat\w*|hit|assault\w*|hurt|injur\w*|attack\w*|steal\w*|stole\w*|theft|"
    r"robb\w*|rap(?:e|ed)|molest\w*|cheat\w*|fraud\w*|threat\w*|kidnap\w*|abduct\w*|dowry|harass\w*|"
    r"forg\w*|bribe\w*|extort\w*|trespass\w*|knife|gun|acid|poison\w*|chaku|maara|maar|chori|dhokha)\b", re.I)
FAMILY_OF_CODE = {"ipc": "penal", "bns": "penal", "crpc": "procedure", "bnss": "procedure",
                  "iea": "evidence", "bsa": "evidence"}


@dataclass
class Route:
    kind: str                                   # case_text | section_lookup | fact_question
    family: str                                 # penal | procedure | evidence
    rules: list[str] = field(default_factory=list)   # RulePlanner rules to run (besides "original")
    decompose: bool = False                     # run issue decomposition (agent/issues.py)
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.kind}, {self.family} law; sub-queries: {', '.join(['original'] + self.rules)}" + \
               ("; decompose into issues" if self.decompose else "") + \
               (f" ({'; '.join(self.reasons)})" if self.reasons else "")


def family_of(query: Query, aq: AnalyzedQuery) -> tuple[str, str]:
    """(family, why). Named sections win, then cue words, then offence / provision ids."""
    codes = [s.split(":")[0] for s in aq.sections if s.split(":")[0] in FAMILY_OF_CODE]
    if codes:
        fam = FAMILY_OF_CODE[codes[0]]
        return fam, f"names {codes[0].upper()} sections"
    text = query.text
    proc, evid = len(PROCEDURE_CUES.findall(text)), len(EVIDENCE_CUES.findall(text))
    if proc or evid:
        fam = "procedure" if proc >= evid else "evidence"
        return fam, f"{fam} words in the question"
    offs = list(aq.offence_ids) + [t for t in aq.expanded_terms if t.startswith("off:")]
    if any(o.startswith("off:bnss") for o in offs):
        return "procedure", "normalised to a CrPC/BNSS provision"
    if any(o.startswith("off:bsa") for o in offs):
        return "evidence", "normalised to an Evidence Act/BSA provision"
    return "penal", "default: an offence question"


def refine_family(rt: Route, query: Query, aq: AnalyzedQuery, top_statute_refs: list[str],
                  cue_weight: float = 0.5, offence_weight: float = 0.5,
                  precedent_codes: list[list[str]] | None = None, precedent_weight: float = 0.5) -> tuple[str, str]:
    """(family, why) from the first round's evidence (see module docstring). `precedent_codes`:
    for each of the top precedents, in rank order, the codes of the sections it cites."""
    codes = [s.split(":")[0] for s in aq.sections if s.split(":")[0] in FAMILY_OF_CODE]
    if codes:
        return FAMILY_OF_CODE[codes[0]], f"names {codes[0].upper()} sections"
    score = {"penal": 0.0, "procedure": 0.0, "evidence": 0.0}
    for r, ref in enumerate(top_statute_refs[:5], start=1):
        fam = FAMILY_OF_CODE.get(ref.split(":")[0])
        if fam:
            score[fam] += 1.0 / r
    score["penal"] += cue_weight * min(2, len(PENAL_CUES.findall(query.text)))
    score["procedure"] += cue_weight * min(2, len(PROCEDURE_CUES.findall(query.text)))
    score["evidence"] += cue_weight * min(2, len(EVIDENCE_CUES.findall(query.text)))
    offs = list(aq.offence_ids) + [t for t in aq.expanded_terms if t.startswith("off:")]
    if any(not o.startswith(("off:bnss", "off:bsa")) for o in offs):
        score["penal"] += offence_weight
    # what the top precedents cite: courts' citations name the area of law better than word overlap
    for r, codes in enumerate(precedent_codes or [], start=1):
        fams = {FAMILY_OF_CODE[c] for c in codes if c in FAMILY_OF_CODE}
        for f in fams:
            score[f] += precedent_weight / r / len(fams)
    fam = max(score, key=score.get)
    return fam, "from the top statutes and cues: " + ", ".join(f"{k} {v:.2f}" for k, v in score.items())


def route(query: Query, aq: AnalyzedQuery, case_words: int = 50) -> Route:
    words = len(query.text.split())
    fam, why = family_of(query, aq)
    content = [t for t in aq.tokens if not t.startswith(("sec:", "off:")) and not t.isdigit()]
    if words >= case_words:
        return Route("case_text", fam, ["cross_code"], decompose=False,
                     reasons=[f"{words} words: a pasted judgment; Boolean / facet variants hurt such queries (E1)", why])
    if aq.sections and len(content) <= 4:
        return Route("section_lookup", fam, ["cross_code", "statutes"], decompose=False,
                     reasons=["names a section", why])
    rules = ["cross_code", "boolean", "facet", "statutes"]
    return Route("fact_question", fam, rules, decompose=True, reasons=[why])


__all__ = ["Route", "route", "family_of", "refine_family", "FAMILY_OF_CODE"]
