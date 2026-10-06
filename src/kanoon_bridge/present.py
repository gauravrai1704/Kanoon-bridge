"""Result presentation: query-biased snippets, highlighting, "why this result", statute versions,
facet counts.  [owner: Gaurav — working]

Used by the CLI, the Streamlit app and the RAG chunk labels; the ranking itself never depends
on anything here.

    snippet      query-biased summary (Tombros & Sanderson, SIGIR 1998, "Advantages of query
                 biased summaries in information retrieval"): score every sentence by the
                 query terms it contains — (distinct matched terms)^2 / query terms, weighted by
                 term weight, with a small bonus for the court's reasoning (ratio / decision
                 zones, cf. SAILER, Li et al. SIGIR 2023, on the value of a judgment's
                 structure) — and show the best one or two sentences in document order. Snippet
                 length and term highlighting follow Turpin et al., SIGIR 2007 ("Fast generation
                 of result snippets in web search"): ~2 sentences, matched terms marked.
    explain      human-readable reasons from the score breakdown and metadata: offence matched
                 across codes, zone of the match, binding vs persuasive for the user's state,
                 old-code judgment still relevant through the crosswalk.
    version_note "BNS 103 <- IPC 302 (same offence)" / "new in BNS (no IPC equivalent)".
    facet_counts court level / decade / code counts over a result list (faceted navigation).
"""

from __future__ import annotations

import re
from collections import Counter

from kanoon_bridge.schema import Code

HIGHLIGHT = ("**", "**")              # markdown bold: renders in Streamlit, readable in a terminal
_ZONE_BONUS = {"ratio": 0.25, "decision": 0.15, "statute": 0.1}
_ZONE_NAMES = {"ratio": "the court's reasoning", "decision": "the decision", "facts": "the facts",
               "arguments": "the arguments", "statute": "the statute text", "other": "the text"}


# --------------------------------------------------------------------------- snippets


def _query_weights(aq) -> dict[str, float]:
    w = dict(aq.weighted_terms()) if hasattr(aq, "weighted_terms") else {}
    return {t: v for t, v in w.items() if v > 0}


def _sentence_terms(sentence: str, res) -> set[str]:
    from kanoon_bridge.text.pipeline import analyze_text

    return set(analyze_text(sentence, res))


def highlight(sentence: str, qterms: dict[str, float], res, marks: tuple[str, str] = HIGHLIGHT) -> str:
    """Wrap query words (matched after the same analysis as the index) and section mentions
    whose section/offence token is in the query."""
    from kanoon_bridge.text.pipeline import analyze_text
    from kanoon_bridge.text.tokenize import extract_sections

    spans = []
    for m in extract_sections(sentence):
        toks = set(analyze_text(sentence[m.start:m.end], res))
        if toks & set(qterms):
            spans.append((m.start, m.end))
    taken = [False] * (len(sentence) + 1)
    for a, b in spans:
        for i in range(a, b):
            taken[i] = True
    for m in re.finditer(r"[A-Za-z][A-Za-z']+", sentence):
        if taken[m.start()]:
            continue
        toks = analyze_text(m.group(0), res)
        if toks and toks[0] in qterms:
            spans.append((m.start(), m.end()))
    out, last = [], 0
    for a, b in sorted(set(spans)):
        if a < last:
            continue
        out += [sentence[last:a], marks[0], sentence[a:b], marks[1]]
        last = b
    out.append(sentence[last:])
    return "".join(out)


def snippet(doc, aq, res, max_sentences: int = 2, max_chars: int = 320, max_scan: int = 400,
            marks: tuple[str, str] = HIGHLIGHT) -> str:
    """Query-biased snippet with highlighting (see module docstring)."""
    from kanoon_bridge.rag._util import split_sentences

    qterms = _query_weights(aq)
    total = sum(qterms.values()) or 1.0
    cands = []
    n = 0
    for para in doc.paragraphs:
        if para.para_id < 0:                                  # statute heading line
            continue
        for s in split_sentences(para.text):
            n += 1
            if n > max_scan:
                break
            if len(s) < 25:
                continue
            matched = _sentence_terms(s, res) & set(qterms)
            if not matched:
                continue
            weight = sum(qterms[t] for t in matched)
            score = (len(matched) ** 2 / max(1, len(qterms))) * (weight / total) + _ZONE_BONUS.get(para.zone, 0.0)
            cands.append((score, n, s))
    if not cands:                                             # no match: the opening lines
        first = next((p.text for p in doc.paragraphs if p.para_id >= 0 and p.text.strip()), doc.text)
        return _cut(" ".join(first.split()), max_chars)
    best = sorted(sorted(cands, reverse=True)[:max_sentences], key=lambda c: c[1])
    per = max_chars // len(best)
    return " … ".join(highlight(_cut(s, per), qterms, res, marks) for _, _, s in best)


def _cut(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


# --------------------------------------------------------------------------- statute versions


def statute_ref(engine, doc_id: str) -> str:
    terms = getattr(getattr(engine, "bridge", None), "statute_terms", {}) or {}
    sec = next((t for t in terms.get(doc_id, []) if t.startswith("sec:")), "")
    return sec[4:] if sec else ""


def _show(ref: str) -> str:
    code, _, num = ref.partition(":")
    return f"{code.upper()} {num}"


def version_note(ref: str, normalizer) -> str:
    """'bns:103' -> 'BNS 103 <- IPC 302 (same offence)'; 'ipc:302' -> 'IPC 302 -> BNS 103 from 1 Jul 2024'."""
    if not ref or normalizer is None or ref.split(":")[0] not in ("ipc", "bns"):
        return ""
    code, _, num = ref.partition(":")
    base = re.match(r"\d+[a-z]*", num)
    base = base.group(0) if base else num
    rows = [r for r in normalizer.crosswalk if (r.bns_section if code == "bns" else r.ipc_section) == base]
    eq = normalizer.equivalents(ref)
    if code == "bns":
        if not eq and (not rows or all(not r.ipc_section for r in rows)):
            return f"{_show(ref)}: new in BNS (no IPC equivalent)"
        rel = {r.relation for r in rows if r.ipc_section} - {""}
        how = {"same": "same offence", "merge": "merges several IPC sections", "split": "part of a split IPC section"}
        detail = ", ".join(how.get(r, r) for r in sorted(rel)) or "same offence"
        return f"{_show(ref)} <- " + ", ".join(_show(e) for e in eq) + f" ({detail})"
    if not eq:
        return f"{_show(ref)}: no BNS equivalent (not carried into BNS)"
    return f"{_show(ref)} -> " + ", ".join(_show(e) for e in eq) + " from 1 Jul 2024"


# --------------------------------------------------------------------------- why this result


def explain(hit, aq, engine, target: str = "precedent") -> list[str]:
    """Plain-language reasons for one result."""
    from kanoon_bridge.rank.authority import binding_status

    reasons: list[str] = []
    comp = hit.components or {}
    res = engine.analyzer.text_res
    norm = res.normalizer
    if target == "statute":
        ref = statute_ref(engine, hit.doc_id)
        note = version_note(ref, norm)
        if note:
            reasons.append(note)
        if ref and aq.code_in_force in (Code.IPC, Code.BNS) and ref.startswith(aq.code_in_force.value + ":"):
            reasons.append(f"in force on the incident date ({aq.code_in_force.value.upper()})")
        return reasons

    meta = engine.facets.metas.get(hit.doc_id)
    zones = {k[5:]: v for k, v in comp.items() if k.startswith("zone_")}
    if zones:
        z = max(zones, key=zones.get)
        reasons.append(f"strongest match in {_ZONE_NAMES.get(z, z)}")
    q_offences = set(aq.offence_ids) | {t for t in aq.expanded_terms if t.startswith("off:")}
    shared: list[tuple[str, str]] = []
    if meta is not None and norm is not None and meta.statutes_cited:
        for ref in meta.statutes_cited:
            for off in norm.offences_for(ref):
                if off in q_offences:
                    shared.append((ref, off))
        if shared:
            ref, off = shared[0]
            mine = [s for s in aq.sections if norm.offences_for(s) and off in norm.offences_for(s)]
            same = f" = {_show(mine[0])}" if mine and mine[0].split(':')[0] != ref.split(':')[0] else ""
            reasons.append(f"cites {_show(ref)}{same} ({norm.label(off)})")
    if comp.get("bridge"):
        reasons.append("cites statutes that ranked high for this query")
    if comp.get("dense"):
        reasons.append(f"semantic similarity {comp['dense']:.2f}")
    if meta is not None:
        status = binding_status(meta, aq.query.state)
        court = meta.court_name or meta.court.replace("_", " ").title()
        if status == "binding":
            reasons.append(f"binding{' in ' + aq.query.state.title() if aq.query.state else ''} ({court})")
        elif status == "persuasive":
            reasons.append(f"persuasive only ({court}; not binding in {aq.query.state.title()})")
        if meta.decision_date and aq.code_in_force == Code.BNS and meta.decision_date.year < 2024 and shared:
            reasons.append(f"decided {meta.decision_date.year} under the IPC; still relevant through the crosswalk")
    return reasons


# --------------------------------------------------------------------------- facets


def facet_counts(hits, engine) -> dict[str, Counter]:
    """Court level, decade and code counts over a result list (for a sidebar / --debug)."""
    out = {"court": Counter(), "decade": Counter(), "code": Counter()}
    for h in hits:
        m = engine.facets.metas.get(h.doc_id)
        if m is None:
            continue
        out["court"][m.court_name or m.court] += 1
        if m.decision_date:
            out["decade"][f"{m.decision_date.year // 10 * 10}s"] += 1
        out["code"][m.code] += 1
    return out


def title_of(docs, doc_id: str) -> str:
    d = docs.get(doc_id) if docs is not None else None
    return (d.title if d is not None and d.title else doc_id)
