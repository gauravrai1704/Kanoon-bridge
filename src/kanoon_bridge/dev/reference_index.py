"""TEMPORARY stand-in for Sharanya's index functions, so the rest of the team can run the
pipeline end to end before index/positional.py, index/zones.py and index/facets.py are done.

    KB_DEV_SHIM=1 make sample          # or any script / test run

  * OFF unless the environment variable KB_DEV_SHIM=1 is set (checked in index/__init__.py).
  * Patches a method ONLY while the real one still raises NotImplementedError, so it switches
    itself off method by method as Sharanya's code lands. It never overrides finished code.
  * Deliberately simple (no smallest-set-first intersection, linear date scan, naive phrase
    check). The graded implementations are Sharanya's [owner: Sharanya Gupta]; delete this
    file once hers pass tests/test_index.py.
"""

from __future__ import annotations

import sys
from collections import defaultdict

_installed: list[str] = []
_originals: list[tuple[object, str, object]] = []      # (class, attribute, original) for uninstall()


def _patch(cls, name: str, new) -> None:
    _originals.append((cls, name, cls.__dict__[name]))
    setattr(cls, name, new)
    _installed.append(f"{cls.__name__}.{name}")


def _still_todo(fn, *args) -> bool:
    try:
        fn(*args)
    except NotImplementedError:
        return True
    except Exception:
        return False
    return False


def install(quiet: bool = False) -> list[str]:
    from kanoon_bridge.index import facets, positional, zones
    from kanoon_bridge.index.facets import DocMeta

    if _installed:
        return _installed

    # -------------------------------------------------------------- PositionalIndex
    def add(self, doc_id, tokens):
        if doc_id in self.doc_len:
            raise ValueError(f"document added twice: {doc_id}")
        for i, t in enumerate(tokens):
            self.postings[t].setdefault(doc_id, []).append(i)
        self.doc_len[doc_id] = len(tokens)

    def phrase(self, terms):
        if not terms:
            return set()
        docs = set(self.postings.get(terms[0], {}))
        for t in terms[1:]:
            docs &= set(self.postings.get(t, {}))
        out = set()
        for d in docs:
            starts = set(self.postings[terms[0]][d])
            for off, t in enumerate(terms[1:], start=1):
                starts &= {p - off for p in self.postings[t][d]}
            if starts:
                out.add(d)
        return out

    if _still_todo(positional.PositionalIndex().add, "x", ["a"]):
        _patch(positional.PositionalIndex, "add", add)
    if _still_todo(positional.PositionalIndex().phrase, ["a"]):
        _patch(positional.PositionalIndex, "phrase", phrase)

    # -------------------------------------------------------------- ZoneIndex
    @classmethod
    def zbuild(cls, docs, analyze):
        z = cls()
        for d in docs:
            per: dict[str, list[str]] = defaultdict(list)
            allt: list[str] = []
            for p in d.paragraphs:
                toks = analyze(p.text, d)
                per[p.zone if p.zone in z.zones else "other"].extend(toks)
                allt.extend(toks)
            for zone, toks in per.items():
                z.zones[zone].add(d.doc_id, toks)
            z.whole.add(d.doc_id, allt)
            z.doc_ids.append(d.doc_id)
        return z

    if _still_todo(zones.ZoneIndex.build, [], lambda t, d: []):
        _patch(zones.ZoneIndex, "build", zbuild)

    # -------------------------------------------------------------- FacetIndex
    @classmethod
    def fbuild(cls, docs):
        f = cls()
        for d in docs:
            m = DocMeta(d.doc_type.value, d.court.value, d.court_name, list(d.states), d.decision_date,
                        d.code.value, list(d.statutes_cited))
            f.metas[d.doc_id] = m
            f.by_type.setdefault(m.doc_type, set()).add(d.doc_id)
            f.by_court.setdefault(m.court, set()).add(d.doc_id)
            f.by_code.setdefault(m.code, set()).add(d.doc_id)
            for s in m.states:
                f.by_state.setdefault(s, set()).add(d.doc_id)
            for s in m.statutes_cited:
                f.by_section.setdefault(s, set()).add(d.doc_id)
            if m.decision_date:
                f.dates.append((m.decision_date, d.doc_id))
        f.dates.sort()
        return f

    def ffilter(self, doc_type=None, court=None, states=None, date_from=None, date_to=None, code=None, cites_any=None):
        sets = []
        if doc_type:
            sets.append(self.by_type.get(doc_type, set()))
        if court:
            sets.append(self.by_court.get(court, set()))
        if code:
            sets.append(self.by_code.get(code, set()))
        if states:
            sets.append(set().union(*(self.by_state.get(s, set()) for s in states)) | self.by_state.get("*", set()))
        if cites_any:
            sets.append(set().union(*(self.by_section.get(s, set()) for s in cites_any)))
        if date_from or date_to:
            from datetime import date as _d

            lo = _d.fromisoformat(str(date_from)) if date_from else None
            hi = _d.fromisoformat(str(date_to)) if date_to else None
            sets.append({doc for dt, doc in self.dates if (lo is None or dt >= lo) and (hi is None or dt <= hi)})
        if not sets:
            return set(self.metas)
        return set.intersection(*sets)

    if _still_todo(facets.FacetIndex.build, []):
        _patch(facets.FacetIndex, "build", fbuild)
    if _still_todo(facets.FacetIndex().filter):
        _patch(facets.FacetIndex, "filter", ffilter)

    if _installed and not quiet:
        print(f"[KB_DEV_SHIM] temporary reference index code in use for: {', '.join(_installed)}", file=sys.stderr)
    return _installed


def uninstall() -> None:
    """Restore the real methods (tests use install()/uninstall() around one test only)."""
    while _originals:
        cls, name, original = _originals.pop()
        setattr(cls, name, original)
    _installed.clear()
