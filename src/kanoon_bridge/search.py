"""The single search entry point.  [owner: D — integration contract, working]

Team rule 4: the CLI, the Streamlit app and the evaluator ALL call this, so the demo shows
exactly what the report measures.

    from kanoon_bridge.search import SearchEngine, SearchOptions
    from kanoon_bridge.schema import Query

    engine = SearchEngine.load()                         # loads indexes from data/processed/index
    result = engine.search(Query("mere bhai ko chaku maara", state="delhi",
                                 incident_date="2025-03-01"))
    result.statutes[0].doc_id, result.precedents[0].components

Pipeline (each stage is switched by SearchOptions; eval/ablation.py walks the ladder):

    analyze query            query/analyzer.py         C (+B)
      -> statute scoring     rank/bm25f.py on statutes D   (filtered to the code in force)
      -> statute bridge      rank/statute_bridge.py    D   (expansion + boosts)
      -> precedent lexical   rank/bm25f.py (or vsm.py) D
      -> dense + fusion      rank/dense.py, fusion.py  C   (alpha from rank/qpp.py)
      -> authority           rank/authority.py         D   (g(d) or g(d | state))
      -> top-K               rank/topk.py              D

Index files expected in data/processed/index/ (written by scripts/02 and 03):
    statutes_zone.pkl  precedents_zone.pkl  facets.pkl  authority.json  [tiers.pkl]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.index import store
from kanoon_bridge.schema import AnalyzedQuery, Code, DocType, Query, ScoredDoc, SearchResult
from kanoon_bridge.rank.topk import net_score, to_scored, top_k


@dataclass
class SearchOptions:
    """Switches for each component (names match configs/eval.yaml ablation_ladder)."""

    lexical: str = "bm25f"         # "bm25f" | "tfidf"
    zones: bool = True             # BM25F zone weights (False = plain BM25)
    bridge: bool = True            # statute bridge
    authority: bool = True         # add lambda * g(d)
    dense: bool = False            # dense channel + fusion
    qpp: bool = False              # QPP-gated alpha (needs dense)
    jurisdiction: bool = True      # g(d | state) instead of g(d)
    code_filter: bool = True       # statutes restricted to the code in force on the incident date
    top_k: int = 10
    max_query_terms: int | None = None   # cap long queries (E1 uses 100); None = config bm25.max_query_terms
    candidate_mode: str = "all"    # "all" | "tiers" | "champions" (efficiency experiment)
    # --- hooks used by the research agent (layer 2); all off in a plain search -----------
    restrict_states: list[str] | None = None        # precedents binding in these states only (SC included)
    require_terms: list[list[str]] | None = None    # Boolean CNF over analysed terms: AND of ORs (postings)
    extra_terms: dict[str, float] | None = None     # added query terms -> weight (pseudo-relevance feedback)
    drop_terms: list[str] | None = None             # analysed terms removed from the query

    @classmethod
    def from_dict(cls, d: dict) -> "SearchOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def baseline(cls) -> "SearchOptions":
        """Plain BM25, nothing else: the 'obvious baseline' in the report."""
        return cls(zones=False, bridge=False, authority=False, dense=False, qpp=False,
                   jurisdiction=False, code_filter=False)


@dataclass
class SearchEngine:
    cfg: Config
    analyzer: object                       # query.analyzer.QueryAnalyzer
    statute_index: object                  # index.zones.ZoneIndex over statutes
    precedent_index: object                # index.zones.ZoneIndex over precedents
    facets: object                         # index.facets.FacetIndex over both
    authority: object | None = None        # rank.authority.Authority
    bridge: object | None = None           # rank.statute_bridge.StatuteBridge
    dense: object | None = None            # rank.dense.DenseRetriever
    tiers: object | None = None            # index.tiers.TieredIndex
    _scorers: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, cfg: Config | None = None) -> "SearchEngine":
        """Load the indexes written by scripts/02 and 03 from data/processed/index/."""
        cfg = cfg or load_config()
        authority = store.load("authority", "json") if store.exists("authority", "json") else None
        statute_terms = store.load("statute_terms", "json") if store.exists("statute_terms", "json") else None
        tiers = store.load("tiers") if store.exists("tiers") else None
        return cls.from_components(cfg, store.load("statutes_zone"), store.load("precedents_zone"),
                                   store.load("facets"), statute_terms, authority, tiers)

    @classmethod
    def from_components(cls, cfg: Config, statute_index, precedent_index, facets, statute_terms=None,
                        authority=None, tiers=None) -> "SearchEngine":
        """Assemble an engine from in-memory parts (load() and the end-to-end tests use this).
        `authority`: an Authority or its to_dict() form."""
        from kanoon_bridge.query.analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer.load(cfg, vocabulary=precedent_index.whole.vocabulary())
        engine = cls(cfg=cfg, analyzer=analyzer, statute_index=statute_index,
                     precedent_index=precedent_index, facets=facets, tiers=tiers)
        if authority is not None:
            from kanoon_bridge.rank.authority import Authority

            engine.authority = Authority.from_dict(authority, facets.metas, cfg) if isinstance(authority, dict) else authority
        if statute_terms is not None:
            from kanoon_bridge.rank.statute_bridge import StatuteBridge

            engine.bridge = StatuteBridge(facets=facets, statute_terms=statute_terms,
                                          top_n=cfg.statute_bridge.top_statutes, boost=cfg.statute_bridge.boost,
                                          normalizer=analyzer.text_res.normalizer)
        if cfg.dense.enabled:
            from kanoon_bridge.rank.dense import DenseRetriever

            engine.dense = DenseRetriever.load(cfg)
        return engine

    def _scorer(self, which: str, opt: SearchOptions):
        key = (which, opt.lexical, opt.zones)
        scorer = self._make_scorer(which, opt, key)
        if hasattr(scorer, "max_query_terms"):
            scorer.max_query_terms = opt.max_query_terms or self.cfg.bm25.get("max_query_terms", 300)
        return scorer

    def _make_scorer(self, which: str, opt: SearchOptions, key):
        if key not in self._scorers:
            zidx = self.statute_index if which == "statute" else self.precedent_index
            if opt.lexical == "tfidf":
                from kanoon_bridge.rank.vsm import TfidfScorer

                self._scorers[key] = TfidfScorer(zidx.whole).prepare()
            else:
                from kanoon_bridge.rank.bm25f import BM25F

                self._scorers[key] = BM25F.from_config(zidx, self.cfg, use_zones=opt.zones)
        return self._scorers[key]

    # ------------------------------------------------------------------ search
    def search(self, query: Query, options: SearchOptions | None = None) -> SearchResult:
        opt = options or SearchOptions()
        t: dict[str, float] = {}
        clock = time.perf_counter

        t0 = clock()
        aq: AnalyzedQuery = self.analyzer.analyze(query)
        terms = aq.weighted_terms()
        for term in opt.drop_terms or []:
            terms.pop(term, None)
        for term, w in (opt.extra_terms or {}).items():
            terms[term] = max(terms.get(term, 0.0), w)
        if opt.drop_terms or opt.extra_terms:
            aq.trace.append(("feedback", f"dropped {opt.drop_terms or []}, added {sorted(opt.extra_terms or {})}"))
        t["analyze"] = clock() - t0

        # --- statutes -------------------------------------------------------------
        t0 = clock()
        statute_cands = None
        if opt.code_filter and aq.code_in_force in (Code.IPC, Code.BNS):
            # drop only the superseded penal code; other Acts (CrPC, Constitution, ...) stay
            superseded = (Code.BNS if aq.code_in_force == Code.IPC else Code.IPC).value
            all_statutes = self.facets.filter(doc_type=DocType.STATUTE.value)
            statute_cands = all_statutes - self.facets.filter(doc_type=DocType.STATUTE.value, code=superseded)
            aq.trace.append(("statute_filter", f"excluded {superseded.upper()} sections (code in force: {aq.code_in_force.value.upper()})"))
        statute_scores = self._scorer("statute", opt).score(terms, statute_cands)
        statutes = to_scored(top_k(statute_scores, opt.top_k), DocType.STATUTE,
                             {d: {"lexical": s} for d, s in statute_scores.items()})
        t["statutes"] = clock() - t0

        # --- statute bridge -------------------------------------------------------
        boosts: dict[str, float] = {}
        if opt.bridge and self.bridge is not None:
            t0 = clock()
            out = self.bridge.run(statutes)
            for term, w in out.expansion.items():
                terms[term] = max(terms.get(term, 0.0), w)
            boosts = out.boosts
            aq.trace.append(("bridge_statutes", ", ".join(out.statutes_used)))
            t["bridge"] = clock() - t0

        # --- precedents: lexical, dense, fusion -----------------------------------
        t0 = clock()
        prec_cands = self._precedent_filter(query)
        if opt.restrict_states:
            binding = self.facets.filter(doc_type=DocType.PRECEDENT.value, states=list(opt.restrict_states))
            prec_cands = binding if prec_cands is None else prec_cands & binding
            aq.trace.append(("facet", f"precedents binding in {', '.join(opt.restrict_states)}: {len(binding)}"))
        if opt.require_terms:
            sel = self.boolean_candidates(opt.require_terms)
            prec_cands = sel if prec_cands is None else prec_cands & sel
            aq.trace.append(("boolean", " AND ".join("(" + " OR ".join(c) + ")" for c in opt.require_terms)
                             + f" -> {len(sel)} docs"))
        if opt.candidate_mode != "all" and self.tiers is not None:
            sel = self.tiers.candidates(list(terms), self.cfg.tiers.min_results_before_tier2,
                                        use_champions=opt.candidate_mode == "champions",
                                        index=self.precedent_index.whole)
            prec_cands = sel if prec_cands is None else prec_cands & sel
        scorer = self._scorer("precedent", opt)
        lexical = scorer.score(terms, prec_cands)
        components = {d: {"lexical": s} for d, s in lexical.items()}
        for d, zones in getattr(scorer, "last_breakdown", {}).items():
            if d in components:
                components[d].update({f"zone_{z}": v for z, v in zones.items()})
        # relevance on a 0-1 scale (divide by the best score) so bridge boosts and lambda*g(d)
        # are comparable across queries; ranking by relevance alone is unchanged
        top_lex = max(lexical.values(), default=0.0) or 1.0
        relevance = {d: s / top_lex for d, s in lexical.items()}

        if opt.dense and self.dense is not None:
            from kanoon_bridge.rank.fusion import fuse

            dense = self.dense.score(aq.transliterated or query.text, prec_cands)
            alpha = self.cfg.fusion.alpha_default
            if opt.qpp:
                from kanoon_bridge.rank import qpp

                idx = self.precedent_index
                f = qpp.pre_retrieval(list(terms), {x: idx.whole.idf(x) for x in terms},
                                      {x: idx.df(x) for x in terms}, idx.n_docs)
                f = qpp.post_retrieval(f, sorted(lexical.values(), reverse=True))
                alpha = qpp.alpha_from_qpp(f, alpha)
            relevance = fuse(lexical, dense, alpha)
            for d, s in dense.items():
                components.setdefault(d, {})["dense"] = s
            aq.trace.append(("alpha", f"{alpha:.2f}"))

        for d, b in boosts.items():
            if d in relevance:
                relevance[d] += b
                components.setdefault(d, {})["bridge"] = b

        # --- authority -----------------------------------------------------------
        final = relevance
        if opt.authority and self.authority is not None:
            lam = self.cfg.authority["lambda"]
            state = query.state if opt.jurisdiction else None
            final = {}
            for d, rel in relevance.items():
                g = self.authority.score(d, state=state, jurisdiction=opt.jurisdiction)
                final[d] = net_score(rel, g, lam)
                components.setdefault(d, {})["authority"] = g
        precedents = to_scored(top_k(final, opt.top_k), DocType.PRECEDENT, components)
        t["precedents"] = clock() - t0

        return SearchResult(query=aq, statutes=statutes, precedents=precedents,
                            timings_ms={k: v * 1000 for k, v in t.items()})

    def boolean_candidates(self, cnf: list[list[str]]) -> set[str]:
        """AND of OR-clauses over analysed terms, by postings union/intersection (rarest clause
        first, so the running intersection stays small)."""
        whole = self.precedent_index.whole
        clauses = [set().union(*(whole.docs_with(t) for t in clause)) for clause in cnf if clause]
        if not clauses:
            return set()
        clauses.sort(key=len)
        out = clauses[0]
        for c in clauses[1:]:
            out = out & c
            if not out:
                break
        return out

    def _precedent_filter(self, query: Query) -> set[str] | None:
        """Hard filters from the query (court:, type:, explicit date range). State is NOT a hard
        filter: it changes authority instead, so persuasive precedent still shows up."""
        f = query.filters
        if not any(k in f for k in ("court", "date_from", "date_to")):
            return self.facets.filter(doc_type=DocType.PRECEDENT.value)
        return self.facets.filter(doc_type=DocType.PRECEDENT.value, court=f.get("court"),
                                  date_from=f.get("date_from"), date_to=f.get("date_to"))


_engine: SearchEngine | None = None


def search(query: Query | str, options: SearchOptions | None = None, **query_fields) -> SearchResult:
    """Module-level convenience: search("chaku maara", state="delhi")."""
    global _engine
    if _engine is None:
        _engine = SearchEngine.load()
    if isinstance(query, str):
        query = Query(text=query, **query_fields)
    return _engine.search(query, options)
