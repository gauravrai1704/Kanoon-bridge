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
    version_norm: bool = True      # match sections through offence ids shared by IPC and BNS (off = text as written)
    ngram: bool = False            # word-trigram BM25 channel for long (case-as-query) queries, rank/ngram.py
    ngram_only: bool = False       # rank by word-trigram BM25 alone, any query length (IL-PCSR's lexical baseline)
    top_k: int = 10
    max_query_terms: int | None = None   # cap long queries (E1 uses 100); None = config bm25.max_query_terms
    candidate_mode: str = "all"    # "all" | "tiers" | "champions" (efficiency experiment)
    # --- hooks used by the research agent (layer 2); all off in a plain search -----------
    restrict_states: list[str] | None = None        # precedents binding in these states only (SC included)
    require_terms: list[list[str]] | None = None    # Boolean CNF over analysed terms: AND of ORs (postings)
    extra_terms: dict[str, float] | None = None     # added query terms -> weight (pseudo-relevance feedback)
    drop_terms: list[str] | None = None             # analysed terms removed from the query
    # --- result-list options -----------------------------------------------------------
    ltr: bool = False                  # re-rank with the learned linear model (rank/ltr.py), if trained
    ltr_short: bool = False            # also re-rank TYPED questions with ltr_short.json (off by default:
                                       # it learns citation popularity, +31% MAP on E1q but -47% nDCG on E6)
    collapse_duplicates: bool = False  # show one judgment per near-duplicate group (index/dedup.py)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def full(cls) -> "SearchOptions":
        """Everything on, including the learned re-ranker when one has been trained."""
        return cls(ltr=True, ngram=True)

    @classmethod
    def interactive(cls) -> "SearchOptions":
        """What the CLI and app use: full system, near-duplicates collapsed."""
        return cls(ltr=True, ngram=True, collapse_duplicates=True)

    @classmethod
    def baseline(cls) -> "SearchOptions":
        """Plain BM25, nothing else: the 'obvious baseline' in the report."""
        return cls(zones=False, bridge=False, authority=False, dense=False, qpp=False,
                   jurisdiction=False, code_filter=False, version_norm=False)

    @classmethod
    def ilpcsr_baseline(cls) -> "SearchOptions":
        """IL-PCSR's strongest lexical baseline (Paul et al. 2025, Table 3): BM25 over word 3-grams
        of the full query text, nothing else."""
        b = cls.baseline()
        b.ngram, b.ngram_only = True, True
        return b


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
    ltr: object | None = None              # rank.ltr.LinearRanker for case queries (scripts/06_train_ltr.py)
    ltr_short: object | None = None        # rank.ltr.LinearRanker for typed questions (06_train_ltr.py --short)
    near_dups: dict | None = None          # doc_id -> group representative (index/dedup.py)
    ngram_statutes: object | None = None   # rank.ngram.NgramIndex over statutes
    ngram_precedents: object | None = None # rank.ngram.NgramIndex over precedents
    _scorers: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, cfg: Config | None = None) -> "SearchEngine":
        """Load the indexes written by scripts/02 and 03 from data/processed/index/."""
        cfg = cfg or load_config()
        authority = store.load("authority", "json") if store.exists("authority", "json") else None
        statute_terms = store.load("statute_terms", "json") if store.exists("statute_terms", "json") else None
        tiers = store.load("tiers") if store.exists("tiers") else None
        surface = store.load("surface_forms", "json") if store.exists("surface_forms", "json") else None
        engine = cls.from_components(cfg, store.load("statutes_zone"), store.load("precedents_zone"),
                                     store.load("facets"), statute_terms, authority, tiers, surface)
        if store.exists("ltr", "json"):
            from kanoon_bridge.rank.ltr import LinearRanker

            try:
                engine.ltr = LinearRanker.from_dict(store.load("ltr", "json"))
            except ValueError as err:
                import warnings

                warnings.warn(f"learning-to-rank model ignored: {err}")
        if store.exists("ltr_short", "json"):
            from kanoon_bridge.rank.ltr import LinearRanker

            try:
                engine.ltr_short = LinearRanker.from_dict(store.load("ltr_short", "json"))
            except ValueError as err:
                import warnings

                warnings.warn(f"short-query learning-to-rank model ignored: {err}")
        if store.exists("near_duplicates", "json"):
            engine.near_dups = store.load("near_duplicates", "json")
        for which in ("statutes", "precedents"):
            if store.exists(f"ngram_{which}"):
                setattr(engine, f"ngram_{which}", store.load(f"ngram_{which}"))
        return engine

    def _is_case_query(self, query: Query) -> bool:
        """A pasted judgment or brief (IL-PCSR's case-as-query task), not a typed question."""
        return len(query.text.split()) >= self.cfg.ngram.min_query_words

    def _use_ngram(self, query: Query, opt: "SearchOptions") -> bool:
        return opt.ngram and (opt.ngram_only or self._is_case_query(query))

    @staticmethod
    def _mix(base: dict[str, float], ngram: dict[str, float], beta: float) -> dict[str, float]:
        """(1 - beta) * base/max + beta * ngram/max over the union of both candidate lists."""
        tb = max(base.values(), default=0.0) or 1.0
        tn = max(ngram.values(), default=0.0) or 1.0
        return {d: (1 - beta) * base.get(d, 0.0) / tb + beta * ngram.get(d, 0.0) / tn for d in set(base) | set(ngram)}

    @classmethod
    def from_components(cls, cfg: Config, statute_index, precedent_index, facets, statute_terms=None,
                        authority=None, tiers=None, surface: dict | None = None) -> "SearchEngine":
        """Assemble an engine from in-memory parts (load() and the end-to-end tests use this).
        `authority`: an Authority or its to_dict() form."""
        from kanoon_bridge.query.analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer.load(cfg, vocabulary=precedent_index.whole.vocabulary())
        if cfg.text.get("spelling", True):
            from kanoon_bridge.text.spell import Speller

            df: dict[str, int] = {}
            for idx in (statute_index.whole, precedent_index.whole):
                for term, plist in idx.postings.items():
                    df[term] = df.get(term, 0) + len(plist)
            analyzer.speller = Speller.build(df, surface or {}, min_df=cfg.text.get("spelling_min_df", 2))
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

            try:
                engine.dense = DenseRetriever.load(cfg)
            except FileNotFoundError as err:
                import warnings

                warnings.warn(f"dense channel off: {err}")
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

                scorer = BM25F.from_config(zidx, self.cfg, use_zones=opt.zones)
                if which == "statute":           # statutes are short and uniform: their own b
                    scorer.b = self.cfg.bm25.get("b_statutes", scorer.b)
                self._scorers[key] = scorer
        return self._scorers[key]

    # ------------------------------------------------------------------ search
    def search(self, query: Query, options: SearchOptions | None = None) -> SearchResult:
        opt = options or SearchOptions()
        t: dict[str, float] = {}
        clock = time.perf_counter

        t0 = clock()
        aq: AnalyzedQuery = self.analyzer.analyze(query)
        terms = aq.weighted_terms()
        if not opt.version_norm:
            # the "text as written" baseline: no offence ids, no cross-code section expansions
            written = set(aq.tokens) - {t for t in aq.tokens if t.startswith("off:")}
            terms = {t: w for t, w in terms.items()
                     if not t.startswith("off:") and not (t.startswith("sec:") and t not in written)}
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
            # drop the superseded codes (IPC/CrPC/IEA after 1 July 2024, BNS/BNSS/BSA before); other Acts stay
            superseded = ["bns", "bnss", "bsa"] if aq.code_in_force == Code.IPC else ["ipc", "crpc", "iea"]
            all_statutes = self.facets.filter(doc_type=DocType.STATUTE.value)
            for code in superseded:
                all_statutes = all_statutes - self.facets.filter(doc_type=DocType.STATUTE.value, code=code)
            statute_cands = all_statutes
            aq.trace.append(("statute_filter", f"excluded {'/'.join(c.upper() for c in superseded)} sections "
                                               f"(in force: {aq.code_in_force.value.upper()} family)"))
        if query.filters.get("code"):                      # explicit code:bns / code:ipc filter
            wanted = self.facets.filter(doc_type=DocType.STATUTE.value, code=query.filters["code"])
            statute_cands = wanted if statute_cands is None else statute_cands & wanted
        statute_scores = self._scorer("statute", opt).score(terms, statute_cands)
        st_components = {d: {"lexical": s} for d, s in statute_scores.items()}
        use_ngram = self._use_ngram(query, opt)
        if use_ngram and self.ngram_statutes is not None:
            ng = self.ngram_statutes.score(query.text, statute_cands, top=300)
            for d, v in ng.items():
                st_components.setdefault(d, {})["ngram"] = v
            statute_scores = self._mix(statute_scores, ng, 1.0 if opt.ngram_only else self.cfg.ngram.beta_statutes)
            aq.trace.append(("ngram_statutes", f"{len(ng)} statutes share word {self.ngram_statutes.n}-grams with the query"))
        statutes = to_scored(top_k(statute_scores, opt.top_k), DocType.STATUTE, st_components)
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
        if aq.boolean is not None:
            from kanoon_bridge.query.boolean import evaluate
            from kanoon_bridge.text.pipeline import analyze_text

            res = self.analyzer.text_res
            fixes = {c.source: c.term for c in aq.corrections if c.applied}
            speller = getattr(self.analyzer, "speller", None)

            def analyze(t: str) -> list[str]:
                return [fixes.get(x, x) for x in analyze_text(t, res, date=query.incident_date)]

            sel = evaluate(aq.boolean, self.precedent_index.whole, analyze,
                           expand=(lambda p: aq.wildcards.get(p.lower()) or speller.expand_wildcard(p.lower()))
                           if speller is not None else None)
            prec_cands = sel if prec_cands is None else prec_cands & sel
            aq.trace.append(("boolean_match", f"{len(sel)} precedents match the Boolean query"))
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
        case_query = self._is_case_query(query) or opt.ngram_only
        dense: dict[str, float] = {}
        if opt.dense and self.dense is not None:
            dense = self.dense.score(aq.transliterated or query.text, prec_cands)
            for d, v in dense.items():
                components.setdefault(d, {})["dense"] = v
        conf = None
        if opt.qpp:
            from kanoon_bridge.rank import qpp

            idx = self.precedent_index
            f = qpp.pre_retrieval(list(terms), {x: idx.whole.idf(x) for x in terms},
                                  {x: idx.df(x) for x in terms}, idx.n_docs)
            f = qpp.post_retrieval(f, sorted(lexical.values(), reverse=True))
            conf = qpp.confidence(f)

        if case_query:
            # a pasted judgment: trigram phrasing dominates (beta tuned on val); dense joins by QPP
            if use_ngram and self.ngram_precedents is not None:
                ng = self.ngram_precedents.score(query.text, prec_cands, top=300)
                for d, v in ng.items():
                    components.setdefault(d, {})["ngram"] = v
                relevance = self._mix(relevance, ng, 1.0 if opt.ngram_only else self.cfg.ngram.beta)
                aq.trace.append(("ngram", f"{len(ng)} precedents share word {self.ngram_precedents.n}-grams; "
                                          f"beta {self.cfg.ngram.beta}"))
            if dense:
                from kanoon_bridge.rank.fusion import fuse

                alpha = self.cfg.fusion.alpha_default if conf is None else 0.4 + 0.5 * conf
                relevance = fuse(relevance, dense, alpha)
                aq.trace.append(("alpha", f"{alpha:.2f}"))
        else:
            # a typed question: BM25F + trigram + dense, weights gated by QPP confidence in the
            # lexical list (confident -> lean on BM25F; weak -> let dense carry more)
            fs = self.cfg.get("fusion_short", {}) or {}
            ng = {}
            # a query that cites a section ("cases under section 103 BNS") is matched through its
            # offence ids; its word trigrams ("cases under section") carry no topic, so no trigram channel
            if (opt.ngram and self.ngram_precedents is not None and fs.get("trigram", 0) > 0
                    and len(query.text.split()) >= 3 and not aq.sections):
                ng = self.ngram_precedents.score(query.text, prec_cands, top=300)
                for d, v in ng.items():
                    components.setdefault(d, {})["ngram"] = v
            w_dense = fs.get("dense_max", 0.5) * (1 - (0.5 if conf is None else conf)) if dense else 0.0
            w_ng = fs.get("trigram", 0.0) if ng else 0.0
            w_lex = max(fs.get("min_lexical", 0.3), 1.0 - w_dense - w_ng)
            total = w_lex + w_dense + w_ng
            if ng or dense:
                tn = max(ng.values(), default=0.0) or 1.0
                td = max(dense.values(), default=0.0) or 1.0
                lo_d = min(dense.values(), default=0.0)
                relevance = {d: (w_lex * relevance.get(d, 0.0) + w_ng * ng.get(d, 0.0) / tn
                                 + w_dense * ((dense.get(d, lo_d) - lo_d) / ((td - lo_d) or 1.0) if dense else 0.0)) / total
                             for d in set(relevance) | set(ng) | set(dense)}
                aq.trace.append(("fusion", f"lexical {w_lex / total:.2f} + trigram {w_ng / total:.2f} + dense "
                                           f"{w_dense / total:.2f}" + (f" (QPP confidence {conf:.2f})" if conf is not None else "")))

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
        # --- learning-to-rank re-ranking of the head (rank/ltr.py) -------------------
        # two learned models: one trained on whole judgments as queries (ltr.json), one on typed-
        # question proxies (ltr_short.json); each re-ranks only queries of its own kind
        # the short model learned from facts excerpts with no section citation; a query citing a
        # section is outside what it saw, so it keeps the hand-set combination
        ranker = self.ltr if case_query else (None if aq.sections or not opt.ltr_short else self.ltr_short)
        if opt.ltr and ranker is not None and final:
            head = to_scored(top_k(final, max(ranker.depth, opt.top_k)), DocType.PRECEDENT, components)
            learned = ranker.score(head, aq, self)
            final = {h.doc_id: float(v) for h, v in zip(head, learned)}
            for d, v in final.items():
                components.setdefault(d, {})["ltr"] = v
            aq.trace.append(("ltr", f"re-ranked the top {len(head)} with the {'case' if case_query else 'short-query'} model"))

        # --- near-duplicate collapse (index/dedup.py) ----------------------------------
        ranked = top_k(final, opt.top_k)
        if opt.collapse_duplicates and self.near_dups:
            kept, seen, hidden = [], set(), 0
            for d, sc in top_k(final, opt.top_k * 3):
                group = self.near_dups.get(d, d)
                if group in seen:
                    hidden += 1
                    continue
                seen.add(group)
                kept.append((d, sc))
                if len(kept) == opt.top_k:
                    break
            ranked = kept
            if hidden:
                aq.trace.append(("duplicates", f"collapsed {hidden} near-duplicate judgment(s)"))
        precedents = to_scored(ranked, DocType.PRECEDENT, components)
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
