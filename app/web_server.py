"""Kanoon-Bridge web app: a small JSON API + the single-page UI in app/web/.  [owner: Gaurav]

    make web                         # = python app/web_server.py   -> http://localhost:8000
    python app/web_server.py --port 9000 --host 0.0.0.0

Standard library only (http.server), so it runs wherever the rest of the project runs.
Everything goes through the same SearchEngine / ResearchAgent / RagPipeline as the CLI and the
evaluator (team rule 4), so the UI shows exactly what the report measures.

Endpoints (JSON):
    GET  /api/health                         index sizes, which layers and models are available
    POST /api/search   {q, state, date, agent, answer, generator, baseline, feedback: [ids]}
                       -> query understanding, statutes, precedents (snippets, reasons),
                          answer (layer 3), agent plan (layer 2), and a step-by-step pipeline
    GET  /api/similar?id=P01&k=6             similar cases
    GET  /api/doc?id=P01                     full text of one document, by zone
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
_MARK_OPEN, _MARK_CLOSE = "\x01", "\x02"


# --------------------------------------------------------------------------- the backend


class Backend:
    """Loads everything once; each request is a pure function of its JSON body."""

    def __init__(self, engine=None, docs=None) -> None:
        from kanoon_bridge.search import SearchEngine

        t0 = time.perf_counter()
        self.engine = engine or SearchEngine.load()
        self.docs = docs
        if self.docs is None:
            try:
                from kanoon_bridge.index.docstore import DocStore

                self.docs = DocStore.load(self.engine.cfg)
            except FileNotFoundError:
                pass
        self._agent = None
        self._rag = None
        self._similar = None
        self.lock = threading.Lock()              # the analyzer keeps per-query state (collision trace)
        self.load_ms = (time.perf_counter() - t0) * 1000

    # lazily built layers ----------------------------------------------------------
    @property
    def agent(self):
        if self._agent is None:
            from kanoon_bridge.agent.research import ResearchAgent

            self._agent = ResearchAgent.load(self.engine, docs=self.docs)
        return self._agent

    @property
    def rag(self):
        if self._rag is None:
            from kanoon_bridge.rag.answer import RagPipeline

            self._rag = RagPipeline.load(self.engine, docs=self.docs)
        return self._rag

    @property
    def similar(self):
        if self._similar is None:
            from kanoon_bridge.rank.similar import SimilarCases

            self._similar = SimilarCases.load(self.engine, self.docs)
        return self._similar

    # ------------------------------------------------------------------------------
    def health(self) -> dict:
        from kanoon_bridge.rag.generate import has_api_key

        e = self.engine
        return {
            "statutes": e.statute_index.n_docs, "precedents": e.precedent_index.n_docs,
            "layers": {"core": True, "agent": True, "answer": True},
            "features": {"dense": e.dense is not None, "ltr": e.ltr is not None, "authority": e.authority is not None,
                         "spelling": getattr(e.analyzer, "speller", None) is not None,
                         "claude": has_api_key(), "near_duplicates": bool(e.near_dups)},
            "load_ms": round(self.load_ms),
        }

    def search(self, body: dict) -> dict:
        from kanoon_bridge.schema import Query
        from kanoon_bridge.search import SearchOptions

        q = (body.get("q") or "").strip()
        if not q:
            return {"error": "Type a question or a section to search."}
        state = (body.get("state") or "").strip().lower() or None
        date = (body.get("date") or "").strip() or None
        opt = SearchOptions.baseline() if body.get("baseline") else SearchOptions.interactive()
        opt.top_k = int(body.get("k") or 10)
        timings: dict[str, float] = {}

        with self.lock:
            t0 = time.perf_counter()
            fb_terms = None
            marked = [d for d in body.get("feedback") or [] if isinstance(d, str)]
            if marked and self.docs is not None:
                from kanoon_bridge.rank.feedback import rocchio_options

                first = self.engine.search(Query(q, state=state, incident_date=date), copy.deepcopy(opt))
                shown = [h.doc_id for h in first.precedents]
                fb = rocchio_options(self.engine, first.query, self.docs, marked,
                                     [d for d in shown[:5] if d not in marked])
                for key, value in fb.items():
                    setattr(opt, key, value)
                fb_terms = fb

            query = Query(q, state=state, incident_date=date)
            if body.get("agent"):
                res = self.agent.run(query, opt)
                aq = res.analyzed
            else:
                res = self.engine.search(query, opt)
                aq = res.query
            timings["search"] = (time.perf_counter() - t0) * 1000

            answer = None
            if body.get("answer"):
                t1 = time.perf_counter()
                try:
                    ans = self.rag.answer(res, generator=body.get("generator") or None)
                    answer = self._answer_json(ans)
                except (RuntimeError, FileNotFoundError) as err:
                    answer = {"error": str(err)}
                timings["answer"] = (time.perf_counter() - t1) * 1000

            out = {
                "query": q, "state": state, "date": str(aq.query.incident_date) if aq.query.incident_date else None,
                "understanding": self._understanding(aq),
                "statutes": [self._statute_json(h, aq) for h in res.statutes],
                "precedents": [self._precedent_json(h, aq, res if body.get("agent") else None) for h in res.precedents],
                "answer": answer,
                "agent": self._agent_json(res) if body.get("agent") else None,
                "feedback": fb_terms and {"added": list(fb_terms["extra_terms"]), "dropped": fb_terms["drop_terms"]},
            }
            out["pipeline"] = self._pipeline(aq, res, out, timings, body)
            out["timings_ms"] = {k: round(v, 1) for k, v in {**(getattr(res, "timings_ms", {}) or {}), **timings}.items()}
        return out

    # ------------------------------------------------------------------------------ pieces
    def _understanding(self, aq) -> dict:
        from kanoon_bridge.query.parser import show

        norm = self.engine.analyzer.text_res.normalizer
        trace = dict(aq.trace)
        sections = list(dict.fromkeys(aq.sections))
        crossings = []
        for ref in sections:
            if norm is not None and ref.split(":")[0] in ("ipc", "bns"):
                for eq in norm.equivalents(ref):
                    crossings.append({"from": _show(ref), "to": _show(eq), "offence": _offence_label(norm, ref)})
        return {
            "language": aq.detected_lang, "normalised": aq.transliterated if aq.transliterated != aq.query.text else None,
            "code_in_force": aq.code_in_force.value, "sections": [_show(s) for s in sections],
            "crossings": crossings,
            "offences": [norm.label(o) for o in dict.fromkeys(aq.offence_ids)] if norm is not None else [],
            "did_you_mean": aq.suggestion,
            "corrections": [{"typed": c.word, "fixed": c.display, "applied": c.applied} for c in aq.corrections],
            "wildcards": {p: t[:12] for p, t in aq.wildcards.items()},
            "boolean": show(aq.boolean) if aq.boolean is not None else None,
            "filters": {k: v for k, v in aq.query.filters.items() if k not in ("date_from", "date_to")},
            "lexicon": trace.get("lexicon"),
            "collision": [v for k, v in aq.trace if k == "collision"],
        }

    def _statute_json(self, h, aq) -> dict:
        from kanoon_bridge import present

        ref = present.statute_ref(self.engine, h.doc_id)
        doc = self.docs.get(h.doc_id) if self.docs is not None else None
        text = ""
        if doc is not None:
            text = " ".join(p.text for p in doc.paragraphs if p.para_id >= 0)[:420]
        return {"id": h.doc_id, "rank": h.rank, "score": round(h.score, 4), "ref": _show(ref) if ref else h.doc_id,
                "title": present.title_of(self.docs, h.doc_id),
                "version": present.version_note(ref, self.engine.analyzer.text_res.normalizer),
                "in_force": bool(ref) and aq.code_in_force.value in ("ipc", "bns") and ref.startswith(aq.code_in_force.value + ":"),
                "code": ref.split(":")[0] if ref else "", "text": text}

    def _precedent_json(self, h, aq, agent_res) -> dict:
        from kanoon_bridge import present
        from kanoon_bridge.rank.authority import binding_status

        meta = self.engine.facets.metas.get(h.doc_id)
        doc = self.docs.get(h.doc_id) if self.docs is not None else None
        snippet = ""
        if doc is not None:
            raw = present.snippet(doc, aq, self.engine.analyzer.text_res, marks=(_MARK_OPEN, _MARK_CLOSE))
            snippet = html.escape(raw).replace(_MARK_OPEN, "<mark>").replace(_MARK_CLOSE, "</mark>")
        status = binding_status(meta, aq.query.state) if meta is not None else "unknown"
        return {
            "id": h.doc_id, "rank": h.rank, "score": round(h.score, 4), "title": present.title_of(self.docs, h.doc_id),
            "court": (meta.court_name or meta.court.replace("_", " ").title()) if meta else "",
            "year": meta.decision_date.year if meta and meta.decision_date else None,
            "code": meta.code if meta else "", "status": status, "snippet": snippet,
            "why": present.explain(h, aq, self.engine),
            "components": {k: round(v, 4) for k, v in (h.components or {}).items()},
            "found_by": agent_res.found_by(h.doc_id) if agent_res is not None else [],
        }

    def _answer_json(self, ans) -> dict:
        from kanoon_bridge import present
        from kanoon_bridge.rag._util import strip_citations

        sources = [{"n": c.n, "id": c.doc_id, "title": present.title_of(self.docs, c.doc_id) if self.docs is not None else (c.title or c.doc_id), "zone": c.zone} for c in ans.chunks]
        return {"text": ans.text, "abstained": ans.abstained, "generator": ans.generator,
                "sentences": [{"text": strip_citations(s), "cite": n} for s, n in ans.sentences],
                "sources": sources, "flags": ans.flags,
                "supported": round(ans.supported_rate, 3) if ans.sentences else None}

    def _agent_json(self, res) -> dict:
        plans = []
        for p in res.plans:
            plans.append([{"id": sq.sq_id, "kind": sq.kind, "text": sq.query.text, "why": sq.rationale,
                           "target": sq.target,
                           "hits": next((len(sr.ranking(sr.subquery.target)) for sr in res.subresults
                                         if sr.subquery is sq), 0)} for sq in p.subqueries])
        return {"rounds": plans, "searches": res.n_searches, "fusion": self.agent.fusion,
                "notes": [v for k, v in res.trace if k.endswith("reflect") or k.endswith("note")]}

    def _pipeline(self, aq, res, out, timings, body) -> list[dict]:
        """The run as a sequence of stages with what each one actually did (for 'How this was found')."""
        tr = aq.trace
        t = getattr(res, "timings_ms", {}) or {}

        def get(step):
            return [v for k, v in tr if k == step]

        u = out["understanding"]
        stages = []

        def add(group, name, detail, ms=None, on=True):
            stages.append({"group": group, "name": name, "detail": detail, "ms": None if ms is None else round(ms, 1),
                           "on": on})

        add("Understand", "Read filters and query syntax",
            (f"Boolean query: {u['boolean']}" if u["boolean"] else "Plain question, no operators")
            + (f"; filters {u['filters']}" if u["filters"] else ""))
        add("Understand", "Detect language", {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish"}
            .get(u["language"], u["language"]) + (f"; normalised to “{u['normalised']}”" if u["normalised"] else ""))
        if u["language"] != "en":
            add("Understand", "Hindi legal lexicon", u["lexicon"] or "no lexicon words")
        if u["wildcards"]:
            add("Understand", "Expand wildcards", "; ".join(f"{p} → {', '.join(v[:6])}" for p, v in u["wildcards"].items()))
        add("Understand", "Check spelling", "; ".join(f"{c['typed']} → {c['fixed']}" for c in u["corrections"])
            or "every word is in the index")
        add("Understand", "Tokenise and stem", (get("tokens") or [""])[0][:220])
        if u["collision"]:
            add("Understand", "Resolve bare section numbers", " | ".join(u["collision"]))
        add("Understand", "Decide the code in force", {"bns": "BNS (incident on or after 1 July 2024)",
                                                       "ipc": "IPC (incident before 1 July 2024)"}
            .get(u["code_in_force"], "no date given: both codes searched"), t.get("analyze"))
        add("Understand", "Cross the IPC–BNS bridge",
            "; ".join(f"{c['from']} ≡ {c['to']} ({c['offence']})" for c in u["crossings"])
            or (", ".join(u["offences"]) or "no sections named in the question"))
        add("Find the law", "Score statutes (BM25F)",
            (get("statute_filter") or ["all statutes"])[0] + f"; top: {', '.join(s['ref'] for s in out['statutes'][:3])}",
            t.get("statutes"))
        bridge = (get("bridge_statutes") or [None])[0]
        add("Find the law", "Statute bridge",
            ("precedents citing these get a boost: " + re.sub(r"\b(ipc|bns):(\w+)", lambda m: f"{m.group(1).upper()} {m.group(2)}", bridge)[:200]) if bridge
            else ("used inside each agent search" if out["agent"] else "off"), t.get("bridge"))
        extra = []
        for k in ("facet", "boolean_match", "feedback", "alpha"):
            extra += [f"{k}: {v}" for v in get(k)]
        ng = get("ngram")
        if ng:
            add("Find cases", "Shared phrasing (word trigrams)", ng[0])
        add("Find cases", "Score precedents by zone (BM25F)",
            f"{len(out['precedents'])} shown" + (f"; {'; '.join(extra)}" if extra else ""), t.get("precedents"))
        e = self.engine
        add("Find cases", "Weigh authority for your state",
            f"PageRank over the citation graph; binding = Supreme Court + {('the ' + out['state'].title() + ' High Court') if out['state'] else 'your High Court'}",
            on=e.authority is not None and not body.get("baseline"))
        add("Find cases", "Dense (multilingual) channel", "meaning-based match" if e.dense is not None else "off (run make dense)",
            on=e.dense is not None)
        ltr_note = (get("ltr") or [None])[0] or ("applied inside every agent search" if e.ltr is not None and out["agent"]
                                                 else "no model trained (run make ltr)" if e.ltr is None else "not used")
        add("Find cases", "Learned re-ranking", ltr_note, on=e.ltr is not None and not body.get("baseline"))
        add("Find cases", "Collapse near-duplicates", (get("duplicates") or ["no duplicates in these results"])[0])
        if out["agent"]:
            for r_i, rnd in enumerate(out["agent"]["rounds"], start=1):
                for sq in rnd:
                    add("Research agent", f"Round {r_i}: {sq['kind'].replace('_', ' ')}",
                        f"“{sq['text'][:120]}”, {sq['hits']} hits. {sq['why']}")
            add("Research agent", f"Fuse {out['agent']['searches']} result lists ({out['agent']['fusion'].upper()})",
                " | ".join(out["agent"]["notes"]))
        if out["answer"] is not None:
            a = out["answer"]
            if a.get("error"):
                add("Answer", "Write the answer", a["error"], timings.get("answer"))
            elif a["abstained"]:
                add("Answer", "Decide whether to answer", a["text"], timings.get("answer"))
            else:
                add("Answer", "Decide whether to answer", "retrieval is strong enough to ground an answer")
                add("Answer", "Pick sources", f"{len(a['sources'])} numbered extracts: top statutes, then the reasoning of top cases")
                add("Answer", "Write the answer", f"{len(a['sentences'])} sentences by {a['generator']}", timings.get("answer"))
                add("Answer", "Check every sentence", f"{round((a['supported'] or 0) * 100)}% supported by the source it cites; "
                    + (f"{len(a['flags'])} flag(s)" if a["flags"] else "no flags"))
        return stages


def _show(ref: str) -> str:
    code, _, num = ref.partition(":")
    if not num:
        return ref
    # short codes as abbreviations (IPC, BNS, CrPC-style); slugs of other Acts as words
    name = code.upper() if len(code) <= 6 else code.replace("_", " ").title().replace(" Of ", " of ").replace(" And ", " and ")
    return f"{name} {num}"


def _offence_label(norm, ref: str) -> str:
    offs = norm.offences_for(ref)
    return norm.label(offs[0]) if offs else ""


# --------------------------------------------------------------------------- HTTP


class Handler(SimpleHTTPRequestHandler):
    backend: Backend = None  # set in main()

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path == "/api/health":
                return self._json(self.backend.health())
            if url.path == "/api/similar":
                doc_id = (qs.get("id") or [""])[0]
                k = int((qs.get("k") or ["6"])[0])
                from kanoon_bridge import present

                b = self.backend
                found = b.similar.find(doc_id, k)
                return self._json({"id": doc_id, "title": present.title_of(b.docs, doc_id),
                                   "similar": [{"id": d, "title": present.title_of(b.docs, d), "score": round(s, 3),
                                                "parts": parts} for d, s, parts in found]})
            if url.path == "/api/doc":
                doc_id = (qs.get("id") or [""])[0]
                d = self.backend.docs.get(doc_id) if self.backend.docs is not None else None
                if d is None:
                    return self._json({"error": f"No document with id {doc_id}."}, 404)
                return self._json({"id": d.doc_id, "title": d.title, "court": d.court_name,
                                   "date": str(d.decision_date) if d.decision_date else None,
                                   "paragraphs": [{"zone": p.zone, "text": p.text} for p in d.paragraphs if p.para_id >= 0]})
        except Exception as err:  # noqa: BLE001 - report to the UI
            return self._json({"error": f"{type(err).__name__}: {err}"}, 500)
        if url.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/search":
            return self._json({"error": "Unknown endpoint."}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._json(self.backend.search(body))
        except Exception as err:  # noqa: BLE001
            return self._json({"error": f"{type(err).__name__}: {err}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kanoon-Bridge web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print("loading indexes ...", flush=True)
    try:
        Handler.backend = Backend()
    except FileNotFoundError as err:
        sys.exit(f"{err}\nBuild the indexes first: make data index graph   (or: make sample)")
    print(f"Kanoon-Bridge is running at http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
