"""Streamlit demo page. Frontend is NOT graded — keep this thin.  [working wrapper]

    make app        (or: streamlit run app/streamlit_app.py)

Query box, state picker, incident date, statutes and precedents with score breakdowns,
the query-analysis trace, and the layer-3 grounded answer panel (checkbox).
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from kanoon_bridge.schema import Query
from kanoon_bridge.search import SearchEngine, SearchOptions

STATES = ["", "delhi", "maharashtra", "uttar-pradesh", "karnataka", "tamil-nadu", "west-bengal",
          "gujarat", "rajasthan", "punjab", "haryana", "kerala", "telangana", "bihar"]

st.set_page_config(page_title="Kanoon-Bridge", layout="wide")
st.title("Kanoon-Bridge")
st.caption("Search Indian criminal law across IPC and BNS, in English, Hindi or Hinglish. Not legal advice.")


@st.cache_resource
def engine() -> SearchEngine:
    return SearchEngine.load()


@st.cache_resource
def rag():
    from kanoon_bridge.rag.answer import RagPipeline

    return RagPipeline.load(engine(), docs=agent().docs)


@st.cache_resource
def agent():
    from kanoon_bridge.agent.research import ResearchAgent

    return ResearchAgent.load(engine())


col1, col2, col3 = st.columns([4, 1, 1])
text = col1.text_input("Describe the incident or search", "mere bhai ko chaku maara")
state = col2.selectbox("Your state", STATES)
date = col3.date_input("Incident date", dt.date.today())
c1, c2, c3 = st.columns(3)
baseline = c1.checkbox("Plain BM25 baseline")
use_agent = c2.checkbox("Research agent (layer 2)")
want_answer = c3.checkbox("Grounded answer (layer 3)")

if text:
    opt = SearchOptions.baseline() if baseline else SearchOptions.interactive()
    query = Query(text, state=state or None, incident_date=date)
    feedback_key = f"fb::{text}::{state}::{date}"
    marked = st.session_state.get(feedback_key, [])
    if marked:                                                    # explicit relevance feedback (Rocchio)
        from kanoon_bridge.rank.feedback import rocchio_options

        first = engine().search(Query(text, state=state or None, incident_date=date), opt)
        fb = rocchio_options(engine(), first.query, agent().docs, marked,
                             [h.doc_id for h in first.precedents[:5] if h.doc_id not in marked])
        for key, value in fb.items():
            setattr(opt, key, value)
        st.caption(f"Refined with your feedback: +{' +'.join(fb['extra_terms'])}")
    res = agent().run(query, opt) if use_agent else engine().search(query, opt)
    analyzed = res.analyzed if use_agent else res.query

    with st.expander("How the query was understood"):
        for step, out in analyzed.trace:
            st.write(f"**{step}** - {out}")
    if use_agent:
        with st.expander(f"What the agent did ({res.n_searches} searches)", expanded=True):
            for step, out in res.trace:
                st.write(f"**{step}** - {out}")

    if analyzed.suggestion:
        st.info(f"Did you mean: **{analyzed.suggestion}** (corrections already included in the search)")

    from kanoon_bridge import present

    docs = agent().docs
    norm = engine().analyzer.text_res.normalizer
    res_text = engine().analyzer.text_res

    with st.sidebar:
        st.subheader("Results by facet")
        for name, counts in present.facet_counts(res.precedents, engine()).items():
            if counts:
                st.caption(name)
                st.write(", ".join(f"{k}: {v}" for k, v in counts.most_common()))

    left, right = st.columns([2, 3])
    left.subheader("Statutes")
    for h in res.statutes:
        note = present.version_note(present.statute_ref(engine(), h.doc_id), norm)
        left.markdown(f"**{h.rank}. {present.title_of(docs, h.doc_id)}** · `{h.doc_id}`"
                      + (f"  \n<small>{note}</small>" if note else ""), unsafe_allow_html=True)
    right.subheader("Precedents")
    for h in res.precedents:
        meta = engine().facets.metas.get(h.doc_id)
        when = f"{meta.court_name or meta.court}, {meta.decision_date.year}" if meta and meta.decision_date else ""
        with right.container(border=True):
            st.markdown(f"**{h.rank}. {present.title_of(docs, h.doc_id)}** · {when} · score {h.score:.3f}")
            if docs is not None and h.doc_id in docs:
                st.markdown(present.snippet(docs[h.doc_id], analyzed, res_text))
            why = present.explain(h, analyzed, engine())
            if use_agent:
                why.append("found by " + ", ".join(res.found_by(h.doc_id)))
            if why:
                st.caption(" · ".join(why))
            if st.checkbox("relevant", key=f"rel::{feedback_key}::{h.doc_id}", value=h.doc_id in marked):
                st.session_state.setdefault(f"pending::{feedback_key}", set()).add(h.doc_id)
            with st.expander("score breakdown"):
                st.json({k: round(v, 4) for k, v in h.components.items()})

    pending = st.session_state.get(f"pending::{feedback_key}", set())
    if pending and st.button(f"Refine with {len(pending)} relevant result(s)"):
        st.session_state[feedback_key] = sorted(pending)
        st.rerun()

    if res.precedents:
        with st.expander("Similar cases"):
            pick = st.selectbox("Find cases similar to", [h.doc_id for h in res.precedents],
                                format_func=lambda d: present.title_of(docs, d))
            from kanoon_bridge.rank.similar import SimilarCases

            for d, sc, parts in SimilarCases.load(engine(), docs).find(pick, 5):
                st.write(f"**{present.title_of(docs, d)}** · {sc:.3f} · "
                         + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()))

    if want_answer:
        st.subheader("Answer")
        try:
            ans = rag().answer(res)
            st.write(ans.text)
            for c in ans.chunks:
                st.caption(c.header())
            for f in ans.flags:
                st.warning(f)
            st.caption(f"generator: {ans.generator} - not legal advice")
        except (RuntimeError, FileNotFoundError) as err:
            st.error(str(err))
