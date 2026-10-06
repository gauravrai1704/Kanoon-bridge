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
    opt = SearchOptions.baseline() if baseline else SearchOptions()
    query = Query(text, state=state or None, incident_date=date)
    res = agent().run(query, opt) if use_agent else engine().search(query, opt)
    analyzed = res.analyzed if use_agent else res.query

    with st.expander("How the query was understood"):
        for step, out in analyzed.trace:
            st.write(f"**{step}** - {out}")
    if use_agent:
        with st.expander(f"What the agent did ({res.n_searches} searches)", expanded=True):
            for step, out in res.trace:
                st.write(f"**{step}** - {out}")

    left, right = st.columns(2)
    left.subheader("Statutes")
    left.dataframe([{"rank": h.rank, "doc": h.doc_id, "score": round(h.score, 3)} for h in res.statutes])
    right.subheader("Precedents")
    right.dataframe([{"rank": h.rank, "doc": h.doc_id, "score": round(h.score, 3),
                      **{k: round(v, 3) for k, v in h.components.items()},
                      **({"found by": ", ".join(res.found_by(h.doc_id))} if use_agent else {})}
                     for h in res.precedents])

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
