"""Streamlit demo page. Frontend is NOT graded — keep this thin.  [working wrapper]

    make app        (or: streamlit run app/streamlit_app.py)

Query box, state picker, incident date, statutes and precedents with score breakdowns,
the query-analysis trace, and (later) the RAG answer panel.
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


col1, col2, col3 = st.columns([4, 1, 1])
text = col1.text_input("Describe the incident or search", "mere bhai ko chaku maara")
state = col2.selectbox("Your state", STATES)
date = col3.date_input("Incident date", dt.date.today())
baseline = st.checkbox("Plain BM25 baseline")

if text:
    opt = SearchOptions.baseline() if baseline else SearchOptions()
    res = engine().search(Query(text, state=state or None, incident_date=date), opt)

    with st.expander("How the query was understood"):
        for step, out in res.query.trace:
            st.write(f"**{step}** - {out}")

    left, right = st.columns(2)
    left.subheader("Statutes")
    left.dataframe([{"rank": h.rank, "doc": h.doc_id, "score": round(h.score, 3)} for h in res.statutes])
    right.subheader("Precedents")
    right.dataframe([{"rank": h.rank, "doc": h.doc_id, "score": round(h.score, 3),
                      **{k: round(v, 3) for k, v in h.components.items()}} for h in res.precedents])

    # TODO(RAG, after the hour-26 gate): grounded answer panel with flags from rag/*_check.py
