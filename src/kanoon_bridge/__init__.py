"""Kanoon-Bridge: legal search across the IPC to BNS change.

Package map (owner in brackets):
    schema.py   shared data classes            [all, agree in hour 1]
    config.py   reads configs/*.yaml            [D]
    search.py   the single search entry point   [D]
    ingest/     raw data -> Documents           [A; crosswalk B]
    text/       tokenise, normalise, translit   [A, B, C]
    index/      postings, zones, facets, tiers  [A; tiers D]
    query/      parse and analyse queries       [A; analyzer C]
    rank/       scoring and fusion              [D; dense + fusion C]
    rag/        grounded answers (stretch)      [first free member]
    eval/       metrics and experiments         [D]
"""

__version__ = "0.1.0"
