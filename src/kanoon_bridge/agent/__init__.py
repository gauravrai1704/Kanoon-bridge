"""Layer 2: the legal research agent.  [owner: Shaurya — working]

The agent sits in FRONT of the core retriever (search.py) and issues several IR queries on
the user's behalf, like a junior researcher:

    plan      agent/plan.py      one question -> several sub-queries (free text, Boolean,
                                 cross-code, facet-restricted, statutes-first; optional LLM)
    execute   agent/executor.py  run each sub-query through search.SearchEngine (rule 4)
    fuse      agent/fuse.py      merge the ranked lists (reciprocal rank fusion or CombSUM)
    reflect   agent/reflect.py   if QPP says retrieval is weak, reformulate with
                                 pseudo-relevance feedback and run one more round (max 2)
    run       agent/research.py  ResearchAgent.run(query) -> AgentResult with a full trace

IR stays central: the agent BUILDS Boolean/facet queries and FUSES ranked lists; any LLM is
optional and only proposes sub-queries.
"""
