"""Evaluate layers 2 and 3.  [agent / RAG owners]

Agent (layer 2), on E1 and E6:
    core single query  vs  agent (RRF)  vs  agent (CombSUM)  vs  agent (LLM planner, optional)
    metrics: MAP, F1@k, nDCG@10 (E6); plus searches per question and the share of questions
    where a reformulation round improved AP

RAG (layer 3), on ~20 questions:
    supported-sentence rate (vs the same LLM without retrieval)
    version errors per answer (with vs without the normaliser)
    checker agreement with a human spot-check
    abstention precision on questions the corpus cannot answer

Output: results/tables/agent.csv, results/tables/rag.csv
"""

from __future__ import annotations


def compare_agent(test_set: str = "e1_ilpcsr") -> list[dict]:
    """TODO: run_eval.load_test_set -> for each system, rankings per query -> metrics.evaluate.
    Sanity check first: an agent whose plan has only the 'original' sub-query must match the core."""
    raise NotImplementedError("TODO: agent vs core comparison")


def evaluate_rag(questions_path: str = "data/queries/rag_questions.jsonl") -> list[dict]:
    """TODO: answer each question with and without retrieval; count flags from the checks."""
    raise NotImplementedError("TODO: RAG evaluation")
