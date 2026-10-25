"""Statutory drift: how far each provision moved when the BNS replaced the IPC.  [owner: Gaurav — working]

Layer 1 treats IPC 302 and BNS 103 as one offence id, which is right, and IPC 124A and BNS 152
as one offence id, which is not: the elements, the mental state and the punishment all changed.
Treating every crosswalk pair as an equivalence lets an IPC-era precedent "transfer" to a BNS
question even when the law under it was rewritten. This package measures the change instead of
assuming it away.

    provisions.py   IPC and BNS section texts, split into clauses; cross-references renumbered
    punishment.py   the punishment a text prescribes (death, life, min/max term, fine, community service)
    elements.py     mental-state words, quantified thresholds (ages, amounts, periods), content words
    diff.py         clause alignment and token diff between an old and a new provision
    model.py        DriftReport per BNS section: drift vector, class, carry-over score, reasons
    gold.py         the two independent change labels we evaluate against (never used to build reports)
    classify.py     logistic regression over the drift vector (numpy), cross-validated
    transfer.py     precedent transferability for a query, and the ranking / explanation hooks

Build:   python scripts/13_build_drift.py     -> data/crosswalk/drift_ipc_bns.csv (+ data/processed/drift.json)
Eval:    python scripts/14_eval_drift.py      -> results/tables/drift_*.csv, results/figures/drift_*.png
"""

__all__ = ["DriftClass", "DriftReport", "DriftTable"]


def __getattr__(name: str):
    if name in __all__:
        from kanoon_bridge.drift import model

        return getattr(model, name)
    raise AttributeError(name)
