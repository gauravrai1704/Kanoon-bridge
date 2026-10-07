"""Citation-shift measurement and High Court ingest (scripts/11_hc_judgments.py) on toy judgments."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("hc", Path(__file__).parents[1] / "scripts" / "11_hc_judgments.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

J = [
    {"id": "a", "court": "Delhi High Court", "state": "delhi", "date": "2024-05-10", "title": "A v State",
     "text": "The accused was charged under Section 302 IPC and Section 34 IPC. Bail under Section 439 CrPC.\n\n2. " + "word " * 20},
    {"id": "b", "court": "Delhi High Court", "state": "delhi", "date": "2024-11-02", "title": "B v State",
     "text": "FIR under Section 103 of the BNS. Anticipatory bail under Section 482 BNSS is sought.\n\n2. " + "word " * 20},
    {"id": "c", "court": "Delhi High Court", "state": "delhi", "date": "2024-11-20", "title": "C v State",
     "text": "Offence under Section 420 IPC committed in 2023; trial under Section 528 BNSS; Section 63 BSA certificate.\n\n2. " + "word " * 20},
]


def test_cited_codes():
    assert hc.cited_codes(J[0]["text"]) == {"ipc": 2, "crpc": 1}
    assert hc.cited_codes(J[1]["text"]) == {"bns": 1, "bnss": 1}


def test_shift_table_by_month():
    rows = {r["month"]: r for r in hc.shift_table(J)}
    assert rows["2024-05"]["bns_share_of_mentions"] == 0.0
    nov = rows["2024-11"]
    assert nov["judgments"] == 2 and nov["bns_share_of_mentions"] == 0.5     # one BNS, one IPC mention
    assert nov["bnss_share_of_mentions"] == 1.0 and nov["bsa_share_of_judgments"] == 0.5


def test_to_documents():
    docs = hc.to_documents(J)
    assert len(docs) == 3 and docs[1].doc_id == "hc:b" and "bns:103" in docs[1].statutes_cited
    assert docs[1].decision_date.isoformat() == "2024-11-02"
