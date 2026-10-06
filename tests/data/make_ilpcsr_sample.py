"""Write a tiny SYNTHETIC sample in IL-PCSR's schema to tests/data/ilpcsr_sample/.

Fictional cases for testing the pipeline only — not real judgments, never use them in results.
Same columns and split names as the real dataset (see ingest/load_ilpcsr.py), so
`python scripts/01_build_corpus.py --sample` exercises exactly the real code path.

    python tests/data/make_ilpcsr_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "ilpcsr_sample"

STATUTES = [
    ("S302", "Section 302 in The Indian Penal Code, 1860", ["Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."]),
    ("S307", "Section 307 in The Indian Penal Code, 1860", ["Whoever does any act with such intention or knowledge that, if he by that act caused death, he would be guilty of murder, shall be punished (attempt to murder)."]),
    ("S498A", "Section 498A in The Indian Penal Code, 1860", ["Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished."]),
    ("S304B", "Section 304B in The Indian Penal Code, 1860", ["Where the death of a woman is caused within seven years of her marriage and she was subjected to cruelty for dowry, such death shall be called dowry death."]),
    ("S506", "Section 506 in The Indian Penal Code, 1860", ["Whoever commits the offence of criminal intimidation shall be punished."]),
    ("S34", "Section 34 in The Indian Penal Code, 1860", ["When a criminal act is done by several persons in furtherance of the common intention of all, each is liable."]),
    ("S438", "Section 438 in The Code Of Criminal Procedure, 1973", ["Direction for grant of bail to person apprehending arrest (anticipatory bail)."]),
    ("A21", "Article 21 in Constitution of India", ["No person shall be deprived of his life or personal liberty except according to procedure established by law."]),
]

# id, title, date, jurisdiction, [(role, text)], statute ids, precedent ids
PRECEDENTS = [
    ("P01", "State v. Arjun (synthetic)", "1998-03-12", "Supreme Court of India",
     [("Facts", "The accused stabbed the deceased with a knife after a quarrel over land."),
      ("PetArg", "Learned counsel for the appellant submitted that there was no intention to kill."),
      ("CourtRes", "We hold that the injury was sufficient in the ordinary course of nature to cause death, so the offence of murder under Section 302 IPC is made out."),
      ("Conclusion", "The appeal is dismissed and the conviction under Section 302 IPC is upheld.")], ["S302"], []),
    ("P02", "Meena v. State (synthetic)", "2005-07-01", "Delhi High Court",
     [("Facts", "The husband and his mother demanded dowry; the wife died of burns within four years of marriage."),
      ("CourtRes", "The presumption of dowry death under Section 304B IPC applies; cruelty under Section 498A IPC is proved."),
      ("Conclusion", "Bail is refused.")], ["S304B", "S498A"], ["P01"]),
    ("P03", "Rakesh v. State of Maharashtra (synthetic)", "2010-11-20", "Bombay High Court",
     [("Facts", "The accused threatened the complainant with dire consequences and showed a knife."),
      ("CourtRes", "Criminal intimidation under Section 506 IPC is established on the evidence."),
      ("Conclusion", "The revision petition is dismissed.")], ["S506"], []),
    ("P04", "Suresh v. State of U.P. (synthetic)", "2012-02-14", "Allahabad High Court",
     [("Facts", "Three accused attacked the victim with sticks and a knife in furtherance of a common plan."),
      ("CourtRes", "Common intention under Section 34 IPC is inferred; the knife injury makes it attempt to murder under Section 307 IPC."),
      ("Conclusion", "Conviction under Section 307 read with Section 34 IPC is affirmed.")], ["S307", "S34"], ["P01"]),
    ("P05", "Kavita v. State (NCT of Delhi) (synthetic)", "2016-09-09", "Delhi High Court",
     [("Facts", "The applicant feared arrest in a matrimonial dispute alleging cruelty."),
      ("CourtRes", "Anticipatory bail under Section 438 CrPC may be granted where the allegations are general; personal liberty under Article 21 weighs heavily."),
      ("Conclusion", "Anticipatory bail is granted.")], ["S438", "A21", "S498A"], ["P02"]),
    ("P06", "Prakash v. State of Maharashtra (synthetic)", "2017-04-03", "Bombay High Court",
     [("Facts", "The deceased was stabbed repeatedly with a knife during a robbery."),
      ("CourtRes", "In our view the multiple knife injuries show intention to cause death; murder under Section 302 IPC."),
      ("Conclusion", "The appeal is dismissed.")], ["S302"], ["P01"]),
    ("P07", "Union of India v. Farooq (synthetic)", "2019-01-22", "Supreme Court of India",
     [("Issue", "Whether bail can be refused solely because the offence is serious."),
      ("CourtRes", "It is well settled that personal liberty under Article 21 cannot be curtailed except by fair procedure; bail is the rule."),
      ("Conclusion", "The petition is allowed and bail is granted.")], ["A21", "S438"], ["P05"]),
    ("P08", "Neha v. State of U.P. (synthetic)", "2014-06-30", "Allahabad High Court",
     [("Facts", "The wife alleged harassment for dowry and cruelty by her husband's relatives."),
      ("CourtRes", "Cruelty under Section 498A IPC is made out against the husband only; general allegations against relatives fail."),
      ("Conclusion", "The petition is partly allowed.")], ["S498A"], ["P02", "P05"]),
]

QUERIES = {
    "train_queries": [
        ("Q01", "Raju v. State (synthetic query)", "2011-05-05", "Delhi High Court",
         [("Facts", "The accused stabbed his neighbour with a knife; the neighbour died in hospital."),
          ("CourtRes", "The offence falls under [SECTION] of the [ACT]; we rely on [PRECEDENT].")], ["S302"], ["P01", "P06"]),
        ("Q02", "Sunita v. State (synthetic query)", "2013-08-19", "Delhi High Court",
         [("Facts", "The bride died of burns within two years of marriage after dowry demands."),
          ("CourtRes", "Dowry death and cruelty are made out under [SECTION].")], ["S304B", "S498A"], ["P02"]),
        ("Q03", "Amit v. State of Maharashtra (synthetic query)", "2018-12-01", "Bombay High Court",
         [("Facts", "The accused threatened the shopkeeper and brandished a knife."),
          ("CourtRes", "Intimidation is proved under [SECTION]; see [PRECEDENT].")], ["S506"], ["P03"]),
    ],
    "dev_queries": [
        ("Q04", "Imran v. State (synthetic query)", "2015-03-03", "Allahabad High Court",
         [("Facts", "Several accused attacked the victim with a knife in furtherance of common intention."),
          ("CourtRes", "Attempt to murder with common intention under [SECTION].")], ["S307", "S34"], ["P04"]),
    ],
    "test_queries": [
        ("Q05", "Pooja v. State (synthetic query)", "2017-10-10", "Delhi High Court",
         [("Facts", "The applicant seeks anticipatory bail in a dowry harassment case."),
          ("CourtRes", "Personal liberty and anticipatory bail under [SECTION]; we follow [PRECEDENT].")], ["S438", "A21"], ["P05", "P07"]),
        ("Q06", "Vikram v. State of Maharashtra (synthetic query)", "2019-07-07", "Bombay High Court",
         [("Facts", "The deceased was stabbed with a knife in a robbery at night."),
          ("CourtRes", "Murder is made out under [SECTION]; [PRECEDENT] applies.")], ["S302"], ["P06", "P01"]),
    ],
}


def judgment_row(i, title, date, juris, paras, secs, precs):
    names = {s[0]: s[1] for s in STATUTES}
    return {"id": i, "case_title": title, "date": date, "jurisdiction": juris,
            "text": [t for _, t in paras], "rhetorical_roles": [r for r, _ in paras],
            "relevant_statutes": [names[s] for s in secs], "relevant_statute_ids": secs,
            "relevant_precedents": [p for p in precs], "relevant_precedent_ids": precs}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    write(OUT / "statutes" / "statute_candidates.jsonl",
          [{"id": i, "provision_name": n, "text": t} for i, n, t in STATUTES])
    write(OUT / "precedents" / "precedent_candidates.jsonl", [judgment_row(*p) for p in PRECEDENTS])
    for split, qs in QUERIES.items():
        write(OUT / "queries" / f"{split}.jsonl", [judgment_row(*q) for q in qs])
    print("wrote synthetic sample to", OUT)


if __name__ == "__main__":
    main()
