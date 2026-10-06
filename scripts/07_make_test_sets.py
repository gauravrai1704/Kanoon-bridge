"""Generate the E2, E3 and E7 test sets automatically, with gold answers.  [owner: Gaurav]

    python scripts/07_make_test_sets.py                   # all three (after make data)
    python scripts/07_make_test_sets.py --sets e3 e7      # some of them
    python scripts/07_make_test_sets.py --sample          # on the synthetic sample (pipeline check)

Needs data/processed/docs.jsonl (make data) and, for E3, the IL-PCSR test split.
Overwrites data/queries/{e2_collision,e3_cross_version,e3_control,e7_temporal}.jsonl and their
qrels in data/queries/qrels/. Every row carries "generated": true, so the report can say how
each set was made. The answers are only as good as the crosswalk (check it: make compare).

E3, cross-version robustness (precedent retrieval)
    For every IPC section that at least 3 precedents applied (IL-PCSR citation labels) and that
    has a BNS equivalent, the same need is asked in both numberings:
        e3_control        "cases under section 302 IPC"
        e3_cross_version  "cases under section 103 BNS"
    Gold (both): the precedents that applied IPC 302. A version-aware system should score about
    the same on both; BM25 on the text as written collapses on the BNS wording, because the
    judgments (all decided before July 2024) never say "BNS 103".

E7, temporal correctness (statute retrieval)
    For each offence with a section in BOTH codes, its name ("criminal intimidation") is asked
    twice: incident on 2024-03-01 (gold: the IPC section(s)) and on 2024-09-01 (gold: the BNS
    section(s)). Headline metric: code_accuracy@1.

E2, section-number collisions (statute retrieval)
    Section numbers that mean different offences in the two codes (302: IPC murder, BNS
    uttering words to wound religious feelings). Each number is asked bare ("punishment under
    section 302") and with context words of the intended offence, once before and once after
    1 July 2024. Gold: the reading in force; "wrong_refs" holds the other code's reading.
    Headline metric: wrong_hit@10 (lower is better).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

BEFORE, AFTER = "2024-03-01", "2024-09-01"
MASK = re.compile(r"\[[A-Z?_ /]+\]")
TITLE_LEAD = re.compile(r"^(punishment\s+(for|of)\s+)", re.I)


def write_set(name: str, rows: list[dict], qrels: list[tuple[str, str, int]] | None, out_dir: Path) -> None:
    q_path = out_dir / f"{name}.jsonl"
    with open(q_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = f"{name}: {len(rows)} queries -> {q_path}"
    if qrels is not None:
        qr_path = out_dir / "qrels" / f"{name}.tsv"
        qr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qr_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query_id", "doc_id", "grade", "judge"])
            w.writerows([q, d, g, "generated"] for q, d, g in qrels)
        msg += f", {len(qrels)} judgments -> {qr_path}"
    print(msg)


def statute_index(docs) -> tuple[dict[str, list[str]], dict[str, str]]:
    """ref ('ipc:302') -> statute doc ids, and doc id -> title."""
    by_ref: dict[str, list[str]] = {}
    titles = {}
    for d in docs:
        if d.doc_type.value != "statute":
            continue
        ref = (d.meta or {}).get("ref") or ""
        if ref:
            by_ref.setdefault(ref, []).append(d.doc_id)
            base = re.match(r"(\w+:\d+[a-z]*)", ref)
            if base and base.group(1) != ref:
                by_ref.setdefault(base.group(1), []).append(d.doc_id)
        titles[d.doc_id] = d.title
    return by_ref, titles


def _show(ref: str) -> str:
    code, _, num = ref.partition(":")
    return f"{code.upper()} {num}"


def _offence_words(label: str) -> str:
    return TITLE_LEAD.sub("", label).strip().rstrip(".").lower()


# --------------------------------------------------------------------------- E3


def make_e3(norm, docs, limit: int | None, rng: random.Random, min_cases: int = 3):
    """Section-only queries for precedents, asked in the old and the new numbering."""
    cited: dict[str, set[str]] = {}
    for d in docs:
        if d.doc_type.value != "precedent":
            continue
        for ref in d.statutes_cited:
            if ref.startswith("ipc:"):
                base = re.match(r"ipc:(\d+[a-z]*)", ref)
                cited.setdefault(base.group(1) if base else ref[4:], set()).add(d.doc_id)
    vague = ("explanation", "definition", "general")
    control, crossed, qrels = [], [], []
    for num, cases in sorted(cited.items(), key=lambda kv: (len(kv[0]), kv[0])):
        eq = [e for e in norm.equivalents(f"ipc:{num}") if e.startswith("bns:")]
        labels = [norm.label(o).lower() for o in norm.offences_for(f"ipc:{num}")]
        if len(cases) < min_cases or not eq or any(v in lab for lab in labels for v in vague):
            continue
        bns = eq[0].split(":", 1)[1]
        base = {"lang": "en", "state": None, "ipc": f"ipc:{num}", "bns": eq[0], "generated": True}
        control.append({"id": f"E3C-{num}", "text": f"cases under section {num} IPC", "variant": "ipc", **base})
        crossed.append({"id": f"E3-{num}", "text": f"cases under section {bns} BNS", "variant": "bns", **base})
        for q in (f"E3C-{num}", f"E3-{num}"):
            qrels += [(q, d, 1) for d in sorted(cases)]
    if limit and len(crossed) > limit:
        keep = set(rng.sample(range(len(crossed)), limit))
        control = [r for i, r in enumerate(control) if i in keep]
        crossed = [r for i, r in enumerate(crossed) if i in keep]
        ids = {r["id"] for r in control + crossed}
        qrels = [x for x in qrels if x[0] in ids]
    return control, crossed, qrels


# --------------------------------------------------------------------------- E7


def make_e7(norm, by_ref, limit: int | None, rng: random.Random):
    rows, qrels = [], []
    for oid, info in sorted(norm.offences.items()):
        label = info.get("label") or ""
        ipc = [f"ipc:{s}" for s in info.get("ipc", [])]
        bns = [f"bns:{s}" for s in info.get("bns", [])]
        ipc_docs = sorted({d for r in ipc for d in by_ref.get(r, [])})
        bns_docs = sorted({d for r in bns for d in by_ref.get(r, [])})
        words = _offence_words(label)
        if not ipc_docs or not bns_docs or len(words.split()) < 1 or len(words) < 6:
            continue
        slug = oid.removeprefix("off:")
        for when, docs, code in ((BEFORE, ipc_docs, "ipc"), (AFTER, bns_docs, "bns")):
            qid = f"E7-{slug}-{code}"
            rows.append({"id": qid, "text": words, "lang": "en", "state": None, "incident_date": when,
                         "expected_code": code, "offence": oid, "generated": True})
            qrels += [(qid, d, 2) for d in docs]
    if limit and len(rows) > 2 * limit:
        offences = sorted({r["offence"] for r in rows})
        keep = set(rng.sample(offences, limit))
        rows = [r for r in rows if r["offence"] in keep]
        ids = {r["id"] for r in rows}
        qrels = [x for x in qrels if x[0] in ids]
    return rows, qrels


# --------------------------------------------------------------------------- E2


def make_e2(norm, by_ref, limit: int | None, rng: random.Random):
    ipc_nums = {r.split(":")[1] for r in norm.section_to_offences if r.startswith("ipc:") and "(" not in r}
    bns_nums = {r.split(":")[1] for r in norm.section_to_offences if r.startswith("bns:") and "(" not in r}
    rows, qrels = [], []
    nums = sorted(ipc_nums & bns_nums, key=lambda n: (len(n), n))
    collisions = []
    vague = ("explanation", "definition", "general")
    for n in nums:
        a, b = set(norm.offences_for(f"ipc:{n}")), set(norm.offences_for(f"bns:{n}"))
        labels = [norm.label(o).lower() for o in a | b]
        if any(v in lab for lab in labels for v in vague):
            continue                                    # definitional sections make poor queries
        if a and b and not (a & b) and by_ref.get(f"ipc:{n}") and by_ref.get(f"bns:{n}"):
            collisions.append(n)
    if limit and len(collisions) > limit:
        collisions = sorted(rng.sample(collisions, limit), key=lambda n: (len(n), n))
    for n in collisions:
        for when, code, other in ((BEFORE, "ipc", "bns"), (AFTER, "bns", "ipc")):
            label = norm.label(norm.offences_for(f"{code}:{n}")[0])
            for style, text in (("bare", f"punishment under section {n}"),
                                ("context", f"section {n} {_offence_words(label)}")):
                qid = f"E2-{n}-{code}-{style}"
                rows.append({"id": qid, "text": text, "lang": "en", "state": None, "incident_date": when,
                             "wrong_refs": [f"{other}:{n}"], "expected": f"{code}:{n}", "style": style, "generated": True})
                qrels += [(qid, d, 2) for d in by_ref[f"{code}:{n}"]]
    return rows, qrels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["e2", "e3", "e7"], choices=["e2", "e3", "e7"])
    ap.add_argument("--limit", type=int, help="cap per set (E3: queries, E7: offences, E2: numbers)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    if args.sample:
        os.environ["KB_ILPCSR_DIR"] = "tests/data/ilpcsr_sample"

    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.schema import read_documents
    from kanoon_bridge.text.version_norm import VersionNormalizer

    cfg = load_config()
    norm = VersionNormalizer.load(cfg)
    docs_path = project_path(cfg.paths.docs)
    if not docs_path.exists():
        raise SystemExit(f"{docs_path} not found - run make data first")
    by_ref, _ = statute_index(read_documents(docs_path))
    out = project_path("data/queries")
    rng = random.Random(args.seed)

    if "e3" in args.sets:
        control, crossed, qrels = make_e3(norm, list(read_documents(docs_path)), args.limit, rng)
        write_set("e3_control", control, [x for x in qrels if x[0].startswith("E3C-")], out)
        write_set("e3_cross_version", crossed, [x for x in qrels if not x[0].startswith("E3C-")], out)
    if "e7" in args.sets:
        rows, qrels = make_e7(norm, by_ref, args.limit, rng)
        write_set("e7_temporal", rows, qrels, out)
    if "e2" in args.sets:
        rows, qrels = make_e2(norm, by_ref, args.limit, rng)
        write_set("e2_collision", rows, qrels, out)
    print("next: make eval")


if __name__ == "__main__":
    main()
