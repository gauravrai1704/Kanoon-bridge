"""Human-annotation kit: sheets for a law student, and scoring once they are filled.  [owner: Gaurav — working]

    python scripts/22_annotation_kit.py make     # writes annotation/*.tsv (needs make build + the drift tables)
    python scripts/22_annotation_kit.py score    # reads the filled sheets, writes results/tables/annotation_*.csv

Sheets (guidelines: annotation/GUIDELINES.md). Columns starting with "LABEL_" are for the
annotator; everything else is context. System outputs are in "system_" columns, which the
annotator should read only AFTER labelling a row (or hide them: they are the last columns).

    drift.tsv       120 BNS sections, stratified by our drift class: IPC and BNS text side by side.
                    LABEL_changed (yes/no), LABEL_kind (none / cosmetic / punishment / elements /
                    structure / other), LABEL_precedent_use (0-3: can an IPC-era precedent on this
                    provision still be relied on for a BNS-era case?)
    treatment.tsv   every judgment our extractor flagged overruled / doubted, and 40 "followed"
                    mentions: LABEL_target_correct (yes/no: is the named case the one the sentence
                    is about?), LABEL_treatment (overruled / doubted / followed / other)
    answers.tsv     60 agent v2 + RAG v2 answers (40 punishment questions, the 20 hand-written
                    questions): LABEL_law_correct (yes / partly / no), LABEL_punishment_correct
                    (yes / no / n.a.), LABEL_misleading (yes / no)

Two annotators: copy a sheet to <sheet>.<name>.tsv per person; `score` reports Cohen's kappa
between them as well as agreement with the system.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict

from kanoon_bridge.config import load_config, project_path

ANN = "annotation"


def _write(name: str, rows: list[dict]) -> None:
    path = project_path(ANN) / f"{name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    keys = [k for k in keys if not k.startswith("system_")] + [k for k in keys if k.startswith("system_")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _cut(text: str, n: int = 1500) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def make_drift(cfg, per_class: int = 20, seed: int = 0) -> None:
    from kanoon_bridge.drift.provisions import load_bns, load_ipc

    bns, ipc = load_bns(cfg), load_ipc(cfg)
    with open(project_path("data/crosswalk/drift_sections.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r["new_section"]]
    by = defaultdict(list)
    for r in rows:
        by[r["drift_class"]].append(r)
    rng = random.Random(seed)
    out = []
    for cls in sorted(by):
        for r in rng.sample(by[cls], min(per_class, len(by[cls]))):
            olds = [s for s in r["old_sections"].replace(";", " ").split() if s]
            out.append({"id": f"D-{r['new_section']}", "bns_section": r["new_section"], "bns_title": r["title"],
                        "ipc_sections": " ".join(olds),
                        "ipc_text": _cut(" || ".join(ipc[s.split(':')[-1]].text for s in olds if s.split(':')[-1] in ipc)),
                        "bns_text": _cut(bns[r["new_section"]].text if r["new_section"] in bns else ""),
                        "LABEL_changed": "", "LABEL_kind": "", "LABEL_precedent_use": "", "LABEL_notes": "",
                        "system_class": r["drift_class"], "system_carry_over": r["carry_over"],
                        "system_reasons": r["reasons"]})
    rng.shuffle(out)
    _write("drift", out)


def make_treatment(cfg, positives: int = 40, seed: int = 0) -> None:
    from kanoon_bridge.rank import treatment as T
    from kanoon_bridge.schema import read_documents

    table = T.load(project_path(cfg.paths.index_dir) / "treatment.json")
    titles = {d.doc_id: d.title for d in read_documents(project_path(cfg.paths.docs))}
    out = []
    for t in table.values():
        for kind, evs in (("strong", t.strong), ("weak", t.weak)):
            for e in evs:
                out.append({"cited_case": titles.get(t.doc_id, t.doc_id), "citing_case": titles.get(e[0], e[0]),
                            "sentence": e[3], "system_cue": e[2], "system_status": t.status})
    pos = [(t, e) for t in table.values() for e in t.positive]
    for t, e in random.Random(seed).sample(pos, min(positives, len(pos))):
        out.append({"cited_case": titles.get(t.doc_id, t.doc_id), "citing_case": titles.get(e[0], e[0]),
                    "sentence": e[3], "system_cue": e[2], "system_status": "followed"})
    random.Random(seed).shuffle(out)
    for i, r in enumerate(out, start=1):
        r.update({"id": f"T-{i:03d}", "LABEL_target_correct": "", "LABEL_treatment": "", "LABEL_notes": ""})
        r.update({k: r.pop(k) for k in ("system_cue", "system_status")})
    _write("treatment", [{"id": r.pop("id"), **r} for r in out])


def make_answers(cfg, n_punishment: int = 40, seed: int = 0) -> None:
    from kanoon_bridge.agent.research import ResearchAgentV2
    from kanoon_bridge.eval import layers_eval as L
    from kanoon_bridge.eval.agent_eval import load_questions
    from kanoon_bridge.rag.answer import RagPipeline
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchEngine, SearchOptions

    engine = SearchEngine.load()
    rag = RagPipeline.load(engine)
    agent = ResearchAgentV2.load(engine, docs=rag.docs)
    qs = random.Random(seed).sample(L.punishment_questions(), n_punishment) + load_questions("data/queries/rag_questions.jsonl")
    out = []
    for row in qs:
        ans = rag.answer(agent.run(Query.from_dict(row), SearchOptions.interactive()), generator="extractive", version=2)
        out.append({"id": row["id"], "question": row["text"], "incident_date": row.get("incident_date", ""),
                    "answer": ans.text, "notes_shown_to_user": " | ".join(getattr(ans, "notes", []) or []),
                    "LABEL_law_correct": "", "LABEL_punishment_correct": "", "LABEL_misleading": "", "LABEL_notes": "",
                    "system_corrections": " | ".join(getattr(ans, "corrections", []) or [])})
    _write("answers", out)


# --------------------------------------------------------------------------- scoring


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def kappa(pairs: list[tuple[str, str]]) -> float:
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _norm(v: str) -> str:
    return (v or "").strip().lower()


def score() -> None:
    d = project_path(ANN)
    rows_out = []
    for sheet in sorted(d.glob("*.tsv")):
        rows = [r for r in _read(sheet) if any(_norm(v) for k, v in r.items() if k.startswith("LABEL_") and k != "LABEL_notes")]
        if not rows:
            continue
        base = sheet.name.split(".")[0]
        if base == "drift":
            sys_changed = lambda r: "no" if r["system_class"] in ("identical", "cosmetic") else "yes"  # noqa: E731
            pairs = [(_norm(r["LABEL_changed"]), sys_changed(r)) for r in rows]
            rows_out.append({"sheet": sheet.name, "what": "changed: annotator vs system", "n": len(pairs),
                             "agreement": round(sum(a == b for a, b in pairs if a) / max(1, sum(1 for a, _ in pairs if a)), 4),
                             "kappa": round(kappa(pairs), 4)})
            co = [(float(r["system_carry_over"]), int(r["LABEL_precedent_use"])) for r in rows
                  if r.get("LABEL_precedent_use", "").strip().isdigit() and r["system_carry_over"]]
            if len(co) > 2:
                import numpy as np

                rows_out.append({"sheet": sheet.name, "what": "carry-over vs precedent_use (Spearman)", "n": len(co),
                                 "agreement": "", "kappa": "",
                                 "spearman": round(float(np.corrcoef(np.argsort(np.argsort([a for a, _ in co])),
                                                                     np.argsort(np.argsort([b for _, b in co])))[0, 1]), 4)})
        elif base == "treatment":
            ok = [_norm(r["LABEL_target_correct"]) == "yes" and _norm(r["LABEL_treatment"]) ==
                  ("followed" if r["system_status"] == "followed" else r["system_status"]) for r in rows]
            for status in ("overruled", "doubted", "followed"):
                sel = [o for o, r in zip(ok, rows) if r["system_status"] == status]
                if sel:
                    rows_out.append({"sheet": sheet.name, "what": f"precision of '{status}'", "n": len(sel),
                                     "agreement": round(sum(sel) / len(sel), 4), "kappa": ""})
        elif base == "answers":
            for col in ("LABEL_law_correct", "LABEL_punishment_correct", "LABEL_misleading"):
                vals = Counter(_norm(r[col]) for r in rows if _norm(r[col]))
                rows_out.append({"sheet": sheet.name, "what": col.removeprefix("LABEL_"), "n": sum(vals.values()),
                                 "agreement": "", "kappa": "", "distribution": dict(vals)})
    # two annotators of the same sheet: <sheet>.<name>.tsv
    groups = defaultdict(list)
    for sheet in d.glob("*.*.tsv"):
        groups[sheet.name.split(".")[0]].append(sheet)
    for base, files in groups.items():
        if len(files) >= 2:
            a, b = (_read(f) for f in files[:2])
            bi = {r["id"]: r for r in b}
            for col in [c for c in a[0] if c.startswith("LABEL_") and c != "LABEL_notes"]:
                pairs = [(_norm(r[col]), _norm(bi[r["id"]][col])) for r in a if r["id"] in bi]
                rows_out.append({"sheet": f"{files[0].name} vs {files[1].name}", "what": col, "n": len(pairs),
                                 "agreement": "", "kappa": round(kappa(pairs), 4)})
    if not rows_out:
        print("no labelled rows yet")
        return
    out = project_path("results/tables") / "annotation_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows_out for k in r)))
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["make", "score"])
    ap.add_argument("--sheets", nargs="*", default=["drift", "treatment", "answers"])
    args = ap.parse_args()
    if args.what == "score":
        score()
        return
    cfg = load_config()
    if "drift" in args.sheets:
        make_drift(cfg)
    if "treatment" in args.sheets:
        make_treatment(cfg)
    if "answers" in args.sheets:
        make_answers(cfg)


if __name__ == "__main__":
    main()
