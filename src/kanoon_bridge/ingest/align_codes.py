"""Align the sections of a replaced code with its replacement by text similarity.  [owner: Gaurav — working]

CrPC (1973) -> BNSS (2023) and Indian Evidence Act (1872) -> BSA (2023) carry most sections
over almost word for word, renumbered. No open machine-readable correspondence table exists
(the government's BPR&D comparison PDFs can be cross-checked with --compare when downloaded),
so we derive one the IR way:

    1. both codes' sections -> title + body text (BNSS/BSA: the bare act; old codes: the Act)
    2. tf-idf vectors (word unigrams + bigrams, sublinear tf), cosine similarity of the bodies
       and of the titles: sim = 0.65 * body + 0.35 * title
    3. every new section takes its best old section if sim >= THRESHOLD; old sections within
       MERGE_RATIO of the best also map to it (merges: BNSS 35 <- CrPC 41 + 41A)
    4. reverse pass: each old section also joins its best new section (catches merges)
    5. relation = same (sim >= 0.8) | modified | merge | new (no old section clears the threshold)

Output: the same CSV shape as ipc_bns.csv (old_section, new_section, relation, note, title),
which ingest/parse_crosswalk.build_offence_ids turns into shared provision ids.
`spot_check()` scores the alignment against pairs known from the official tables.

    python -m kanoon_bridge.ingest.align_codes            # writes data/crosswalk/crpc_bnss.csv, iea_bsa.csv
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

THRESHOLD = 0.30
MERGE_RATIO = 0.85
SAME = 0.80

# Pairs from the government correspondence tables, used only to measure the alignment.
KNOWN = {
    ("crpc", "bnss"): [("41", "35"), ("41a", "35"), ("91", "94"), ("107", "126"), ("125", "144"), ("144", "163"),
                       ("154", "173"), ("156", "175"), ("161", "180"), ("164", "183"), ("167", "187"),
                       ("173", "193"), ("190", "210"), ("197", "218"), ("200", "223"), ("313", "351"),
                       ("319", "358"), ("320", "359"), ("397", "438"), ("401", "442"), ("436a", "479"),
                       ("437", "480"), ("438", "482"), ("439", "483"), ("482", "528")],
    ("iea", "bsa"): [("3", "2"), ("17", "15"), ("24", "22"), ("25", "23"), ("26", "23"), ("27", "23"),
                     ("32", "26"), ("45", "39"), ("60", "55"), ("61", "56"), ("63", "58"), ("65b", "63"),
                     ("101", "104"), ("106", "109"), ("113a", "117"), ("113b", "118"), ("114", "119"),
                     ("118", "124"), ("133", "138"), ("134", "139"), ("137", "142"), ("145", "148"),
                     ("154", "157"), ("157", "160"), ("165", "168")],
}

PAIRS = {  # old code -> (new code, old file, new file, old act name, new act name)
    "crpc": ("bnss", "crpc.json", "bnss_sections.json", "Code of Criminal Procedure, 1973",
             "Bharatiya Nagarik Suraksha Sanhita, 2023"),
    "iea": ("bsa", "iea.json", "bsa_sections.json", "Indian Evidence Act, 1872", "Bharatiya Sakshya Adhiniyam, 2023"),
}

_CONTEXT = re.compile(r"^\[Context:[^\]]*\]\s*", re.S)
_LEAD = re.compile(r"^\s*\d{1,3}[A-Z]?\.\s+[^\n]*?[—–-]{1,2}\s*")


@dataclass
class Section:
    number: str
    title: str
    text: str


def load_old(path: str | Path) -> list[Section]:
    """crpc.json / iea.json: [{section, section_title, section_desc}]."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out, seen = [], set()
    for r in rows:
        num = str(r.get("section", "")).strip().lower()
        if not num or num in seen:
            continue
        seen.add(num)
        out.append(Section(num, str(r.get("section_title", "")).strip(), str(r.get("section_desc", ""))))
    return out


def load_new(path: str | Path) -> list[Section]:
    """bnss_sections.json / bsa_sections.json: [{section_number, section_title, text}]."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out, seen = [], set()
    for r in rows:
        num = str(r.get("section_number", "")).strip().lower()
        if not num or num in seen:
            continue
        seen.add(num)
        body = _LEAD.sub("", _CONTEXT.sub("", r.get("text", "")), count=1)
        out.append(Section(num, str(r.get("section_title", "")).strip(), body))
    return out


def _vectors(texts: list[str]):
    """tf-idf rows (unigrams + bigrams, sublinear tf, l2-normalised) as a scipy CSR matrix."""
    import math

    import numpy as np
    from scipy import sparse

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    toks = [re.findall(r"[a-z]{2,}", t.lower()) for t in texts]
    for i, ws in enumerate(toks):
        grams = ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]
        counts: dict[int, int] = {}
        for g in grams:
            j = vocab.setdefault(g, len(vocab))
            counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(i)
            cols.append(j)
            vals.append(1 + math.log(c))
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), max(len(vocab), 1)))
    df = np.bincount(m.indices, minlength=m.shape[1])
    idf = np.log((1 + len(texts)) / (1 + df)) + 1
    m = m.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return sparse.diags(1 / norms) @ m


def similarity(old: list[Section], new: list[Section]):
    """[len(new) x len(old)] combined cosine similarity."""
    n = len(new)
    body = _vectors([s.text for s in new] + [s.text for s in old])
    title = _vectors([s.title for s in new] + [s.title for s in old])
    sb = (body[:n] @ body[n:].T).toarray()
    st = (title[:n] @ title[n:].T).toarray()
    return 0.65 * sb + 0.35 * st


def align(old: list[Section], new: list[Section], threshold: float = THRESHOLD,
          merge_ratio: float = MERGE_RATIO) -> list[dict]:
    """Crosswalk rows: one per (old, new) pair, plus 'new' rows for unmatched new sections."""
    sim = similarity(old, new)
    rows = []
    for i, s in enumerate(new):
        order = sim[i].argsort()[::-1]
        best = float(sim[i, order[0]]) if len(order) else 0.0
        if best < threshold:
            rows.append({"old": "", "new": s.number, "relation": "new", "sim": round(best, 3), "title": s.title})
            continue
        for j in order[:4]:
            v = float(sim[i, j])
            if j != order[0] and (v < threshold or v < merge_ratio * best):
                break
            rows.append({"old": old[j].number, "new": s.number, "relation": "same" if v >= SAME else "modified",
                         "sim": round(v, 3), "title": s.title})
    # reverse pass: an old section whose best match is a new section it was merged into
    # (BNSS 35 absorbs CrPC 41 and 41A; BSA 23 absorbs IEA 25-27) - the forward pass keeps
    # only one old section per new one unless their scores are close
    have = {(r["old"], r["new"]) for r in rows}
    for j, o in enumerate(old):
        i = int(sim[:, j].argmax())
        v = float(sim[i, j])
        if v >= threshold and (o.number, new[i].number) not in have:
            rows.append({"old": o.number, "new": new[i].number, "relation": "merge", "sim": round(v, 3),
                         "title": new[i].title})
    rows.sort(key=lambda r: (_key(r["new"]), _key(r["old"])))
    return rows


def _key(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)", num or "")
    return (int(m.group(1)), m.group(2)) if m else (10**6, num)


def spot_check(rows: list[dict], known: list[tuple[str, str]]) -> dict:
    pairs = {(r["old"], r["new"]) for r in rows if r["old"]}
    hit = [k for k in known if k in pairs]
    return {"known": len(known), "found": len(hit), "accuracy": round(len(hit) / len(known), 3) if known else 0.0,
            "missed": [k for k in known if k not in pairs]}


def write(rows: list[dict], path: Path, old_code: str, new_code: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {old_code.upper()} -> {new_code.upper()} correspondence, aligned by text similarity "
                f"(ingest/align_codes.py) from {source}\n")
        w = csv.writer(f)
        w.writerow([f"{old_code}_section", f"{new_code}_section", "relation", "note", "title"])
        for r in rows:
            w.writerow([r["old"], r["new"], r["relation"], f"sim={r['sim']}", r["title"]])


def build_provision_ids(rows: list[dict], old_code: str, new_code: str) -> list[dict]:
    """Shared ids for aligned sections: connected components of the old<->new pair graph, named
    after the lowest new section ('off:bnss482_direction_grant_bail_person_apprehending')."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    titles = {}
    for r in rows:
        n = f"{new_code}:{r['new']}"
        find(n)
        titles[r["new"]] = r["title"]
        if r["old"]:
            parent[find(f"{old_code}:{r['old']}")] = find(n)
    groups: dict[str, list[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    out = []
    for members in groups.values():
        olds = sorted((m.split(":", 1)[1] for m in members if m.startswith(old_code + ":")), key=_key)
        news = sorted((m.split(":", 1)[1] for m in members if m.startswith(new_code + ":")), key=_key)
        if not news:
            continue
        label = titles.get(news[0], "")
        slug = "_".join(re.findall(r"[a-z]+", label.lower()))[:48].strip("_")
        out.append({"offence_id": f"off:{new_code}{news[0]}_{slug}", "label": label, "old_code": old_code,
                    "new_code": new_code, "old_sections": ";".join(olds), "new_sections": ";".join(news)})
    return sorted(out, key=lambda r: (r["new_code"], _key(r["new_sections"].split(";")[0])))


def write_provision_ids(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# Shared ids for CrPC<->BNSS and IEA<->BSA sections (ingest/align_codes.py)\n")
        w = csv.DictWriter(f, fieldnames=["offence_id", "label", "old_code", "new_code", "old_sections", "new_sections"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import argparse

    from kanoon_bridge.config import load_config, project_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="indian-legal-mcp checkout (default: paths.procedure_dir)")
    args = ap.parse_args()
    cfg = load_config()
    root = Path(args.source or project_path(cfg.paths.procedure_dir)) / "data" / "raw"
    ids: list[dict] = []
    for old_code, (new_code, old_file, new_file, _, _) in PAIRS.items():
        old, new = load_old(root / old_file), load_new(root / new_file)
        rows = align(old, new)
        out = project_path(cfg.paths[f"{old_code}_{new_code}"])
        write(rows, out, old_code, new_code, "GSMS-B/indian-legal-mcp (MIT)")
        chk = spot_check(rows, KNOWN[(old_code, new_code)])
        n_pairs = sum(1 for r in rows if r["old"])
        print(f"{old_code}->{new_code}: {len(old)} old, {len(new)} new sections, {n_pairs} pairs, "
              f"{sum(1 for r in rows if r['relation'] == 'new')} new; known pairs found {chk['found']}/{chk['known']} "
              f"({chk['accuracy']:.0%}); missed {chk['missed']} -> {out}")
        ids += build_provision_ids(rows, old_code, new_code)
    write_provision_ids(ids, project_path(cfg.paths.provision_ids))
    print(f"provision ids: {len(ids)} -> {project_path(cfg.paths.provision_ids)}")


if __name__ == "__main__":
    main()
