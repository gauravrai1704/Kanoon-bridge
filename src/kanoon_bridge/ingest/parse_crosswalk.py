"""Build data/crosswalk/ipc_bns.csv and offence_ids.csv.  [owner: Gaurav — working]

Primary source: the `ipcReference` of every BNS section in data/raw/bns-study-platform
(358 sections; each names the IPC section(s) it corresponds to, e.g. BNS 103 -> "302 IPC",
BNS 351 -> "503, 506, 507 IPC", BNS 127 -> "340, 342-348 IPC", BNS 111 -> "No IPC equivalent").
Second source (optional cross-check): the government comparison PDF, parsed best-effort with
pdfplumber; `compare_sources()` reports where the two disagree so they can be hand-checked.

Outputs (formats in data/crosswalk/README.md)
    ipc_bns.csv       one row per (IPC, BNS) pair: ipc_section, bns_section, relation, note
    offence_ids.csv   canonical offences = connected groups of IPC and BNS sections that map
                      to each other; id from the BNS heading ("Punishment for murder" -> off:murder)

    python -m kanoon_bridge.ingest.parse_crosswalk            # writes both CSVs
    python -m kanoon_bridge.ingest.parse_crosswalk --compare  # also cross-check against the PDF
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.ingest.load_statutes import iter_section_records

STOP = {"of", "the", "and", "or", "by", "to", "in", "for", "a", "an", "with", "on", "etc", "any", "who", "is",
        "certain", "cases", "case", "person", "persons", "punishment", "offence", "offences", "under", "from",
        "being", "not", "as", "at", "which", "such", "other", "be", "his", "her", "their", "same", "act", "when",
        "that", "there", "than", "are", "its", "may", "made", "done", "one", "but", "without", "where", "himself"}

_NUM = r"\d{1,3}[a-z]?"


# --------------------------------------------------------------------------- IPC reference strings


def _expand_range(a: str, b: str) -> list[str]:
    """'342','348' -> 342..348; '6','52a' -> 6..52 plus 52a; '171','i' -> 171i."""
    if re.fullmatch(r"[a-z]{1,2}", b):                    # "171-I" is section 171I
        return [a + b]
    ma, mb = re.match(r"(\d+)([a-z]?)", a), re.match(r"(\d+)([a-z]?)", b)
    lo, hi = int(ma.group(1)), int(mb.group(1))
    if hi < lo or hi - lo > 80:
        return [a, b]
    out = [str(n) for n in range(lo, hi + 1)]
    if ma.group(2):
        out[0] = a
    if mb.group(2):
        out.append(b)
    return out


def parse_ipc_reference(text: str) -> tuple[list[str], str]:
    """'340, 342-348 IPC' -> (['340','342',...,'348'], ''); 'No IPC equivalent' -> ([], 'new').

    Sub-sections are kept ('376(1)'); a bare '(2)' after '228A(1)' becomes '228a(2)'.
    The second value is a short status: '' | 'new' | 'not retained' | 'explanation' | 'repeal'.
    """
    low = (text or "").lower().strip()
    if not low or low.startswith("none") or "no ipc equivalent" in low:
        status = "repeal" if "repeal" in low else "new"
        return [], status
    status = "not retained" if "not retained" in low else "explanation" if "explanation" in low else ""
    low = re.sub(r"\((?!\s*\d+\s*\))[^)]*\)", " ", low)           # drop notes like (explanation), keep (1)
    if re.fullmatch(r"\s*new(\s+(section|provision|offence))?\s*\.?\s*", low) or low.startswith("new section"):
        return [], "new"
    low = re.sub(r"\bipc\b|\bchapter\s+[ivxl]+\b|\bss?\.|\bsections?\b|\bsecs?\.?", " ", low)
    out: list[str] = []
    base = None
    for part in re.split(r",|\band\b|&", low):
        part = part.strip().replace(" ", "")
        if not part:
            continue
        if re.fullmatch(r"\(\d+\)", part) and base:              # "(2)" continues the previous section
            out.append(base + part)
            continue
        m = re.fullmatch(rf"({_NUM})-({_NUM}|[a-z]{{1,2}})", part)
        if m:
            out += _expand_range(m.group(1), m.group(2))
            base = out[-1]
            continue
        m = re.fullmatch(rf"({_NUM})((?:\(\d+\))*)", part)
        if m:
            out.append(m.group(1) + m.group(2))
            base = m.group(1)
    return out, status


# --------------------------------------------------------------------------- rows from the JSON source


def extract_rows_from_bns_json(root: str | Path) -> list[dict]:
    """One row per (IPC, BNS) pair, plus 'new' rows for BNS sections with no IPC counterpart."""
    pairs: list[dict] = []
    for rec in iter_section_records(root):
        bns = str(rec["number"]).strip().lower()
        ref = rec.get("ipcReference") or ""
        ref = ref.get("section", "") if isinstance(ref, dict) else str(ref)
        ipcs, status = parse_ipc_reference(ref)
        title = (rec.get("title") or "").strip()
        if not ipcs:
            pairs.append({"ipc_section": "", "bns_section": bns, "relation": status or "new", "note": "", "title": title})
        for ipc in ipcs:
            pairs.append({"ipc_section": ipc, "bns_section": bns, "relation": "", "note": status, "title": title})
    _set_relations(pairs)
    return pairs


def _set_relations(rows: list[dict]) -> None:
    per_ipc: dict[str, set[str]] = defaultdict(set)
    per_bns: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["ipc_section"]:
            per_ipc[_base(r["ipc_section"])].add(r["bns_section"])
            per_bns[r["bns_section"]].add(_base(r["ipc_section"]))
    for r in rows:
        if not r["ipc_section"]:
            continue
        split = len(per_ipc[_base(r["ipc_section"])]) > 1
        merge = len(per_bns[r["bns_section"]]) > 1
        r["relation"] = "split" if split and not merge else "merge" if merge and not split else "split+merge" if split else "same"
        if r["note"] == "not retained":
            r["relation"] = "modified"


def _base(section: str) -> str:
    return re.sub(r"\(.*", "", section)


# --------------------------------------------------------------------------- PDF source (cross-check)


_BNS_HEAD = ("bns", "nyaya sanhita", "bharatiya")
_IPC_HEAD = ("ipc", "penal code", "i.p.c")
_BNS_CELL = re.compile(rf"^(?:section\s*|sec\.?\s*|s\.\s*)?({_NUM}(?:\(\d+\))?)", re.I)


def _find_columns(cells: list[str]) -> tuple[int | None, int | None]:
    """Header row -> (BNS column, IPC column). A 'section(s)' column wins over a 'subject' one."""
    bns_col = ipc_col = None
    for i, c in enumerate(cells):
        c = " ".join(c.split())
        if bns_col is None and any(h in c for h in _BNS_HEAD) and "summary" not in c:
            bns_col = i
        elif ipc_col is None and any(h in c for h in _IPC_HEAD) and "summary" not in c:
            ipc_col = i
    return bns_col, ipc_col


def extract_rows_from_pdf(pdf_path: str | Path, debug: bool = False) -> list[dict]:
    """Best-effort rows from the government comparison PDF (BPR&D / MHA comparison summary).

    Table mode (pdfplumber.extract_tables): a header row naming the BNS and IPC columns
    ("BNS", "Bharatiya Nyaya Sanhita", "IPC", "Indian Penal Code") fixes the columns; they carry
    over to continuation pages without a header. Cells may hold several lines ("103\n(1)").
    Text mode (no ruled tables found): a line starting with a BNS number whose remainder
    contains an IPC reference ("... 302 IPC" / "Section 302").
    Validate with compare_sources() before trusting it; `debug=True` prints what was seen.
    """
    import pdfplumber

    rows: list[dict] = []
    stats = {"pages": 0, "tables": 0, "table_rows": 0, "text_rows": 0}
    bns_col = ipc_col = None
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = []
        for page in pdf.pages:
            stats["pages"] += 1
            tables = page.extract_tables() or []
            stats["tables"] += len(tables)
            for table in tables:
                if debug and stats["tables"] <= 2:
                    print("table sample:", table[:4])
                for raw in table:
                    cells = [" ".join((c or "").split()).lower() for c in raw]
                    b, i = _find_columns(cells)
                    if b is not None and i is not None:
                        bns_col, ipc_col = b, i
                        continue
                    if bns_col is None or max(bns_col, ipc_col) >= len(cells):
                        continue
                    m = _BNS_CELL.match(cells[bns_col].replace(" (", "("))
                    ipcs, _ = parse_ipc_reference(cells[ipc_col])
                    if m:
                        stats["table_rows"] += 1
                        for ipc in ipcs:
                            rows.append({"ipc_section": ipc, "bns_section": m.group(1), "relation": "", "note": "pdf"})
            if not tables:
                texts.append(page.extract_text() or "")
        if not rows:                                   # text-mode fallback
            line_re = re.compile(rf"^\s*(?:\d+\.\s+)?({_NUM}(?:\(\d+\))?)\s+(.+?(?:ipc|section)[^\n]*)$", re.I)
            for line in "\n".join(texts).splitlines():
                m = line_re.match(line)
                if not m:
                    continue
                ref = re.findall(rf"(?:section|sec\.?|s\.)\s*({_NUM}(?:\(\d+\))?(?:\s*(?:,|and|to|-)\s*{_NUM})*)|"
                                 rf"({_NUM}(?:\(\d+\))?(?:\s*(?:,|and|to|-)\s*{_NUM})*)\s*(?:of\s+the\s+)?ipc", m.group(2), re.I)
                for a, b in ref:
                    ipcs, _ = parse_ipc_reference(a or b)
                    stats["text_rows"] += 1
                    for ipc in ipcs:
                        rows.append({"ipc_section": ipc, "bns_section": m.group(1).lower(), "relation": "", "note": "pdf-text"})
    if debug:
        print("pdf:", stats, "pairs:", len(rows), "columns (bns, ipc):", (bns_col, ipc_col))
    _set_relations(rows)
    return rows


def compare_sources(json_rows: list[dict], pdf_rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Pairs found in only one source, keyed 'only_json' / 'only_pdf' (base sections)."""
    a = {(_base(r["ipc_section"]), _base(r["bns_section"])) for r in json_rows if r["ipc_section"]}
    b = {(_base(r["ipc_section"]), _base(r["bns_section"])) for r in pdf_rows if r["ipc_section"]}
    return {"only_json": sorted(a - b), "only_pdf": sorted(b - a)}


# --------------------------------------------------------------------------- writing


def write_crosswalk(rows: list[dict], out_path: str | Path) -> None:
    """ipc_bns.csv, sorted by IPC section (new BNS-only sections last)."""
    def key(r):
        m = re.match(r"(\d+)", r["ipc_section"])
        return (0, int(m.group(1)), r["ipc_section"]) if m else (1, int(re.match(r"(\d+)", r["bns_section"]).group(1)), "")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# Generated by ingest/parse_crosswalk.py from BNS section correspondences (bns-study-platform)\n")
        f.write("# Spot-check against the government comparison table before relying on a row.\n")
        w = csv.DictWriter(f, fieldnames=["ipc_section", "bns_section", "relation", "note"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=key))


def _slug(title: str) -> str:
    """'Punishment for murder' -> 'murder'; 'Husband or relative of husband of a woman subjecting
    her to cruelty' -> 'husband_relative_husband_woman_subjecting'."""
    t = re.sub(r"^(punishment\s+(for|of)\s+)", "", title.lower())
    words = [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP or w == "murder"]
    return "_".join(words[:5]) or "offence"


def _keywords(titles: list[str]) -> list[str]:
    words: list[str] = []
    for t in titles:
        for w in re.findall(r"[a-z]+", t.lower()):
            if len(w) > 2 and w not in STOP and w not in words:
                words.append(w)
    return words[:12]


def build_offences(rows: list[dict]) -> list[dict]:
    """Group sections into canonical offences: connected components of the IPC-BNS mapping graph."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    titles: dict[str, str] = {}
    for r in rows:
        b = "bns:" + r["bns_section"]
        find(b)
        titles[r["bns_section"]] = r.get("title", "")
        if r["ipc_section"]:
            union("ipc:" + _base(r["ipc_section"]), b)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in list(parent):
        groups[find(node)].append(node)

    def num_key(s):
        return (int(re.match(r"\d+", s).group()), s)

    comps = []
    for members in groups.values():
        bns = sorted({m[4:] for m in members if m.startswith("bns:")}, key=num_key)
        ipc = sorted({m[4:] for m in members if m.startswith("ipc:")}, key=num_key)
        if bns:
            comps.append((bns, ipc, set(members)))
    comps.sort(key=lambda c: num_key(c[0][0]))          # deterministic ids: lowest BNS section first

    out, used = [], set()
    for bns, ipc, members in comps:
        label = titles.get(bns[0], "") or f"BNS {bns[0]}"
        oid = "off:" + _slug(label)
        if oid in used:
            oid = f"{oid}_bns{bns[0]}"
        used.add(oid)
        # keep sub-section forms too, so 'ipc:376(1)' and 'ipc:376' both resolve
        ipc_full = sorted({r["ipc_section"] for r in rows if r["ipc_section"] and "ipc:" + _base(r["ipc_section"]) in members} | set(ipc))
        out.append({"offence_id": oid, "label": label, "ipc_sections": ";".join(ipc_full),
                    "bns_sections": ";".join(bns), "keywords": ";".join(_keywords([titles.get(b, "") for b in bns]))})
    return sorted(out, key=lambda o: int(re.match(r"\d+", o["bns_sections"]).group()))


def build_offence_ids(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# Generated by ingest/parse_crosswalk.py: connected IPC<->BNS section groups, named by BNS heading\n")
        w = csv.DictWriter(f, fieldnames=["offence_id", "label", "ipc_sections", "bns_sections", "keywords"])
        w.writeheader()
        w.writerows(build_offences(rows))


# Kept for the stub API name used elsewhere.
def extract_rows(source: str | Path) -> list[dict]:
    """Rows from either source: the bns-study-platform directory, or a PDF."""
    source = Path(source)
    return extract_rows_from_pdf(source) if source.suffix == ".pdf" else extract_rows_from_bns_json(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="bns-study-platform dir (default from config) or a PDF")
    ap.add_argument("--compare", action="store_true", help="cross-check against the government PDF")
    ap.add_argument("--pdf", help="PDF path for --compare (default: paths.crosswalk_pdf)")
    ap.add_argument("--debug", action="store_true", help="print what the PDF parser sees")
    args = ap.parse_args()
    cfg = load_config()
    rows = extract_rows(args.source or project_path(cfg.paths.bns_dir))
    write_crosswalk(rows, project_path(cfg.paths.crosswalk))
    build_offence_ids(rows, project_path(cfg.paths.offence_ids))
    n_pairs = sum(1 for r in rows if r["ipc_section"])
    print(f"crosswalk: {n_pairs} IPC-BNS pairs, {sum(1 for r in rows if not r['ipc_section'])} new BNS sections")
    if args.compare:
        pdf = Path(args.pdf) if args.pdf else project_path(cfg.paths.crosswalk_pdf)
        if not pdf.exists():
            raise SystemExit(f"PDF not found: {pdf}\nDownload it by hand (see SETUP_AND_RUN.md, step 4).")
        pdf_rows = extract_rows_from_pdf(pdf, debug=args.debug)
        diff = compare_sources(rows, pdf_rows)
        both = {(_base(r["ipc_section"]), _base(r["bns_section"])) for r in rows if r["ipc_section"]}
        agree = len(both) - len(diff["only_json"])
        print(f"PDF pairs: {len({(_base(r['ipc_section']), _base(r['bns_section'])) for r in pdf_rows})}  "
              f"agree: {agree}/{len(both)} JSON pairs  only in JSON: {len(diff['only_json'])}  "
              f"only in PDF: {len(diff['only_pdf'])}")
        out = project_path("data/processed/crosswalk_compare.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "ipc_section", "bns_section"])
            for k, v in diff.items():
                w.writerows([k, a, b] for a, b in v)
        print(f"disagreements written to {out} (hand-check each; fix the JSON-derived row in ipc_bns.csv if the PDF is right)")
        if not pdf_rows:
            print("No rows read from the PDF: rerun with --debug to see what pdfplumber found.")


if __name__ == "__main__":
    main()
