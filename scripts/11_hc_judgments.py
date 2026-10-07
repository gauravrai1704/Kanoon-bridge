"""Post-2024 High Court judgments: fetch, measure the IPC -> BNS citation shift, add to the corpus.  [owner: Gaurav]

Source: the Indian High Court Judgments open dataset (eCourts, CC-BY-4.0), public S3 bucket
`indian-high-court-judgments` (ap-south-1), read over plain HTTPS - no AWS account needed.
https://registry.opendata.aws/indian-high-court-judgments/   docs: github.com/vanga/indian-high-court-judgments

    python scripts/11_hc_judgments.py fetch --years 2023 2024 2025 --per-month 40     # Colab: ~30-60 min
    python scripts/11_hc_judgments.py shift      # monthly citation shares -> results/tables/citation_shift.csv + figure
    python scripts/11_hc_judgments.py ingest     # -> data/processed/hc_docs.jsonl (01_build_corpus.py adds them)
    python scripts/11_hc_judgments.py testset    # -> E10: IPC-worded queries vs BNS-era judgments

Steps of `fetch`, per court bench and year:
    metadata/parquet/year=Y/court=C/bench=B/metadata.parquet       (title, description, decision_date, pdf_link, cnr)
    keep criminal matters (case-type words in title/description: CRL, BAIL, CRM, CRR, CRA, ...)
    sample up to --per-month judgments per month (seeded), download each PDF:
    data/pdf/year=Y/court=C/bench=B/<pdf file name>
    text with pypdf -> data/raw/hc/judgments.jsonl  {id, court, state, date, title, text}

`shift` (the measurement): for every judgment, the sections it cites per code (our section-aware
tokenizer: "u/s 302 IPC", "Section 103 of the BNS", "BNSS 482", "Section 63 BSA"...). Per month:
    share of judgments citing the new code, share citing the old code, share citing both,
    and the new code's share of all section mentions - for IPC/BNS, CrPC/BNSS and IEA/BSA.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

BUCKET = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
# (court code, bench, court name as our court table knows it, state slug)
COURTS = [
    ("7_26", "dhcdb", "Delhi High Court", "delhi"),
    ("3_22", "phhc", "Punjab and Haryana High Court", "punjab"),
    ("9_13", "cishclko", "Allahabad High Court", "uttar-pradesh"),
    ("32_4", "highcourtofkerala", "Kerala High Court", "kerala"),
    ("24_17", "gujarathc", "Gujarat High Court", "gujarat"),
]
CRIMINAL = re.compile(r"\b(CRL|CRIMINAL|BAIL|CRM|CRR|CRA|CR\.?\s?A|CR\.?\s?R|CR\.?\s?M|ABA|BA|CRMP|CRLMC|CRLA|CRLP|CRLR)\b", re.I)
FAMILIES = (("ipc", "bns"), ("crpc", "bnss"), ("iea", "bsa"))


def _get(url: str, timeout: int = 60) -> bytes | None:
    import requests

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
        except Exception:                                          # noqa: BLE001 - retry, then give up
            time.sleep(2 * (attempt + 1))
    return None


def pdf_text(data: bytes, max_pages: int = 40) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
    except Exception:                                              # noqa: BLE001 - a broken PDF is skipped
        return ""


def fetch(years: list[int], per_month: int, out: Path, seed: int = 7) -> None:
    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        seen = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    rng = random.Random(seed)
    n_new = 0
    with open(out, "a", encoding="utf-8") as f:
        for court, bench, cname, state in COURTS:
            for year in years:
                url = f"{BUCKET}/metadata/parquet/year={year}/court={court}/bench={bench}/metadata.parquet"
                blob = _get(url, timeout=300)
                if blob is None:
                    print(f"{cname} {year}: no metadata at {url}")
                    continue
                df = pd.read_parquet(io.BytesIO(blob))
                text_cols = [c for c in ("title", "description") if c in df.columns]
                crim = df[df[text_cols].astype(str).agg(" ".join, axis=1).str.contains(CRIMINAL, na=False)].copy()
                crim["decision_date"] = pd.to_datetime(crim["decision_date"], errors="coerce")
                crim = crim.dropna(subset=["decision_date", "pdf_link"])
                crim["month"] = crim["decision_date"].dt.strftime("%Y-%m")
                print(f"{cname} {year}: {len(df)} judgments, {len(crim)} criminal")
                for month, grp in sorted(crim.groupby("month")):
                    rows = grp.to_dict("records")
                    rng.shuffle(rows)
                    got = 0
                    for r in rows:
                        if got >= per_month:
                            break
                        name = Path(str(r["pdf_link"])).name
                        jid = f"{court}:{Path(name).stem}"
                        if jid in seen:
                            got += 1
                            continue
                        data = _get(f"{BUCKET}/data/pdf/year={year}/court={court}/bench={bench}/{name}")
                        text = pdf_text(data) if data else ""
                        if len(text.split()) < 150:                 # orders of a few lines say nothing
                            continue
                        f.write(json.dumps({"id": jid, "court": cname, "state": state,
                                            "date": r["decision_date"].date().isoformat(),
                                            "title": str(r.get("title") or "").strip()[:300],
                                            "cnr": str(r.get("cnr") or ""), "text": text}, ensure_ascii=False) + "\n")
                        seen.add(jid)
                        got += 1
                        n_new += 1
                    f.flush()
                    print(f"  {month}: {got} judgments")
    print(f"fetched {n_new} new judgments -> {out}")


# --------------------------------------------------------------------------- the measurement


def cited_codes(text: str) -> Counter:
    """Section mentions per code in one judgment (codes the text names; bare numbers ignored)."""
    from kanoon_bridge.text.tokenize import extract_sections

    c = Counter()
    for m in extract_sections(text.lower()):
        if m.code.value in ("ipc", "bns", "crpc", "bnss", "iea", "bsa"):
            c[m.code.value] += 1
    return c


def shift_table(judgments: list[dict]) -> list[dict]:
    by_month: dict[str, list[Counter]] = defaultdict(list)
    for j in judgments:
        by_month[j["date"][:7]].append(cited_codes(j["text"]))
    rows = []
    for month in sorted(by_month):
        cs = by_month[month]
        row = {"month": month, "judgments": len(cs)}
        for old, new in FAMILIES:
            any_ = [c for c in cs if c[old] or c[new]]
            row[f"{new}_share_of_judgments"] = round(sum(1 for c in cs if c[new]) / len(cs), 4)
            row[f"{old}_share_of_judgments"] = round(sum(1 for c in cs if c[old]) / len(cs), 4)
            row[f"{old}_and_{new}"] = round(sum(1 for c in cs if c[old] and c[new]) / len(cs), 4)
            mentions = sum(c[old] + c[new] for c in cs)
            row[f"{new}_share_of_mentions"] = round(sum(c[new] for c in cs) / mentions, 4) if mentions else ""
            row[f"{old}_{new}_citing"] = len(any_)
        rows.append(row)
    return rows


def plot_shift(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = [r["month"] for r in rows]
    x = list(range(len(months)))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    styles = {("ipc", "bns"): ("#2563c9", "BNS share of IPC+BNS section mentions"),
              ("crpc", "bnss"): ("#6b7280", "BNSS share of CrPC+BNSS mentions"),
              ("iea", "bsa"): ("#b45309", "BSA share of IEA+BSA mentions")}
    for (old, new), (color, label) in styles.items():
        ys = [r[f"{new}_share_of_mentions"] for r in rows]
        pts = [(i, y) for i, y in zip(x, ys) if y != ""]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2 if new == "bns" else 1.5,
                    marker="o", ms=3.5, label=label)
    if "2024-07" in months:
        k = months.index("2024-07")
        ax.axvline(k, color="#9ca3af", lw=1, ls="--")
        ax.text(k + 0.2, 0.95, "BNS in force\n1 Jul 2024", fontsize=8.5, color="#4b5563", va="top")
    ax.set_xticks(x[::3], months[::3], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of mentions", color="#5d6375")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    last = next((r for r in reversed(rows) if r["bns_share_of_mentions"] != ""), None)
    title = "High Court criminal judgments: how fast citations move from IPC to BNS"
    if last:
        title = f"By {last['month']}, BNS is {float(last['bns_share_of_mentions']):.0%} of IPC/BNS section citations in High Court judgments"
    fig.suptitle(title, x=0.02, ha="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def shift(src: Path) -> None:
    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.eval.ablation import write_table

    judgments = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    rows = shift_table(judgments)
    ev = load_config("eval.yaml")
    tables, figs = project_path(ev.outputs.tables), project_path(ev.outputs.figures)
    write_table(rows, tables / "citation_shift.csv")
    plot_shift(rows, figs / "citation_shift.png")
    fig_dir = project_path("docs/report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_shift(rows, fig_dir / "fig_citation_shift.png")
    for r in rows:
        print(f"  {r['month']}  n={r['judgments']:4d}  BNS share of mentions {r['bns_share_of_mentions']}  "
              f"cites BNS {r['bns_share_of_judgments']:.2f}  cites IPC {r['ipc_share_of_judgments']:.2f}")
    print("written", tables / "citation_shift.csv", figs / "citation_shift.png")


# --------------------------------------------------------------------------- corpus + test set


def to_documents(judgments: list[dict]):
    from kanoon_bridge.ingest.segment import segment_judgment
    from kanoon_bridge.schema import Document, DocType

    docs = []
    for j in judgments:
        paras = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\n(?=\d+\.)", j["text"]) if len(p.split()) >= 8]
        if not paras:
            continue
        from kanoon_bridge.text.tokenize import extract_sections

        sections = cited_codes(j["text"])
        refs = sorted({f"{m.code.value}:{m.section}" for m in extract_sections(j["text"].lower())
                       if m.code.value not in ("?", "other")})
        doc = Document(doc_id=f"hc:{j['id']}", doc_type=DocType.PRECEDENT, title=j.get("title") or "",
                       paragraphs=segment_judgment(paras, None), decision_date=date.fromisoformat(j["date"]),
                       statutes_cited=refs,
                       meta={"jurisdiction": j["court"], "source": "hc-open-data", "cnr": j.get("cnr", ""),
                             "codes_cited": dict(sections)})
        docs.append(doc)
    return docs


def ingest(src: Path) -> None:
    from kanoon_bridge.config import project_path
    from kanoon_bridge.schema import write_documents

    judgments = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    docs = to_documents(judgments)
    out = project_path("data/processed/hc_docs.jsonl")
    write_documents(docs, out)
    print(f"{len(docs)} High Court judgments -> {out}; rebuild with: make data index graph")


def testset(src: Path, min_cases: int = 3) -> None:
    """E10: IPC-numbered queries, BNS-era judgments. For each IPC section with a BNS equivalent,
    'cases under section 302 IPC' must find the post-July-2024 High Court judgments that cite
    BNS 103 (and not IPC 302). Gold comes from the BNS sections each judgment cites."""
    import csv

    from kanoon_bridge.config import project_path
    from kanoon_bridge.text.tokenize import extract_sections
    from kanoon_bridge.text.version_norm import VersionNormalizer

    norm = VersionNormalizer.load()
    judgments = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    cites: dict[str, set[str]] = defaultdict(set)              # bns section -> judgments citing it
    for j in judgments:
        if j["date"] < "2024-07-01":
            continue
        for m in extract_sections(j["text"].lower()):
            if m.code.value == "bns":
                cites[m.section.split("(")[0]].add(f"hc:{j['id']}")
    rows, qrels = [], []
    for bns, docs in sorted(cites.items(), key=lambda kv: -len(kv[1])):
        ipc = [e for e in norm.equivalents(f"bns:{bns}") if e.startswith("ipc:")]
        if len(docs) < min_cases or not ipc:
            continue
        num = ipc[0].split(":", 1)[1]
        qid = f"E10-{num}"
        rows.append({"id": qid, "text": f"cases under section {num} IPC", "lang": "en", "state": None,
                     "bns": f"bns:{bns}", "ipc": ipc[0], "generated": True})
        qrels += [(qid, d, 1) for d in sorted(docs)]
    qdir = project_path("data/queries")
    with open(qdir / "e10_bns_era.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(qdir / "qrels" / "e10_bns_era.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "doc_id", "grade", "judge"])
        w.writerows([q, d, g, "generated"] for q, d, g in qrels)
    print(f"E10: {len(rows)} queries, {len(qrels)} judgments")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kanoon_bridge.config import project_path

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "shift", "ingest", "testset"])
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--per-month", type=int, default=40)
    ap.add_argument("--src", default="data/raw/hc/judgments.jsonl")
    args = ap.parse_args()
    src = project_path(args.src)
    if args.step == "fetch":
        fetch(args.years, args.per_month, src)
    elif args.step == "shift":
        shift(src)
    elif args.step == "ingest":
        ingest(src)
    else:
        testset(src)


if __name__ == "__main__":
    main()
