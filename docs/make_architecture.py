"""Generate docs/architecture.svg (and .png for the report).

    python docs/make_architecture.py          # needs: pip install cairosvg  (PNG only)

Edit the content lists below and re-run; layout is computed, so boxes stay aligned.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W = 1500
OUT = Path(__file__).resolve().parent

C = {  # fill, stroke per band
    "data": ("#f4f4f2", "#8a8a85"),
    "split": ("#fff7e0", "#c99a1e"),
    "offline": ("#eaf1fb", "#3d6fb6"),
    "artifact": ("#f7f7f7", "#9a9a9a"),
    "l1": ("#e6f5f1", "#1f8a70"),
    "l2": ("#f1ebfa", "#6b48b5"),
    "l3": ("#fdf0e6", "#c4651b"),
    "out": ("#f4f4f2", "#8a8a85"),
    "eval": ("#ecf6e8", "#3f8a2c"),
}
INK, QUIET = "#1d1d1b", "#55554f"
CHAR_W = 0.6  # DejaVu Sans width per font-size unit (worst case)

svg: list[str] = []
warnings: list[str] = []


def text(x, y, s, size=13, weight="normal", fill=INK, anchor="start", style=""):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}"{style}>{escape(s)}</text>')


def check_fit(s, size, width, where):
    if len(s) * size * CHAR_W > width:
        warnings.append(f"too long ({where}): {s!r}")


def box(x, y, w, h, title, lines, band, owner="", title_size=15, size=13, cols=None):
    fill, stroke = C[band]
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="{stroke}" stroke-width="1.4"/>')
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="8" fill="{fill}"/>')
    svg.append(f'<rect x="{x}" y="{y + 22}" width="{w}" height="8" fill="{fill}"/>')
    text(x + 12, y + 21, title, title_size, "bold")
    check_fit(title + "  " + owner, title_size, w - 24, title)
    if owner:
        text(x + w - 12, y + 21, owner, 11.5, "normal", QUIET, "end")
    columns = cols or [lines]
    cw = (w - 24) / len(columns)
    for c, col in enumerate(columns):
        for i, line in enumerate(col):
            check_fit(line, size, cw - 12, title)
            text(x + 12 + c * cw, y + 52 + i * 18, line, size, fill=INK if not line.startswith("->") else stroke)


def band(y, h, label, key, sub=""):
    fill, stroke = C[key]
    svg.append(f'<rect x="14" y="{y}" width="{W - 28}" height="{h}" rx="12" fill="{fill}" fill-opacity="0.55" stroke="{stroke}" stroke-opacity="0.5"/>')
    text(30, y + 26, label, 17, "bold", stroke)
    if sub:
        text(W - 30, y + 26, sub, 13, "normal", QUIET, "end")


def arrow(x1, y1, x2, y2, color="#55554f", dashed=False, label="", lx=None, ly=None, anchor="start"):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#arr)"/>')
    if label:
        text(lx if lx is not None else x2 + 8, ly if ly is not None else (y1 + y2) / 2 + 4, label, 12, "normal", QUIET, anchor)


def path(d, color="#55554f", dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#arr)"/>')


def row(y, h, items, band_key, x0=30, x1=W - 30, gap=20):
    n = len(items)
    w = (x1 - x0 - gap * (n - 1)) / n
    xs = []
    for i, item in enumerate(items):
        x = x0 + i * (w + gap)
        box(round(x), y, round(w), h, item[0], item[2], band_key, item[1])
        xs.append((round(x), round(w)))
    return xs


def chain(xs, y, h):
    for (xa, wa), (xb, _) in zip(xs, xs[1:]):
        arrow(xa + wa + 2, y + h / 2, xb - 3, y + h / 2)


# --------------------------------------------------------------------------- content

DATA = [
    ("IL-PCSR", "IIT Kgp + Kanpur", ["6,271 query judgments", "  train 5,017 / val 627 /", "  test 627", "3,183 precedents", "936 statutes (IPC era)", "qrels = what each cited"]),
    ("BNS bare act", "via GitHub", ["358 sections, text", "verified vs Gazette", "+ IPC reference each", "-> BNS statute docs"]),
    ("IPC-BNS crosswalk", "derived", ["496 section pairs", "-> ipc_bns.csv", "-> offence_ids.csv", "Govt PDF: cross-check"]),
    ("Lexicons", "ours", ["Hinglish legal terms", "Hindi + legal stop words", "court -> states table"]),
    ("Hand-built queries", "ours", ["E2 collision", "E3 cross-version", "E4 en / hi / Hinglish", "E6 jurisdiction", "E7 temporal", "2 judges + kappa"]),
    ("Optional", "", ["PoliceDrishti: 190 facts", "-> BNS charges (E5)", "NLLB-E5 weights", "(Hindi-BEIR, NAACL'25)"]),
]

SPLITS = [
    ("TRAIN  5,017 queries", "", ["-> citation-graph edges for g(d)"]),
    ("VAL  627 queries", "", ["-> tune lambda, alpha, zone weights, k"]),
    ("TEST  627 queries", "", ["-> E1 / E3 results only, never tuned"]),
    ("HAND-BUILT SETS", "", ["-> E2, E4, E6, E7 (written first)"]),
]

OFFLINE = [
    ("01 build corpus", "owner A", ["load IL-PCSR + BNS sections", "segment: facts / arguments /", "  ratio / decision zones", "metadata: court, states, date,", "  code in force", "-> docs.jsonl (one schema)"]),
    ("02 build index", "owner A", ["text pipeline (same as queries):", "tokenize, stop words, stem,", "collision, version norm", "  (IPC 302 = BNS 103 = off:murder)", "-> zone index x2, facet index", "-> statute terms"]),
    ("03 build graph", "owner D", ["TRAIN qrels + precedent", "  citations -> citation graph", "-> PageRank g(d)", "-> per-state g(d | state)", "-> tiers + champion lists"]),
    ("04 encode dense", "owner C, optional", ["zone paragraphs -> NLLB-E5", "one-time, on a GPU", "-> embeddings/*.npy", "doc score = max over", "  its paragraphs"]),
]

L1 = [
    ("Query", "", ["text: en / hi / Hinglish", "+ user's state", "+ incident date", "+ filters: code:, court:"]),
    ("Analyze", "C + B", ["detect language", "translit + spelling norm", "lexicon + phonetic", "same text pipeline", "code in force from date", "+ other-code sections"]),
    ("Rank statutes", "D", ["BM25F over statutes", "only the code in force:", "IPC before 1 Jul 2024,", "BNS from then on"]),
    ("Statute bridge", "D", ["top statutes ->", "expand precedent query", "boost precedents that", "cite those statutes"]),
    ("Rank precedents", "D, C", ["BM25F with zone", "  weights", "+ dense (optional),", "  fused with alpha", "alpha set by QPP"]),
    ("Authority, top-K", "D", ["rel + lambda g(d|state)", "SC, own HC: binding", "other HC: persuasive", "heap top-K"]),
]

L2 = [
    ("Plan", "", ["original", "cross-code (IPC<->BNS)", "Boolean: top-idf AND", "facet: binding only", "statutes-first"]),
    ("Execute", "", ["each sub-query =", "one Layer 1 search", "(same engine, rule 4)"]),
    ("Fuse: RRF", "", ["sum of w / (60 + rank)", "no score normalisation", "CombSUM to compare"]),
    ("Reflect: QPP", "", ["weak retrieval?", "low max-idf, small gap", "-> one more round", "   (max 2 rounds)"]),
    ("Fused ranking", "", ["statutes + precedents", "trace of every", "  sub-query (--debug)"]),
]

L3 = [
    ("Abstain?", "", ["QPP says weak ->", "'not enough grounding'", "no LLM call"]),
    ("Chunk", "", ["top statutes +", "ratio / decision paras", "numbered [1]..[8]"]),
    ("Generate", "", ["LLM, temperature 0", "3-5 sentences", "one [n] per sentence"]),
    ("Citation check", "", ["tf-idf cosine:", "sentence vs its chunk", "< 0.2 -> unsupported"]),
    ("Version check", "", ["IPC cited for a", "post-July-2024 case?", "bare '302'? -> flag"]),
    ("Answer + flags", "", ["cited answer", "+ flags + sources", "'not legal advice'"]),
]

# --------------------------------------------------------------------------- layout

y = 20
text(30, y + 26, "Kanoon-Bridge: end-to-end architecture", 24, "bold")
text(30, y + 50, "Three layers on one core retriever. Layers 2 and 3 are stretch goals, gated at hour 26 and ~hour 30.", 14, "normal", QUIET)

# A. data sources
yA = 90
band(yA, 238, "Data sources", "data", "data/raw (gitignored) + data/crosswalk, data/lexicons, data/queries (committed)")
xsA = row(yA + 44, 178, DATA, "data", gap=14)

# split usage
yS = yA + 252
band(yS, 112, "How each split is used", "split", "rule 3: no test leakage")
row(yS + 40, 60, SPLITS, "split", gap=16)

# B. offline
yB = yS + 128
arrow(W / 2, yA + 238, W / 2, yB - 2)
band(yB, 290, "Offline build", "offline", "make data / index / graph / dense  (scripts/01-04, run once)")
xsB = row(yB + 44, 166, OFFLINE, "offline")
ya = yB + 226
svg.append(f'<rect x="30" y="{ya}" width="{W - 60}" height="48" rx="8" fill="#ffffff" stroke="{C["artifact"][1]}" stroke-dasharray="5 3"/>')
text(46, ya + 21, "data/processed/", 14, "bold")
text(46, ya + 40, "docs.jsonl  |  statutes_zone.pkl  |  precedents_zone.pkl  |  facets.pkl  |  statute_terms.json  |  authority.json  |  tiers.pkl  |  embeddings/", 13, "normal", QUIET)
for x, w in xsB:
    arrow(x + w / 2, yB + 210, x + w / 2, ya - 2)

# C. Layer 1
y1 = yB + 330
arrow(W / 2, ya + 48, W / 2, y1 - 2, label="loaded once by SearchEngine.load()", lx=W / 2 + 10)
band(y1, 236, "Layer 1  Core retriever", "l1", "search.py  SearchEngine.search(query)  - the only path to results (rule 4)")
xs1 = row(y1 + 44, 172, L1, "l1", gap=22)
chain(xs1, y1 + 44, 172)

# D. Layer 2
y2 = y1 + 290
band(y2, 262, "Layer 2  Research agent", "l2", "agent/  ResearchAgent.run(query)  - stretch goal, from hour 26")
xs2 = row(y2 + 44, 150, L2, "l2", gap=28)
chain(xs2, y2 + 44, 150)
# Execute <-> Layer 1
ex, ew = xs2[1]
path(f"M {ex + ew / 2 - 30} {y2 + 44} L {ex + ew / 2 - 30} {y1 + 236 + 4}", C["l2"][1])
path(f"M {ex + ew / 2 + 30} {y1 + 236} L {ex + ew / 2 + 30} {y2 + 40}", C["l1"][1])
text(ex + ew / 2 + 40, y1 + 236 + 33, "ranked lists back", 12, "normal", QUIET)
text(ex + ew / 2 - 40, y1 + 236 + 33, "sub-queries", 12, "normal", QUIET, "end")
# reflect loop back to plan
rx, rw = xs2[3]
px, pw = xs2[0]
yl = y2 + 44 + 150 + 30
svg.append(f'<path d="M {rx + rw / 2} {y2 + 194} L {rx + rw / 2} {yl} L {px + pw / 2} {yl} L {px + pw / 2} {y2 + 198}" '
           f'fill="none" stroke="{C["l2"][1]}" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#arr)"/>')
text((px + rx + rw) / 2, yl - 8, "if weak: pseudo-relevance feedback adds top-document terms, then plan again", 12, "normal", QUIET, "middle")

# E. Layer 3
y3 = y2 + 302
fx, fw = xs2[4]
gy = y3 - 18
band(y3, 236, "Layer 3  RAG answer", "l3", "rag/  answer(result, docs)  - stretch goal, only if Layer 2 is done by ~hour 30")
xs3 = row(y3 + 44, 160, L3, "l3", gap=22)
ax, aw = xs3[0]
path(f"M {fx + fw / 2} {y2 + 194} L {fx + fw / 2} {gy} L {ax + aw - 22} {gy} L {ax + aw - 22} {y3 + 40}", C["l2"][1])
text((fx + fw / 2 + ax + aw / 2) / 2, gy - 6, "fused ranking (or the Layer 1 result) goes to Layer 3", 12, "normal", QUIET, "middle")
chain(xs3, y3 + 44, 160)

# F. outputs + evaluation
yF = y3 + 276
band(yF, 236, "Outputs and evaluation", "eval", "")
box(30, yF + 44, 430, 172, "Outputs", [
    "app/cli.py --debug",
    "  shows every rewrite, postings, scores",
    "app/streamlit_app.py  (demo page)",
    "--agent -> Layer 2,  --answer -> Layer 3",
    "no flag -> Layer 1 only (dashed line, left)",
], "out", "")
box(480, yF + 44, W - 510, 172, "Evaluation", [], "eval", "owner D, scripts/05_run_all_evals.py", size=12.5, cols=[
    ["E1 IL-PCSR test: F1@k, MAP, MRR,", "   next to the published baselines", "E2 collision: wrong-offence rate, top-10",
     "E3 cross-version: Recall@20 of", "   BNS-worded queries", "-> results/tables, results/figures"],
    ["E4 language: P@5 for en / hi / Hinglish", "E6 jurisdiction: binding share in", "   top-5, nDCG@10",
     "E7 temporal: correct code of top statute", "Ablation ladder; efficiency (tiers)", "Agent vs core; RAG supported sentences"],
])
a6x, a6w = xs3[5]
arrow(a6x + a6w / 2, y3 + 204, a6x + a6w / 2, yF + 42, C["l3"][1], label="answers scored", lx=a6x + a6w / 2 - 8, ly=yF + 30, anchor="end")
# Layer 1 straight to outputs when the agent is off: elbow down the left margin
path(f"M 30 {y1 + 150} L 22 {y1 + 150} L 22 {yF + 130} L 28 {yF + 130}", C["l1"][1], dashed=True)
H = yF + 260
head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="Inter, Helvetica, Arial, DejaVu Sans, sans-serif">'
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 0L10 5L0 10z" fill="#55554f"/></marker></defs>')
doc = head + "\n".join(svg) + "</svg>"
(OUT / "architecture.svg").write_text(doc, encoding="utf-8")
print(f"wrote {OUT / 'architecture.svg'} ({W}x{H})")
for w in warnings:
    print("WARN", w)
try:
    import cairosvg

    cairosvg.svg2png(bytestring=doc.encode(), write_to=str(OUT / "architecture.png"), output_width=W * 2)
    print("wrote", OUT / "architecture.png")
except ImportError:
    print("pip install cairosvg to also write architecture.png")
