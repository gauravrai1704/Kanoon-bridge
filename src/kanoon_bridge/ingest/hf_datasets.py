"""Download the extra Indian legal datasets from the Hugging Face Hub.  [owner: Gaurav — working]

Gated datasets need a read token of an account that accepted the dataset's terms on its page:
env HF_TOKEN. Files are listed with their sizes first; `--max-gb` stops a run that would download
more than that.

    name             repo (license)                                         used for
    bail             SnehaDeshmukh/IndianBailJudgments-1200 (CC-BY-4.0)    E19 facts -> charged sections
    open-india-law   vaquill/open-india-law (CC-BY-4.0, gated)              BNS-era judgments (post 1 July 2024)
                                                                            -> data/raw/hc/judgments.jsonl format
    il-tur           Exploration-Lab/IL-TUR (CC-BY-NC-SA-4.0, gated)        ILSI (statute identification), IL-PCR
                                                                            (prior-case retrieval): second benchmark
    nyayaanumana     L-NLProc/NyayaAnumana-Classification-Data (Apache-2.0, gated)   scale (2.28M cases, pre-BNS)
    legalseg         L-NLProc/LegalSeg_CSV (Apache-2.0, gated)              rhetorical roles (segment.py check)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HF = "https://huggingface.co"
DATASETS = {
    "bail": ("SnehaDeshmukh/IndianBailJudgments-1200", r"indian_bail_judgments\.json$", "data/raw/bail"),
    "open-india-law": ("vaquill/open-india-law", r"_judgments\.parquet$", "data/raw/open-india-law"),
    "il-tur": ("Exploration-Lab/IL-TUR", r"^(lsi|pcr)/", "data/raw/il-tur"),
    "nyayaanumana": ("L-NLProc/NyayaAnumana-Classification-Data", r"\.(parquet|csv|jsonl?)$", "data/raw/nyayaanumana"),
    "legalseg": ("L-NLProc/LegalSeg_CSV", r"\.(parquet|csv|jsonl?)$", "data/raw/legalseg"),
}


def _headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def list_files(repo: str, pattern: str) -> list[tuple[str, int]]:
    import requests

    r = requests.get(f"{HF}/api/datasets/{repo}/tree/main", params={"recursive": "true"}, headers=_headers(), timeout=60)
    if r.status_code in (401, 403):
        raise PermissionError(f"{repo}: access denied - accept its terms on {HF}/datasets/{repo} and set HF_TOKEN")
    r.raise_for_status()
    return [(f["path"], int(f.get("size") or 0)) for f in r.json() if f.get("type") == "file" and re.search(pattern, f["path"])]


def download(name: str, root: Path, max_gb: float = 20.0, only: str | None = None) -> list[Path]:
    import requests

    repo, pattern, dest = DATASETS[name]
    files = list_files(repo, pattern)
    if only:
        files = [f for f in files if re.search(only, f[0])]
    total = sum(s for _, s in files) / 2**30
    print(f"{name}: {len(files)} files, {total:.2f} GiB")
    if total > max_gb:
        raise RuntimeError(f"{name}: {total:.1f} GiB > --max-gb {max_gb}; narrow it with --only <regex>")
    out = []
    for path, size in files:
        target = root / dest / path
        if target.exists() and target.stat().st_size == size:
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(f"{HF}/datasets/{repo}/resolve/main/{path}", headers=_headers(), stream=True, timeout=600) as r:
            if r.status_code in (401, 403):
                raise PermissionError(f"{repo}: access denied - accept its terms on {HF}/datasets/{repo} and set HF_TOKEN"
                                      " (large files also need the host us.aws.cdn.hf.co allowed)")
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"  {path} ({size / 2**20:.1f} MiB)")
        out.append(target)
    return out


_NEW_CODES = re.compile(r"\b(BNS|BNSS|BSA|Bharatiya\s+(?:Nyaya|Nagarik|Sakshya))", re.I)


def open_india_law_to_jsonl(root: Path, out: Path, since: str = "2024-07-01", require_new_code: bool = True) -> int:
    """BNS-era judgments from open-india-law's parquet files into the scripts/11 format
    ({id, court, state, date, title, text}). Columns are found by name (text / date / title / court)."""
    import json

    import pyarrow.parquet as pq

    from kanoon_bridge.ingest.metadata import CourtTable  # noqa: F401  (court names resolved at ingest)

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for path in sorted((root / DATASETS["open-india-law"][2]).glob("*_judgments.parquet")):
            court_slug = path.name.removeprefix("in_").removesuffix("_judgments.parquet")
            pf = pq.ParquetFile(path)
            cols = pf.schema_arrow.names
            pick = lambda *names: next((c for c in cols if c.lower() in names), None)  # noqa: E731
            c_text, c_date = pick("text", "content", "judgment_text", "body"), pick("date", "decision_date", "judgment_date")
            c_title, c_id = pick("title", "case_title", "name"), pick("id", "doc_id", "cnr")
            if not c_text or not c_date:
                print(f"  {path.name}: no text/date column in {cols} - skipped")
                continue
            for batch in pf.iter_batches(columns=[c for c in (c_text, c_date, c_title, c_id) if c], batch_size=2000):
                for r in batch.to_pylist():
                    d = str(r[c_date] or "")[:10]
                    text = r[c_text] or ""
                    if d < since or (require_new_code and not _NEW_CODES.search(text[:20000])):
                        continue
                    f.write(json.dumps({"id": f"oil-{court_slug}-{r.get(c_id) or n}", "court": court_slug.replace("-", " ").title()
                                        + (" High Court" if court_slug != "supreme-court" else ""),
                                        "state": court_slug, "date": d, "title": r.get(c_title) or "", "text": text},
                                       ensure_ascii=False) + "\n")
                    n += 1
    return n


__all__ = ["download", "list_files", "open_india_law_to_jsonl", "DATASETS"]
