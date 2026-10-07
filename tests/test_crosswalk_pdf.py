"""PDF cross-check parser on a small generated PDF shaped like the government comparison table."""

from __future__ import annotations

import pytest


def _make_pdf(path, header, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.27, 4))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="left",
                 colWidths=[0.08, 0.14, 0.34, 0.2, 0.24])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    fig.savefig(path)
    plt.close(fig)


def test_pdf_table_rows_and_compare(tmp_path):
    pytest.importorskip("pdfplumber")
    from kanoon_bridge.ingest.parse_crosswalk import compare_sources, extract_rows_from_pdf

    pdf = tmp_path / "cmp.pdf"
    _make_pdf(pdf, ["Sl. No.", "BNS Sections", "Subject", "IPC Sections", "Comparison Summary"],
              [["1", "103", "Punishment for murder", "302", "same"],
               ["2", "351", "Criminal intimidation", "503, 506, 507", "merged"],
               ["3", "111", "Organised crime", "New Section", "new"],
               ["4", "127", "Wrongful confinement", "340, 342-344", "merged"]])
    rows = extract_rows_from_pdf(pdf)
    pairs = {(r["ipc_section"], r["bns_section"]) for r in rows}
    assert {("302", "103"), ("506", "351"), ("343", "127")} <= pairs
    assert not any(r["bns_section"] == "111" for r in rows)
    diff = compare_sources([{"ipc_section": "302", "bns_section": "103"},
                            {"ipc_section": "498a", "bns_section": "85"}], rows)
    assert ("498a", "85") in diff["only_json"] and ("506", "351") in diff["only_pdf"]
