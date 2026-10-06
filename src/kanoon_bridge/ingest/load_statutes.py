"""Load BNS bare-act text into one Document per section.  [owner: A]

Source: India Code (https://www.indiacode.nic.in), the Bharatiya Nyaya Sanhita, 2023.
Save the text or PDF in data/raw/bns/.

IL-PCSR's statute pool is IPC-era; BNS sections must be added so BNS-era queries have
statutes to retrieve. Each section becomes:

    Document(doc_id="bns:103", doc_type=STATUTE, code=BNS, section="103",
             title="Punishment for murder", paragraphs=[Paragraph(text, zone="statute")],
             decision_date=date(2024, 7, 1))

Section ids use schema.section_ref() so they line up with the crosswalk.
"""

from __future__ import annotations

from pathlib import Path

from kanoon_bridge.schema import Document


def parse_bns(path: str | Path) -> list[Document]:
    """TODO(A): split the bare act on section headings ("103. Punishment for murder.—"),
    one Document per section, sub-sections kept as separate paragraphs. Log the count:
    the BNS has 358 sections."""
    raise NotImplementedError("TODO(A): parse BNS bare act")
