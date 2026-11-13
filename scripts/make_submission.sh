#!/usr/bin/env bash
# Build the CSD358 course submission ZIP: report PDF + GitHub link, source code, datasets.
#
#   bash scripts/make_submission.sh <report.pdf> <Member1> <Member2> [<Member3> <Member4>] [--with-restricted]
#   e.g. bash scripts/make_submission.sh ~/Downloads/kanoon_bridge_acm.pdf GauravRai ShauryaSharma SharanyaGupta
#   -> ../CSD358_GauravRai_ShauryaSharma_SharanyaGupta.zip
#
# Contents:
#   report/       the report PDF and GITHUB.txt (repository link)
#   source/       every committed file at HEAD (code, configs, tests, docs, results, benchmarks,
#                 crosswalks, lexicons) - the same as the GitHub repository
#   dataset/      the raw data the experiments used (data/raw), with DATASETS.md (sources, licences)
#
# IL-PCSR (CC-BY-NC-SA 4.0, gated on Hugging Face) and bns-study-platform (no licence declared) are
# left out unless --with-restricted is given; DATASETS.md says where to get them. data/processed
# (the built indexes, ~700 MB) is never included: `make build` rebuilds it.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 3 ] || { sed -n '2,5p' "$0"; exit 1; }
PDF="$1"; shift
RESTRICTED=0
MEMBERS=()
for a in "$@"; do
    if [ "$a" = "--with-restricted" ]; then RESTRICTED=1; else MEMBERS+=("$a"); fi
done
[ -f "$PDF" ] || { echo "report PDF not found: $PDF"; exit 1; }
[ ${#MEMBERS[@]} -ge 2 ] || { echo "give 2-4 member names"; exit 1; }

NAME="CSD358_$(IFS=_; echo "${MEMBERS[*]}")"
OUT="$(cd .. && pwd)/$NAME.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME/report" "$STAGE/$NAME/source" "$STAGE/$NAME/dataset"

cp "$PDF" "$STAGE/$NAME/report/$NAME.pdf"
URL="$(git remote get-url origin 2>/dev/null | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##; s#https://[^@]*@#https://#')"
printf 'GitHub repository: %s\nCommit: %s\n' "$URL" "$(git rev-parse --short HEAD)" > "$STAGE/$NAME/report/GITHUB.txt"

git archive HEAD | tar -x -C "$STAGE/$NAME/source"

for d in data/raw/*/; do
    n="$(basename "$d")"
    if [ $RESTRICTED -eq 0 ] && { [ "$n" = "ilpcsr" ] || [ "$n" = "bns-study-platform" ]; }; then
        echo "  skipped data/raw/$n (restricted; use --with-restricted to include)"
        continue
    fi
    mkdir -p "$STAGE/$NAME/dataset/raw"
    cp -r "$d" "$STAGE/$NAME/dataset/raw/$n"
done
cat > "$STAGE/$NAME/dataset/DATASETS.md" <<'EOF'
# Datasets used by Kanoon-Bridge

Copy `raw/` to `data/raw/` in the source folder, then `make build` (IL-PCSR corpus) or `make open`
(open corpus only). Benchmarks, crosswalks and lexicons are already in `source/data/`.

| Folder | Source | Licence |
| --- | --- | --- |
| raw/ilpcsr | IL-PCSR, https://huggingface.co/datasets/Exploration-Lab/IL-PCSR (Paul et al., EMNLP 2025) | CC-BY-NC-SA 4.0, gated: accept the terms on Hugging Face |
| raw/open-laws | India Code via https://github.com/kanoonhq/open-laws | CC0 |
| raw/bns-study-platform | https://github.com/PSKprem/bns-study-platform | no licence declared |
| raw/hc | eCourts Indian High Court Judgments, https://registry.opendata.aws/indian-high-court-judgments/ | CC-BY 4.0 |
| raw/bail | IndianBailJudgments-1200, https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200 | CC-BY 4.0 |
| raw/indian-legal-mcp, raw/il-tur | statute text and metadata used by the ingest scripts | see each source |

A folder missing here was left out for licence reasons: `python scripts/00_fetch_data.py` fetches it.
EOF

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" "$NAME")
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
