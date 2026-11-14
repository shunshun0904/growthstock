#!/usr/bin/env bash
# Build static Pages assets. Credentials are used only by the Python collector.
set -euo pipefail
catalog_output=${1:?output directory required}
catalog_work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/supply-assets.XXXXXX")
trap 'rm -rf "$catalog_work"' EXIT
mkdir -p "$catalog_work/raw"
gh release view data-raw --json assets --jq '.assets[].name' > "$catalog_work/assets.txt"
python3 - "$catalog_work/assets.txt" <<'PY' > "$catalog_work/bars.txt"
import pathlib, re, sys
names = [x for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if re.fullmatch(r'bars_[0-9]{4}\.parquet', x)]
if not names:
    raise SystemExit('No cached daily-bar assets available')
# Select available years rather than the current year, including New Year holidays.
selected = sorted(names)[-2:]
assets = set(pathlib.Path(sys.argv[1]).read_text().splitlines())
selected += [x.replace('bars_', 'margin_') for x in selected if x.replace('bars_', 'margin_') in assets]
print('\n'.join(selected))
PY
while IFS= read -r asset; do
  bash scripts/gh_release_download.sh data-raw "$catalog_work/raw" "$asset"
done < "$catalog_work/bars.txt"
python3 scripts/build_supply_demand_catalog.py --bars-dir "$catalog_work/raw" --output-dir "$catalog_output"
