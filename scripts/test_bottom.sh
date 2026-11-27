#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/test_bottom.sh /path/to/bottom_checkpoint.pth" >&2
  exit 2
fi

CHECKPOINT="$1"
CONFIG="configs/tec_nas/bottom_view.py"
ANNOTATION="data/Bottom_View_V2/annotations/instances_val2017.json"
IMAGES="data/Bottom_View_V2/images/val"

for file in "$CONFIG" "architectures/res50_like.txt" "tools/test.py" "$ANNOTATION" "$CHECKPOINT"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done
if [[ ! -d "$IMAGES" ]]; then
  echo "Missing validation image directory: $IMAGES" >&2
  exit 1
fi

echo "Evaluating TEC-NAS Bottom View (16 classes)"
exec python tools/test.py "$CONFIG" "$CHECKPOINT" --eval bbox
