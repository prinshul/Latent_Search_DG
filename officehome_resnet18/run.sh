#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${DATA:-data/OfficeHomeDataset_10072016}"
python ltp.py --data "$DATA" "$@"
