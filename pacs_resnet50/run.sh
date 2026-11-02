#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ " $* " == *" officehome "* ]]; then
  DATA="${DATA:-data/OfficeHome}"
  if [ ! -d "$DATA" ]; then
    echo "Office-Home not found at $DATA (see README), or set DATA=/path/to/OfficeHome"
    exit 1
  fi
else
  DATA="${DATA:-data/Homework3-PACS/PACS}"
  if [ ! -d "$DATA/photo" ]; then
    mkdir -p data
    git clone -q --depth 1 https://github.com/MachineLearning2020/Homework3-PACS data/Homework3-PACS
    DATA=data/Homework3-PACS/PACS
  fi
fi
python ltp.py --data "$DATA" "$@"
