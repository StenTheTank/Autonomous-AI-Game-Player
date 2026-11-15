#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/run_ppo_oracle.py" --checkpoint "${SCRIPT_DIR}/brass_ppo_best.pt"
