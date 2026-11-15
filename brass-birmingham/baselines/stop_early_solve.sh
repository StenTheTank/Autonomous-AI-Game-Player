#!/usr/bin/env bash
# Stop-Early Baseline: Plays through the 10-round Canal Era (19 actions), collects
# Canal Era VP, and deliberately halts before completing the 10-round Rail Era.
# Used by judges & ./run_eval.sh to verify that stopping early earns <= 0.08 credit
# and cannot exploit Canal-era loan-and-quit strategies.
set -euo pipefail

for step in $(seq 1 19); do
    brass play 0 >/dev/null
done

brass status
