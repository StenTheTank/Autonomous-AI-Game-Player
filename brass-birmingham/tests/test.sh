#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

on_unexpected_err() {
    if [ ! -s /logs/verifier/reward.txt ]; then
        echo "0.000000" > /logs/verifier/reward.txt
    fi
}
trap on_unexpected_err EXIT

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${TESTS_DIR}/verify_brass.py" \
    --tests-dir "${TESTS_DIR}" \
    --trajectory-path "${BRASS_TRAJECTORY_PATH:-/app/trajectory.jsonl}" \
    --verifier-log-dir /logs/verifier
