#!/usr/bin/env bash
# One-command reproducible Harbor evaluation suite for Brass: Birmingham.
# Runs the exact judging protocol:
#   1. Do-Nothing baseline (-a nop)                  -> expected 0.000000
#   2. Pass-Spam baseline (39 passes, 0 VP)          -> expected 0.000000
#   3. Stop-Early baseline (halt after Canal Era)    -> expected <= 0.080000
#   4. Reference run (-a oracle, single-pass 350K)   -> expected ~0.85+ (WIN vs Mautoma)
#   5. Frontier Agent runs (3x -a opencode -m MODEL) -> runs when MODEL is provided
#
# Usage:
#   ./run_eval.sh                                    # Runs nop + pass-spam + stop-early + oracle
#   MODEL=anthropic/claude-sonnet-4-5 ./run_eval.sh  # Also runs 3x OpenCode frontier trials

set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${TASK_DIR}/harbor_runs}"
export PYTHONPATH="${TASK_DIR}:${PYTHONPATH:-}"

echo "========================================================================"
echo "Brass: Birmingham — Harbor Evaluation Suite"
echo "Task Directory : ${TASK_DIR}"
echo "Logs Directory : ${OUT_DIR}"
echo "========================================================================"

echo "[1/4] Running Do-Nothing Baseline (-a nop)..."
harbor run -p "${TASK_DIR}" -a nop -o "${OUT_DIR}" --job-name eval-nop -y

echo "[2/4] Running Pass-Spam Baseline (39 passes -> 0 VP)..."
harbor run -p "${TASK_DIR}" -a baselines.harbor_agents:PassSpamAgent -o "${OUT_DIR}" --job-name eval-pass-spam -y

echo "[3/4] Running Stop-Early Baseline (halt after Canal Era)..."
harbor run -p "${TASK_DIR}" -a baselines.harbor_agents:StopEarlyAgent -o "${OUT_DIR}" --job-name eval-stop-early -y

echo "[4/4] Running Single-Pass Neuroevolution Reference Run (-a oracle)..."
harbor run -p "${TASK_DIR}" -a oracle -o "${OUT_DIR}" --job-name eval-ga-oracle -y

if [ -n "${MODEL:-}" ]; then
    TRIALS="${TRIALS:-3}"
    EXTRA_ARGS=()
    if [[ "${MODEL}" == google/* ]]; then
        if [ -z "${GOOGLE_GENERATIVE_AI_API_KEY:-}" ] && [ -n "${GEMINI_API_KEY:-}" ]; then
            export GOOGLE_GENERATIVE_AI_API_KEY="${GEMINI_API_KEY}"
        fi
        if [ -n "${GOOGLE_GENERATIVE_AI_API_KEY:-}" ]; then
            EXTRA_ARGS+=(--api-key-name GOOGLE_GENERATIVE_AI_API_KEY)
        fi
    fi
    echo "[5/5] Running ${TRIALS}x OpenCode Frontier Agent Trials (-a opencode -m ${MODEL})..."
    harbor run -p "${TASK_DIR}" -a opencode -m "${MODEL}" "${EXTRA_ARGS[@]}" -k "${TRIALS}" -n "${TRIALS}" -o "${OUT_DIR}" --job-name eval-opencode-frontier -y
else
    echo "[Info] Set MODEL=<provider/model> (e.g. MODEL=google/gemini-3.8-flash or anthropic/claude-sonnet-4-5) to also run 3x OpenCode frontier trials."
fi

echo "========================================================================"
echo "Evaluation Summary (${OUT_DIR}):"
python3 - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
for job in sorted(out.iterdir()):
    res_file = job / "result.json"
    if not res_file.exists():
        continue
    data = json.loads(res_file.read_text(encoding="utf-8"))
    evals = data.get("stats", {}).get("evals", {})
    for k, v in evals.items():
        metrics = v.get("metrics") or [{}]
        mean_rew = metrics[0].get("mean", 0.0)
        print(f"  {job.name:<28s} | agent={k:<26s} | trials={v.get('n_trials', 0)} | mean_reward={mean_rew:.6f}")
PY
echo "========================================================================"
