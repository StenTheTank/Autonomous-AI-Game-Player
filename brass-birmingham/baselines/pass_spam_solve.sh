#!/usr/bin/env bash
# Pass-Spam Baseline: Completes all 39 turns by passing every single turn (0 VP).
# Verifies that reaching game_over=True without building/scoring earns 0.000000 credit.
set -euo pipefail

python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/brass")
import brass_cli

brass_cli.ensure_daemon_running()
for _ in range(45):
    st = brass_cli._http_json("GET", "/status")["state"]
    if st.get("game_over"):
        break
    actions = brass_cli._http_json("GET", "/legal").get("actions") or []
    pass_idx = next((i for i, a in enumerate(actions) if a.get("action_type") == "pass"), len(actions) - 1)
    res = brass_cli._http_json("POST", "/play", {"action_id": int(pass_idx)})
    if res.get("game_over"):
        break
print(brass_cli._http_json("GET", "/status").get("hud", ""))
PY
