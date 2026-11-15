#!/usr/bin/env python3
"""CLI entrypoint (`brass`) for interacting with the Brass: Birmingham Harbor environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def _daemon_base_url() -> str:
    host = os.environ.get("BRASS_DAEMON_HOST", "127.0.0.1")
    port = int(os.environ.get("BRASS_DAEMON_PORT", "18765"))
    return f"http://{host}:{port}"


def _engine_root() -> Path:
    env_root = os.environ.get("BRASS_ENGINE_ROOT")
    if env_root:
        return Path(env_root)
    opt_root = Path("/opt/brass")
    if opt_root.exists():
        return opt_root
    return Path(__file__).resolve().parent


def _http_json(method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    url = f"{_daemon_base_url()}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {"ok": False, "error": f"HTTP {err.code}: {raw}"}


def ensure_daemon_running() -> None:
    try:
        health = _http_json("GET", "/health", timeout=1.5)
        if health.get("ok"):
            return
    except Exception:
        pass

    root = _engine_root()
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing_pp}" if existing_pp else str(root)
    env.setdefault("BRASS_WEBAPP_DIR", str(root / "webapp"))
    if "BRASS_TRAJECTORY_PATH" not in env:
        default_traj = Path("/app/trajectory.jsonl") if os.name != "nt" else (root / "trajectory.jsonl")
        env["BRASS_TRAJECTORY_PATH"] = str(default_traj)

    log_path = Path(tempfile.gettempdir()) / "brass_daemon.log"
    log_file = log_path.open("ab")
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(root),
        "env": env,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen([sys.executable, "-m", "brass_engine.brass_daemon"], **popen_kwargs)

    deadline = time.time() + 25.0
    while time.time() < deadline:
        try:
            health = _http_json("GET", "/health", timeout=1.5)
            if health.get("ok"):
                return
        except Exception:
            time.sleep(0.25)

    tail = ""
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
    raise RuntimeError(f"Failed to start Brass engine daemon within 25s.\nDaemon log:\n{tail}")


def cmd_rules() -> int:
    root = _engine_root()
    sys.path.insert(0, str(root))
    from brass_engine.formatter import RULES_REFERENCE

    print(RULES_REFERENCE)
    return 0


def cmd_status(as_json: bool) -> int:
    ensure_daemon_running()
    res = _http_json("GET", "/status")
    if as_json:
        out = {k: v for k, v in res.items() if k != "hud"}
        print(json.dumps(out, indent=2))
    else:
        if not res.get("ok"):
            print(f"Error: {res.get('error')}", file=sys.stderr)
            return 1
        print(res.get("hud", ""))
    return 0 if res.get("ok") else 1


def cmd_legal(as_json: bool) -> int:
    ensure_daemon_running()
    res = _http_json("GET", "/legal")
    if as_json:
        out = {k: v for k, v in res.items() if k != "formatted"}
        print(json.dumps(out, indent=2))
    else:
        if not res.get("ok"):
            print(f"Error: {res.get('error')}", file=sys.stderr)
            return 1
        print(res.get("formatted", ""))
    return 0 if res.get("ok") else 1


def cmd_play(action_id: int, as_json: bool, brief: bool) -> int:
    ensure_daemon_running()
    res = _http_json("POST", "/play", {"action_id": int(action_id)})
    if as_json:
        out = {k: v for k, v in res.items() if k not in ("hud", "formatted_next_legal")}
        print(json.dumps(out, indent=2))
        return 0 if res.get("ok") else 1

    if not res.get("ok"):
        print(f"ERROR: {res.get('error')}", file=sys.stderr)
        if res.get("formatted_legal"):
            print(res["formatted_legal"])
        elif res.get("hud"):
            print(res["hud"])
        return 1

    print(f">>> EXECUTED STEP #{res['executed_step']}: [{res['executed_action_id']}] {res['executed_description']}")
    print(res.get("hud", ""))
    if not brief and not res.get("game_over"):
        print()
        print(res.get("formatted_next_legal", ""))
    return 0


def cmd_history(as_json: bool) -> int:
    ensure_daemon_running()
    res = _http_json("GET", "/history")
    if as_json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    hist = res.get("history") or []
    if not hist:
        print("No actions played yet (`0` steps recorded).")
        return 0
    print(f"MATCH TRAJECTORY ({len(hist)} steps played):")
    for rec in hist:
        print(f"  Step {rec.get('step', 0):2d}: {rec.get('description', '')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brass",
        description="Brass: Birmingham (Solo vs. Mautoma) CLI for Harbor agent evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show current board HUD, resources, hand cards, and VP scoreboard.")
    p_status.add_argument("--json", action="store_true", help="Output structured JSON state.")

    p_legal = sub.add_parser("legal", help="List all currently legal macro-actions (0..K-1) with costs & effects.")
    p_legal.add_argument("--json", action="store_true", help="Output structured JSON candidate actions.")

    p_play = sub.add_parser("play", help="Execute legal macro-action <action_id> and advance the simulation.")
    p_play.add_argument("action_id", type=int, help="Integer index from `brass legal` (0..K-1).")
    p_play.add_argument("--json", action="store_true", help="Output structured JSON result + next legal actions.")
    p_play.add_argument("--brief", action="store_true", help="Omit printing next turn's legal actions.")

    sub.add_parser("rules", help="Print Brass: Birmingham mechanics, tactical tips, and Harbor reward rubric.")

    p_hist = sub.add_parser("history", help="Show all moves recorded so far in this match.")
    p_hist.add_argument("--json", action="store_true", help="Output structured JSON trajectory.")

    args = parser.parse_args()
    if args.command == "rules":
        return cmd_rules()
    if args.command == "status":
        return cmd_status(as_json=args.json)
    if args.command == "legal":
        return cmd_legal(as_json=args.json)
    if args.command == "play":
        return cmd_play(args.action_id, as_json=args.json, brief=args.brief)
    if args.command == "history":
        return cmd_history(as_json=args.json)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
