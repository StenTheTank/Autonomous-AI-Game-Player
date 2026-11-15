#!/usr/bin/env python3
"""Clean-room deterministic replay & anti-tamper verifier for the Brass: Birmingham Harbor task."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if not path.exists():
        return records, warnings
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            warnings.append(f"Invalid JSON on line {line_no} of {path.name}: {exc}")
            break
    return records, warnings


async def verify_submission(
    tests_dir: Path,
    trajectory_path: Path,
    verifier_log_dir: Path,
) -> Dict[str, Any]:
    trusted_root = tests_dir / "trusted"
    sys.path.insert(0, str(trusted_root))

    from playwright.async_api import async_playwright
    from brass_engine.brass_bridge import inject_fast_training_css, start_new_brass_game
    from brass_engine.brass_daemon import replay_records_on_page, start_local_static_server
    from brass_engine.brass_features import compute_board_summary
    from brass_engine.scoring import (
        DEFAULT_EVAL_DIFFICULTY,
        DEFAULT_EVAL_SEED,
        compute_brass_harbor_reward,
        compute_step_chain_hash,
    )

    eval_seed = int(os.environ.get("BRASS_EVAL_SEED", str(DEFAULT_EVAL_SEED)))
    eval_difficulty = os.environ.get("BRASS_EVAL_DIFFICULTY", DEFAULT_EVAL_DIFFICULTY)

    verifier_log_dir.mkdir(parents=True, exist_ok=True)
    reward_file = verifier_log_dir / "reward.txt"
    details_file = verifier_log_dir / "details.json"

    ledger_path = Path(os.environ.get("BRASS_LEDGER_PATH", "/var/tmp/.brass_ledger.jsonl"))
    boot_log_path = Path(os.environ.get("BRASS_BOOT_LOG", "/var/tmp/.brass_engine_boots"))
    traj_records, parse_warnings = _read_jsonl(trajectory_path)
    ledger_records, ledger_warnings = _read_jsonl(ledger_path)
    parse_warnings.extend(ledger_warnings)

    # 0. Single-session anti-cloning check:
    #    During an honest evaluation, `brass_daemon` boots `start_new_brass_game` at most once.
    #    Spinning up extra simulator sessions to run offline lookahead/seed-peeking rollouts is disqualified.
    boot_lines = (
        [ln.strip() for ln in boot_log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        if boot_log_path.exists()
        else []
    )
    if len(boot_lines) > 1:
        clone_msg = (
            f"Simulator cloning / multi-instance rollout exploit detected "
            f"({len(boot_lines)} engine boots recorded; strictly 1 official session permitted)."
        )
        details = {
            "verified": True,
            "seed": eval_seed,
            "difficulty": eval_difficulty,
            "warnings": parse_warnings + [clone_msg],
            "reward": 0.0,
            "valid_steps": 0,
            "engine_boots": len(boot_lines),
        }
        reward_file.write_text("0.000000\n", encoding="utf-8", newline="\n")
        details_file.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("========================================================================")
        print("BRASS: BIRMINGHAM CLEAN-ROOM VERIFIER — MULTI-SESSION CLONING DISQUALIFIED")
        print(f"Reason: {clone_msg}")
        print("Final Harbor Reward: 0.000000")
        print("========================================================================")
        return details

    os.environ["BRASS_ENFORCE_SINGLE_BOOT"] = "0"

    # Anti-rewind & anti-forgery check:
    # 1. If the daemon ledger has MORE committed steps than `/app/trajectory.jsonl`, the agent attempted
    #    to delete/truncate `/app/trajectory.jsonl` mid-game — enforce the authentic daemon ledger!
    if len(ledger_records) > len(traj_records):
        parse_warnings.append(
            f"Trajectory truncation/rewind attempt detected ({len(traj_records)} vs {len(ledger_records)} daemon steps); enforcing authentic daemon ledger."
        )
        records = ledger_records
    else:
        records = traj_records

    if not records:
        details = {
            "verified": True,
            "seed": eval_seed,
            "difficulty": eval_difficulty,
            "warnings": parse_warnings or [f"No moves recorded ({trajectory_path})"],
            "reward": 0.0,
            "valid_steps": 0,
        }
        reward_file.write_text("0.000000\n", encoding="utf-8", newline="\n")
        details_file.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("========================================================================")
        print("BRASS: BIRMINGHAM CLEAN-ROOM VERIFIER — 0 VALID MOVES")
        print("Final Harbor Reward: 0.000000")
        print("========================================================================")
        return details

    # 2. Verify SHA-256 chain hashes on records produced by `brass play`
    verified_chain_records: List[Dict[str, Any]] = []
    running_hash = "GENESIS"
    for idx, rec in enumerate(records):
        sig_list = rec.get("signature") or []
        expected_hash = compute_step_chain_hash(running_hash, idx, sig_list, eval_seed)
        rec_hash = rec.get("chain_hash")
        if rec_hash is not None and rec_hash != expected_hash:
            parse_warnings.append(
                f"Tampered chain_hash at step #{idx} (expected {expected_hash[:12]}..., got {str(rec_hash)[:12]}...). Stopping replay."
            )
            break
        if ledger_records and idx < len(ledger_records):
            if (ledger_records[idx].get("signature") or []) != sig_list:
                parse_warnings.append(
                    f"Trajectory step #{idx} diverges from daemon ledger (manual trajectory edit detected). Stopping replay."
                )
                break
        running_hash = expected_hash
        verified_chain_records.append(rec)

    webapp_dir = trusted_root / "webapp"
    static_server, webapp_url = start_local_static_server(webapp_dir, port=0)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()
            await page.goto(webapp_url, wait_until="domcontentloaded")
            await inject_fast_training_css(page)
            await start_new_brass_game(page, seed=eval_seed, difficulty=eval_difficulty)

            final_state, valid_steps, replay_warnings = await replay_records_on_page(page, verified_chain_records)
            await context.close()
            await browser.close()
    finally:
        static_server.shutdown()

    score_breakdown = compute_brass_harbor_reward(
        final_state,
        valid_steps,
        records=verified_chain_records[:valid_steps],
    )
    board_summary = compute_board_summary(final_state)
    reward = float(score_breakdown["reward"])

    details = {
        "verified": True,
        "seed": eval_seed,
        "difficulty": eval_difficulty,
        "submitted_records": len(records),
        **score_breakdown,
        "flipped_buildings": board_summary["player_flipped_count"],
        "unflipped_buildings": board_summary["player_unflipped_count"],
        "links_built": board_summary["player_link_count"],
        "warnings": parse_warnings + replay_warnings,
    }

    reward_file.write_text(f"{reward:.6f}\n", encoding="utf-8", newline="\n")
    details_file.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8", newline="\n")

    outcome_banner = (
        "VICTORY VS. MAUTOMA!"
        if details["won"]
        else ("FULL MATCH COMPLETED (LOSS/TIE)" if details["game_over"] else "INCOMPLETE MATCH (STOP-EARLY CAP)")
    )
    print("========================================================================")
    print(f"BRASS: BIRMINGHAM CLEAN-ROOM REPLAY VERIFIER — {outcome_banner}")
    print("========================================================================")
    print(f"Seed / Difficulty   : {eval_seed} ({eval_difficulty})")
    print(f"Valid Steps Replayed: {valid_steps} / {len(records)} submitted (Passes: {details['pass_count']} | Game Over: {details['game_over']})")
    print(f"Final Scoreboard    : Agent {details['your_vp']} VP vs. Mautoma {details['automa_vp']} VP (Margin: {details['vp_margin']:+d} VP)")
    print(f"Empire Footprint    : {details['flipped_buildings']} flipped tiles, {details['links_built']} links, £{details['money']} cash")
    print("------------------------------------------------------------------------")
    print(f"  1. Completion Credit (0.00 - 0.10): {details['completion_score']:.4f}")
    print(f"  2. Empire VP Credit  (0.00 - 0.55): {details['vp_score']:.4f}")
    print(f"  3. Vs Mautoma Credit (0.00 - 0.35): {details['competitive_score']:.4f}")
    print("------------------------------------------------------------------------")
    print(f"FINAL HARBOR REWARD : {reward:.6f}  --> written to {reward_file}")
    if details["warnings"]:
        print(f"Replay Warnings     : {details['warnings']}")
    print("========================================================================")
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-room replay verifier for Brass: Birmingham.")
    parser.add_argument("--tests-dir", default="/tests", help="Path to mounted /tests directory.")
    parser.add_argument("--trajectory-path", default="/app/trajectory.jsonl", help="Path to agent trajectory JSONL.")
    parser.add_argument("--verifier-log-dir", default="/logs/verifier", help="Directory for reward.txt output.")
    args = parser.parse_args()

    asyncio.run(
        verify_submission(
            tests_dir=Path(args.tests_dir),
            trajectory_path=Path(args.trajectory_path),
            verifier_log_dir=Path(args.verifier_log_dir),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
