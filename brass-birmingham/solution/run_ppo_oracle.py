#!/usr/bin/env python3
"""
Single-Pass Neuroevolution Reference Policy (`BrassActorCritic`, 350K parameters) for Brass: Birmingham.

Plays a full 20-round (39-action) match strictly via the official `brass` CLI daemon
(`GET /status` -> `GET /legal` -> `POST /play`) in a single online pass (`N=1` greedy argmax)
with ZERO lookahead rollouts, ZERO simulator cloning, and ZERO seed peeking.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play Brass: Birmingham single-pass (N=1) using the GA-evolved BrassActorCritic policy."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "brass_ppo_best.pt"),
        help="Path to evolved PyTorch BrassActorCritic checkpoint (.pt).",
    )
    parser.add_argument(
        "--engine-root",
        default=os.environ.get("BRASS_ENGINE_ROOT", "/opt/brass"),
        help="Path to brass_engine root.",
    )
    args = parser.parse_args()

    solution_dir = Path(__file__).resolve().parent
    if str(solution_dir) not in sys.path:
        sys.path.insert(0, str(solution_dir))

    engine_root = Path(args.engine_root)
    if not engine_root.exists():
        engine_root = solution_dir.parent / "environment"
    sys.path.insert(0, str(engine_root))

    import brass_cli
    from brass_engine.brass_features import ACTION_DIM, encode_action, encode_state
    from model import BrassActorCritic, load_brass_checkpoint

    brass_cli.ensure_daemon_running()
    device = torch.device("cpu")
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        model, meta = load_brass_checkpoint(ckpt_path, device=device)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(
            f"[GA Reference Policy] Loaded {ckpt_path.name} "
            f"({param_count:,} params | algo={meta.get('algorithm', 'ga')} | "
            f"gen={meta.get('generation', '?')} | champ_avg={meta.get('champion_avg_vp', meta.get('rolling_avg_vp', '?'))} VP)",
            flush=True,
        )
    else:
        model = BrassActorCritic().to(device)
        model.eval()
        print("[GA Reference Policy] Checkpoint not found; using initialized BrassActorCritic.", flush=True)

    for step_idx in range(60):
        st_res = brass_cli._http_json("GET", "/status")
        state = st_res["state"]
        if state.get("game_over"):
            break

        leg_res = brass_cli._http_json("GET", "/legal")
        actions = leg_res.get("actions") or []
        if not actions:
            break

        s_vec = encode_state(state)
        a_mat = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
        for idx, act in enumerate(actions):
            a_mat[idx] = encode_action(state, act)

        with torch.inference_mode():
            s_t = torch.from_numpy(s_vec).to(device)
            a_t = torch.from_numpy(a_mat).to(device)
            logits, _ = model.forward_single(s_t, a_t)
            if len(actions) > 1:
                unsafe_loan = a_t[:, 40] > 0.5
                if not bool(torch.all(unsafe_loan)):
                    logits = logits.masked_fill(unsafe_loan, -1e9)
            best_idx = int(torch.argmax(logits).item())

        chosen_desc = actions[best_idx].get("description", actions[best_idx].get("action_type", ""))
        era_str = str(state.get("era", "canal")).upper()
        rd_num = int(state.get("turn", 1))
        print(f"  [Step {step_idx + 1:2d} | {era_str:<5s} Rd {rd_num:2d}] -> [{best_idx:2d}] {chosen_desc}", flush=True)

        play_res = brass_cli._http_json("POST", "/play", {"action_id": best_idx})
        if not play_res.get("ok"):
            print(f"[GA Reference Policy] Error at step {step_idx}: {play_res.get('error')}", file=sys.stderr)
            return 1
        if play_res.get("game_over"):
            break

    final_status = brass_cli._http_json("GET", "/status")
    print()
    print(final_status.get("hud", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
