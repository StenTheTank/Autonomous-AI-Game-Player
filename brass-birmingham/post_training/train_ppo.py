#!/usr/bin/env python3
"""
Self-contained Parallel Headless PPO Post-Training Script for Brass: Birmingham.

Trains `BrassActorCritic` (288-dim board/economy state encoder + 48-dim variable
candidate-action scorer + GAE-lambda + clipped PPO) directly against the bundled
offline Brass: Birmingham simulator (`environment/webapp/`).

Usage:
    python post_training/train_ppo.py --workers 6 --episodes 200 --lr 3e-4 \
        --out solution/brass_ppo_best.pt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

TASK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = TASK_ROOT / "environment"
SOLUTION_ROOT = TASK_ROOT / "solution"
for _p in (SOLUTION_ROOT, ENGINE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from playwright.async_api import async_playwright
from brass_engine.brass_bridge import (
    enumerate_legal_actions,
    execute_macro_action,
    extract_brass_state,
    inject_fast_training_css,
    start_new_brass_game,
)
from brass_engine.brass_daemon import start_local_static_server
from brass_engine.brass_features import (
    ACTION_DIM,
    STATE_DIM,
    compute_board_summary,
    encode_action,
    encode_state,
    get_connected_component_and_network,
)
from model import BrassActorCritic, get_rl_device, load_brass_checkpoint, save_brass_checkpoint


def board_potential(state: Dict[str, Any], summary: Dict[str, Any]) -> float:
    era_is_canal = state.get("era", "canal") == "canal"
    turn = int(state.get("turn", 1))
    unflipped_weight = 0.25 if turn <= 7 else (0.12 if turn <= 9 else 0.02)
    banked_vp = float(state.get("your_vp", 0) - state.get("automa_vp", 0))
    era_proj_vp = float(summary["flipped_tile_vp"] + summary["connected_link_vp"])
    pending_vp = unflipped_weight * float(summary["unflipped_tile_vp_at_risk"])
    income_term = (0.35 if era_is_canal else 0.15) * float(summary["income_pounds"])
    cash_term = 0.04 * min(float(state.get("money", 0)), 40.0)
    return (banked_vp + era_proj_vp + pending_vp + income_term + cash_term) / 25.0


@dataclass
class StepTransition:
    state_vec: np.ndarray
    cand_vecs: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float
    done: bool


async def run_training(args: argparse.Namespace) -> None:
    device = get_rl_device()
    out_path = Path(args.out)

    if out_path.exists() and not args.fresh:
        model, meta = load_brass_checkpoint(out_path, device=device)
        model.train()
        best_rolling_vp = float(meta.get("rolling_avg_vp", -999.0))
        start_ep = int(meta.get("episodes_trained", 0))
        print(f"Resumed {out_path} (ep={start_ep}, best_roll20={best_rolling_vp:.1f} VP) on {device}")
    else:
        model = BrassActorCritic().to(device)
        model.train()
        best_rolling_vp = -999.0
        start_ep = 0
        print(f"Initialized fresh BrassActorCritic on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    static_server, webapp_url = start_local_static_server(ENGINE_ROOT / "webapp", port=0)

    recent_vps: deque = deque(maxlen=20)
    recent_wins: deque = deque(maxlen=20)
    completed = 0
    best_vp = 0
    t0 = time.time()

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            pages = []
            contexts = []
            for _ in range(args.workers):
                ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
                pg = await ctx.new_page()
                await pg.goto(webapp_url, wait_until="domcontentloaded")
                await inject_fast_training_css(pg)
                contexts.append(ctx)
                pages.append(pg)

            async def collect_one_episode(page: Any) -> Tuple[List[StepTransition], int, int, bool]:
                await start_new_brass_game(page, difficulty=args.difficulty)
                st = await extract_brass_state(page)
                summ = compute_board_summary(st)
                prev_phi = board_potential(st, summ)
                traj: List[StepTransition] = []

                for _ in range(55):
                    if st.get("game_over"):
                        break
                    legal = await enumerate_legal_actions(page, st)
                    if not legal:
                        break
                    s_vec = encode_state(st)
                    c_mat = np.zeros((len(legal), ACTION_DIM), dtype=np.float32)
                    for i, act in enumerate(legal):
                        c_mat[i] = encode_action(st, act)

                    with torch.no_grad():
                        s_t = torch.from_numpy(s_vec).to(device)
                        c_t = torch.from_numpy(c_mat).to(device)
                        logits, val = model.forward_single(s_t, c_t)
                        dist = Categorical(logits=logits)
                        chosen_t = dist.sample()
                        a_idx = int(chosen_t.item())
                        lp = float(dist.log_prob(chosen_t).item())
                        v_num = float(val.item())

                    await execute_macro_action(page, legal[a_idx])
                    next_st = await extract_brass_state(page)
                    next_summ = compute_board_summary(next_st)
                    done = bool(next_st.get("game_over"))
                    next_phi = board_potential(next_st, next_summ)

                    rew = 0.995 * next_phi - prev_phi
                    if done:
                        margin = float(next_st.get("your_vp", 0) - next_st.get("automa_vp", 0))
                        rew += float(next_st.get("your_vp", 0)) / 30.0 + margin / 40.0 + (1.5 if margin > 0 else 0.0)

                    traj.append(StepTransition(s_vec, c_mat, a_idx, lp, v_num, rew, done))
                    st, summ, prev_phi = next_st, next_summ, next_phi
                    if done:
                        break

                y_vp = int(st.get("your_vp", 0))
                a_vp = int(st.get("automa_vp", 0))
                return traj, y_vp, a_vp, bool(y_vp > a_vp)

            while completed < args.episodes:
                batch_n = min(args.workers, args.episodes - completed)
                results = await asyncio.gather(*(collect_one_episode(pages[i]) for i in range(batch_n)))

                flat: List[StepTransition] = []
                advantages: List[float] = []
                returns: List[float] = []

                for traj, y_vp, a_vp, won in results:
                    if not traj:
                        continue
                    completed += 1
                    best_vp = max(best_vp, y_vp)
                    recent_vps.append(y_vp)
                    recent_wins.append(1.0 if won else 0.0)

                    gae = 0.0
                    ep_adv = [0.0] * len(traj)
                    ep_ret = [0.0] * len(traj)
                    for t in reversed(range(len(traj))):
                        nxt_v = 0.0 if (t == len(traj) - 1 or traj[t].done) else traj[t + 1].value
                        mask = 0.0 if traj[t].done else 1.0
                        delta = traj[t].reward + 0.995 * nxt_v * mask - traj[t].value
                        gae = delta + 0.995 * 0.95 * mask * gae
                        ep_adv[t] = gae
                        ep_ret[t] = gae + traj[t].value
                    flat.extend(traj)
                    advantages.extend(ep_adv)
                    returns.extend(ep_ret)

                # PPO minibatch update
                bsz = len(flat)
                max_k = max(tr.cand_vecs.shape[0] for tr in flat)
                s_np = np.zeros((bsz, STATE_DIM), dtype=np.float32)
                c_np = np.zeros((bsz, max_k, ACTION_DIM), dtype=np.float32)
                m_np = np.zeros((bsz, max_k), dtype=bool)
                a_np = np.zeros(bsz, dtype=np.int64)
                lp_np = np.zeros(bsz, dtype=np.float32)

                for i, tr in enumerate(flat):
                    k = tr.cand_vecs.shape[0]
                    s_np[i] = tr.state_vec
                    c_np[i, :k] = tr.cand_vecs
                    m_np[i, :k] = True
                    a_np[i] = tr.action_idx
                    lp_np[i] = tr.log_prob

                s_t = torch.from_numpy(s_np).to(device)
                c_t = torch.from_numpy(c_np).to(device)
                m_t = torch.from_numpy(m_np).to(device)
                a_t = torch.from_numpy(a_np).to(device)
                old_lp_t = torch.from_numpy(lp_np).to(device)
                adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
                ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                for _ in range(4):
                    perm = torch.randperm(bsz, device=device)
                    for start in range(0, bsz, 64):
                        mb = perm[start : start + 64]
                        logits, vals = model.forward_padded_batch(s_t[mb], c_t[mb], m_t[mb])
                        dist = Categorical(logits=logits)
                        new_lp = dist.log_prob(a_t[mb])
                        ratio = torch.exp(new_lp - old_lp_t[mb])
                        surr1 = ratio * adv_t[mb]
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[mb]
                        loss = (
                            -torch.min(surr1, surr2).mean()
                            + 0.5 * F.smooth_l1_loss(vals, ret_t[mb])
                            - 0.015 * dist.entropy().mean()
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                roll_vp = float(np.mean(recent_vps))
                win_pct = 100.0 * float(np.mean(recent_wins))
                gpm = completed / max(1e-6, (time.time() - t0) / 60.0)
                saved_note = ""
                if len(recent_vps) >= 10 and roll_vp >= best_rolling_vp:
                    best_rolling_vp = roll_vp
                    save_brass_checkpoint(
                        model,
                        optimizer=optimizer,
                        path=out_path,
                        metadata={
                            "episodes_trained": start_ep + completed,
                            "rolling_avg_vp": round(roll_vp, 2),
                            "rolling_win_rate_pct": round(win_pct, 2),
                            "best_vp": best_vp,
                        },
                    )
                    saved_note = f" [SAVED -> {out_path.name}]"

                print(
                    f"[Ep {completed:4d}/{args.episodes}] Roll20={roll_vp:5.1f} VP "
                    f"(Win={win_pct:4.1f}%, Best={best_vp:3d} VP) | {gpm:4.1f} games/min{saved_note}",
                    flush=True,
                )

            for ctx in contexts:
                await ctx.close()
            await browser.close()
    finally:
        static_server.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description="Train BrassActorCritic via parallel headless PPO.")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--difficulty", type=str, default="easy")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out", type=str, default=str(TASK_ROOT / "solution" / "brass_ppo_best.pt"))
    asyncio.run(run_training(p.parse_args()))


if __name__ == "__main__":
    main()
