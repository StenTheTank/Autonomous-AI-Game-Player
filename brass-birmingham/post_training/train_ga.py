#!/usr/bin/env python3
"""
Self-contained Neuroevolution (Genetic Algorithm) Post-Training Script for Brass: Birmingham.

Evolves the 317K-parameter `BrassActorCritic` (`state_net`, `action_net`, `policy_head`,
and `action_prior` weights & biases) directly via:
  1. `self_play` mode: 2-Player population mirror-match self-play (`A vs B` and `B vs A` on shared seeds)
  2. `solo_mautoma` mode: Paired-seed Common Random Numbers evaluation against the Mautoma solo opponent
  3. Strict Elitism (Genome #0 is always the reigning Champion -> eliminates PPO's 0-VP policy collapse!)
  4. Tournament Selection + Neuron-Row / BLX-alpha Crossover + Fan-In Scaled Weight & Bias Mutation

Usage:
    # Train via 2P Population Self-Play:
    python post_training/train_ga.py --mode self_play --workers 6 --generations 20 --pop-size 12 \
        --games-per-ind 2 --out solution/brass_ppo_best.pt

    # Fine-tune or train against Solo Mautoma on paired random seeds:
    python post_training/train_ga.py --mode solo_mautoma --workers 6 --generations 10 --pop-size 12 \
        --games-per-ind 3 --out solution/brass_ppo_best.pt

    # Evaluate a checkpoint single-pass (N=1 greedy) across held-out unseen seeds:
    python post_training/train_ga.py --eval-only --checkpoint solution/brass_ppo_best.pt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
    compute_board_summary,
    encode_action_list,
    encode_state,
)
from model import (
    BrassActorCritic,
    get_rl_device,
    load_brass_checkpoint,
    save_brass_checkpoint,
)
from brass_engine.scoring import compute_brass_harbor_reward

Genome = Dict[str, torch.Tensor]
HELDOUT_SEEDS = [202611, 1001, 1002, 1003, 1004, 1005]


def get_actor_parameter_names(model: BrassActorCritic) -> List[str]:
    return [name for name, _ in model.named_parameters() if not name.startswith("value_head.")]


def extract_genome(model: BrassActorCritic) -> Genome:
    state = model.state_dict()
    return {name: state[name].detach().cpu().float().clone() for name in get_actor_parameter_names(model)}


def apply_genome(
    model: BrassActorCritic,
    genome: Genome,
    calibrated_value_hint: Optional[float] = None,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in genome:
                param.copy_(genome[name].to(device=param.device, dtype=param.dtype))
        if calibrated_value_hint is not None:
            last_linear = model.value_head[-1]
            if isinstance(last_linear, torch.nn.Linear):
                last_linear.weight.zero_()
                if last_linear.bias is not None:
                    last_linear.bias.fill_(float(calibrated_value_hint) / 15.0)


def clone_genome(genome: Genome) -> Genome:
    return {k: v.clone() for k, v in genome.items()}


def crossover_genomes(
    parent_a: Genome,
    parent_b: Genome,
    rng: Optional[random.Random] = None,
) -> Genome:
    r = rng or random
    child: Genome = {}
    if r.random() < 0.30:
        alpha = r.uniform(-0.10, 1.10)
        for name in parent_a:
            child[name] = (alpha * parent_a[name] + (1.0 - alpha) * parent_b[name]).clone()
        return child

    row_masks: Dict[str, torch.Tensor] = {}
    for name, ta in parent_a.items():
        tb = parent_b[name]
        if name.endswith(".weight") and ta.dim() == 2:
            prefix = name[: -len(".weight")]
            mask_1d = torch.rand(ta.shape[0]) < 0.5
            row_masks[prefix] = mask_1d
            child[name] = torch.where(mask_1d.unsqueeze(1).expand_as(ta), ta, tb).clone()
        elif name.endswith(".bias") and ta.dim() == 1:
            prefix = name[: -len(".bias")]
            mask_1d = row_masks.get(prefix)
            if mask_1d is None or mask_1d.shape != ta.shape:
                mask_1d = torch.rand(ta.shape[0]) < 0.5
            child[name] = torch.where(mask_1d, ta, tb).clone()
        else:
            mask = torch.rand_like(ta) < 0.5
            child[name] = torch.where(mask, ta, tb).clone()
    return child


def mutate_genome(
    genome: Genome,
    mutation_rate: float = 0.20,
    mutation_sigma: float = 0.05,
) -> Genome:
    mutated: Genome = {}
    base_rate = float(max(0.01, min(1.0, mutation_rate)))
    base_sigma = float(max(0.001, mutation_sigma))

    for name, tensor in genome.items():
        t = tensor.clone()
        if name.startswith("action_prior."):
            rate = min(1.0, base_rate * 1.85)
            sigma = base_sigma * 1.90
        elif name.startswith("policy_head.2."):
            rate = min(1.0, base_rate * 1.60)
            sigma = base_sigma * 0.65
        elif ".0." in name and ("state_net" in name or "action_net" in name):
            rate = base_rate * 0.75
            sigma = base_sigma * 0.35
        elif name.endswith(".weight") and t.dim() == 2:
            fan_in = max(1, t.shape[1])
            fan_scale = min(1.0, math.sqrt(64.0 / float(fan_in)))
            rate = base_rate
            sigma = base_sigma * fan_scale
        elif name.endswith(".bias"):
            rate = min(1.0, base_rate * 1.15)
            sigma = base_sigma * 0.55
        else:
            rate = base_rate
            sigma = base_sigma

        mask = torch.rand_like(t) < rate
        if not bool(mask.any()):
            mask.view(-1)[random.randrange(t.numel())] = True
        t.add_(torch.randn_like(t) * sigma * mask.float())
        mutated[name] = t
    return mutated


def tournament_select(
    ranked_indices: List[int],
    fitnesses: Dict[int, float],
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> int:
    r = rng or random
    contenders = r.sample(ranked_indices, min(len(ranked_indices), max(2, k)))
    return max(contenders, key=lambda idx: fitnesses[idx])


def build_self_play_schedule(
    pop_size: int,
    games_per_individual: int,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int, int]]:
    r = rng or random
    schedule: List[Tuple[int, int, int]] = []
    if pop_size < 2:
        return schedule
    mirror_rounds = games_per_individual // 2
    has_odd = (games_per_individual % 2) == 1
    for _ in range(mirror_rounds):
        s_val = r.randint(1, 999999)
        order = list(range(pop_size))
        r.shuffle(order)
        for i in range(0, pop_size - 1, 2):
            a, b = order[i], order[i + 1]
            schedule.append((a, b, s_val))
            schedule.append((b, a, s_val))
        if pop_size % 2 == 1:
            schedule.append((order[-1], order[0], s_val))
            schedule.append((order[0], order[-1], s_val))
    if has_odd:
        s_val = r.randint(1, 999999)
        order = list(range(pop_size))
        r.shuffle(order)
        for i in range(0, pop_size - 1, 2):
            schedule.append((order[i], order[i + 1], s_val))
        if pop_size % 2 == 1:
            schedule.append((order[-1], order[0], s_val))
    return schedule


def compute_ga_game_fitness(
    your_vp: int,
    opp_vp: int,
    income_pounds: int = 0,
    unflipped_risk_vp: float = 0.0,
) -> float:
    vp = float(your_vp)
    opp = float(opp_vp)
    margin = vp - opp
    win_bonus = 28.0 if vp > opp else (8.0 if vp == opp and vp > 0 else 0.0)
    collapse_penalty = -80.0 if vp <= 0.0 else (-35.0 * ((45.0 - vp) / 45.0) if vp < 45.0 else 0.0)
    tie_breaker = 0.12 * float(income_pounds) - 0.30 * float(unflipped_risk_vp)
    return float(vp + 0.35 * margin + win_bonus + collapse_penalty + tie_breaker)


def _did_turn_advance(prev_state: Dict[str, Any], next_state: Dict[str, Any]) -> bool:
    return (
        prev_state.get("turn") != next_state.get("turn")
        or prev_state.get("actions_left") != next_state.get("actions_left")
        or prev_state.get("era") != next_state.get("era")
        or bool(next_state.get("game_over"))
        or len(prev_state.get("hand") or []) != len(next_state.get("hand") or [])
        or prev_state.get("mat") != next_state.get("mat")
    )


async def _guaranteed_fallback(page: Any, state: Dict[str, Any], legal: List[Dict[str, Any]]) -> None:
    safe_loan = next((a for a in legal if a.get("action_type") == "loan"), None)
    if safe_loan is not None:
        res = await execute_macro_action(page, safe_loan, visual_delay_ms=0)
        if res.get("ok"):
            return
    hand_len = max(1, len(state.get("hand") or []))
    for slot in range(hand_len):
        res = await execute_macro_action(page, {"action_type": "pass", "card_slot": slot}, visual_delay_ms=0)
        if res.get("ok"):
            return


async def play_solo_greedy(
    page: Any,
    model: BrassActorCritic,
    device: torch.device,
    seed: int,
    difficulty: str = "easy",
) -> Dict[str, Any]:
    await start_new_brass_game(page, seed=seed, difficulty=difficulty)
    await inject_fast_training_css(page)
    state = await extract_brass_state(page, reset_open_flow=True)
    steps = 0

    while not bool(state.get("game_over")) and steps < 65:
        legal = await enumerate_legal_actions(page, state)
        if not legal:
            break
        s_vec = encode_state(state)
        a_vecs = encode_action_list(state, legal)
        with torch.inference_mode():
            s_t = torch.from_numpy(s_vec).to(device)
            a_t = torch.from_numpy(a_vecs).to(device)
            logits, _ = model.forward_single(s_t, a_t)
            # Never choose bankruptcy/Round-10 loan if a safe alternative exists
            if len(legal) > 1:
                unsafe_loan = a_t[:, 40] > 0.5
                if not bool(torch.all(unsafe_loan)):
                    logits = logits.masked_fill(unsafe_loan, -1e9)
            act_idx = int(torch.argmax(logits).item())

        prev_state = state
        res = await execute_macro_action(page, legal[act_idx], visual_delay_ms=0)
        if not res.get("ok"):
            await _guaranteed_fallback(page, state, legal)
        state = await extract_brass_state(page, reset_open_flow=True)
        if not _did_turn_advance(prev_state, state) and not state.get("game_over"):
            await _guaranteed_fallback(page, state, legal)
            state = await extract_brass_state(page, reset_open_flow=True)
        steps += 1

    summ = compute_board_summary(state)
    your_vp = int(state.get("your_vp", 0))
    automa_vp = int(state.get("automa_vp", 0))
    harbor_info = compute_brass_harbor_reward(state, steps)
    fit = compute_ga_game_fitness(
        your_vp=your_vp,
        opp_vp=automa_vp,
        income_pounds=int(summ.get("income_pounds", 0)),
        unflipped_risk_vp=float(summ.get("unflipped_tile_vp_at_risk", 0.0)),
    )
    return {
        "seed": seed,
        "your_vp": your_vp,
        "automa_vp": automa_vp,
        "won": your_vp > automa_vp,
        "steps": steps,
        "fitness": round(fit, 3),
        "harbor_reward": harbor_info["reward"],
    }


async def play_2p_self_play_match(
    page: Any,
    model_p0: BrassActorCritic,
    model_p1: BrassActorCritic,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    await start_new_brass_game(page, seed=seed, mode="2p", player_names=["Genome P1", "Genome P2"])
    await inject_fast_training_css(page)
    state = await extract_brass_state(page, reset_open_flow=True)
    steps = 0

    while not bool(state.get("game_over")) and steps < 95:
        active_seat = int(state.get("active_player", 0))
        active_model = model_p0 if active_seat == 0 else model_p1
        legal = await enumerate_legal_actions(page, state)
        if not legal:
            break
        s_vec = encode_state(state)
        a_vecs = encode_action_list(state, legal)
        with torch.inference_mode():
            s_t = torch.from_numpy(s_vec).to(device)
            a_t = torch.from_numpy(a_vecs).to(device)
            logits, _ = active_model.forward_single(s_t, a_t)
            if len(legal) > 1:
                unsafe_loan = a_t[:, 40] > 0.5
                if not bool(torch.all(unsafe_loan)):
                    logits = logits.masked_fill(unsafe_loan, -1e9)
            act_idx = int(torch.argmax(logits).item())

        prev_state = state
        res = await execute_macro_action(page, legal[act_idx], visual_delay_ms=0)
        if not res.get("ok"):
            await _guaranteed_fallback(page, state, legal)
        state = await extract_brass_state(page, reset_open_flow=True)
        if not _did_turn_advance(prev_state, state) and not state.get("game_over"):
            await _guaranteed_fallback(page, state, legal)
            state = await extract_brass_state(page, reset_open_flow=True)
        steps += 1

    p0_state = await extract_brass_state(page, reset_open_flow=False, perspective_player=0)
    p1_state = await extract_brass_state(page, reset_open_flow=False, perspective_player=1)
    p0_vp, p1_vp = int(p0_state.get("your_vp", 0)), int(p1_state.get("your_vp", 0))
    p0_sum, p1_sum = compute_board_summary(p0_state), compute_board_summary(p1_state)

    return {
        "seed": seed,
        "p0_vp": p0_vp,
        "p1_vp": p1_vp,
        "p0_won": p0_vp > p1_vp,
        "p1_won": p1_vp > p0_vp,
        "p0_fitness": round(
            compute_ga_game_fitness(p0_vp, p1_vp, int(p0_sum["income_pounds"]), float(p0_sum["unflipped_tile_vp_at_risk"])),
            3,
        ),
        "p1_fitness": round(
            compute_ga_game_fitness(p1_vp, p0_vp, int(p1_sum["income_pounds"]), float(p1_sum["unflipped_tile_vp_at_risk"])),
            3,
        ),
        "steps": steps,
    }


async def evaluate_checkpoint_suite(
    pages: List[Any],
    model: BrassActorCritic,
    device: torch.device,
    seeds: List[int],
    difficulty: str = "easy",
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(seeds):
        batch = seeds[idx : idx + len(pages)]
        batch_res = await asyncio.gather(
            *(play_solo_greedy(pages[i], model, device, batch[i], difficulty=difficulty) for i in range(len(batch)))
        )
        results.extend(batch_res)
        idx += len(batch)

    mean_vp = float(np.mean([r["your_vp"] for r in results]))
    mean_opp = float(np.mean([r["automa_vp"] for r in results]))
    win_rate = 100.0 * float(np.mean([1.0 if r["won"] else 0.0 for r in results]))
    mean_harbor = float(np.mean([r["harbor_reward"] for r in results]))
    return {
        "games": results,
        "mean_vp": round(mean_vp, 2),
        "mean_automa_vp": round(mean_opp, 2),
        "win_rate_pct": round(win_rate, 1),
        "mean_harbor_reward": round(mean_harbor, 4),
    }


async def run_ga(args: argparse.Namespace) -> None:
    device = get_rl_device()
    out_path = Path(args.out)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_path
    static_server, webapp_url = start_local_static_server(ENGINE_ROOT / "webapp", port=0)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--mute-audio"],
            )
            contexts = []
            pages = []
            for _ in range(args.workers):
                ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
                pg = await ctx.new_page()
                await pg.goto(webapp_url, wait_until="domcontentloaded")
                await inject_fast_training_css(pg)
                contexts.append(ctx)
                pages.append(pg)

            if args.eval_only:
                eval_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
                model, meta = load_brass_checkpoint(ckpt_path, device=device)
                model.eval()
                param_cnt = sum(p.numel() for p in model.parameters())
                print(f"Loaded {ckpt_path.name} ({param_cnt:,} params) | meta={meta}")
                suite = await evaluate_checkpoint_suite(pages, model, device, eval_seeds, difficulty=args.difficulty)
                for g in suite["games"]:
                    tag = "WIN " if g["won"] else "LOSS"
                    print(
                        f"  Seed {g['seed']:6d}: [{tag}] Agent {g['your_vp']:3d} VP vs Mautoma {g['automa_vp']:3d} VP "
                        f"({g['your_vp'] - g['automa_vp']:+d} VP) -> Harbor Reward = {g['harbor_reward']:.6f}"
                    )
                print(
                    f"Summary ({len(eval_seeds)} seeds): Mean {suite['mean_vp']:.1f} VP vs {suite['mean_automa_vp']:.1f} VP | "
                    f"Win Rate {suite['win_rate_pct']:.1f}% | Mean Harbor Reward {suite['mean_harbor_reward']:.4f}"
                )
                for ctx in contexts:
                    await ctx.close()
                await browser.close()
                return

            # Build initial population (Genome #0 = warm prior, Genome #1 = resumed checkpoint if present)
            fresh_model = BrassActorCritic()
            seeds_list: List[Genome] = [extract_genome(fresh_model)]
            if ckpt_path.exists() and not args.fresh:
                loaded_m, loaded_meta = load_brass_checkpoint(ckpt_path, device=torch.device("cpu"))
                # Put existing champion at #0 so strict elitism preserves it!
                seeds_list = [extract_genome(loaded_m), extract_genome(fresh_model)]
                print(f"Seeded Gen 1 from {ckpt_path.name} (meta={loaded_meta}) + domain anchor")
            else:
                print("Seeded Gen 1 from fresh BrassActorCritic")

            pop_size = max(4, args.pop_size)
            population: List[Genome] = [clone_genome(g) for g in seeds_list[:pop_size]]
            sigmas = [0.5 * args.sigma, 1.0 * args.sigma, 1.5 * args.sigma]
            idx = 0
            while len(population) < pop_size:
                parent = seeds_list[idx % len(seeds_list)]
                population.append(mutate_genome(parent, mutation_rate=args.mutation_rate, mutation_sigma=sigmas[idx % 3]))
                idx += 1

            worker_m0 = [BrassActorCritic().to(device).eval() for _ in range(args.workers)]
            worker_m1 = [BrassActorCritic().to(device).eval() for _ in range(args.workers)]
            saver_model = BrassActorCritic().to(torch.device("cpu"))
            rng = random.Random(args.rng_seed)
            stagnant_gens = 0
            completed_matches = 0
            t0 = time.time()

            log_csv_path = Path(args.log_csv) if args.log_csv else None
            if log_csv_path:
                log_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with log_csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(
                        ["generation", "mode", "pop_mean_vp", "champion_vp", "champion_fitness", "win_rate_pct", "sigma"]
                    )

            for gen in range(1, args.generations + 1):
                prog = (gen - 1) / max(1, args.generations - 1)
                cos_factor = 0.55 + 0.45 * (0.5 * (1.0 + math.cos(math.pi * prog)))
                active_sigma = round(args.sigma * cos_factor * (1.45 if stagnant_gens >= 4 else 1.0), 5)
                results_by_genome: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(pop_size)}
                lock = asyncio.Lock()

                if args.mode == "self_play":
                    sched = build_self_play_schedule(pop_size, args.games_per_ind, rng=rng)
                    q_sp: asyncio.Queue[Optional[Tuple[int, int, int]]] = asyncio.Queue()
                    for item in sched:
                        q_sp.put_nowait(item)
                    for _ in range(args.workers):
                        q_sp.put_nowait(None)

                    async def worker_sp(w_id: int) -> None:
                        nonlocal completed_matches
                        while True:
                            job = await q_sp.get()
                            if job is None:
                                break
                            p0_i, p1_i, s_val = job
                            apply_genome(worker_m0[w_id], population[p0_i])
                            apply_genome(worker_m1[w_id], population[p1_i])
                            res = await play_2p_self_play_match(pages[w_id], worker_m0[w_id], worker_m1[w_id], device, s_val)
                            async with lock:
                                results_by_genome[p0_i].append(
                                    {"your_vp": res["p0_vp"], "opp_vp": res["p1_vp"], "won": res["p0_won"], "fitness": res["p0_fitness"]}
                                )
                                results_by_genome[p1_i].append(
                                    {"your_vp": res["p1_vp"], "opp_vp": res["p0_vp"], "won": res["p1_won"], "fitness": res["p1_fitness"]}
                                )
                                completed_matches += 1

                    await asyncio.gather(*(worker_sp(w) for w in range(args.workers)))
                else:
                    # Exclude evaluation seed 202611 and held-out seeds 1001..1010 from training!
                    gen_seeds: List[int] = []
                    while len(gen_seeds) < args.games_per_ind:
                        cand_s = rng.randint(5000, 999999)
                        if cand_s != 202611:
                            gen_seeds.append(cand_s)
                    q_solo: asyncio.Queue[Optional[Tuple[int, int]]] = asyncio.Queue()
                    for g_i in range(pop_size):
                        for s_val in gen_seeds:
                            q_solo.put_nowait((g_i, s_val))
                    for _ in range(args.workers):
                        q_solo.put_nowait(None)

                    async def worker_solo(w_id: int) -> None:
                        nonlocal completed_matches
                        loaded_g = -1
                        while True:
                            job = await q_solo.get()
                            if job is None:
                                break
                            g_i, s_val = job
                            if loaded_g != g_i:
                                apply_genome(worker_m0[w_id], population[g_i])
                                loaded_g = g_i
                            res = await play_solo_greedy(
                                pages[w_id], worker_m0[w_id], device, s_val, difficulty=args.difficulty
                            )
                            async with lock:
                                results_by_genome[g_i].append(res)
                                completed_matches += 1

                    await asyncio.gather(*(worker_solo(w) for w in range(args.workers)))

                mean_vp_map = {i: float(np.mean([x["your_vp"] for x in gl])) for i, gl in results_by_genome.items() if gl}
                win_map = {i: float(np.mean([1.0 if x["won"] else 0.0 for x in gl])) for i, gl in results_by_genome.items() if gl}
                fit_map = {i: float(np.mean([x["fitness"] for x in gl])) for i, gl in results_by_genome.items() if gl}

                ranked = sorted(
                    fit_map.keys(),
                    key=lambda idx: (round(fit_map[idx], 4), round(mean_vp_map[idx], 3), round(win_map[idx], 3), 1 if idx == 0 else 0),
                    reverse=True,
                )
                win_idx = ranked[0]
                champ_vp = mean_vp_map[win_idx]
                champ_fit = fit_map[win_idx]
                champ_win_pct = 100.0 * win_map[win_idx]
                pop_mean = float(np.mean(list(mean_vp_map.values())))
                dethroned = (gen == 1) or (win_idx != 0 and champ_fit > fit_map.get(0, -1e9) + 1e-4)
                champion_genome = clone_genome(population[win_idx])

                if dethroned:
                    apply_genome(saver_model, champion_genome, calibrated_value_hint=champ_vp)
                    save_brass_checkpoint(
                        saver_model,
                        optimizer=None,
                        path=out_path,
                        metadata={
                            "algorithm": "genetic_algorithm",
                            "training_mode": args.mode,
                            "generation": gen,
                            "completed_episodes": completed_matches,
                            "champion_avg_vp": round(champ_vp, 2),
                            "champion_fitness": round(champ_fit, 3),
                            "generation_mean_vp": round(pop_mean, 2),
                            "rolling_avg_vp": round(champ_vp, 2),
                            "rolling_win_rate_pct": round(champ_win_pct, 1),
                        },
                    )
                    stagnant_gens = 0
                    badge = f" [PROMOTED -> {out_path.name}]"
                else:
                    stagnant_gens += 1
                    badge = " [Crown Defended]"

                if log_csv_path:
                    with log_csv_path.open("a", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerow(
                            [gen, args.mode, round(pop_mean, 2), round(champ_vp, 2), round(champ_fit, 3), round(champ_win_pct, 1), active_sigma]
                        )

                gpm = completed_matches / max(1e-6, (time.time() - t0) / 60.0)
                print(
                    f"[Gen {gen:2d}/{args.generations}] PopMean={pop_mean:5.1f} VP | "
                    f"Champ={champ_vp:5.1f} VP (Win={champ_win_pct:5.1f}%, Fit={champ_fit:6.1f}) | "
                    f"{gpm:4.1f} matches/min{badge}",
                    flush=True,
                )

                if gen >= args.generations:
                    break

                # Breed next generation
                next_pop: List[Genome] = [clone_genome(population[ranked[e]]) for e in range(min(args.elite_count, len(ranked)))]
                if len(next_pop) < pop_size:
                    next_pop.append(
                        mutate_genome(champion_genome, mutation_rate=max(0.08, args.mutation_rate * 0.65), mutation_sigma=active_sigma * 0.45)
                    )
                reserve_exp = 1 if (pop_size - len(next_pop)) >= 2 else 0
                while len(next_pop) < pop_size - reserve_exp:
                    pa = tournament_select(ranked, fit_map, k=3, rng=rng)
                    pb = tournament_select(ranked, fit_map, k=3, rng=rng)
                    if pa == pb and len(ranked) > 1:
                        pb = ranked[1] if pa == ranked[0] else ranked[0]
                    child = crossover_genomes(population[pa], population[pb], rng=rng)
                    next_pop.append(mutate_genome(child, mutation_rate=args.mutation_rate, mutation_sigma=active_sigma))
                if reserve_exp:
                    next_pop.append(
                        mutate_genome(champion_genome, mutation_rate=min(0.60, args.mutation_rate * 1.65), mutation_sigma=active_sigma * 1.75)
                    )
                population = next_pop

            for ctx in contexts:
                await ctx.close()
            await browser.close()
    finally:
        static_server.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description="Neuroevolution (GA) Trainer & Multi-Seed Evaluator for Brass: Birmingham.")
    p.add_argument("--mode", choices=["self_play", "solo_mautoma"], default="self_play")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop-size", type=int, default=12)
    p.add_argument("--games-per-ind", type=int, default=2)
    p.add_argument("--elite-count", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--mutation-rate", type=float, default=0.20)
    p.add_argument("--difficulty", type=str, default="easy")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--seeds", type=str, default="202611,1001,1002,1003,1004,1005")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--out", type=str, default=str(TASK_ROOT / "solution" / "brass_ppo_best.pt"))
    p.add_argument("--log-csv", type=str, default="")
    asyncio.run(run_ga(p.parse_args()))


if __name__ == "__main__":
    main()
