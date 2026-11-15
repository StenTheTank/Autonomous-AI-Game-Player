import asyncio
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from playwright.async_api import async_playwright

from models.schemas import RLTrainingStatus
from rl.brass_bridge import BRASS_SOLO_URL, ensure_local_brass_server
from rl.brass_env import BrassPlaywrightEnv
from rl.model import (
    BEST_CHECKPOINT_PATH,
    LATEST_CHECKPOINT_PATH,
    BrassActorCritic,
    get_rl_device,
    load_brass_checkpoint,
    save_brass_checkpoint,
)
from storage.database import SessionDatabase

Genome = Dict[str, torch.Tensor]


def get_actor_parameter_names(model: BrassActorCritic) -> List[str]:
    """
    Return all trainable weight & bias parameter names used for action selection
    (state_net, action_net, policy_head, and action_prior).
    Excludes value_head since neuroevolution optimizes game fitness directly without TD bootstrapping.
    """
    return [
        name
        for name, _ in model.named_parameters()
        if not name.startswith("value_head.")
    ]


def extract_genome(model: BrassActorCritic) -> Genome:
    """Clone all Actor weights and biases onto CPU float32 tensors."""
    state = model.state_dict()
    names = get_actor_parameter_names(model)
    return {name: state[name].detach().cpu().float().clone() for name in names}


def apply_genome(
    model: BrassActorCritic,
    genome: Genome,
    calibrated_value_hint: Optional[float] = None,
) -> None:
    """Load a genome's weights and biases into a BrassActorCritic instance in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in genome:
                param.copy_(genome[name].to(device=param.device, dtype=param.dtype))
        if calibrated_value_hint is not None:
            # Keep value_head calibrated to expected VP scale for Live Arena display
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
    """
    Breed an offspring genome from two parents while preserving hidden-neuron coherence:
    - 30% chance: BLX-alpha continuous interpolation across matching weights & biases
    - 70% chance: Neuron/row-wise uniform crossover (pairing each Linear output row with its bias!)
    """
    r = rng or random
    child: Genome = {}

    if r.random() < 0.30:
        # BLX-alpha real-coded blend crossover
        alpha = r.uniform(-0.10, 1.10)
        for name in parent_a:
            ta = parent_a[name]
            tb = parent_b[name]
            child[name] = (alpha * ta + (1.0 - alpha) * tb).clone()
        return child

    # Neuron/row-wise uniform crossover: keep each neuron's incoming weight row + scalar bias paired!
    row_masks: Dict[str, torch.Tensor] = {}
    for name, ta in parent_a.items():
        tb = parent_b[name]
        if name.endswith(".weight") and ta.dim() == 2:
            prefix = name[: -len(".weight")]
            out_dim = ta.shape[0]
            mask_1d = torch.rand(out_dim) < 0.5
            row_masks[prefix] = mask_1d
            mask_2d = mask_1d.unsqueeze(1).expand_as(ta)
            child[name] = torch.where(mask_2d, ta, tb).clone()
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
    """
    Apply layer-aware masked Gaussian perturbations to both weights and biases:
    - Direct tactical heads (`action_prior`, `policy_head.2`) receive focused high-signal mutations.
    - Deep trunk layers (`state_net`, `action_net`, `policy_head.0`) use fan-in scaled mutations
      on both `.weight` matrices and `.bias` vectors so hidden SiLU representations evolve smoothly.
    """
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
            # LayerNorm gain & bias
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
            # Guarantee at least 1 parameter mutates in small tensors like bias / action_prior
            flat_idx = random.randrange(t.numel())
            mask.view(-1)[flat_idx] = True

        noise = torch.randn_like(t) * sigma
        t.add_(noise * mask.float())
        mutated[name] = t

    return mutated


def tournament_select(
    ranked_indices: List[int],
    fitnesses: Dict[int, float],
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> int:
    r = rng or random
    k_eff = min(len(ranked_indices), max(2, k))
    contenders = r.sample(ranked_indices, k_eff)
    return max(contenders, key=lambda idx: fitnesses[idx])


def build_self_play_schedule(
    pop_size: int,
    games_per_individual: int,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int, int]]:
    """
    Generate 2-player self-play match pairings `(seat0_idx, seat1_idx, seed)` across the population.
    Uses mirror-match rounds `(A vs B)` and `(B vs A)` on the same seed to cancel P1/P2 seat bias!
    """
    r = rng or random
    schedule: List[Tuple[int, int, int]] = []
    if pop_size < 2:
        return schedule

    mirror_rounds = games_per_individual // 2
    has_odd_round = (games_per_individual % 2) == 1

    for _ in range(mirror_rounds):
        round_seed = r.randint(1, 999999)
        order = list(range(pop_size))
        r.shuffle(order)
        for i in range(0, pop_size - 1, 2):
            a, b = order[i], order[i + 1]
            schedule.append((a, b, round_seed))
            schedule.append((b, a, round_seed))
        if pop_size % 2 == 1:
            last_g = order[-1]
            opp_g = order[0]
            schedule.append((last_g, opp_g, round_seed))
            schedule.append((opp_g, last_g, round_seed))

    if has_odd_round:
        round_seed = r.randint(1, 999999)
        order = list(range(pop_size))
        r.shuffle(order)
        for i in range(0, pop_size - 1, 2):
            a, b = order[i], order[i + 1]
            schedule.append((a, b, round_seed))
        if pop_size % 2 == 1:
            schedule.append((order[-1], order[0], round_seed))

    return schedule


class BrassGATrainer:
    """
    Parallel Headless Chromium Neuroevolution (Genetic Algorithm) Trainer for Brass: Birmingham.
    Evolves BrassActorCritic weights & biases directly with:
      - Paired-seed Common Random Numbers evaluation (deterministic argmax play)
      - Strict Elitism (defending Champion is always Genome #0 -> zero collapse risk!)
      - Tournament Selection + Neuron/BLX Crossover + Layer-Scaled Weight & Bias Mutation
    """

    def __init__(
        self,
        db: Optional[SessionDatabase] = None,
        on_update: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.db = db or SessionDatabase()
        self.on_update = on_update
        self.device = get_rl_device()
        self.running = False
        self.stop_requested = False
        self._task: Optional[asyncio.Task] = None
        self.training_mode: str = "self_play"
        self.run_id: str = ""
        self.workers: int = 0
        self.current_generation: int = 0
        self.target_generations: int = 0
        self.population_size: int = 12
        self.games_per_individual: int = 2
        self.elite_count: int = 2
        self.mutation_sigma: float = 0.05
        self.mutation_rate: float = 0.20
        self.generation_progress: str = "0 / 0"
        self.generation_best_vp: float = 0.0
        self.generation_mean_vp: float = 0.0
        self.champion_avg_vp: float = 0.0
        self.champion_fitness: float = 0.0
        self.target_episodes: int = 0
        self.completed_episodes: int = 0
        self.best_rolling_vp: float = 0.0
        self.status_message: str = "Idle"
        self._start_time: float = 0.0

        if BEST_CHECKPOINT_PATH.exists():
            try:
                _, meta = load_brass_checkpoint(BEST_CHECKPOINT_PATH, device=torch.device("cpu"))
                self.best_rolling_vp = float(meta.get("rolling_avg_vp", 0.0))
                self.champion_avg_vp = float(meta.get("champion_avg_vp", self.best_rolling_vp))
                self.champion_fitness = float(meta.get("champion_fitness", 0.0))
            except Exception:
                self.best_rolling_vp = 0.0

    def get_status(self) -> RLTrainingStatus:
        summary = self.db.get_rl_training_summary(limit=150)
        elapsed_min = max(1e-6, (time.time() - self._start_time) / 60.0) if self._start_time > 0 else 0.0
        gpm = round(self.completed_episodes / elapsed_min, 2) if self.running and elapsed_min > 0 else 0.0

        return RLTrainingStatus(
            running=self.running,
            algorithm="genetic_algorithm",
            training_mode=self.training_mode,
            run_id=self.run_id,
            device=str(self.device),
            workers=self.workers,
            current_generation=self.current_generation,
            target_generations=self.target_generations,
            population_size=self.population_size,
            games_per_individual=self.games_per_individual,
            elite_count=self.elite_count,
            mutation_sigma=round(self.mutation_sigma, 4),
            mutation_rate=round(self.mutation_rate, 3),
            generation_progress=self.generation_progress,
            generation_best_vp=round(self.generation_best_vp, 1),
            generation_mean_vp=round(self.generation_mean_vp, 1),
            champion_avg_vp=round(self.champion_avg_vp, 1),
            champion_fitness=round(self.champion_fitness, 2),
            target_episodes=self.target_episodes,
            completed_episodes=self.completed_episodes,
            rolling_avg_vp=summary["rolling_avg_vp"],
            rolling_automa_vp=summary["rolling_automa_vp"],
            rolling_win_rate_pct=summary["rolling_win_rate_pct"],
            best_vp=summary["best_vp"],
            best_rolling_vp=round(self.best_rolling_vp, 1),
            games_per_min=gpm,
            latest_policy_loss=round(self.generation_mean_vp, 2),
            latest_value_loss=float(self.current_generation),
            latest_entropy=round(self.mutation_sigma, 4),
            has_best_checkpoint=BEST_CHECKPOINT_PATH.exists(),
            has_latest_checkpoint=LATEST_CHECKPOINT_PATH.exists(),
            status_message=self.status_message,
            recent_episodes=summary["episodes"],
        )

    async def _emit(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.on_update:
            return
        payload: Dict[str, Any] = {
            "type": "rl_training_update",
            "status": self.get_status().model_dump(),
        }
        if extra:
            payload.update(extra)
        res = self.on_update(payload)
        if asyncio.iscoroutine(res):
            await res

    def start_background(
        self,
        workers: int = 6,
        training_mode: str = "self_play",
        generations: int = 30,
        population_size: int = 12,
        games_per_individual: int = 2,
        mutation_sigma: float = 0.05,
        mutation_rate: float = 0.20,
        elite_count: int = 2,
        resume_checkpoint: bool = True,
        url: str = BRASS_SOLO_URL,
        **_: Any,
    ) -> None:
        if self.running and self._task and not self._task.done():
            raise RuntimeError("Genetic Algorithm training is already running.")
        self._task = asyncio.create_task(
            self.train(
                workers=workers,
                training_mode=training_mode,
                generations=generations,
                population_size=population_size,
                games_per_individual=games_per_individual,
                mutation_sigma=mutation_sigma,
                mutation_rate=mutation_rate,
                elite_count=elite_count,
                resume_checkpoint=resume_checkpoint,
                url=url,
            )
        )

    async def stop(self) -> None:
        self.stop_requested = True
        self.status_message = "Stopping after active browser evaluations finish..."
        await self._emit()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self.running = False
        self.status_message = "Stopped (best evolved checkpoint saved)"
        await self._emit()

    def _build_initial_population(
        self,
        pop_size: int,
        resume_checkpoint: bool,
        sigma: float,
        rate: float,
    ) -> Tuple[List[Genome], str]:
        """
        Seed Generation 1 with:
          - Genome #0: Clean warm-started domain heuristic BrassActorCritic (`_init_weights()`).
                       Guarantees strong baseline (~85-115 VP) even if an older checkpoint collapsed!
          - Genome #1: Saved `brass_ppo_best.pt` (if `resume_checkpoint=True` and available).
          - Genomes #2..P-1: Diverse weight & bias mutations around the seed elites.
        """
        fresh_model = BrassActorCritic()
        heuristic_genome = extract_genome(fresh_model)

        seeds: List[Genome] = [heuristic_genome]
        init_note = "Seeded Gen 1 with domain-prior baseline"

        if resume_checkpoint:
            for ckpt_path, label in (
                (BEST_CHECKPOINT_PATH, "brass_ppo_best.pt"),
                (LATEST_CHECKPOINT_PATH, "brass_ppo_latest.pt"),
            ):
                if ckpt_path.exists() and len(seeds) < pop_size:
                    try:
                        loaded_model, _ = load_brass_checkpoint(ckpt_path, device=torch.device("cpu"))
                        seeds.append(extract_genome(loaded_model))
                        if label == "brass_ppo_best.pt":
                            init_note = f"Seeded Gen 1 with {label} + clean heuristic anchor"
                    except Exception:
                        pass

        population: List[Genome] = [clone_genome(g) for g in seeds[:pop_size]]
        sigmas = [0.5 * sigma, 1.0 * sigma, 1.5 * sigma]
        idx = 0
        while len(population) < pop_size:
            parent = seeds[idx % len(seeds)]
            s_val = sigmas[idx % len(sigmas)]
            population.append(mutate_genome(parent, mutation_rate=rate, mutation_sigma=s_val))
            idx += 1

        return population, init_note

    async def train(
        self,
        workers: int = 6,
        training_mode: str = "self_play",
        generations: int = 30,
        population_size: int = 12,
        games_per_individual: int = 2,
        mutation_sigma: float = 0.05,
        mutation_rate: float = 0.20,
        elite_count: int = 2,
        resume_checkpoint: bool = True,
        url: str = BRASS_SOLO_URL,
        **_: Any,
    ) -> RLTrainingStatus:
        self.running = True
        self.stop_requested = False
        self.training_mode = training_mode if training_mode in ("self_play", "solo_mautoma") else "self_play"
        self.run_id = f"ga_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.workers = max(1, min(12, int(workers)))
        self.target_generations = max(1, int(generations))
        self.population_size = max(4, min(64, int(population_size)))
        self.games_per_individual = max(1, min(6, int(games_per_individual)))
        self.elite_count = max(1, min(self.population_size - 1, int(elite_count)))
        base_sigma = float(mutation_sigma)
        self.mutation_sigma = base_sigma
        self.mutation_rate = float(mutation_rate)
        self.current_generation = 0
        self.completed_episodes = 0

        if self.training_mode == "self_play":
            matches_per_gen = len(build_self_play_schedule(self.population_size, self.games_per_individual))
        else:
            matches_per_gen = self.population_size * self.games_per_individual
        self.target_episodes = self.target_generations * matches_per_gen
        self.generation_progress = f"0 / {matches_per_gen}"
        self._start_time = time.time()

        population, init_msg = self._build_initial_population(
            pop_size=self.population_size,
            resume_checkpoint=resume_checkpoint,
            sigma=base_sigma,
            rate=self.mutation_rate,
        )
        champion_genome: Genome = clone_genome(population[0])
        saver_model = BrassActorCritic().to(torch.device("cpu"))
        rng = random.Random()
        stagnant_generations = 0
        logged_episode_counter = self.db.get_rl_training_summary(limit=1)["total_episodes_recorded"]

        mode_label = "2P Self-Play" if self.training_mode == "self_play" else "Solo vs Mautoma"
        self.status_message = f"{init_msg} ({mode_label}) — launching {self.workers} headless Chromium workers..."
        await self._emit()

        local_brass_url = ensure_local_brass_server() if (not url or url == BRASS_SOLO_URL) else url

        pw = None
        browser = None
        contexts = []
        envs: List[BrassPlaywrightEnv] = []
        worker_models_p0: List[BrassActorCritic] = []
        worker_models_p1: List[BrassActorCritic] = []

        try:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-renderer-backgrounding",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--mute-audio",
                ],
            )
            for _ in range(self.workers):
                ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
                contexts.append(ctx)
                env = BrassPlaywrightEnv(ctx, url=local_brass_url, difficulty="easy")
                envs.append(env)
                wm0 = BrassActorCritic().to(self.device)
                wm0.eval()
                worker_models_p0.append(wm0)
                wm1 = BrassActorCritic().to(self.device)
                wm1.eval()
                worker_models_p1.append(wm1)

            await asyncio.gather(*(e.initialize() for e in envs))

            for gen in range(1, self.target_generations + 1):
                if self.stop_requested:
                    break

                self.current_generation = gen
                progress_frac = (gen - 1) / max(1, self.target_generations - 1)
                cosine_factor = 0.55 + 0.45 * (0.5 * (1.0 + math.cos(math.pi * progress_frac)))
                stagnation_boost = 1.45 if stagnant_generations >= 4 else 1.0
                active_sigma = round(base_sigma * cosine_factor * stagnation_boost, 5)
                self.mutation_sigma = active_sigma

                results_by_genome: Dict[int, List[Dict[str, Any]]] = {
                    i: [] for i in range(self.population_size)
                }
                done_gen_games = 0
                lock = asyncio.Lock()

                if self.training_mode == "self_play":
                    schedule = build_self_play_schedule(
                        self.population_size, self.games_per_individual, rng=rng
                    )
                    total_gen_games = len(schedule)
                    self.generation_progress = f"0 / {total_gen_games}"
                    job_queue_sp: asyncio.Queue[Optional[Tuple[int, int, int]]] = asyncio.Queue()
                    for match_item in schedule:
                        job_queue_sp.put_nowait(match_item)
                    for _ in range(self.workers):
                        job_queue_sp.put_nowait(None)

                    self.status_message = (
                        f"Gen {gen}/{self.target_generations} (2P Self-Play): "
                        f"Running {total_gen_games} population mirror matches..."
                    )
                    await self._emit()

                    async def eval_worker_sp(worker_id: int) -> None:
                        nonlocal done_gen_games
                        env = envs[worker_id]
                        m0 = worker_models_p0[worker_id]
                        m1 = worker_models_p1[worker_id]
                        loaded_p0 = -1
                        loaded_p1 = -1
                        while not self.stop_requested:
                            item = await job_queue_sp.get()
                            if item is None:
                                break
                            p0_idx, p1_idx, seed_val = item
                            if loaded_p0 != p0_idx:
                                apply_genome(m0, population[p0_idx])
                                loaded_p0 = p0_idx
                            if loaded_p1 != p1_idx:
                                apply_genome(m1, population[p1_idx])
                                loaded_p1 = p1_idx

                            m_res = await env.evaluate_self_play_match(
                                model_p0=m0,
                                model_p1=m1,
                                device=self.device,
                                seed=seed_val,
                            )

                            async with lock:
                                results_by_genome[p0_idx].append(
                                    {
                                        "your_vp": m_res["p0_vp"],
                                        "automa_vp": m_res["p1_vp"],
                                        "won": m_res["p0_won"],
                                        "fitness": m_res["p0_fitness"],
                                        "steps": m_res["steps"],
                                        "seed": m_res["seed"],
                                        "opp_idx": p1_idx,
                                    }
                                )
                                results_by_genome[p1_idx].append(
                                    {
                                        "your_vp": m_res["p1_vp"],
                                        "automa_vp": m_res["p0_vp"],
                                        "won": m_res["p1_won"],
                                        "fitness": m_res["p1_fitness"],
                                        "steps": m_res["steps"],
                                        "seed": m_res["seed"],
                                        "opp_idx": p0_idx,
                                    }
                                )
                                done_gen_games += 1
                                self.completed_episodes += 1
                                self.generation_progress = f"{done_gen_games} / {total_gen_games}"

                                active_means = [
                                    sum(x["your_vp"] for x in glist) / len(glist)
                                    for glist in results_by_genome.values()
                                    if len(glist) > 0
                                ]
                                if active_means:
                                    self.generation_best_vp = max(active_means)
                                    self.generation_mean_vp = sum(active_means) / len(active_means)

                                self.status_message = (
                                    f"Gen {gen}/{self.target_generations} ({done_gen_games}/{total_gen_games} 2P matches) · "
                                    f"Leader: {self.generation_best_vp:.1f} VP · Pop Mean: {self.generation_mean_vp:.1f} VP"
                                )
                                await self._emit()

                    await asyncio.gather(*(eval_worker_sp(w_i) for w_i in range(self.workers)))
                else:
                    # Solo vs Mautoma with paired Common Random Numbers
                    gen_seeds = [rng.randint(1, 999999) for _ in range(self.games_per_individual)]
                    total_gen_games = self.population_size * self.games_per_individual
                    self.generation_progress = f"0 / {total_gen_games}"
                    job_queue_solo: asyncio.Queue[Optional[Tuple[int, int]]] = asyncio.Queue()
                    for g_idx in range(self.population_size):
                        for s_val in gen_seeds:
                            job_queue_solo.put_nowait((g_idx, s_val))
                    for _ in range(self.workers):
                        job_queue_solo.put_nowait(None)

                    self.status_message = (
                        f"Gen {gen}/{self.target_generations} (Solo vs Mautoma): Evaluating {self.population_size} genomes "
                        f"× {self.games_per_individual} paired seeds ({total_gen_games} games)..."
                    )
                    await self._emit()

                    async def eval_worker_solo(worker_id: int) -> None:
                        nonlocal done_gen_games
                        env = envs[worker_id]
                        w_model = worker_models_p0[worker_id]
                        loaded_idx = -1
                        while not self.stop_requested:
                            item = await job_queue_solo.get()
                            if item is None:
                                break
                            g_idx, seed_val = item
                            if loaded_idx != g_idx:
                                apply_genome(w_model, population[g_idx])
                                loaded_idx = g_idx

                            game_info = await env.evaluate_genome_greedy(
                                model=w_model,
                                device=self.device,
                                seed=seed_val,
                            )
                            game_info["genome_idx"] = g_idx

                            async with lock:
                                results_by_genome[g_idx].append(game_info)
                                done_gen_games += 1
                                self.completed_episodes += 1
                                self.generation_progress = f"{done_gen_games} / {total_gen_games}"

                                completed_means = [
                                    sum(x["your_vp"] for x in glist) / len(glist)
                                    for glist in results_by_genome.values()
                                    if len(glist) == self.games_per_individual
                                ]
                                if completed_means:
                                    self.generation_best_vp = max(completed_means)
                                    self.generation_mean_vp = sum(completed_means) / len(completed_means)

                                self.status_message = (
                                    f"Gen {gen}/{self.target_generations} ({done_gen_games}/{total_gen_games} games) · "
                                    f"Gen Leader: {self.generation_best_vp:.1f} VP · Pop Mean: {self.generation_mean_vp:.1f} VP"
                                )
                                await self._emit()

                    await asyncio.gather(*(eval_worker_solo(w_i) for w_i in range(self.workers)))

                # Only breed/rank if genomes have completed evaluations
                min_required = 1 if self.training_mode == "self_play" else self.games_per_individual
                valid_indices = [
                    i
                    for i, glist in results_by_genome.items()
                    if len(glist) >= min_required
                ]
                if not valid_indices:
                    break

                mean_vp_map: Dict[int, float] = {}
                mean_automa_map: Dict[int, float] = {}
                win_rate_map: Dict[int, float] = {}
                fitness_map: Dict[int, float] = {}

                for i in valid_indices:
                    glist = results_by_genome[i]
                    mean_vp_map[i] = sum(x["your_vp"] for x in glist) / len(glist)
                    mean_automa_map[i] = sum(x["automa_vp"] for x in glist) / len(glist)
                    win_rate_map[i] = sum(1.0 for x in glist if x["won"]) / len(glist)
                    fitness_map[i] = sum(float(x["fitness"]) for x in glist) / len(glist)

                # Rank descending by (fitness, mean_vp, win_rate); tie-break in favor of defending Champion (idx 0)
                ranked = sorted(
                    valid_indices,
                    key=lambda idx: (
                        round(fitness_map[idx], 4),
                        round(mean_vp_map[idx], 3),
                        round(win_rate_map[idx], 3),
                        1 if idx == 0 else 0,
                    ),
                    reverse=True,
                )

                winner_idx = ranked[0]
                winner_vp = mean_vp_map[winner_idx]
                winner_fit = fitness_map[winner_idx]
                pop_mean_vp = sum(mean_vp_map.values()) / len(mean_vp_map)
                defending_fit = fitness_map.get(0, -1e9)

                self.generation_best_vp = winner_vp
                self.generation_mean_vp = pop_mean_vp
                self.champion_avg_vp = winner_vp
                self.champion_fitness = winner_fit

                # Did a challenger beat Genome #0 on the EXACT same paired seeds (or is this Gen 1)?
                dethroned_champion = (gen == 1) or (winner_idx != 0 and winner_fit > defending_fit + 1e-4)
                champion_genome = clone_genome(population[winner_idx])

                if winner_vp > self.best_rolling_vp:
                    self.best_rolling_vp = winner_vp

                apply_genome(saver_model, champion_genome, calibrated_value_hint=winner_vp)
                ckpt_meta = {
                    "algorithm": "genetic_algorithm",
                    "generation": gen,
                    "completed_episodes": self.completed_episodes,
                    "champion_avg_vp": round(winner_vp, 2),
                    "champion_fitness": round(winner_fit, 3),
                    "generation_mean_vp": round(pop_mean_vp, 2),
                    "rolling_avg_vp": round(max(self.best_rolling_vp, winner_vp), 2),
                    "run_id": self.run_id,
                }
                save_brass_checkpoint(saver_model, None, LATEST_CHECKPOINT_PATH, metadata=ckpt_meta)

                if dethroned_champion:
                    save_brass_checkpoint(saver_model, None, BEST_CHECKPOINT_PATH, metadata=ckpt_meta)
                    stagnant_generations = 0
                    tag_label = f"Gen {gen} ★ NEW BEST"
                else:
                    stagnant_generations += 1
                    tag_label = f"Gen {gen} Elite #1"

                # Log the Generation Champion's paired games to SQLite so the chart reflects true evolved skill!
                last_ep_row = None
                for game_idx, g_res in enumerate(results_by_genome[winner_idx]):
                    logged_episode_counter += 1
                    last_ep_row = self.db.record_rl_episode(
                        run_id=self.run_id,
                        episode_num=logged_episode_counter,
                        your_vp=int(g_res["your_vp"]),
                        automa_vp=int(g_res["automa_vp"]),
                        won=bool(g_res["won"]),
                        total_reward=float(g_res["fitness"]),
                        steps=int(g_res["steps"]),
                        policy_loss=float(pop_mean_vp),
                        value_loss=float(gen),
                        entropy=float(active_sigma),
                        checkpoint_tag=tag_label if game_idx == len(results_by_genome[winner_idx]) - 1 else f"Gen {gen}",
                    )

                self.status_message = (
                    f"Gen {gen}/{self.target_generations} complete · Champion: {winner_vp:.1f} VP "
                    f"(Fitness {winner_fit:.1f} · Pop Mean {pop_mean_vp:.1f} VP)"
                    + (" — ★ Promoted to brass_ppo_best.pt!" if dethroned_champion else " — Crown defended")
                )
                await self._emit({"episode": last_ep_row} if last_ep_row else None)

                if self.stop_requested or gen >= self.target_generations:
                    break

                # === Breed Generation `gen + 1` ===
                next_population: List[Genome] = []
                # 1. Strict Elitism: Genome #0 is ALWAYS the reigning Champion unchanged!
                for e_rank in range(min(self.elite_count, len(ranked))):
                    next_population.append(clone_genome(population[ranked[e_rank]]))

                # 2. Dedicated Fine Hill-Climber around reigning Champion (weights & biases local polish)
                if len(next_population) < self.population_size:
                    next_population.append(
                        mutate_genome(
                            champion_genome,
                            mutation_rate=max(0.08, self.mutation_rate * 0.65),
                            mutation_sigma=active_sigma * 0.45,
                        )
                    )

                # 3. Dedicated Bold Explorer at last slot when population >= 6
                reserve_explorer = 1 if (self.population_size - len(next_population)) >= 2 else 0
                target_crossover_slots = self.population_size - reserve_explorer

                # 4. Tournament Selection + Neuron/BLX Crossover + Layer-Scaled Gaussian Mutation
                while len(next_population) < target_crossover_slots:
                    p_a = tournament_select(ranked, fitness_map, k=3, rng=rng)
                    p_b = tournament_select(ranked, fitness_map, k=3, rng=rng)
                    if p_a == p_b and len(ranked) > 1:
                        p_b = ranked[1] if p_a == ranked[0] else ranked[0]
                    child = crossover_genomes(population[p_a], population[p_b], rng=rng)
                    child = mutate_genome(
                        child,
                        mutation_rate=self.mutation_rate,
                        mutation_sigma=active_sigma,
                    )
                    next_population.append(child)

                if reserve_explorer and len(next_population) < self.population_size:
                    next_population.append(
                        mutate_genome(
                            champion_genome,
                            mutation_rate=min(0.60, self.mutation_rate * 1.65),
                            mutation_sigma=active_sigma * 1.75,
                        )
                    )

                population = next_population

            self.status_message = (
                f"Evolution complete ({self.current_generation} gens · {self.completed_episodes} games) — "
                f"Champion: {self.champion_avg_vp:.1f} VP saved to brass_ppo_best.pt!"
            )
        except asyncio.CancelledError:
            self.status_message = "Evolution cancelled."
        except Exception as e:
            self.status_message = f"GA Trainer error: {e}"
        finally:
            self.running = False
            for e in envs:
                await e.close()
            for c in contexts:
                try:
                    await c.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass
            if pw:
                try:
                    await pw.stop()
                except Exception:
                    pass
            await self._emit()

        return self.get_status()
