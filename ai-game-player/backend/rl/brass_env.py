import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from rl.brass_bridge import (
    BRASS_SOLO_URL,
    ensure_local_brass_server,
    enumerate_legal_actions,
    execute_macro_action,
    extract_brass_state,
    inject_fast_training_css,
    start_new_brass_game,
)
from rl.brass_features import (
    compute_board_summary,
    encode_action_list,
    encode_state,
)


def board_potential(state: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> float:
    """
    Potential function Phi(s) combining banked VP, live flipped tile VP,
    live connected link VP, and income track earnings power.
    Unflipped tiles contribute 0 until flipped!
    """
    s = summary or compute_board_summary(state)
    banked_vp = float(state.get("your_vp", 0))
    flipped_vp = float(s["flipped_tile_vp"])
    link_vp = float(s["connected_link_vp"])
    # Weight income higher in Canal Era (2 income phases ahead) than late Rail Era
    era_mult = 1.1 if state.get("era") == "canal" else 0.45
    income_term = era_mult * float(s["income_pounds"])
    return banked_vp + flipped_vp + link_vp + income_term


def compute_shaped_step_reward(
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    chosen_action: Dict[str, Any],
) -> float:
    """
    Compute dense Brass: Birmingham step reward r_t + terminal victory margin bonus.
    """
    prev_sum = compute_board_summary(prev_state)
    next_sum = compute_board_summary(next_state)

    phi_prev = board_potential(prev_state, prev_sum)
    # At gameOver, solo.kdc.sh already folds final Rail Era flipped_vp + link_vp into next_state['your_vp']!
    if next_state.get("game_over"):
        phi_next = float(next_state.get("your_vp", 0)) + 0.15 * float(next_sum["income_pounds"])
    else:
        phi_next = board_potential(next_state, next_sum)

    reward = (phi_next - phi_prev) / 15.0

    turn_advanced = (
        prev_state.get("turn") != next_state.get("turn")
        or prev_state.get("actions_left") != next_state.get("actions_left")
        or prev_state.get("era") != next_state.get("era")
        or bool(next_state.get("game_over"))
        or len(prev_state.get("hand") or []) != len(next_state.get("hand") or [])
        or prev_state.get("mat") != next_state.get("mat")
    )

    # Small bonus for multi-tile tech development (unlocking L2+ tiles) ONLY when state actually changed
    act_type = chosen_action.get("action_type")
    if turn_advanced and act_type == "develop":
        n_dev = len(chosen_action.get("industries") or [])
        reward += 0.04 * n_dev
    elif turn_advanced and act_type == "pass":
        # Strongly discourage wasting 1 of only 39 actions on Pass
        reward -= 0.35
    elif not turn_advanced:
        reward -= 0.15

    # Era-transition / game-end penalty for leaving unflipped tiles stranded or scoring 0 in an era
    era_changed = prev_state.get("era") != next_state.get("era")
    if era_changed or next_state.get("game_over"):
        stranded_vp = float(prev_sum["unflipped_tile_vp_at_risk"])
        reward -= min(0.8, stranded_vp / 30.0)
        if era_changed and prev_sum["flipped_tile_vp"] == 0 and prev_sum["connected_link_vp"] == 0:
            reward -= 1.5

    # Terminal VP margin, win bonus, and catastrophic 0-VP / low-VP penalty
    if next_state.get("game_over"):
        your_vp = float(next_state.get("your_vp", 0))
        automa_vp = float(next_state.get("automa_vp", 0))
        margin = (your_vp - automa_vp) / 25.0
        if your_vp > automa_vp:
            win_bonus = 2.0
        elif your_vp == automa_vp:
            win_bonus = 0.25
        else:
            win_bonus = -0.35

        # Heavy penalty for scoring 0 VP or failing to score meaningfully (< 40 VP)
        zero_vp_penalty = 0.0
        if your_vp <= 0.0:
            zero_vp_penalty = -6.0
        elif your_vp < 40.0:
            zero_vp_penalty = -3.0 * ((40.0 - your_vp) / 40.0)

        # Absolute VP milestone bonus to push past 120 -> 150+ VP
        milestone_bonus = max(0.0, (your_vp - 90.0) / 50.0)
        reward += margin + win_bonus + zero_vp_penalty + milestone_bonus

    return float(reward)


def compute_ga_game_fitness(
    your_vp: int,
    automa_vp: int,
    income_pounds: int = 0,
    unflipped_risk_vp: float = 0.0,
) -> float:
    """
    Fitness score for a completed 20-round Brass: Birmingham game in Genetic Algorithm training.
    Directly optimizes Victory Points, beating Mautoma, and avoiding 0-VP / unflipped collapses.
    """
    vp = float(your_vp)
    opp = float(automa_vp)
    margin = vp - opp

    if vp > opp:
        win_bonus = 28.0
    elif vp == opp and vp > 0:
        win_bonus = 8.0
    else:
        win_bonus = 0.0

    # Harsh evolutionary culling penalty for 0-VP or ultra-low-VP collapses
    collapse_penalty = 0.0
    if vp <= 0.0:
        collapse_penalty = -80.0
    elif vp < 45.0:
        collapse_penalty = -35.0 * ((45.0 - vp) / 45.0)

    # Small tie-breaker for healthy end-game income and flipping built tiles
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


class BrassPlaywrightEnv:
    """
    Single headless Chromium environment running Brass: Birmingham
    at macro-action granularity (~39 steps per full 20-round Brass match).
    """

    def __init__(self, browser_context, url: Optional[str] = None, difficulty: str = "easy"):
        self.context = browser_context
        self.url = ensure_local_brass_server() if (not url or url == BRASS_SOLO_URL) else url
        self.difficulty = difficulty
        self.page = None
        self.state: Dict[str, Any] = {}
        self.legal_actions: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.episode_reward: float = 0.0

    async def initialize(self) -> None:
        if self.page is None or self.page.is_closed():
            self.page = await self.context.new_page()
            await self.page.goto(self.url, wait_until="networkidle", timeout=30000)
            await inject_fast_training_css(self.page)

    async def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        await self.initialize()
        await start_new_brass_game(
            self.page,
            seed=seed if seed is not None else random.randint(1, 999999),
            difficulty=self.difficulty,
        )
        await inject_fast_training_css(self.page)
        self.step_count = 0
        self.episode_reward = 0.0
        self.state = await extract_brass_state(self.page, reset_open_flow=True)
        self.legal_actions = await enumerate_legal_actions(self.page, self.state)
        s_vec = encode_state(self.state)
        a_vecs = encode_action_list(self.state, self.legal_actions)
        return s_vec, a_vecs, self.legal_actions

    async def _run_guaranteed_fallback(self) -> Dict[str, Any]:
        safe_loan = next((a for a in self.legal_actions if a.get("action_type") == "loan"), None)
        if safe_loan is not None:
            res = await execute_macro_action(self.page, safe_loan, visual_delay_ms=0)
            if res.get("ok"):
                return safe_loan
        hand_len = max(1, len(self.state.get("hand") or []))
        for slot in range(hand_len):
            pass_act = {"action_type": "pass", "card_slot": slot, "description": f"Fallback Pass slot #{slot + 1}"}
            res = await execute_macro_action(self.page, pass_act, visual_delay_ms=0)
            if res.get("ok"):
                return pass_act
        return {"action_type": "pass", "card_slot": 0}

    async def step(
        self, action_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        if not self.legal_actions:
            s_vec = encode_state(self.state)
            return s_vec, np.zeros((0, 48), dtype=np.float32), 0.0, True, {
                "your_vp": int(self.state.get("your_vp", 0)),
                "automa_vp": int(self.state.get("automa_vp", 0)),
                "won": int(self.state.get("your_vp", 0)) > int(self.state.get("automa_vp", 0)),
                "steps": self.step_count,
                "total_reward": self.episode_reward,
            }

        idx = max(0, min(len(self.legal_actions) - 1, int(action_idx)))
        chosen_action = self.legal_actions[idx]
        prev_state = self.state

        exec_res = await execute_macro_action(self.page, chosen_action, visual_delay_ms=0)
        if not exec_res.get("ok"):
            chosen_action = await self._run_guaranteed_fallback()

        next_state = await extract_brass_state(self.page, reset_open_flow=True)
        if not _did_turn_advance(prev_state, next_state) and not next_state.get("game_over"):
            chosen_action = await self._run_guaranteed_fallback()
            next_state = await extract_brass_state(self.page, reset_open_flow=True)

        self.step_count += 1
        reward = compute_shaped_step_reward(prev_state, next_state, chosen_action)
        self.episode_reward += reward
        self.state = next_state

        done = bool(next_state.get("game_over")) or self.step_count >= 65
        if done:
            self.legal_actions = []
            a_vecs = np.zeros((0, 48), dtype=np.float32)
        else:
            self.legal_actions = await enumerate_legal_actions(self.page, self.state)
            if not self.legal_actions:
                done = True
                a_vecs = np.zeros((0, 48), dtype=np.float32)
            else:
                a_vecs = encode_action_list(self.state, self.legal_actions)

        end_summary = compute_board_summary(self.state) if done else {}
        your_vp = int(self.state.get("your_vp", 0))
        automa_vp = int(self.state.get("automa_vp", 0))
        fitness = (
            compute_ga_game_fitness(
                your_vp=your_vp,
                automa_vp=automa_vp,
                income_pounds=int(end_summary.get("income_pounds", 0)),
                unflipped_risk_vp=float(end_summary.get("unflipped_tile_vp_at_risk", 0.0)),
            )
            if done
            else 0.0
        )

        s_vec = encode_state(self.state)
        info = {
            "your_vp": your_vp,
            "automa_vp": automa_vp,
            "won": your_vp > automa_vp,
            "steps": self.step_count,
            "total_reward": round(self.episode_reward, 4),
            "fitness": round(fitness, 3),
            "action_desc": chosen_action.get("description", ""),
        }
        return s_vec, a_vecs, reward, done, info

    async def evaluate_genome_greedy(self, model, device, seed: int) -> Dict[str, Any]:
        """
        Play one full deterministic 20-round game on `seed` using greedy argmax(logits).
        Used by Genetic Algorithm workers for zero-variance paired-seed evaluation.
        """
        import torch

        s_vec, a_vecs, _ = await self.reset(seed=seed)
        done = False
        info: Dict[str, Any] = {
            "your_vp": 0,
            "automa_vp": 0,
            "won": False,
            "steps": 0,
            "total_reward": 0.0,
            "fitness": -80.0,
            "seed": seed,
        }
        while not done and a_vecs.shape[0] > 0:
            with torch.inference_mode():
                s_t = torch.from_numpy(s_vec).to(device)
                a_t = torch.from_numpy(a_vecs).to(device)
                logits, _ = model.forward_single(s_t, a_t)
                act_idx = int(torch.argmax(logits).item())
            s_vec, a_vecs, _, done, info = await self.step(act_idx)

        info["seed"] = int(seed)
        return info

    async def evaluate_self_play_match(
        self,
        model_p0,
        model_p1,
        device,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Play one full official 2-player Brass: Birmingham match (39 turns per player = 78 turns)
        between `model_p0` (Seat 0) and `model_p1` (Seat 1) using deterministic argmax(logits).
        """
        import torch

        await self.initialize()
        await start_new_brass_game(
            self.page,
            seed=seed,
            mode="2p",
            player_names=["Genome P1", "Genome P2"],
        )
        await inject_fast_training_css(self.page)

        self.step_count = 0
        self.episode_reward = 0.0
        self.state = await extract_brass_state(self.page, reset_open_flow=True)

        while not bool(self.state.get("game_over")) and self.step_count < 95:
            active_seat = int(self.state.get("active_player", 0))
            active_model = model_p0 if active_seat == 0 else model_p1

            legal = await enumerate_legal_actions(self.page, self.state)
            if not legal:
                break
            self.legal_actions = legal

            s_vec = encode_state(self.state)
            a_vecs = encode_action_list(self.state, legal)
            with torch.inference_mode():
                s_t = torch.from_numpy(s_vec).to(device)
                a_t = torch.from_numpy(a_vecs).to(device)
                logits, _ = active_model.forward_single(s_t, a_t)
                act_idx = int(torch.argmax(logits).item())

            chosen_action = legal[max(0, min(len(legal) - 1, act_idx))]
            prev_state = self.state

            exec_res = await execute_macro_action(self.page, chosen_action, visual_delay_ms=0)
            if not exec_res.get("ok"):
                await self._run_guaranteed_fallback()

            next_state = await extract_brass_state(self.page, reset_open_flow=True)
            if not _did_turn_advance(prev_state, next_state) and not next_state.get("game_over"):
                await self._run_guaranteed_fallback()
                next_state = await extract_brass_state(self.page, reset_open_flow=True)

            self.state = next_state
            self.step_count += 1

        # Extract final scores & board summaries from both seats' perspectives
        p0_state = await extract_brass_state(self.page, reset_open_flow=False, perspective_player=0)
        p1_state = await extract_brass_state(self.page, reset_open_flow=False, perspective_player=1)

        p0_vp = int(p0_state.get("your_vp", 0))
        p1_vp = int(p1_state.get("your_vp", 0))
        p0_sum = compute_board_summary(p0_state)
        p1_sum = compute_board_summary(p1_state)

        p0_fit = compute_ga_game_fitness(
            your_vp=p0_vp,
            automa_vp=p1_vp,
            income_pounds=int(p0_sum.get("income_pounds", 0)),
            unflipped_risk_vp=float(p0_sum.get("unflipped_tile_vp_at_risk", 0.0)),
        )
        p1_fit = compute_ga_game_fitness(
            your_vp=p1_vp,
            automa_vp=p0_vp,
            income_pounds=int(p1_sum.get("income_pounds", 0)),
            unflipped_risk_vp=float(p1_sum.get("unflipped_tile_vp_at_risk", 0.0)),
        )

        return {
            "seed": int(seed),
            "p0_vp": p0_vp,
            "p1_vp": p1_vp,
            "p0_fitness": round(p0_fit, 3),
            "p1_fitness": round(p1_fit, 3),
            "p0_won": p0_vp > p1_vp,
            "p1_won": p1_vp > p0_vp,
            "steps": self.step_count,
        }

    async def close(self) -> None:
        if self.page and not self.page.is_closed():
            try:
                await self.page.close()
            except Exception:
                pass
