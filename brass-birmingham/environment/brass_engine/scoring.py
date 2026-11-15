"""Canonical trajectory signature matching, SHA-256 chain integrity, & exploit-resistant [0.0, 1.0] Harbor reward scoring for Brass: Birmingham."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_EVAL_SEED = 202611
DEFAULT_EVAL_DIFFICULTY = "easy"
EXPECTED_TOTAL_ACTIONS = 39


def income_space_to_pounds_track(space: int) -> int:
    """Official Brass: Birmingham 0..99 income track mapping -> pounds collected per round."""
    s = max(0, min(99, int(space)))
    if s <= 10:
        return s - 10  # spaces 0..9 yield -10..-1 £/round; space 10 yields £0
    if s <= 30:
        return (s - 9) // 2  # +1..+10
    if s <= 60:
        return 10 + (s - 28) // 3  # +11..+20
    return 20 + (s - 57) // 4  # +21..+30


def action_signature(action: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Canonical, deterministic tuple signature uniquely identifying a legal Brass macro-action.
    Used to record moves in `/app/trajectory.jsonl` and replay them inside the clean-room verifier.
    """
    t = str(action.get("action_type", "pass"))
    if t == "build":
        return (
            "build",
            int(action.get("card_slot", 0)),
            str(action.get("city", "")),
            str(action.get("industry", "")),
            int(action.get("level", 1)),
            int(action.get("btn_index", 0)),
            str(action.get("option_text", "")),
        )
    if t == "network":
        lids = action.get("link_ids") or ([str(action.get("link_id", ""))] if action.get("link_id") else [])
        return (
            "network",
            int(action.get("card_slot", 0)),
            tuple(str(x) for x in lids),
            int(action.get("btn_index", 0)),
            str(action.get("option_text", "")),
        )
    if t == "sell":
        return (
            "sell",
            int(action.get("card_slot", 0)),
            tuple(int(i) for i in (action.get("sell_indices") or [])),
        )
    if t == "develop":
        return (
            "develop",
            int(action.get("card_slot", 0)),
            tuple(str(ind) for ind in (action.get("industries") or [])),
        )
    if t == "scout":
        return (
            "scout",
            tuple(int(s) for s in (action.get("discard_slots") or [])),
        )
    # loan or pass
    return (t, int(action.get("card_slot", 0)))


def compute_step_chain_hash(prev_hash: str, step_index: int, sig_json_list: List[Any], seed: int) -> str:
    payload = json.dumps([prev_hash, int(step_index), sig_json_list, int(seed)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_action_record(
    step_index: int,
    action: Dict[str, Any],
    description: str,
    prev_hash: str = "GENESIS",
    seed: int = DEFAULT_EVAL_SEED,
) -> Dict[str, Any]:
    """Compact, hash-chained JSON record written per step to `/app/trajectory.jsonl`."""
    sig = action_signature(action)
    sig_list = [list(x) if isinstance(x, tuple) else x for x in sig]
    chain_hash = compute_step_chain_hash(prev_hash, step_index, sig_list, seed)
    return {
        "step": int(step_index),
        "action_type": str(action.get("action_type", "pass")),
        "signature": sig_list,
        "chain_hash": chain_hash,
        "description": description,
    }


def parse_record_signature(record: Dict[str, Any]) -> Tuple[Any, ...]:
    raw_sig = record.get("signature") or []
    return tuple(tuple(x) if isinstance(x, list) else x for x in raw_sig)


def compute_brass_harbor_reward(
    state: Dict[str, Any],
    valid_steps: int,
    expected_steps: int = EXPECTED_TOTAL_ACTIONS,
    records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute exploit-resistant [0.0, 1.0] Harbor evaluation score from authentic engine state:

    Anti-Reward-Hacking Guarantees:
      - Do-Nothing (`valid_steps == 0`) or Zero-VP / Pass-Spam (`your_vp <= 0`) -> `0.000000`
      - Stop-Early (`not game_over`):
          * Deducts unpaid negative-income debt liability across all remaining rounds
            (eliminates the "take 3 Canal loans, score L1 tiles at Turn 19, and quit" loophole!).
          * Caps incomplete-episode credit strictly at `<= 0.08` so stopping early NEVER competes
            with completing the full 20-round Canal + Rail game.
      - Completed Match (`game_over == True` and `your_vp > 0`):
          1. Engaged Completion (`0.00 - 0.10`): `0.10 * min(1.0, your_vp / 35.0)`
          2. Empire Victory Points (`0.00 - 0.55`): `0.55 * min(1.0, your_vp / 150.0)`
          3. Competitive Margin vs. Mautoma (`0.00 - 0.35`):
             - Win (`margin > 0`):  `0.22 + 0.13 * min(1.0, margin / 30.0)`
             - Loss/Tie (`<= 0`):   `0.20 * max(0.0, 1.0 - abs(margin) / 100.0) * min(1.0, your_vp / 50.0)`
    """
    game_over = bool(state.get("game_over", False))
    your_vp = int(state.get("your_vp", 0))
    automa_vp = int(state.get("automa_vp", 0))
    margin = your_vp - automa_vp
    won = bool(game_over and margin > 0 and your_vp > 0)
    era = str(state.get("era", "canal"))
    turn = int(state.get("turn", 1))
    income_space = int(state.get("income_space", 10))
    income_pounds = income_space_to_pounds_track(income_space)

    pass_count = 0
    if records:
        pass_count = sum(
            1
            for r in records[:valid_steps]
            if str(r.get("action_type", "")) == "pass"
            or (isinstance(r.get("signature"), list) and r["signature"] and r["signature"][0] == "pass")
        )

    if valid_steps <= 0 or your_vp <= 0:
        return {
            "reward": 0.0,
            "valid_steps": max(0, int(valid_steps)),
            "pass_count": pass_count,
            "game_over": game_over,
            "era": era,
            "turn": turn,
            "your_vp": your_vp,
            "automa_vp": automa_vp,
            "vp_margin": margin,
            "won": False,
            "money": int(state.get("money", 0)),
            "income_space": income_space,
            "income_pounds": income_pounds,
            "completion_score": 0.0,
            "vp_score": 0.0,
            "competitive_score": 0.0,
        }

    if not game_over:
        # Stop-Early / Incomplete Episode: deduct remaining negative-income liability & strictly cap <= 0.08
        rem_rounds = max(0, 10 - turn + 1) + (10 if era == "canal" else 0)
        debt_liability = rem_rounds * abs(min(0, income_pounds))
        effective_vp = max(0.0, float(your_vp) - float(debt_liability))
        progress = min(1.0, float(valid_steps) / float(max(1, expected_steps)))

        completion_score = 0.03 * progress * min(1.0, effective_vp / 35.0)
        vp_score = 0.05 * progress * min(1.0, effective_vp / 140.0)
        competitive_score = 0.0
    else:
        # Full 20-Round Completed Match
        engagement_gate = min(1.0, float(your_vp) / 35.0)
        completion_score = 0.10 * engagement_gate
        vp_score = 0.55 * min(1.0, float(your_vp) / 150.0)

        if margin > 0:
            competitive_score = 0.22 + 0.13 * min(1.0, float(margin) / 30.0)
        else:
            vp_floor_gate = min(1.0, float(your_vp) / 50.0)
            competitive_score = 0.20 * max(0.0, 1.0 - float(abs(margin)) / 100.0) * vp_floor_gate

        # Discourage degenerate excessive passing (>15 passes out of 39 actions)
        if pass_count > 15:
            pass_penalty_mult = max(0.25, 1.0 - 0.04 * float(pass_count - 15))
            completion_score *= pass_penalty_mult
            vp_score *= pass_penalty_mult
            competitive_score *= pass_penalty_mult

    total_reward = max(0.0, min(1.0, completion_score + vp_score + competitive_score))
    return {
        "reward": round(total_reward, 6),
        "valid_steps": int(valid_steps),
        "pass_count": pass_count,
        "game_over": game_over,
        "era": era,
        "turn": turn,
        "your_vp": your_vp,
        "automa_vp": automa_vp,
        "vp_margin": margin,
        "won": won,
        "money": int(state.get("money", 0)),
        "income_space": income_space,
        "income_pounds": income_pounds,
        "completion_score": round(completion_score, 6),
        "vp_score": round(vp_score, 6),
        "competitive_score": round(competitive_score, 6),
    }
