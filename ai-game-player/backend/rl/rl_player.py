from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rl.brass_bridge import enumerate_legal_actions, extract_brass_state
from rl.brass_features import compute_board_summary, encode_action_list, encode_state
from rl.model import (
    ACTION_DIM,
    BEST_CHECKPOINT_PATH,
    LATEST_CHECKPOINT_PATH,
    STATE_DIM,
    BrassActorCritic,
    get_rl_device,
    load_brass_checkpoint,
)


class BrassRLAgentPlayer:
    """
    Inference wrapper for playing Brass: Birmingham visually in the Live Browser Arena
    (provider='brass_rl', $0 API cost) AND for providing top-3 RL Advisor move rankings
    to cloud LLM sessions (Gemini/OpenAI/Claude).
    """

    def __init__(self, checkpoint_preference: str = "brass-ppo-best"):
        self.preference = checkpoint_preference or "brass-ppo-best"
        self.device = get_rl_device()
        self.model: Optional[BrassActorCritic] = None
        self.checkpoint_meta: Dict[str, Any] = {}
        self.checkpoint_source: str = "untrained-init"
        self._loaded_mtime: float = 0.0
        self.reload_if_updated()

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        if "latest" in self.preference.lower():
            if LATEST_CHECKPOINT_PATH.exists():
                return LATEST_CHECKPOINT_PATH
            if BEST_CHECKPOINT_PATH.exists():
                return BEST_CHECKPOINT_PATH
        else:
            if BEST_CHECKPOINT_PATH.exists():
                return BEST_CHECKPOINT_PATH
            if LATEST_CHECKPOINT_PATH.exists():
                return LATEST_CHECKPOINT_PATH
        return None

    def reload_if_updated(self) -> None:
        ckpt_path = self._resolve_checkpoint_path()
        if ckpt_path is not None and ckpt_path.exists():
            mtime = ckpt_path.stat().st_mtime
            if self.model is None or mtime > self._loaded_mtime:
                self.model, self.checkpoint_meta = load_brass_checkpoint(ckpt_path, device=self.device)
                self.model.eval()
                self._loaded_mtime = mtime
                self.checkpoint_source = ckpt_path.name
        elif self.model is None:
            self.model = BrassActorCritic(STATE_DIM, ACTION_DIM).to(self.device)
            self.model.eval()
            self.checkpoint_source = "untrained-init"

    def has_trained_checkpoint(self) -> bool:
        return BEST_CHECKPOINT_PATH.exists() or LATEST_CHECKPOINT_PATH.exists()

    async def evaluate_board(
        self, page, greedy: bool = True, temperature: float = 0.25
    ) -> Dict[str, Any]:
        """
        Inspect live Brass page, score all legal macro-actions with BrassActorCritic,
        and return ranked moves + V(s_t) state value.
        """
        self.reload_if_updated()
        state = await extract_brass_state(page, reset_open_flow=True)

        if state.get("in_setup_modal"):
            return {
                "mode": "setup_modal",
                "state": state,
                "value": 0.0,
                "chosen_action": None,
                "ranked_actions": [],
            }

        if state.get("game_over"):
            return {
                "mode": "game_over",
                "state": state,
                "value": float(state.get("your_vp", 0) - state.get("automa_vp", 0)) / 25.0,
                "chosen_action": None,
                "ranked_actions": [],
            }

        legal = await enumerate_legal_actions(page, state)
        if not legal:
            return {
                "mode": "no_moves",
                "state": state,
                "value": 0.0,
                "chosen_action": None,
                "ranked_actions": [],
            }

        s_vec = encode_state(state)
        a_vecs = encode_action_list(state, legal)

        with torch.no_grad():
            assert self.model is not None
            self.model.eval()
            s_t = torch.from_numpy(s_vec).to(self.device)
            a_t = torch.from_numpy(a_vecs).to(self.device)
            logits, val = self.model.forward_single(s_t, a_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            logits_np = logits.cpu().numpy()
            value_est = float(val.item())

        order = np.argsort(-logits_np)
        if greedy or temperature <= 0.05:
            chosen_idx = int(order[0])
        else:
            scaled = logits_np / max(0.05, float(temperature))
            scaled -= scaled.max()
            p_temp = np.exp(scaled)
            p_temp /= p_temp.sum()
            chosen_idx = int(np.random.choice(len(legal), p=p_temp))

        ranked = []
        for rank_i, idx in enumerate(order[:5]):
            ranked.append(
                {
                    "rank": rank_i + 1,
                    "index": int(idx),
                    "prob_pct": round(float(probs[idx]) * 100.0, 1),
                    "logit": round(float(logits_np[idx]), 3),
                    "action": legal[int(idx)],
                }
            )

        return {
            "mode": "play",
            "state": state,
            "summary": compute_board_summary(state),
            "value": round(value_est, 3),
            "chosen_action": legal[chosen_idx],
            "chosen_prob_pct": round(float(probs[chosen_idx]) * 100.0, 1),
            "total_legal_actions": len(legal),
            "ranked_actions": ranked,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_meta": self.checkpoint_meta,
        }

    async def get_advisor_hint(self, page) -> Optional[str]:
        """
        Generate a non-intrusive Top-3 Neural Move Recommendation block for cloud LLMs
        when playing on solo.kdc.sh/brass-birmingham with a trained checkpoint.
        """
        if not self.has_trained_checkpoint():
            return None
        try:
            # Only advise when on root action selection (don't cancel an active LLM mid-turn flow!)
            in_mid_flow = await page.evaluate(
                "() => !!document.querySelector('[data-testid=\"flow-cancel\"]')"
            )
            if in_mid_flow:
                return None
            eval_res = await self.evaluate_board(page, greedy=True)
            if eval_res.get("mode") != "play" or not eval_res.get("ranked_actions"):
                return None
            top_lines = [
                f"  {r['rank']}. [{r['prob_pct']:.1f}%] {r['action']['description']}"
                for r in eval_res["ranked_actions"][:3]
            ]
            avg_vp = self.checkpoint_meta.get("rolling_avg_vp", "N/A")
            return (
                f"BRASS RL NEURAL CRITIC ADVISOR ({self.checkpoint_source} · avg {avg_vp} VP · "
                f"State Value V(s)={eval_res['value']:+.2f} across {eval_res['total_legal_actions']} legal macro-moves):\n"
                + "\n".join(top_lines)
            )
        except Exception:
            return None
