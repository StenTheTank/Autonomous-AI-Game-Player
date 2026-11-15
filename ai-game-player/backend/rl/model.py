import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "brass_ppo_best.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "brass_ppo_latest.pt"

STATE_DIM = 288
ACTION_DIM = 48
HIDDEN_DIM = 256


def get_rl_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BrassActorCritic(nn.Module):
    """
    Candidate-Action Actor-Critic for Brass: Birmingham.
    Scores variable-count legal macro-actions (s_t, {a_1 .. a_K}) -> logits (K,)
    and predicts scalar state value V(s_t).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.state_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_net = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Combines [h_s, h_a, h_s * h_a] -> scalar logit per legal candidate action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # Differentiable domain-feature skip connection (warm-started for sensible Ep-1 play, refined by PPO)
        self.action_prior = nn.Linear(action_dim, 1, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small initial gain on neural residual head so PPO shapes residuals smoothly around action_prior
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.08)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

        with torch.no_grad():
            w = torch.zeros(self.action_dim)
            w[0] = 0.45   # build base
            w[1] = 0.25   # network base
            w[2] = 2.10   # sell base (always prioritize flipping built consumers!)
            w[3] = 0.10   # develop base
            w[4] = -0.35  # loan base
            w[5] = -0.55  # scout base
            w[6] = -2.50  # pass base (avoid passing unless forced)
            w[12] = 0.55  # brewery build bonus (beer unlocks consumer sales)
            w[16] = -0.60 # penalize discarding wildcards / high-value cards
            w[18] = 1.10  # build tile VP
            w[19] = 0.85  # build income spaces
            w[24] = 0.70  # build connected to matching merchant
            w[25] = 0.80  # build with available beer / no beer needed
            w[26] = -2.20 # avoid late-era unflippable traps!
            w[27] = 1.35  # immediate coal/iron market sale flip/cash!
            w[28] = 0.90  # build in city already touched by player links
            w[29] = 1.80  # network immediate connected link VP!
            w[30] = 0.65  # network connects merchant port
            w[31] = 0.35  # network expands to new city
            w[32] = 0.90  # multi-building sell bundle bonus
            w[33] = 1.60  # sell total VP
            w[34] = 0.80  # sell total income gain
            w[36] = 0.55  # develop 2 tiles over 1 tile
            w[37] = 0.45  # develop canal-only tile
            w[38] = 0.40  # cheap iron develop
            w[39] = 1.65  # timely safe loan when cash <= £7 and income >= -4
            w[40] = -2.40 # block bankruptcy / Round-10 loans!
            w[41] = 0.75  # scout when 0 beer on board
            self.action_prior.weight.copy_(w.unsqueeze(0))

    def encode_state_batch(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.state_net(state_tensor)

    def forward_single(
        self, state_vec: torch.Tensor, action_vecs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a single game state and its K legal candidate actions.
        state_vec: (state_dim,) or (1, state_dim)
        action_vecs: (K, action_dim)
        Returns:
            logits: (K,)
            value: scalar tensor ()
        """
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        h_s = self.state_net(state_vec)  # (1, H)
        value = self.value_head(h_s).squeeze(-1).squeeze(0)  # ()

        k = action_vecs.shape[0]
        if k == 0:
            return torch.zeros(0, device=state_vec.device), value

        h_a = self.action_net(action_vecs)  # (K, H)
        h_s_exp = h_s.expand(k, -1)  # (K, H)
        joint = torch.cat([h_s_exp, h_a, h_s_exp * h_a], dim=-1)  # (K, 3H)
        prior_logits = self.action_prior(action_vecs).squeeze(-1)  # (K,)
        logits = self.policy_head(joint).squeeze(-1) + prior_logits  # (K,)
        return logits, value

    def forward_padded_batch(
        self,
        states: torch.Tensor,
        padded_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward pass over B rollout transitions for fast PPO minibatch updates.
        states: (B, state_dim)
        padded_actions: (B, max_K, action_dim)
        action_mask: (B, max_K) boolean tensor (True = valid legal candidate)
        Returns:
            masked_logits: (B, max_K) with -1e9 on padded slots
            values: (B,)
        """
        bsz, max_k, _ = padded_actions.shape
        h_s = self.state_net(states)  # (B, H)
        values = self.value_head(h_s).squeeze(-1)  # (B,)

        flat_actions = padded_actions.reshape(bsz * max_k, self.action_dim)
        h_a = self.action_net(flat_actions).reshape(bsz, max_k, self.hidden_dim)  # (B, max_K, H)
        h_s_exp = h_s.unsqueeze(1).expand(bsz, max_k, self.hidden_dim)  # (B, max_K, H)

        joint = torch.cat([h_s_exp, h_a, h_s_exp * h_a], dim=-1)  # (B, max_K, 3H)
        prior_logits = self.action_prior(flat_actions).reshape(bsz, max_k)  # (B, max_K)
        logits = self.policy_head(joint).squeeze(-1) + prior_logits  # (B, max_K)
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        return masked_logits, values


def save_brass_checkpoint(
    model: BrassActorCritic,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(path.parent, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden_dim": model.hidden_dim,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp_path = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_brass_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BrassActorCritic, Dict[str, Any]]:
    dev = device or get_rl_device()
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model = BrassActorCritic(
        state_dim=int(ckpt.get("state_dim", STATE_DIM)),
        action_dim=int(ckpt.get("action_dim", ACTION_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", HIDDEN_DIM)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    return model, dict(ckpt.get("metadata") or {})
