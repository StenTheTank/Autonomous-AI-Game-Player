"""Rich terminal HUD, tactical legal-move annotations, and rules reference for Brass CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from brass_engine.brass_bridge import describe_macro_action
from brass_engine.brass_features import (
    TILE_STATS,
    coal_market_price,
    compute_board_summary,
    encode_action,
    get_connected_component_and_network,
    iron_market_price,
)
from brass_engine.scoring import compute_brass_harbor_reward

RULES_REFERENCE = """================================================================================
BRASS: BIRMINGHAM (SOLO VS. MAUTOMA) — TACTICAL & SCORING GUIDE
================================================================================
OVERVIEW
- Match length: 20 rounds total = Canal Era (Rounds 1-10) + Rail Era (Rounds 1-10).
- Actions: Round 1 of Canal Era gives 1 action; all other rounds give 2 actions (~39 total).
- Turn Order & Tie-Breakers: You and the Mautoma AI play each round; turn order next round
  goes to whoever spent LESS money this round.
- End-of-Era Scoring (happens twice: end of Canal Round 10 AND end of Rail Round 10):
  1. LINK VP: Each of your placed Canal/Rail links scores the sum of link-dots (•) on
     ALL FLIPPED tiles + Merchant ports adjacent to both ends of that link.
  2. BUILDING VP: Every FLIPPED industry tile you own scores its printed VP.
     * Unflipped tiles score 0 VP!
     * Canal Transition: After Canal Era scoring, ALL Canal links AND all Level-1 (L1)
       industry tiles on the board are removed! Level 2+ (L2+) tiles survive and SCORE TWICE
       (once in Canal Era and again in Rail Era!).

CORE MECHANICS & CONSTRAINTS
1. BUILD: Place an industry tile in a city.
   - Location card builds ANY industry in that city (even outside your network).
   - Industry card builds that industry ONLY in a city already in your personal network
     (a city containing one of your tiles or adjacent to one of your links).
   - Coal consumption: Needs a connected route of links to an unflipped Coal Mine (free)
     OR to a Merchant port (buys from Coal Market; £1-8/cube).
   - Iron consumption: Bought from ANY unflipped Iron Works anywhere on the board (free,
     no link route needed!) OR from the Iron Market (£1-6/cube).
   - Coal Mines & Iron Works auto-sell cubes into empty slots on their market immediately
     upon building! If all cubes on the tile sell out immediately, the tile FLIPS right away
     (scoring VP & boosting your £ income immediately!).

2. SELL (Cotton, Manufactured Goods, Pottery):
   - Requires: (a) your unflipped tile is connected via links to a Merchant port showing
     that industry icon, AND (b) sufficient Beer (from ANY of your unflipped Breweries
     anywhere on the board, OR an opponent unflipped Brewery connected to the tile, OR
     the Merchant port's own bonus beer barrel!).
   - Flipping a tile immediately advances your Income track and unlocks its VP + link dots!
   - Multiple buildings can be sold in a SINGLE Sell action!

3. NETWORK:
   - Canal Era: £3 per canal link (max 1 per action).
   - Rail Era:
     * Single Rail: £5 + 1 Coal for 1 rail link.
     * Double Rail: £15 + 2 Coal (1 per link, resolved sequentially so link 2 can extend link 1!) + 1 Beer (own brewery anywhere or connected opponent brewery; merchant beer not allowed) for 2 rail links in 1 action!

4. DEVELOP:
   - Discard 1 card + pay 1-2 Iron to remove 1 or 2 lowest-level tiles from your Player Mat
     permanently (either 2 distinct tracks OR the same industry track twice `×2`!), unlocking L2+ tiles (which survive Canal -> Rail transition and score twice!).
   - NOTE: Lightbulb Pottery tiles (L1 & L3) CANNOT be developed away — build & sell L1 Pottery!

5. LOAN:
   - Discard 1 card -> gain +£30 cash immediately, move BACK 3 income tiers (up to -£6/round).
   - Taking Loans in early/mid Era when cash <= £6 fuels expansion; avoid Loans in Round 9-10
     or when income is already deeply negative (bankruptcy shortfall costs -1 VP per £1 missing!).

6. SCOUT:
   - Discard 3 cards (requires no Wild cards already in hand) -> gain 1 Wild Location + 1 Wild Industry.

HARBOR REWARD RUBRIC (/logs/verifier/reward.txt in [0.0, 1.0])
- Zero-VP / Pass-Spam (`your_vp <= 0`): 0.000000
- Incomplete / Stop-Early (`game_over == false`): strictly capped <= 0.08 (after deducting
  unpaid negative-income debt across remaining rounds). Always finish all 20 rounds!
- Full 20-Round Match (`game_over == true` and `your_vp > 0`):
    1. Engaged Completion (0.00 - 0.10): 0.10 * min(1.0, your_vp / 35.0)
    2. Empire VP          (0.00 - 0.55): 0.55 * min(1.0, your_vp / 150.0)
    3. Vs Mautoma         (0.00 - 0.35):
         * Win (your_vp > automa_vp):  0.22 + 0.13 * min(1.0, (your_vp - automa_vp) / 30.0)
         * Loss/Tie:                   0.20 * max(0.0, 1.0 - (automa_vp - your_vp) / 100.0) * min(1.0, your_vp / 50.0)
================================================================================
"""


def annotate_action_hint(action: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Compute concise, truthful tactical telemetry from the exact feature vector for CLI display."""
    feat = encode_action(state, action)
    t = action.get("action_type", "pass")
    notes: List[str] = []

    if t == "build":
        ind = str(action.get("industry", ""))
        lvl = int(action.get("level", 1))
        vp, inc_sp, dots, beer_need, _, _, _ = TILE_STATS.get((ind, lvl), (3, 2, 1, 0, 0, True, False))
        notes.append(f"tile={vp}VP,+{inc_sp}inc,{dots}•")
        if lvl >= 2 and state.get("era") == "canal":
            notes.append("survives->Rail(scores 2x)")
        if feat[27] >= 0.99:
            notes.append("AUTO-FLIPS via market!")
        elif feat[27] > 0.1:
            notes.append("sells cubes to market")
        if beer_need > 0:
            if feat[24] > 0.5 and feat[25] > 0.5:
                notes.append("merchant+beer ready to Sell!")
            elif feat[24] < 0.5:
                notes.append("not yet linked to matching merchant")
            elif feat[25] < 0.5:
                notes.append(f"needs {beer_need} beer")
        if feat[26] > 0.5:
            notes.append("WARN: late-era unflipped risk")
    elif t == "network":
        imm_vp = int(round(float(feat[29]) * 10.0))
        lids = action.get("link_ids") or ([action.get("link_id")] if action.get("link_id") else [])
        if len(lids) == 2:
            notes.append("2x RAIL")
        notes.append(f"currently +{imm_vp} link VP")
        if feat[30] > 0.5:
            notes.append("connects merchant port")
        if feat[31] > 0.5:
            notes.append("expands network")
    elif t == "sell":
        sales = action.get("sales") or []
        tot_vp = sum(TILE_STATS.get((s.get("industry", ""), int(s.get("level", 1))), (3, 2, 1, 0, 0, True, False))[0] for s in sales)
        tot_inc = sum(TILE_STATS.get((s.get("industry", ""), int(s.get("level", 1))), (3, 2, 1, 0, 0, True, False))[1] for s in sales)
        notes.append(f"FLIPS -> +{tot_vp} VP, +{tot_inc} income spaces!")
    elif t == "develop":
        inds = action.get("industries") or []
        notes.append(f"unlocks higher tech ({len(inds)} tile{'s' if len(inds) != 1 else ''})")
    elif t == "loan":
        if feat[40] > 0.5:
            notes.append("WARN: late/low-income loan risk")
        elif feat[39] > 0.5:
            notes.append("replenishes low cash")
    return f"  [{'; '.join(notes)}]" if notes else ""


def format_status_hud(
    state: Dict[str, Any],
    valid_steps: int,
    records: Optional[List[Dict[str, Any]]] = None,
) -> str:
    summary = compute_board_summary(state)
    score_info = compute_brass_harbor_reward(state, valid_steps, records=records)
    era_str = str(state.get("era", "canal")).upper()
    turn = int(state.get("turn", 1))
    actions_left = int(state.get("actions_left", 0))
    game_over = bool(state.get("game_over", False))

    lines: List[str] = []
    lines.append("=" * 78)
    if game_over:
        outcome = "VICTORY!" if score_info["won"] else "MATCH COMPLETE"
        lines.append(
            f"BRASS: BIRMINGHAM — {outcome} (Steps played: {valid_steps} | Harbor Score: {score_info['reward']:.4f})"
        )
    else:
        lines.append(
            f"BRASS: BIRMINGHAM — {era_str} ERA · Round {turn}/10 · Actions left this round: {actions_left} (Step #{valid_steps})"
        )
    lines.append("=" * 78)

    inc_lbs = summary["income_pounds"]
    inc_sign = f"+£{inc_lbs}" if inc_lbs >= 0 else f"-£{abs(inc_lbs)}"
    coal_c = int(state.get("coal_cubes", 0))
    iron_c = int(state.get("iron_cubes", 0))
    lines.append(
        f"Economy   : Cash £{state.get('money', 0)} (spent £{state.get('spent', 0)} this round) | "
        f"Income {inc_sign}/rd (space {state.get('income_space', 10)}) | Deck {state.get('draw_pile', 0)} cards"
    )
    lines.append(
        f"Markets   : Coal £{coal_market_price(coal_c)} ({coal_c} cubes) | "
        f"Iron £{iron_market_price(iron_c)} ({iron_c} cubes) | "
        f"Beer: {summary['player_beer_barrels']} yours + {summary['merchant_beer_barrels']} merchant"
    )
    lines.append(
        f"Scoreboard: Banked VP: You {state.get('your_vp', 0)} vs Mautoma {state.get('automa_vp', 0)} | "
        f"Board Era Projection: +{summary['flipped_tile_vp']} tile VP +{summary['connected_link_vp']} link VP "
        f"({summary['player_unflipped_count']} unflipped / {summary['unflipped_tile_vp_at_risk']} VP pending)"
    )

    # Merchants
    merch_parts: List[str] = []
    for m in state.get("merchants") or []:
        mid = m.get("id", "?")
        tiles = [t for t in (m.get("tiles") or []) if t and t != "blank"]
        beers = sum(1 for b in (m.get("beer") or []) if b)
        if tiles:
            merch_parts.append(f"{mid}:{'/'.join(tiles)}(beer={beers})")
    if merch_parts:
        lines.append("Merchants : " + " | ".join(merch_parts))

    # Player mat next tech levels
    mat = state.get("mat") or {}
    mat_parts: List[str] = []
    for ind in ("cotton", "goods", "pottery", "coal", "iron", "brewery"):
        levels = mat.get(ind) or []
        lowest_lvl = 0
        total_rem = sum(int(x) for x in levels) if levels and isinstance(levels[0], (int, float)) else len(levels)
        if levels and isinstance(levels[0], (int, float)):
            for lvl_idx, cnt in enumerate(levels):
                if int(cnt) > 0:
                    lowest_lvl = lvl_idx + 1
                    break
        elif levels and isinstance(levels[0], dict):
            lowest_lvl = int(levels[0].get("level", 1))
        if lowest_lvl > 0:
            mat_parts.append(f"{ind}:L{lowest_lvl}({total_rem} left)")
        else:
            mat_parts.append(f"{ind}:empty")
    lines.append("Player Mat: " + " | ".join(mat_parts))

    # Hand cards
    hand_labels = state.get("hand_labels") or []
    if hand_labels:
        cards_str = " | ".join(f"[{i}] {lbl}" for i, lbl in enumerate(hand_labels))
        lines.append(f"Hand ({len(hand_labels)}) : {cards_str}")

    # Built player tiles & links summary
    my_tiles: List[str] = []
    for loc, slots in (state.get("board") or {}).items():
        if not isinstance(slots, list):
            continue
        for tile in slots:
            if isinstance(tile, dict) and tile.get("owner") == 0:
                status_tag = "FLIPPED" if tile.get("flipped") else f"unflipped(res={tile.get('resources', 0)})"
                my_tiles.append(f"{loc}:{tile.get('industry')} L{tile.get('level')}[{status_tag}]")
    if my_tiles:
        lines.append("Your Tiles: " + ", ".join(my_tiles))

    my_links = [lid for lid, own in (state.get("links") or {}).items() if own == 0]
    if my_links:
        lines.append(f"Your Links ({len(my_links)}): " + ", ".join(my_links))

    recent_log = [str(x).strip() for x in (state.get("recent_log") or []) if str(x).strip()]
    if recent_log:
        lines.append("Recent Log: " + " || ".join(recent_log[-3:]))

    if game_over:
        lines.append("-" * 78)
        lines.append(
            f"FINAL HARBOR SCORE PREVIEW: {score_info['reward']:.4f} "
            f"(Completion={score_info['completion_score']:.3f} + "
            f"VP={score_info['vp_score']:.3f} + "
            f"VsMautoma={score_info['competitive_score']:.3f})"
        )
    return "\n".join(lines)


def format_legal_actions(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    if state.get("game_over"):
        return "Game is already over (`game_over: true`). No further actions needed."
    if not actions:
        return "No legal actions available."

    lines = [f"LEGAL ACTIONS ({len(actions)} available — execute with `brass play <id>`):"]
    for idx, act in enumerate(actions):
        desc = describe_macro_action(act)
        hint = annotate_action_hint(act, state)
        lines.append(f"  [{idx:2d}] {desc}{hint}")
    return "\n".join(lines)
