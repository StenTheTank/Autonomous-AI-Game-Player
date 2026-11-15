import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

STATE_DIM = 288
ACTION_DIM = 48

# Canonical 22 board locations in solo.kdc.sh/brass-birmingham (20 cities + 2 farm breweries)
CANONICAL_LOCATIONS: List[str] = [
    "belper",
    "derby",
    "leek",
    "stoke",
    "stone",
    "uttoxeter",
    "stafford",
    "burton",
    "cannock",
    "tamworth",
    "walsall",
    "wolverhampton",
    "coalbrookdale",
    "dudley",
    "kidderminster",
    "worcester",
    "birmingham",
    "coventry",
    "nuneaton",
    "redditch",
    "farmNorth",
    "farmSouth",
]

CITY_DISPLAY_TO_ID: Dict[str, str] = {
    "belper": "belper",
    "derby": "derby",
    "leek": "leek",
    "stoke-on-trent": "stoke",
    "stoke": "stoke",
    "stone": "stone",
    "uttoxeter": "uttoxeter",
    "stafford": "stafford",
    "burton-on-trent": "burton",
    "burton": "burton",
    "cannock": "cannock",
    "tamworth": "tamworth",
    "walsall": "walsall",
    "wolverhampton": "wolverhampton",
    "coalbrookdale": "coalbrookdale",
    "dudley": "dudley",
    "kidderminster": "kidderminster",
    "worcester": "worcester",
    "birmingham": "birmingham",
    "coventry": "coventry",
    "nuneaton": "nuneaton",
    "redditch": "redditch",
    "farm brewery (north)": "farmNorth",
    "farmnorth": "farmNorth",
    "farm brewery (south)": "farmSouth",
    "farmsouth": "farmSouth",
}

MERCHANT_IDS: List[str] = [
    "shrewsbury",
    "gloucester",
    "oxford",
    "warrington",
    "nottingham",
]

INDUSTRIES: List[str] = ["cotton", "goods", "pottery", "coal", "iron", "brewery"]
INDUSTRY_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(INDUSTRIES)}

ACTION_TYPES: List[str] = ["build", "network", "sell", "develop", "loan", "scout", "pass"]
ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ACTION_TYPES)}

# Canonical 39 links matching solo.kdc.sh/brass-birmingham SVG link IDs
CANONICAL_LINKS: List[Tuple[str, Tuple[str, ...]]] = [
    ("belper-derby", ("belper", "derby")),
    ("belper-leek", ("belper", "leek")),
    ("birmingham-coventry", ("birmingham", "coventry")),
    ("birmingham-dudley", ("birmingham", "dudley")),
    ("birmingham-nuneaton", ("birmingham", "nuneaton")),
    ("birmingham-oxford", ("birmingham", "oxford")),
    ("birmingham-redditch", ("birmingham", "redditch")),
    ("birmingham-tamworth", ("birmingham", "tamworth")),
    ("birmingham-walsall", ("birmingham", "walsall")),
    ("birmingham-worcester", ("birmingham", "worcester")),
    ("burton-cannock", ("burton", "cannock")),
    ("burton-derby", ("burton", "derby")),
    ("burton-stone", ("burton", "stone")),
    ("burton-tamworth", ("burton", "tamworth")),
    ("burton-walsall", ("burton", "walsall")),
    ("cannock-stafford", ("cannock", "stafford")),
    ("cannock-farmNorth", ("cannock", "farmNorth")),
    ("cannock-walsall", ("cannock", "walsall")),
    ("cannock-wolverhampton", ("cannock", "wolverhampton")),
    ("coalbrookdale-kidderminster", ("coalbrookdale", "kidderminster")),
    ("coalbrookdale-shrewsbury", ("coalbrookdale", "shrewsbury")),
    ("coalbrookdale-wolverhampton", ("coalbrookdale", "wolverhampton")),
    ("coventry-nuneaton", ("coventry", "nuneaton")),
    ("derby-nottingham", ("derby", "nottingham")),
    ("derby-uttoxeter", ("derby", "uttoxeter")),
    ("dudley-kidderminster", ("dudley", "kidderminster")),
    ("dudley-wolverhampton", ("dudley", "wolverhampton")),
    ("gloucester-redditch", ("gloucester", "redditch")),
    ("gloucester-worcester", ("gloucester", "worcester")),
    ("kidderminster-farmSouth-worcester", ("kidderminster", "farmSouth", "worcester")),
    ("leek-stoke", ("leek", "stoke")),
    ("nuneaton-tamworth", ("nuneaton", "tamworth")),
    ("oxford-redditch", ("oxford", "redditch")),
    ("stafford-stone", ("stafford", "stone")),
    ("stoke-stone", ("stoke", "stone")),
    ("stoke-warrington", ("stoke", "warrington")),
    ("stone-uttoxeter", ("stone", "uttoxeter")),
    ("tamworth-walsall", ("tamworth", "walsall")),
    ("walsall-wolverhampton", ("walsall", "wolverhampton")),
]
LINK_ID_TO_ENDPOINTS: Dict[str, Tuple[str, ...]] = dict(CANONICAL_LINKS)

# Brass: Birmingham official tile stats: (vp, income_spaces, link_dots, beer_needed, initial_cubes, can_develop, canal_only)
TILE_STATS: Dict[Tuple[str, int], Tuple[int, int, int, int, int, bool, bool]] = {
    # Cotton
    ("cotton", 1): (5, 5, 1, 1, 0, True, True),
    ("cotton", 2): (5, 4, 2, 1, 0, True, False),
    ("cotton", 3): (9, 3, 1, 1, 0, True, False),
    ("cotton", 4): (12, 2, 1, 1, 0, True, False),
    # Manufactured Goods
    ("goods", 1): (3, 5, 2, 1, 0, True, True),
    ("goods", 2): (5, 1, 1, 1, 0, True, False),
    ("goods", 3): (4, 4, 0, 0, 0, True, False),
    ("goods", 4): (3, 6, 1, 1, 0, True, False),
    ("goods", 5): (8, 2, 2, 2, 0, True, False),
    ("goods", 6): (7, 6, 1, 1, 0, True, False),
    ("goods", 7): (9, 4, 0, 0, 0, True, False),
    ("goods", 8): (11, 1, 1, 2, 0, True, False),
    # Pottery (lightbulb icons on L1 & L3 cannot be developed)
    ("pottery", 1): (10, 5, 1, 1, 0, False, False),
    ("pottery", 2): (1, 1, 1, 1, 0, True, False),
    ("pottery", 3): (11, 5, 1, 2, 0, False, False),
    ("pottery", 4): (1, 1, 1, 1, 0, True, False),
    ("pottery", 5): (20, 5, 1, 2, 0, True, False),
    # Coal Mine
    ("coal", 1): (1, 4, 2, 0, 2, True, True),
    ("coal", 2): (2, 7, 1, 0, 3, True, False),
    ("coal", 3): (3, 6, 1, 0, 4, True, False),
    ("coal", 4): (4, 5, 1, 0, 5, True, False),
    # Iron Works
    ("iron", 1): (3, 3, 1, 0, 4, True, True),
    ("iron", 2): (5, 3, 1, 0, 4, True, False),
    ("iron", 3): (7, 2, 1, 0, 5, True, False),
    ("iron", 4): (9, 1, 1, 0, 6, True, False),
    # Brewery (1 barrel in Canal, 2 barrels in Rail for L2-L4)
    ("brewery", 1): (4, 4, 2, 0, 1, True, True),
    ("brewery", 2): (5, 5, 2, 0, 1, True, False),
    ("brewery", 3): (7, 5, 2, 0, 1, True, False),
    ("brewery", 4): (10, 5, 2, 0, 1, True, False),
}


def income_space_to_pounds(space: int) -> int:
    """Convert Brass Birmingham income track space (0..99) to £ income (-10..+30)."""
    s = max(0, min(99, int(space)))
    if s <= 10:
        return s - 10
    if s <= 30:
        return (s - 9) // 2
    if s <= 60:
        return 10 + (s - 28) // 3
    return min(30, 20 + (s - 57) // 4)


def coal_market_price(cubes: int) -> int:
    c = max(0, int(cubes))
    if c == 0:
        return 8
    return max(1, 8 - (c + 1) // 2)


def iron_market_price(cubes: int) -> int:
    c = max(0, int(cubes))
    if c == 0:
        return 6
    return max(1, 6 - (c + 1) // 2)


def normalize_city_id(raw_city: str) -> str:
    cleaned = (raw_city or "").strip().lower()
    return CITY_DISPLAY_TO_ID.get(cleaned, cleaned)


def get_connected_component_and_network(state: Dict[str, Any]) -> Tuple[set, Dict[str, set]]:
    """
    Build adjacency graph of placed links (both players for coal/merchant transport,
    plus perspective player's personal network of cities with own tiles or adjacent to own links).
    """
    links = state.get("links") or {}
    board = state.get("board") or {}
    self_id = int(state.get("self_player_id", 0))

    adj: Dict[str, set] = {}
    player_network: set = set()

    for link_id, owner in links.items():
        if owner is None:
            continue
        endpoints = LINK_ID_TO_ENDPOINTS.get(link_id)
        if not endpoints:
            parts = link_id.split("-")
            endpoints = tuple(parts)
        for u in endpoints:
            adj.setdefault(u, set())
            if owner == self_id:
                player_network.add(u)
            for v in endpoints:
                if u != v:
                    adj[u].add(v)

    for loc, slots in board.items():
        if not isinstance(slots, list):
            continue
        for tile in slots:
            if isinstance(tile, dict) and tile.get("owner") == self_id:
                player_network.add(loc)

    return player_network, adj


def reachable_nodes(start: str, adj: Dict[str, set]) -> set:
    visited = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return visited


def compute_board_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute exact domain metrics used for state encoding, spectator HUD, and dense reward shaping."""
    board = state.get("board") or {}
    links = state.get("links") or {}
    era = state.get("era", "canal")
    mode = state.get("mode", "solo")
    self_id = int(state.get("self_player_id", 0))
    opp_id = 1 - self_id

    flipped_tile_vp = 0
    unflipped_tile_vp_at_risk = 0
    player_flipped_count = 0
    player_unflipped_count = 0
    player_beer_barrels_on_board = 0
    player_remaining_mine_cubes = 0

    opp_flipped_tile_vp = 0
    opp_unflipped_tile_vp_at_risk = 0
    city_link_dots: Dict[str, int] = {}

    for loc, slots in board.items():
        dots = 0
        if isinstance(slots, list):
            for tile in slots:
                if not isinstance(tile, dict):
                    continue
                ind = tile.get("industry", "")
                lvl = int(tile.get("level", 1))
                flipped = bool(tile.get("flipped", False))
                owner = int(tile.get("owner", -1))
                res = int(tile.get("resources", 0))
                vp, _, link_dots, _, _, _, _ = TILE_STATS.get((ind, lvl), (3, 2, 1, 0, 0, True, False))
                # Official Brass rules: flipped tiles contribute link dots (in solo Mautoma mode, Automa unflipped tiles also count in engine; in 2p only flipped tiles count)
                if flipped or (mode == "solo" and owner == 1):
                    dots += link_dots
                if owner == self_id:
                    if flipped:
                        flipped_tile_vp += vp
                        player_flipped_count += 1
                    else:
                        player_unflipped_count += 1
                        unflipped_tile_vp_at_risk += vp
                        if ind == "brewery":
                            player_beer_barrels_on_board += max(1, res)
                        elif ind in ("coal", "iron"):
                            player_remaining_mine_cubes += res
                elif owner == opp_id:
                    if flipped:
                        opp_flipped_tile_vp += vp
                    else:
                        opp_unflipped_tile_vp_at_risk += vp
        city_link_dots[loc] = dots

    # Merchant ports give 2 link dots per merchant link
    for m in state.get("merchants") or []:
        mid = m.get("id")
        if mid and any(t not in ("blank", None) for t in (m.get("tiles") or [])):
            city_link_dots[mid] = 2

    connected_link_vp = 0
    player_link_count = 0
    opp_connected_link_vp = 0
    opp_link_count = 0
    for link_id, owner in links.items():
        endpoints = LINK_ID_TO_ENDPOINTS.get(link_id, tuple(link_id.split("-")))
        lvp = sum(city_link_dots.get(ep, 0) for ep in endpoints)
        if owner == self_id:
            player_link_count += 1
            connected_link_vp += lvp
        elif owner == opp_id:
            opp_link_count += 1
            opp_connected_link_vp += lvp

    merchant_beer_total = 0
    for m in state.get("merchants") or []:
        for has_beer in m.get("beer") or []:
            if has_beer:
                merchant_beer_total += 1

    income_space = int(state.get("income_space", 10))
    return {
        "era": era,
        "flipped_tile_vp": flipped_tile_vp,
        "unflipped_tile_vp_at_risk": unflipped_tile_vp_at_risk,
        "player_flipped_count": player_flipped_count,
        "player_unflipped_count": player_unflipped_count,
        "player_beer_barrels": player_beer_barrels_on_board,
        "merchant_beer_barrels": merchant_beer_total,
        "player_remaining_mine_cubes": player_remaining_mine_cubes,
        "connected_link_vp": connected_link_vp,
        "player_link_count": player_link_count,
        "opp_flipped_tile_vp": opp_flipped_tile_vp,
        "opp_unflipped_tile_vp_at_risk": opp_unflipped_tile_vp_at_risk,
        "opp_connected_link_vp": opp_connected_link_vp,
        "opp_link_count": opp_link_count,
        "city_link_dots": city_link_dots,
        "income_pounds": income_space_to_pounds(income_space),
        "income_space": income_space,
    }


def score_card_discard_cost(card: Dict[str, Any], player_network: set, era: str) -> float:
    """
    Lower score = BETTER discard candidate (lowest opportunity cost to throw away for Network/Develop/Sell/Loan/Scout).
    Wildcards have highest cost (never discard if avoidable!).
    """
    kind = card.get("kind", "")
    if kind == "wildLocation":
        return 10.0
    if kind == "wildIndustry":
        return 9.5
    if kind == "industry":
        inds = card.get("industries") or []
        if "pottery" in inds:
            return 2.5
        if "brewery" in inds:
            return 5.5
        if "cotton" in inds or "goods" in inds:
            return 4.5
        return 3.5
    if kind == "location":
        city = card.get("city", "")
        # Location cards outside our network are extra valuable because they let us teleport!
        if city and city not in player_network:
            return 5.0 if city in ("birmingham", "coventry", "derby", "stoke", "coalbrookdale") else 4.0
        return 2.0
    return 3.0


def rank_discard_slots(state: Dict[str, Any]) -> List[int]:
    """Return hand slot indices sorted from BEST discard (lowest utility) to MOST valuable."""
    hand = state.get("hand") or []
    player_net, _ = get_connected_component_and_network(state)
    era = state.get("era", "canal")
    scored = []
    for idx, card in enumerate(hand):
        cost = score_card_discard_cost(card, player_net, era)
        scored.append((cost, idx))
    scored.sort(key=lambda x: x[0])
    return [idx for _, idx in scored]


def encode_state(state: Dict[str, Any]) -> np.ndarray:
    """Encode full Brass: Birmingham state into a 288-dim float32 vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    summary = compute_board_summary(state)
    player_net, adj = get_connected_component_and_network(state)

    era_is_rail = 1.0 if state.get("era") == "rail" else 0.0
    turn = float(state.get("turn", 1))
    turns_remaining_in_era = max(0.0, 10.0 - turn)
    actions_left = float(state.get("actions_left", 1))
    money = float(state.get("money", 17))
    income_space = float(state.get("income_space", 10))
    income_pounds = float(summary["income_pounds"])
    your_vp = float(state.get("your_vp", 0))
    automa_vp = float(state.get("automa_vp", 0))
    coal_cubes = float(state.get("coal_cubes", 13))
    iron_cubes = float(state.get("iron_cubes", 8))
    draw_pile = float(state.get("draw_pile", 11))

    # [0..19] Global scalar features (20 dims)
    vec[0] = era_is_rail
    vec[1] = turn / 10.0
    vec[2] = turns_remaining_in_era / 10.0
    vec[3] = 1.0 if turns_remaining_in_era <= 2.0 else 0.0  # Late-era crunch indicator!
    vec[4] = actions_left / 2.0
    vec[5] = min(2.0, money / 40.0)
    vec[6] = (income_pounds + 10.0) / 40.0
    vec[7] = income_space / 99.0
    vec[8] = your_vp / 150.0
    vec[9] = automa_vp / 150.0
    vec[10] = np.clip((your_vp - automa_vp) / 60.0, -1.5, 1.5)
    vec[11] = coal_cubes / 14.0
    vec[12] = coal_market_price(int(coal_cubes)) / 8.0
    vec[13] = iron_cubes / 10.0
    vec[14] = iron_market_price(int(iron_cubes)) / 6.0
    vec[15] = draw_pile / 20.0
    vec[16] = summary["flipped_tile_vp"] / 80.0
    vec[17] = summary["unflipped_tile_vp_at_risk"] / 50.0
    vec[18] = summary["connected_link_vp"] / 50.0
    vec[19] = (summary["player_beer_barrels"] + summary["merchant_beer_barrels"]) / 8.0

    # [20..51] Hand summary: 20 city counts + 6 industry counts + 2 wild counts + 4 aggregate hand stats (32 dims)
    hand = state.get("hand") or []
    city_idx_map = {c: i for i, c in enumerate(CANONICAL_LOCATIONS[:20])}
    wild_loc = 0
    wild_ind = 0
    for card in hand:
        kind = card.get("kind", "")
        if kind == "location":
            c = card.get("city", "")
            if c in city_idx_map:
                vec[20 + city_idx_map[c]] += 0.5
        elif kind == "industry":
            for ind in card.get("industries") or []:
                if ind in INDUSTRY_TO_IDX:
                    vec[40 + INDUSTRY_TO_IDX[ind]] += 0.5
        elif kind == "wildLocation":
            wild_loc += 1
        elif kind == "wildIndustry":
            wild_ind += 1
    vec[46] = float(wild_loc)
    vec[47] = float(wild_ind)
    vec[48] = len(hand) / 8.0
    vec[49] = 1.0 if state.get("availability", {}).get("sell") else 0.0
    vec[50] = 1.0 if state.get("availability", {}).get("scout") else 0.0
    vec[51] = 1.0 if state.get("availability", {}).get("develop") else 0.0

    # [52..87] Player Mat tech stacks: 6 industries * 6 features (36 dims)
    mat = state.get("mat") or {}
    for i, ind in enumerate(INDUSTRIES):
        levels = mat.get(ind) or []
        lowest_lvl = 0
        total_rem = sum(int(x) for x in levels)
        for lvl_idx, cnt in enumerate(levels):
            if int(cnt) > 0:
                lowest_lvl = lvl_idx + 1
                break
        base = 52 + i * 6
        if lowest_lvl > 0:
            vp, inc, dots, beer_req, cubes, can_dev, canal_only = TILE_STATS.get(
                (ind, lowest_lvl), (3, 2, 1, 0, 0, True, False)
            )
            vec[base + 0] = lowest_lvl / 8.0
            vec[base + 1] = total_rem / 12.0
            vec[base + 2] = vp / 20.0
            vec[base + 3] = inc / 7.0
            vec[base + 4] = float(dots) / 2.0
            vec[base + 5] = 1.0 if (canal_only and era_is_rail > 0.5) else (1.0 if can_dev else 0.0)

    # [88..219] Board locations: 22 locations * 6 features = 132 dims
    self_id = int(state.get("self_player_id", 0))
    opp_id = 1 - self_id
    board = state.get("board") or {}
    city_dots = summary["city_link_dots"]
    for loc_i, loc in enumerate(CANONICAL_LOCATIONS):
        slots = board.get(loc) or []
        total_slots = max(1, len(slots))
        empty_slots = 0
        my_flipped = 0
        my_unflipped = 0
        opp_tiles = 0
        loc_resources = 0
        for tile in slots:
            if tile is None:
                empty_slots += 1
            elif isinstance(tile, dict):
                loc_resources += int(tile.get("resources", 0))
                if tile.get("owner") == self_id:
                    if tile.get("flipped"):
                        my_flipped += 1
                    else:
                        my_unflipped += 1
                else:
                    opp_tiles += 1
        b = 88 + loc_i * 6
        vec[b + 0] = empty_slots / float(total_slots)
        vec[b + 1] = my_flipped / float(total_slots)
        vec[b + 2] = my_unflipped / float(total_slots)
        vec[b + 3] = opp_tiles / float(total_slots)
        vec[b + 4] = min(1.0, loc_resources / 5.0)
        vec[b + 5] = (1.0 if loc in player_net else 0.0) * 0.5 + min(0.5, city_dots.get(loc, 0) / 8.0)

    # [220..258] Canonical 39 links ownership: +1.0 = perspective player, -0.5 = opponent, 0.0 = open (39 dims)
    links = state.get("links") or {}
    for l_i, (link_id, _) in enumerate(CANONICAL_LINKS):
        owner = links.get(link_id)
        if owner == self_id:
            vec[220 + l_i] = 1.0
        elif owner == opp_id:
            vec[220 + l_i] = -0.5

    # [259..287] Merchant ports (5 merchants * 5 dims = 25 dims) + 4 global strategic ratios = 29 dims -> total 288!
    merchants_by_id = {m.get("id"): m for m in (state.get("merchants") or []) if isinstance(m, dict)}
    all_player_reach = set()
    for node in player_net:
        all_player_reach |= reachable_nodes(node, adj)

    for m_i, mid in enumerate(MERCHANT_IDS):
        m = merchants_by_id.get(mid) or {}
        tiles = m.get("tiles") or []
        beers = m.get("beer") or []
        b = 259 + m_i * 5
        vec[b + 0] = 1.0 if ("cotton" in tiles or "any" in tiles) else 0.0
        vec[b + 1] = 1.0 if ("goods" in tiles or "any" in tiles) else 0.0
        vec[b + 2] = 1.0 if ("pottery" in tiles or "any" in tiles) else 0.0
        vec[b + 3] = sum(1.0 for x in beers if x) / 2.0
        vec[b + 4] = 1.0 if mid in all_player_reach else 0.0

    vec[284] = summary["player_beer_barrels"] / 4.0
    vec[285] = summary["player_remaining_mine_cubes"] / 10.0
    vec[286] = summary["player_link_count"] / 14.0
    vec[287] = (summary["player_flipped_count"] / max(1.0, summary["player_flipped_count"] + summary["player_unflipped_count"]))

    return vec


def encode_action(state: Dict[str, Any], action: Dict[str, Any]) -> np.ndarray:
    """Encode a legal Brass macro-action candidate into a 48-dim float32 feature vector."""
    feat = np.zeros(ACTION_DIM, dtype=np.float32)
    summary = compute_board_summary(state)
    player_net, adj = get_connected_component_and_network(state)

    act_type = action.get("action_type", "pass")
    if act_type in ACTION_TO_IDX:
        feat[ACTION_TO_IDX[act_type]] = 1.0

    money = float(state.get("money", 0))
    turn = float(state.get("turn", 1))
    turns_left = max(0.0, 10.0 - turn)
    cost = float(action.get("cost", 0))

    # [7..12] Primary industry one-hot
    ind = action.get("industry") or ""
    if not ind and action.get("industries"):
        ind = action["industries"][0]
    if ind in INDUSTRY_TO_IDX:
        feat[7 + INDUSTRY_TO_IDX[ind]] = 1.0

    # [13..20] Economic & cost features
    feat[13] = min(1.5, cost / 25.0)
    post_cash = money + (30.0 if act_type == "loan" else -cost)
    feat[14] = np.clip(post_cash / 40.0, -0.5, 2.0)
    feat[15] = 1.0 if post_cash < 4.0 else 0.0

    hand = state.get("hand") or []
    card_slot = action.get("card_slot")
    discard_cost = 3.0
    if isinstance(card_slot, int) and 0 <= card_slot < len(hand):
        discard_cost = score_card_discard_cost(hand[card_slot], player_net, state.get("era", "canal"))
    feat[16] = discard_cost / 10.0

    # Action-specific rich strategic features
    if act_type == "build":
        lvl = int(action.get("level", 1))
        city = normalize_city_id(action.get("city", ""))
        vp, inc, dots, beer_req, cubes, _, _ = TILE_STATS.get((ind, lvl), (4, 3, 1, 1, 0, True, False))
        if ind == "brewery" and state.get("era") == "rail" and lvl >= 2:
            cubes = 2
        feat[17] = lvl / 8.0
        feat[18] = vp / 20.0
        feat[19] = inc / 7.0
        feat[20] = dots / 2.0
        feat[21] = beer_req / 2.0
        feat[22] = cubes / 6.0
        feat[23] = 1.0 if action.get("is_overbuild") else 0.0

        # Check merchant connectivity & beer availability for consumer tiles!
        reach = reachable_nodes(city, adj) if city else set()
        has_matching_merchant = False
        has_merchant_beer = False
        for m in state.get("merchants") or []:
            if m.get("id") in reach:
                mtiles = m.get("tiles") or []
                if ind in mtiles or "any" in mtiles:
                    has_matching_merchant = True
                    if any(m.get("beer") or []):
                        has_merchant_beer = True
        total_avail_beer = summary["player_beer_barrels"] + (1 if has_merchant_beer else 0)
        feat[24] = 1.0 if has_matching_merchant else 0.0
        feat[25] = 1.0 if (beer_req == 0 or total_avail_beer >= beer_req) else 0.0
        # Late-era unflipped trap indicator: building consumer without beer or huge cube mine in Round 9-10!
        late_unflippable = False
        if turns_left <= 1.0 and beer_req > 0 and (not has_matching_merchant or total_avail_beer < beer_req):
            late_unflippable = True
        if turns_left <= 1.0 and ind in ("coal", "iron") and cubes >= 4:
            late_unflippable = True
        feat[26] = 1.0 if late_unflippable else 0.0
        # Immediate market sale potential for coal/iron
        if ind == "coal" and int(state.get("coal_cubes", 13)) < 14 and any(m in reach for m in MERCHANT_IDS):
            feat[27] = min(1.0, (14 - int(state.get("coal_cubes", 13))) / max(1, cubes))
        elif ind == "iron" and int(state.get("iron_cubes", 8)) < 10:
            feat[27] = min(1.0, (10 - int(state.get("iron_cubes", 8))) / max(1, cubes))

        # Player links already touching this city (immediate link VP multiplier when flipped!)
        self_id = int(state.get("self_player_id", 0))
        links = state.get("links") or {}
        touching_player_links = 0
        for lid, owner in links.items():
            if owner == self_id and city in LINK_ID_TO_ENDPOINTS.get(lid, tuple(lid.split("-"))):
                touching_player_links += 1
        feat[28] = min(1.0, (touching_player_links * dots) / 6.0)

    elif act_type == "network":
        link_ids = action.get("link_ids") or ([action.get("link_id", "")] if action.get("link_id") else [])
        city_dots = summary["city_link_dots"]
        immediate_link_vp = 0
        connects_merchant = False
        expands_new_city = False
        sim_net = set(player_net)
        for lid in link_ids:
            endpoints = LINK_ID_TO_ENDPOINTS.get(lid, tuple(lid.split("-")))
            immediate_link_vp += sum(city_dots.get(ep, 0) for ep in endpoints)
            if any(ep in MERCHANT_IDS for ep in endpoints):
                connects_merchant = True
            if any(ep not in sim_net and ep not in MERCHANT_IDS for ep in endpoints):
                expands_new_city = True
            for ep in endpoints:
                sim_net.add(ep)
        feat[29] = min(1.8, immediate_link_vp / 10.0)
        feat[30] = 1.0 if connects_merchant else 0.0
        feat[31] = 1.0 if expands_new_city else 0.0

    elif act_type == "sell":
        sales = action.get("sales") or []
        total_vp = 0
        total_inc = 0
        total_dots = 0
        for s in sales:
            s_ind = s.get("industry", "cotton")
            s_lvl = int(s.get("level", 1))
            vp, inc, dots, _, _, _, _ = TILE_STATS.get((s_ind, s_lvl), (5, 3, 1, 1, 0, True, False))
            total_vp += vp
            total_inc += inc
            total_dots += dots
        feat[32] = len(sales) / 4.0
        feat[33] = total_vp / 25.0
        feat[34] = total_inc / 10.0
        feat[35] = total_dots / 6.0

    elif act_type == "develop":
        inds = action.get("industries") or []
        feat[36] = len(inds) / 2.0  # Explicitly rewards Develop 2 tiles over 1 tile!
        unlocks_non_canal = 0.0
        mat = state.get("mat") or {}
        skip_by_ind: Dict[str, int] = {}
        for d_ind in inds:
            skip = skip_by_ind.get(d_ind, 0)
            skip_by_ind[d_ind] = skip + 1
            levels = mat.get(d_ind) or []
            seen = 0
            for idx_l, cnt in enumerate(levels):
                c = int(cnt)
                if c > 0:
                    if seen + c > skip:
                        _, _, _, _, _, _, canal_only = TILE_STATS.get((d_ind, idx_l + 1), (3, 2, 1, 0, 0, True, False))
                        if canal_only:
                            unlocks_non_canal += 0.5
                        break
                    seen += c
        feat[37] = unlocks_non_canal
        feat[38] = 1.0 if cost <= 4.0 else 0.0  # Cheap iron develop bonus

    elif act_type == "loan":
        inc_space = int(state.get("income_space", 10))
        # Take a loan when low on cash (<= £6) with turns left to invest it; avoid wasteful loans when cash >= £11 or on final turn!
        feat[39] = 1.0 if (money <= 6.0 and turns_left >= 1.0) else 0.0
        feat[40] = 1.0 if (money >= 11.0 or (turns_left <= 0.0 and money >= 5.0) or (inc_space < 4 and money >= 6.0)) else 0.0

    elif act_type == "scout":
        feat[41] = 1.0 if summary["player_beer_barrels"] == 0 else 0.0
        feat[42] = 1.0 if len(hand) >= 5 else 0.0
        feat[43] = turns_left / 10.0

    elif act_type == "pass":
        feat[44] = 1.0

    # Interaction context features [45..47]
    feat[45] = 1.0 if state.get("era") == "rail" else 0.0
    feat[46] = turns_left / 10.0
    feat[47] = 1.0  # Bias term

    return feat


def encode_action_list(state: Dict[str, Any], actions: List[Dict[str, Any]]) -> np.ndarray:
    if not actions:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack([encode_action(state, a) for a in actions], axis=0)
