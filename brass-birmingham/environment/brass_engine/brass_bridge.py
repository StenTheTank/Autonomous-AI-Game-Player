import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Dict, Any, List, Optional

from brass_engine.brass_features import (
    CITY_DISPLAY_TO_ID,
    LINK_ID_TO_ENDPOINTS,
    MERCHANT_IDS,
    TILE_STATS,
    compute_board_summary,
    get_connected_component_and_network,
    iron_market_price,
    normalize_city_id,
    rank_discard_slots,
)

BRASS_SOLO_URL = "http://localhost:8000/brass-birmingham/"
_LOCAL_SERVER_URL: Optional[str] = None
_LOCAL_SERVER_LOCK = threading.Lock()


class _QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def ensure_local_brass_server() -> str:
    """Serve backend/webapp over an ephemeral 127.0.0.1 HTTP port for offline/CLI RL workers."""
    global _LOCAL_SERVER_URL
    with _LOCAL_SERVER_LOCK:
        if _LOCAL_SERVER_URL is not None:
            return _LOCAL_SERVER_URL
        webapp_dir = Path(__file__).resolve().parent.parent / "webapp"
        if not (webapp_dir / "index.html").exists():
            return "https://solo.kdc.sh/brass-birmingham/"
        handler = partial(_QuietStaticHandler, directory=str(webapp_dir))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True, name="brass-local-webapp")
        t.start()
        _LOCAL_SERVER_URL = f"http://127.0.0.1:{port}/"
        return _LOCAL_SERVER_URL

_BUILD_OPT_RE = re.compile(
    r"^(?P<city>.+?):\s*(?P<industry>[a-zA-Z]+)\s+L(?P<level>\d+)\s*[—\-]\s*£(?P<cost>\d+)(?P<over>\s*\(overbuild\))?",
    re.IGNORECASE,
)
_NET_OPT_RE = re.compile(
    r"^(?P<links>[a-zA-Z0-9\-]+(?:\s*\+\s*[a-zA-Z0-9\-]+)?)\s*[—\-]\s*£(?P<cost>\d+)",
    re.IGNORECASE,
)
_SELL_OPT_RE = re.compile(
    r"^(?P<city>.+?):\s*(?P<industry>[a-zA-Z]+)\s+L(?P<level>\d+)\s*\(beer\s*[×x]\s*(?P<beer>\d+)\)",
    re.IGNORECASE,
)


def format_card_label(card: Dict[str, Any]) -> str:
    kind = card.get("kind", "")
    if kind == "location":
        return str(card.get("city", "location"))
    if kind == "industry":
        inds = card.get("industries") or []
        return "/".join(inds) if inds else "industry"
    if kind == "wildLocation":
        return "Wild location"
    if kind == "wildIndustry":
        return "Wild industry"
    return str(kind or "card")


def describe_macro_action(action: Dict[str, Any]) -> str:
    t = action.get("action_type", "pass")
    if t == "build":
        over = " (overbuild)" if action.get("is_overbuild") else ""
        return (
            f"Build {action.get('city_display', action.get('city'))}: "
            f"{action.get('industry')} L{action.get('level')} (£{action.get('cost')}){over} "
            f"[card #{action.get('card_slot')}: {action.get('card_label')}]"
        )
    if t == "network":
        lids = action.get("link_ids") or ([action.get("link_id")] if action.get("link_id") else [])
        if len(lids) == 2:
            return (
                f"Network 2x rail: {lids[0]} + {lids[1]} (£{action.get('cost')}, +1 beer) "
                f"[discard #{action.get('card_slot')}: {action.get('card_label')}]"
            )
        return (
            f"Network {lids[0] if lids else action.get('link_id')} (£{action.get('cost')}) "
            f"[discard #{action.get('card_slot')}: {action.get('card_label')}]"
        )
    if t == "sell":
        sales = action.get("sales") or []
        names = ", ".join(f"{s.get('city_display', s.get('city'))} {s.get('industry')} L{s.get('level')}" for s in sales)
        return f"Sell {len(sales)}x ({names}) [discard #{action.get('card_slot')}: {action.get('card_label')}]"
    if t == "develop":
        inds = "+".join(action.get("industries") or [])
        return f"Develop {len(action.get('industries') or [])}x ({inds}, ~£{action.get('cost')}) [discard #{action.get('card_slot')}: {action.get('card_label')}]"
    if t == "loan":
        return f"Loan +£30 (-3 income tiers) [discard #{action.get('card_slot')}: {action.get('card_label')}]"
    if t == "scout":
        labels = ", ".join(action.get("discard_labels") or [])
        return f"Scout (+Wild Location & +Wild Industry) [discard 3: {labels}]"
    if t == "pass":
        return f"Pass [discard #{action.get('card_slot')}: {action.get('card_label')}]"
    return t


async def inject_fast_training_css(page) -> None:
    """Disable CSS transitions/animations in headless RL workers for maximum steps/sec."""
    await page.evaluate(
        """() => {
            if (!document.getElementById('__brass_rl_fast_css')) {
                const st = document.createElement('style');
                st.id = '__brass_rl_fast_css';
                st.textContent = '* { transition: none !important; animation: none !important; scroll-behavior: auto !important; }';
                document.head.appendChild(st);
            }
        }"""
    )


async def start_new_brass_game(
    page,
    seed: Optional[int] = None,
    difficulty: str = "easy",
    mode: str = "solo",
    player_names: Optional[List[str]] = None,
) -> None:
    """Reset Brass: Birmingham into a fresh Round 1 Canal Era match ('solo' or official '2p' ruleset)."""
    if os.environ.get("BRASS_ENFORCE_SINGLE_BOOT") == "1":
        boot_log = Path(os.environ.get("BRASS_BOOT_LOG", "/var/tmp/.brass_engine_boots"))
        try:
            boot_log.parent.mkdir(parents=True, exist_ok=True)
            with boot_log.open("a", encoding="utf-8", newline="\n") as bf:
                bf.write(f"{time.time():.6f}:{os.getpid()}\n")
        except OSError:
            pass
    chosen_seed = seed if seed is not None else random.randint(1, 999999)
    names = list(player_names[:2]) if player_names and len(player_names) >= 2 else None
    await page.evaluate(
        """async ({ seed, difficulty, mode, playerNames }) => {
            const sleep = (ms) => new Promise(r => setTimeout(r, ms));
            let startBtn = document.querySelector('[data-testid="start-game"]');
            if (!startBtn) {
                const newGameBtn = document.querySelector('[data-testid="new-game"]');
                if (newGameBtn) {
                    newGameBtn.click();
                    await sleep(15);
                }
                startBtn = document.querySelector('[data-testid="start-game"]');
            }
            if (startBtn) {
                const selectSetter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
                const inputSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;

                const modeSel = document.querySelector('[data-testid="game-mode"]');
                if (modeSel && modeSel.value !== mode) {
                    if (selectSetter) selectSetter.call(modeSel, mode);
                    modeSel.dispatchEvent(new Event('change', { bubbles: true }));
                    await sleep(15);
                }
                if (mode === '2p' && Array.isArray(playerNames) && playerNames.length >= 2) {
                    const p1Input = document.querySelector('[data-testid="p1-name"]');
                    if (p1Input && inputSetter) {
                        inputSetter.call(p1Input, String(playerNames[0]));
                        p1Input.dispatchEvent(new Event('input', { bubbles: true }));
                        p1Input.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    const p2Input = document.querySelector('[data-testid="p2-name"]');
                    if (p2Input && inputSetter) {
                        inputSetter.call(p2Input, String(playerNames[1]));
                        p2Input.dispatchEvent(new Event('input', { bubbles: true }));
                        p2Input.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                } else {
                    const diffSel = document.querySelector('[data-testid="difficulty"]');
                    if (diffSel && diffSel.value !== difficulty) {
                        if (selectSetter) selectSetter.call(diffSel, difficulty);
                        diffSel.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
                const seedInput = document.querySelector('[data-testid="seed"]');
                if (seedInput) {
                    if (inputSetter) inputSetter.call(seedInput, String(seed));
                    seedInput.dispatchEvent(new Event('input', { bubbles: true }));
                    seedInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
                await sleep(10);
                startBtn.click();
                await sleep(25);
            }
        }""",
        {
            "seed": int(chosen_seed),
            "difficulty": difficulty,
            "mode": mode,
            "playerNames": names,
        },
    )


async def extract_brass_state(
    page,
    reset_open_flow: bool = True,
    perspective_player: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract exact Brass: Birmingham state from localStorage + live DOM button availability.
    Supports both 'solo' (vs Mautoma) and '2p' (head-to-head seat perspective).
    """
    raw = await page.evaluate(
        """async (resetFlow) => {
            if (resetFlow) {
                const cancelBtn = document.querySelector('[data-testid="flow-cancel"]');
                if (cancelBtn) {
                    cancelBtn.click();
                    await new Promise(r => setTimeout(r, 5));
                }
            }
            const startBtn = document.querySelector('[data-testid="start-game"]');
            let saved = null;
            try {
                const s = localStorage.getItem('bbsolo-save-v1');
                if (s) saved = JSON.parse(s);
            } catch (e) {}

            const availability = {};
            for (const act of ['build', 'network', 'sell', 'develop', 'loan', 'scout', 'pass']) {
                const btn = document.querySelector(`[data-testid="action-${act}"]`);
                availability[act] = !!(btn && !btn.disabled);
            }
            const chipLabels = Array.from(document.querySelectorAll('[data-testid^="hand-card-"]')).map(
                el => (el.innerText || el.textContent || '').trim()
            );
            const eraTurnText = (document.querySelector('[data-testid="era-turn"]')?.innerText || '').trim();
            return {
                in_setup_modal: !!startBtn,
                saved,
                availability,
                chip_labels: chipLabels,
                era_turn_text: eraTurnText,
            };
        }""",
        reset_open_flow,
    )

    saved = raw.get("saved") or {}
    mode = saved.get("mode", "solo")
    players = saved.get("players") or [{}, {}]
    p0 = players[0] if len(players) > 0 else {}
    p1 = players[1] if len(players) > 1 else {}

    active_player = int(saved.get("activePlayer", 0)) if mode == "2p" else 0
    turn_order = [int(x) for x in (saved.get("turnOrder") or [0, 1])]
    turn_index = int(saved.get("turnIndex", 0))
    default_names = ["Player 1", "Player 2"] if mode == "2p" else ["You", "Automa"]
    raw_names = saved.get("playerNames") or default_names
    player_names = [
        str(raw_names[0]) if len(raw_names) > 0 else default_names[0],
        str(raw_names[1]) if len(raw_names) > 1 else default_names[1],
    ]

    self_id = int(perspective_player) if perspective_player in (0, 1) else active_player
    opp_id = 1 - self_id
    self_p = players[self_id] if len(players) > self_id else {}
    opp_p = players[opp_id] if len(players) > opp_id else {}

    hand = self_p.get("hand") or []
    dom_chips = raw.get("chip_labels") or []
    has_masked_chips = any("Hidden #" in str(lbl) for lbl in dom_chips)
    if self_id == active_player and len(dom_chips) == len(hand) and dom_chips and not has_masked_chips:
        chip_labels = dom_chips
    else:
        chip_labels = [format_card_label(c) for c in hand]

    players_info = []
    for seat_idx, p_obj in enumerate((p0, p1)):
        players_info.append(
            {
                "seat": seat_idx,
                "name": player_names[seat_idx],
                "money": int(p_obj.get("money", 17)),
                "spent": int(p_obj.get("spent", 0)),
                "income_space": int(p_obj.get("incomeSpace", 10)),
                "vp": int(p_obj.get("vp", 0)),
                "hand_count": len(p_obj.get("hand") or []),
                "discard_count": len(p_obj.get("discard") or []),
                "mat": p_obj.get("mat") or {},
            }
        )

    return {
        "in_setup_modal": bool(raw.get("in_setup_modal")),
        "mode": mode,
        "active_player": active_player,
        "self_player_id": self_id,
        "opp_player_id": opp_id,
        "turn_order": turn_order,
        "turn_index": turn_index,
        "player_names": player_names,
        "players_info": players_info,
        "era": saved.get("era", "canal"),
        "turn": int(saved.get("turn", 1)),
        "actions_left": int(saved.get("actionsLeft", 1)),
        "game_over": bool(saved.get("gameOver", False)),
        "money": int(self_p.get("money", 17)),
        "spent": int(self_p.get("spent", 0)),
        "income_space": int(self_p.get("incomeSpace", 10)),
        "your_vp": int(self_p.get("vp", 0)),
        "automa_vp": int(opp_p.get("vp", 0)),
        "draw_pile": len(saved.get("drawPile") or []),
        "coal_cubes": int(saved.get("coalCubes", 13)),
        "iron_cubes": int(saved.get("ironCubes", 8)),
        "hand": hand,
        "hand_labels": chip_labels,
        "mat": self_p.get("mat") or {},
        "automa_mat": opp_p.get("mat") or {},
        "board": saved.get("board") or {},
        "links": saved.get("links") or {},
        "merchants": saved.get("merchants") or [],
        "availability": raw.get("availability") or {},
        "era_turn_text": raw.get("era_turn_text", ""),
        "last_log": (saved.get("log") or [""])[-1] if saved.get("log") else "",
        "recent_log": (saved.get("log") or [])[-10:],
        "seed": None if os.environ.get("BRASS_ENFORCE_SINGLE_BOOT") == "1" else saved.get("seed"),
        "difficulty": saved.get("difficulty", ""),
    }


async def enumerate_legal_actions(page, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Non-destructively enumerate all legal Brass macro-actions in <15ms using the in-page flow + cancel button.
    Never clicks cards inside Loan, Scout, or Pass (which commit immediately); derives those directly.
    """
    if state.get("game_over") or state.get("in_setup_modal"):
        return []

    hand = state.get("hand") or []
    hand_labels = state.get("hand_labels") or []
    if not hand:
        return []

    discard_order = rank_discard_slots(state)
    best_discard_slot = discard_order[0] if discard_order else 0

    # Group hand slots by unique card key, picking the lowest-opportunity-cost slot per unique card
    unique_card_slots: List[int] = []
    seen_card_keys = set()
    for slot in discard_order:
        if slot >= len(hand):
            continue
        c = hand[slot]
        key = (
            c.get("kind"),
            c.get("city", ""),
            tuple(c.get("industries") or []),
        )
        if key not in seen_card_keys:
            seen_card_keys.add(key)
            unique_card_slots.append(slot)

    probed = await page.evaluate(
        """async ({ uniqueSlots, discardOrder }) => {
            const tick = () => new Promise(r => setTimeout(r, 0));
            const cancelFlow = async () => {
                const c = document.querySelector('[data-testid="flow-cancel"]');
                if (c) {
                    c.click();
                    await tick();
                }
            };
            const isCardSelectable = (slot) => {
                const chip = document.querySelector(`[data-testid="hand-card-${slot}"]`);
                if (!chip) return false;
                const op = parseFloat(chip.style.opacity || '1');
                return op > 0.5;
            };
            const pickFirstSelectable = (slots) => {
                for (const s of slots) {
                    if (isCardSelectable(s)) return s;
                }
                return null;
            };

            await cancelFlow();
            const result = {
                builds: [],
                networks: [],
                develop: null,
                sell: null,
            };

            // 1. Probe BUILD across unique selectable card types
            const buildBtn = document.querySelector('[data-testid="action-build"]');
            if (buildBtn && !buildBtn.disabled) {
                buildBtn.click();
                await tick();
                const selectableUnique = uniqueSlots.filter(isCardSelectable);
                for (const slot of selectableUnique) {
                    const bBtn = document.querySelector('[data-testid="action-build"]');
                    if (bBtn && !bBtn.classList.contains('selected')) {
                        bBtn.click();
                        await tick();
                    }
                    const chip = document.querySelector(`[data-testid="hand-card-${slot}"]`);
                    if (!chip || !isCardSelectable(slot)) continue;
                    chip.click();
                    await tick();
                    const opts = Array.from(document.querySelectorAll('[data-testid="build-options"] > button'));
                    opts.forEach((btn, idx) => {
                        if (!btn.disabled) {
                            result.builds.push({
                                card_slot: slot,
                                btn_index: idx,
                                text: (btn.innerText || btn.textContent || '').trim(),
                            });
                        }
                    });
                    await cancelFlow();
                }
                await cancelFlow();
            }

            // 2. Probe NETWORK with lowest-utility selectable card
            const netBtn = document.querySelector('[data-testid="action-network"]');
            if (netBtn && !netBtn.disabled) {
                netBtn.click();
                await tick();
                const netSlot = pickFirstSelectable(discardOrder);
                if (netSlot !== null) {
                    document.querySelector(`[data-testid="hand-card-${netSlot}"]`).click();
                    await tick();
                    const opts = Array.from(document.querySelectorAll('[data-testid="network-options"] > button'));
                    opts.forEach((btn, idx) => {
                        if (!btn.disabled) {
                            result.networks.push({
                                card_slot: netSlot,
                                btn_index: idx,
                                text: (btn.innerText || btn.textContent || '').trim(),
                            });
                        }
                    });
                }
                await cancelFlow();
            }

            // 3. Probe DEVELOP (check single & valid 2-industry combinations!)
            const devBtn = document.querySelector('[data-testid="action-develop"]');
            if (devBtn && !devBtn.disabled) {
                devBtn.click();
                await tick();
                const devSlot = pickFirstSelectable(discardOrder);
                if (devSlot !== null) {
                    document.querySelector(`[data-testid="hand-card-${devSlot}"]`).click();
                    await tick();
                    const indBtns = Array.from(
                        document.querySelectorAll('[data-testid="develop-options"] > button:not([data-testid="confirm-develop"])')
                    );
                    const enabledIndBtns = indBtns.filter(b => !b.disabled);
                    const industries = enabledIndBtns.map(b =>
                        (b.getAttribute('data-industry') || (b.innerText || b.textContent || '').trim().split(/\s+/)[0]).trim()
                    );
                    const maxCounts = {};
                    enabledIndBtns.forEach((b, idx) => {
                        const indName = industries[idx];
                        const mc = parseInt(b.getAttribute('data-max-count') || '1', 10);
                        maxCounts[indName] = (mc >= 2) ? 2 : 1;
                    });
                    // Test whether selecting 2 tiles is allowed (either 2 distinct industries OR 1 same-industry ×2)
                    let canDevelopTwo = false;
                    if (enabledIndBtns.length >= 2) {
                        enabledIndBtns[0].click();
                        await tick();
                        enabledIndBtns[1].click();
                        await tick();
                        const conf = document.querySelector('[data-testid="confirm-develop"]');
                        const confTxt = (conf?.innerText || '').trim();
                        if (conf && !conf.disabled && confTxt.includes('2')) {
                            canDevelopTwo = true;
                        }
                    } else if (enabledIndBtns.length === 1 && maxCounts[industries[0]] >= 2) {
                        enabledIndBtns[0].click();
                        await tick();
                        enabledIndBtns[0].click();
                        await tick();
                        const conf = document.querySelector('[data-testid="confirm-develop"]');
                        const confTxt = (conf?.innerText || '').trim();
                        if (conf && !conf.disabled && confTxt.includes('2')) {
                            canDevelopTwo = true;
                        }
                    }
                    result.develop = {
                        card_slot: devSlot,
                        industries,
                        max_counts: maxCounts,
                        can_develop_two: canDevelopTwo,
                    };
                }
                await cancelFlow();
            }

            // 4. Probe SELL (test maximal multi-building bundle + individual sales)
            const sellBtn = document.querySelector('[data-testid="action-sell"]');
            if (sellBtn && !sellBtn.disabled) {
                sellBtn.click();
                await tick();
                const sellSlot = pickFirstSelectable(discardOrder);
                if (sellSlot !== null) {
                    document.querySelector(`[data-testid="hand-card-${sellSlot}"]`).click();
                    await tick();
                    const sBtns = Array.from(
                        document.querySelectorAll('[data-testid="sell-options"] > button:not([data-testid="confirm-sell"])')
                    );
                    const items = sBtns.map((b, idx) => ({
                        btn_index: idx,
                        text: (b.innerText || b.textContent || '').trim(),
                    }));
                    // Greedily toggle all eligible buildings to see the exact maximal legal combo!
                    for (const b of sBtns) {
                        b.click();
                        await tick();
                    }
                    const selectedIndices = [];
                    sBtns.forEach((b, idx) => {
                        if ((b.className || '').includes('selected')) {
                            selectedIndices.push(idx);
                        }
                    });
                    result.sell = {
                        card_slot: sellSlot,
                        items,
                        greedy_combo_indices: selectedIndices,
                    };
                }
                await cancelFlow();
            }

            return result;
        }""",
        {"uniqueSlots": unique_card_slots, "discardOrder": discard_order},
    )

    actions: List[Dict[str, Any]] = []

    def card_lbl(slot: int) -> str:
        if 0 <= slot < len(hand_labels):
            return hand_labels[slot]
        if 0 <= slot < len(hand):
            return format_card_label(hand[slot])
        return f"card-{slot}"

    # 1. Deduplicated BUILD macro-actions (keep cheapest-discard-card option per (city, industry, level, cost, overbuild))
    seen_builds = set()
    for b in probed.get("builds") or []:
        txt = b["text"]
        m = _BUILD_OPT_RE.match(txt)
        if not m:
            continue
        city_disp = m.group("city").strip()
        city_id = normalize_city_id(city_disp)
        ind = m.group("industry").lower()
        lvl = int(m.group("level"))
        cost = int(m.group("cost"))
        is_over = bool(m.group("over"))
        dedup_key = (city_id, ind, lvl, cost, is_over)
        if dedup_key in seen_builds:
            continue
        seen_builds.add(dedup_key)
        slot = int(b["card_slot"])
        act = {
            "action_type": "build",
            "card_slot": slot,
            "card_label": card_lbl(slot),
            "btn_index": int(b["btn_index"]),
            "option_text": txt,
            "city": city_id,
            "city_display": city_disp,
            "industry": ind,
            "level": lvl,
            "cost": cost,
            "is_overbuild": is_over,
        }
        act["description"] = describe_macro_action(act)
        actions.append(act)

    # 2. NETWORK macro-actions (all single links + top-ranked Rail Era double-rail pairs)
    net_summary = compute_board_summary(state)
    net_city_dots = net_summary["city_link_dots"]
    net_player_nodes, _ = get_connected_component_and_network(state)
    single_net_actions: List[Dict[str, Any]] = []
    double_net_candidates: List[tuple[float, Dict[str, Any]]] = []

    for n in probed.get("networks") or []:
        txt = n["text"]
        m = _NET_OPT_RE.match(txt)
        raw_links = m.group("links") if m else txt.split("—")[0].strip()
        link_ids = [part.strip() for part in raw_links.split("+") if part.strip()]
        link_id = link_ids[0] if link_ids else raw_links
        cost = int(m.group("cost")) if m else (15 if len(link_ids) == 2 else 3)
        slot = int(n["card_slot"])
        act = {
            "action_type": "network",
            "card_slot": slot,
            "card_label": card_lbl(slot),
            "btn_index": int(n["btn_index"]),
            "option_text": txt,
            "link_id": link_id,
            "link_ids": link_ids,
            "is_double_rail": len(link_ids) == 2,
            "cost": cost,
        }
        act["description"] = describe_macro_action(act)
        if len(link_ids) >= 2:
            imm_vp = 0
            conn_merch = 0
            new_city = 0
            for lid in link_ids:
                eps = LINK_ID_TO_ENDPOINTS.get(lid, tuple(lid.split("-")))
                imm_vp += sum(net_city_dots.get(ep, 0) for ep in eps)
                if any(ep in MERCHANT_IDS for ep in eps):
                    conn_merch = 1
                if any(ep not in net_player_nodes and ep not in MERCHANT_IDS for ep in eps):
                    new_city += 1
            score = float(imm_vp) + 2.5 * conn_merch + 1.0 * new_city - 0.15 * float(cost)
            double_net_candidates.append((score, act))
        else:
            single_net_actions.append(act)

    actions.extend(single_net_actions)
    if double_net_candidates:
        double_net_candidates.sort(key=lambda item: item[0], reverse=True)
        actions.extend(act for _, act in double_net_candidates[:10])

    # 3. SELL macro-actions (maximal multi-sell combo + individual building options)
    sell_info = probed.get("sell")
    if sell_info and sell_info.get("items"):
        s_slot = int(sell_info["card_slot"])
        parsed_items = []
        for it in sell_info["items"]:
            m = _SELL_OPT_RE.match(it["text"])
            city_disp = m.group("city").strip() if m else it["text"]
            ind = m.group("industry").lower() if m else "cotton"
            lvl = int(m.group("level")) if m else 1
            beer_req = int(m.group("beer")) if m else 1
            parsed_items.append(
                {
                    "btn_index": int(it["btn_index"]),
                    "text": it["text"],
                    "city": normalize_city_id(city_disp),
                    "city_display": city_disp,
                    "industry": ind,
                    "level": lvl,
                    "beer_needed": beer_req,
                }
            )

        combo_idxs = sell_info.get("greedy_combo_indices") or []
        if combo_idxs:
            combo_sales = [it for it in parsed_items if it["btn_index"] in combo_idxs]
            act = {
                "action_type": "sell",
                "card_slot": s_slot,
                "card_label": card_lbl(s_slot),
                "sell_indices": combo_idxs,
                "sales": combo_sales,
                "cost": 0,
            }
            act["description"] = describe_macro_action(act)
            actions.append(act)

        if len(parsed_items) > 1:
            for it in parsed_items:
                if [it["btn_index"]] == combo_idxs:
                    continue
                act = {
                    "action_type": "sell",
                    "card_slot": s_slot,
                    "card_label": card_lbl(s_slot),
                    "sell_indices": [it["btn_index"]],
                    "sales": [it],
                    "cost": 0,
                }
                act["description"] = describe_macro_action(act)
                actions.append(act)

    # 4. DEVELOP macro-actions (prioritize 2-industry combos when legal, plus 1-industry options)
    dev_info = probed.get("develop")
    if dev_info and dev_info.get("industries"):
        d_slot = int(dev_info["card_slot"])
        inds = dev_info["industries"]
        # Estimate iron cost (0 if player has unflipped iron on board, else market price)
        free_iron = 0
        for slots in (state.get("board") or {}).values():
            for tile in slots or []:
                if isinstance(tile, dict) and tile.get("industry") == "iron" and int(tile.get("resources", 0)) > 0:
                    free_iron += int(tile["resources"])
        iron_cubes_mkt = int(state.get("iron_cubes", 8))
        money_now = int(state.get("money", 0))
        unit_iron_cost = 0 if free_iron >= 1 else iron_market_price(iron_cubes_mkt)
        two_iron_cost = (
            0
            if free_iron >= 2
            else (
                iron_market_price(iron_cubes_mkt)
                if free_iron == 1
                else iron_market_price(iron_cubes_mkt) + iron_market_price(max(0, iron_cubes_mkt - 1))
            )
        )

        ind_priority = {"cotton": 0, "goods": 1, "iron": 2, "brewery": 3, "coal": 4, "pottery": 5}
        sorted_inds = sorted(inds, key=lambda x: ind_priority.get(x.lower(), 9))
        max_counts = dev_info.get("max_counts") or {}
        pairs: List[List[str]] = []

        if dev_info.get("can_develop_two") and money_now >= two_iron_cost:
            # Same-industry double-develops (e.g. ['cotton', 'cotton'] when max_count >= 2)
            same_pairs = [[ind, ind] for ind in sorted_inds if int(max_counts.get(ind, 1)) >= 2]
            # Distinct-industry pairs
            diff_pairs: List[List[str]] = []
            for i in range(len(sorted_inds)):
                for j in range(i + 1, len(sorted_inds)):
                    diff_pairs.append([sorted_inds[i], sorted_inds[j]])
            pairs.extend(diff_pairs[:3])
            pairs.extend(same_pairs[:2])
            for pair in pairs:
                act = {
                    "action_type": "develop",
                    "card_slot": d_slot,
                    "card_label": card_lbl(d_slot),
                    "industries": pair,
                    "cost": two_iron_cost,
                }
                act["description"] = describe_macro_action(act)
                actions.append(act)
        if not pairs and money_now >= unit_iron_cost:
            for single_ind in sorted_inds[:2]:
                act = {
                    "action_type": "develop",
                    "card_slot": d_slot,
                    "card_label": card_lbl(d_slot),
                    "industries": [single_ind],
                    "cost": unit_iron_cost,
                }
                act["description"] = describe_macro_action(act)
                actions.append(act)

    # 5. LOAN, SCOUT, PASS (derived directly without destructive card clicks)
    avail = state.get("availability") or {}
    if avail.get("loan"):
        act = {
            "action_type": "loan",
            "card_slot": best_discard_slot,
            "card_label": card_lbl(best_discard_slot),
            "cost": 0,
        }
        act["description"] = describe_macro_action(act)
        actions.append(act)

    if avail.get("scout") and len(discard_order) >= 3:
        s3 = discard_order[:3]
        act = {
            "action_type": "scout",
            "card_slot": s3[0],
            "discard_slots": s3,
            "discard_labels": [card_lbl(x) for x in s3],
            "cost": 0,
        }
        act["description"] = describe_macro_action(act)
        actions.append(act)

    # Only include Pass if no productive action exists OR in very late-era edge cases as fallback
    if avail.get("pass") and (len(actions) == 0 or int(state.get("turn", 1)) >= 10):
        act = {
            "action_type": "pass",
            "card_slot": best_discard_slot,
            "card_label": card_lbl(best_discard_slot),
            "cost": 0,
        }
        act["description"] = describe_macro_action(act)
        actions.append(act)

    return actions


async def _resolve_element_center(page, selector: str, index: int = 0, text_match: Optional[str] = None) -> Optional[tuple[int, int]]:
    """Scroll target DOM element into view if needed and return its viewport (cx, cy)."""
    pt = await page.evaluate(
        """({ selector, index, textMatch }) => {
            const list = Array.from(document.querySelectorAll(selector));
            let el = null;
            if (textMatch) {
                const want = textMatch.trim().toLowerCase();
                el = list.find(e => (e.getAttribute('data-industry') || '').trim().toLowerCase() === want)
                  || list.find(e => (e.innerText || e.textContent || '').trim().toLowerCase() === want)
                  || list.find(e => (e.innerText || e.textContent || '').trim().toLowerCase().includes(want));
            }
            if (!el && index >= 0 && index < list.length) {
                el = list[index];
            }
            if (!el || el.disabled) return null;
            try { el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (e) {}
            const r = el.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) return null;
            return [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)];
        }""",
        {"selector": selector, "index": int(index), "textMatch": text_match},
    )
    if isinstance(pt, list) and len(pt) == 2:
        return int(pt[0]), int(pt[1])
    return None


async def execute_macro_action(
    page,
    action: Dict[str, Any],
    visual_delay_ms: int = 0,
    browser_mgr: Any = None,
    on_substep: Any = None,
) -> Dict[str, Any]:
    """
    Execute a Brass macro-action either atomically in <10ms (headless RL training)
    or step-by-step with real Playwright mouse movement (Live Arena playback).
    """
    if visual_delay_ms <= 0 and browser_mgr is None:
        ok = await page.evaluate(
            """async (act) => {
                const tick = () => new Promise(r => setTimeout(r, 0));
                const cancel = document.querySelector('[data-testid="flow-cancel"]');
                if (cancel) { cancel.click(); await tick(); }

                const t = act.action_type;
                const mainBtn = document.querySelector(`[data-testid="action-${t}"]`);
                if (!mainBtn || mainBtn.disabled) return false;
                mainBtn.click();
                await tick();

                const finishCheck = async () => {
                    await tick();
                    const errEl = document.querySelector('[data-testid="flow-error"]');
                    const cancelStillOpen = document.querySelector('[data-testid="flow-cancel"]');
                    if (errEl || cancelStillOpen) {
                        cancelStillOpen?.click();
                        await tick();
                        return false;
                    }
                    return true;
                };

                if (t === 'scout') {
                    for (const s of (act.discard_slots || [])) {
                        const chip = document.querySelector(`[data-testid="hand-card-${s}"]`);
                        if (!chip) return false;
                        chip.click();
                        await tick();
                    }
                    return await finishCheck();
                }

                const cardChip = document.querySelector(`[data-testid="hand-card-${act.card_slot}"]`);
                if (!cardChip) return false;
                cardChip.click();
                await tick();

                if (t === 'loan' || t === 'pass') {
                    return await finishCheck();
                }

                if (t === 'build') {
                    const opts = Array.from(document.querySelectorAll('[data-testid="build-options"] > button'));
                    let target = opts[act.btn_index];
                    if (!target || (act.option_text && (target.innerText || '').trim() !== act.option_text)) {
                        target = opts.find(b => (b.innerText || '').trim() === act.option_text) || target;
                    }
                    if (!target || target.disabled) {
                        document.querySelector('[data-testid="flow-cancel"]')?.click();
                        return false;
                    }
                    target.click();
                    return await finishCheck();
                }

                if (t === 'network') {
                    const opts = Array.from(document.querySelectorAll('[data-testid="network-options"] > button'));
                    let target = opts[act.btn_index];
                    if (!target || (act.option_text && (target.innerText || '').trim() !== act.option_text)) {
                        target = opts.find(b => (b.innerText || '').trim() === act.option_text) || target;
                    }
                    if (!target || target.disabled) {
                        document.querySelector('[data-testid="flow-cancel"]')?.click();
                        return false;
                    }
                    target.click();
                    return await finishCheck();
                }

                if (t === 'develop') {
                    for (const ind of (act.industries || [])) {
                        const want = ind.toLowerCase();
                        const indBtns = Array.from(
                            document.querySelectorAll('[data-testid="develop-options"] > button:not([data-testid="confirm-develop"])')
                        );
                        const btn = indBtns.find(b => {
                            const attr = (b.getAttribute('data-industry') || '').trim().toLowerCase();
                            if (attr) return attr === want;
                            const txt = (b.innerText || b.textContent || '').trim().toLowerCase();
                            return txt === want || txt.startsWith(want + ' ');
                        });
                        if (btn && !btn.disabled) {
                            btn.click();
                            await tick();
                        }
                    }
                    const conf = document.querySelector('[data-testid="confirm-develop"]');
                    if (!conf || conf.disabled) {
                        document.querySelector('[data-testid="flow-cancel"]')?.click();
                        return false;
                    }
                    conf.click();
                    return await finishCheck();
                }

                if (t === 'sell') {
                    const sBtns = Array.from(
                        document.querySelectorAll('[data-testid="sell-options"] > button:not([data-testid="confirm-sell"])')
                    );
                    for (const idx of (act.sell_indices || [])) {
                        if (sBtns[idx] && !sBtns[idx].disabled) {
                            sBtns[idx].click();
                            await tick();
                        }
                    }
                    const conf = document.querySelector('[data-testid="confirm-sell"]');
                    if (!conf || conf.disabled) {
                        document.querySelector('[data-testid="flow-cancel"]')?.click();
                        return false;
                    }
                    conf.click();
                    return await finishCheck();
                }
                return false;
            }""",
            action,
        )
        return {"ok": bool(ok), "last_xy": (640, 360), "clicks": 1}

    # Visual step-by-step mode for Live Arena!
    last_xy = (640, 360)
    clicks_done = 0

    async def visual_click(selector: str, index: int = 0, text_match: Optional[str] = None, label: str = "") -> bool:
        nonlocal last_xy, clicks_done
        xy = await _resolve_element_center(page, selector, index=index, text_match=text_match)
        if xy is None:
            return False
        last_xy = xy
        clicks_done += 1
        if browser_mgr is not None:
            await browser_mgr.execute_action(
                {"type": "click_coordinates", "x": xy[0], "y": xy[1], "button": "left", "clicks": 1},
                default_duration_ms=80,
            )
        else:
            await page.mouse.click(xy[0], xy[1])
        if visual_delay_ms > 0:
            await asyncio.sleep(visual_delay_ms / 1000.0)
        if on_substep is not None:
            await on_substep(xy, label)
        return True

    # Reset any stale open flow first
    cancel_xy = await _resolve_element_center(page, '[data-testid="flow-cancel"]')
    if cancel_xy is not None:
        await visual_click('[data-testid="flow-cancel"]', label="cancel")

    t = action.get("action_type", "pass")
    if not await visual_click(f'[data-testid="action-{t}"]', label=t.capitalize()):
        return {"ok": False, "last_xy": last_xy, "clicks": clicks_done}

    if t == "scout":
        for slot in action.get("discard_slots") or []:
            if not await visual_click(f'[data-testid="hand-card-{slot}"]', label=f"card-{slot}"):
                return {"ok": False, "last_xy": last_xy, "clicks": clicks_done}
        return {"ok": True, "last_xy": last_xy, "clicks": clicks_done}

    slot = int(action.get("card_slot", 0))
    if not await visual_click(f'[data-testid="hand-card-{slot}"]', label=f"card-{slot}"):
        return {"ok": False, "last_xy": last_xy, "clicks": clicks_done}

    if t in ("loan", "pass"):
        return {"ok": True, "last_xy": last_xy, "clicks": clicks_done}

    if t == "build":
        ok = await visual_click(
            '[data-testid="build-options"] > button',
            index=int(action.get("btn_index", 0)),
            text_match=action.get("option_text"),
            label=action.get("option_text", "build option"),
        )
        return {"ok": ok, "last_xy": last_xy, "clicks": clicks_done}

    if t == "network":
        ok = await visual_click(
            '[data-testid="network-options"] > button',
            index=int(action.get("btn_index", 0)),
            text_match=action.get("option_text"),
            label=action.get("option_text", "network option"),
        )
        return {"ok": ok, "last_xy": last_xy, "clicks": clicks_done}

    if t == "develop":
        for ind in action.get("industries") or []:
            await visual_click(
                '[data-testid="develop-options"] > button:not([data-testid="confirm-develop"])',
                text_match=ind,
                label=f"develop {ind}",
            )
        ok = await visual_click('[data-testid="confirm-develop"]', label="Confirm Develop")
        return {"ok": ok, "last_xy": last_xy, "clicks": clicks_done}

    if t == "sell":
        for idx in action.get("sell_indices") or []:
            await visual_click(
                '[data-testid="sell-options"] > button:not([data-testid="confirm-sell"])',
                index=int(idx),
                label=f"sell #{idx}",
            )
        ok = await visual_click('[data-testid="confirm-sell"]', label="Confirm Sell")
        return {"ok": ok, "last_xy": last_xy, "clicks": clicks_done}

    return {"ok": False, "last_xy": last_xy, "clicks": clicks_done}
