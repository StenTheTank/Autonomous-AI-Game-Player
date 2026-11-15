# Brass: Birmingham — Harbor RL Environment & Neuroevolution Post-Training

A containerized **Harbor** (`laude-institute/harbor`) environment built around **Brass: Birmingham** (BoardGameGeek's #1 ranked economic strategy game). An agent plays all **20 rounds** (`39` sequential macro-decisions across the **Canal Era** and **Rail Era**) against the **Mautoma** solo opponent, managing coupled coal and iron commodity markets, spatial canal/rail networks, beer supply chains, tech-tree tile development, and nonlinear debt/income tracks.

---

## 1. One-Command Evaluation (`Craft`)

Run the complete judging protocol (`do-nothing`, `pass-spam`, `stop-early`, `single-pass reference oracle`, and optional `3x opencode` frontier trials) with one command:

```bash
cd brass-birmingham
./run_eval.sh
```

To include **3 repeated OpenCode frontier agent runs** in the same sweep, pass any LiteLLM/OpenCode `MODEL`:

```bash
GOOGLE_GENERATIVE_AI_API_KEY=<your_key> MODEL=google/gemini-3.8-flash ./run_eval.sh
# or run individual Harbor jobs directly:
harbor run -p ./brass-birmingham -a opencode -m google/gemini-3.8-flash --api-key-name GOOGLE_GENERATIVE_AI_API_KEY -k 3 -n 3
harbor run -p ./brass-birmingham -a oracle
harbor run -p ./brass-birmingham -a nop
```

*(Windows PowerShell equivalent: `.\run_eval.ps1 -Model "google/gemini-3.8-flash" -Trials 3`)*

---

## 2. What the Agent Has to Do

Inside `/workspace`, the agent interacts with a 100% offline **Brass: Birmingham** engine via the `brass` CLI:

- **Objective**: Play a complete 20-round game (`seed=202611` by default; configurable via `BRASS_EVAL_SEED`) until `game_over == true`, maximizing **Victory Points (VP)** and outscoring Mautoma.
- **CLI Observation & Action Space** (all information required for optimal play is exposed transparently):
  - `brass rules` — complete rulebook summary, industry tile table, resource transport rules, and exact reward formula.
  - `brass status [--json]` — current era/round, cash, income space & `£/round` payout, coal/iron market prices, merchant ports & bonus beer, player mat tech stacks, hand cards, placed buildings/links, and projected era VP.
  - `brass legal [--json]` — enumerates all currently legal macro-actions (`0 .. K-1`: `build`, `network`, `sell`, `develop`, `loan`, `scout`, `pass`) with exact `£` costs and immediate tactical consequences.
  - `brass play <action_id> [--json]` — executes legal move `<action_id>`, resolves Mautoma's counter-turn, and records a hash-chained move signature.

Agents running in **OpenCode** (`-a opencode`) can either play interactively turn-by-turn in bash or inspect `brass status --json` / `brass legal --json` and write a Python planner in `/workspace`.

---

## 3. The Empirical Gap (`Reference` vs. `Frontier Agents` vs. `Baselines`)

All runs below use **single-pass online execution** (`N=1`, zero lookahead rollouts or simulator cloning) and are verified from scratch by [`tests/verify_brass.py`](tests/verify_brass.py) (logs saved in [`harbor_runs/`](harbor_runs/) and [`post_training/post_training_benchmark.json`](post_training/post_training_benchmark.json)):

| Agent / Baseline | Moves Completed | Score (`seed=202611`) | Multi-Run / Held-Out Summary | Harbor Reward (`/logs/verifier/reward.txt`) |
| :--- | :---: | :---: | :---: | :---: |
| **Do-Nothing (`harbor run -a nop`)** | `0 / 39` | `0 VP` | `0.0 VP` | **`0.000000`** |
| **Pass-Spam (`PassSpamAgent`)** | `39 / 39` | `0 VP` vs `129 VP` | `0.0 VP` | **`0.000000`** |
| **Stop-Early (`StopEarlyAgent`, halt after Canal Era)** | `19 / 39` | `16 VP` vs `55 VP` (`Rail Rd 1`) | Canal-only (`-£1/rd` debt deducted) | **`0.003549`** *(capped `<= 0.08`)* |
| **Untrained `BrassActorCritic` (`350K` params, zero prior)** | `39 / 39` | `97 VP` vs `140 VP` | `24.3 VP` *(6 unseen seeds)* | `0.1482` *(mean)* |
| **Post-Trained PPO (`350K` params, 600 episodes)** | `39 / 39` | `96 VP` vs `151 VP` | `54.1 VP` *(21% `0-VP` collapses)* | `~0.44` *(mean)* |
| **OpenCode Frontier (`-a opencode -m google/gemini-3.8-flash`, `k=3`)** | `39`, `39`, `20 / 39` | `106 vs 133`, `82 vs 126`, `41 vs 54` | **`0%` wins** (`0.6347`, `0.5127`, `0.0229`) | **`0.3901`** *(mean; best: `0.634667`)* |
| **Direct API Frontier — `claude-sonnet-5-5` (3 full matches)** | `39 / 39` | `91`, `46`, `77 VP` | **`71.3 VP`** (`46–91 VP`, `0%` wins) | `~0.43` *(mean equiv.)* |
| **GA-Evolved Reference (`-a oracle`, `350K` params, `N=1`)** | **`39 / 39`** | **`135 VP` vs `127 VP` (WIN)** | **`138.0 VP`** *(6 unseen seeds)* | **`0.849667`** *(6-seed mean: `0.7997`)* |

In direct 2-player mirror-seed matches (`seed=202611`), our GA-evolved `350K`-parameter model also defeated `claude-sonnet-5-5` **`135 VP vs 91 VP`** and **`141 VP vs 46 VP`**.

---

## 4. Meaningful Difficulty: Where & Why Frontier Agents Fail

Every piece of state needed to win—card hand, city slot icons, merchant port beer/demand, coal/iron market prices, and exact action costs—is visible in `brass status` and `brass legal`. Frontier LLMs fall short because Brass couples **three distinct resource transport rules** with a **hard Round-10 era wipe**:

1. **Early Brewery Starvation via Cash-Conservation Bias**:
   - *Observed Failure*: In Canal Round 2 (`£9` cash), `claude-sonnet-5-5` rejected building `Walsall: brewery L1 (£7)`, reasoning that *"keeping £6 cash via a £3 canal link is a safer trade."*
   - *Consequence*: Entering mid-Canal Era with **zero personal beer barrels** blocks multi-tile `sell` actions (your own unflipped Breweries supply beer *anywhere* on the map, whereas opponent Breweries require link connectivity) and starves Rail Era **Double-Rail (`£15` + 2 Coal + 1 Brewery Beer)** turns, where Merchant beer is illegal.
2. **The Round-10 Canal Wipe & Stranded Consumer Tiles**:
   - *Observed Failure*: Frontier agents routinely build `Cotton` or `Manufactured Goods` tiles in Rounds 8–10 without verifying that both a connected matching Merchant port *and* a reachable Beer barrel remain before Round 10 ends.
   - *Consequence*: Unflipped tiles score **0 VP** and **0 link-dots (`•`)**, and every **Level-1 (L1)** tile and Canal link is permanently removed after Canal Round 10.
3. **Negative-Income Debt Spirals**:
   - *Observed Failure*: Taking multiple `+£30` loans (`-3` income tiers each) without flipping tiles to recover income pushed `claude-sonnet-5-5` to `income_space = 1` (`-£9/round`). By Rail Round 3, automatic upkeep drained its cash to `£1`, trapping it in forced `scout`/`pass` turns while its opponent compounded `+93` projected Rail VP.
4. **Over-Developing, Debt Drag & Rail-Link Starvation (`eval-opencode-frontier`: `google/gemini-3.8-flash`)**:
   - *Observed Failure (`harbor_runs/eval-opencode-frontier/`)*: Across both completed 39-turn OpenCode matches (`8UWFHdK`: `106 vs 133 VP`, `0.634667`; `6YNc7UP`: `82 vs 126 VP`, `0.512667`), `google/gemini-3.8-flash` spent **15–16 of its 39 turns (~40% of the entire game)** on non-board preparation (`9–10 develop` + `4 loan` + `1–2 scout`). Taking 4 loans (`-12` income spaces) dragged its final income down to `+£3/round` (`income_space = 16`), and even after flipping `8` buildings in `8UWFHdK`, it built **only `3` Rail links** (`5` in `6YNc7UP`) versus **`12` Rail links** and **`+£17/round`** income (`3` loans, `0` scouts, `16` builds, `18` network actions) in our GA Reference victory (`135 vs 127 VP`, `0.849667`). Meanwhile, the third OpenCode trial (`cuwDBFR`) played interactively turn-by-turn with long per-move reasoning and exhausted the `900s` budget at turn `20 / 39` (`41 vs 54 VP`, `0.022894`).

---

## 5. Reward Integrity & Anti-Exploit Architecture

[`environment/brass_engine/scoring.py`](environment/brass_engine/scoring.py) and [`tests/verify_brass.py`](tests/verify_brass.py) are engineered specifically to hold up under adversarial reward-hacking audits:

### Piecewise Scoring Formula (`[0.0, 1.0]`)
- **Zero-Engagement / Pass-Spam Gate (`valid_steps == 0` or `your_vp <= 0`)**: Returns exact **`0.000000`**. Passing 39 times to reach `game_over == true` earns zero credit.
- **Incomplete / Stop-Early Episodes (`game_over == false`)**:
  - First deducts unpaid negative-income debt across all remaining rounds: $\text{VP}_{\text{eff}} = \max(0,\; \text{VP} - R_{\text{rem}} \cdot |\min(0, \text{income\_£})|)$.
  - Caps total incomplete credit strictly at **`<= 0.08`**:
    $$R_{\text{incomplete}} = \text{progress} \cdot \left(0.03 \cdot \min\!\left(1, \frac{\text{VP}_{\text{eff}}}{35}\right) + 0.05 \cdot \min\!\left(1, \frac{\text{VP}_{\text{eff}}}{140}\right)\right) \le 0.08$$
  - **Blocks "Canal Loan-and-Quit"**: Taking 3 Canal loans (`+£90`, `-£9/round` income), scoring `~60 VP` at Canal end (Turn 19), and quitting before 10 rounds of Rail Era debt repayment deducts `10 × £9 = 90 VP` liability (`0.000000` reward).
- **Completed 20-Round Match (`game_over == true` and `your_vp > 0`)**:
  $$R_{\text{complete}} = \underbrace{0.10 \cdot \min\!\left(1, \frac{\text{VP}}{35}\right)}_{\text{Completion }(0.00-0.10)} + \underbrace{0.55 \cdot \min\!\left(1, \frac{\text{VP}}{150}\right)}_{\text{Empire VP }(0.00-0.55)} + \underbrace{R_{\text{vs}}(\text{VP} - \text{VP}_{\text{automa}})}_{\text{Vs. Mautoma }(0.00-0.35)}$$
  where **Win** ($\Delta > 0$) gives $0.22 + 0.13 \cdot \min(1, \Delta / 30)$ and **Loss/Tie** ($\Delta \le 0$) gives $0.20 \cdot \max(0, 1 - |\Delta| / 100) \cdot \min(1, \text{VP} / 50)$ (with a decay multiplier if `pass_count > 15`).

### Clean-Room Replay & Anti-Tamper Defenses
1. **Isolated Trusted Verifier (`/tests/trusted/webapp`)**: Even though OpenCode agents run with shell access inside `/workspace`, `/tests` is mounted **only after the agent container phase finishes**. `verify_brass.py` launches a fresh Chromium instance against `/tests/trusted/webapp` and replays every move signature against the authentic rules engine.
2. **SHA-256 Chain Hashing & Append-Only Daemon Ledger**: Every `brass play` step records a cumulative SHA-256 `chain_hash` (`GENESIS -> step 0 -> ... -> step t`) in both `/app/trajectory.jsonl` and an internal daemon ledger (`/var/tmp/.brass_ledger.jsonl`). Attempting to truncate `/app/trajectory.jsonl` mid-game for a "mulligan rewind" (peeking at card draws and restarting `seed=202611`) or hand-editing JSON signatures is caught and rejected by `verify_brass.py`.
3. **Single-Session Anti-Cloning Guard (`BRASS_ENFORCE_SINGLE_BOOT=1`) & Reference Isolation**: Any attempt to import `/opt/brass` internal helpers to spin up extra simulator sessions for offline seed-peeking rollouts is logged to `/var/tmp/.brass_engine_boots` and disqualified (`0.000000`) by `verify_brass.py`. All neural policy architecture and evolved weights (`model.py` + `brass_ppo_best.pt`) live strictly inside `solution/` (never mounted during agent runs), and [`solution/solve.sh`](solution/solve.sh) -> [`solution/run_ppo_oracle.py`](solution/run_ppo_oracle.py) executes a pure `N=1` online greedy pass (`39` `brass` calls in ~4s).

---

## 6. Post-Training a 350K-Parameter Model: Why Neuroevolution Beat PPO (`Tiebreaker`)

We built a compact **350,418-parameter (`1.4 MB`)** Candidate-Action neural policy (`BrassActorCritic` in [`solution/model.py`](solution/model.py): `288`-dim board/market/topology state encoder + `48`-dim legal macro-action scorer with bilinear interaction $[h_s, h_a, h_s \odot h_a]$) and compared two post-training paradigms at **~40 full 20-round games/minute** across 6 headless Chromium workers:

1. **Failure of Clipped PPO on Delayed Era Horizons ([`post_training/train_ppo.py`](post_training/train_ppo.py))**:
   - Across **600 episodes** (6 runs documented in [`post_training/post_training_benchmark.json`](post_training/post_training_benchmark.json)), PPO averaged only **`48.2–60.1 VP`** (`0%` win rate vs. Mautoma) and suffered **`21.0%` catastrophic `0-VP` collapses** (`126 / 600` games ending at `0 VP`).
   - *Root Cause*: In Brass, building a Coal Mine -> Brewery -> Cotton Mill -> Merchant Link -> `Sell` chain spans 5–9 turns before banking VP at Round 10. High-variance GAE returns destabilized the Value Head, causing policy updates that over-selected `loan` and `pass` into irreversible bankruptcy.
2. **Breakthrough via 2-Player Self-Play Neuroevolution ([`post_training/train_ga.py`](post_training/train_ga.py))**:
   - Pivoting to a **Genetic Algorithm** evolving all `317,393` Actor weights and biases directly—using **mirror-seeded 2-Player Self-Play** (`A vs B` and `B vs A` on identical card shuffles to cancel turn-order variance), neuron-row + BLX-$\alpha$ crossover, fan-in scaled Gaussian mutation, and **strict elitism**—eliminated `0-VP` collapses (`0 / 480` games) and drove steady generational improvement ([`post_training/ga_selfplay_40gen_log.csv`](post_training/ga_selfplay_40gen_log.csv)):
     - **Gen 1 Population Mean**: `56.8 VP` (Champion: `104.5 VP`)
     - **Gen 20 Population Mean**: `114.2 VP`
     - **Gen 39 Peak Population Mean**: **`131.1 VP`** (Champion: **`145.5 VP`**)
     - **Held-Out 6-Seed Solo Transfer (`N=1` greedy)**: Jumped from **`24.3 VP`** (random init) and **`90.0 VP`** (untrained Gen-0 prior) to **`138.0 VP`** (`33.3%` win rate vs. Mautoma, **`135 VP vs 127 VP` WIN** on `seed=202611`).

```bash
# Reproduce 2P Self-Play Neuroevolution post-training locally:
python post_training/train_ga.py --mode self_play --workers 6 --generations 40 --pop-size 12 \
    --games-per-ind 2 --out solution/brass_ppo_best.pt --log-csv post_training/ga_run.csv

# Evaluate single-pass (N=1) generalization across 6 unseen Solo Mautoma seeds:
python post_training/train_ga.py --eval-only --checkpoint solution/brass_ppo_best.pt
```

### Credits
- Offline Brass web simulator adapted and extended from [`KDC-Solo/brass-birmingham`](https://github.com/KDC-Solo/brass-birmingham) (adding full 2-Player human/AI hotseat mode, sequential Rail Era Double-Rail placement, and same-industry `×2` Develop parity), implementing *Brass: Birmingham* (Roxley Games) and the *Mautoma* solo system.
