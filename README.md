# Brass: Birmingham — AfterQuery Harbor RL Environment Submission

> [!IMPORTANT]
> **For AfterQuery Judges:** The Harbor (`laude-institute/harbor`) task submission is located in the **[`brass-birmingham/`](brass-birmingham/)** directory.
> Pass `-p ./brass-birmingham` to `harbor run` (detailed rubric, failure-mode analysis, and reward architecture are in **[`brass-birmingham/README.md`](brass-birmingham/README.md)**).

---

## Quick Start for Judges (One-Command Evaluation)

```bash
git clone https://github.com/StenTheTank/Autonomous-AI-Game-Player.git
cd Autonomous-AI-Game-Player/brass-birmingham

# Run Do-Nothing, Pass-Spam, Stop-Early, and the 350K-param GA Reference Oracle:
./run_eval.sh

# Include 3x OpenCode frontier agent trials (e.g. Gemini or Claude):
GOOGLE_GENERATIVE_AI_API_KEY=<your_key> MODEL=google/gemini-3.8-flash ./run_eval.sh
# or:
harbor run -p ./brass-birmingham -a opencode -m <provider/model> -k 3
```

*(Windows PowerShell: `.\run_eval.ps1 -Model "google/gemini-3.8-flash" -Trials 3`)*

---

## Verified Harbor Results (`brass-birmingham/harbor_runs/`)

All 5 evaluation jobs below were executed in Docker via Harbor and verified from scratch by [`brass-birmingham/tests/verify_brass.py`](brass-birmingham/tests/verify_brass.py) (score in `/logs/verifier/reward.txt`):

| Harbor Job | Agent / Baseline | Moves | Match Score (`seed=202611`) | Harbor Reward |
| :--- | :--- | :---: | :---: | :---: |
| [`eval-nop`](brass-birmingham/harbor_runs/eval-nop/result.json) | Do-Nothing (`-a nop`) | `0 / 39` | `0 VP` | **`0.000000`** |
| [`eval-pass-spam`](brass-birmingham/harbor_runs/eval-pass-spam/result.json) | Pass-Spam (`PassSpamAgent`) | `39 / 39` | `0 VP vs 129 VP` | **`0.000000`** |
| [`eval-stop-early`](brass-birmingham/harbor_runs/eval-stop-early/result.json) | Stop-Early (`StopEarlyAgent`, halt after Canal Era) | `19 / 39` | `16 VP vs 55 VP` | **`0.003549`** |
| [`eval-opencode-frontier`](brass-birmingham/harbor_runs/eval-opencode-frontier/result.json) | OpenCode (`-a opencode -m google/gemini-3.8-flash`, `k=3`) | `39`, `39`, `20 / 39` | `106 vs 133`, `82 vs 126`, `41 vs 54` (`0%` wins) | **`0.390076`** *(mean)* |
| [`eval-ga-oracle`](brass-birmingham/harbor_runs/eval-ga-oracle/result.json) | **GA-Evolved Reference (`-a oracle`, `350K` params, `N=1`)** | **`39 / 39`** | **`135 VP vs 127 VP` (WIN)** | **`0.849667`** |

---

## Repository Structure

| Directory | Purpose |
| :--- | :--- |
| **[`brass-birmingham/`](brass-birmingham/)** | **Main Harbor Task Submission**: `instruction.md`, `task.toml`, `environment/Dockerfile` (Node 22 + `opencode-ai` pre-baked), `tests/verify_brass.py`, `solution/` (`N=1` GA reference policy), `harbor_runs/` (all 5 verified logs), and `post_training/` (PPO vs. 40-gen 2P Self-Play Neuroevolution trainer & benchmarks). |
| [`kdc-brass-birmingham/`](kdc-brass-birmingham/) | Extended offline *Brass: Birmingham* TypeScript simulator source (adding 2-Player hotseat mode, sequential Rail Era Double-Rail, and same-industry `×2` Develop parity) compiled into `brass-birmingham/environment/webapp/`. |
| [`ai-game-player/`](ai-game-player/) | Interactive FastAPI + React/Vite web UI & Playwright harness used during development for live LLM vs. RL arena duels and training visualization. |
