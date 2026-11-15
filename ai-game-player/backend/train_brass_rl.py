"""
Standalone CLI runner for parallel Brass: Birmingham Genetic Algorithm (Neuroevolution) training.

Example usage:
    python train_brass_rl.py --workers 6 --generations 40 --pop-size 12 --games-per-genome 2 --sigma 0.05
"""

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rl.brass_bridge import BRASS_SOLO_URL
from rl.ga_trainer import BrassGATrainer
from storage.database import SessionDatabase


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evolve Brass: Birmingham PyTorch Model Weights & Biases via Genetic Algorithm"
    )
    p.add_argument("--workers", type=int, default=6, help="Parallel headless Chromium tabs (1..12)")
    p.add_argument(
        "--mode",
        type=str,
        default="self_play",
        choices=["self_play", "solo_mautoma"],
        help="self_play = 2P population vs itself; solo_mautoma = solo vs Mautoma",
    )
    p.add_argument("--generations", type=int, default=40, help="Number of evolutionary generations")
    p.add_argument("--pop-size", type=int, default=12, help="Population size per generation (4..64)")
    p.add_argument("--games-per-genome", type=int, default=2, help="Paired evaluation seeds per genome (1..6)")
    p.add_argument("--sigma", type=float, default=0.05, help="Gaussian mutation standard deviation")
    p.add_argument("--mutation-rate", type=float, default=0.20, help="Fraction of weights & biases mutated (0.01..1.0)")
    p.add_argument("--elites", type=int, default=2, help="Top elite genomes preserved unchanged each generation")
    p.add_argument("--fresh", action="store_true", help="Ignore saved checkpoints and evolve purely from heuristic anchor")
    p.add_argument("--url", type=str, default=BRASS_SOLO_URL, help="Brass Birmingham solo URL")
    return p


async def run_cli() -> None:
    args = build_arg_parser().parse_args()
    db = SessionDatabase()

    def on_progress(msg: dict) -> None:
        st = msg.get("status") or {}
        ep = msg.get("episode")
        if ep:
            res_tag = "WIN " if ep.get("won") else "LOSS"
            ckpt_note = f" [{ep['checkpoint_tag']}]" if ep.get("checkpoint_tag") else ""
            print(
                f"[Gen {st.get('current_generation', 0):>3d}/{st.get('target_generations', args.generations)}] "
                f"{res_tag} | Champion: {st.get('champion_avg_vp', 0):>5.1f} VP "
                f"(Last Game {ep['your_vp']:>3d} vs Automa {ep['automa_vp']:>3d}) | "
                f"PopMean: {st.get('generation_mean_vp', 0):>5.1f} VP | "
                f"Fitness: {st.get('champion_fitness', 0):>6.1f} | "
                f"sigma={st.get('mutation_sigma', 0):.4f} | {st.get('games_per_min', 0):>4.1f} g/min"
                f"{ckpt_note}",
                flush=True,
            )
        elif st.get("status_message") and ("complete" in st["status_message"] or "Seeded" in st["status_message"]):
            print(f"[GA] {st['status_message']}", flush=True)

    trainer = BrassGATrainer(db=db, on_update=on_progress)
    total_games = args.generations * args.pop_size * args.games_per_genome
    print(
        f"=== Brass: Birmingham Genetic Algorithm (Neuroevolution) Trainer ===\n"
        f"Device: {trainer.device} | Workers: {args.workers} | Generations: {args.generations} | "
        f"Pop: {args.pop_size} | Seeds/Genome: {args.games_per_genome} ({total_games} total games) | "
        f"Sigma: {args.sigma} | MutRate: {args.mutation_rate} | Elites: {args.elites} | Resume: {not args.fresh}",
        flush=True,
    )
    final_status = await trainer.train(
        workers=args.workers,
        training_mode=args.mode,
        generations=args.generations,
        population_size=args.pop_size,
        games_per_individual=args.games_per_genome,
        mutation_sigma=args.sigma,
        mutation_rate=args.mutation_rate,
        elite_count=args.elites,
        resume_checkpoint=not args.fresh,
        url=args.url,
    )
    print(
        f"\n=== Evolution Complete ===\n"
        f"Generations: {final_status.current_generation} ({final_status.completed_episodes} games) | "
        f"Champion Avg: {final_status.champion_avg_vp:.1f} VP | "
        f"Rolling Avg: {final_status.rolling_avg_vp:.1f} VP vs Automa {final_status.rolling_automa_vp:.1f} VP | "
        f"Win Rate: {final_status.rolling_win_rate_pct:.1f}% | All-Time Best Single Game: {final_status.best_vp} VP",
        flush=True,
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nInterrupted by user — latest evolved checkpoint preserved.")
