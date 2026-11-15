"""
Backward-compatible module shim redirecting legacy `BrassPPOTrainer` imports
to the Neuroevolution / Genetic Algorithm optimizer (`BrassGATrainer`).
"""

from rl.ga_trainer import (
    BrassGATrainer,
    BrassGATrainer as BrassPPOTrainer,
    apply_genome,
    clone_genome,
    crossover_genomes,
    extract_genome,
    get_actor_parameter_names,
    mutate_genome,
    tournament_select,
)

__all__ = [
    "BrassGATrainer",
    "BrassPPOTrainer",
    "get_actor_parameter_names",
    "extract_genome",
    "apply_genome",
    "clone_genome",
    "crossover_genomes",
    "mutate_genome",
    "tournament_select",
]
