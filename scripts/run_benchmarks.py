"""Run the manuscript benchmark experiment and save pkl outputs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.experiment_common import ALGORITHM_NAMES, build_algorithm, dump_pickle
from src import DF12, DF13, DMOP1, DMOP2, DMOP3, F5, F6, F7


PROBLEMS = {"DMOP1": DMOP1, "DMOP2": DMOP2, "DMOP3": DMOP3, "F5": F5, "F6": F6, "F7": F7, "DF12": DF12, "DF13": DF13}


def build_reference_fronts(problem, max_iterations, reference_size):
    fronts = []
    for counter in range(max_iterations + 1):
        problem.update(counter)
        fronts.append(problem.get_reference_front(reference_size))
    problem.reinit_problem()
    return fronts


def run_one(name, problem, population_size, max_iterations, weights_path, reference_fronts, reference_point):
    algorithm = build_algorithm(name, problem, population_size, population_size * max_iterations, weights_path)
    algorithm.run()
    _, fronts = algorithm.result()
    if len(fronts) != len(reference_fronts):
        raise RuntimeError(f"{name} produced {len(fronts)} fronts; expected {len(reference_fronts)}")
    igds = [InvertedGenerationalDistance(reference).compute(front) for front, reference in zip(fronts, reference_fronts)]
    reference_hvs = [HyperVolume(reference_point).compute(front) for front in reference_fronts]
    hvds = [reference_hv - HyperVolume(reference_point).compute(front) for front, reference_hv in zip(fronts, reference_hvs)]
    return fronts, igds, hvds, algorithm.total_computing_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument("--tau", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "results" / "reproduced_benchmark")
    args = parser.parse_args()
    trials = args.trials
    for problem_name in args.problems:
        for tau_t in args.tau:
            n_changes, first_change_iter = 10, 0
            max_iterations = 3 * n_changes * tau_t + first_change_iter
            for algorithm_name in args.algorithms:
                records = {"total_fronts": [], "IGDs": [], "HVDs": [], "times": []}
                for trial in range(trials):
                    random.seed(trial)
                    np.random.seed(trial)
                    problem = PROBLEMS[problem_name](n_changes, tau_t, first_change_iter)
                    population_size = 100
                    reference_size = 900 if problem.number_of_objectives() == 3 else 500
                    reference_fronts = build_reference_fronts(problem, max_iterations, reference_size)
                    reference_point = problem.get_reference_HVpoint(0.5)
                    fronts, igds, hvds, runtime = run_one(
                        algorithm_name, problem, population_size, max_iterations,
                        PACKAGE_ROOT / "data" / "weights", reference_fronts, reference_point,
                    )
                    records["total_fronts"].append(fronts)
                    records["IGDs"].append(igds)
                    records["HVDs"].append(hvds)
                    records["times"].append(runtime)
                    print(f"{algorithm_name} {problem_name} tau={tau_t} trial={trial}: {runtime:.3f}s")
                stem = f"{algorithm_name}_{problem_name}_{tau_t}"
                for suffix, values in records.items():
                    dump_pickle(args.output / f"{stem}_{suffix}.pkl", values)


if __name__ == "__main__":
    main()
