"""Run the manuscript S1--S3 trajectory experiment with the public data files."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from jmetal.core.quality_indicator import HyperVolume

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.experiment_common import ALGORITHM_NAMES, build_algorithm, dump_pickle
from src import ThreeDTrajectoryModel


CONSTRAINTS = {"max_range": 1400, "max_z": 300, "min_z": 100, "max_threat": 100}


def load_scenario(name: str):
    with (PACKAGE_ROOT / "data" / "scenarios" / f"{name}.pkl").open("rb") as handle:
        return pickle.load(handle)


def run_one(name, scenario, weights_path, tau_t):
    problem = ThreeDTrajectoryModel(
        vars=15, objs=2, scenario=scenario, constraints=CONSTRAINTS, change_tau=tau_t
    )
    population_size, max_iterations = 100, 100
    algorithm = build_algorithm(name, problem, population_size, population_size * max_iterations, weights_path)
    algorithm.run()
    archive, fronts = algorithm.result()
    reference_point = [CONSTRAINTS["max_range"] + 1, CONSTRAINTS["max_threat"] + 1]
    hvs = [HyperVolume(reference_point).compute(front) for front in fronts]
    return archive, fronts, hvs, algorithm.total_computing_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--scenarios", nargs="+", choices=["S1", "S2", "S3"], default=["S1", "S2", "S3"])
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "results" / "reproduced_trajectory")
    args = parser.parse_args()
    trials = args.trials
    for scenario_name in args.scenarios:
        scenario = load_scenario(scenario_name)
        for algorithm_name in args.algorithms:
            records = {"total_archive": [], "total_fronts": [], "HVs": [], "times": []}
            for trial in range(trials):
                random.seed(trial)
                np.random.seed(trial)
                archive, fronts, hvs, runtime = run_one(
                    algorithm_name, scenario, PACKAGE_ROOT / "data" / "weights", args.tau
                )
                records["total_archive"].append(archive)
                records["total_fronts"].append(fronts)
                records["HVs"].append(hvs)
                records["times"].append(runtime)
                print(f"{algorithm_name} {scenario_name} trial={trial}: {runtime:.3f}s")
            stem = f"{algorithm_name}_{scenario_name}"
            for suffix, values in records.items():
                dump_pickle(args.output / f"{stem}_{suffix}.pkl", values)


if __name__ == "__main__":
    main()
