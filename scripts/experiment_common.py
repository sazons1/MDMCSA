"""Shared factories for the main experiments."""

from __future__ import annotations

import pickle
from pathlib import Path

from jmetal.operator import DifferentialEvolutionCrossover, PolynomialMutation, SBXCrossover
from jmetal.util.aggregation_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import DominanceComparator, DominanceWithConstraintsComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

from src import DBCSAII, DNSGAII_AB, MDMCSA, MOEADSVR, SGEA


ALGORITHM_NAMES = ("MDMCSA", "DNSGAII", "DNSGAIIB", "SGEA", "MOEADSVR", "DBCSAII")


def comparator_for(problem):
    return (
        DominanceWithConstraintsComparator()
        if problem.number_of_constraints() > 0
        else DominanceComparator()
    )


def build_algorithm(name, problem, population_size, max_evaluations, weights_path):
    comparator = comparator_for(problem)
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    termination = StoppingByEvaluations(max_evaluations)
    if name == "MDMCSA":
        objectives = problem.number_of_objectives()
        if population_size % objectives:
            raise ValueError("population_size must be divisible by the number of objectives")
        return MDMCSA(
            problem=problem,
            swarm_size=population_size // objectives,
            mutation=mutation,
            leaders=CrowdingDistanceArchive(population_size, comparator),
            swarm_comparator=comparator,
            termination_criterion=termination,
            awareness_probability=0.05,
            flight_length_max=1.5,
            flight_length_min=1.0,
        )
    if name in {"DNSGAII", "DNSGAIIB"}:
        return DNSGAII_AB(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=mutation,
            crossover=SBXCrossover(probability=0.9, distribution_index=10),
            dominance_comparator=comparator,
            termination_criterion=termination,
            diversity_mechanism=0 if name == "DNSGAII" else 1,
            diversity_propotion=0.4 if name == "DNSGAIIB" else 0.2,
        )
    if name == "SGEA":
        return SGEA(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            dominance_comparator=comparator,
            termination_criterion=termination,
        )
    if name == "MOEADSVR":
        return MOEADSVR(
            problem=problem,
            population_size=population_size,
            mutation=mutation,
            crossover=DifferentialEvolutionCrossover(CR=0.5, F=0.5),
            aggregation_function=Tschebycheff(dimension=problem.number_of_objectives()),
            neighbor_size=20,
            neighbourhood_selection_probability=0.8,
            weight_files_path=str(weights_path),
            solution_comparator=comparator,
            termination_criterion=termination,
            feasible_rules=problem.number_of_constraints() > 0,
            q=2,
        )
    if name == "DBCSAII":
        return DBCSAII(
            problem=problem,
            swarm_size=population_size,
            mutation=mutation,
            leaders=CrowdingDistanceArchive(population_size, comparator),
            max_iterations=max_evaluations // population_size,
            swarm_comparator=comparator,
            termination_criterion=termination,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def dump_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
