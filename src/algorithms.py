from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Any, List, Optional, TypeVar

import numpy
import numpy as np
from jmetal.algorithm.multiobjective import DynamicNSGAII, NSGAII, MOEAD
from jmetal.algorithm.multiobjective.moead import Permutation
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import DynamicProblem, FloatProblem, Problem
from jmetal.core.solution import FloatSolution, Solution
from jmetal.operator import BinaryTournamentSelection, DifferentialEvolutionCrossover, NaryRandomSolutionSelection
from jmetal.util.aggregation_function import AggregationFunction
from jmetal.util.archive import BoundedArchive, CrowdingDistanceArchive, NonDominatedSolutionsArchive
from jmetal.util.comparator import DominanceComparator, Comparator, ObjectiveComparator, SolutionAttributeComparator, MultiComparator
from jmetal.util.constraint_handling import is_feasible, overall_constraint_violation_degree
from jmetal.util.density_estimator import CrowdingDistance, KNearestNeighborDensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from sklearn.svm import SVR

from .mdmcsa import MOCSA

S = TypeVar("S")
R = TypeVar("R")

def get_non_dominated_solutions(solutions: List[Solution], comparator) -> List[Solution]:
    archive = NonDominatedSolutionsArchive(dominance_comparator=comparator)
    for solution in solutions:
        archive.add(solution)
    return archive.solution_list


class FarthestFirstSelection(Selection):
    def __init__(self, max_population_size: int):
        super(FarthestFirstSelection, self).__init__()
        self.max_population_size = max_population_size

    def execute(self, front: List[S]):
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        objectives_num = len(front[0].objectives)
        new_solution_list = []
        for i in range(objectives_num):
            min_sol = min(front, key=lambda x: x.objectives[i])
            new_solution_list.append(min_sol)
            front.remove(min_sol)

        while len(new_solution_list) < self.max_population_size:
            front_size = len(front)
            new_solution_size = len(new_solution_list)
            distance_matrix = np.zeros(shape=(front_size, new_solution_size))
            # Compute distance matrix
            for i in range(front_size):
                for j in range(new_solution_size):
                    distance_matrix[i, j] = euclidean(front[i].objectives, new_solution_list[j].objectives)
            # Gets the minimum distance of all solutions
            min_distance_matrix = np.min(distance_matrix, axis=1)
            # Gets the farthest nearest solution
            new_solution_list.append(front.pop(int(np.argmax(min_distance_matrix))))
            
        return new_solution_list, front

    def get_name(self) -> str:
        return "Farthest First Selection"

class EnvironmentSelection(Selection):

    def __init__(self, max_population_size: int):
        super(EnvironmentSelection, self).__init__()
        self.max_population_size = max_population_size

    def execute(self, front: List[S]):
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        leaders = []
        elitists = []

        for sol in front:
            if sol.attributes["assigned_fitness"]==0:
                leaders.append(copy(sol))

        if len(leaders) < self.max_population_size:
            front.sort(key=lambda x: x.attributes["assigned_fitness"])
            for i in range(self.max_population_size):
                elitists.append(copy(front[i]))
        elif len(leaders) == self.max_population_size:
            for solution in leaders:
                elitists.append(copy(solution))
        else:
            density_estimator = KNearestNeighborDensityEstimator()
            density_estimator.compute_density_estimator(leaders)
            density_estimator.sort(leaders)
            leaders = leaders[:self.max_population_size]
            for solution in leaders:
                elitists.append(copy(solution))

        return leaders, elitists

    def get_name(self) -> str:
        return "Environment Selection"


class DNSGAII_AB(DynamicNSGAII):
    def __init__(
            self,
            problem: DynamicProblem[S],
            population_size: int,
            offspring_population_size: int,
            mutation: Mutation,
            crossover: Crossover,
            selection: Selection = BinaryTournamentSelection(
                MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])
            ),
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            dominance_comparator: DominanceComparator = DominanceComparator(),
            detection_size: float = 0.1,
            diversity_mechanism: int = 0,  # default A, the re-initialization version
            diversity_propotion: float = 0.2,
    ):
        super(DNSGAII_AB, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator,
        )
        self.iter_front = []
        self.completed_iterations = 0
        self.detection_size = detection_size
        self.diversity_mechanism = diversity_mechanism
        self.diversity_propotion = diversity_propotion
        self.change_archive = {}


    def update_fronts(self):
        s_copy = [s for s in self.solutions]
        nd = get_non_dominated_solutions(s_copy, self.dominance_comparator)
        ps_objs = np.array([s.objectives for s in nd])
        self.iter_front.append(ps_objs)

    def change_detection(self):
        change_flag = False
        for solution in random.sample(self.solutions, int(self.population_size * self.detection_size)):
            temp_obj = solution.objectives[:]
            temp_con = solution.constraints[:]
            cmp_solution = copy(solution)
            solution_copy = self.evaluate([cmp_solution])[0]
            if solution_copy.objectives != temp_obj or solution_copy.constraints != temp_con:
                change_flag = True
                break
        return change_flag

    def change_response(self):
        diversity_indi = random.sample(self.solutions, int(self.population_size * self.diversity_propotion))
        if self.diversity_mechanism:
            for solution in diversity_indi:
                self.mutation_operator.execute(solution)
        else:
            for solution in diversity_indi:
                new_solution = self.population_generator.new(self.problem)
                solution.variables = new_solution.variables

    def init_progress(self) -> None:
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)
        self.update_fronts()

    def step(self):
        if self.change_detection() or self.problem.the_problem_has_changed():
            self.change_archive[self.completed_iterations] = [copy(solution) for solution in
                                                              get_non_dominated_solutions(self.solutions,self.dominance_comparator)]
            self.change_response()
            self.restart()
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)

    def update_progress(self):
        self.evaluations += self.offspring_population_size
        if self.evaluations % self.population_size == 0:
            self.completed_iterations += 1
            self.update_fronts()
            self.problem.update(self.completed_iterations, self.solutions)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion.is_met:
            self.change_archive[self.completed_iterations] = get_non_dominated_solutions(self.solutions,self.dominance_comparator)
            return self.termination_criterion.is_met

    def result(self) -> R:
        return self.change_archive, self.iter_front

    def get_name(self) -> str:
        if self.diversity_mechanism:
            return "DNSGAIIB"
        else:
            return "DNSGAII"



class SGEA(NSGAII):
    def __init__(
            self,
            problem: DynamicProblem,
            population_size: int,
            offspring_population_size: int,
            mutation: Mutation,
            crossover: Crossover,
            selection: Selection = BinaryTournamentSelection(comparator=SolutionAttributeComparator('assigned_fitness')),
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            dominance_comparator: DominanceComparator = DominanceComparator(),
            detection_size: float = 0.1,
    ):
        super(SGEA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator,
        )
        self.completed_iterations = 0
        self.detection_size = detection_size
        self.archive = []
        self.change_archive = {}
        self.centroid_tM1 = []
        self.elist_population = []
        self.environment_selection = EnvironmentSelection(population_size)
        self.iter_front = []

    def update_fronts(self):
        s_copy = [s for s in self.archive]
        nd = get_non_dominated_solutions(s_copy, self.dominance_comparator)
        ps_objs = np.array([s.objectives for s in nd])
        self.iter_front.append(ps_objs)

    def change_detection(self):
        change_flag = False
        for solution in random.sample(self.solutions, int(self.population_size * self.detection_size)):
            temp_obj = solution.objectives[:]
            temp_con = solution.constraints[:]
            cmp_solution = copy(solution)
            solution_copy = self.evaluate([cmp_solution])[0]
            if solution_copy.objectives != temp_obj or solution_copy.constraints != temp_con:
                change_flag = True
                break
        return change_flag

    def fitness_assignment(self, solutions):
        ranked_list = FastNonDominatedRanking(self.dominance_comparator).compute_ranking(solutions)
        rank_length = [len(rank) for rank in ranked_list]
        for rank in ranked_list:
            assigned_value = sum(rank_length[0:rank[0].attributes["dominance_ranking"]])
            for solution in rank:
                solution.attributes["assigned_fitness"] = assigned_value

    def change_response(self):
        # centroid of PS at time t
        variable_length = self.problem.number_of_variables()
        centroid_t = []
        for v in range(variable_length):
            centroid_t.append(sum([s.variables[v] for s in self.archive]) / len(self.archive))
        self.archive.clear()
        # select old solutions R
        old_solutions, self.solutions = FarthestFirstSelection(int(self.population_size * 0.5)).execute(self.solutions)
        # re-evaluate old solutions
        old_solutions = self.evaluate(old_solutions)
        if len(self.centroid_tM1) > 0:
            # 维度适配 centroid_tM1
            if len(self.centroid_tM1) < variable_length:
                self.centroid_tM1.extend([0.0] * (variable_length - len(self.centroid_tM1)))
            elif len(self.centroid_tM1) > variable_length:
                self.centroid_tM1 = self.centroid_tM1[:variable_length]
            # centroid of R
            centroid_r = []
            for v in range(variable_length):
                centroid_r.append(sum([s.variables[v] for s in old_solutions]) / len(old_solutions))
            # nondominated solutions in R fill A
            self.archive = get_non_dominated_solutions(old_solutions,self.dominance_comparator)
            # centroid of A
            centroid_a = []
            for v in range(variable_length):
                centroid_a.append(sum([s.variables[v] for s in self.archive]) / len(self.archive))
            # move rest solutions
            step_size = euclidean(centroid_t, self.centroid_tM1)
            moving_direction = []
            moving_dis = euclidean(centroid_a, centroid_r)
            for x in zip(centroid_a, centroid_r):
                if moving_dis:
                    moving_direction.append((x[0] - x[1]) / moving_dis)
                else:
                    moving_direction.append(0)
            noise_sigma = step_size / (2 * math.sqrt(variable_length))
            for solution in self.solutions:
                gaussian_noise = norm.rvs(0, noise_sigma, size=variable_length)
                for i in range(variable_length):
                    solution.variables[i] = solution.variables[i] + step_size * moving_direction[i] + gaussian_noise[i]
                    if solution.variables[i] < self.problem.lower_bound[i]:
                        solution.variables[i] = self.problem.lower_bound[i]
                    elif solution.variables[i] > self.problem.upper_bound[i]:
                        solution.variables[i] = self.problem.upper_bound[i]
        else:
            # reinitialize solutions
            for solution in self.solutions:
                new_solutions = self.population_generator.new(self.problem)
                solution.variables = new_solutions.variables
        self.solutions = self.evaluate(self.solutions)
        self.solutions.extend(old_solutions)
        self.archive.clear()
        self.fitness_assignment(self.solutions)
        self.archive = [copy(sol) for sol in self.solutions if sol.attributes["assigned_fitness"]==0]
        # update A
        self.centroid_tM1 = centroid_t
        # update centroid
        self.elist_population.clear()

    def APselection(self, population: List[S], archive: List[S]):
        mating_population = []

        if random.random() < 0.5:
            c=0
            while len(mating_population) < self.mating_pool_size:
                solution = self.selection_operator.execute(population)
                if c>4 or solution not in mating_population:
                    mating_population.append(solution)
                c+=1
        else:
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)
            c=0
            while len(mating_population) < self.mating_pool_size:
                random_solution = random.choice(archive)
                if c>4 or random_solution not in mating_population:
                    mating_population.append(random_solution)
                c+=1
        return mating_population

    def step(self):
        # steady state manner
        responded = False
        for i in range(self.population_size):
            if not responded:
                if self.change_detection() or self.problem.the_problem_has_changed():
                    self.change_archive[self.completed_iterations] = [copy(solution) for solution in self.archive]
                    self.change_response()
                    responded = True
            mating_population = self.APselection(self.solutions, self.archive)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)
            self.replacement(self.solutions, offspring_population)
        self.solutions.extend(self.elist_population)
        self.fitness_assignment(self.solutions)
        self.archive, self.solutions = self.environment_selection.execute(self.solutions)
        # P contains N elists
        self.elist_population = [copy(s) for s in self.solutions]

    def replacement(self, population: List[S], offspring_population: List[S]):
        """This method joins the current and offspring populations to produce the population of the next generation

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        offspring = offspring_population[0]
        offspring.attributes["assigned_fitness"] = 0
        for solution in population:
            if self.dominance_comparator.compare(offspring, solution) == -1:
                solution.attributes["assigned_fitness"] += 1
            elif self.dominance_comparator.compare(offspring, solution) == 1:
                offspring.attributes["assigned_fitness"] += 1
            else:
                pass
        worst_solution = max(population, key=lambda x: x.attributes["assigned_fitness"])
        worst_ind = population.index(worst_solution)
        if offspring.attributes["assigned_fitness"] <= worst_solution.attributes["assigned_fitness"]:
            population[worst_ind] = offspring
            if offspring.attributes["assigned_fitness"] == 0:
                remove_list = []
                for solution in self.archive:
                    if self.dominance_comparator.compare(offspring, solution) == -1:
                        remove_list.append(solution)
                for solution in remove_list:
                    self.archive.remove(solution)
                if len(self.archive) < self.population_size:
                    self.archive.append(copy(offspring))

    def init_progress(self) -> None:
        self.fitness_assignment(self.solutions)
        self.archive, self.solutions = self.environment_selection.execute(self.solutions)
        self.update_fronts()

    def update_progress(self):
        self.evaluations += self.population_size
        if self.evaluations % self.population_size == 0:
            self.completed_iterations += 1
            self.update_fronts()
            self.problem.update(self.completed_iterations, self.archive)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion.is_met:
            self.change_archive[self.completed_iterations] = self.archive
            return self.termination_criterion.is_met

    def result(self) -> R:
        return self.change_archive, self.iter_front

    def get_name(self) -> str:
        return "SGEA"



class MOEADSVR(MOEAD):
    def __init__(
            self,
            problem: Problem,
            population_size: int,
            mutation: Mutation,
            crossover: DifferentialEvolutionCrossover,
            aggregation_function: AggregationFunction,
            neighbourhood_selection_probability: float,
            neighbor_size: int,
            weight_files_path: str,
            solution_comparator: Comparator = store.default_comparator,
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            feasible_rules=False,
            q=4
    ):
        """
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            selection=NaryRandomSolutionSelection(2),
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            solution_comparator=solution_comparator
        )
        # no limitation of max replace
        # self.max_number_of_replaced_solutions = 2
        self.iter_front = []
        self.fitness_function = aggregation_function
        self.neighbourhood = WeightVectorNeighborhood(
            number_of_weight_vectors=population_size,
            neighborhood_size=neighbor_size,
            weight_vector_size=problem.number_of_objectives(),
            weights_path=weight_files_path,
        )
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.permutation = None
        self.current_subproblem = 0
        self.neighbor_type = None
        self.completed_iterations = 0
        self.detection_size = 0.1
        self.t_environment = 1
        self.change_archive = {}
        self.q = q
        self.saved_solution_variables = []
        self.SVR = SVR(gamma="auto", C=1000, epsilon=0.05)
        self.feasible_rules = feasible_rules
        if self.feasible_rules:
            self.feasible_flag = False
            self.best_cv = -np.inf

    def update_fronts(self):
        s_copy = [s for s in self.solutions]
        nd = get_non_dominated_solutions(s_copy, self.dominance_comparator)
        ps_objs = np.array([s.objectives for s in nd])
        self.iter_front.append(ps_objs)

    def change_detection(self):
        change_flag = False
        for solution in random.sample(self.solutions, int(
                self.population_size * self.detection_size)):
            temp_obj = solution.objectives[:]
            temp_con = solution.constraints[:]
            cmp_solution = copy(solution)
            solution_copy = self.evaluate([cmp_solution])[0]
            if solution_copy.objectives != temp_obj or solution_copy.constraints != temp_con:
                change_flag = True
                break
        return change_flag

    def feasible_update(self,solutions):
        # once the feasible flag is True, then only the feasible solutions can be used for updating the z*
        if self.feasible_flag:
            for solution in solutions:
                if is_feasible(solution):
                    self.fitness_function.update(solution.objectives)
        else:
            feasible_solutions = []
            infeasible_solutions = []
            for solution in solutions:
                if is_feasible(solution):
                    feasible_solutions.append(solution)
                else:
                    infeasible_solutions.append(solution)
            if len(feasible_solutions) != 0:
                # reinit the z* with infeasible solutions' information
                self.fitness_function.__init__(dimension=self.problem.number_of_objectives())
                for solution in feasible_solutions:
                    self.fitness_function.update(solution.objectives)
                self.feasible_flag = True
            else:
                # for solution in infeasible_solutions:
                #     self.fitness_function.update(solution.objectives)
                cvs = [overall_constraint_violation_degree(x) for x in infeasible_solutions]
                self.best_cv = np.max(cvs)
                best_infeasible = [s for s, cv in zip(solutions, cvs) if cv == self.best_cv]
                for solution in best_infeasible:
                    self.fitness_function.update(solution.objectives)

    def change_response(self):
        last_environment_solutions = [
            solution.variables[:] for solution in self.solutions]
        self.saved_solution_variables.append(last_environment_solutions)
        self.t_environment += 1
        
        if self.t_environment >= self.q + 2:
            num_vars = self.problem.number_of_variables()
            for i in range(self.population_size):
                for j in range(num_vars):
                    training_X = numpy.array([[self.saved_solution_variables[t + x][i][j] for t in range(self.q)]
                                                for x in range(self.t_environment - self.q - 1)])
                    training_Y = numpy.array([self.saved_solution_variables[x + self.q][i][j] for
                                                x in range(self.t_environment - self.q - 1)])
                    predict_X = numpy.array(
                        [self.saved_solution_variables[x + self.t_environment - self.q - 1][i][j] for
                            x in range(self.q)])
                    
                    if len(training_X) == 1:
                        training_X = training_X.reshape(1, -1)
                    
                    self.SVR = self.SVR.fit(training_X, training_Y)
                    new_variable = self.SVR.predict(predict_X.reshape(1, -1))
                    new_variable = float(new_variable)
                    
                    if new_variable < self.problem.lower_bound[j]:
                        new_variable = self.problem.lower_bound[j]
                    elif new_variable > self.problem.upper_bound[j]:
                        new_variable = self.problem.upper_bound[j]
                    self.solutions[i].variables[j] = new_variable
        self.solutions = self.evaluate(self.solutions)
        self.fitness_function.__init__(dimension=self.problem.number_of_objectives())
        if self.feasible_rules:
            self.feasible_flag = False
            self.best_cv=-np.inf
            self.feasible_update(self.solutions)
        else:
            for solution in self.solutions:
                self.fitness_function.update(solution.objectives)
        self.permutation = Permutation(self.population_size)

    def update_current_subproblem_neighborhood(self, new_solution, population):
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)

        if self.feasible_rules:
            for k in permuted_neighbors_indexes:
                current_sol = population[k]
                s1 = is_feasible(current_sol)
                s2 = is_feasible(new_solution)
                # 1. Comparison of feasibility rules
                if s2 and (not s1):
                    population[k] = deepcopy(new_solution)  # The new solution is feasible and can be directly replaced.
                    continue
                elif s1 and (not s2):
                    continue  # The current solution is feasible, keep it.
                elif (not s2) and (not s1):
                    # When neither is feasible, choose the solution with a better cv (solution with better cv dominates)
                    if overall_constraint_violation_degree(current_sol) < overall_constraint_violation_degree(new_solution):
                        population[k] = deepcopy(new_solution)
                    continue
                else:
                    f1 = self.fitness_function.compute(current_sol.objectives,
                                                       self.neighbourhood.weight_vectors[k])
                    f2 = self.fitness_function.compute(new_solution.objectives,
                                                       self.neighbourhood.weight_vectors[k])
                    if f2 < f1:
                        population[k] = deepcopy(new_solution)
        else:
            for i in range(len(permuted_neighbors_indexes)):
                k = permuted_neighbors_indexes[i]

                f1 = self.fitness_function.compute(population[k].objectives, self.neighbourhood.weight_vectors[k])
                f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

                if f2 < f1:
                    population[k] = deepcopy(new_solution)

        return population

    def init_progress(self) -> None:
        if self.feasible_rules:
            self.feasible_update(self.solutions)
        else:
            for solution in self.solutions:
                self.fitness_function.update(solution.objectives)

        self.permutation = Permutation(self.population_size)
        self.update_fronts()  # Record the initial population front.

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        new_solution = offspring_population[0]
        if not self.feasible_rules:
            self.fitness_function.update(new_solution.objectives)
        else:
            if self.feasible_flag:
                if is_feasible(new_solution):
                    self.fitness_function.update(new_solution.objectives)
            else:
                if is_feasible(new_solution):
                    self.fitness_function.__init__(dimension=self.problem.number_of_objectives())
                    self.fitness_function.update(new_solution.objectives)
                    self.feasible_flag=True
                elif overall_constraint_violation_degree(new_solution) >= self.best_cv:
                    self.best_cv=overall_constraint_violation_degree(new_solution)
                    self.fitness_function.update(new_solution.objectives)
        new_population = self.update_current_subproblem_neighborhood(new_solution, population)

        return new_population
    def step(self):
        if self.evaluations % self.population_size == 0:
            if self.change_detection() or self.problem.the_problem_has_changed():
                self.change_archive[self.completed_iterations] = [copy(s) for s in get_non_dominated_solutions(self.solutions,self.solution_comparator)]
                self.change_response()
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)

    def update_progress(self):
        self.evaluations += self.offspring_population_size
        if self.evaluations % self.population_size == 0:
            self.completed_iterations += 1
            self.update_fronts()
            self.problem.update(self.completed_iterations, self.solutions)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion.is_met:
            self.change_archive[self.completed_iterations] = get_non_dominated_solutions(self.solutions,self.solution_comparator)
            return self.termination_criterion.is_met


    def result(self) -> R:
        return self.change_archive, self.iter_front

    def get_name(self) -> str:
        return "MOEADSVR"



class DBCSAII(MOCSA):
    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            mutation: Mutation,
            leaders: Optional[BoundedArchive],
            max_iterations: int,
            swarm_comparator: Comparator = DominanceComparator(),
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            swarm_generator: Generator = store.default_generator,
            swarm_evaluator: Evaluator = store.default_evaluator,
    ):
        super(DBCSAII, self).__init__(problem=problem, swarm_size=swarm_size, mutation=mutation,
                                      termination_criterion=termination_criterion, swarm_generator
                                      =swarm_generator, swarm_evaluator=swarm_evaluator, swarm_comparator
                                      =swarm_comparator, leaders=leaders)
        self.pos_tau = [0 for _ in range(swarm_size)]
        self.pos_tau_1 = [0 for _ in range(swarm_size)]
        # np and nq
        # self.non_dom_counter_np = 0
        # self.dom_counter_nq = 0
        self.change_archive = {}
        self.completed_iterations = 0
        self.max_iterations = max_iterations
        self.detection_size = 0.1

    @staticmethod
    def beta_distribution(p, q, x0, x1, x):
        if x0 >= x1:
            raise ValueError
        else:
            if x < x0 or x > x1:
                beta = 0
            else:
                xc = (p * x1 + q * x0) / (p + q)
                beta = math.pow((x - x0) / (xc - x0), p) * math.pow((x1 - x) / (x1 - xc), q)
            return beta

    def change_detection(self):
        change_flag = False
        for solution in random.sample(self.solutions, int(self.swarm_size * self.detection_size)):
            cmp_solution = copy(solution)
            solution_copy = self.evaluate([cmp_solution])[0]
            if self.swarm_comparator.compare(solution, solution_copy) != 0:
                change_flag = True
                break
        return change_flag

    def change_response(self):
        # Aboud et al. (2022), Algorithm 1: after a detected change, compare
        # POS(t) under the new environment with the retained POS(t-1) snapshot.
        # ``pos_tau`` still carries the pre-change objectives at this point, so
        # evaluate copies before the dominance-count rule below is applied.
        current_population = self.evaluate([copy(s) for s in self.pos_tau])
        previous_population = [copy(s) for s in self.pos_tau_1]
        new_population = []
        replacement_indices = []
        for index, solution_p in enumerate(current_population):
            non_dom_p=dom_q=0
            for solution_q in previous_population:
                dom_cmp = self.swarm_comparator.compare(solution_p, solution_q)
                if dom_cmp == -1:
                    non_dom_p += 1
                elif dom_cmp == 1:
                    dom_q += 1
            if non_dom_p>=dom_q:
                new_population.append(solution_p)
            else:
                new_population.append(self.swarm_generator.new(self.problem))
                replacement_indices.append(index)
        if replacement_indices:
            replacements = self.evaluate([new_population[index] for index in replacement_indices])
            for index, replacement in zip(replacement_indices, replacements):
                new_population[index] = replacement
        self.solutions = new_population
        self.leaders.__init__(self.swarm_size)
        self.initialize_memory(self.solutions)
        self.initialize_leaders(self.solutions)

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        for solution in swarm:
            for i in range(self.problem.number_of_variables()):
                if random.random() <= (1 / self.problem.number_of_variables()):
                    r1 = random.random()
                    r2 = random.random()
                    if i % 3 == 0:
                        temp = solution.variables[i]
                        if r1 < 0.5:
                            solution.variables[i] = temp + (self.problem.upper_bound[i] - temp) * \
                                                    (r2 * (1 - self.completed_iterations / self.max_iterations))
                        elif r1 >= 0.5:
                            solution.variables[i] = temp - (temp + self.problem.lower_bound[i]) * \
                                                    (r2 * (1 - self.completed_iterations / self.max_iterations))
                    elif i % 3 == 1:
                        temp = solution.variables[i]
                        r = random.random()
                        cmp = temp + (r - 0.5 * (1 / self.problem.number_of_variables()))
                        solution.variables[i] = cmp
                    if solution.variables[i] < self.problem.lower_bound[i]:
                        solution.variables[i] = self.problem.lower_bound[i]
                    elif solution.variables[i] > self.problem.upper_bound[i]:
                        solution.variables[i] = self.problem.upper_bound[i]

    def update_position(self, swarm: List[FloatSolution]) -> None:
        follow = [math.floor(self.swarm_size * random.uniform(0, 1)) for _ in
                  range(self.swarm_size)]  # crows for chasing
        mean_fit = [sum([solution.objectives[m] for solution in self.solutions]) / len(self.solutions)
                    for m in range(self.problem.number_of_objectives())]
        for i in range(self.swarm_size):
            feasible_position = 1
            crow = copy(swarm[i])
            follow_crow = copy(swarm[follow[i]].attributes["memory"])
            if sum(crow.objectives) >= sum(mean_fit):
                for j in range(len(crow.variables)):
                    crow.variables[j] = crow.variables[j] + self.beta_distribution(5, 5, -1, 1, random.random())*(follow_crow.variables[j] - crow.variables[j])
                    if (crow.variables[j] > crow.upper_bound[j]) or (
                            crow.variables[j] < crow.lower_bound[j]):
                        feasible_position = 0  # if exceed search space, keep original position
                        break
            else:
                for j in range(len(crow.variables)):
                    """
                    orginal paper used beta-2 to generate variable, and the variable will only range in [0,1],
                    may have the problem of exceeding search space
                    """
                    # crow.variables[j] = self.beta_distribution(50, 50, crow.lower_bound[j], crow.upper_bound[j], crow.variables[j])
                    crow.variables[j] = crow.lower_bound[j] + self.beta_distribution(50, 50, -1, 1, random.random()) * (crow.upper_bound[j] - crow.lower_bound[j])
            if feasible_position:
                swarm[i] = copy(crow)

    def init_progress(self) -> None:
        self.initialize_memory(self.solutions)
        self.initialize_leaders(self.solutions)
        self.leaders.compute_density_estimator()
        self.update_fronts() 

    def step(self):
        if self.change_detection() or self.problem.the_problem_has_changed():
            self.change_archive[self.completed_iterations] = [copy(solution) for solution in self.leaders.solution_list]
            self.change_response()
        # population t and population t-1 in the paper
        self.pos_tau_1 = [copy(s) for s in self.solutions]
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_memory(self.solutions)
        self.update_leaders(self.solutions)
        self.pos_tau = [copy(s) for s in self.solutions]

    def update_progress(self):
        self.evaluations += self.swarm_size
        if self.evaluations % self.swarm_size == 0:
            self.completed_iterations += 1
            self.update_fronts()
            self.problem.update(self.completed_iterations, self.leaders.solution_list)
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion.is_met:
            self.change_archive[self.completed_iterations] = self.leaders.solution_list
            return self.termination_criterion.is_met

    def result(self) -> R:
        return self.change_archive, self.iter_front

    def get_name(self) -> str:
        return "DBCSAII"
