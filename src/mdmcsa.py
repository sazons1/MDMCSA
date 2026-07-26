from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from copy import copy
from itertools import chain
from typing import List, Optional, TypeVar

import numpy
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import FloatProblem, Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.archive import BoundedArchive, NonDominatedSolutionsArchive
from jmetal.util.comparator import Comparator, DominanceComparator, ObjectiveComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class _SplitSelection:
    """Preserve the manuscript's objective-wise binary split selection."""

    def __init__(self, maximum_size, objective_id):
        self.maximum_size = int(maximum_size)
        self.objective_id = int(objective_id)

    @staticmethod
    def _binary_points(count):
        points = []
        level = 1
        while len(points) < count:
            denominator = 2 ** level
            for index in range(2 ** (level - 1)):
                points.append((2 * index + 1) / denominator)
                if len(points) == count:
                    return points
            level += 1
        return points

    def execute(self, solutions):
        if not solutions:
            raise ValueError("Split selection requires at least one solution")
        pool = list(solutions)
        pool.sort(key=lambda solution: solution.objectives[self.objective_id])
        if len(pool) <= self.maximum_size:
            return [copy(solution) for solution in pool]
        lower = pool[0].objectives[self.objective_id]
        upper = pool[-1].objectives[self.objective_id]
        selected = []
        for point in [0.0, 1.0] + self._binary_points(self.maximum_size - 2):
            target = lower + point * (upper - lower)
            index = min(
                range(len(pool)),
                key=lambda item: abs(pool[item].objectives[self.objective_id] - target),
            )
            selected.append(copy(pool.pop(index)))
        return selected

class CrowSearchAlgorithm(Algorithm[FloatSolution, List[FloatSolution]], ABC):
    def __init__(self, problem: Problem[S], swarm_size: int):
        super(CrowSearchAlgorithm, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_memory(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_memory(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def init_progress(self) -> None:
        self.initialize_memory(self.solutions)

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_memory(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f"{self.get_name()}.{self.problem.name()}"

class SingleCSA(CrowSearchAlgorithm):
    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            mutation: Mutation,
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            swarm_generator: Generator = store.default_generator,
            swarm_evaluator: Evaluator = store.default_evaluator,
            swarm_comparator: Comparator = ObjectiveComparator(0),
            awareness_probability: float = 0.05,
            flight_length: float = 2
    ):
        super(SingleCSA, self).__init__(problem=problem, swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.mutation_operator = mutation
        self.swarm_comparator = swarm_comparator

        self.awareness_probability = awareness_probability
        self.flight_length = flight_length

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_memory(self, swarm: List[FloatSolution]) -> None:
        for crow in swarm:
            crow.attributes["memory"] = copy(crow)

    def update_position(self, swarm: List[FloatSolution]) -> None:
        follow = [math.floor(self.swarm_size * random.uniform(0, 1)) for _ in
                  range(self.swarm_size)]  # crows for chasing
        for i in range(self.swarm_size):
            feasible_position = 1
            crow = copy(swarm[i])
            follow_crow = copy(swarm[follow[i]].attributes["memory"])
            if random.random() > self.awareness_probability:
                for j in range(len(crow.variables)):
                    crow.variables[j] = crow.variables[j] + self.flight_length * random.random() * (
                                follow_crow.variables[j] - crow.variables[j])
                    if (crow.variables[j] > crow.upper_bound[j]) or (
                            crow.variables[j] < crow.lower_bound[j]):
                        feasible_position = 0  # if exceed search space, keep original position
                        break
            else:
                for j in range(len(crow.variables)):
                    crow.variables[j] = crow.lower_bound[j] + random.random() * (
                                crow.upper_bound[j] - crow.lower_bound[j])
            if feasible_position:
                swarm[i] = copy(crow)

    def update_memory(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.swarm_comparator.compare(swarm[i], swarm[i].attributes["memory"])
            if flag != 1:
                swarm[i].attributes["memory"] = copy(swarm[i])

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def result(self) -> R:
        self.solutions.sort(key=lambda x: x.objectives[0])
        return self.solutions[0]

    def get_name(self) -> str:
        return "SingleCSA"

class MOCSA(SingleCSA):
    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            mutation: Mutation,
            leaders: Optional[BoundedArchive],
            swarm_comparator: Comparator = DominanceComparator(),
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            swarm_generator: Generator = store.default_generator,
            swarm_evaluator: Evaluator = store.default_evaluator,
            awareness_probability=0.2,
            flight_length=2,
    ):
        super(MOCSA, self).__init__(problem=problem, swarm_size=swarm_size,mutation=mutation,
                                    termination_criterion=termination_criterion,swarm_generator
                                    =swarm_generator,swarm_evaluator=swarm_evaluator,swarm_comparator
                                    =swarm_comparator,awareness_probability=awareness_probability,
                                    flight_length=flight_length)
        self.leaders = leaders
        self.iter_front=[]

    def update_fronts(self):
        # ``leaders`` is the algorithm's archive.
        archived = self.leaders.solution_list
        ps_objs = np.array([s.objectives for s in archived])
        self.iter_front.append(ps_objs)

    def initialize_leaders(self, swarm: List[FloatSolution]) -> None:
        for crow in swarm:
            self.leaders.add(copy(crow))

    def update_leaders(self, swarm: List[FloatSolution]) -> None:
        for crow in swarm:
            self.leaders.add(copy(crow))

    def init_progress(self) -> None:
        self.initialize_memory(self.solutions)
        self.initialize_leaders(self.solutions)
        self.update_fronts()

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.update_leaders(self.solutions)
        self.update_fronts()
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def result(self) -> List[FloatSolution]:
        return self.leaders.solution_list

    def get_name(self) -> str:
        return "MOCSA"

class MDMCSA(Algorithm):
    def __init__(self, problem, swarm_size, mutation, leaders,
                 swarm_comparator=DominanceComparator(),
                 termination_criterion=store.default_termination_criteria,
                 swarm_generator=store.default_generator,
                 swarm_evaluator=store.default_evaluator,
                 awareness_probability=0.1, flight_length_max=2.0,
                 flight_length_min=1.0, detection_size=0.1):
        Algorithm.__init__(self)
        self.problem = problem
        self.swarm_size = int(swarm_size)
        self.total_swarm_size = self.swarm_size * self.problem.number_of_objectives()
        if int(leaders.maximum_size) != self.total_swarm_size:
            raise ValueError(
                "leaders.maximum_size must equal swarm_size * number_of_objectives"
            )
        self.mutation_operator = mutation
        self.leaders = leaders
        self.swarm_comparator = swarm_comparator
        self.termination_criterion = termination_criterion
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.observable.register(termination_criterion)
        self.awareness_probability = awareness_probability
        self.flight_length_max = flight_length_max
        self.flight_length_min = flight_length_min
        self.detection_size = detection_size
        self.completed_iterations = 0
        self.evaluations = 0
        self.change_archive = {}
        self.iter_front = []
        self.centroid_tM1 = []
        self.interval = max(
            1,
            int(termination_criterion.max_evaluations / self.total_swarm_size / 10),
        )
        self.elite_mutate_swarm = []
        self.selector = BinaryTournamentSelection(CrowdingDistance.get_comparator())

    def update_leaders(self, swarm_list):
        for swarm in swarm_list:
            for solution in swarm:
                self.leaders.add(copy(solution))

    def update_fronts(self):
        """Record the shared archive exactly as maintained by MDMCSA."""
        archived = self.leaders.solution_list
        objectives = (
            np.array([solution.objectives for solution in archived])
            if archived
            else np.zeros((0, self.problem.number_of_objectives()))
        )
        self.iter_front.append(objectives)

    def change_detection(self):
        """Perform one sampling-based change check for the current generation."""
        sample_size = max(1, int(self.swarm_size * self.detection_size))
        for swarm in self.solutions:
            for solution in random.sample(swarm, min(sample_size, len(swarm))):
                previous_objectives = list(solution.objectives)
                previous_constraints = list(solution.constraints)
                self.evaluate([[solution]])
                if solution.objectives != previous_objectives or solution.constraints != previous_constraints:
                    return True
        return False

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def create_initial_solutions(self) -> List[List[S]]:
            co_swarm = []
            for i in range(self.problem.number_of_objectives()):  # co-evolution swarm
                co_swarm.append([self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)])
            return co_swarm

    def evaluate(self, swarm_list: List[List[S]]):
            for swarm in swarm_list:
                self.swarm_evaluator.evaluate(swarm, self.problem)
            return swarm_list

    def initialize_memory(self, swarm_list: List[List[S]]) -> None:
            for swarm in swarm_list:
                for crow in swarm:
                    crow.attributes["memory"] = copy(crow)
                    crow.attributes["memory_time"] = 1

    def initialize_leaders(self, swarm_list: List[List[S]]) -> None:
            for objId, swarm in enumerate(swarm_list):
                for solution in swarm:
                    self.leaders.add(copy(solution))

    @staticmethod
    def roulette(cumsum):
            r = random.uniform(0, 1)
            index = numpy.searchsorted(cumsum, r * cumsum[-1])
            return index

    @staticmethod
    def fuch(i):
            xi=random.uniform(-1,1)
            while xi==0:
                xi=random.uniform(-1,1)
            num_list=[]
            for x in range(i):
                xi=math.cos(1/(xi**2))
                num_list.append(xi)
            return num_list

    def update_memory(self, swarm_list: List[List[S]]) -> None:
            for objId, swarm in enumerate(swarm_list):
                comparator = ObjectiveComparator(objId)
                for solution in swarm:
                    flag = self.swarm_comparator.compare(solution, solution.attributes["memory"])
                    if flag == -1:
                        solution.attributes["memory"] = copy(solution)
                        solution.attributes["memory_time"] = 1
                    elif flag == 0:
                        if solution.attributes["memory_time"] == 1:
                            self.leaders.add(solution)
                        # Pareto check
                        if comparator.compare(solution, solution.attributes["memory"])==-1:
                            solution.attributes["memory"] = copy(solution)
                            solution.attributes["memory_time"] = 1
                    else:
                        solution.attributes["memory_time"] += 1

    def stopping_condition_is_met(self) -> bool:
            if self.termination_criterion.is_met:
                self.change_archive[self.completed_iterations] = self.leaders.solution_list
                return self.termination_criterion.is_met

    def merge_sort(self, swarm_list: List[List[S]]):
            total = list(chain(*swarm_list))
            total.extend(self.elite_mutate_swarm)
            total.extend(self.leaders.solution_list)
            for i in range(self.problem.number_of_objectives()):
                # Select sequentially from the shared pool. SplitSelection removes
                # selected candidates; reinsert one copy afterwards so information
                # can still transfer between objective swarms without independently
                # cloning the entire candidate pool for every swarm.
                swarm_list[i] = _SplitSelection(self.swarm_size, i).execute(total)
                total.extend(copy(solution) for solution in swarm_list[i])

    def init_progress(self) -> None:
            self.initialize_memory(self.solutions)
            self.initialize_leaders(self.solutions)
            self.update_elite_mutate_swarm()
            self.merge_sort(self.solutions)
            self.update_fronts()  # Record the initial population front.
            observable_data = self.observable_data()
            self.observable.notify_all(**observable_data)

    def update_position(self, swarm_list: List[List[S]]) -> None:
            for objId, swarm in enumerate(swarm_list):
                ap_list = self.fuch(self.swarm_size)
                flight_length = self.flight_length_min + self.flight_length_max*random.random() *math.cos(
                    (self.completed_iterations % self.interval / self.interval) * (math.pi / 2))
                for i in range(self.swarm_size):
                    crow = swarm[i]
                    if random.random()<0.5:
                        rou_list = [1 / (c.objectives[objId] + 0.001) for c in self.leaders.solution_list]
                        cumsum = numpy.cumsum(rou_list)
                        follow_crow = self.leaders.solution_list[self.roulette(cumsum)]
                    else:
                        follow_crow = self.selector.execute(self.leaders.solution_list)
                    follow_crow_m = follow_crow.attributes["memory"]
                    if random.random() > (ap_list[i]/2+1)*self.awareness_probability:
                        for j in range(len(crow.variables)):
                            crow.variables[j] = crow.variables[j] + flight_length * random.random()*(follow_crow_m.variables[j] - crow.variables[j])
                            if crow.variables[j] > crow.upper_bound[j]:
                                crow.variables[j] = crow.upper_bound[j]  # if exceed search space, confine it
                            elif crow.variables[j] < crow.lower_bound[j]:
                                crow.variables[j] = crow.lower_bound[j]
                    else:
                        for j in range(len(crow.variables)):
                            crow.variables[j] = crow.lower_bound[j] + random.random() * (
                                    crow.upper_bound[j] - crow.lower_bound[j])

    def perturbation(self, swarm_list: List[List[S]]) -> None:
            for objId, swarm in enumerate(swarm_list):
                for solution in swarm[:int(self.swarm_size)]:
                    if solution.attributes["memory_time"]==1:
                        continue
                    self.mutation_operator.execute(solution)
                    self.evaluate([[solution]])
                    comparator = ObjectiveComparator(objId)
                    flag = self.swarm_comparator.compare(solution, solution.attributes["memory"])
                    if flag == -1:
                        solution.attributes["memory"] = copy(solution)
                        solution.attributes["memory_time"] = 1
                    elif flag == 0:
                        if solution.attributes["memory_time"] == 1:
                            self.leaders.add(solution)
                        # Pareto check
                        if comparator.compare(solution, solution.attributes["memory"]) == -1:
                            solution.attributes["memory"] = copy(solution)
                            solution.attributes["memory_time"] = 1
                    else:
                        solution.attributes["memory_time"] += 1

    def update_elite_mutate_swarm(self):
            self.elite_mutate_swarm = [copy(s) for s in self.leaders.solution_list]
            for elite in self.elite_mutate_swarm:
                i = random.randint(0,self.problem.number_of_variables()-1)
                elite.variables[i] += (elite.upper_bound[i] - elite.lower_bound[i]) * random.gauss(0,1)
                if elite.variables[i] > elite.upper_bound[i]:
                    elite.variables[i] = elite.upper_bound[i]  # if exceed search space, confine it
                elif elite.variables[i] < elite.lower_bound[i]:
                    elite.variables[i] = elite.lower_bound[i]
                self.evaluate([[elite]])
                if (self.swarm_comparator.compare(elite, elite.attributes["memory"]))<1:
                    self.leaders.add(elite)

    def change_response(self):
            variable_length = self.problem.number_of_variables()
            # Reset the historical centroid because cross-dimensional prediction is undefined.
            if len(self.centroid_tM1) > 0 and len(self.centroid_tM1) != variable_length:
                self.centroid_tM1 = []
            # centroid of PS at time t
            centroid_t = []
            for v in range(variable_length):
                centroid_t.append(
                    sum([s.variables[v] for s in self.leaders.solution_list]) / self.leaders.size())
            for objId, swarm in enumerate(self.solutions):
                # reuse memories
                memory_swarm = []
                for solution in swarm:
                    if solution.attributes["memory_time"]>1:
                        memory_swarm.append(solution.attributes["memory"])
                swarm.extend(memory_swarm)
                # filter out the old solutions R in current sub-swarm
                old_solutions = _SplitSelection(int(self.swarm_size * 0.5), objId).execute(swarm)
                # re-evaluate old solutions for obj_id
                old_solutions = self.evaluate([old_solutions])[0]
                # random split other solutions into P and R
                random.shuffle(swarm)
                swarm_p = swarm[0:int(len(swarm) / 2)]
                swarm_r = swarm[int(len(swarm) / 2):]
                # if not first environmental change
                if len(self.centroid_tM1)>0:
                    # centroid of R_obj
                    centroid_r = []
                    for v in range(variable_length):
                        centroid_r.append(sum([s.variables[v] for s in old_solutions]) / len(old_solutions))
                    archive = NonDominatedSolutionsArchive(
                        dominance_comparator=self.swarm_comparator
                    )
                    for solution in old_solutions:
                        archive.add(solution)
                    nondom_r = archive.solution_list
                    # centroid of A_obj
                    centroid_a = []
                    for v in range(variable_length):
                        centroid_a.append(sum([s.variables[v] for s in nondom_r]) / len(nondom_r))
                    # step size and moving_direction
                    step_size = euclidean(centroid_t, self.centroid_tM1)
                    mov_dis = euclidean(centroid_a, centroid_r)
                    moving_direction = []
                    for x in zip(centroid_a, centroid_r):
                        if mov_dis:
                            moving_direction.append((x[0] - x[1]) / mov_dis)
                        else:
                            moving_direction.append(0)
                    gaussian_noise = norm.rvs(0, step_size / (2 * math.sqrt(variable_length)), size=variable_length)
                    # move the non-selected solutions
                    for solution in swarm_p:
                        for i in range(variable_length):
                            solution.variables[i] = solution.variables[i] + step_size * moving_direction[i] + \
                                                    gaussian_noise[i]
                            if solution.variables[i] < solution.lower_bound[i]:
                                solution.variables[i] = solution.lower_bound[i]
                            elif solution.variables[i] > solution.upper_bound[i]:
                                solution.variables[i] = solution.upper_bound[i]
                else:
                    swarm_p = [self.swarm_generator.new(self.problem) for _ in range(len(swarm_p))]
                swarm_r = [self.swarm_generator.new(self.problem) for _ in range(len(swarm_r))]
                self.solutions[objId]= _SplitSelection(self.swarm_size, objId).execute(swarm_p+swarm_r+old_solutions)
            # update centroid
            self.centroid_tM1 = centroid_t
            # re-init archive
            self.evaluate(self.solutions)
            self.leaders.__init__(self.total_swarm_size)
            self.initialize_memory(self.solutions)
            self.initialize_leaders(self.solutions)
            self.update_elite_mutate_swarm()
            self.merge_sort(self.solutions)

    def step(self):
        if self.change_detection() or self.problem.the_problem_has_changed():
            self.change_archive[self.completed_iterations] = [
                copy(solution) for solution in self.leaders.solution_list
            ]
            self.change_response()
        self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_memory(self.solutions)
        self.perturbation(self.solutions)
        self.update_leaders(self.solutions)
        self.update_elite_mutate_swarm()
        self.merge_sort(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.total_swarm_size
        self.completed_iterations += 1
        self.update_fronts()
        self.problem.update(self.completed_iterations, self.leaders.solution_list)
        self.observable.notify_all(**self.observable_data())

    def result(self) -> R:
        return self.change_archive, self.iter_front

    def get_name(self) -> str:
        return "MDMCSA"
