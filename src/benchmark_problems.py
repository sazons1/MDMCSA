"""dynamic benchmark problems used by the experiments."""

import math
import random
from abc import ABC, abstractmethod
from math import cos, floor, pi, pow, sin, sqrt

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class DynamicProblem(FloatProblem, ABC):
    def __init__(self, nT=10, tau_T=10, first_change_iter=0):
        """
        DMOP base
        :param nT:severity of change
        :param tau_T:frequency of change
        """
        super(DynamicProblem, self).__init__()
        self.vars = None
        self.objs = None
        self.cons = None
        self.tau_T = tau_T
        self.nT = nT
        self.time = 0
        self.first_change_iter = first_change_iter
        self.problem_modified = False

    def number_of_variables(self) -> int:
        return self.vars

    def number_of_objectives(self) -> int:
        return self.objs

    def number_of_constraints(self) -> int:
        return self.cons

    def update(self, counter, solutions=None):
        last_time = self.time
        if self.first_change_iter:
            real_counter = counter-self.first_change_iter+self.tau_T
            if real_counter<0:
                self.time=0
            else:
                self.time = (1.0 / self.nT) * (real_counter // self.tau_T)
        else:
            self.time = (1.0 / self.nT) * (counter // self.tau_T)
        if self.time!=last_time:
            self.changed()
        else:
            self.clear_changed()

    def changed(self):
        self.problem_modified = True

    def the_problem_has_changed(self) -> bool:
        return self.problem_modified

    def clear_changed(self) -> None:
        self.problem_modified = False

    def reinit_problem(self)->None:
        self.time=0
        self.problem_modified=False

    @abstractmethod
    def evaluate(self, solution: FloatSolution):
        pass

class DMOP1(DynamicProblem):
    """Problem DMOP1.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
    """

    def __init__(self,  nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(DMOP1, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter)
        self.vars = numVariables
        self.objs = 2
        self.cons = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = self.vars * [0.0]
        self.upper_bound = self.vars * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 1.0 + (9/(self.vars-1))*sum([v**2 for v in solution.variables[1:]])
        return g

    def __eval_h(self, f: float, g: float) -> float:
        H = 1.25+0.75*sin(0.5 * pi * self.time)

        return 1.0 - pow(f / g, H)

    def get_reference_front(self, num):
        H = 1.25+0.75*sin(0.5 * pi * self.time)
        step = 1/(num-1)
        f1s = [step*i for i in range(num)]
        f2s = []
        for f1 in f1s:
            f2 = 1-pow(f1, H)
            f2s.append(f2)
        points = list(zip(f1s, f2s))
        return np.array(points)

    def get_reference_HVpoint(self, plus_value):
        maxf1 = 1.0
        maxf2 = 1.0
        return np.array([maxf1+plus_value, maxf2+plus_value])

    def name(self):
        return "DMOP1"

class DMOP2(DMOP1):
    """Problem DMOP2.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
    """

    def __init__(self, nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(DMOP2, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter, numVariables=numVariables)


    def __eval_g(self, solution: FloatSolution):
        gT = abs(sin(0.5 * pi * self.time))
        g = 1.0 + sum([pow(v - gT, 2) for v in solution.variables[1:]])
        return g

    def __eval_h(self, f: float, g: float) -> float:
        H = 1.25+0.75*sin(0.5 * pi * self.time)
        return 1.0 - pow(f / g, H)

    def name(self):
        return "DMOP2"

class DMOP3(DMOP1):
    """Problem DMOP3.

        .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
        """

    def __init__(self, nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(DMOP3, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter, numVariables=numVariables)
        self.r = random.randint(0, self.vars - 1)

    def changed(self):
        self.r = random.randint(0, self.vars - 1)
        self.problem_modified = True

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[self.r], g)

        solution.objectives[0] = solution.variables[self.r]
        solution.objectives[1] = h * g

        return solution

    def __eval_g(self, solution: FloatSolution):
        gT = abs(sin(0.5 * pi * self.time))
        g = 1.0 + sum([pow(v - gT, 2) for v in solution.variables[:self.r]+solution.variables[self.r+1:]])
        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f/g)

    def get_reference_front(self, num):
        step = 1/(num-1)
        f1s = [step*i for i in range(num)]
        f2s = []
        for f1 in f1s:
            f2 = 1-sqrt(f1)
            f2s.append(f2)
        points = list(zip(f1s, f2s))
        return np.array(points)

    def name(self):
        return "DMOP3"

class F5(DynamicProblem):
    """Problem F5.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
    """

    def __init__(self, nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(F5, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter)
        self.vars = numVariables
        self.objs = 2
        self.cons = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = self.vars * [0.0]
        self.upper_bound = self.vars * [5.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        H = self.__eval_h()
        a = self.__eval_a()
        b = self.__eval_b()

        solution.objectives[0] = pow(abs(solution.variables[0] - a), H) + sum(
            [pow(self.__eval_yi(a, b, H, i, solution.variables[i], solution.variables[0]), 2) for i in
             range(0, self.vars, 2)])

        solution.objectives[1] = pow(abs(solution.variables[0] - a - 1), H) + sum(
            [pow(self.__eval_yi(a, b, H, i, solution.variables[i], solution.variables[0]), 2) for i in
             range(1, self.vars, 2)])

        return solution

    def __eval_h(self) -> float:
        H = 0.75*sin(pi*self.time)+1.25
        return H

    def __eval_a(self):
        return 2*cos(pi*self.time)+2

    def __eval_b(self):
        return 2*sin(pi*self.time)+2

    def __eval_yi(self, a, b, H, i, xi, x1):
        return xi-b-1+pow(abs(x1-a), H+i/self.vars)

    def get_reference_front(self, num):
        H = self.__eval_h()
        step = 1 / (num - 1)
        f1s = [step * i for i in range(num)]
        f2s = []
        for f1 in f1s:
            f2 = pow(1 - pow(f1, 1/H),H)
            f2s.append(f2)
        points = list(zip(f1s, f2s))
        return np.array(points)

    def get_reference_HVpoint(self, plus_value):
        maxf1 = 1.0
        maxf2 = 1.0
        return np.array([maxf1+plus_value, maxf2+plus_value])

    def name(self):
        return "F5"

class F6(F5):
    """Problem F6.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
    """

    def __init__(self, nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(F6, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter, numVariables=numVariables)

    def __eval_a(self):
        return 2*cos(1.5*pi*self.time)*sin(0.5*pi*self.time)+2

    def __eval_b(self):
        return 2*cos(1.5*pi*self.time)*cos(0.5*pi*self.time)+2

    def name(self):
        return "F6"

class F7(F5):
    """Problem F7.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 10.
    """

    def __init__(self, nT=10, tau_T=10, first_change_iter=0, numVariables: int = 10):
        """:param numVariables: Number of decision variables of the problem."""
        super(F7, self).__init__(nT=nT, tau_T=tau_T, first_change_iter=first_change_iter, numVariables=numVariables)

    def __eval_a(self):
        return 1.7 * (1-sin(pi * self.time)) * sin(pi * self.time) + 3.4

    def __eval_b(self):
        return 1.4 * (1-sin(pi * self.time)) * cos(pi * self.time) + 2.1

    def name(self):
        return "F7"

def _filter_nondominated_points(points, atol=1.0e-12):
    """Return the nondominated rows of a minimization point set.

    This helper constructs the analytical reference fronts of DF12 and DF13.
    Reference-front generation is outside the optimization loop, so the simple
    quadratic check is preferable to adding another dependency.
    """

    points = np.unique(np.asarray(points, dtype=float), axis=0)
    keep = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        no_worse = np.all(points <= point + atol, axis=1)
        strictly_better = np.any(points < point - atol, axis=1)
        if np.any(no_worse & strictly_better):
            keep[i] = False

    return points[keep]

class DF12(DynamicProblem):
    """CEC 2018 DF12: tri-objective dynamic multimodal/disconnected-PF benchmark."""

    def __init__(self, nT=10, tau_T=10, first_change_iter=0,
                 numVariables: int = 10):
        super(DF12, self).__init__(
            nT=nT,
            tau_T=tau_T,
            first_change_iter=first_change_iter
        )

        if numVariables < 3:
            raise ValueError("DF12 requires at least three decision variables")

        self.vars = numVariables
        self.objs = 3
        self.cons = 0
        self.obj_directions = [self.MINIMIZE] * self.objs
        self.obj_labels = ["f1", "f2", "f3"]

        self.lower_bound = [0.0, 0.0] + [-1.0] * (self.vars - 2)
        self.upper_bound = [1.0, 1.0] + [1.0] * (self.vars - 2)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        k = 10.0 * sin(pi * self.time)
        r = 1.0

        linkage = sin(self.time * x1)

        tmp1 = [xi - linkage for xi in solution.variables[2:]]

        tmp2_x1 = abs(sin(floor(k * (2.0 * x1 - r)) * pi / 2.0))
        tmp2_x2 = abs(sin(floor(k * (2.0 * x2 - r)) * pi / 2.0))

        g = (1.0 + sum(value * value for value in tmp1) + tmp2_x1 * tmp2_x2)

        solution.objectives[0] = (g * cos(0.5 * pi * x2) * cos(0.5 * pi * x1))
        solution.objectives[1] = (g * sin(0.5 * pi * x2) * cos(0.5 * pi * x1))
        solution.objectives[2] = (g * sin(0.5 * pi * x1))

        return solution

    def get_reference_front(self, num):
        if num < 2:
            raise ValueError("num must be at least 2")

        x1, x2 = np.meshgrid(np.linspace(0.0, 1.0, num), np.linspace(0.0, 1.0, num), indexing="xy")

        k = 10.0 * sin(pi * self.time)
        r = 1.0

        tmp2 = np.abs(np.sin(np.floor(k * (2.0 * x1 - r)) * pi / 2.0) * np.sin(np.floor(k * (2.0 * x2 - r)) * pi / 2.0 ))

        g = 1.0 + tmp2

        f1 = g * np.cos(0.5 * pi * x2) * np.cos(0.5 * pi * x1)
        f2 = g * np.sin(0.5 * pi * x2) * np.cos(0.5 * pi * x1)
        f3 = g * np.sin(0.5 * pi * x1)

        candidates = np.column_stack(
            (f1.ravel(), f2.ravel(), f3.ravel())
        )

        return _filter_nondominated_points(candidates)

    def get_reference_HVpoint(self, plus_value):
        return np.array([2.0 + plus_value] * self.objs)

    def name(self):
        return "DF12"

class DF13(DynamicProblem):
    """CEC 2018 DF13: tri-objective problem with dynamic PF connectivity."""

    def __init__(self, nT=10, tau_T=10, first_change_iter=0,
                 numVariables: int = 10):
        super(DF13, self).__init__(
            nT=nT, tau_T=tau_T, first_change_iter=first_change_iter
        )
        if numVariables < 3:
            raise ValueError("DF13 requires at least three decision variables")

        self.vars = numVariables
        self.objs = 3
        self.cons = 0
        self.obj_directions = [self.MINIMIZE] * self.objs
        self.obj_labels = ["f1", "f2", "f3"]

        self.lower_bound = [0.0, 0.0] + [-1.0] * (self.vars - 2)
        self.upper_bound = [1.0, 1.0] + [1.0] * (self.vars - 2)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1, x2 = solution.variables[0], solution.variables[1]
        G = sin(0.5 * pi * self.time)
        p = floor(6.0 * G)
        g = 1.0 + sum(pow(xi - G, 2) for xi in solution.variables[2:])

        sin_x1 = sin(0.5 * pi * x1)
        sin_x2 = sin(0.5 * pi * x2)
        solution.objectives[0] = g * pow(cos(0.5 * pi * x1), 2)
        solution.objectives[1] = g * pow(cos(0.5 * pi * x2), 2)
        solution.objectives[2] = (
            g * pow(sin_x1, 2)
            + sin_x1 * pow(cos(p * pi * x1), 2)
            + pow(sin_x2, 2)
            + sin_x2 * pow(cos(p * pi * x2), 2)
        )
        return solution

    def get_reference_front(self, num):
        if num < 2:
            raise ValueError("num must be at least 2")

        x1, x2 = np.meshgrid(
            np.linspace(0.0, 1.0, num),
            np.linspace(0.0, 1.0, num),
            indexing="xy",
        )
        G = sin(0.5 * pi * self.time)
        p = floor(6.0 * G)
        sin_x1 = np.sin(0.5 * pi * x1)
        sin_x2 = np.sin(0.5 * pi * x2)

        f1 = np.power(np.cos(0.5 * pi * x1), 2)
        f2 = np.power(np.cos(0.5 * pi * x2), 2)
        f3 = (
            np.power(sin_x1, 2)
            + sin_x1 * np.power(np.cos(p * pi * x1), 2)
            + np.power(sin_x2, 2)
            + sin_x2 * np.power(np.cos(p * pi * x2), 2)
        )
        candidates = np.column_stack((f1.ravel(), f2.ravel(), f3.ravel()))
        return _filter_nondominated_points(candidates)

    def get_reference_HVpoint(self, plus_value):
        return np.array(
            [1.0 + plus_value, 1.0 + plus_value, 4.0 + plus_value]
        )

    def name(self):
        return "DF13"
