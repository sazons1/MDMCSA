"""Dynamic three-dimensional trajectory-planning model used in the manuscript."""

from __future__ import annotations

import math

import numpy as np
from jmetal.core.problem import FloatProblem
from scipy.spatial.distance import euclidean


class ThreeDTrajectoryModel(FloatProblem):
    """A standalone trajectory model with generation-based change detection.
    """

    def __init__(
        self,
        vars: int = 15,
        objs: int = 2,
        cons: int = 4,
        scenario: dict | list | None = None,
        constraints: dict | None = None,
        change_tau: int = 20,
    ):
        if scenario is None or constraints is None:
            raise ValueError("scenario and constraints are required")
        if vars % 3:
            raise ValueError("vars must contain equal yaw, pitch, and length blocks")
        if change_tau <= 0:
            raise ValueError("change_tau must be positive")

        super().__init__()
        self.vars = int(vars)
        self.objs = int(objs)
        self.cons = int(cons)
        self.constraints = constraints
        self.change_tau = int(change_tau)
        self.scenarios, self.scenario_name = self._normalise_scenario(scenario)
        self.time = 0
        self.environment_index = 0
        self.problem_modified = False
        self.code_length = self.vars // 3
        self.obj_directions = [self.MINIMIZE] * self.objs
        self.obj_labels = ["trajectory length", "threat cost"]

        motion_length = 50.0
        self.lower_bound = self.vars * [-math.pi / 4]
        self.upper_bound = self.vars * [math.pi / 4]
        self.lower_bound[: self.code_length] = self.code_length * [-math.pi / 3]
        self.upper_bound[: self.code_length] = self.code_length * [math.pi / 3]
        self.lower_bound[2 * self.code_length :] = self.code_length * [
            motion_length / self.constraints["max_range"]
        ]
        self.upper_bound[2 * self.code_length :] = self.code_length * [
            1.0 - motion_length * self.code_length / self.constraints["max_range"]
        ]
        self._set_scenario(0)

    @staticmethod
    def _normalise_scenario(scenario):
        if isinstance(scenario, dict):
            indices = sorted(key for key in scenario if isinstance(key, int))
            if not indices:
                raise ValueError("scenario dictionary must contain integer environment keys")
            return [scenario[index] for index in indices], scenario["name"]
        if not scenario:
            raise ValueError("scenario sequence must not be empty")
        return list(scenario), "Three-dimensional trajectory planning"

    def number_of_variables(self) -> int:
        return self.vars

    def number_of_objectives(self) -> int:
        return self.objs

    def number_of_constraints(self) -> int:
        return self.cons

    @staticmethod
    def _field(item, key):
        return item[key] if isinstance(item, dict) else getattr(item, key)

    def _set_scenario(self, environment_index: int) -> None:
        state = self.scenarios[environment_index]
        self.start_pos = np.asarray(state["start_pos"], dtype=float)
        self.end_pos = np.asarray(state["end_pos"], dtype=float)
        self._set_threats(state["threats"])

    def _set_threats(self, threats) -> None:
        self.threats = list(threats)
        size = len(self.threats)
        self._threat_centers_xy = np.empty((size, 2), dtype=float)
        self._threat_inner_radii = np.empty(size, dtype=float)
        self._threat_outer_radii = np.empty(size, dtype=float)
        self._threat_heights = np.empty(size, dtype=float)
        self._threat_inner_levels = np.empty(size, dtype=float)
        self._threat_outer_levels = np.empty(size, dtype=float)
        for index, threat in enumerate(self.threats):
            self._threat_centers_xy[index] = (
                self._field(threat, "center_x"),
                self._field(threat, "center_y"),
            )
            self._threat_inner_radii[index] = self._field(threat, "inner_radius")
            self._threat_outer_radii[index] = self._field(threat, "outer_radius")
            self._threat_heights[index] = self._field(threat, "height")
            self._threat_inner_levels[index] = self._field(threat, "inner_threat")
            self._threat_outer_levels[index] = self._field(threat, "outer_threat")

    def update(self, counter: int, solutions=None) -> None:
        """Apply the environment scheduled at the completed iteration count."""
        self.time = int(counter)
        next_index = min(self.time // self.change_tau, len(self.scenarios) - 1)
        if next_index != self.environment_index:
            self.environment_index = next_index
            self._set_scenario(next_index)
            self.problem_modified = True
        else:
            self.problem_modified = False

    def the_problem_has_changed(self) -> bool:
        """Return the model-change flag used by the manuscript algorithms."""
        return self.problem_modified

    def clear_changed(self) -> None:
        """Clear a pending change flag when an external caller requires it."""
        self.problem_modified = False

    def reinit_problem(self) -> None:
        self.time = 0
        self.environment_index = 0
        self.problem_modified = False
        self._set_scenario(0)

    def cal_threat(self, start, end) -> float:
        if not self.threats:
            return 0.0
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        segment_start = start[:2]
        vector = end[:2] - segment_start
        length_squared = float(np.dot(vector, vector))
        if length_squared == 0.0:
            closest = np.broadcast_to(segment_start, self._threat_centers_xy.shape)
        else:
            projection = ((self._threat_centers_xy - segment_start) @ vector) / length_squared
            projection = np.clip(projection, 0.0, 1.0)
            closest = segment_start + projection[:, None] * vector
        distances = np.linalg.norm(self._threat_centers_xy - closest, axis=1)
        active = (
            (distances < self._threat_outer_radii + 1.0)
            & (min(start[2], end[2]) < self._threat_heights + 1.0)
        )
        if not np.any(active):
            return 0.0
        outer_band = active & (distances > self._threat_inner_radii + 1.0)
        inner_band = active & ~outer_band
        value = np.sum(
            self._threat_outer_levels[outer_band]
            * (self._threat_outer_radii[outer_band] + 1.0 - distances[outer_band])
        )
        value += np.sum(
            self._threat_outer_levels[inner_band]
            * (self._threat_outer_radii[inner_band] - self._threat_inner_radii[inner_band])
            + self._threat_inner_levels[inner_band]
            * (self._threat_inner_radii[inner_band] + 1.0 - distances[inner_band])
        )
        return float(value)

    @staticmethod
    def calculate_yaw_pitch(first, second):
        dx, dy, dz = np.asarray(second) - np.asarray(first)
        yaw = math.atan2(dy, dx) % (2 * math.pi)
        horizontal = math.sqrt(dx * dx + dy * dy)
        pitch = math.atan2(dz, horizontal) if horizontal else (math.pi / 2 if dz > 0 else -math.pi / 2 if dz < 0 else 0.0)
        return yaw, pitch

    @staticmethod
    def _bound_violation(value, lower, upper) -> float:
        if value < lower:
            return value - lower
        if value > upper:
            return upper - value
        return 0.0

    def evaluate(self, solution):
        solution.objectives[:] = [0.0] * self.objs
        solution.constraints[:] = [0.0] * self.cons
        delta_yaws = solution.variables[: self.code_length]
        delta_pitches = solution.variables[self.code_length : 2 * self.code_length]
        length_factors = solution.variables[2 * self.code_length :]
        path = [self.start_pos.copy()]
        current_yaw, current_pitch = self.calculate_yaw_pitch(self.start_pos, self.end_pos)
        x, y, z = self.start_pos
        for delta_yaw, delta_pitch, length_factor in zip(delta_yaws, delta_pitches, length_factors):
            current_yaw += delta_yaw
            current_pitch += delta_pitch
            length = length_factor * self.constraints["max_range"]
            point = np.asarray(
                [
                    x + length * math.cos(current_pitch) * math.cos(current_yaw),
                    y + length * math.cos(current_pitch) * math.sin(current_yaw),
                    z + length * math.sin(current_pitch),
                ]
            )
            solution.objectives[0] += length
            solution.objectives[1] += self.cal_threat(path[-1], point)
            solution.constraints[2] += self._bound_violation(point[0], 0.0, 1000.0)
            solution.constraints[2] += self._bound_violation(point[1], 0.0, 1000.0)
            solution.constraints[2] += self._bound_violation(
                point[2], self.constraints["min_z"], self.constraints["max_z"]
            )
            path.append(point)
            x, y, z = point
        path.append(self.end_pos.copy())
        solution.objectives[0] += euclidean(path[-2], path[-1])
        solution.objectives[1] += self.cal_threat(path[-2], path[-1])
        yaw_end, pitch_end = self.calculate_yaw_pitch(path[-2], path[-1])
        if abs(yaw_end - current_yaw) > self.upper_bound[0]:
            solution.constraints[3] += self.upper_bound[0] - abs(yaw_end - current_yaw)
        if abs(pitch_end - current_pitch) > self.upper_bound[self.code_length]:
            solution.constraints[3] += self.upper_bound[self.code_length] - abs(pitch_end - current_pitch)
        if solution.objectives[0] > self.constraints["max_range"]:
            solution.constraints[0] += self.constraints["max_range"] - solution.objectives[0]
        if solution.objectives[1] > self.constraints["max_threat"]:
            solution.constraints[1] += self.constraints["max_threat"] - solution.objectives[1]
        solution.attributes["path"] = path
        return solution

    def name(self) -> str:
        return self.scenario_name
