"""Self-contained source modules for the MDMCSA manuscript experiments."""

from .mdmcsa import MDMCSA
from .trajectory_problem import ThreeDTrajectoryModel
from .benchmark_problems import DMOP1, DMOP2, DMOP3, F5, F6, F7, DF12, DF13
from .algorithms import DNSGAII_AB, SGEA, MOEADSVR, DBCSAII

__all__ = [
    "MDMCSA", "ThreeDTrajectoryModel",
    "DNSGAII_AB", "SGEA", "MOEADSVR", "DBCSAII",
    "DMOP1", "DMOP2", "DMOP3", "F5", "F6", "F7", "DF12", "DF13",
]
