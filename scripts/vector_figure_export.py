"""Export vector figures from the public MDMCSA reproduction results.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import QuadMesh
from matplotlib.patches import Circle
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import art3d
from scipy.interpolate import CubicSpline

# Embed TrueType fonts in the PDF rather than Type-3 glyph paths.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

# Publication-oriented vector export settings.
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 16
TITLE_SIZE = 24
EXPORT_DPI = 600
PLANAR_BACKGROUND_DPI = 300


REPRODUCIBILITY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RESULT_ROOT = REPRODUCIBILITY_ROOT / "results" / "benchmark"
TRAJECTORY_RESULT_ROOT = REPRODUCIBILITY_ROOT / "results" / "trajectory"
TRAJECTORY_SCENARIO_ROOT = REPRODUCIBILITY_ROOT / "data" / "scenarios"
TRAJECTORY_TERRAIN_FILE = TRAJECTORY_SCENARIO_ROOT / "terrain_seed42_p0.7.pkl"
FIGURE_ROOT = REPRODUCIBILITY_ROOT / "results" / "figures"

TRIALS = 30
BENCHMARK_PROBLEMS = ("DMOP1", "F5")
TAUS = (5, 10, 20)
TRAJECTORY_SCENARIOS = ("S1", "S2", "S3")
TRAJECTORY_ITERATIONS = 100
TRAJECTORY_TAU = 20
TRAJECTORY_SNAPSHOTS = (20, 40, 60, 80, 100)
TRAJECTORY_MHV_INDICES = (19, 39, 59, 79, 99)

ALGORITHMS = (
    ("MDMCSA", "MDMCSA", "black", "-"),
    ("DNSGAII", "DNSGA-II-A", "#228b22", "-."),
    ("DNSGAIIB", "DNSGA-II-B", "#e58500", (0, (3, 1, 1, 1, 1, 1))),
    ("SGEA", "SGEA", "#c62828", "--"),
    ("MOEADSVR", "MOEA/D-SVR", "#7b1fa2", ":"),
    ("DBCSAII", "DBCSA-II", "#0a1fbe", (0, (5, 1))),
)


def _install_pickle_compatibility_module() -> None:
    """Provide the legacy threat class without importing its Tk plotting module."""
    if "Draw_maps" in sys.modules:
        return
    module = types.ModuleType("Draw_maps")
    cylinder = type("CylinderThreat", (), {})
    cylinder.__module__ = "Draw_maps"
    module.CylinderThreat = cylinder
    sys.modules["Draw_maps"] = module


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    _install_pickle_compatibility_module()
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_metric(path: Path, *, label: str) -> np.ndarray:
    values = np.asarray(_load_pickle(path), dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{label}: expected a non-empty trial-by-iteration array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite metric values")
    return values


def _benchmark_file(root: Path, algorithm: str, problem: str, tau: int, metric: str) -> Path:
    return root / f"{algorithm}_{problem}_{tau}_{metric}s.pkl"


def load_aligned_benchmark_series(problem: str, tau: int, metric: str) -> dict[str, np.ndarray]:
    expected_length = 30 * tau
    result: dict[str, np.ndarray] = {}
    for code, _, _, _ in ALGORITHMS:
        values = _load_metric(
            _benchmark_file(BENCHMARK_RESULT_ROOT, code, problem, tau, metric),
            label=f"{code}/{problem}/tau={tau}/{metric}",
        )
        if values.shape[1] == expected_length + 1:
            values = values[:, 1:]
        if values.shape[1] != expected_length:
            raise ValueError(
                f"{code}/{problem}/tau={tau}: expected F_1..F_{expected_length}, got {values.shape[1]} columns"
            )
        result[code] = values
    if any(values.shape[1] != expected_length for values in result.values()):
        raise RuntimeError(f"{problem}/tau={tau}/{metric}: historical series lengths differ")
    return result


def _save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ROOT / name
    _enlarge_axis_text(fig)
    fig.savefig(output, format="pdf", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output


def _enlarge_axis_text(fig: plt.Figure) -> None:
    for axis in fig.axes:
        is_3d = hasattr(axis, "zaxis")
        xy_label_size = AXIS_LABEL_SIZE if is_3d else AXIS_LABEL_SIZE + 1
        axis.xaxis.label.set_size(xy_label_size)
        axis.yaxis.label.set_size(xy_label_size)
        if is_3d:
            axis.zaxis.label.set_size(AXIS_LABEL_SIZE)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        if is_3d:
            axis.tick_params(axis="z", labelsize=TICK_LABEL_SIZE)
            axis.xaxis.set_major_locator(LinearLocator(6))
            axis.yaxis.set_major_locator(LinearLocator(6))
            axis.zaxis.set_major_locator(LinearLocator(6))


def _rasterize_planar_terrain(axis) -> None:
    for collection in axis.collections:
        if isinstance(collection, QuadMesh):
            collection.set_rasterized(True)


def _line_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=color, linestyle=style, linewidth=1.25, label=label)
            for _, label, color, style in ALGORITHMS]


def _export_legend(name: str, *, include_markers: bool = False) -> Path:
    fig_width = 7.16 if include_markers else 8.0
    legend_font_size = 10 if include_markers else 14
    fig, ax = plt.subplots(figsize=(fig_width, 0.34))
    ax.axis("off")
    handles = _line_handles()
    if include_markers:
        handles = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#1b5e20", markersize=7, label="Start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#8e0000", markersize=8, label="Target"),
            *handles,
        ]
    legend = ax.legend(handles=handles, loc="center", ncol=len(handles), fontsize=legend_font_size,
                       handlelength=1.4 if include_markers else 2.2,
                       columnspacing=0.55 if include_markers else 1.05, frameon=True)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    output = FIGURE_ROOT / name
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output


def export_benchmark_curves(metric: str, problems: tuple[str, ...] = BENCHMARK_PROBLEMS) -> list[Path]:
    outputs: list[Path] = []
    for problem in problems:
        for tau in TAUS:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            fig.subplots_adjust(bottom=0.15, top=0.92, right=0.95)
            series = load_aligned_benchmark_series(problem, tau, metric)
            x = np.arange(1, next(iter(series.values())).shape[1] + 1)
            for code, _, color, style in ALGORITHMS:
                y = np.mean(series[code], axis=0)
                ax.plot(x, y, color=color, linestyle=style, linewidth=1.7)
            ax.set_xlim(0, int(x[-1]))
            tick_interval = 20 if tau == 5 else 50
            ax.set_xticks(np.arange(0, int(x[-1]) + 1, tick_interval))
            ax.set_xlabel("Iteration Number", fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE)
            if problem == "F5":
                ax.set_ylim(0.0, 2.5)
                ax.set_yticks(np.arange(0.0, 2.5 + 1e-9, 0.5))
            elif problem == "DMOP1":
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, 0.2))
            ax.tick_params(labelsize=TICK_LABEL_SIZE, width=0.8, length=3.2)
            display_problem = "dMOP1" if problem == "DMOP1" else problem
            ax.set_title(f"{metric} curves on {display_problem} with $\\tau_t={tau}$", fontsize=TITLE_SIZE, pad=5)
            outputs.append(_save_figure(fig, f"{metric}_curves_on_{display_problem}_tau{tau}.pdf"))
    return outputs


def load_trajectory_series(scenario: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for code, _, _, _ in ALGORITHMS:
        values = _load_metric(
            TRAJECTORY_RESULT_ROOT / f"{code}_{scenario}_HVs.pkl",
            label=f"{code}/{scenario}/HV",
        )
        if values.shape[1] == TRAJECTORY_ITERATIONS + 1:
            values = values[:, 1:]
        if values.shape[1] != TRAJECTORY_ITERATIONS:
            raise ValueError(f"{code}/{scenario}: expected {TRAJECTORY_ITERATIONS} HV values, got {values.shape[1]}")
        result[code] = values
    return result


def export_trajectory_hv_curves() -> list[Path]:
    outputs: list[Path] = []
    for scenario in TRAJECTORY_SCENARIOS:
        series = load_trajectory_series(scenario)
        fig, axis = plt.subplots(figsize=(6.4, 4.8))
        x = np.arange(1, TRAJECTORY_ITERATIONS + 1)
        for code, label, color, style in ALGORITHMS:
            axis.plot(x, np.mean(series[code], axis=0), color=color, linestyle=style,
                      linewidth=1.7, label=label)
        axis.set_xlim(0, TRAJECTORY_ITERATIONS)
        axis.set_xticks(np.arange(0, TRAJECTORY_ITERATIONS + 1, 20))
        axis.set_xlabel("Iteration Number", fontsize=AXIS_LABEL_SIZE)
        axis.set_ylabel("HV", fontsize=AXIS_LABEL_SIZE)
        axis.set_title(f"HV curves on {scenario}", fontsize=TITLE_SIZE, pad=5)
        axis.tick_params(labelsize=TICK_LABEL_SIZE, width=0.8, length=3.2)
        outputs.append(_save_figure(fig, f"HV_curves_on_{scenario}.pdf"))
    return outputs


def _select_preference_solution(archive):
    from jmetal.util.constraint_handling import is_feasible, overall_constraint_violation_degree,number_of_violated_constraints
    from sklearn.preprocessing import MinMaxScaler

    if not archive:
        raise ValueError("cannot select a path from an empty archive")
    feasible = [solution for solution in archive if is_feasible(solution)]
    if feasible:
        normalized = MinMaxScaler().fit_transform(
            np.asarray([solution.objectives for solution in feasible], dtype=float)
        )
        return feasible[int(np.argmin(normalized @ np.array([0.5, 0.5])))]
    return archive[int(np.argmin([number_of_violated_constraints(solution) for solution in archive]))]


def _best_archive_trial(code: str, scenario: str) -> dict:
    hvs = _load_metric(TRAJECTORY_RESULT_ROOT / f"{code}_{scenario}_HVs.pkl", label=f"{code}/{scenario}/HV")
    archives = _load_pickle(TRAJECTORY_RESULT_ROOT / f"{code}_{scenario}_total_archive.pkl")
    if len(archives) != TRIALS:
        raise ValueError(f"{code}/{scenario}: archive trials {len(archives)} != {TRIALS}")
    scores = np.mean(hvs[:, TRAJECTORY_MHV_INDICES], axis=1)
    return archives[int(np.argmax(scores))]


def _load_scenario(scenario: str) -> dict:
    value = _load_pickle(TRAJECTORY_SCENARIO_ROOT / f"{scenario}.pkl")
    if value["name"] != scenario:
        raise ValueError(f"scenario name mismatch: expected {scenario}, got {value['name']}")
    return value


def _paths_for_snapshot(scenario: str, archives: dict[str, dict], snapshot: int) -> dict[str, np.ndarray]:
    paths: dict[str, np.ndarray] = {}
    for code, _, _, _ in ALGORITHMS:
        solution = _select_preference_solution(archives[code][snapshot])
        path = np.asarray(solution.attributes["path"], dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"{scenario}/{code}/t={snapshot}: invalid path shape {path.shape}")
        paths[code] = path
    return paths


def _smooth_path(path: np.ndarray, samples: int = 100) -> np.ndarray:
    """Use the natural cubic spline employed by the historical renderer."""
    if len(path) < 2:
        return path
    if len(path) == 2:
        weights = np.linspace(0.0, 1.0, samples)
        return np.asarray([path[0] * (1.0 - weight) + path[1] * weight for weight in weights])
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    parameter = np.concatenate(([0.0], np.cumsum(distances)))
    if parameter[-1] == 0.0:
        parameter = np.linspace(0.0, 1.0, len(path))
    else:
        parameter /= parameter[-1]
    sample_parameter = np.linspace(0.0, 1.0, samples)
    return np.column_stack([
        CubicSpline(parameter, path[:, coordinate], bc_type="natural")(sample_parameter)
        for coordinate in range(3)
    ])


def _threat_value(threat, name: str):
    return threat[name] if isinstance(threat, dict) else getattr(threat, name)


def _draw_threats_3d(axis, threats) -> None:
    for threat in threats:
        center_x = _threat_value(threat, "center_x")
        center_y = _threat_value(threat, "center_y")
        inner_radius = _threat_value(threat, "inner_radius")
        outer_radius = _threat_value(threat, "outer_radius")
        height = _threat_value(threat, "height")
        z = np.linspace(0.0, height, 50)
        theta = np.linspace(0.0, 2.0 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = center_x + outer_radius * np.cos(theta_grid)
        y_grid = center_y + outer_radius * np.sin(theta_grid)
        axis.plot_surface(x_grid, y_grid, z_grid, color="orange", alpha=0.6)

        for theta_level in np.linspace(0.0, 2.0 * np.pi, 10):
            x_line = center_x + outer_radius * np.cos(theta_level)
            y_line = center_y + outer_radius * np.sin(theta_level)
            z_line = np.linspace(1.0, height - 1.0, 100)
            axis.plot([x_line] * 100, [y_line] * 100, z_line,
                      color="black", linewidth=2.0, alpha=0.6)

        for surface_height in (0.0, height):
            inner_circle = Circle(
                (center_x, center_y), inner_radius, facecolor="red",
                alpha=0.6, edgecolor="black", linewidth=2.0,
            )
            axis.add_patch(inner_circle)
            art3d.pathpatch_2d_to_3d(inner_circle, z=surface_height, zdir="z")
            outer_circle = Circle(
                (center_x, center_y), outer_radius, facecolor="orange",
                alpha=0.6, edgecolor="black", linewidth=2.0,
            )
            axis.add_patch(outer_circle)
            art3d.pathpatch_2d_to_3d(outer_circle, z=surface_height, zdir="z")

        center = Circle(
            (center_x, center_y), 3.0, facecolor="black",
            alpha=1.0, edgecolor="black", linewidth=2.0, zorder=10,
        )
        axis.add_patch(center)
        art3d.pathpatch_2d_to_3d(center, z=height, zdir="z")


def _draw_paths(axis, paths: dict[str, np.ndarray], *, planar: bool) -> None:
    for code, label, color, style in ALGORITHMS:
        path = paths[code]
        smooth = _smooth_path(path)
        if planar:
            axis.plot(smooth[:, 0], smooth[:, 1], color=color, linestyle=style,
                      linewidth=2.0, label=label, zorder=10)
            axis.scatter(path[1:-1, 0], path[1:-1, 1], color=color, s=30, zorder=10)
        else:
            axis.plot(smooth[:, 0], smooth[:, 1], smooth[:, 2], color=color,
                      linestyle=style, linewidth=2.0, label=label, zorder=10)
            axis.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2], color=color,
                         s=30, zorder=10)


def _save_trajectory_figure(figure: plt.Figure, name: str, *, planar: bool) -> Path:
    output = FIGURE_ROOT / name
    _enlarge_axis_text(figure)
    if planar:
        _rasterize_planar_terrain(figure.axes[0])
        dpi = PLANAR_BACKGROUND_DPI
    else:
        dpi = EXPORT_DPI
    # Matplotlib's 3D tight bounding box does not reliably include the rotated
    # z-axis label. The original trajectory renderer used a larger 3D margin.
    figure.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight",
                   pad_inches=0.02 if planar else 0.3)
    plt.close(figure)
    return output


def _render_snapshot(scenario: str, snapshot: int, state: dict, paths: dict[str, np.ndarray], terrain) -> list[Path]:
    title_suffix = f"{scenario} at {snapshot}th iteration"
    threats = state["threats"]
    primary_code = ALGORITHMS[0][0]
    start = paths[primary_code][0]
    target = paths[primary_code][-1]
    x, y, z = terrain

    figure_3d = plt.figure(figsize=(11, 10))
    axis_3d = figure_3d.add_subplot(111, projection="3d")
    axis_3d.plot_surface(x, y, z, cmap="terrain", linewidth=0, antialiased=True, alpha=0.5)
    _draw_threats_3d(axis_3d, threats)
    _draw_paths(axis_3d, paths, planar=False)
    axis_3d.scatter(*start, color="#1b5e20", s=100, marker="^", zorder=11)
    axis_3d.scatter(*target, color="#8e0000", s=100, marker="*", zorder=11)
    axis_3d.set_xlim(0, 1000)
    axis_3d.set_ylim(0, 1000)
    axis_3d.set_zlim(0, 310)
    axis_3d.set_xlabel(r"$x$ coordinate (m)")
    axis_3d.set_ylabel(r"$y$ coordinate (m)")
    axis_3d.set_zlabel(r"$z$ coordinate (m)")
    axis_3d.set_title(f"3D UAV trajectories in {title_suffix}", fontsize=TITLE_SIZE, pad=3)
    axis_3d.view_init(elev=35, azim=135)
    figure_3d.subplots_adjust(left=0.0, right=0.8, bottom=0.0, top=0.8)

    figure_2d, axis_2d = plt.subplots(figsize=(8, 8))
    axis_2d.pcolormesh(x, y, z, cmap="terrain", shading="auto")
    axis_2d.contour(x, y, z, colors="black", linewidths=0.5)
    _draw_paths(axis_2d, paths, planar=True)
    axis_2d.scatter(start[0], start[1], color="#1b5e20", s=200, marker="^", zorder=11)
    axis_2d.scatter(target[0], target[1], color="#8e0000", s=200, marker="*", zorder=11)
    for index, threat in enumerate(threats, start=1):
        center_x = _threat_value(threat, "center_x")
        center_y = _threat_value(threat, "center_y")
        outer_radius = _threat_value(threat, "outer_radius")
        inner_radius = _threat_value(threat, "inner_radius")
        axis_2d.add_patch(Circle((center_x, center_y), outer_radius,
                                 facecolor="orange", alpha=1.0, edgecolor="black", linewidth=1.5, zorder=5))
        axis_2d.add_patch(Circle((center_x, center_y), inner_radius,
                                 facecolor="red", alpha=1.0, edgecolor="black", linewidth=1.5, zorder=6))
        axis_2d.text(center_x, center_y, fr"$T_{index}$", ha="center", va="center",
                     color="white", fontsize=TITLE_SIZE - 4, zorder=7)
    axis_2d.plot([0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0],
                 color="black", linewidth=2.0)
    axis_2d.set_xlim(0, 1000)
    axis_2d.set_ylim(0, 1000)
    axis_2d.set_aspect("equal", adjustable="box")
    axis_2d.set_xlabel(r"$x$ coordinate (m)")
    axis_2d.set_ylabel(r"$y$ coordinate (m)")
    axis_2d.set_title(f"Planar projection of UAV trajectories in {title_suffix}", fontsize=TITLE_SIZE, y=1.03, pad=6)
    figure_2d.tight_layout()

    return [
        _save_trajectory_figure(figure_3d, f"3D UAV trajectories in {title_suffix}.pdf", planar=False),
        _save_trajectory_figure(figure_2d, f"projection of UAV trajectories in {title_suffix}.pdf", planar=True),
    ]


def export_trajectory_paths(scenario: str, snapshots: tuple[int, ...] = TRAJECTORY_SNAPSHOTS) -> list[Path]:
    state_sequence = _load_scenario(scenario)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_TERRAIN_FILE.open("rb") as handle:
        terrain = pickle.load(handle)
    archives = {code: _best_archive_trial(code, scenario) for code, _, _, _ in ALGORITHMS}
    outputs: list[Path] = []
    for snapshot in snapshots:
        paths = _paths_for_snapshot(scenario, archives, snapshot)
        state = state_sequence[snapshot // TRAJECTORY_TAU - 1]
        outputs.extend(_render_snapshot(scenario, snapshot, state, paths, terrain))
    return outputs


def main() -> None:
    """Export the complete figure set used by the manuscript."""
    outputs = [
        *export_benchmark_curves("IGD", BENCHMARK_PROBLEMS),
        *export_benchmark_curves("HVD", BENCHMARK_PROBLEMS),
        _export_legend("only_legend.pdf"),
        *export_trajectory_hv_curves(),
    ]
    for scenario in TRAJECTORY_SCENARIOS:
        outputs.extend(export_trajectory_paths(scenario, TRAJECTORY_SNAPSHOTS))
    outputs.append(_export_legend("only_legend_scen.pdf", include_markers=True))
    print("Vector figures exported:")
    for output in outputs:
        print(f"  {output.relative_to(REPRODUCIBILITY_ROOT)}")


if __name__ == "__main__":
    main()
