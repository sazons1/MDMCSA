# MDMCSA Reproducibility Package

This package contains the independent source code, data, main-experiment results, and runners for the MDMCSA manuscript. It includes MDMCSA, DNSGA-II-A/B, SGEA, MOEA/D-SVR, and DBCSA-II on the benchmark and three-dimensional trajectory experiments.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Run commands from this directory, or set `PYTHONPATH` to this directory.

## Main Experiments

```powershell
python scripts/run_benchmarks.py
python scripts/run_trajectories.py
```

`run_benchmarks.py` reproduces the dynamic benchmark experiments and saves fronts, IGD, HVD, and algorithm runtime. `run_trajectories.py` reproduces the S1--S3 three-dimensional trajectory experiments and saves archives, fronts, HV, and algorithm runtime.

`vector_figure_export.py` exports the complete manuscript-style figure set, including benchmark curves, trajectory HV curves, and S1--S3 trajectory panels, from the packaged results into `results/figures`. It does not run an optimizer.

```powershell
python scripts/vector_figure_export.py
```

The default commands run the manuscript configurations with 30 trials. Use `--algorithms`, `--problems`, `--scenarios`, `--tau`, and `--trials` to select a subset.


## Contents

- `src/`: algorithms and problem models.
- `data/`: pure-data S1--S3 scenarios, terrain data, and MOEA/D weights.
- `results/`: saved main-experiment outputs only.
- `scripts/`: benchmark and trajectory experiment runners.
