# CareAI

CareAI is the active codebase for an MSc thesis workflow built around hourly ICU trajectories from MIMIC-IV. The repository turns raw MIMIC-code concept tables into an hourly transition dataset, derives a 30-day readmission dataset from those transitions, and then uses both artifacts to run simple policy simulation experiments.

The repository is organized as a practical pipeline rather than a general-purpose package. A new reader should think of it in three stages:

1. Build an hourly state-action-state dataset from MIMIC-IV.
2. Collapse those trajectories into episode-level features and train a readmission model.
3. Use the transition data and readmission table inside an hourly simulator for rollout, counterfactual, and hardening experiments.

## What This Repository Contains

The current thesis track focuses on three modules:

- `transition_hourly_mimiccode`: builds hourly ICU transitions from MIMIC-code concepts stored in Postgres
- `readmission`: builds an episode table and trains a baseline 30-day readmission model
- `hourly_sim`: runs simulated hourly policy rollouts, counterfactual plots, and hardening checks

Earlier exploratory tracks were moved out of this repository into sibling folders such as `../CareAI-second-track/`.

## Project Flow

The intended execution order is:

```text
MIMIC-IV + mimic-code concepts in Postgres
  -> hourly transition dataset
  -> episode-level readmission dataset
  -> hourly simulation and robustness analysis
```

In practice:

- The transition step is the data foundation.
- The readmission step depends on the transition output.
- The simulation step depends on both the transition output and the readmission episode table.

## Repository Layout

```text
CareAI/
  configs/      YAML configuration files for each pipeline stage
  data/         Processed outputs written by the pipeline
  reports/      QA summaries, model summaries, and simulation outputs
  scripts/      Command-line entry points for transition, readmission, and simulation runs
  src/careai/   Python implementation
  tests/        Unit and integration tests
```

Key entry points:

- `scripts/transitions/run_hourly_mimiccode.py`
- `scripts/readmission/build_episode_table.py`
- `scripts/readmission/qa_episode_table.py`
- `scripts/readmission/train_readmission_head.py`
- `scripts/sim/run_hourly_sim.py`
- `scripts/sim/plot_hourly_counterfactual.py`
- `scripts/sim/harden_hourly_sim.py`

## Requirements

### Python

- Python 3.10 or newer
- Core dependencies are defined in `pyproject.toml`

Install the project in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Data and Database

The transition pipeline expects access to a local or reachable Postgres database containing MIMIC-IV tables and derived MIMIC-code concepts.

The default config assumes:

- Postgres host: `localhost`
- Postgres port: `5432`
- Database name: `mimic`
- Username from environment variable: `PGUSER`
- Password from environment variable: `PGPASSWORD`

The default transition config also assumes the sibling repository path:

- `../../external_repos_top3/mimic-code`

Schema defaults in `configs/transition_hourly_mimiccode.yaml`:

- `mimiciv_hosp_typed`
- `mimiciv_icu`
- `mimiciv_derived`

Before running the transition build, set your database credentials in the shell:

```powershell
$env:PGUSER="your_username"
$env:PGPASSWORD="your_password"
```

## Quick Start

If you are new to the project, use this order.

### 1. Build the hourly transition dataset

```powershell
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml
```

Useful variants:

```powershell
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml --skip-concept-build
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml --limit-stays 50
```

Primary outputs:

- `data/processed/transitions_hourly_mimiccode.csv`
- `data/processed/transitions_hourly_mimiccode_sample2pct.csv`
- `data/processed/manifest_transition_hourly_mimiccode.json`
- `reports/transition_hourly_mimiccode/qa_summary.json`
- `reports/transition_hourly_mimiccode/qa_summary.md`

### 2. Build and validate the readmission dataset

```powershell
python scripts/readmission/build_episode_table.py --config configs/readmission.yaml
python scripts/readmission/qa_episode_table.py --config configs/readmission.yaml
python scripts/readmission/train_readmission_head.py --config configs/readmission.yaml
```

Primary outputs:

- `data/processed/readmission_episode.csv`
- `reports/readmission/readmission_qa_summary.json`
- `reports/readmission/readmission_qa_summary.md`
- `reports/readmission/readmission_model_summary.json`
- `reports/readmission/readmission_model_summary.md`
- `reports/readmission/readmission_predictions_valid.csv`
- `reports/readmission/readmission_predictions_test.csv`

### 3. Run hourly simulation experiments

Standard simulation run:

```powershell
python scripts/sim/run_hourly_sim.py --config configs/hourly_sim.yaml
```

Quick smoke run:

```powershell
python scripts/sim/run_hourly_sim.py --config configs/hourly_sim.yaml --n-rollouts 50 --max-steps 24
```

Primary outputs:

- `reports/hourly_sim/hourly_sim_summary.json`
- `reports/hourly_sim/hourly_sim_summary.md`
- `reports/hourly_sim/hourly_sim_policy_metrics.csv`
- `reports/hourly_sim/hourly_sim_trajectories.csv`

### 4. Generate counterfactual plots

```powershell
python scripts/sim/plot_hourly_counterfactual.py --config configs/hourly_sim.yaml --max-steps 48
```

Primary outputs:

- `reports/hourly_sim/hourly_sim_counterfactual_paths.csv`
- `reports/hourly_sim/hourly_sim_counterfactual_risks.csv`
- `reports/hourly_sim/hourly_sim_counterfactual_plot.png`
- `reports/hourly_sim/hourly_sim_counterfactual_summary.json`
- `reports/hourly_sim/hourly_sim_counterfactual_summary.md`

### 5. Run the hardening workflow

```powershell
python scripts/sim/harden_hourly_sim.py --config configs/hourly_sim_hardening.yaml --stop-on-pass
```

Outputs are written into timestamped folders under:

- `reports/hourly_sim/hardening_runs/`

## Configuration Files

The main configs live in `configs/`:

- `transition_hourly_mimiccode.yaml`: database access, concept build behavior, cohort filters, feature definitions, outputs, and split settings
- `readmission.yaml`: episode construction, readmission labeling horizon, baseline model settings, and feature list
- `hourly_sim.yaml`: state variables, action policies, simulation settings, dynamics model, and readmission model inputs
- `hourly_sim_hardening.yaml`: variant grid, QA thresholds, repeated seeds, and hardening run output structure

For most work, you should edit configs rather than hardcode paths or parameters.

## Current Modeling Assumptions

This repository currently makes a few practical assumptions that are useful to understand before reading the code:

- The transition representation is hourly and ICU-centric.
- The action space is a compact treatment combination over vasoactive support, ventilation, and CRRT.
- The readmission model is a baseline logistic regression over episode-level summary features.
- The simulator is intended for structured experimentation and comparison, not for clinical deployment.

## Testing

Run the test suite with:

```powershell
pytest
```

Targeted examples:

```powershell
pytest tests/unit
pytest tests/integration
pytest tests/integration/test_hourly_sim_smoke.py
```

## Outputs and Artifacts

Most generated artifacts fall into two categories:

- `data/processed/`: reusable intermediate datasets such as transition tables and episode tables
- `reports/`: human-readable summaries, QA reports, plots, model diagnostics, and simulation outputs

If you are trying to understand whether a run worked, start by checking the corresponding report directory.

## Who This README Is For

This README is written for a reader who is new to the repository and wants to answer four questions quickly:

1. What problem is this project solving?
2. What are the main pipeline stages?
3. What do I need before I can run it?
4. Which command should I run first?

If you are that reader, start with the transition pipeline, confirm the outputs under `data/processed/` and `reports/transition_hourly_mimiccode/`, and only then move downstream.
