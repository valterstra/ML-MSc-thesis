# CareAI

CareAI is the active codebase for an MSc thesis workflow built around MIMIC-IV. The repository contains two parallel pipelines:

- **Daily hospital pipeline** (active thesis track): builds a hospital-wide daily state-action-state dataset from raw MIMIC-IV tables, trains a LightGBM transition model per output variable, and runs rollout evaluation of the resulting simulator.
- **Hourly ICU pipeline** (earlier track): builds hourly ICU transitions from MIMIC-code concepts, derives a 30-day readmission model, and runs hourly simulation experiments.

The repository is organized as a practical pipeline rather than a general-purpose package.

## What This Repository Contains

### Daily hospital pipeline (primary track)

- `hosp_daily`: builds a daily (s, a, s') dataset directly from MIMIC-IV hosp/ICU tables — hospital-wide, not ICU-only
- `sim_daily`: trains one LightGBM model per state output and runs rollout evaluation

### Hourly ICU pipeline (earlier track)

- `transition_hourly_mimiccode`: builds hourly ICU transitions from MIMIC-code concepts stored in Postgres
- `readmission`: builds an episode table and trains a baseline 30-day readmission model
- `hourly_sim`: runs simulated hourly policy rollouts, counterfactual plots, and hardening checks

## Project Flow

### Daily pipeline

```text
MIMIC-IV in Postgres (mimiciv_hosp, mimiciv_icu, mimiciv_derived schemas)
  -> scripts/hosp_daily/build_hosp_daily.py   (builds hosp_daily_transitions.csv)
  -> scripts/sim/train_daily_transition.py    (trains LightGBM transition models)
  -> scripts/sim/evaluate_daily_sim.py        (rollout evaluation and metrics)
```

### Hourly pipeline

```text
MIMIC-IV + mimic-code concepts in Postgres
  -> hourly transition dataset
  -> episode-level readmission dataset
  -> hourly simulation and robustness analysis
```

## Repository Layout

```text
CareAI/
  configs/              YAML configuration files for each pipeline stage
  data/                 Processed outputs written by the pipeline (gitignored)
  models/               Trained model artifacts (gitignored)
  reports/              QA summaries, model summaries, and simulation outputs (gitignored)
  scripts/
    hosp_daily/         Daily dataset build entry points
    sim/                Transition model training and evaluation
    readmission/        Readmission episode build and model training
    transitions/        Hourly transition build
  src/careai/
    hosp_daily/         Daily dataset SQL extraction and assembly
    sim_daily/          LightGBM transition model, rollout env, evaluation
    readmission/        Readmission episode builder and baseline model
    sim_hourly/         Hourly simulator
    transitions/        Hourly transition build logic
  tests/                Unit and integration tests
  docs/                 Reference notes (MIMIC schema, cohort decisions)
```

Key entry points — daily pipeline:

- `scripts/hosp_daily/build_hosp_daily.py`
- `scripts/sim/train_daily_transition.py`
- `scripts/sim/evaluate_daily_sim.py`

Key entry points — hourly pipeline:

- `scripts/transitions/run_hourly_mimiccode.py`
- `scripts/readmission/build_episode_table.py`
- `scripts/readmission/train_readmission_head.py`
- `scripts/sim/run_hourly_sim.py`
- `scripts/sim/harden_hourly_sim.py`

## Requirements

### Python

- Python 3.10 or newer
- All dependencies are defined in `pyproject.toml`: pandas, numpy, scikit-learn, lightgbm, psycopg2-binary, joblib, PyYAML, scipy, statsmodels

Install the project in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Data and Database

Both pipelines require MIMIC-IV loaded into a local Postgres database.

- **Daily pipeline** uses `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived` schemas directly from the MIMIC-IV Postgres dump.
- **Hourly pipeline** additionally requires mimic-code derived concept tables; see `configs/transition_hourly_mimiccode.yaml` for the expected sibling repo path.

Default database config (shared by both pipelines):

- Postgres host: `localhost`
- Postgres port: `5432`
- Database name: `mimic`
- Username from environment variable: `PGUSER`
- Password from environment variable: `PGPASSWORD`

Set credentials before running:

```powershell
$env:PGUSER="your_username"
$env:PGPASSWORD="your_password"
```

## Quick Start

### Daily pipeline (start here for the current thesis track)

Prerequisites: MIMIC-IV loaded into Postgres with schemas `mimiciv_hosp`, `mimiciv_icu`, and `mimiciv_derived`. Set credentials:

```powershell
$env:PGUSER="your_username"
$env:PGPASSWORD="your_password"
```

#### Step 1 — Build the daily dataset

```powershell
python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml
```

To build a 5,000-episode sample instead of the full dataset:

```powershell
python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml --sample-only
```

Primary outputs:

- `data/processed/hosp_daily_transitions.csv`
- `data/processed/hosp_daily_transitions_sample5k.csv`
- `reports/hosp_daily/build_manifest.json`

#### Step 2 — Train the transition model

```powershell
python scripts/sim/train_daily_transition.py
```

To train on the sample dataset:

```powershell
python scripts/sim/train_daily_transition.py --csv data/processed/hosp_daily_transitions_sample5k.csv
```

Primary outputs (written to `models/sim_daily/`):

- One `.joblib` file per output variable (continuous labs, binary flags, discharge)
- `models/sim_daily/metadata.json`

#### Step 3 — Evaluate the simulator

```powershell
python scripts/sim/evaluate_daily_sim.py
```

Primary outputs:

- `reports/sim_daily/single_step_metrics.json`
- `reports/sim_daily/simulated_trajectories.csv`
- `reports/sim_daily/rollout_comparison.json`

---

### Hourly pipeline (earlier track)

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
