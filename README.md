# CareAI

Active CARE-AI workspace for the hourly transition pipeline and its downstream models.

## Scope

This repository now contains only the current thesis track:

- `transition_hourly_mimiccode`: hourly ICU state-action-state transitions built from MIMIC-code concepts in Postgres
- `readmission`: episode-level 30-day readmission model built from the hourly transition data
- `hourly_sim`: multi-step hourly simulator, counterfactual plots, and hardening runs built on the hourly transition data plus the readmission table

Legacy earlier-track work has moved to `../CareAI-second-track/`.

## Transition

Build hourly ICU transitions using MIMIC-code concepts:

```powershell
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml
```

Useful flags:

```powershell
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml --skip-concept-build
python scripts/transitions/run_hourly_mimiccode.py --config configs/transition_hourly_mimiccode.yaml --limit-stays 50
```

Outputs:

- `data/processed/transitions_hourly_mimiccode.csv`
- `data/processed/transitions_hourly_mimiccode_sample2pct.csv`
- `data/processed/manifest_transition_hourly_mimiccode.json`
- `reports/transition_hourly_mimiccode/qa_summary.json`
- `reports/transition_hourly_mimiccode/qa_summary.md`

## Readmission

Build the episode table and train the baseline readmission head:

```powershell
python scripts/readmission/build_episode_table.py --config configs/readmission.yaml
python scripts/readmission/qa_episode_table.py --config configs/readmission.yaml
python scripts/readmission/train_readmission_head.py --config configs/readmission.yaml
```

Outputs:

- `data/processed/readmission_episode.csv`
- `reports/readmission/readmission_qa_summary.json`
- `reports/readmission/readmission_qa_summary.md`
- `reports/readmission/readmission_model_summary.json`
- `reports/readmission/readmission_model_summary.md`
- `reports/readmission/readmission_predictions_valid.csv`
- `reports/readmission/readmission_predictions_test.csv`

## Hourly Simulation

Run hourly policy rollouts:

```powershell
python scripts/sim/run_hourly_sim.py --config configs/hourly_sim.yaml
```

Quick smoke run:

```powershell
python scripts/sim/run_hourly_sim.py --config configs/hourly_sim.yaml --n-rollouts 50 --max-steps 24
```

Outputs:

- `reports/hourly_sim/hourly_sim_summary.json`
- `reports/hourly_sim/hourly_sim_summary.md`
- `reports/hourly_sim/hourly_sim_policy_metrics.csv`
- `reports/hourly_sim/hourly_sim_trajectories.csv`

Generate fixed-action counterfactual plots:

```powershell
python scripts/sim/plot_hourly_counterfactual.py --config configs/hourly_sim.yaml --max-steps 48
```

Outputs:

- `reports/hourly_sim/hourly_sim_counterfactual_paths.csv`
- `reports/hourly_sim/hourly_sim_counterfactual_risks.csv`
- `reports/hourly_sim/hourly_sim_counterfactual_plot.png`
- `reports/hourly_sim/hourly_sim_counterfactual_summary.json`
- `reports/hourly_sim/hourly_sim_counterfactual_summary.md`

Run the simulator hardening workflow:

```powershell
python scripts/sim/harden_hourly_sim.py --config configs/hourly_sim_hardening.yaml --stop-on-pass
```

Outputs are written to timestamped run folders under `reports/hourly_sim/hardening_runs/`.
