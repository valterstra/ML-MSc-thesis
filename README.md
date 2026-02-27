# CareAI

Standalone workspace for CARE-AI temporal pipeline work.

## Current scope (v1.1)
- Build admission-step transition dataset from MIMIC-derived stage-02 data.
- Derive observed action classes from discharge destination:
  - `A_LOW_SUPPORT`
  - `A_HIGH_SUPPORT`
  - `A_TERMINAL`
  - `A_UNKNOWN`
- Run deterministic 2% sample for fast iteration.
- Validate schema and produce QA summary report.

## Inputs
- `Yuyang-Clean-Rewrite/mimic/pipeline_v2/data/02_cohort_lace.csv`

## Outputs
- `data/processed/transitions_v1_1.csv`
- `data/processed/transitions_v1_1_sample2pct.csv`
- `data/processed/manifest_transition_v1_1.json`
- `reports/transition_v1_1/qa_summary.json`
- `reports/transition_v1_1/qa_summary.md`

## Run
From repository root:

```powershell
python CareAI/scripts/transitions/run_v1.py --sample-frac 0.02
python CareAI/scripts/transitions/validate_v1.py --input CareAI/data/processed/transitions_v1_1_sample2pct.csv
python CareAI/scripts/transitions/qa_v1.py --input CareAI/data/processed/transitions_v1_1_sample2pct.csv --output-dir CareAI/reports/transition_v1_1
```

## Causal baseline (next step)
Estimate adjusted effect of `A_HIGH_SUPPORT` vs `A_LOW_SUPPORT` on 30d readmission:

```powershell
python CareAI/scripts/effects/estimate_v1.py --config CareAI/configs/causal_effect_v1.yaml
```

For a faster smoke test:

```powershell
python CareAI/scripts/effects/estimate_v1.py --config CareAI/configs/causal_effect_v1.yaml --input CareAI/data/processed/transitions_v1_1_sample2pct.csv --bootstrap-resamples 50
```

## Sim v1 (one-step bandit)
Run baseline policy simulation on the action-aware transition dataset:

```powershell
python CareAI/scripts/sim/run_v1.py --config CareAI/configs/sim_v1.yaml
```

Quick smoke run:

```powershell
python CareAI/scripts/sim/run_v1.py --config CareAI/configs/sim_v1.yaml --n-episodes 2000
```

Optional causal-weighted training mode (IPW on train split):

```powershell
python CareAI/scripts/sim/run_v1.py --config CareAI/configs/sim_v1.yaml --weighting-mode ipw_stabilized
```

When enabled, an additional file is written:
- `reports/sim_v1/sim_v1_train_weights.csv`

## Transition v2 (multi-step chains)
Build event-driven multi-step transitions side-by-side with v1:

```powershell
python CareAI/scripts/transitions/run_v2_multi.py --config CareAI/configs/transition_v2_multi.yaml --sample-frac 0.02
python CareAI/scripts/transitions/validate_v2_multi.py --input CareAI/data/processed/transitions_v2_multi_sample2pct.csv
python CareAI/scripts/transitions/qa_v2.py --input CareAI/data/processed/transitions_v2_multi_sample2pct.csv --output-dir CareAI/reports/transition_v2_multi
```

## Longitudinal v1 (long table + padded tensor)
Build additive longitudinal artifacts from transition v2:

```powershell
python CareAI/scripts/transitions/build_longitudinal_v1.py --config CareAI/configs/longitudinal_v1.yaml
python CareAI/scripts/transitions/qa_longitudinal_v1.py --config CareAI/configs/longitudinal_v1.yaml
```

Outputs:
- `data/processed/longitudinal_v1_long.csv`
- `data/processed/longitudinal_v1_tensor/` (`.npy` arrays + `episode_index.csv` + `metadata.json`)
- `reports/longitudinal_v1/qa_summary.json`
- `reports/longitudinal_v1/qa_summary.md`

Legacy script paths in `CareAI/scripts/*.py` are still supported and forward to these canonical stage-based paths.

## Transition v3 (MIMIC-code hourly treatment pipeline)
Build hourly ICU transitions using MIMIC-code concepts materialized in Postgres:

```powershell
python CareAI/scripts/transitions/run_v3_mimiccode_hourly.py --config CareAI/configs/transition_v3_mimiccode_hourly.yaml
```

Useful flags:

```powershell
python CareAI/scripts/transitions/run_v3_mimiccode_hourly.py --config CareAI/configs/transition_v3_mimiccode_hourly.yaml --skip-concept-build
python CareAI/scripts/transitions/run_v3_mimiccode_hourly.py --config CareAI/configs/transition_v3_mimiccode_hourly.yaml --limit-stays 50
```

Outputs:
- `data/processed/transitions_v3_mimiccode_hourly.csv`
- `data/processed/transitions_v3_mimiccode_hourly_sample2pct.csv`
- `data/processed/manifest_transition_v3_mimiccode_hourly.json`
- `reports/transition_v3_mimiccode_hourly/qa_summary.json`
- `reports/transition_v3_mimiccode_hourly/qa_summary.md`

## Readmission head v1 (episode-level 30d target)
Build episode-level table and train baseline readmission head from v3 transitions:

```powershell
python CareAI/scripts/readmission/build_episode_table_v1.py --config CareAI/configs/readmission_head_v1.yaml
python CareAI/scripts/readmission/qa_episode_table_v1.py --config CareAI/configs/readmission_head_v1.yaml
python CareAI/scripts/readmission/train_readmission_head_v1.py --config CareAI/configs/readmission_head_v1.yaml
```

Outputs:
- `data/processed/readmission_episode_v1.csv`
- `reports/readmission_head_v1/readmission_head_v1_qa_summary.json`
- `reports/readmission_head_v1/readmission_head_v1_qa_summary.md`
- `reports/readmission_head_v1/readmission_head_v1_model_summary.json`
- `reports/readmission_head_v1/readmission_head_v1_model_summary.md`
- `reports/readmission_head_v1/readmission_head_v1_predictions_valid.csv`
- `reports/readmission_head_v1/readmission_head_v1_predictions_test.csv`

## Sim hourly v1 (multi-step deterministic rollout)
Run hourly state-action-state simulation and policy comparison using the v3 transitions and readmission episode table:

```powershell
python CareAI/scripts/sim/run_hourly_v1.py --config CareAI/configs/sim_hourly_v1.yaml
```

Quick smoke run:

```powershell
python CareAI/scripts/sim/run_hourly_v1.py --config CareAI/configs/sim_hourly_v1.yaml --n-rollouts 50 --max-steps 24
```

Outputs:
- `reports/sim_hourly_v1/sim_hourly_v1_summary.json`
- `reports/sim_hourly_v1/sim_hourly_v1_summary.md`
- `reports/sim_hourly_v1/sim_hourly_v1_policy_metrics.csv`
- `reports/sim_hourly_v1/sim_hourly_v1_trajectories.csv`

Counterfactual visualization from one initial state with fixed repeated actions:

```powershell
python CareAI/scripts/sim/plot_hourly_counterfactual_v1.py --config CareAI/configs/sim_hourly_v1.yaml --max-steps 48
```

Outputs:
- `reports/sim_hourly_v1/sim_hourly_v1_counterfactual_paths.csv`
- `reports/sim_hourly_v1/sim_hourly_v1_counterfactual_risks.csv`
- `reports/sim_hourly_v1/sim_hourly_v1_counterfactual_plot.png`
- `reports/sim_hourly_v1/sim_hourly_v1_counterfactual_summary.json`
- `reports/sim_hourly_v1/sim_hourly_v1_counterfactual_summary.md`
