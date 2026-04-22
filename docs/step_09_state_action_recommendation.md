# Step 09 State and Action Recommendation

Purpose:
- Record the current recommended expanded Step 09 set after comparing:
  - Step 03 state -> readmission results
  - Step 04b robust action -> state results

Context:
- Step 03 still provides the outcome anchor:
  which discharge states are most relevant for readmission.
- Step 04b now provides the stronger controllability filter:
  which actions robustly shift which states under the multivariate random-graph setup.
- The old `09b` action-analysis branch is retained only as legacy step `21`
  and is not a downstream dependency of the active selected-set pipeline.

## Recommended Main Set

States:
- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

Actions:
- `ivfluid`
- `diuretic`
- `vasopressor`
- `mechvent`
- `antibiotic`

Static confounders carried with the selected set:

- `age`
- `charlson_score`
- `prior_ed_visits_6m`

## Borderline Additions

States:
- `Shock_Index`

Actions:
- `insulin`

## Decision Table

### States

| State | Step 03 signal | Step 04b support | Recommendation | Reason |
|---|---:|---|---|---|
| `Hb` | very strong | `vasopressor 1.00`, `ivfluid 0.97`, `antibiotic 0.96`, `mechvent 0.94` | keep | strongest readmission state, strongly and repeatedly modifiable |
| `BUN` | very strong | `diuretic 1.00`, `ivfluid 0.95`, `insulin 0.83` | keep | strongest renal readmission state with clean actionable coverage |
| `Creatinine` | strong | `diuretic 0.94`, `ivfluid 0.87` | keep | strong readmission state with robust renal/fluid action support |
| `Phosphate` | strong | `ivfluid 1.00`, `antibiotic 0.58`, `diuretic 0.57` | keep | still important for readmission and now clearly controllable |
| `HR` | strong | `mechvent 1.00`, `vasopressor 0.77`, `antibiotic 0.28` | keep | outcome-relevant and still controllable, though action story changed |
| `Chloride` | moderate | `diuretic 1.00`, `ivfluid 0.97`, `insulin 0.89` | keep | not as high in Step 03, but very strongly controllable in Step 04b |
| `Shock_Index` | moderate | `insulin 0.55`, `diuretic 0.51`, `antibiotic 0.37` | borderline | still relevant, but the robust action story weakened versus old Step 04 |
| `Ht` | very strong | `vasopressor 1.00`, `ivfluid 0.95`, `antibiotic 0.95`, `mechvent 0.98` | not default | strong, but mostly redundant with `Hb` |
| `PT` | moderate | weak relative to other main states | drop | weaker controllability story than the states above |

### Actions

| Action | Step 04b support | Recommendation | Reason |
|---|---|---|---|
| `ivfluid` | `BUN 0.95`, `Hb 0.97`, `Ht 0.95`, `Creatinine 0.87`, `Phosphate 1.00`, `Chloride 0.97`, `cumulated_balance 1.00`, `input_4hourly_tev 1.00` | keep | best overall action by breadth and strength of coverage |
| `diuretic` | `BUN 1.00`, `Creatinine 0.94`, `Chloride 1.00`, `cumulated_balance 1.00`, `output_total 0.97` | keep | strongest renal/fluid action |
| `vasopressor` | `Hb 1.00`, `Ht 1.00`, `HR 0.77` | keep | still strong and clinically central for hemodynamic intervention |
| `mechvent` | `HR 1.00`, `RR 1.00`, `Hb 0.94`, `Ht 0.98`, `Temp_C 0.98` | keep | much stronger in robust Step 04b than in earlier intuition; important if acceptable in intervention set |
| `antibiotic` | `Hb 0.96`, `Ht 0.95`, `input_4hourly_tev 0.97`, `Phosphate 0.58` | keep | weaker than before, but still useful for broader clinically realistic coverage |
| `insulin` | `RR 0.95`, `Glucose 0.90`, `BUN 0.83`, `Chloride 0.89` | borderline | real signal, but less aligned with the clearest Step 03 core |
| `anticoagulant` | `PTT 1.00`, `output_total 0.92` | drop for main set | strong, but mostly for `PTT`, which is not in the main Step 03-driven state set |
| `sedation` | mixed moderate support | drop for main set | less clean and likely more procedural/confounded |
| `steroid` | unusable in Step 04b | drop | no usable variation in the current run |

## Main Interpretation

- The renal/fluid axis is now even more defensible than before.
- The old `HR` / `Shock_Index` plus `antibiotic` story weakened under the robust Step 04b setup.
- `mechvent` is much more credible after Step 04b than it looked from the earlier Tier-2 framing.
- If a slightly larger set is acceptable, the recommended main set above is the best current compromise between:
  - readmission relevance
  - action modifiability
  - non-redundancy

## Implementation Status

This recommendation is no longer only a note. It has already been pushed into
parallel preprocessing / simulator tracks.

Implemented outputs:

- selected replay dataset:
  - `data/processed/icu_readmit/rl_dataset_selected.parquet`
- selected scaler file:
  - `data/processed/icu_readmit/scaler_params_selected.json`
- selected static context:
  - `data/processed/icu_readmit/static_context_selected.parquet`
- selected-state severity surrogate:
  - `models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib`
  - `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
- selected-set active CARE-Sim training entrypoint:
  - `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`
- selected-set active Colab notebook:
  - `notebooks/step_11a_caresim_selected_causal_colab.ipynb`

Current selected-set training design:

- full simulator input state:
  - 6 dynamic selected states
  - 3 static confounders
  - total state dim = `9`
- action dim = `5`
- action space = `32` binary combinations
- Step 11a selected CARE-Sim track trains:
  - next-state head
  - terminal head
- Step 11a selected CARE-Sim track does **not** train a reward head

Pending:

- selected-set Step 12a / Step 13a should still be wired to use the trained
  severity surrogate to compute dense reward from predicted state transitions

## What `mechvent` Means

`mechvent` is the binary mechanical ventilation flag:
- column constant: `C_MECHVENT = 'mechvent'`
- defined in [columns.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/columns.py)

In this pipeline it is treated as both:
- a clinical intervention/action decision
- and a state-like ongoing support status

That dual role is documented in:
- [columns.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/columns.py)
- [icu_readmit_pipeline.md](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/docs/icu_readmit_pipeline.md)
