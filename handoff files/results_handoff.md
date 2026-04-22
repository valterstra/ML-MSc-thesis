# Results Handoff

Code-grounded RESULTS handoff for the active ICU-readmission thesis pipeline.

Purpose: give a thesis-writing model enough exact information to draft the Results chapter without reopening the codebase.

Date prepared: 2026-04-14  
Repo: `CareAI` main selected-causal branch

## Evidence rules used in this handoff

- `Confirmed from code/output` = directly verified from scripts, saved JSON/CSV/parquet files, model configs, or logs in this repo.
- `Inferred from surrounding context` = reconstructed from notebooks, artifact naming, or cross-file comparison. These are still useful, but should not be presented as hard output facts unless rechecked.

## Priority materials read first

Read first as requested:

1. `handoff files/data_handoff.md`
2. `handoff files/methods_handoff.md`
3. thesis-planning/support docs found in repo:
   - `README.md`
   - `STATUS.md`
   - `THESIS_GUIDE.md`
   - `docs/chatgpt_thesis_project_instructions.md`
   - `docs/thesis_codex_handoff_template.md`
4. `AGENTS.md`:
   - not present in this repo
5. final-script / final-output identification docs:
   - `README.md`
   - `STATUS.md`
   - `docs/caresim_playbook.md`
   - `docs/icu_readmit_artifacts.md`
   - `docs/icu_readmit_offline_ddqn_explainer.md`
   - `scripts/package_icu_readmit_artifacts.py`

Important implication:

- `README.md`, `STATUS.md`, and the two existing handoff files describe the active selected-causal thesis pipeline.
- `THESIS_GUIDE.md` contains older full-dataset branches and should not be treated as the source of final Results numbers for the thesis.

---

## A. Final Results Pipeline

### A1. Canonical final pipeline

`Confirmed from code/output`

The active thesis branch is:

1. Step `09`: select state variables associated with readmission
2. Step `10a`: build selected replay dataset
3. Step `10b`: train selected severity surrogate
4. Step `10c`: train selected terminal readmission model
5. Step `11a`: train CARE-Sim selected-causal simulator
6. Step `11b`: train MarkovSim selected-causal baseline
7. Step `12a`: evaluate CARE-Sim
8. Step `12b`: evaluate MarkovSim
9. Step `13a`: train/evaluate CARE-Sim control policies
10. Step `13b`: train/evaluate MarkovSim control policies
11. Step `14`: train offline DDQN and compare policies on held-out logged data

Legacy step `21` preserves the old action-selection diagnostics branch that is
not consumed by the active selected-set pipeline.

The active scripts are:

- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
- `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
- `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`
- `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`
- `scripts/icu_readmit/step_11b_markovsim_train.py`
- `scripts/icu_readmit/step_12a_caresim_evaluate.py`
- `scripts/icu_readmit/step_12b_markovsim_evaluate.py`
- `scripts/icu_readmit/step_13a_caresim_control.py`
- `scripts/icu_readmit/step_13b_markovsim_control.py`
- `scripts/icu_readmit/step_14_offline_selected.py`

### A2. Final artifact folders

`Confirmed from code/output`

Main artifact roots:

- `data/processed/icu_readmit/`
- `models/icu_readmit/`
- `reports/icu_readmit/`
- `logs/`

Final selected-causal report folders:

- `reports/icu_readmit/severity_selected/`
- `reports/icu_readmit/terminal_readmit_selected/`
- `reports/icu_readmit/caresim_selected_causal/`
- `reports/icu_readmit/markovsim_selected_causal/`
- `reports/icu_readmit/caresim_control_selected_causal/`
- `reports/icu_readmit/markovsim_control_selected_causal/`
- `reports/icu_readmit/offline_selected/`

Final selected-causal model folders:

- `models/icu_readmit/severity_selected/`
- `models/icu_readmit/terminal_readmit_selected/`
- `models/icu_readmit/caresim_selected_causal/`
- `models/icu_readmit/markovsim_selected_causal/`
- `models/icu_readmit/caresim_control_selected_causal/`
- `models/icu_readmit/markovsim_control_selected_causal/`
- `models/icu_readmit/offline_selected/`

### A3. Final data objects

`Confirmed from code/output`

Core final data objects:

- `data/processed/icu_readmit/ICUdataset.csv`
- `data/processed/icu_readmit/rl_dataset_selected.parquet`
- `data/processed/icu_readmit/static_context_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`

These are the data objects the final results branch actually uses.

### A4. Exact likely execution order

`Confirmed from code/output` for script names and interfaces  
`Inferred from surrounding context` for the exact command sequence below

```bash
python scripts/icu_readmit/step_10a_rl_preprocess_selected.py
python scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py
python scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py
python scripts/icu_readmit/step_11a_caresim_train_selected_causal.py --save-dir models/icu_readmit/caresim_selected_causal --causal-constraints --n-models 5 --d-model 256 --n-heads 8 --n-layers 4 --n-epochs 30 --batch-size 64 --lr 1e-3
python scripts/icu_readmit/step_11b_markovsim_train.py --data data/processed/icu_readmit/rl_dataset_selected.parquet --save-dir models/icu_readmit/markovsim_selected_causal
python scripts/icu_readmit/step_12a_caresim_evaluate.py --data data/processed/icu_readmit/rl_dataset_selected.parquet --model-dir models/icu_readmit/caresim_selected_causal --report-dir reports/icu_readmit/caresim_selected_causal ...
python scripts/icu_readmit/step_12b_markovsim_evaluate.py --data data/processed/icu_readmit/rl_dataset_selected.parquet --model-dir models/icu_readmit/markovsim_selected_causal --report-dir reports/icu_readmit/markovsim_selected_causal --use-severity-reward --severity-mode handcrafted --use-terminal-readmit-reward --terminal-reward-scale 15.0
python scripts/icu_readmit/step_13a_caresim_control.py train-ddqn ...
python scripts/icu_readmit/step_13a_caresim_control.py eval ...
python scripts/icu_readmit/step_13b_markovsim_control.py train-ddqn ...
python scripts/icu_readmit/step_13b_markovsim_control.py eval ...
python scripts/icu_readmit/step_14_offline_selected.py train-ddqn --dqn-steps 100000 ...
python scripts/icu_readmit/step_14_offline_selected.py eval --physician-steps 35000 --reward-steps 30000 --env-steps 60000 ...
```

### A5. Notebook-grounded full-run parameterization

`Confirmed from code/output`

Notebooks present:

- `notebooks/step_11a_caresim_selected_causal_colab.ipynb`
- `notebooks/step_13a_caresim_selected_colab.ipynb`
- `notebooks/step_13b_markovsim_selected_colab.ipynb`
- `notebooks/step_14_offline_selected_colab.ipynb`

Key notebook parameters:

- CARE-Sim training notebook:
  - `--n-models 5`
  - `--d-model 256`
  - `--n-heads 8`
  - `--n-layers 4`
  - `--n-epochs 30`
  - `--batch-size 64`
  - `--lr 1e-3`
- CARE-Sim control notebook:
  - handcrafted severity reward
  - terminal readmit reward on
  - `terminal_reward_scale 15.0`
  - `history_len 5`
  - `observation_window 5`
  - `rollout_steps 5`
  - `planner_horizon 3`
  - `uncertainty_penalty 0.25`
  - DDQN training `train_steps 20000`
  - evaluation `episodes_per_split 100`
- MarkovSim control notebook mirrors the CARE-Sim control settings
- Offline notebook:
  - DDQN train `--dqn-steps 100000`
  - OPE support models: `physician 35000`, `reward 30000`, `env 60000`

### A6. Outputs that are final and usable

`Confirmed from code/output`

Usable final artifacts:

- `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
- `reports/icu_readmit/severity_selected/severity_surrogate_coefficients.csv`
- `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_metrics.json`
- `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_feature_importance.csv`
- `reports/icu_readmit/caresim_selected_causal/caresim_one_step_val.json`
- `reports/icu_readmit/caresim_selected_causal/caresim_one_step_test.json`
- `reports/icu_readmit/markovsim_selected_causal/markovsim_one_step_val.json`
- `reports/icu_readmit/markovsim_selected_causal/markovsim_one_step_test.json`
- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`
- `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`
- `reports/icu_readmit/offline_selected/step_17_eval_results.json`
- `reports/icu_readmit/offline_selected/step_17_action_stats.json`

Important caveat:

- The offline selected outputs are usable as evidence of what is currently saved, but they are not a full final OPE benchmark. They come from a reduced run limited to 400 stays per split.

### A7. Outputs that are exploratory, superseded, or should not be used

`Confirmed from code/output`

Do not use as final Results evidence:

- `THESIS_GUIDE.md` result numbers
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance_smoke/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness_smoke/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21d_graphs_with_confounders/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21e_drug_physiology_graphs/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21f_discharge_readmission_graphs/`
- `reports/icu_readmit/legacy/`
- `models/icu_readmit/legacy/`
- `models/icu_readmit/caresim-20260402T121057Z-1-001.zip`

### A8. Naming mismatch that matters

`Confirmed from code/output`

Current scripts write newer names:

- `step_13a_summary.json`
- `step_13b_markovsim_summary.json`
- `step_14_eval_results.json`
- `step_14_action_stats.json`

But the populated saved artifacts on disk still use older names:

- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`
- `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`
- `reports/icu_readmit/offline_selected/step_17_eval_results.json`
- `reports/icu_readmit/offline_selected/step_17_action_stats.json`

Conclusion:

- cite the older-named files as the actual populated results
- mention the naming mismatch under limitations / reproducibility notes

---

## B. Main Results Objects

| Result name | Generator | Output file(s) | Main text or appendix | Classification | Status |
|---|---|---|---|---|---|
| Final selected replay cohort and interface | `step_10a_rl_preprocess_selected.py` | `data/processed/icu_readmit/rl_dataset_selected.parquet`, `static_context_selected.parquet`, `scaler_params_selected.json` | Main text | Core baseline setup | Confirmed |
| Selection justification | `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance/variable_selection.json`, `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness/step_21b_transition_responsiveness.json`, `reports/icu_readmit/step_09_state_action_selection/*.json` | JSON summaries | Appendix or brief transition in main text | Diagnostic / justification | Confirmed |
| Severity surrogate | `step_10b_train_selected_severity_surrogate.py` | `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`, `severity_surrogate_coefficients.csv` | Main text short summary; coefficients appendix | Support model | Confirmed |
| Terminal readmission model | `step_10c_train_selected_terminal_readmit.py` | `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_metrics.json`, `terminal_readmit_selected_feature_importance.csv` | Main text short summary; feature importance appendix | Support model | Confirmed |
| CARE-Sim one-step fidelity | `step_12a_caresim_evaluate.py` | `caresim_one_step_val.json`, `caresim_one_step_test.json` | Main text | Core model result | Confirmed |
| CARE-Sim rollout / counterfactual diagnostics | `step_12a_caresim_evaluate.py` | `caresim_rollout_*.json`, `caresim_counterfactual_val.csv` | Appendix only | Diagnostic | Confirmed, but settings are mismatched |
| MarkovSim one-step fidelity | `step_12b_markovsim_evaluate.py` | `markovsim_one_step_val.json`, `markovsim_one_step_test.json` | Main text | Baseline simulator result | Confirmed |
| MarkovSim rollout / counterfactual diagnostics | `step_12b_markovsim_evaluate.py` | `markovsim_rollout_*.json`, `markovsim_counterfactual_val.csv` | Appendix mostly | Diagnostic | Confirmed |
| CARE-Sim control comparison | `step_13a_caresim_control.py` | `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json` plus diagnostics CSV/JSON | Main text | Core control result | Confirmed |
| MarkovSim control comparison | `step_13b_markovsim_control.py` | `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json` plus diagnostics CSV/JSON | Main text, but caveated | Baseline / extension | Confirmed |
| Offline held-out comparison | `step_14_offline_selected.py` | `reports/icu_readmit/offline_selected/step_17_eval_results.json`, `step_17_action_stats.json` | Main text only if labeled provisional; otherwise appendix | Robustness / real-data check | Confirmed, but reduced run only |

---

## C. Exact Empirical Results

## C1. Final sample and selected interface

`Confirmed from code/output`

### Final cohort facts

- `data/processed/icu_readmit/ICUdataset.csv`
  - rows: `1,500,857`
  - ICU stays: `61,771`
  - stay-level `readmit_30d`: `0.20692557996470837`
- `data/processed/icu_readmit/rl_dataset_selected.parquet`
  - rows: `1,500,857`
  - ICU stays: `61,771`
  - terminal rows: `61,771`
  - observed action IDs: `32`

### Split facts

- stay counts:
  - train: `43,239`
  - val: `9,265`
  - test: `9,267`
- row counts:
  - train: `1,057,632`
  - val: `222,408`
  - test: `220,817`
- stay-level readmission rate on terminal rows:
  - train: `0.2064571336062351`
  - val: `0.2042093901780896`
  - test: `0.21182691270098197`

### Longitudinal structure facts

- mean stay length: `24.297113532240047` four-hour blocs
- median stay length: `14`
- p25: `9`
- p75: `25`
- p90: `49`
- p95: `77`
- max: `1359`

### Final state and action definitions

From `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

- dynamic states: `Hb`, `BUN`, `Creatinine`, `Phosphate`, `HR`, `Chloride`
- static states: `age`, `charlson_score`, `prior_ed_visits_6m`
- binary actions:
  - `vasopressor_b`
  - `ivfluid_b`
  - `antibiotic_b`
  - `diuretic_b`
  - `mechvent_b`
- action encoding weights:
  - vasopressor `1`
  - ivfluid `2`
  - antibiotic `4`
  - diuretic `8`
  - mechvent `16`

### Observed treatment frequency facts

Row-level prevalence:

- `vasopressor_b`: `0.2015375215626805`
- `ivfluid_b`: `0.6498886969244905`
- `antibiotic_b`: `0.19106350571706698`
- `diuretic_b`: `0.08424720009967639`
- `mechvent_b`: `0.3564823297622625`

Stay-level ever-treated prevalence:

- vasopressor: `0.3013064382962879`
- ivfluid: `0.8544786388434702`
- antibiotic: `0.561169480824335`
- diuretic: `0.32806656845445276`
- mechvent: `0.46290330413948294`

Most common action IDs:

- `0`: `367,621`
- `2`: `337,143`
- `18`: `145,084`
- `19`: `110,429`
- `16`: `89,990`
- `6`: `89,018`
- `22`: `62,299`
- `23`: `59,299`
- `3`: `51,937`
- `10`: `25,898`

Useful decodings:

- `0` = none
- `2` = ivfluid
- `18` = ivfluid + mechvent
- `19` = vasopressor + ivfluid + mechvent
- `23` = vasopressor + ivfluid + antibiotic + mechvent

### Interpretation

`Inferred from surrounding context`

This is a large longitudinal ICU decision dataset with a moderately imbalanced but non-rare 30-day readmission outcome and a highly sparse-but-not-empty 32-action treatment space. The empirical action distribution is concentrated in a few combinations, which matters when interpreting learned policy behavior and offline evaluation support.

## C2. Selection-stage evidence that justifies the final selected set

`Confirmed from code/output`

### Legacy Step 21a variable relevance

From `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance/variable_selection.json`:

- time-varying branch:
  - val AUC: `0.6182243461049692`
  - features modeled: `90`
  - top variables include: `Platelets_count`, `WBC_count`, `Hb`, `HR`, `PT`, `PTT`, `BUN`, `Glucose`, `Creatinine`
- static branch:
  - val AUC: `0.6961403295333831`
  - features modeled: `31`
  - top variables include: `age`, `charlson_score`, `discharge_disposition`, `race`, `prior_ed_visits_6m`
- action branch:
  - val AUC: `0.5928141297382727`
  - features modeled: `22`

### Legacy Step 21b action responsiveness

From `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness/step_21b_transition_responsiveness.json`:

- `action_share_threshold`: `0.05`
- features modeled: `85`
- responsive features: `24`

Top action-responsive variables:

- `Hb`: `action_share 0.1168`, `R^2 0.9607`
- `Ht`: `action_share 0.0969`
- `MeanBP`: `action_share 0.0903`
- `Glucose`: `action_share 0.0889`
- `mechvent`: `AUC 0.9946`, `action_share 0.0881`
- `Chloride`: `action_share 0.0684`, `R^2 0.9561`

### Robust causal-state outputs

From `reports/icu_readmit/step_09_state_action_selection/random_stability_summary.json`:

- `last_Hb`: `freq_definite 0.9712`
- `last_Ht`: `0.9624`
- `last_BUN`: `0.9083`
- `last_input_total`: `0.8870`
- `last_Phosphate`: `0.5573`
- `last_HR`: `0.5414`
- `last_PT`: `0.5385`
- `last_Creatinine`: `0.5175`

From `reports/icu_readmit/step_09_state_action_selection/action_state_summary_robust.json`:

- `diuretic -> BUN`: frequency `1.0`
- `diuretic -> Chloride`: frequency `1.0`
- `ivfluid -> Phosphate`: frequency `1.0`
- `mechvent -> HR`: frequency `1.0`
- `vasopressor -> Hb`: frequency `1.0`

### Interpretation

`Inferred from surrounding context`

The step 09 outputs support the final selected set, but the actual canonical final state/action interface is what step 10a hard-codes and writes into `rl_dataset_selected.parquet`. Use step 09 as justification, not as the final definition.

## C3. Support model results used in reward design

### Severity surrogate

`Confirmed from code/output`

From `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`:

- train:
  - `n_rows 1,057,632`
  - `MAE 1.862747851755136`
  - `RMSE 2.338395111760472`
  - `R^2 0.27908079511608475`
  - `Spearman 0.5119421671931618`
- val:
  - `n_rows 222,408`
  - `MAE 1.8578440600240316`
  - `RMSE 2.336196964241669`
  - `R^2 0.2788672744501649`
  - `Spearman 0.5166161912287539`
- test:
  - `n_rows 220,817`
  - `MAE 1.8366567447965123`
  - `RMSE 2.297552390264984`
  - `R^2 0.2732469164707003`
  - `Spearman 0.5008183032555072`

Coefficients from `severity_surrogate_coefficients.csv`:

- `Creatinine 0.903154`
- `BUN 0.545850`
- `Hb -0.434151`
- `HR 0.205414`
- `Phosphate -0.092649`
- `Chloride 0.065115`

Interpretation:

- strength: weak-to-moderate
- substantive use: reward shaping support model, not a headline predictive result

### Terminal readmission model

`Confirmed from code/output`

From `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_metrics.json`:

- train:
  - `n_rows 43,239`
  - `readmit_rate 0.2064571336062351`
  - `AUC 0.689479099312386`
  - `AUPRC 0.3949748411997885`
  - `Brier 0.1504892485693187`
  - `logloss 0.4721699834673634`
- val:
  - `n_rows 9,265`
  - `readmit_rate 0.2042093901780896`
  - `AUC 0.6421542918866592`
  - `AUPRC 0.33047743420292663`
  - `Brier 0.15444788786168992`
  - `logloss 0.4834697603925375`
- test:
  - `n_rows 9,267`
  - `readmit_rate 0.21182691270098197`
  - `AUC 0.6489203467879763`
  - `AUPRC 0.3476652057423911`
  - `Brier 0.157861804664386`
  - `logloss 0.49126303283438655`

Top feature importances from `terminal_readmit_selected_feature_importance.csv`:

- `s_prior_ed_visits_6m 19834.155816`
- `s_charlson_score 6490.759527`
- `s_Hb 5603.740018`
- `s_BUN 5237.052079`
- `s_age 4995.568344`
- `s_HR 3963.660251`
- `s_Creatinine 3483.068304`
- `s_Chloride 2435.812656`
- `s_Phosphate 2178.563464`

Interpretation:

- strength: moderate
- substantive use: terminal reward source, not a standalone flagship predictive model

### Caveat on uncertainty reporting

`Confirmed from code/output`

No p-values, bootstrap confidence intervals, or saved formal hypothesis tests exist for these support models in the active artifact set.

## C4. Exact simulator fidelity results

### CARE-Sim training spec

`Confirmed from code/output`

From `models/icu_readmit/caresim_selected_causal/run_meta.json` and `ensemble_config.json`:

- training stays:
  - train `43,239`
  - val `9,265`
- ensemble size: `5`
- epochs: `30`
- learning rate: `0.001`
- batch size: `64`
- architecture:
  - `state_dim 9`
  - `action_dim 5`
  - `d_model 256`
  - `n_heads 8`
  - `n_layers 4`
  - `dropout 0.1`
  - `max_seq_len 80`
- settings:
  - causal constraints: `True`
  - freeze static context: `True`
  - time feature: `True`
  - reward head: `False`

### MarkovSim training spec

`Confirmed from code/output`

From `models/icu_readmit/markovsim_selected_causal/run_meta.json`:

- training rows: `1,057,632`
- feature dimension: `14`
- `transition_train_mse 0.10653316229581833`
- `terminal_train_accuracy 0.5478881123112765`
- config:
  - `ridge_alpha 1.0`
  - `terminal_c 1.0`
  - `max_iter 1000`

### CARE-Sim one-step results

`Confirmed from code/output`

From `reports/icu_readmit/caresim_selected_causal/caresim_one_step_val.json`:

- `n_stays 9,265`
- `n_rows 199,258`
- `n_nonterminal_rows 189,993`
- `next_state_mse 0.06783302429281862`
- `reward_mae 1.1143156878210705`
- `terminal_accuracy 0.9536279597306005`
- `mean_uncertainty 0.020686472240833146`

Per-feature val MSE:

- `Hb 0.07409730777587897`
- `BUN 0.05265508101027139`
- `Creatinine 0.04824511931350828`
- `Phosphate 0.145175731375099`
- `HR 0.20983677378292276`
- `Chloride 0.08048720537768438`

From `reports/icu_readmit/caresim_selected_causal/caresim_one_step_test.json`:

- `n_stays 9,267`
- `n_rows 198,761`
- `n_nonterminal_rows 189,494`
- `next_state_mse 0.0680623747749937`
- `reward_mae 1.103205339174348`
- `terminal_accuracy 0.9535925055720187`
- `mean_uncertainty 0.020657865344905584`

Per-feature test MSE:

- `Hb 0.07433717672012818`
- `BUN 0.053893483025873865`
- `Creatinine 0.04737507387877342`
- `Phosphate 0.14411130369559128`
- `HR 0.2125451794389697`
- `Chloride 0.08029915621560386`

### MarkovSim one-step results

`Confirmed from code/output`

From `reports/icu_readmit/markovsim_selected_causal/markovsim_one_step_val.json`:

- `n_rows 222,408`
- `next_state_mse 0.07206787914037704`
- `reward_mae 0.05209772661328316`
- `terminal_accuracy 0.551841660371929`
- `terminal_brier 0.20914970338344574`
- `mean_uncertainty 0.20878218114376068`

Per-feature val MSE:

- `Hb 0.07744856923818588`
- `BUN 0.056075941771268845`
- `Creatinine 0.05082378536462784`
- `Phosphate 0.16331535577774048`
- `HR 0.2272515594959259`
- `Chloride 0.07369557023048401`

From `reports/icu_readmit/markovsim_selected_causal/markovsim_one_step_test.json`:

- `n_rows 220,817`
- `next_state_mse 0.07180938124656677`
- `reward_mae 0.05172107741236687`
- `terminal_accuracy 0.5428793978724464`
- `terminal_brier 0.21431775391101837`
- `mean_uncertainty 0.2087821662425995`

Per-feature test MSE:

- `Hb 0.07756689190864563`
- `BUN 0.05648665875196457`
- `Creatinine 0.048990827053785324`
- `Phosphate 0.1585925668478012`
- `HR 0.23161481320858002`
- `Chloride 0.0730329379439354`

### Direct simulator comparison

`Confirmed from code/output`

CARE-Sim vs MarkovSim:

- val next-state MSE difference: `0.0042348548475584225`
- test next-state MSE difference: `0.0037470064715730694`
- relative improvement in next-state MSE:
  - val: about `5.88%` lower
  - test: about `5.22%` lower
- terminal accuracy difference:
  - val: `+0.4017862993586715`
  - test: `+0.41071310769957226`

Clean interpretation:

- CARE-Sim is clearly better on the cleanest simulator-quality metric available in the repo: one-step held-out state prediction.
- CARE-Sim is also much better at the saved done/terminal prediction task.

### Rollout comparability caveat

`Confirmed from code/output`

Saved CARE-Sim evaluation is not aligned with saved MarkovSim evaluation:

- CARE-Sim summary:
  - `rollout_patients 20`
  - `counterfactual_patients 3`
  - `reward_source severity`
  - `severity_mode handcrafted`
  - `use_terminal_readmit_reward false`
- CARE-Sim eval log:
  - `logs/step_15_caresim_eval.log` starts with `smoke=True`
- MarkovSim summary:
  - `rollout_patients 200`
  - `counterfactual_patients 10`
  - `use_terminal_readmit_reward true`
  - `smoke false`

Therefore:

- one-step CARE-Sim vs MarkovSim is the defensible main-text simulator comparison
- rollout reward comparisons are not apples-to-apples in the current saved artifact set

## C5. Exact control results inside CARE-Sim

`Confirmed from code/output`

From `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`:

Config used:

- `history_len 5`
- `observation_window 5`
- `rollout_steps 5`
- `planner_horizon 3`
- `uncertainty_penalty 0.25`
- `uncertainty_threshold 1.0`
- `use_severity_reward true`
- `severity_mode handcrafted`
- `use_terminal_readmit_reward true`
- `terminal_reward_scale 15.0`

Validation mean discounted return:

- `ddqn 8.692149344347015`
- `planner 8.582858349683406`
- `random 8.592901410564357`
- `repeat_last 8.680327229551217`

Validation additional DDQN facts:

- episodes: `100`
- mean raw reward total: `9.069316809277517`
- mean uncertainty: `0.01721114821173251`
- mean rollout steps: `5.0`
- std discounted return: `2.3205909611509123`
- p25: `7.89227122788689`
- p50: `9.406750460154157`
- p75: `10.153921504556818`

Test mean discounted return:

- `ddqn 8.836255628219314`
- `planner 8.78556112665876`
- `random 8.789412113939656`
- `repeat_last 8.806716251419907`

Pairwise diagnostics from `step_16_diagnostics_test.json`:

- `ddqn_minus_planner mean_diff 0.05069450156055609`, `win_rate 0.58`
- `ddqn_minus_random mean_diff 0.046843514279660194`, `win_rate 0.58`
- `ddqn_minus_repeat_last mean_diff 0.02953937679940732`, `win_rate 0.52`

Action patterns:

- DDQN uses action IDs such as `2`, `18`, `23`, `22`
- planner uses a broader action set than repeat-last
- repeat-last is heavily concentrated in actions `0` and `2`

Interpretation:

- strength: weak / mixed
- important substantive comparison: DDQN is numerically best, but only marginally better than planner, random, and especially repeat-last
- what not to overclaim: do not describe this as a large control gain

## C6. Exact control results inside MarkovSim

`Confirmed from code/output`

From `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`:

Config used:

- same main reward settings as CARE-Sim control
- `use_severity_reward true`
- `severity_mode handcrafted`
- `use_terminal_readmit_reward true`
- `terminal_reward_scale 15.0`

Validation mean discounted return:

- `ddqn 8.976655859579582`
- `planner 8.973150204867125`
- `random 8.609596448199428`
- `repeat_last 8.783070112458793`

Validation additional facts:

- DDQN mean rollout steps: `1.22`
- planner mean rollout steps: `1.0`
- DDQN mean uncertainty: `0.20878209173679352`

Test mean discounted return:

- `ddqn 9.101420106375041`
- `planner 9.131546391099691`
- `random 8.812488372628884`
- `repeat_last 8.97896232580588`

Test mean rollout steps:

- `ddqn 1.29`
- `planner 1.0`
- `random 3.56`
- `repeat_last 2.91`

Pairwise diagnostics from test:

- `ddqn_minus_planner -0.030126284724650114`, `win_rate 0.26`
- `ddqn_minus_random 0.2889317337461568`, `win_rate 0.75`
- `ddqn_minus_repeat_last 0.12245778056916246`, `win_rate 0.52`
- `planner_minus_random 0.31905801847080695`, `win_rate 0.79`

Action pattern:

- planner uses exactly one action: `0` on all validation and test episodes

Interpretation:

- numerical ranking looks competitive
- substantive strength is weak because the environment effectively terminates after about one step and the planner's best policy is always no action
- what not to overclaim: do not present this as persuasive evidence that MarkovSim is a better decision model than CARE-Sim

## C7. Exact offline held-out comparison

`Confirmed from code/output`

From `reports/icu_readmit/offline_selected/step_17_eval_results.json`:

Metadata:

- `obs_dim 70`
- `window_len 5`
- comparison policies:
  - `caresim_ddqn`
  - `markovsim_ddqn`
  - `offline_ddqn`
- reward config:
  - `severity_mode handcrafted`
  - terminal reward model on
  - `terminal_reward_scale 15.0`

Logged policy empirical trajectory return:

- val:
  - mean `7.2400803565979`
  - std `2.697385787963867`
  - trajectories `400`
- test:
  - mean `7.29922342300415`
  - std `2.673602819442749`
  - trajectories `400`

Doubly robust OPE quick-check:

- val:
  - `offline_ddqn mean 0.0625932763788066`, `std 0.26190146765186245`, `n_valid 398`
  - `caresim_ddqn mean -2.460073740973099`, `std 8.067124905759313`, `n_valid 338`
  - `markovsim_ddqn mean -0.8717816211236864`, `std 3.9077578330479037`, `n_valid 394`
- test:
  - `offline_ddqn mean 0.04820438391557281`, `std 0.25821966464129387`, `n_valid 400`
  - `caresim_ddqn mean -2.2666017163189287`, `std 7.597194553842551`, `n_valid 340`
  - `markovsim_ddqn mean -0.49651748054280365`, `std 3.0002127003139956`, `n_valid 394`

Action-agreement stats from `step_17_action_stats.json`:

Validation exact agreement with logged actions:

- `caresim_ddqn 24.279603782080144%`
- `offline_ddqn 17.323277802791534%`
- `markovsim_ddqn 7.7104907699234575%`

Test exact agreement with logged actions:

- `caresim_ddqn 30.063326297078103%`
- `offline_ddqn 13.254082879680034%`
- `markovsim_ddqn 7.154760582157538%`

Important ranking from saved OPE outputs:

- val: `offline_ddqn > markovsim_ddqn > caresim_ddqn`
- test: `offline_ddqn > markovsim_ddqn > caresim_ddqn`

### Critical caveat

`Confirmed from code/output`

`logs/step_17_offline_selected.log` shows:

- `Split=train limited to 400 stays`
- `Split=val limited to 400 stays`
- `Split=test limited to 400 stays`

The same log also shows the reduced support-model training used in this saved run.

Conclusion:

- these are real saved results
- they are not the intended full-run final OPE benchmark from the notebook configuration
- they should be described as provisional, reduced-run, or robustness-style evidence

---

## D. Figure and Table Walkthrough

## D1. Likely Results chapter order

### 1. Final selected cohort and decision problem

- proposed subsection: `Results 1. Final selected cohort and action space`
- title: `Selected replay cohort and 9-state/5-action intervention interface`
- source files:
  - `data/processed/icu_readmit/rl_dataset_selected.parquet`
  - `data/processed/icu_readmit/static_context_selected.parquet`
  - `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
- reader should notice first:
  - `61,771` stays
  - readmission rate about `20.7%`
  - final action space has `32` combinations
- second-order patterns:
  - action distribution is highly concentrated in a few combos
  - ivfluids and mechvent dominate observed action mass
- do not overclaim:
  - this is not the full ICU feature universe, only the selected RL interface
- connection to thesis argument:
  - defines the exact empirical decision problem used for all downstream results

### 2. Reward support models

- proposed subsection: `Results 2. Support models used in reward construction`
- title: `Selected severity surrogate and terminal readmission model`
- source files:
  - `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
  - `reports/icu_readmit/severity_selected/severity_surrogate_coefficients.csv`
  - `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_metrics.json`
  - `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_feature_importance.csv`
- reader should notice first:
  - severity surrogate is only moderate
  - terminal readmit model is also moderate
- second-order patterns:
  - creatinine and BUN dominate surrogate severity
  - prior ED visits and comorbidity dominate terminal readmission risk
- do not overclaim:
  - these are support models for reward, not the thesis' main predictive contribution
- connection to thesis argument:
  - explains the reward signal used in simulator control and offline comparison

### 3. Simulator fidelity

- proposed subsection: `Results 3. Held-out simulator fidelity`
- title: `CARE-Sim versus MarkovSim on one-step held-out prediction`
- source files:
  - `reports/icu_readmit/caresim_selected_causal/caresim_one_step_test.json`
  - `reports/icu_readmit/markovsim_selected_causal/markovsim_one_step_test.json`
- reader should notice first:
  - CARE-Sim test next-state MSE `0.0681` vs MarkovSim `0.0718`
  - CARE-Sim terminal accuracy `0.9536` vs MarkovSim `0.5429`
- second-order patterns:
  - both simulators struggle most on `HR` and `Phosphate`
  - MarkovSim has much higher mean uncertainty
- do not overclaim:
  - rollout outputs are not saved under matching evaluation settings
- connection to thesis argument:
  - supports choosing CARE-Sim as the preferred rich world model

### 4. Policy results inside CARE-Sim

- proposed subsection: `Results 4. Policy comparison inside CARE-Sim`
- title: `CARE-Sim DDQN versus planner and simple baselines`
- source files:
  - `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`
  - `reports/icu_readmit/caresim_control_selected_causal/step_16_diagnostics_test.json`
- reader should notice first:
  - DDQN is best on test, but only slightly
- second-order patterns:
  - repeat-last is almost tied with DDQN
  - improvement margins are very small
- do not overclaim:
  - this is weak or mixed evidence of control improvement
- connection to thesis argument:
  - better simulator fidelity does not automatically produce large policy gains

### 5. Policy results inside MarkovSim

- proposed subsection: `Results 5. Policy comparison inside MarkovSim`
- title: `MarkovSim control results and degenerate rollout behavior`
- source files:
  - `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`
  - `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_diagnostics_test.json`
- reader should notice first:
  - numeric returns are slightly higher than CARE-Sim
- second-order patterns:
  - planner always chooses action `0`
  - episodes usually end after about one step
- do not overclaim:
  - this does not support a strong claim that MarkovSim is the superior simulator
- connection to thesis argument:
  - clarifies why MarkovSim is a useful baseline but not the preferred environment

### 6. Held-out offline comparison

- proposed subsection: `Results 6. Held-out offline policy comparison`
- title: `Reduced doubly robust comparison on held-out logged data`
- source files:
  - `reports/icu_readmit/offline_selected/step_17_eval_results.json`
  - `reports/icu_readmit/offline_selected/step_17_action_stats.json`
  - `logs/step_17_offline_selected.log`
- reader should notice first:
  - saved reduced-run OPE ranks `offline_ddqn` first on both val and test
- second-order patterns:
  - CARE-Sim DDQN has the highest action agreement with clinicians, but the worst saved DR value
- do not overclaim:
  - the saved run is not the intended full benchmark
- connection to thesis argument:
  - adds a real-data check on top of simulator-only control results

### 7. Appendix diagnostics

- proposed subsection: `Appendix`
- title: `Selection diagnostics, per-variable errors, counterfactual sweeps, action traces`
- source files:
  - `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance/`
  - `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness/`
  - `reports/icu_readmit/step_09_state_action_selection/`
  - `reports/icu_readmit/caresim_selected_causal/caresim_rollout_*.json`
  - `reports/icu_readmit/markovsim_selected_causal/markovsim_rollout_*.json`
  - `reports/icu_readmit/caresim_selected_causal/caresim_counterfactual_val.csv`
  - `reports/icu_readmit/markovsim_selected_causal/markovsim_counterfactual_val.csv`
  - `reports/icu_readmit/caresim_control_selected_causal/step_16_policy_traces_*.csv`
  - `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_policy_traces_*.csv`

---

## E. Baseline vs Extensions

### E1. Core baseline results

Definitely main Results:

- final selected cohort and state/action interface from `step_10a`
- support-model performance from `step_10b` and `step_10c`
- CARE-Sim vs MarkovSim one-step held-out fidelity
- CARE-Sim control comparison
- MarkovSim control comparison as explicit simpler baseline

### E2. Mechanism / heterogeneity / supporting diagnostics

Probably appendix or short transition material:

- step 09 selection rankings and robust action-state links
- coefficient table for severity surrogate
- feature-importance table for terminal readmit model
- per-feature one-step MSE tables

### E3. Robustness checks

Main text only if clearly labeled:

- reduced held-out offline OPE comparison

### E4. Failed or inconclusive tests

Should be described honestly:

- CARE-Sim policy gains over planner / repeat-last are small
- MarkovSim control results are hard to interpret because the environment often ends after one step
- saved offline OPE is reduced-run and not final-full-benchmark quality

### E5. Not worth discussing

- `THESIS_GUIDE.md` result sections
- `step_21a_outcome_relevance_smoke`
- `step_09b_smoke`
- legacy `step_21c` to `step_21f` branches
- `reports/icu_readmit/legacy`
- `models/icu_readmit/legacy`
- archived zip model bundle

---

## F. Final Model Choice

### F1. Preferred simulator specification

`Confirmed from code/output`

Preferred rich simulator in the active codebase:

- `models/icu_readmit/caresim_selected_causal/`

Why it is preferred:

- it is the active selected-causal branch
- it is the richer transformer world model
- it outperforms MarkovSim on the cleanest held-out fidelity metrics

Relevant configuration:

- 5-member ensemble
- transformer
- `state_dim 9`
- `action_dim 5`
- causal constraints on
- static context frozen
- reward head off

### F2. Preferred baseline specification

`Confirmed from code/output`

Explicit simpler baseline:

- `models/icu_readmit/markovsim_selected_causal/`

Why retained:

- methods handoff explicitly treats MarkovSim as the final simpler baseline simulator, not just an exploratory branch

### F3. Preferred control specification

`Confirmed from code/output`

Saved step 13 control reports use:

- handcrafted severity reward
- terminal readmission reward on
- `terminal_reward_scale 15.0`
- `history_len 5`
- `observation_window 5`
- `rollout_steps 5`
- `planner_horizon 3`

### F4. Alternatives tried or present but not emphasized

`Confirmed from code/output` for existence  
`Inferred from surrounding context` for the judgment that they are not final

- full-dataset / tier2 branches
- surrogate-severity reward mode
- CARE-Sim reward-head variants
- step `09c*` selection variants
- legacy older step-number branches discussed in older docs

### F5. Mismatch between code and likely thesis assumptions

#### CARE-Sim selected-state indexing bug

`Confirmed from code/output`

In `src/careai/icu_readmit/caresim/model.py`:

- `DYNAMIC_STATE_IDX = (0, 1, 2, 3, 4)`
- `STATIC_STATE_IDX = (5, 6, 7)`

But the selected state order is:

1. `Hb`
2. `BUN`
3. `Creatinine`
4. `Phosphate`
5. `HR`
6. `Chloride`
7. `age`
8. `charlson_score`
9. `prior_ed_visits_6m`

Implication:

- `Chloride` at index `5` is treated as static by CARE-Sim when `freeze_static_context=True`
- it is copied through rather than predicted dynamically
- MarkovSim does not have this bug; its dynamic/static indexing is correct

This should be mentioned if the thesis describes the final CARE-Sim implementation in detail.

#### CARE-Sim step 12 reward mismatch

`Confirmed from code/output`

Saved CARE-Sim evaluation does not use the same reward specification as saved step 13 control:

- saved CARE-Sim eval: terminal reward off
- saved CARE-Sim control: terminal reward on

#### Offline step 14 benchmark mismatch

`Confirmed from code/output`

Notebook intent:

- full DDQN training
- full support-model training
- no small stay cap

Saved output:

- only 400 stays per split
- reduced support-model training

---

## G. Uncertainties and Open Issues

| Issue | Confirmed fact | Why it matters |
|---|---|---|
| No `AGENTS.md` | Repo search found none | No extra project-agent instructions exist beyond current docs |
| CARE-Sim indexing bug | `src/careai/icu_readmit/caresim/model.py` freezes `Chloride` as static | Could affect the validity of selected CARE-Sim results |
| CARE-Sim step 12 saved eval is smoke-like | `caresim_summary.json` and log show small rollout settings and terminal reward off | Rollout and reward metrics are not directly comparable to MarkovSim |
| Offline OPE saved output is reduced-run | `logs/step_17_offline_selected.log` limits all splits to 400 stays | Offline comparison is provisional, not full-thesis-ready |
| Step 13/14 filenames do not match current scripts | Current scripts write `step_13*` and `step_14*`, but saved outputs are `step_16*` and `step_17*` | Can confuse a writer if not noted explicitly |
| Offline training version ambiguity | `models/icu_readmit/offline_selected/ddqn/` has older full checkpoints and later partial rerun artifacts | Need human confirmation if the saved OPE used the intended trained policy snapshot |
| No polished figure files | Active report folders mainly contain JSON/CSV, not `.png/.pdf/.svg` | Thesis figures must be made from these raw outputs |
| `THESIS_GUIDE.md` is outdated for final Results | Contains old branches and old result sets | Do not import numbers from it into the final Results chapter |
| No formal uncertainty estimates for main comparisons | No saved p-values/bootstrap intervals for simulator/control/OPE comparisons | Writer should avoid implying formal statistical significance |

### Human confirmation points most worth checking

1. Whether there is a later full selected-causal CARE-Sim evaluation not stored here
2. Whether there is a full step 14 OPE run stored elsewhere
3. Whether the CARE-Sim indexing bug should be fixed and rerun before thesis lock
4. Whether any external figure exports already exist outside the repo

---

## H. Results facts ChatGPT should use

- Final selected replay dataset: `61,771` ICU stays and `1,500,857` four-hour rows.
- Final stay-level 30-day readmission rate: about `20.69%`.
- Split stays: `43,239` train, `9,265` val, `9,267` test.
- Final RL interface:
  - dynamic states: `Hb`, `BUN`, `Creatinine`, `Phosphate`, `HR`, `Chloride`
  - static context: `age`, `charlson_score`, `prior_ed_visits_6m`
  - actions: `vasopressor`, `ivfluid`, `antibiotic`, `diuretic`, `mechvent`
  - action space size: `32`
- Most common actions are no action (`0`), ivfluid (`2`), and ivfluid+mechvent (`18`).
- Severity surrogate is only moderate:
  - test `R^2 0.2732469164707003`
  - test `MAE 1.8366567447965123`
  - test `RMSE 2.297552390264984`
- Terminal readmission model is moderate:
  - test `AUC 0.6489203467879763`
  - test `AUPRC 0.3476652057423911`
  - test `Brier 0.157861804664386`
- Cleanest simulator comparison:
  - CARE-Sim test next-state MSE `0.0680623747749937`
  - MarkovSim test next-state MSE `0.07180938124656677`
  - CARE-Sim test terminal accuracy `0.9535925055720187`
  - MarkovSim test terminal accuracy `0.5428793978724464`
- Main simulator-fidelity takeaway:
  - CARE-Sim clearly beats MarkovSim on one-step held-out prediction.
- CARE-Sim control result:
  - test mean discounted return
    - DDQN `8.836255628219314`
    - planner `8.78556112665876`
    - random `8.789412113939656`
    - repeat-last `8.806716251419907`
- CARE-Sim control interpretation:
  - DDQN is best numerically, but only marginally.
- MarkovSim control result:
  - test mean discounted return
    - DDQN `9.101420106375041`
    - planner `9.131546391099691`
    - random `8.812488372628884`
    - repeat-last `8.97896232580588`
- MarkovSim control caveat:
  - planner always selects action `0`
  - mean rollout length is about one step
  - do not overinterpret slightly higher return numbers
- Saved offline held-out comparison is reduced-run only:
  - each split limited to `400` stays
  - not the notebook-intended final full benchmark
- Saved offline OPE ranking:
  - validation: `offline_ddqn > markovsim_ddqn > caresim_ddqn`
  - test: `offline_ddqn > markovsim_ddqn > caresim_ddqn`
- Exact saved test DR means:
  - `offline_ddqn 0.04820438391557281`
  - `markovsim_ddqn -0.49651748054280365`
  - `caresim_ddqn -2.2666017163189287`
- Important caveats that must be preserved:
  - CARE-Sim step 12 saved evaluation is a smoke-sized / mismatched run
  - offline step 14 saved evaluation is reduced-run, not full-run
  - CARE-Sim selected implementation currently treats `Chloride` as frozen static context due to a hard-coded index bug
  - active saved step 13/14 outputs still use older `step_16` / `step_17` filenames

---

## Bottom-line guidance for thesis writing

Most defensible main Results narrative:

1. define the final selected cohort and state/action interface
2. report the moderate support-model performance used to construct reward
3. show that CARE-Sim is the stronger simulator on one-step held-out fidelity
4. show that simulator-control gains are small and mixed rather than dramatic
5. present MarkovSim as a simpler baseline with degeneracy caveats
6. present offline OPE only as a reduced/provisional held-out check unless a fuller run exists elsewhere

What should not be said:

- do not say the offline benchmark is fully final unless a full run is found
- do not say CARE-Sim and MarkovSim rollout evaluations are directly comparable in the current saved artifacts
- do not say policy improvements are large
- do not import numbers from `THESIS_GUIDE.md`
