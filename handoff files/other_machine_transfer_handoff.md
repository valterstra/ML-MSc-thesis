# CareAI Transfer Handoff For Other Machine

This document explains how to reconstruct the working `CareAI` environment on another computer using the materials that now exist in three places:

1. GitHub `main`
2. the existing Drive export zip from the Colab/Drive `CareAI` folder
3. the new local-only transfer zip created on this machine

This is not a general explanation of the ICU-readmit pipeline. It is a practical file-placement and recovery guide so another agent can take the transferred materials and restore a usable working project tree.

## Goal

After following this handoff, the other machine should have:

- the current repo code, docs, notebooks, and scripts
- the existing Drive artifacts and trained models
- the missing local-only processed artifacts that were not present on Drive

This should be enough to continue the active ICU-readmit work, including the new non-causal branch.

## Canonical Sources

Use these sources in this order:

1. **GitHub repo**
   - source of truth for code, notebooks, docs, and repo structure
2. **Drive CareAI export zip**
   - source of truth for many large artifacts already produced in Colab / Drive
3. **Local-only core transfer zip**
   - fills the important gaps that existed locally but were not in the Drive export

Do not treat the Drive zip as a full repo snapshot. It is not.

## Required Inputs

The other machine should have access to:

### A. GitHub repo
- repository: `git@github.com:valterstra/ML-MSc-thesis.git`
- branch: `main`
- important commit pushed from this machine:
  - `0b47de3`
  - message: `Add non-causal ICU readmit pipeline and evaluation updates`

### B. Existing Drive export zip
- filename:
  - `CareAI-20260422T155239Z-3-001.zip`

This zip came from the existing Drive `CareAI` folder and contains mostly:
- `data/`
- `models/`
- `reports/`
- `caresim_colab.zip`

It does **not** contain the full repo codebase.

### C. New local-only core transfer zip
- filename:
  - `CareAI_local_only_core_step8plus_20260422.zip`

This zip was created specifically to fill the local-only gaps that were missing from both GitHub and the Drive export.

## What GitHub Already Covers

GitHub `main` now contains the current working project code and structure, including:

- `src/`
- `scripts/`
- `notebooks/`
- `docs/`
- `handoff files/`
- `.gitignore`
- current step 09 / 10 / 11 / 12 non-causal code
- DAG-aware additions
- step 13 / 14 code updates

So the other machine should **always** start by cloning or pulling the repo.

## What The Drive Export Zip Covers

The Drive export zip provides a large artifact layer, including many of the previously produced runtime outputs:

- flat replay parquet files under `CareAI/data/`
- trained simulator and controller models under `CareAI/models/`
- evaluation outputs under `CareAI/reports/`
- `CareAI/caresim_colab.zip`

Important examples already present in that Drive export:

- `CareAI/data/rl_dataset_selected.parquet`
- `CareAI/data/rl_dataset_noncausal.parquet`
- `CareAI/models/icu_readmit/caresim_noncausal/best_model.pt`
- `CareAI/models/icu_readmit/caresim_noncausal/model.pt`
- `CareAI/models/icu_readmit/caresim_noncausal/train_config.json`
- `CareAI/models/icu_readmit/caresim_noncausal/train_metrics.json`
- selected causal CARE-Sim ensemble artifacts
- selected causal MarkovSim artifacts
- selected causal DAG-aware artifacts
- many selected-causal control and offline artifacts

But the Drive export was missing several step-8+ processed files and newer local-only artifacts.

## What The Local-Only Core Transfer Zip Covers

The local-only core transfer zip contains the important missing pieces that were present locally on this machine but absent from the Drive export and absent from GitHub.

Included paths:

- `data/processed/icu_readmit/ICUdataset.csv`
- `data/processed/icu_readmit/icu_cohort_summary.csv`
- `data/processed/icu_readmit/static_context_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/step_09_state_action_selection/`
- `data/processed/icu_readmit/step_09_noncausal_interface/`
- `data/processed/icu_readmit/step_10_noncausal/`
- `models/icu_readmit/severity_selected/`
- `reports/icu_readmit/caresim_noncausal_smoke/`
- `README_transfer.txt`

These are the main missing artifacts that matter for active work from step 8 onward.

## Important Thing Not Included In The Core Transfer Zip

The core transfer zip deliberately excludes:

- `data/interim/icu_readmit/`

Reason:
- it is very large
- it is the main rebuild / reproducibility tree
- it slowed transfer packaging substantially

This means the restored environment will be good for active continuation of the pipeline, but not fully equivalent to having every raw/intermediate rebuild artifact from step 8.

If later needed, `data/interim/icu_readmit/` should be transferred separately.

## Reconstruction Procedure On The Other Machine

The following sequence should be followed exactly.

### Step 1. Clone or update the GitHub repo

If the repo is not yet cloned:

```powershell
git clone git@github.com:valterstra/ML-MSc-thesis.git
```

Then go to the CareAI directory within that repo layout and update `main`:

```powershell
git checkout main
git pull origin main
```

Expected result:
- repo code and structure are current
- commit `0b47de3` should be present

### Step 2. Establish the repo root

The target repo root is the `CareAI` directory itself. All transferred files should end up relative to that root.

Examples of correct target locations:

- `CareAI/src/...`
- `CareAI/scripts/...`
- `CareAI/data/processed/...`
- `CareAI/models/...`
- `CareAI/reports/...`

Do not unpack any transfer zip into a sibling directory. They should be unpacked so that the internal relative paths land inside the existing repo root.

### Step 3. Restore the Drive export artifacts

Take:
- `CareAI-20260422T155239Z-3-001.zip`

Unpack its contents into the repo root in a way that preserves the relative `CareAI/...` paths.

Because the zip contains a top-level `CareAI/` folder, there are two acceptable methods:

#### Method A
Unzip to a temporary location first, then copy the contents of the extracted `CareAI/` directory into the repo root.

#### Method B
Unzip directly one level above the repo root and then merge carefully.

Preferred approach:
- extract to temp
- inspect
- copy only the contents of the inner `CareAI/` directory into the actual repo root

The goal is to merge:
- `data/`
- `models/`
- `reports/`
- `caresim_colab.zip`

into the cloned repo.

### Step 4. Restore the local-only core transfer zip

Take:
- `CareAI_local_only_core_step8plus_20260422.zip`

This zip is already structured with repo-relative paths. It should be unpacked directly at the repo root so the following paths land correctly:

- `data/processed/icu_readmit/...`
- `models/icu_readmit/severity_selected/...`
- `reports/icu_readmit/caresim_noncausal_smoke/...`

This step fills the key gaps that were not present in the Drive export.

### Step 5. Do not overwrite repo code from the artifact zips

The repo code should come from GitHub `main`, not from the Drive export.

The artifact zips are there to fill:
- `data/`
- `models/`
- `reports/`

They should not be treated as a replacement for:
- `src/`
- `scripts/`
- `notebooks/`
- `docs/`

## Expected Final File Layout

After the merge, the repo should have all of these classes of content:

### Repo code layer from GitHub
- `src/`
- `scripts/`
- `notebooks/`
- `docs/`
- `handoff files/`

### Artifact layer from Drive
- `data/rl_dataset_selected.parquet`
- `data/rl_dataset_noncausal.parquet`
- `models/icu_readmit/...`
- `reports/icu_readmit/...`
- `caresim_colab.zip`

### Processed data layer from local-only transfer
- `data/processed/icu_readmit/ICUdataset.csv`
- `data/processed/icu_readmit/static_context_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/step_09_noncausal_interface/...`
- `data/processed/icu_readmit/step_10_noncausal/...`

### Additional important local-only items
- `models/icu_readmit/severity_selected/...`
- `reports/icu_readmit/caresim_noncausal_smoke/...`

## Why These Extra Files Matter

### `ICUdataset.csv`
This is the broad processed ICU dataset used as the basis for step 8 onward and is important for understanding or rebuilding later preprocessing steps.

### `static_context_selected.parquet` and `scaler_params_selected.json`
These are part of the selected-causal processed branch and are useful for continuing or inspecting the older branch cleanly.

### `step_09_noncausal_interface/`
This captures the broad non-causal interface selection artifacts and makes the new branch inspectable rather than relying only on the flattened replay parquet.

### `step_10_noncausal/`
This contains the structured non-causal replay build, including static context and preprocessing metadata, not just the flat Drive-level `rl_dataset_noncausal.parquet`.

### `severity_selected/`
This fills an important gap in the selected-causal reward machinery. It was not in the Drive export and should be present if older selected-causal reward logic is revisited.

### `caresim_noncausal_smoke/`
This preserves the local non-causal evaluation smoke results and gives the next agent a known-good evaluation snapshot for the new simulator.

## Verification Checklist

After reconstruction, verify the following.

### Git / code verification

Run:

```powershell
git status
git log --oneline -n 3
```

Expected:
- on `main`
- commit `0b47de3` present

### File existence verification

Check these paths exist:

- `data/processed/icu_readmit/ICUdataset.csv`
- `data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_dataset.parquet`
- `data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet`
- `models/icu_readmit/caresim_noncausal/best_model.pt`
- `models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib`
- `reports/icu_readmit/caresim_noncausal_smoke/caresim_noncausal_summary.json`
- `scripts/icu_readmit/step_11a_caresim_train_noncausal.py`
- `scripts/icu_readmit/step_12a_caresim_evaluate_noncausal.py`
- `notebooks/step_11a_caresim_noncausal_colab.ipynb`
- `notebooks/step_12a_caresim_noncausal_colab.ipynb`

### Semantic verification

The other agent should confirm:

- GitHub provided the codebase
- Drive export provided most runtime artifacts
- local-only core transfer provided the missing step-8+ processed artifacts

## Known Remaining Gap

The main remaining gap after this procedure is:

- `data/interim/icu_readmit/`

If later work requires full rebuildability from raw/intermediate step 8 inputs, that directory should be transferred separately.

For normal continuation of the current active work, that gap is acceptable in the short term.

## Recommended Priority

If time is short, prioritize exactly this order:

1. GitHub repo on `main`
2. Drive export zip
3. local-only core transfer zip

That combination is the minimum practical setup for continuing current ICU-readmit work.

## Summary For The Next Agent

If another agent receives this handoff and the three inputs above, the correct interpretation is:

- GitHub is the source of code
- Drive zip is the source of many existing artifacts
- local-only core zip is the patch set for important missing processed files

The next agent should not try to infer the repo structure from the Drive zip alone. It should first restore the repo from GitHub and then merge the two artifact packages into that repo root.
