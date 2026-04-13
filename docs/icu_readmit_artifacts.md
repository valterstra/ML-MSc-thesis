# ICU Readmit Artifact Distribution

GitHub contains the active ICU-readmission code path, not the generated artifacts.

The missing pieces are the trained and derived files under:
- `data/processed/icu_readmit/`
- `models/icu_readmit/`
- `reports/icu_readmit/`

## Recommended Distribution Method

Use a single zip bundle containing the active artifact subset only.

Build it locally with:

```powershell
python scripts/package_icu_readmit_artifacts.py
```

This creates:
- `dist/icu_readmit_main_artifacts/`
- `dist/icu_readmit_main_artifacts.zip`

## What To Upload

Upload `dist/icu_readmit_main_artifacts.zip` to a shareable storage location:
- Google Drive
- OneDrive
- Zenodo
- Hugging Face Hub
- GitHub Release asset if the size is acceptable

## What The Recipient Does

1. Clone the `main` branch.
2. Download `icu_readmit_main_artifacts.zip`.
3. Extract it at the repository root.
4. Confirm the bundled `data/`, `models/`, and `reports/` paths now exist.
5. Run the active scripts and notebooks normally.

## Included Artifacts

Required runtime artifacts:
- `data/processed/icu_readmit/rl_dataset_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/static_context_selected.parquet`
- `models/icu_readmit/severity_selected/`
- `models/icu_readmit/terminal_readmit_selected/`
- `models/icu_readmit/caresim_selected_causal/`
- `models/icu_readmit/caresim_control_selected_causal/`
- `models/icu_readmit/markovsim_selected_causal/`
- `models/icu_readmit/markovsim_control_selected_causal/`
- `models/icu_readmit/offline_selected/`

Optional inspection artifacts:
- `data/processed/icu_readmit/ICUdataset.csv`
- `reports/icu_readmit/caresim_selected_causal/`
- `reports/icu_readmit/markovsim_selected_causal/`
- `reports/icu_readmit/caresim_control_selected_causal/`
- `reports/icu_readmit/markovsim_control_selected_causal/`
- `reports/icu_readmit/offline_selected/`

## Manifest

The bundle includes:
- `artifact_manifest.json`
- `README_ARTIFACTS.md`

Those files tell the recipient exactly what was packaged and where it belongs.
