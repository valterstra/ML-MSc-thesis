"""
prepare_colab_upload.py

Creates caresim_colab.zip containing the active selected-track ICU readmit
source files needed for the current Google Colab notebooks, including:
  - Step 11a: selected CARE-Sim training
  - Step 13a: selected CARE-Sim control
  - Step 13b: selected MarkovSim control
  - Step 14: selected offline RL comparison branch

Run from the repo root:
    python scripts/prepare_colab_upload.py

Output:
    caresim_colab.zip   (created in the repo root, ready to upload to Google Drive)

NOT included (upload separately to Drive):
    active selected parquet / model folders
    trained CARE-Sim / MarkovSim / offline model artifacts
"""
import os
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ZIP = os.path.join(REPO_ROOT, "caresim_colab.zip")

FILES_TO_ZIP = [
    # Package init files
    os.path.join("src", "careai", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "__init__.py"),
    # CareSimGPT source
    os.path.join("src", "careai", "icu_readmit", "caresim", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "model.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "dataset.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "train.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "ensemble.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "simulator.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "severity.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "readmit.py"),
    # Step 13 control layer
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "actions.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "observation.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "planner.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "evaluation.py"),
    os.path.join("src", "careai", "icu_readmit", "caresim", "control", "ddqn.py"),
    # MarkovSim source
    os.path.join("src", "careai", "icu_readmit", "markovsim", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "model.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "train.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "simulator.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "actions.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "observation.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "planner.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "evaluation.py"),
    os.path.join("src", "careai", "icu_readmit", "markovsim", "control", "ddqn.py"),
    # Reused RL network
    os.path.join("src", "careai", "icu_readmit", "rl", "__init__.py"),
    os.path.join("src", "careai", "icu_readmit", "rl", "networks.py"),
    os.path.join("src", "careai", "icu_readmit", "rl", "continuous.py"),
    os.path.join("src", "careai", "icu_readmit", "rl", "evaluation.py"),
    # Colab scripts
    os.path.join("scripts", "icu_readmit", "step_14_caresim_smoke_test.py"),
    os.path.join("scripts", "icu_readmit", "step_14_caresim_train.py"),
    os.path.join("scripts", "icu_readmit", "step_14_caresim_train_selected.py"),
    os.path.join("scripts", "icu_readmit", "step_11a_caresim_train_selected_causal.py"),
    os.path.join("scripts", "icu_readmit", "step_12a_caresim_evaluate.py"),
    os.path.join("scripts", "icu_readmit", "step_13a_caresim_control.py"),
    os.path.join("scripts", "icu_readmit", "step_11b_markovsim_train.py"),
    os.path.join("scripts", "icu_readmit", "step_12b_markovsim_evaluate.py"),
    os.path.join("scripts", "icu_readmit", "step_13b_markovsim_control.py"),
    os.path.join("scripts", "icu_readmit", "step_14_offline_selected.py"),
]

missing = []
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path in FILES_TO_ZIP:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.exists(abs_path):
            zf.write(abs_path, rel_path)
            print(f"  + {rel_path}")
        else:
            missing.append(rel_path)
            print(f"  ! MISSING: {rel_path}")

print()
if missing:
    print(f"WARNING: {len(missing)} file(s) missing from zip (see above).")
else:
    size_kb = os.path.getsize(OUTPUT_ZIP) / 1024
    print(f"Created: {OUTPUT_ZIP}  ({size_kb:.0f} KB)")
    print()
    print("Next steps:")
    print("  1. Upload caresim_colab.zip to Google Drive (anywhere, e.g. MyDrive/)")
    print("  2. Upload the selected parquet to MyDrive/CareAI/data/")
    print("     - rl_dataset_selected.parquet")
    print("  3. Open the matching notebook in Colab:")
    print("     - notebooks/step_11a_caresim_selected_causal_colab.ipynb")
    print("     - notebooks/step_13a_caresim_selected_colab.ipynb")
    print("     - notebooks/step_13b_markovsim_selected_colab.ipynb")
    print("     - notebooks/step_14_offline_selected_colab.ipynb")
