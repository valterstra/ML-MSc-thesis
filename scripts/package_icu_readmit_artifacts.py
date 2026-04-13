from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUNDLE_NAME = "icu_readmit_main_artifacts"
STAGING_DIR = DIST_DIR / BUNDLE_NAME
ZIP_PATH = DIST_DIR / f"{BUNDLE_NAME}.zip"


ARTIFACTS: list[dict[str, str]] = [
    {
        "path": "data/processed/icu_readmit/ICUdataset.csv",
        "kind": "optional",
        "reason": "Lets a reader inspect the final 01-08 cohort output without rerunning extraction.",
    },
    {
        "path": "data/processed/icu_readmit/rl_dataset_selected.parquet",
        "kind": "required",
        "reason": "Selected replay dataset used by steps 10a, 11a, 11b, 12a, 12b, 13a, 13b, and 14.",
    },
    {
        "path": "data/processed/icu_readmit/scaler_params_selected.json",
        "kind": "required",
        "reason": "Selected preprocessing metadata used downstream by the simulator/control stack.",
    },
    {
        "path": "data/processed/icu_readmit/static_context_selected.parquet",
        "kind": "required",
        "reason": "Selected static context table used by the active ICU-readmission branch.",
    },
    {
        "path": "models/icu_readmit/severity_selected",
        "kind": "required",
        "reason": "Selected severity surrogate for reward computation when using surrogate mode.",
    },
    {
        "path": "models/icu_readmit/terminal_readmit_selected",
        "kind": "required",
        "reason": "Terminal readmission model used by steps 12a, 12b, 13a, 13b, and 14.",
    },
    {
        "path": "models/icu_readmit/caresim_selected_causal",
        "kind": "required",
        "reason": "Trained CARE-Sim selected causal world model.",
    },
    {
        "path": "models/icu_readmit/caresim_control_selected_causal",
        "kind": "required",
        "reason": "CARE-Sim planner/DDQN control outputs used for comparison.",
    },
    {
        "path": "models/icu_readmit/markovsim_selected_causal",
        "kind": "required",
        "reason": "Trained MarkovSim selected causal baseline.",
    },
    {
        "path": "models/icu_readmit/markovsim_control_selected_causal",
        "kind": "required",
        "reason": "MarkovSim planner/DDQN control outputs used for comparison.",
    },
    {
        "path": "models/icu_readmit/offline_selected",
        "kind": "required",
        "reason": "Offline DDQN and OPE support models for step 14.",
    },
    {
        "path": "reports/icu_readmit/caresim_selected_causal",
        "kind": "optional",
        "reason": "Held-out CARE-Sim evaluation outputs for inspection and write-up.",
    },
    {
        "path": "reports/icu_readmit/markovsim_selected_causal",
        "kind": "optional",
        "reason": "Held-out MarkovSim evaluation outputs for inspection and write-up.",
    },
    {
        "path": "reports/icu_readmit/caresim_control_selected_causal",
        "kind": "optional",
        "reason": "CARE-Sim control evaluation summaries and traces.",
    },
    {
        "path": "reports/icu_readmit/markovsim_control_selected_causal",
        "kind": "optional",
        "reason": "MarkovSim control evaluation summaries and traces.",
    },
    {
        "path": "reports/icu_readmit/offline_selected",
        "kind": "optional",
        "reason": "Offline RL comparison reports and OPE summaries.",
    },
]


def copy_path(src: Path, dst_root: Path) -> dict[str, object]:
    rel = src.relative_to(PROJECT_ROOT)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
        kind = "dir"
    else:
        shutil.copy2(src, dst)
        size = src.stat().st_size
        kind = "file"
    return {
        "path": str(rel).replace("\\", "/"),
        "type": kind,
        "size_bytes": size,
    }


def build_bundle_readme(manifest: dict) -> str:
    lines = [
        "# ICU Readmit Artifact Bundle",
        "",
        "This bundle is meant to be extracted at the root of a clone of the `main` branch.",
        "",
        "## How To Use",
        "",
        "1. Clone the repository.",
        "2. Extract this archive at the repository root so the bundled `data/`, `models/`, and `reports/` folders merge into place.",
        "3. Verify the paths now exist exactly as listed in `artifact_manifest.json`.",
        "4. Run the active pipeline entrypoints from `scripts/icu_readmit/`.",
        "",
        "## What This Bundle Contains",
        "",
        "- selected processed datasets",
        "- trained severity and terminal readmission models",
        "- trained CARE-Sim and MarkovSim model folders",
        "- trained CARE-Sim, MarkovSim, and offline-DDQN control/comparison outputs",
        "- optional reports used for inspection and thesis write-up",
        "",
        "## Notes",
        "",
        "- The repository code stays on GitHub.",
        "- This bundle supplies the generated artifacts that are intentionally gitignored.",
        "- If a recipient only wants to inspect results, they do not need to rerun training.",
        "- If they want to retrain from scratch, the code can still do that without this bundle, assuming they have the required database/data pipeline upstream.",
        "",
        f"Bundle entries: {len(manifest['entries'])}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []

    for item in ARTIFACTS:
        src = PROJECT_ROOT / item["path"]
        if src.exists():
            entry = copy_path(src, STAGING_DIR)
            entry["bundle_kind"] = item["kind"]
            entry["reason"] = item["reason"]
            copied.append(entry)
        else:
            missing.append(item)

    manifest = {
        "bundle_name": BUNDLE_NAME,
        "extract_into": ".",
        "entries": copied,
        "missing": missing,
    }

    with open(STAGING_DIR / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(STAGING_DIR / "README_ARTIFACTS.md", "w", encoding="utf-8") as f:
        f.write(build_bundle_readme(manifest))

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(DIST_DIR / BUNDLE_NAME), "zip", root_dir=STAGING_DIR)

    total_bytes = sum(int(entry["size_bytes"]) for entry in copied)
    summary = {
        "zip_path": str(ZIP_PATH),
        "staging_dir": str(STAGING_DIR),
        "copied_entries": len(copied),
        "missing_entries": len(missing),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
