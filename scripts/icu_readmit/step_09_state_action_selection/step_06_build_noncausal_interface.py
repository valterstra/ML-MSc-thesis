"""
Step 09 side track -- build broad non-causal state/action interface.

Purpose
-------
Materialize a broad predictive interface from step-08 `ICUdataset.csv` without
the causal narrowing used by the active selected-causal branch.

This side track keeps:
  - a broad dynamic state space
  - broad static context
  - broad binary action space
  - SOFA as an auxiliary reward/support column

It excludes:
  - identifiers from the feature lists
  - outcome leakage columns from the model interface
  - zero-variance columns
  - extremely sparse variables judged unsuitable as default inputs

Outputs
-------
  data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_dataset.parquet
  data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_spec.json
  data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_missingness.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

META_COLS = [
    "bloc",
    "icustayid",
    "timestep",
]

STATIC_CONTEXT_COLS = [
    "gender",
    "age",
    "Weight_kg",
    "race",
    "insurance",
    "marital_status",
    "admission_type",
    "admission_location",
    "charlson_score",
    "re_admission",
    "prior_ed_visits_6m",
]

DYNAMIC_STATE_COLS = [
    "HR",
    "SysBP",
    "MeanBP",
    "DiaBP",
    "NIBP_Diastolic",
    "Arterial_BP_Sys",
    "Arterial_BP_Dia",
    "RR",
    "RR_Spontaneous",
    "RR_Total",
    "SpO2",
    "Temp_C",
    "Interface",
    "FiO2_1",
    "O2flow",
    "PEEP",
    "TidalVolume",
    "TidalVolume_Observed",
    "MinuteVentil",
    "PAWmean",
    "PAWpeak",
    "PAWplateau",
    "Pain_Level",
    "GCS",
    "mechvent",
    "Potassium",
    "Sodium",
    "Chloride",
    "Glucose",
    "BUN",
    "Creatinine",
    "Magnesium",
    "Calcium",
    "Ionised_Ca",
    "CO2_mEqL",
    "SGOT",
    "SGPT",
    "Total_bili",
    "Direct_bili",
    "Albumin",
    "Hb",
    "Ht",
    "RBC_count",
    "WBC_count",
    "Platelets_count",
    "PTT",
    "PT",
    "INR",
    "Arterial_pH",
    "paO2",
    "paCO2",
    "Arterial_BE",
    "Arterial_lactate",
    "HCO3",
    "Phosphate",
    "Anion_Gap",
    "Alkaline_Phosphatase",
    "Fibrinogen",
    "Neuts_pct",
    "Lymphs_pct",
    "Monos_pct",
]

ACTION_SOURCE_COLS = [
    "vasopressor_dose",
    "max_dose_vaso",
    "ivfluid_dose",
    "input_4hourly_tev",
    "antibiotic_active",
    "anticoagulant_active",
    "diuretic_active",
    "insulin_active",
    "opioid_active",
    "sedation_active",
    "transfusion_active",
    "electrolyte_active",
    "mechvent",
]

ACTION_COLS = [
    "vasopressor_active",
    "ivfluid_active",
    "antibiotic_active",
    "anticoagulant_active",
    "diuretic_active",
    "insulin_active",
    "opioid_active",
    "sedation_active",
    "transfusion_active",
    "electrolyte_active",
    "mechvent_active",
]

AUX_COLS = [
    "SOFA",
    "readmit_30d",
]

EXCLUDED_COLS = {
    "identifier_or_ordering": ["bloc", "icustayid", "timestep"],
    "outcome_or_terminal_leakage": [
        "readmit_30d",
        "discharge_disposition",
        "died_in_hosp",
        "died_within_48h_of_out_time",
        "delay_end_of_record_and_discharge_or_death",
    ],
    "duplicates_removed": [
        "Temp_F",
        "FiO2_100",
        "GCS_Eye",
        "GCS_Verbal",
        "GCS_Motor",
    ],
    "zero_variance_removed": [
        "extubated",
        "cam_icu",
        "drg_severity",
        "drg_mortality",
        "steroid_active",
    ],
    "too_sparse_removed": [
        "Basos_pct",
        "Eos_pct",
        "SVR",
        "CI",
        "Total_protein",
        "ACT",
        "CRP",
        "PAPmean",
        "PAPdia",
        "PAPsys",
        "SvO2",
        "ETCO2",
    ],
    "not_in_default_keep_optional_for_later": [
        "CVP",
        "Troponin",
        "LDH",
        "PaO2_FiO2",
        "Shock_Index",
        "SIRS",
        "input_total",
        "output_total",
        "output_4hourly",
        "cumulated_balance",
        "median_dose_vaso",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "ICUdataset.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "step_09_noncausal_interface"),
    )
    parser.add_argument(
        "--log",
        default=str(PROJECT_ROOT / "logs" / "step_09_noncausal_interface.log"),
    )
    return parser


def setup_logging(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")],
    )


def build_binary_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vasopressor_active"] = (
        (out["vasopressor_dose"].fillna(0) > 0) | (out["max_dose_vaso"].fillna(0) > 0)
    ).astype("int8")
    out["ivfluid_active"] = (
        (out["ivfluid_dose"].fillna(0) > 0) | (out["input_4hourly_tev"].fillna(0) > 0)
    ).astype("int8")

    for col in [
        "antibiotic_active",
        "anticoagulant_active",
        "diuretic_active",
        "insulin_active",
        "opioid_active",
        "sedation_active",
        "transfusion_active",
        "electrolyte_active",
        "mechvent",
    ]:
        out[col] = out[col].fillna(0).astype("int8")
    out["mechvent_active"] = out["mechvent"]

    return out


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_to_read = META_COLS + STATIC_CONTEXT_COLS + DYNAMIC_STATE_COLS + ACTION_SOURCE_COLS + AUX_COLS
    cols_to_read = list(dict.fromkeys(cols_to_read))

    logging.info("Loading ICUdataset: %s", input_path)
    df = pd.read_csv(input_path, usecols=cols_to_read)
    logging.info("Loaded %d rows", len(df))

    df = build_binary_actions(df)

    final_cols = META_COLS + STATIC_CONTEXT_COLS + DYNAMIC_STATE_COLS + ACTION_COLS + AUX_COLS
    final_df = df[final_cols].copy()

    missingness = []
    for col in final_cols:
        s = final_df[col]
        missingness.append(
            {
                "column": col,
                "group": (
                    "meta" if col in META_COLS else
                    "static" if col in STATIC_CONTEXT_COLS else
                    "dynamic" if col in DYNAMIC_STATE_COLS else
                    "action" if col in ACTION_COLS else
                    "aux"
                ),
                "nonnull_rows": int(s.notna().sum()),
                "coverage": float(s.notna().mean()),
                "nunique_nonnull": int(s.nunique(dropna=True)),
            }
        )

    missing_df = pd.DataFrame(missingness).sort_values(["group", "column"]).reset_index(drop=True)

    spec = {
        "purpose": "Broad non-causal predictive interface built from step-08 ICUdataset.csv",
        "input": str(input_path),
        "output_dataset": str((out_dir / "noncausal_interface_dataset.parquet").resolve()),
        "meta_cols": META_COLS,
        "static_context_cols": STATIC_CONTEXT_COLS,
        "dynamic_state_cols": DYNAMIC_STATE_COLS,
        "action_cols": ACTION_COLS,
        "aux_cols": AUX_COLS,
        "derived_actions": {
            "vasopressor_active": "1 if vasopressor_dose > 0 or max_dose_vaso > 0 else 0",
            "ivfluid_active": "1 if ivfluid_dose > 0 or input_4hourly_tev > 0 else 0",
        },
        "excluded_cols": EXCLUDED_COLS,
        "n_rows": int(len(final_df)),
        "n_stays": int(final_df["icustayid"].nunique()),
    }

    dataset_path = out_dir / "noncausal_interface_dataset.parquet"
    spec_path = out_dir / "noncausal_interface_spec.json"
    missing_path = out_dir / "noncausal_interface_missingness.csv"

    final_df.to_parquet(dataset_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    logging.info("Saved dataset: %s", dataset_path)
    logging.info("Saved spec:    %s", spec_path)
    logging.info("Saved stats:   %s", missing_path)


if __name__ == "__main__":
    main()
