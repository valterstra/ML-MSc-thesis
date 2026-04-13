"""
ICU Feature Audit — explore_features.py

Reads MIMIC-IV ICU CSV files directly to compute per-item coverage
(% of distinct ICU stays with at least one measurement).

Outputs per-table CSV files to reports/icu_readmit/feature_audit/
and a comparison report against the current AI-Clinician feature set.

Usage:
    python scripts/icu_readmit/explore_features.py
    python scripts/icu_readmit/explore_features.py --icu-dir path/to/icu/csvs
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_ICU_DIR = Path(r"C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\downloads\icu")
OUT_DIR = Path("reports/icu_readmit/legacy/step_30_33/feature_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

Path("logs/legacy/icu_readmit").mkdir(parents=True, exist_ok=True)
LOG_FILE = "logs/legacy/icu_readmit/step_34_explore_features_legacy.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 500_000  # rows per chunk for large files

# ---------------------------------------------------------------------------
# Current AI-Clinician itemids — used to flag "already known" items
# ---------------------------------------------------------------------------
KNOWN_ITEMIDS = set([
    # REF_VITALS
    226707, 226730,
    581, 580, 224639, 226512,
    198,
    228096,
    211, 220045,
    220179, 225309, 6701, 6, 227243, 224167, 51, 455,
    220181, 220052, 225312, 224322, 6702, 443, 52, 456,
    8368, 8441, 225310, 8555, 8440,
    220210, 3337, 224422, 618, 3603, 615,
    220277, 646, 834,
    3655, 223762,
    223761, 678,
    220074, 113,
    492, 220059,
    491, 220061,
    8448, 220060,
    116, 1372, 1366, 228368, 228177,
    626,
    467, 226732,
    223835, 3420, 160, 727,
    190,
    470, 471, 223834, 227287, 194, 224691,
    220339, 506, 505, 224700,
    224686, 224684, 684, 224421, 3083, 2566, 654, 3050, 681, 2311,
    224687, 450, 448, 445,
    224697, 444,
    224695, 535,
    224696, 543,
    # REF_LABS
    829, 1535, 227442, 227464, 3792, 50971, 50822,
    837, 220645, 4194, 3725, 3803, 226534, 1536, 4195, 3726, 50983, 50824,
    788, 220602, 1523, 4193, 3724, 226536, 3747, 50902, 50806,
    225664, 807, 811, 1529, 220621, 226537, 3744, 50809, 50931,
    781, 1162, 225624, 3737, 51006, 52647,
    791, 1525, 220615, 3750, 50912, 51081,
    821, 1532, 220635, 50960,
    786, 225625, 1522, 3746, 50893, 51624,
    816, 225667, 3766, 50808,
    777, 787, 50804,
    770, 3801, 50878, 220587,
    769, 3802, 50861,
    1538, 848, 225690, 51464, 50885,
    803, 1527, 225651, 50883,
    3807, 1539, 849, 50976,
    772, 1521, 227456, 3727, 50862,
    227429, 851, 51002, 51003,
    227444, 50889,
    814, 220228, 50811, 51222,
    813, 220545, 3761, 226540, 51221, 50810,
    4197, 3799, 51279,
    1127, 1542, 220546, 4200, 3834, 51300, 51301,
    828, 227457, 3789, 51265,
    825, 1533, 227466, 3796, 51275, 52923, 52165, 52166, 52167,
    824, 1286, 51274, 227465,
    1671, 1520, 768, 220507,
    815, 1530, 227467, 51237,
    780, 1126, 3839, 4753, 50820,
    779, 490, 3785, 3838, 3837, 50821, 220224, 226063, 226770, 227039,
    778, 3784, 3836, 3835, 50818, 220235, 226062, 227036,
    776, 224828, 3736, 4196, 3740, 74, 50802,
    225668, 1531, 50813,
    227443, 50882, 50803,
    1817, 228640,
    823, 227686, 223772,
    # Phosphate candidates
    50970, 220736,
])


def count_coverage_chunked(csv_path, itemid_col, stayid_col, value_col=None, total_stays=None):
    """
    Read a large CSV in chunks and compute per-itemid distinct stay_id coverage.
    Returns a dict: {itemid: n_distinct_stays}
    """
    stay_sets = defaultdict(set)
    n_chunks = 0

    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in [itemid_col, stayid_col, value_col] if value_col else c in [itemid_col, stayid_col],
                              chunksize=CHUNK_SIZE, low_memory=False):
        # Filter to rows with a numeric value if value_col provided
        if value_col and value_col in chunk.columns:
            chunk = chunk.dropna(subset=[value_col])

        for row_itemid, row_stayid in zip(chunk[itemid_col], chunk[stayid_col]):
            if pd.notna(row_itemid) and pd.notna(row_stayid):
                stay_sets[int(row_itemid)].add(int(row_stayid))

        n_chunks += 1
        if n_chunks % 10 == 0:
            log.info("  processed %d chunks (%d items seen so far)...", n_chunks, len(stay_sets))

    return {itemid: len(stays) for itemid, stays in stay_sets.items()}


def build_coverage_df(coverage_dict, d_items, total_stays):
    """Join coverage counts with d_items catalog, compute pct_stays."""
    df = pd.DataFrame([
        {"itemid": k, "n_stays": v}
        for k, v in coverage_dict.items()
    ])
    if df.empty:
        return df
    df = df.merge(d_items, on="itemid", how="left")
    df["pct_stays"] = (df["n_stays"] * 100.0 / total_stays).round(1)
    df["known"] = df["itemid"].isin(KNOWN_ITEMIDS)
    df = df.sort_values("n_stays", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icu-dir", type=Path, default=DEFAULT_ICU_DIR)
    args = parser.parse_args()

    icu_dir = args.icu_dir
    log.info("=== ICU Feature Audit ===")
    log.info("Reading from: %s", icu_dir)

    # Load d_items catalog
    log.info("Loading d_items...")
    d_items = pd.read_csv(icu_dir / "d_items.csv", low_memory=False)
    d_items = d_items[["itemid", "label", "category", "unitname", "linksto"]].copy()
    log.info("  %d items in catalog", len(d_items))

    # Total ICU stays
    log.info("Loading icustays...")
    icustays = pd.read_csv(icu_dir / "icustays.csv", usecols=["stay_id"], low_memory=False)
    total_stays = icustays["stay_id"].nunique()
    log.info("  Total ICU stays: %d", total_stays)

    results = {}

    # ------------------------------------------------------------------
    # Chartevents
    # ------------------------------------------------------------------
    log.info("Processing chartevents (large file — this will take a while)...")
    cov = count_coverage_chunked(
        icu_dir / "chartevents.csv",
        itemid_col="itemid", stayid_col="stay_id", value_col="valuenum"
    )
    df_chart = build_coverage_df(cov, d_items, total_stays)
    df_chart.to_csv(OUT_DIR / "chartevents_coverage.csv", index=False)
    results["chartevents"] = df_chart
    log.info("Chartevents: %d unique items with numeric values", len(df_chart))

    # ------------------------------------------------------------------
    # Outputevents
    # ------------------------------------------------------------------
    log.info("Processing outputevents...")
    cov = count_coverage_chunked(
        icu_dir / "outputevents.csv",
        itemid_col="itemid", stayid_col="stay_id", value_col="value"
    )
    df_out = build_coverage_df(cov, d_items, total_stays)
    df_out.to_csv(OUT_DIR / "outputevents_coverage.csv", index=False)
    results["outputevents"] = df_out
    log.info("Outputevents: %d unique items", len(df_out))

    # ------------------------------------------------------------------
    # Procedurevents
    # ------------------------------------------------------------------
    log.info("Processing procedurevents...")
    cov = count_coverage_chunked(
        icu_dir / "procedureevents.csv",
        itemid_col="itemid", stayid_col="stay_id"
    )
    df_proc = build_coverage_df(cov, d_items, total_stays)
    df_proc.to_csv(OUT_DIR / "procedurevents_coverage.csv", index=False)
    results["procedurevents"] = df_proc
    log.info("Procedurevents: %d unique items", len(df_proc))

    # ------------------------------------------------------------------
    # Inputevents
    # ------------------------------------------------------------------
    log.info("Processing inputevents (action space reference)...")
    cov = count_coverage_chunked(
        icu_dir / "inputevents.csv",
        itemid_col="itemid", stayid_col="stay_id", value_col="amount"
    )
    df_inp = build_coverage_df(cov, d_items, total_stays)
    df_inp.to_csv(OUT_DIR / "inputevents_coverage.csv", index=False)
    results["inputevents"] = df_inp
    log.info("Inputevents: %d unique items", len(df_inp))

    # ------------------------------------------------------------------
    # Datetimeevents
    # ------------------------------------------------------------------
    log.info("Processing datetimeevents...")
    cov = count_coverage_chunked(
        icu_dir / "datetimeevents.csv",
        itemid_col="itemid", stayid_col="stay_id"
    )
    df_dt = build_coverage_df(cov, d_items, total_stays)
    df_dt.to_csv(OUT_DIR / "datetimeevents_coverage.csv", index=False)
    results["datetimeevents"] = df_dt
    log.info("Datetimeevents: %d unique items", len(df_dt))

    # ------------------------------------------------------------------
    # Comparison: high-coverage items NOT in current feature set
    # ------------------------------------------------------------------
    COVERAGE_THRESHOLD = 10.0

    log.info("")
    log.info("=== UNKNOWN HIGH-COVERAGE ITEMS (>=%.0f%% of stays, not in current feature set) ===", COVERAGE_THRESHOLD)

    report_rows = []
    for table, df in results.items():
        if df.empty or "pct_stays" not in df.columns:
            continue
        unknown = df[(df["pct_stays"] >= COVERAGE_THRESHOLD) & (~df["known"])].copy()
        unknown["table"] = table
        report_rows.append(unknown)
        if len(unknown) > 0:
            log.info("")
            log.info("--- %s (%d unknown items >= %.0f%%) ---", table, len(unknown), COVERAGE_THRESHOLD)
            for _, row in unknown.iterrows():
                log.info(
                    "  itemid=%-8d  pct=%5.1f%%  label=%-45s  cat=%s",
                    row["itemid"], row["pct_stays"], str(row.get("label", "")), str(row.get("category", ""))
                )

    if report_rows:
        combined = pd.concat(report_rows, ignore_index=True)
        combined = combined.sort_values("pct_stays", ascending=False)
        combined.to_csv(OUT_DIR / "unknown_high_coverage.csv", index=False)
        log.info("")
        log.info("Saved unknown high-coverage items: %s", OUT_DIR / "unknown_high_coverage.csv")

    # ------------------------------------------------------------------
    # Known items: confirm coverage of our current feature set
    # ------------------------------------------------------------------
    log.info("")
    log.info("=== KNOWN ITEMS COVERAGE (confirming our assumed features are well-measured) ===")
    all_known_rows = []
    for table, df in results.items():
        if df.empty:
            continue
        known = df[df["known"]].copy()
        known["table"] = table
        all_known_rows.append(known)

    if all_known_rows:
        known_combined = pd.concat(all_known_rows, ignore_index=True)
        known_combined = known_combined.sort_values("pct_stays", ascending=False)
        known_combined.to_csv(OUT_DIR / "known_items_coverage.csv", index=False)
        log.info("")
        log.info("Top 40 known items by coverage:")
        for _, row in known_combined.head(40).iterrows():
            log.info(
                "  [%-15s] itemid=%-8d  pct=%5.1f%%  label=%s",
                row["table"], row["itemid"], row["pct_stays"], str(row.get("label", ""))
            )

    log.info("")
    log.info("=== DONE. All results in %s ===", OUT_DIR)


if __name__ == "__main__":
    main()
