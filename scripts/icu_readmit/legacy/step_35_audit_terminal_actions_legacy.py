"""
Terminal action audit for ICU readmission pipeline.

Looks at actions in three windows:
  1. ICU last 24h -- inputevents + procedureevents right before ICU discharge
  2. Hospital discharge window -- prescriptions newly started at hospital discharge
  3. Service transitions at ICU discharge -- what services see the patient post-ICU

Output: reports/icu_readmit/feature_audit/terminal_actions/
"""

import os
import sys
import logging
import pandas as pd
from collections import defaultdict
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ICU_DIR   = r"C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\downloads\icu"
HOSP_DIR  = r"C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\downloads\hosp"
OUT_DIR   = r"C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\CareAI\reports\icu_readmit\legacy\step_30_33\feature_audit\terminal_actions"
os.makedirs(OUT_DIR, exist_ok=True)

LAST_ICU_HOURS = 24          # window before ICU discharge
HOSP_DISCH_HOURS = 48        # window around hospital discharge for "discharge meds"
MIN_STAY_COVERAGE = 0.02     # report items present in >=2% of stays


# ── 1. Load icustays ────────────────────────────────────────────────────────
log.info("Loading icustays...")
icu = pd.read_csv(os.path.join(ICU_DIR, "icustays.csv"),
                  usecols=["stay_id", "hadm_id", "subject_id", "intime", "outtime"],
                  parse_dates=["intime", "outtime"])
TOTAL_STAYS = len(icu)
log.info(f"  {TOTAL_STAYS} ICU stays, {icu['hadm_id'].nunique()} unique hadm_ids")

# Lookup dicts
stay_outtime  = dict(zip(icu["stay_id"],  icu["outtime"]))
stay_intime   = dict(zip(icu["stay_id"],  icu["intime"]))
stay_to_hadm  = dict(zip(icu["stay_id"],  icu["hadm_id"]))

# hadm -> last ICU discharge time
hadm_icu_out = icu.groupby("hadm_id")["outtime"].max().to_dict()

# hadm -> subject_id
hadm_to_subj = dict(zip(icu["hadm_id"], icu["subject_id"]))

icu_stay_ids  = set(icu["stay_id"].astype(int))
icu_hadm_ids  = set(icu["hadm_id"].astype(int))


# ── 2. Load d_items for labels ──────────────────────────────────────────────
log.info("Loading d_items...")
ditems = pd.read_csv(os.path.join(ICU_DIR, "d_items.csv"),
                     usecols=["itemid", "label", "category"])
item_label = dict(zip(ditems["itemid"], ditems["label"]))
item_cat   = dict(zip(ditems["itemid"], ditems["category"]))


# ── 3. ICU inputevents -- last 24h ──────────────────────────────────────────
log.info(f"Scanning inputevents (last {LAST_ICU_HOURS}h of ICU stay)...")
terminal_input = defaultdict(set)   # itemid -> set of stay_ids

for i, chunk in enumerate(pd.read_csv(
        os.path.join(ICU_DIR, "inputevents.csv"),
        usecols=["stay_id", "itemid", "starttime"],
        parse_dates=["starttime"],
        chunksize=500_000)):
    for row in chunk.itertuples(index=False):
        sid = int(row.stay_id)
        if sid not in stay_outtime:
            continue
        outtime = stay_outtime[sid]
        if pd.isna(outtime) or pd.isna(row.starttime):
            continue
        hours_before = (outtime - row.starttime).total_seconds() / 3600
        if 0 <= hours_before <= LAST_ICU_HOURS:
            terminal_input[int(row.itemid)].add(sid)
    if (i + 1) % 10 == 0:
        log.info(f"  inputevents: {i+1} chunks done")

log.info(f"  {len(terminal_input)} unique items seen in terminal window")

rows = []
for iid, stays in terminal_input.items():
    cov = len(stays) / TOTAL_STAYS
    if cov >= MIN_STAY_COVERAGE:
        rows.append({
            "itemid": iid,
            "label": item_label.get(iid, ""),
            "category": item_cat.get(iid, ""),
            "n_stays": len(stays),
            "coverage_pct": round(cov * 100, 1),
        })
df_input_terminal = pd.DataFrame(rows).sort_values("n_stays", ascending=False)
df_input_terminal.to_csv(os.path.join(OUT_DIR, "terminal_inputevents.csv"), index=False)
log.info(f"  Saved terminal_inputevents.csv ({len(df_input_terminal)} items >= {MIN_STAY_COVERAGE*100:.0f}%)")


# ── 4. ICU procedureevents -- last 24h ──────────────────────────────────────
log.info(f"Scanning procedureevents (last {LAST_ICU_HOURS}h)...")
terminal_proc = defaultdict(set)

for i, chunk in enumerate(pd.read_csv(
        os.path.join(ICU_DIR, "procedureevents.csv"),
        usecols=["stay_id", "itemid", "starttime"],
        parse_dates=["starttime"],
        chunksize=200_000)):
    for row in chunk.itertuples(index=False):
        sid = int(row.stay_id)
        if sid not in stay_outtime:
            continue
        outtime = stay_outtime[sid]
        if pd.isna(outtime) or pd.isna(row.starttime):
            continue
        hours_before = (outtime - row.starttime).total_seconds() / 3600
        if 0 <= hours_before <= LAST_ICU_HOURS:
            terminal_proc[int(row.itemid)].add(sid)

log.info(f"  {len(terminal_proc)} unique items seen in terminal window")

rows = []
for iid, stays in terminal_proc.items():
    cov = len(stays) / TOTAL_STAYS
    if cov >= MIN_STAY_COVERAGE:
        rows.append({
            "itemid": iid,
            "label": item_label.get(iid, ""),
            "category": item_cat.get(iid, ""),
            "n_stays": len(stays),
            "coverage_pct": round(cov * 100, 1),
        })
df_proc_terminal = pd.DataFrame(rows).sort_values("n_stays", ascending=False)
df_proc_terminal.to_csv(os.path.join(OUT_DIR, "terminal_procedureevents.csv"), index=False)
log.info(f"  Saved terminal_procedureevents.csv ({len(df_proc_terminal)} items)")


# ── 5. Hospital admissions -- get discharge times ────────────────────────────
log.info("Loading admissions for hospital discharge times...")
admissions = pd.read_csv(os.path.join(HOSP_DIR, "admissions.csv"),
                         usecols=["hadm_id", "dischtime", "discharge_location"],
                         parse_dates=["dischtime"])
admissions = admissions[admissions["hadm_id"].isin(icu_hadm_ids)]
hadm_dischtime = dict(zip(admissions["hadm_id"], admissions["dischtime"]))
hadm_discloc   = dict(zip(admissions["hadm_id"], admissions["discharge_location"]))
log.info(f"  {len(admissions)} admissions with ICU stays")
# Discharge location breakdown
loc_counts = admissions["discharge_location"].value_counts().head(10)
log.info("  Discharge locations:\n" + loc_counts.to_string())
loc_counts.reset_index().rename(columns={"index":"discharge_location","discharge_location":"n_hadm"}).to_csv(
    os.path.join(OUT_DIR, "discharge_locations.csv"), index=False)


# ── 6. Hospital prescriptions near hospital discharge ───────────────────────
log.info(f"Scanning prescriptions (within {HOSP_DISCH_HOURS}h of hospital discharge)...")
# "discharge medications" = newly started prescriptions within HOSP_DISCH_HOURS before discharge
disch_rx = defaultdict(set)   # drug (normalized) -> set of hadm_ids

for i, chunk in enumerate(pd.read_csv(
        os.path.join(HOSP_DIR, "prescriptions.csv"),
        usecols=["hadm_id", "drug", "drug_type", "starttime"],
        parse_dates=["starttime"],
        chunksize=500_000)):
    chunk = chunk[chunk["hadm_id"].isin(icu_hadm_ids)]
    if chunk.empty:
        continue
    for row in chunk.itertuples(index=False):
        hid = int(row.hadm_id)
        dischtime = hadm_dischtime.get(hid)
        if dischtime is None or pd.isna(row.starttime):
            continue
        hours_before = (dischtime - row.starttime).total_seconds() / 3600
        if 0 <= hours_before <= HOSP_DISCH_HOURS:
            drug_norm = str(row.drug).strip().upper() if pd.notna(row.drug) else "UNKNOWN"
            disch_rx[drug_norm].add(hid)
    if (i + 1) % 10 == 0:
        log.info(f"  prescriptions: {i+1} chunks done")

log.info(f"  {len(disch_rx)} unique drug names at hospital discharge")

N_HADM = len(admissions)
rows = []
for drug, hadms in disch_rx.items():
    cov = len(hadms) / N_HADM
    if cov >= MIN_STAY_COVERAGE:
        rows.append({"drug": drug, "n_hadm": len(hadms), "coverage_pct": round(cov * 100, 1)})
df_disch_rx = pd.DataFrame(rows).sort_values("n_hadm", ascending=False)
df_disch_rx.to_csv(os.path.join(OUT_DIR, "discharge_prescriptions.csv"), index=False)
log.info(f"  Saved discharge_prescriptions.csv ({len(df_disch_rx)} drugs >= {MIN_STAY_COVERAGE*100:.0f}%)")


# ── 7. Services table -- transitions at ICU discharge ───────────────────────
log.info("Scanning services at ICU discharge...")
services = pd.read_csv(os.path.join(HOSP_DIR, "services.csv"),
                       parse_dates=["transfertime"])
services = services[services["hadm_id"].isin(icu_hadm_ids)]

# For each service row, find if it's within 24h of any ICU outtime for that hadm
svc_rows = []
for row in services.itertuples(index=False):
    hid = int(row.hadm_id)
    icu_out = hadm_icu_out.get(hid)
    if icu_out is None or pd.isna(row.transfertime):
        continue
    hours_diff = abs((row.transfertime - icu_out).total_seconds()) / 3600
    if hours_diff <= 24:
        svc_rows.append({
            "hadm_id": hid,
            "transfertime": row.transfertime,
            "prev_service": row.prev_service,
            "curr_service": row.curr_service,
            "hours_from_icu_out": round((row.transfertime - icu_out).total_seconds() / 3600, 1),
        })

df_svc = pd.DataFrame(svc_rows)
if not df_svc.empty:
    df_svc.to_csv(os.path.join(OUT_DIR, "icu_discharge_services_raw.csv"), index=False)
    # Summarize transitions
    svc_summary = df_svc.groupby(["prev_service", "curr_service"]).size().reset_index(name="n")
    svc_summary["coverage_pct"] = (svc_summary["n"] / N_HADM * 100).round(1)
    svc_summary = svc_summary.sort_values("n", ascending=False)
    svc_summary.to_csv(os.path.join(OUT_DIR, "icu_discharge_services_summary.csv"), index=False)
    log.info(f"  Saved {len(svc_summary)} service transitions within 24h of ICU discharge")
    log.info("  Top transitions:\n" + svc_summary.head(15).to_string(index=False))
else:
    log.info("  No service transitions found near ICU discharge")


# ── 8. Print summary for review ─────────────────────────────────────────────
log.info("\n=== TERMINAL INPUT EVENTS (last 24h ICU) ===")
log.info(df_input_terminal.head(30).to_string(index=False))

log.info("\n=== TERMINAL PROCEDURE EVENTS (last 24h ICU) ===")
log.info(df_proc_terminal.head(30).to_string(index=False))

log.info("\n=== DISCHARGE PRESCRIPTIONS (hosp discharge window) ===")
log.info(df_disch_rx.head(40).to_string(index=False))

log.info("\nDone. All outputs in: " + OUT_DIR)
