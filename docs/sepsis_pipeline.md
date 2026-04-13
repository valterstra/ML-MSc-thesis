# Sepsis RL Pre-Processing Pipeline

## Purpose

This pipeline ports the [AI-Clinician-MIMICIV](https://github.com/microsoft/AI-Clinician-MIMICIV)
BigQuery pipeline to local PostgreSQL, producing **MKdataset.csv** -- the exact input
format expected by the sepsisRL reinforcement learning framework.

The goal is to create a Sepsis-3 ICU cohort with 4-hourly state-action sequences
that can train and evaluate a fluid/vasopressor dosing policy.

---

## Data Source

**MIMIC-IV** (local PostgreSQL, db=`mimic`).

Key schema notes:
- `mimiciv_icu.*` -- ICU stays, chartevents, inputevents, outputevents
- `mimiciv_hosp.*` -- admissions, patients, prescriptions, labevents (all-text columns)
- `mimiciv_hosp_typed.*` -- typed views over `mimiciv_hosp` used for join correctness
- `mimiciv_derived.*` -- pre-computed concepts (sofa, sepsis3, etc.) -- **not used in this pipeline**

---

## Final Output Summary

**MKdataset.csv**: `data/processed/sepsis/MKdataset.csv`
- **325,159 rows** from **37,071 Sepsis-3 ICU stays** (max SOFA >= 2)
- 85 columns: demographics, vitals, labs, vent params, fluid I/O, vasopressors, outcomes
- 4-hour time resolution (~8.7 blocs per stay average)
- Column renames for sepsisRL compat: `output_step` -> `output_4hourly`, `input_step` -> `input_4hourly_tev`

**Cohort filtering** (step 08):
- 39,152 ICU stays in -> 37,071 out
- Removed: 25 outlier stays (extreme UO/bili/intake), 298 treatment-stopped-before-death, 1,429 died-in-ICU

---

## 8-Step Pipeline

### Step 01 -- Extract raw data from PostgreSQL
**Script:** `scripts/sepsis/step_01_extract.py`
**Output:** `data/interim/sepsis/raw_data/`

Extracts 14 tables from MIMIC-IV via psycopg2:
`abx, culture, microbio, ce (batched), labs_ce, labs_le, mechvent, mechvent_pe,
preadm_fluid, fluid_mv, vaso_mv, preadm_uo, uo, demog`

Chartevents (ce) are extracted in 10 batches of 1M stay_id range each
(stay_ids cluster ~30M-40M in MIMIC-IV).

**Known issue -- CSV corruption:** The PostgreSQL COPY output occasionally merges two rows
into one line (missing newline). Found 53 merged rows in `fluid_mv.csv` and 1 in
`vaso_mv.csv`. Fixed by a repair script that splits them back on extraction.

---

### Step 02 -- Preprocess raw data
**Script:** `scripts/sepsis/step_02_preprocess.py`
**Output:** `data/interim/sepsis/intermediates/`

- Remaps chartevents and lab itemids to simplified sequential IDs (REF_VITALS / REF_LABS)
- Combines microbio + culture -> `bacterio.csv`
- Imputes missing `icustayid` in bacterio and abx from demog (by subject_id/hadm_id time window)
- Computes normalized infusion rate for fluid_mv
- Removes null-icustayid rows from mechvent, vaso_mv

Bacterio imputation iterates ~3.15M rows (pandas progress_apply) -- takes ~17 min.

---

### Step 03 -- Calculate sepsis onset times
**Script:** `scripts/sepsis/step_03_sepsis_onset.py`
**Output:** `data/interim/sepsis/intermediates/sepsis_onset.csv`

Sepsis-3 proxy (no explicit sepsis label in MIMIC-IV):
- Antibiotic administration (abx) within +/-24h of a microbiological culture (bacterio)
- Onset time = the earlier of antibiotic start or culture time

**Result:** ~49,793 sepsis onset records from 53,906 unique ICU stays.

---

### Step 04 -- Build patient states (unbinned)
**Script:** `scripts/sepsis/step_04_patient_states_fast.py` *(optimized version)*
**Output:** `data/interim/sepsis/intermediates/patient_states/patient_states.csv` (2,630,151 rows)
           `data/interim/sepsis/intermediates/patient_states/qstime.csv` (39,152 stays)

For each sepsis patient, collects all chart/lab/mechvent events within a
-49h/+25h window around onset time. One row per unique timestamp.

**Optimization:** Pre-groups all DataFrames by icustayid before the patient loop and
converts per-timestamp lookups to numpy array operations. ~9x speedup vs original.

Checkpointing every 200 patients with **fixed column ordering** (`.reindex(columns=...)`)
-- safe to resume with `--resume` after crash.

**Critical bug fix -- checkpoint column ordering:** The original checkpoint logic used
`pd.DataFrame(list_of_dicts).to_csv(mode='a', header=False)`. Different checkpoint batches
have different dict key insertion orders because patients have different events. This scrambled
columns silently after the first batch -- e.g., Sodium column contained Platelet values.
Fixed with `.reindex(columns=_STATES_COLUMNS)` before every checkpoint write. The same fix
was applied to step 06.

---

### Step 05 -- Impute patient states
**Script:** `scripts/sepsis/step_05_impute_states.py`
**Output:** `data/interim/sepsis/intermediates/patient_states/patient_states_imputed.csv`

1. **Sentinel filter**: Any value > 10,000 in clinical columns -> NaN (catches MIMIC-IV 999,999 placeholders). Removed 724 values across 33 columns.
2. **Unit corrections**: Temp_C > 45 with missing Temp_F moved to Temp_F (wrong-unit recording). FiO2 < 1 multiplied by 100 (fraction -> percentage).
3. **Complete physiological bounds** for all 63 chart + lab columns (see below).
4. **GCS estimation** from RASS sedation scale (all GCS values are NaN in MIMIC-IV because the total GCS itemid 198 is CareVue-only).
5. **FiO2 estimation** from O2 flow rate and delivery device interface type.
6. **Cross-column imputation**: BP (SysBP/MeanBP/DiaBP from each other), Temp (C<->F conversion), Hb<->Ht, Total_bili<->Direct_bili.
7. **Direct_bili clamp**: 2,933 negative values from formula artifact (`Total_bili * 0.6934 - 0.1752` goes negative when Total_bili < 0.25) clamped to 0.
8. **Sample-and-hold** forward-fill within each ICU stay for 65 columns. Vectorized implementation using pandas groupby + ffill (~50-200x faster than original per-element loop).

#### Complete Outlier Bounds Table

The original AI-Clinician code only had bounds for 30 of 63 columns. We extended this to
cover ALL clinical columns. The original bounds are preserved where they existed; new bounds
are marked with (*).

**Vitals:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| Height_cm | 50* | 250* | tallest human ~251 cm |
| Weight_kg | 0.5* | 300 | bariatric upper bound |
| GCS | 3* | 15* | GCS scale definition |
| RASS | -5* | 4* | RASS scale definition |
| HR | 0* | 250 | asystole=0 may be charted |
| SysBP | 0 | 300 | |
| MeanBP | 0 | 200 | |
| DiaBP | 0 | 200 | |
| RR | 0* | 80 | |
| SpO2 | 0* | 100* | percentage, was (None, 150) + clamp |
| Temp_C | 25* | 45* | was (None, 90) -- 90 far too generous |
| Temp_F | 77* | 113* | was NONE -- critical gap in original |

**Hemodynamics (all new*):**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| CVP | -20 | 40 | cmH2O |
| PAPsys | 5 | 120 | mmHg |
| PAPmean | 5 | 80 | mmHg |
| PAPdia | 0 | 60 | mmHg |
| CI | 0.5 | 10 | L/min/m2 |
| SVR | 100 | 4000 | dyn*s/cm5 (100% NaN in MIMIC-IV) |

**Respiratory:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| FiO2_100 | 20 | 100* | was (20, None) -- no upper bound! |
| FiO2_1 | 0.2* | 1.0* | was (None, 1.5) -- fraction max is 1.0 |
| O2flow | 0* | 70 | was (None, 70) |
| PEEP | 0 | 40 | |
| TidalVolume | 0* | 1800 | |
| MinuteVentil | 0* | 50 | |
| PAWmean | 0* | 50* | was NONE |
| PAWpeak | 0* | 80* | was NONE |
| PAWplateau | 0* | 60* | was NONE |

**Labs -- Electrolytes/Chemistry:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| Potassium | 1 | 12* | was (1, 15) |
| Sodium | 95 | 178 | unchanged |
| Chloride | 70 | 150 | unchanged |
| Glucose | 1 | 1000 | unchanged |
| BUN | 1* | 300* | was NONE |
| Creatinine | 0.1* | 50* | was (None, 150) |
| Magnesium | 0.5* | 10 | was (None, 10) |
| Calcium | 4* | 20 | was (None, 20); <4 incompatible with life |
| Ionised_Ca | 0.2* | 5 | |
| CO2_mEqL | 5* | 60* | was (None, 120) |

**Labs -- Liver/Enzymes:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| SGOT | 0* | 10000 | massive hepatic necrosis possible |
| SGPT | 0* | 10000 | |
| Total_bili | 0* | 100* | was NONE |
| Direct_bili | 0* | 70* | was NONE; negatives from formula clamped post-imputation |
| Total_protein | 1* | 15* | was NONE |
| Albumin | 0.5* | 7* | was NONE |
| Troponin | 0* | 100* | was NONE |
| CRP | 0* | 500* | was NONE |

**Labs -- Hematology:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| Hb | 1* | 25* | was (None, 20); polycythemia ~22 |
| Ht | 5* | 70* | was (None, 65) |
| RBC_count | 0.5* | 10* | was NONE |
| WBC_count | 0* | 500 | unchanged |
| Platelets_count | 1* | 2000 | unchanged |

**Labs -- Coagulation:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| PTT | 10* | 200* | was NONE; 0 is lab error |
| PT | 7* | 200* | was NONE |
| ACT | 50* | 600* | was NONE |
| INR | 0.5* | 25* | was (None, 20) |

**Labs -- Blood Gas:**
| Column | Lower | Upper | Notes |
|--------|-------|-------|-------|
| Arterial_pH | 6.7 | 7.8* | was (6.7, 8); >7.8 essentially fatal |
| paO2 | 10* | 700 | was (None, 700); 0 impossible in living patient |
| paCO2 | 5* | 200 | was (None, 200); 0 impossible |
| Arterial_BE | -50 | 40* | was (-50, None) -- no upper bound! |
| Arterial_lactate | 0.1* | 30 | |
| HCO3 | 2* | 55* | was NONE; 133 was in data |
| ETCO2 | 0* | 100* | was NONE; 374 was in data |
| SvO2 | 0* | 100* | was NONE; percentage cannot exceed 100 |

---

### Step 06 -- Combine states with actions
**Script:** `scripts/sepsis/step_06_states_actions_vec.py` *(vectorized version)*
**Output:** `data/interim/sepsis/intermediates/states_actions.csv` (340,691 rows)

Joins patient states with fluid and vasopressor actions. Computes per-bloc fluid input,
vasopressor dose (median + max), and urine output. Adds demographics from demog table.

Same checkpoint column-ordering fix as step 04 applied here.

---

### Step 07 -- Final imputation
**Script:** `scripts/sepsis/step_07_impute_final.py`
**Output:** `data/interim/sepsis/intermediates/states_actions_final.csv` (340,691 rows)

1. **Feature corrections**: Gender encoding (2/1 -> 1/0), age >150 clamped to 91, NaN mechvent -> 0, NaN elixhauser -> median.
2. **Type conversions**: `re_admission` (bool -> int), `died_within_48h_of_out_time` (string "True"/"False" -> int), `died_in_hosp` (to int). Required for downstream numeric comparisons in step 08.
3. **Linear interpolation** on 6 columns with <5% missingness: HR, MeanBP, RR, SpO2, FiO2_100, FiO2_1.
4. **KNN imputation** on all 63 chart + lab columns (~20 min). Batch size 10,000, nan_euclidean metric, 4 jobs. Skips columns with >90% missing in a batch.
5. **Computed features**: PaO2/FiO2 ratio, Shock Index (HR/SysBP), SOFA score, SIRS score.

---

### Step 08 -- Build Sepsis-3 cohort
**Script:** `scripts/sepsis/step_08_cohort.py`
**Output:** `data/processed/sepsis/sepsis_cohort.csv`, `data/processed/sepsis/MKdataset.csv`

Filters to Sepsis-3 cohort:
- Remove 25 outlier stays (extreme UO >12,000, bilirubin >10,000, fluid intake >10,000)
- Remove 298 stays where treatment was stopped before death
- Remove 1,429 stays where patient died in ICU during data collection window
- Keep only patients with max SOFA >= 2 -> **37,071 stays, 325,159 rows**

---

## Bugs Found and Fixed

### Bug 1: Checkpoint column scrambling (CRITICAL)
**Where:** Steps 04 and 06
**Impact:** All lab/chart values scrambled after first checkpoint batch. Sodium column contained Platelet values, etc.
**Root cause:** `pd.DataFrame(list_of_dicts)` column order depends on dict key insertion order, which varies between patients. Appending to CSV with `header=False` writes in the new DataFrame's order, not the file header's order.
**Fix:** `.reindex(columns=FIXED_COLUMNS)` before every checkpoint write.

### Bug 2: SpO2 row corruption
**Where:** Step 05, `remove_outliers()`
**Impact:** 25,142 rows had ALL columns set to 100 (including icustayid).
**Root cause:** `df[df[C_SPO2] > 100] = 100` sets entire row, not just SpO2 column.
**Fix:** `df.loc[df[C_SPO2] > 100, C_SPO2] = 100`

### Bug 3: MIMIC-IV sentinel values (999,999)
**Where:** Raw labevents/chartevents data
**Impact:** 724 values of 999,999 across 33 columns survived into the pipeline. Corrupts z-score/min-max normalization in sepsisRL.
**Root cause:** MIMIC-IV uses 999,999 as a placeholder for invalid/unmeasurable results. The original AI-Clinician pipeline didn't filter these.
**Fix:** Blanket sentinel filter: any value > 10,000 in clinical columns -> NaN.

### Bug 4: Incomplete outlier bounds (26 columns with no bounds)
**Where:** Step 05, `fill_outliers()` spec
**Impact:** Mid-range garbage values (100-10,000) survived in CVP, PAP, CI, PAWmean, ETCO2, SvO2, HCO3, Height, FiO2, Temp_F, and others.
**Root cause:** The original AI-Clinician code only specified bounds for 30 of 63 columns. The original authors likely ran on a cleaner MIMIC extract where these gaps didn't matter.
**Fix:** Complete physiological bounds for all 63 columns (see table above). Also tightened existing bounds (Temp_C from 90 to 45, FiO2_100 added upper=100, Arterial_BE added upper=40).

### Bug 5: Temp_F -> Temp_C cross-imputation of bad values
**Where:** Step 05, `estimate_vitals()`
**Impact:** Bad Temp_F values (e.g., 8211) got converted to Temp_C via `(Temp_F - 32) / 1.8 = 4544`.
**Root cause:** Temp_F had NO outlier bounds at all in the original code. Bad values survived filtering, then cross-imputation created new bad values in Temp_C.
**Fix:** Added Temp_F bounds (77, 113) and tightened Temp_C to (25, 45). Also fixed the wrong-unit correction to clear the source column.

### Bug 6: NAType crash in vectorized sample_and_hold
**Where:** `src/careai/sepsis/imputation.py`, `sample_and_hold()`
**Impact:** `TypeError: float() argument must be a string or a real number, not 'NAType'`
**Root cause:** `pd.NA` (nullable integer dtype from checkpoint reads) cannot be cast with `.astype(float)`.
**Fix:** `pd.to_numeric(series, errors='coerce').values` instead of `series.values.astype(float)`.

### Bug 7: Type mismatches in demographic columns
**Where:** Step 07/08
**Impact:** `re_admission` stored as bool, `died_within_48h_of_out_time` stored as string "True"/"False". Numeric comparisons (`== 1`) in step 08 silently failed.
**Fix:** `pd.to_numeric(col, errors='coerce').fillna(0).astype(int)` in step 07's `correct_features()`.

### Bug 8: Direct_bili formula artifact
**Where:** Step 05, `estimate_vitals()`
**Impact:** 2,933 negative Direct_bili values from `Total_bili * 0.6934 - 0.1752` when Total_bili < 0.25.
**Fix:** Post-cross-imputation clamp: `Direct_bili < 0 -> 0`.

---

## Data Quality Assessment (Final MKdataset)

### Missingness patterns (clinically reasonable)
- **Continuous monitors** (HR, MeanBP, RR, SpO2, FiO2): < 5% missing
- **Basic labs** (K, Na, Glucose, BUN, Cr, WBC, Plt): 9-17% missing
- **Coag panel** (PTT, PT, INR, HCO3): 28-31% missing
- **ABG** (pH, paO2, paCO2, lactate): 56-85% missing (only with arterial line)
- **Vent params** (PEEP, TV, MV, PAW): ~60% missing (~40% intubated)
- **Specialized** (CVP, Albumin, Troponin): 64-85% missing
- **Rare** (PAP, CI, SVR, SvO2, CRP, ACT): 90-100% missing (PA catheter required)

### Known data characteristics (not bugs)
- **RBC_count 74% missing** vs WBC/Platelets 13%: Different itemid coverage in MIMIC-IV, not a pipeline error. Hb (20%) and Ht (12%) carry the same clinical information.
- **SGPT 90% missing** vs SGOT 56%: AST is ordered more frequently than ALT in ICU settings. Known MIMIC pattern.
- **GCS 100% originally NaN**: MIMIC-IV lacks the CareVue total GCS itemid (198). All GCS values estimated from RASS sedation scale. See "Known Limitations" below.
- **SVR 100% NaN**: CareVue-only measurement. No MIMIC-IV equivalent.
- **19 negative DiaBP values**: Formula artifact from BP cross-imputation (`DiaBP = (3*MeanBP - SysBP)/2` when SysBP > 3*MeanBP). Negligible (19 of 340K rows).

---

## Known Limitations (MIMIC-III vs MIMIC-IV)

The original AI-Clinician pipeline was developed on MIMIC-III (CareVue + MetaVision mix).
MIMIC-IV contains only MetaVision data. Three variables are affected:

| Variable | Status | Impact | Workaround |
|----------|--------|--------|------------|
| **GCS** | CareVue itemid 198 not in MIMIC-IV | HIGH -- SOFA neurological component uses proxy | Estimated from RASS. MIMIC-IV has real GCS components (220739/223900/223901) but they were never added to the extraction query. |
| **FiO2_1** | CareVue itemid 190 not in MIMIC-IV | NONE | Auto-filled as FiO2_100 / 100 in step 05. |
| **SVR** | CareVue itemid 626 not in MIMIC-IV | MINOR | 100% NaN. Not used in any derived score. KNN imputes from neighbors. |

### To fix GCS properly (future work):
1. Add MetaVision GCS component itemids to `REF_VITALS[2]` in `columns.py`: 220739 (Eye), 223900 (Verbal), 223901 (Motor)
2. Re-run step 01 with `--only ce` to re-extract chartevents
3. Sum the three components to get GCS total (3-15)
4. Re-run steps 02-08

---

## Performance

| Step | Time | Rows In | Rows Out | Notes |
|------|------|---------|----------|-------|
| 01 | ~20 min | DB | 14 CSVs | Requires DB credentials |
| 02 | ~20 min | 14 CSVs | intermediates | Bacterio imputation is slow |
| 03 | ~1 min | intermediates | 49,793 onsets | |
| 04 | ~10 min | intermediates | 2,630,151 | Optimized from ~90 min |
| 05 | ~5 min | 2,630,151 | 2,630,151 | Vectorized SAH, was ~20 min |
| 06 | ~3 min | 2,630,151 | 340,691 | Vectorized version |
| 07 | ~30 min | 340,691 | 340,691 | KNN is the bottleneck |
| 08 | ~1 min | 340,691 | 325,159 | Cohort filtering |
| **Total** | **~90 min** | | | **Steps 04-08: ~50 min** |

---

## File Locations

```
data/interim/sepsis/
  raw_data/          <- step 01 output (CSV extracts from PostgreSQL)
  intermediates/     <- steps 02-06 intermediate files
    patient_states/  <- step 04-05 output

data/processed/sepsis/
  sepsis_cohort.csv  <- step 08: one row per ICU stay, cohort summary
  MKdataset.csv      <- step 08: full dataset, sepsisRL input (325,159 rows, 37,071 stays)

logs/
  step_05_final.log
  step_06_final.log
  step_07_final.log
  step_08_final.log

scripts/sepsis/
  step_01_extract.py
  step_02_preprocess.py
  step_03_sepsis_onset.py
  step_04_patient_states_fast.py      <- USE THIS (optimized)
  step_05_impute_states.py
  step_06_states_actions_vec.py       <- USE THIS (vectorized)
  step_07_impute_final.py
  step_08_cohort.py

src/careai/sepsis/
  queries.py          <- all SQL queries (PostgreSQL dialect)
  columns.py          <- REF_VITALS, REF_LABS, column name constants
  imputation.py       <- fill_outliers, knn_impute, sample_and_hold (vectorized)
  derived_features.py <- compute_sofa, compute_sirs, compute_pao2_fio2
  utils.py            <- load_csv, load_intermediate_or_raw_csv
```

---

## Run Order (full pipeline from scratch)

```bash
source ../.venv/Scripts/activate

# Step 01: Extract (requires DB credentials)
PGUSER=postgres PGPASSWORD=<pw> python scripts/sepsis/step_01_extract.py

# Steps 02-03: Preprocess and onset
python scripts/sepsis/step_02_preprocess.py
python scripts/sepsis/step_03_sepsis_onset.py

# Step 04: Patient states (optimized)
python scripts/sepsis/step_04_patient_states_fast.py \
    data/interim/sepsis/intermediates/patient_states \
    --log logs/step_04.log

# Steps 05-08: Chained (stops on first failure)
python scripts/sepsis/step_05_impute_states.py \
    data/interim/sepsis/intermediates/patient_states/patient_states.csv \
    data/interim/sepsis/intermediates/patient_states/patient_states_imputed.csv \
    --log logs/step_05.log && \
python scripts/sepsis/step_06_states_actions_vec.py \
    data/interim/sepsis/intermediates/patient_states/patient_states_imputed.csv \
    data/interim/sepsis/intermediates/patient_states/qstime.csv \
    data/interim/sepsis/intermediates/states_actions.csv \
    --log logs/step_06.log && \
python scripts/sepsis/step_07_impute_final.py \
    data/interim/sepsis/intermediates/states_actions.csv \
    data/interim/sepsis/intermediates/states_actions_final.csv \
    --log logs/step_07.log && \
python scripts/sepsis/step_08_cohort.py \
    data/interim/sepsis/intermediates/states_actions_final.csv \
    data/interim/sepsis/intermediates/patient_states/qstime.csv \
    data/processed/sepsis/ \
    --log logs/step_08.log
```
