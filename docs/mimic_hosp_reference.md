# MIMIC-IV Hospital Data Reference
## For Hospital-Stay Transition Model Construction

### Mental Model

The goal is to build a state-action-transition table at the **hospital admission level**, where:

- **Episode** = one hospital admission (`hadm_id`), framed by `admissions.admittime` → `admissions.dischtime`
- **Time step** = one calendar day (or configurable window, e.g. 6-hour blocks)
- **State** = {care unit, clinical service, active lab values, active medications}
- **Action** = {medication changes, care unit escalation/de-escalation, discharge}
- **Outcome** = 30-day readmission (next `admittime` within 30 days of `dischtime`)

The episode ends at hospital discharge — which is the exact moment the readmission clock starts, making the label clean and direct.

---

## Table Roles at a Glance

| Table | Role | Time-varying |
|---|---|---|
| `admissions` | Episode frame | No (per admission) |
| `transfers` | State + Action (care unit) | Yes (per unit move) |
| `services` | State (clinical service) | Yes (per service change) |
| `labevents` | State (physiology) | Yes (~1–3×/day) |
| `prescriptions` | Action (medications) | Yes (start/stop) |
| `diagnoses_icd` | Static covariate | No (discharge-time) |
| `drgcodes` | Static covariate | No (discharge-time) |
| `patients` | Static covariate | No (per patient) |
| `microbiologyevents` | State flag (infection) | Yes (irregular) |
| `emar` | Action (dose given) | Yes (per dose) |
| `pharmacy` | Action (medication order) | Yes (start/stop) |
| `omr` | Pre-admission baseline | Date-level (outpatient) |
| `d_labitems` | Reference/lookup | No |
| `d_icd_diagnoses` | Reference/lookup | No |
| `d_icd_procedures` | Reference/lookup | No |
| `d_hcpcs` | Reference/lookup | No |
| `emar_detail` | Too granular — skip | Yes |
| `poe` | Too broad — skip | Yes |
| `poe_detail` | Too unstructured — skip | No |
| `hcpcsevents` | Billing only — skip | Date-level |
| `provider` | No clinical value — skip | No |

---

## Detailed Table Descriptions

---

### `admissions.csv`
**Role: Episode frame — one row per hospital admission**

Every patient-hospital encounter. This is the backbone of the transition model — each row defines one episode.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Unique patient identifier | Join key |
| `hadm_id` | int | Unique hospital admission identifier — the primary episode key | Episode ID |
| `admittime` | timestamp | Datetime patient was admitted to hospital | Episode start |
| `dischtime` | timestamp | Datetime patient was discharged | Episode end; readmission clock starts here |
| `deathtime` | timestamp | Datetime of in-hospital death (null if survived) | Terminal episode flag |
| `admission_type` | str | Acuity of admission: ELECTIVE, URGENT, EW EMER. (emergency walk-in), EU OBSERVATION, DIRECT EMER., DIRECT OBSERVATION, AMBULATORY OBSERVATION | Static state feature; strong readmission predictor |
| `admit_provider_id` | str | Anonymised ID of admitting provider | Not useful |
| `admission_location` | str | Where patient came from: EMERGENCY ROOM, TRANSFER FROM HOSPITAL, PHYSICIAN REFERRAL, WALK-IN/SELF REFERRAL, PACU, AMBULATORY SURGERY TRANSFER, INTERNAL TRANSFER TO OR FROM PSYCH | Static feature; transfer-from-hospital flag is useful |
| `discharge_location` | str | Where patient went after discharge: HOME, HOME HEALTH CARE, SKILLED NURSING FACILITY, REHAB, CHRONIC/LONG TERM ACUTE CARE, HOSPICE, AGAINST ADVICE, DIED, etc. | Strong readmission predictor — SNF/rehab discharges have high readmission risk |
| `insurance` | str | Payer: Medicare, Medicaid, Other | Static covariate; proxy for socioeconomic status |
| `language` | str | Primary language | Static covariate |
| `marital_status` | str | Marital status | Static covariate |
| `race` | str | Patient race/ethnicity | Static covariate |
| `edregtime` | timestamp | Time patient registered in ED (null if not ED admission) | ED visit flag; time in ED = edouttime - edregtime |
| `edouttime` | timestamp | Time patient left ED | ED boarding time |
| `hospital_expire_flag` | int | 1 = patient died during this admission | Exclude from readmission analysis or treat as terminal |

**Key derived features:**
- `hospital_los_days` = (dischtime - admittime) in days
- `ed_admission` = 1 if edregtime is not null
- `ed_los_hours` = (edouttime - edregtime) in hours

---

### `transfers.csv`
**Role: State (care unit location) + Action (escalation/discharge) — the temporal backbone**

Every physical unit movement during the hospital stay. This table lets you reconstruct exactly where the patient was at any point in time, and what transitions occurred.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `transfer_id` | int | Unique transfer event identifier | Row ID |
| `eventtype` | str | Type of event: `ED` (emergency dept), `admit` (ward/unit admission), `discharge` | Episode boundary marker; `discharge` marks episode end |
| `careunit` | str | Physical unit name: Emergency Department, Medical Intensive Care Unit (MICU), Surgical Intensive Care Unit (SICU), Cardiac Vascular Intensive Care Unit (CVICU), Coronary Care Unit (CCU), Neuro Surgical Intensive Care Unit (NSICU), Transplant, Medicine, Surgery, Cardiology, Hematology/Oncology, Psychiatry, etc. | Primary location state variable |
| `intime` | timestamp | Datetime patient entered this unit | Time-step boundary |
| `outtime` | timestamp | Datetime patient left this unit (null for current location) | Time-step boundary |

**Key derived features:**
- `is_icu` = 1 if careunit contains "Intensive Care" or "ICU" or "CCU"
- `icu_escalation` = transition from non-ICU to ICU unit
- `icu_stepdown` = transition from ICU to non-ICU ward
- At any time t: patient's current unit = careunit where intime ≤ t < outtime
- `post_icu_los_days` = time between ICU outtime and hospital dischtime

**Important:** `eventtype = 'discharge'` with `careunit = 'UNKNOWN'` is the terminal row — it marks the moment of hospital discharge and should be the final state of each episode.

---

### `services.csv`
**Role: State (clinical service) — which medical team is responsible**

Clinical service changes during the admission. Different from physical location — a patient can be on the Medicine service while physically in an ICU bed, or on Surgery service while on a general ward.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `transfertime` | timestamp | Datetime of service change | Time boundary |
| `prev_service` | str | Previous clinical service (null on admission) | Service transition |
| `curr_service` | str | Current clinical service | Service state variable |

**Common service codes:**
- `MED` = General Medicine
- `SURG` = General Surgery
- `CSURG` = Cardiac Surgery
- `NSURG` = Neurosurgery
- `TRAUM` = Trauma
- `ORTHO` = Orthopaedics
- `GYN` = Gynaecology
- `OBS` = Obstetrics
- `PSYCH` = Psychiatry
- `NMED` = Neurology
- `OMED` = Oncology Medicine
- `CMED` = Cardiology Medicine
- `MICU`, `SICU`, `TSICU`, `CSRU`, `CCU` = ICU services

**Key derived feature:**
- `service_escalation` = change from ward service to ICU service

---

### `labevents.csv`
**Role: PRIMARY STATE SOURCE — time-stamped physiology throughout the hospital stay**

The most important time-varying table. Every lab draw for every patient across the entire hospital stay (not just ICU). At 18 GB this is the largest table. Covers inpatient and outpatient draws — filter by `hadm_id IS NOT NULL` to get inpatient only.

| Column | Type | Description | Model use |
|---|---|---|---|
| `labevent_id` | int | Unique event identifier | Row ID |
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier (NULL = outpatient draw) | Filter: keep only non-null for inpatient model |
| `specimen_id` | int | Specimen identifier — links multiple tests from same draw | Grouping |
| `itemid` | int | Lab test identifier — join to `d_labitems` for name | State variable selector |
| `order_provider_id` | str | Ordering provider (anonymised) | Not useful |
| `charttime` | timestamp | When the sample was drawn/resulted | Time anchor for state |
| `storetime` | timestamp | When result was entered in system | Usually slightly after charttime |
| `value` | str | Raw result value (may be text e.g. "NEGATIVE") | — |
| `valuenum` | float | Numeric result value | PRIMARY state variable value |
| `valueuom` | str | Unit of measurement (mg/dL, mEq/L, etc.) | Unit check |
| `ref_range_lower` | float | Normal range lower bound | Abnormality flag computation |
| `ref_range_upper` | float | Normal range upper bound | Abnormality flag computation |
| `flag` | str | "abnormal" if outside reference range, null otherwise | Direct abnormality state feature |
| `priority` | str | ROUTINE or STAT (urgent) | STAT flag = clinical concern |
| `comments` | str | Free text notes | Not useful |

**Key itemids for state variables (from d_labitems):**
- `50912` = Creatinine (mg/dL) — kidney function
- `51006` = Urea Nitrogen / BUN (mg/dL) — kidney function
- `50983` = Sodium (mEq/L) — electrolytes
- `50971` = Potassium (mEq/L) — electrolytes
- `50882` = Bicarbonate (mEq/L) — acid-base
- `50868` = Anion Gap — acid-base
- `50931` = Glucose (mg/dL) — metabolic
- `50862` = Albumin (g/dL) — nutrition/severity
- `51222` = Hemoglobin (g/dL) — anemia
- `51265` = Platelets (K/uL) — coagulation
- `51301` = WBC (K/uL) — infection/inflammation
- `50960` = Magnesium (mg/dL) — electrolytes
- `50893` = Calcium (mg/dL) — electrolytes
- `50970` = Phosphate (mg/dL) — electrolytes
- `50813` = Lactate (mmol/L) — tissue perfusion/shock
- `50885` = Total Bilirubin (mg/dL) — liver function
- `50910` = ALT/SGPT — liver function
- `50954` = LDH — tissue damage

**Usage pattern for daily state:**
```sql
SELECT hadm_id, DATE(charttime) AS lab_date, itemid, AVG(valuenum) AS daily_value
FROM mimiciv_hosp.labevents
WHERE hadm_id IS NOT NULL AND valuenum IS NOT NULL
  AND itemid IN (50912, 51006, 50983, 50971, 50882, 50931, 51222, 51301, 50813)
GROUP BY hadm_id, DATE(charttime), itemid
```
Then pivot on itemid and forward-fill missing days.

---

### `prescriptions.csv`
**Role: PRIMARY ACTION SOURCE — what medications were ordered and when**

Clean medication orders with start/stop times. Built from physician orders, this is the best source for defining medication actions in the transition model.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `pharmacy_id` | int | Links to pharmacy dispensing record | Cross-reference |
| `poe_id` | str | Links to provider order entry | Cross-reference |
| `poe_seq` | int | Order sequence number | — |
| `order_provider_id` | str | Ordering provider (anonymised) | Not useful |
| `starttime` | timestamp | When medication order started | Action onset |
| `stoptime` | timestamp | When medication order ended | Action offset |
| `drug_type` | str | MAIN, ADDITIVE, or BASE | Filter: keep MAIN |
| `drug` | str | Drug name (free text) | Action classification |
| `formulary_drug_cd` | str | Hospital formulary code | Drug lookup |
| `gsn` | str | Generic Sequence Number | Drug classification |
| `ndc` | str | National Drug Code | Drug classification |
| `prod_strength` | str | Strength and form (e.g. "40mg Tablet") | Dose context |
| `form_rx` | str | Prescribed form | — |
| `dose_val_rx` | str | Dose value | — |
| `dose_unit_rx` | str | Dose unit | — |
| `form_val_disp` | str | Dispensed form value | — |
| `form_unit_disp` | str | Dispensed form unit | — |
| `doses_per_24_hrs` | float | Frequency per day | — |
| `route` | str | Route of administration: PO (oral), IV, IH (inhaled), etc. | IV flag = more acute |

**Key action flags derivable from `drug` column (use LIKE matching):**
- `any_antibiotic` — match antibiotic drug names (see mimic-code `antibiotic.sql` for full list)
- `any_anticoagulant` — Heparin, Warfarin, Enoxaparin, Apixaban, Rivaroxaban
- `any_diuretic` — Furosemide, Torsemide, Bumetanide, Metolazone, Hydrochlorothiazide
- `any_steroid` — Methylprednisolone, Prednisone, Dexamethasone, Hydrocortisone
- `any_insulin` — Insulin (various types)
- `any_vasopressor` — Norepinephrine, Epinephrine, Dopamine, Vasopressin, Phenylephrine (overlap with ICU inputevents)
- `any_opioid` — Morphine, Hydromorphone, Oxycodone, Fentanyl

**Usage pattern for daily action flags:**
```sql
SELECT hadm_id, generate_series::date AS day,
  MAX(CASE WHEN drug ILIKE '%antibiotic_name%' THEN 1 ELSE 0 END) AS any_antibiotic
FROM mimiciv_hosp.prescriptions
CROSS JOIN generate_series(starttime::date, stoptime::date, '1 day')
GROUP BY hadm_id, day
```

---

### `diagnoses_icd.csv`
**Role: Static covariate — discharge diagnoses and comorbidity scoring**

All ICD-coded diagnoses for each admission. Assigned at discharge, so not time-varying during the stay. Primary diagnosis (seq_num = 1) indicates the main reason for admission.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `seq_num` | int | Priority ranking — 1 = primary diagnosis | Filter seq_num=1 for primary |
| `icd_code` | str | ICD code (join to `d_icd_diagnoses` for description) | Comorbidity scoring |
| `icd_version` | int | 9 = ICD-9, 10 = ICD-10 | Version handling required |

**Key uses:**
- Elixhauser comorbidity score (better than Charlson for readmission prediction) — requires mapping ICD codes to comorbidity categories
- Primary diagnosis category as a state feature (cardiac, respiratory, renal, etc.)
- `mimic-code` provides `charlson.sql` which already maps these codes

---

### `drgcodes.csv`
**Role: Static covariate — case complexity and severity**

DRG (Diagnosis Related Group) codes assigned at discharge. Two systems: APR-DRG (All Patient Refined, has severity/mortality scores) and MS-DRG (Medicare, widely used).

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `drg_type` | str | APR or HCFA (MS-DRG) | Filter by type |
| `drg_code` | str | DRG code | Case type |
| `description` | str | DRG description text | Case type label |
| `drg_severity` | int | APR severity: 1=Minor, 2=Moderate, 3=Major, 4=Extreme (APR only) | Direct severity feature |
| `drg_mortality` | int | APR mortality risk: 1–4 scale (APR only) | Direct mortality risk feature |

**Key uses:**
- `drg_severity` (APR) is a single-number case complexity score — strong readmission predictor
- `drg_mortality` captures expected mortality risk

---

### `patients.csv`
**Role: Static patient demographics**

One row per patient across all their admissions.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `gender` | str | M or F | Demographic covariate |
| `anchor_age` | int | Patient age at `anchor_year` — use to compute age at admission | Age covariate |
| `anchor_year` | int | De-identified reference year for this patient | Age calculation: age_at_admit = anchor_age + (YEAR(admittime) - anchor_year) |
| `anchor_year_group` | str | Broad year group (e.g. "2014 - 2016") — actual year is shifted for privacy | Calendar context |
| `dod` | date | Date of death (null if alive at data extraction) | Mortality outcome |

---

### `microbiologyevents.csv`
**Role: State flag — infection/sepsis signal**

Culture orders, results, and sensitivities. Covers blood cultures, urine cultures, sputum, wound swabs, and serology. Irregular timing — drawn when infection is suspected.

| Column | Type | Description | Model use |
|---|---|---|---|
| `microevent_id` | int | Unique event ID | Row ID |
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier (null = outpatient) | Episode join key |
| `micro_specimen_id` | int | Specimen ID — groups tests from same specimen | Grouping |
| `order_provider_id` | str | Ordering provider | Not useful |
| `chartdate` | timestamp | Date specimen ordered | Time anchor |
| `charttime` | timestamp | Time specimen ordered (more precise) | Time anchor |
| `spec_itemid` | int | Specimen type code | Specimen classification |
| `spec_type_desc` | str | Specimen type text: BLOOD CULTURE, URINE, SPUTUM, MRSA SCREEN, etc. | Infection site |
| `test_seq` | int | Test sequence within specimen | — |
| `storedate` / `storetime` | timestamp | When result was reported | Result timing |
| `test_itemid` | int | Test type code | — |
| `test_name` | str | Test name (e.g. BLOOD CULTURE, HCV VIRAL LOAD) | Test classification |
| `org_itemid` | int | Organism code (null = no growth) | Positive culture flag |
| `org_name` | str | Organism name (null = no growth or pending) | Pathogen type |
| `isolate_num` | int | Isolate number for multiple organisms | — |
| `quantity` | str | Quantity of growth | — |
| `ab_itemid` | int | Antibiotic sensitivity test code | — |
| `ab_name` | str | Antibiotic name tested | Sensitivity result |
| `dilution_text` / `dilution_comparison` / `dilution_value` | str/float | MIC dilution values | Resistance detail |
| `interpretation` | str | S=Sensitive, R=Resistant, I=Intermediate | Resistance flag |
| `comments` | str | Free text result notes | — |

**Key derived state features:**
- `culture_ordered` = any culture drawn on this day (flag of clinical suspicion)
- `positive_blood_culture` = org_name IS NOT NULL AND spec_type_desc = 'BLOOD CULTURE'
- `any_resistant_organism` = interpretation = 'R' for key antibiotics

---

### `emar.csv`
**Role: Alternative action source — actual dose administration events**

Electronic Medication Administration Record. Every dose actually given (or not given, or held). More precise than `prescriptions` but much larger (5.9 GB). Prefer `prescriptions` for daily action flags; use `emar` only if you need dose-level timing.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `hadm_id` | int | Admission identifier | Episode join key |
| `emar_id` | str | Unique administration event ID | Row ID |
| `emar_seq` | int | Sequence number | — |
| `poe_id` | str | Links to provider order | Cross-reference |
| `pharmacy_id` | int | Links to pharmacy record | Cross-reference |
| `enter_provider_id` | str | Nurse who charted the administration | Not useful |
| `charttime` | timestamp | When dose was given/charted | Precise action time |
| `medication` | str | Drug name | Action classification |
| `event_txt` | str | What happened: Administered, Not Given, Held, Partial Administration, Refused, Restarted, Stopped | Dose event type — filter to "Administered" |
| `scheduletime` | timestamp | When dose was scheduled | Compliance check |
| `storetime` | timestamp | When record was saved | — |

---

### `omr.csv`
**Role: Pre-admission baseline — outpatient measurements**

Outpatient measurement record. Date-level only, typically recorded in clinic visits before or after admission. Blood pressure values are stored as strings ("110/65") requiring parsing.

| Column | Type | Description | Model use |
|---|---|---|---|
| `subject_id` | int | Patient identifier | Join key |
| `chartdate` | date | Date of measurement | Time anchor (date only) |
| `seq_num` | int | Sequence within date | — |
| `result_name` | str | Measurement type: Blood Pressure, Weight (Lbs), Height (Inches), BMI (kg/m2), Weight (kg) | Measurement classifier |
| `result_value` | str | Value as text (BP = "systolic/diastolic") | Baseline covariate |

**Limited use:** Outpatient data, no timestamps, date-level only. Best use is as a pre-admission baseline BP or BMI covariate — find the most recent measurement before `admittime`.

---

### `pharmacy.csv`
**Role: Alternative to prescriptions — pharmacy dispensing view**

The pharmacy-side record of medication orders. Overlaps heavily with `prescriptions`. Contains additional pharmacy logistics fields but same core medication/timing information. Use `prescriptions` instead for a transition model — they cover the same clinical content.

---

### Reference Tables (lookup only, no patient rows)

**`d_labitems.csv`** — maps `itemid` → lab test name, fluid, category. Essential for selecting which itemids to extract from `labevents`.
Columns: `itemid`, `label`, `fluid`, `category`

**`d_icd_diagnoses.csv`** — maps ICD code → text description. Join to `diagnoses_icd` to get readable diagnosis names.
Columns: `icd_code`, `icd_version`, `long_title`

**`d_icd_procedures.csv`** — maps ICD procedure codes → descriptions. Join to `procedures_icd` if that table is loaded.
Columns: `icd_code`, `icd_version`, `long_title`

**`d_hcpcs.csv`** — maps HCPCS billing codes → descriptions. Only useful if using `hcpcsevents`.
Columns: `code`, `category`, `long_description`, `short_description`

---

### Tables to Skip

**`emar_detail.csv`** (8.1 GB) — line-level dose detail behind each emar event. Exact infusion rates, product codes, barcode scans. Far too granular — `prescriptions` or `emar` is sufficient.

**`poe.csv`** (4.8 GB) — all provider orders including lab orders, imaging orders, nursing orders, dietary orders. Too broad; medication orders are better captured by `prescriptions`.

**`poe_detail.csv`** — free-text field values behind provider orders. Mostly unstructured. Exception: code status (DNR/DNI) is buried here, but extraction requires text parsing.

**`hcpcsevents.csv`** — billing procedure codes with date only. Administrative data, no clinical value for a transition model.

**`provider.csv`** — just a list of anonymised provider IDs. No clinical content.

---

## Recommended State Vector (daily resolution)

```
State at day d for admission hadm_id:
  Location:
    - careunit         (from transfers: unit active at start of day d)
    - is_icu           (binary: careunit is an ICU)
    - curr_service     (from services: service active at start of day d)

  Labs (from labevents, daily average, forward-filled):
    - creatinine       (itemid 50912)
    - bun              (itemid 51006)
    - sodium           (itemid 50983)
    - potassium        (itemid 50971)
    - bicarbonate      (itemid 50882)
    - glucose          (itemid 50931)
    - wbc              (itemid 51301)
    - hemoglobin       (itemid 51222)
    - lactate          (itemid 50813)
    - bilirubin_total  (itemid 50885)

  Infection signal:
    - culture_ordered  (binary: any culture drawn on day d)
    - positive_culture (binary: any positive culture on or before day d)

  Static (constant across episode):
    - age
    - gender
    - admission_type
    - charlson_score
    - drg_severity     (APR-DRG)
```

## Recommended Action Vector (daily resolution)

```
Action on day d:
  - any_antibiotic     (binary: antibiotic active from prescriptions)
  - any_anticoagulant  (binary)
  - any_diuretic       (binary)
  - any_vasopressor    (binary — from prescriptions or ICU inputevents)
  - icu_escalation     (binary: transferred to ICU on day d)
  - icu_stepdown       (binary: transferred from ICU on day d)
  - discharged         (binary: eventtype='discharge' on day d)
```

## Episode Outcome

```
Label: readmit_30d = 1 if next hadm_id admittime within 30 days of dischtime
Source: admissions table self-join (already implemented in label_from_admissions.py)
```
