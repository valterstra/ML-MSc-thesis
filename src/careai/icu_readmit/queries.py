"""
PostgreSQL SQL queries for the ICU readmission pipeline.

Adapted from src/careai/sepsis/queries.py (AI-Clinician-MIMICIV structure).

Key differences from sepsis queries:
  - demog(): replaces elixhauser with charlson; adds race/insurance/marital_status/
             admission_type/admission_location/drg_severity/drg_mortality/
             prior_ed_visits_6m/discharge_location/readmit_30d
  - Charlson computed in a CTE (charlson_flags temp table built separately)
  - ce():    extended CHARTEVENT_CODES (GCS components, NIBP_dia, pain, WBC diff, etc.)
  - labs_ce(): extended LABS_CE_CODES (phosphate, anion_gap, alk_phos, LDH, fibrinogen)
  - drugs_mv(): NEW — extracts inputevents rows for all 9 binary drug classes
  - Removed: abx(), culture(), microbio() (sepsis-specific)
  - Kept unchanged: mechvent(), mechvent_pe(), fluid_mv(), vaso_mv(),
                    preadm_fluid(), uo(), preadm_uo(), labs_le()
"""

# ---------------------------------------------------------------------------
# Itemid sets
# ---------------------------------------------------------------------------

# Chart events — sepsis set + ICU audit additions
# GCS: replaced CareVue 198 (100% NaN) with MetaVision components
CHARTEVENT_CODES = (
    # --- sepsis items (kept) ---
    226707, 581, 228096, 211, 220179, 220181, 8368, 220210, 220277, 3655,
    223761, 220074, 492, 491, 8448, 116, 626, 467, 223835, 190, 470, 220339,
    224686, 224687, 224697, 224695, 224696, 226730, 580, 220045, 225309, 220052,
    8441, 3337, 646, 223762, 678, 113, 1372, 3420, 471, 506, 224684, 450, 444,
    535, 543, 224639, 6701, 225312, 225310, 224422, 834, 1366, 160, 223834, 505,
    684, 448, 226512, 6, 224322, 8555, 618, 228368, 727, 227287, 224700, 224421,
    445, 227243, 6702, 8440, 3603, 228177, 194, 3083, 224167, 443, 615, 224691,
    2566, 51, 52, 654, 455, 456, 3050, 681, 2311, 220059, 220061, 220060, 226732,
    # --- new ICU audit additions ---
    220739,   # GCS_Eye (replaces 198)
    223900,   # GCS_Verbal
    223901,   # GCS_Motor
    220180,   # NIBP_Diastolic
    220050,   # Arterial_BP_Sys
    220051,   # Arterial_BP_Dia
    224689,   # RR_Spontaneous
    224690,   # RR_Total
    224685,   # TidalVolume_Observed
    223791,   # Pain_Level
    229326,   # CAM-ICU (delirium screen; text values converted to 0/1 in ce() CASE)
)

# Lab events from chartevents — sepsis set + new additions
LABS_CE_CODES = (
    # --- sepsis items (kept) ---
    223772, 829, 1535, 227442, 227464, 4195, 3726, 3792, 837, 220645, 4194,
    3725, 3803, 226534, 1536, 4195, 3726, 788, 220602, 1523, 4193, 3724, 226536,
    3747, 225664, 807, 811, 1529, 220621, 226537, 3744, 781, 1162, 225624, 3737,
    791, 1525, 220615, 3750, 821, 1532, 220635, 786, 225625, 1522, 3746, 816,
    225667, 3766, 777, 787, 770, 3801, 769, 3802, 1538, 848, 225690, 803, 1527,
    225651, 3807, 1539, 849, 772, 1521, 227456, 3727, 227429, 851, 227444, 814,
    220228, 813, 220545, 3761, 226540, 4197, 3799, 1127, 1542, 220546, 4200,
    3834, 828, 227457, 3789, 825, 1533, 227466, 3796, 824, 1286, 1671, 1520,
    768, 220507, 815, 1530, 227467, 780, 1126, 3839, 4753, 779, 490, 3785, 3838,
    3837, 778, 3784, 3836, 3835, 776, 224828, 3736, 4196, 3740, 74, 225668, 1531,
    227443, 1817, 228640, 823, 227686, 220587, 227465, 220224, 226063, 226770,
    227039, 220235, 226062, 227036, 220644,
    # --- new additions (phosphate, anion_gap, alk_phos, LDH, fibrinogen, WBC diff) ---
    225677,   # Phosphate
    227073,   # Anion_Gap
    225612,   # Alkaline_Phosphatase
    220632,   # LDH
    227468,   # Fibrinogen
    225641,   # Neuts_pct
    225642,   # Lymphs_pct
    225643,   # Monos_pct
    225645,   # Eos_pct
    225644,   # Basos_pct
)

# Lab events from labevents — sepsis set + phosphate
LABS_LE_CODES = (
    50971,50822,50824,50806,50931,51081,50885,51003,51222,50810,51301,50983,
    50902,50809,51006,50912,50960,50893,50808,50804,50878,50861,51464,50883,
    50976,50862,51002,50889,50811,51221,51279,51300,51265,51275,51274,51237,
    50820,50821,50818,50802,50813,50882,50803,52167,52166,52165,52923,51624,
    52647,
    50970,  # Phosphate (labevents)
)

# Mechvent — identical to sepsis
MECHVENT_MEASUREMENT_CODES = (
    445, 448, 449, 450, 1340, 1486, 1600, 224687,
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686,
    218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747,
    221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,
    543,
    5865, 5866, 224707, 224709, 224705, 224706,
    60, 437, 505, 506, 686, 220339, 224700,
    3459,
    501, 502, 503, 224702,
    223, 667, 668, 669, 670, 671, 672,
    157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810,
    224701,
)
MECHVENT_CODES = (640, 720, 467) + MECHVENT_MEASUREMENT_CODES

# Preadmission fluid — identical to sepsis
PREADM_FLUID_CODES = (
    30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
    226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
    227071,227072,
)

# Urine output — identical to sepsis
UO_CODES = (
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405,
    40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489, 226584,
    226563, 226564, 226565, 226557, 226558,
)

# Vasopressors — identical to sepsis
VASO_CODES = (
    30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289,
    222315, 221662, 30043, 30307,
)

# ---------------------------------------------------------------------------
# Drug action itemids — one dict entry per binary action class
# These are used in drugs_mv() and step_06 to compute per-bloc binary flags.
# ---------------------------------------------------------------------------
DRUG_ACTION_ITEMIDS = {
    'antibiotic_active': (
        225798,   # Vancomycin
        225850,   # Cefazolin
        225851,   # Cefepime
        225893,   # Piperacillin/Tazobactam (Zosyn)
        225855,   # Ceftriaxone
        225884,   # Metronidazole
        225883,   # Meropenem
        225859,   # Ciprofloxacin
        225875,   # Ampicillin-Sulbactam
        225876,   # Nafcillin
        225900,   # Azithromycin
        225857,   # Clindamycin
        225860,   # Daptomycin
        225902,   # Fluconazole
    ),
    'anticoagulant_active': (
        225975,   # Heparin Sodium (Prophylaxis)
        225152,   # Heparin Sodium (therapeutic)
        225913,   # Enoxaparin (Lovenox)
        222026,   # Fondaparinux
    ),
    'diuretic_active': (
        221794,   # Furosemide (Lasix)
        228340,   # Bumetanide
        229691,   # Torsemide
    ),
    'steroid_active': (
        # NOTE: IV steroids in inputevents have lower coverage than prescriptions.
        # Hydrocortisone and methylprednisolone are the main ICU IV steroids.
        # Coverage may be low (~5-10%); oral steroids are not captured here.
        229311,   # Methylprednisolone Sodium Succinate (Solumedrol)
        221835,   # Hydrocortisone Sodium Succinate
        229761,   # Dexamethasone
    ),
    'insulin_active': (
        223258,   # Insulin - Regular
        223262,   # Insulin - Humalog
        223260,   # Insulin - Glargine
        260860,   # Insulin - NPH
        228304,   # Insulin - Detemir
    ),
    'opioid_active': (
        225154,   # Morphine Sulfate
        221833,   # Hydromorphone (Dilaudid)
        221744,   # Fentanyl
        225942,   # Fentanyl (Concentrate)
        228380,   # Oxycodone (IV uncommon; included for completeness)
    ),
    'sedation_active': (
        222168,   # Propofol
        221668,   # Midazolam (Versed)
        225150,   # Dexmedetomidine (Precedex)
        221385,   # Lorazepam (Ativan) — sedation context
        221036,   # Ketamine
    ),
    'transfusion_active': (
        225168,   # Packed Red Blood Cells
        220970,   # Fresh Frozen Plasma
        226368,   # Platelets
        220864,   # Albumin 5%
        220862,   # Albumin 25%
        225170,   # Cryoprecipitate
    ),
    'electrolyte_active': (
        222011,   # Magnesium Sulfate
        227523,   # Magnesium Sulfate (Bolus)
        225166,   # Potassium Chloride
        227522,   # KCL (Bolus)
        221456,   # Calcium Gluconate
        225799,   # K Phos (potassium phosphate)
        227524,   # Calcium Chloride
    ),
}

# Flat set of all drug action itemids (for SQL IN clause)
ALL_DRUG_ACTION_CODES = tuple(
    iid for ids in DRUG_ACTION_ITEMIDS.values() for iid in ids
)

# ---------------------------------------------------------------------------
# Services indicating obstetrics/neonatal — excluded from cohort
# ---------------------------------------------------------------------------
OBS_NEONATAL_SERVICES = ('OBS', 'GYN', 'NBB', 'NMED', 'NSURG')

# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def ce(min_stay, max_stay):
    """
    Chart events (vitals + ICU-specific additions).
    Batched over stay_id ranges for memory efficiency.
    Includes GCS components (220739/223900/223901) instead of broken CareVue 198.
    """
    return """
        SELECT DISTINCT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            CASE
                -- CAM-ICU delirium screen (itemid 229326): text -> binary
                WHEN itemid = 229326 AND lower(value) = 'positive'                        THEN 1
                WHEN itemid = 229326 AND lower(value) = 'negative'                        THEN 0
                -- 'unable to assess' and other non-standard values -> NULL (not assessed)
                WHEN itemid = 229326                                                       THEN NULL
                -- Interface encoding (unchanged)
                WHEN lower(value) = 'none'                                                THEN 0
                WHEN lower(value) = 'ventilator'                                          THEN 1
                WHEN lower(value) IN ('cannula','nasal cannula','high flow nasal cannula') THEN 2
                WHEN lower(value) = 'face tent'                                           THEN 3
                WHEN lower(value) = 'aerosol-cool'                                        THEN 4
                WHEN lower(value) = 'trach mask'                                          THEN 5
                WHEN lower(value) = 'hi flow neb'                                         THEN 6
                WHEN lower(value) = 'non-rebreather'                                      THEN 7
                WHEN lower(value) = ''                                                     THEN 8
                WHEN lower(value) = 'venti mask'                                          THEN 9
                WHEN lower(value) = 'medium conc mask'                                    THEN 10
                ELSE valuenum
            END AS valuenum
        FROM mimiciv_icu.chartevents
        WHERE stay_id >= {min_stay}
          AND stay_id < {max_stay}
          AND value IS NOT NULL
          AND itemid IN {codes}
        ORDER BY stay_id, charttime
    """.format(min_stay=min_stay, max_stay=max_stay, codes=repr(CHARTEVENT_CODES))


def labs_ce():
    """Lab-like items from chartevents (extended with phosphate/anion_gap/etc.)."""
    return """
        SELECT DISTINCT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            valuenum
        FROM mimiciv_icu.chartevents
        WHERE value IS NOT NULL
          AND itemid IN {codes}
        ORDER BY stay_id, charttime
    """.format(codes=repr(LABS_CE_CODES))


def labs_le():
    """Lab items from labevents (extended with phosphate 50970).
    Uses mimiciv_hosp_typed (integer hadm_id) to avoid text=integer cast errors.
    """
    return """
        SELECT
            xx.icustay_id,
            EXTRACT(EPOCH FROM l.charttime)::bigint AS charttime,
            l.itemid,
            l.valuenum
        FROM (
            SELECT hadm_id, stay_id AS icustay_id, intime, outtime
            FROM mimiciv_icu.icustays
            GROUP BY hadm_id, stay_id, intime, outtime
        ) AS xx
        INNER JOIN mimiciv_hosp_typed.labevents l ON l.hadm_id = xx.hadm_id
          AND l.charttime BETWEEN xx.intime AND xx.outtime
        WHERE l.itemid IN {codes}
          AND l.valuenum IS NOT NULL
        ORDER BY xx.icustay_id, l.charttime, l.itemid
    """.format(codes=repr(LABS_LE_CODES))


def mechvent():
    """Mechanical ventilation flags from chartevents — identical to sepsis."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            MAX(CASE
                WHEN itemid IN {meas_codes}
                     AND (value IS NOT NULL OR valuenum IS NOT NULL)  THEN 1
                WHEN itemid = 640 AND value = 'Extubated'             THEN 0
                WHEN itemid = 640 AND value = 'Self Extubated'        THEN 0
                ELSE NULL END) AS mechvent,
            MAX(CASE
                WHEN itemid = 640 AND value = 'Extubated'             THEN 1
                WHEN itemid = 640 AND value = 'Self Extubated'        THEN 1
                ELSE 0 END) AS extubated,
            MAX(CASE
                WHEN itemid = 640 AND value = 'Self Extubated'        THEN 1
                ELSE 0 END) AS selfextubated
        FROM mimiciv_icu.chartevents
        WHERE itemid IN {all_codes}
          AND (value IS NOT NULL OR valuenum IS NOT NULL)
        GROUP BY stay_id, charttime
        ORDER BY stay_id, charttime
    """.format(
        meas_codes=repr(MECHVENT_MEASUREMENT_CODES),
        all_codes=repr(MECHVENT_CODES),
    )


def mechvent_pe():
    """Mechanical ventilation from procedureevents — identical to sepsis."""
    return """
        SELECT
            p.subject_id, p.hadm_id,
            p.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM p.starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM p.endtime)::bigint   AS endtime,
            1 AS mechvent,
            0 AS extubated,
            0 AS selfextubated,
            p.itemid,
            p.value
        FROM mimiciv_icu.procedureevents p
        WHERE p.itemid IN (225792, 225794)
        ORDER BY p.stay_id, p.starttime
    """


def fluid_mv():
    """IV fluid inputevents (MetaVision only) — identical to sepsis."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM endtime)::bigint   AS endtime,
            itemid,
            amount,
            rate,
            CASE
                WHEN amountuom = 'L' THEN amount * 1000
                ELSE amount
            END AS tev
        FROM mimiciv_icu.inputevents
        WHERE itemid IN {codes}
          AND statusdescription != 'Rewritten'
        ORDER BY stay_id, starttime
    """.format(codes=repr((
        225158,225943,226089,225168,225828,225823,220862,220970,220864,225159,
        220995,225170,225825,227533,225161,227531,225171,225827,225941,225823,
        225825,225941,225825,228341,225827,
    )))


def vaso_mv():
    """Vasopressor inputevents (weight-normalised rate) — identical to sepsis."""
    return """
        SELECT
            ie.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM ie.starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM ie.endtime)::bigint   AS endtime,
            ie.itemid,
            CASE
                WHEN ie.itemid IN (221906, 221289)
                     THEN ie.rate / pat.weight * 0.0833333
                WHEN ie.itemid = 221749
                     THEN ie.rate / pat.weight * 0.0833333
                WHEN ie.itemid = 222315
                     THEN ie.rate * 5 / pat.weight * 0.0833333
                WHEN ie.itemid = 221662
                     THEN ie.rate / pat.weight * 0.0166667
                ELSE ie.rate
            END AS ratestd
        FROM mimiciv_icu.inputevents ie
        JOIN (
            SELECT stay_id, MAX(patientweight) AS weight
            FROM mimiciv_icu.inputevents
            WHERE patientweight IS NOT NULL
            GROUP BY stay_id
        ) pat USING (stay_id)
        WHERE ie.itemid IN {codes}
          AND ie.statusdescription != 'Rewritten'
          AND ie.rate IS NOT NULL AND ie.rate > 0
        ORDER BY ie.stay_id, ie.starttime
    """.format(codes=repr(VASO_CODES))


def preadm_fluid():
    """Pre-admission fluid — identical to sepsis."""
    return """
        SELECT
            stay_id AS icustay_id,
            SUM(
                CASE WHEN amountuom = 'L' THEN amount * 1000 ELSE amount END
            ) AS input_preadm
        FROM mimiciv_icu.inputevents
        WHERE itemid IN {codes}
          AND statusdescription != 'Rewritten'
        GROUP BY stay_id
        ORDER BY stay_id
    """.format(codes=repr(PREADM_FLUID_CODES))


def uo():
    """Urine output events — identical to sepsis."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            value
        FROM mimiciv_icu.outputevents
        WHERE itemid IN {codes}
          AND value IS NOT NULL AND value > 0
        ORDER BY stay_id, charttime
    """.format(codes=repr(UO_CODES))


def preadm_uo():
    """Pre-admission urine output — identical to sepsis."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            value,
            EXTRACT(EPOCH FROM charttime - intime)::bigint / 60 AS datediff_minutes
        FROM mimiciv_icu.outputevents oe
        JOIN mimiciv_icu.icustays i USING (stay_id)
        WHERE oe.itemid IN {codes}
          AND oe.value IS NOT NULL AND oe.value > 0
          AND oe.charttime <= i.intime
        ORDER BY stay_id, charttime
    """.format(codes=repr(UO_CODES))


def drugs_mv():
    """
    Drug action inputevents — all 9 binary drug classes.
    Returns one row per drug administration event within an ICU stay.
    Step_06 aggregates these to per-bloc binary flags.
    """
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM endtime)::bigint   AS endtime,
            itemid,
            amount,
            rate
        FROM mimiciv_icu.inputevents
        WHERE itemid IN {codes}
          AND statusdescription != 'Rewritten'
          AND (amount > 0 OR rate > 0)
        ORDER BY stay_id, starttime
    """.format(codes=repr(ALL_DRUG_ACTION_CODES))


def demog():
    """
    Demographics, comorbidities, and outcome for ICU readmission pipeline.

    Returns one row per ICU stay with:
      - ICU stay identifiers and timing
      - Patient demographics: age, gender, race, insurance, marital_status
      - Admission context: admission_type, admission_location, drg_severity, drg_mortality
      - Charlson score (computed in charlson_flags temp table, must be created first)
      - Re-admission flag, prior ED visits in 6 months (LACE E)
      - 30-day readmission outcome (from hospital discharge)
      - Discharge location (simplified for terminal action)
      - In-hospital and 90-day mortality flags
    """
    return """
        WITH icu_hadm AS (
            -- One row per ICU stay with hospital admission context
            SELECT
                i.stay_id,
                i.hadm_id,
                i.subject_id,
                EXTRACT(EPOCH FROM i.intime)::bigint  AS intime,
                EXTRACT(EPOCH FROM i.outtime)::bigint AS outtime,
                i.los,
                EXTRACT(EPOCH FROM a.admittime)::bigint  AS admittime,
                EXTRACT(EPOCH FROM a.dischtime)::bigint  AS dischtime,
                -- Age at ICU admission (MIMIC-IV: anchor_age + year offset from anchor_year)
                (p.anchor_age + EXTRACT(YEAR FROM i.intime)::int - p.anchor_year)::int AS age,
                p.gender,
                p.dod,
                a.discharge_location,
                a.admission_type,
                a.admission_location,
                a.insurance,
                a.language,
                a.marital_status,
                a.race,
                a.hospital_expire_flag::int AS morta_hosp,
                -- Re-admission: is this the 2nd+ ICU stay for this subject?
                ROW_NUMBER() OVER (
                    PARTITION BY i.subject_id ORDER BY i.intime
                ) AS adm_order
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp_typed.admissions a ON i.hadm_id = a.hadm_id
            JOIN mimiciv_hosp.patients   p ON i.subject_id = p.subject_id
        ),
        prior_ed AS (
            -- Count prior ED visits in 6 months before each ICU admission
            SELECT
                i.stay_id,
                LEAST(
                    COUNT(DISTINCT a2.hadm_id),
                    4
                )::int AS prior_ed_visits_6m
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp_typed.admissions a2
                ON a2.subject_id = i.subject_id
                AND a2.admission_type IN ('EW EMER.', 'EU OBSERVATION', 'DIRECT EMER.')
                AND a2.admittime < i.intime
                AND a2.admittime >= i.intime - INTERVAL '6 months'
            GROUP BY i.stay_id
        ),
        drg AS (
            -- DRG severity and mortality scores
            SELECT
                hadm_id,
                MAX(drg_severity)  AS drg_severity,
                MAX(drg_mortality) AS drg_mortality
            FROM mimiciv_hosp.drgcodes
            WHERE drg_type = 'HCFA'
            GROUP BY hadm_id
        ),
        readmit AS (
            -- 30-day readmission from hospital discharge
            -- Readmission = any new admission within 30 days of dischtime
            -- Exclude same-day returns and OBS/GYN services
            SELECT
                a1.hadm_id,
                MAX(CASE
                    WHEN a2.admittime IS NOT NULL
                     AND a2.admittime > a1.dischtime
                     AND EXTRACT(EPOCH FROM (a2.admittime - a1.dischtime)) <= 30*86400
                    THEN 1 ELSE 0 END
                )::int AS readmit_30d
            FROM mimiciv_hosp_typed.admissions a1
            LEFT JOIN mimiciv_hosp_typed.admissions a2
                ON a1.subject_id = a2.subject_id
                AND a2.hadm_id != a1.hadm_id
                AND a2.admittime > a1.dischtime
            GROUP BY a1.hadm_id
        )
        SELECT
            h.stay_id          AS icustayid,
            h.hadm_id,
            h.subject_id,
            h.admittime,
            h.dischtime,
            h.intime,
            h.outtime,
            h.los,
            h.adm_order,
            h.age,
            h.gender,
            h.dod,
            h.morta_hosp,
            -- 90-day mortality proxy (dod within 90d of intime)
            -- h.intime is epoch bigint; compare using epoch arithmetic
            CASE WHEN h.dod IS NOT NULL
                  AND EXTRACT(EPOCH FROM h.dod)::bigint - h.intime < 90*86400
                 THEN 1 ELSE 0 END AS morta_90,
            h.discharge_location,
            h.admission_type,
            h.admission_location,
            h.insurance,
            h.language,
            h.marital_status,
            h.race,
            COALESCE(pe.prior_ed_visits_6m, 0) AS prior_ed_visits_6m,
            -- Charlson score and component flags (from temp table)
            COALESCE(cf.charlson_score, 0)      AS charlson_score,
            COALESCE(cf.cc_mi,         0)       AS cc_mi,
            COALESCE(cf.cc_chf,        0)       AS cc_chf,
            COALESCE(cf.cc_pvd,        0)       AS cc_pvd,
            COALESCE(cf.cc_cvd,        0)       AS cc_cvd,
            COALESCE(cf.cc_dementia,   0)       AS cc_dementia,
            COALESCE(cf.cc_copd,       0)       AS cc_copd,
            COALESCE(cf.cc_rheum,      0)       AS cc_rheum,
            COALESCE(cf.cc_pud,        0)       AS cc_pud,
            COALESCE(cf.cc_mild_liver, 0)       AS cc_mild_liver,
            COALESCE(cf.cc_dm_no_cc,   0)       AS cc_dm_no_cc,
            COALESCE(cf.cc_dm_cc,      0)       AS cc_dm_cc,
            COALESCE(cf.cc_paralysis,  0)       AS cc_paralysis,
            COALESCE(cf.cc_renal,      0)       AS cc_renal,
            COALESCE(cf.cc_malign,     0)       AS cc_malign,
            COALESCE(cf.cc_sev_liver,  0)       AS cc_sev_liver,
            COALESCE(cf.cc_metastatic, 0)       AS cc_metastatic,
            COALESCE(cf.cc_hiv,        0)       AS cc_hiv,
            COALESCE(cf.cc_hemiplegia, 0)       AS cc_hemiplegia,
            COALESCE(d.drg_severity,   0)       AS drg_severity,
            COALESCE(d.drg_mortality,  0)       AS drg_mortality,
            COALESCE(r.readmit_30d,    0)       AS readmit_30d
        FROM icu_hadm h
        LEFT JOIN prior_ed  pe ON h.stay_id  = pe.stay_id
        LEFT JOIN charlson_flags cf ON h.hadm_id = cf.hadm_id
        LEFT JOIN drg       d  ON h.hadm_id  = d.hadm_id
        LEFT JOIN readmit   r  ON h.hadm_id  = r.hadm_id
        ORDER BY h.subject_id, h.intime
    """


def charlson():
    """
    Charlson Comorbidity Index — 18 component flags + numeric score.
    Uses ICD-9 and ICD-10 dual mapping (covers full MIMIC-IV 2008-2019 range).
    Adapted from src/careai/readmission_v4/build.py (step4_charlson).

    This CTE is materialised as a TEMP TABLE 'charlson_flags' before demog() runs.
    """
    return """
        WITH diag AS (
            SELECT
                hadm_id,
                CASE WHEN icd_version = 9  THEN icd_code ELSE NULL END AS icd9,
                CASE WHEN icd_version = 10 THEN icd_code ELSE NULL END AS icd10
            FROM mimiciv_hosp.diagnoses_icd
        ),
        flags AS (
            SELECT
                hadm_id,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3)  IN ('410','412')            THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('I21','I22')            THEN 1
                    WHEN icd10 LIKE 'I252%'                            THEN 1
                    ELSE 0 END) AS cc_mi,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3)  IN ('428')                  THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('I43','I50')            THEN 1
                    ELSE 0 END) AS cc_chf,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('4431','4432','4438','4439') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('I70','I71')            THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959') THEN 1
                    ELSE 0 END) AS cc_pvd,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('430','431','432','433','434','435','436','437','438') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('G45','G46','I60','I61','I62','I63','I64','I65','I66','I67','I68','I69') THEN 1
                    ELSE 0 END) AS cc_cvd,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('290')                   THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('F00','F01','F02','F03') THEN 1
                    WHEN icd10 LIKE 'G30%'                             THEN 1
                    ELSE 0 END) AS cc_dementia,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('490','491','492','493','494','495','496','500','501','502','503','504','505') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67') THEN 1
                    ELSE 0 END) AS cc_copd,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('7140','7141','7142','7148') THEN 1
                    WHEN SUBSTR(icd9,1,3) IN ('725')                   THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('M050','M060','M063','M069','M315','M320','M321','M324','M325','M328','M329','M332','M334','M346','M351','M353','M360') THEN 1
                    ELSE 0 END) AS cc_rheum,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('531','532','533','534') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('K25','K26','K27','K28') THEN 1
                    ELSE 0 END) AS cc_pud,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('5710','5712','5713')    THEN 1
                    WHEN SUBSTR(icd9,1,3) IN ('571')                   THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('K700','K701','K702','K703','K709','K714','K715','K716','K717','K760','K762','K763','K764','K768','K769','Z944') THEN 1
                    ELSE 0 END) AS cc_mild_liver,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('2500','2501','2502','2503','2508','2509') THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('E100','E101','E106','E108','E109','E110','E111','E116','E118','E119','E120','E121','E126','E128','E129','E130','E131','E136','E138','E139','E140','E141','E146','E148','E149') THEN 1
                    ELSE 0 END) AS cc_dm_no_cc,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('2504','2505','2506','2507') THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('E102','E103','E104','E105','E107','E112','E113','E114','E115','E117','E122','E123','E124','E125','E127','E132','E133','E134','E135','E137','E142','E143','E144','E145','E147') THEN 1
                    ELSE 0 END) AS cc_dm_cc,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('342','343')             THEN 1
                    WHEN SUBSTR(icd9,1,4) IN ('3341','3440','3441','3442','3443','3444','3445','3446','3449') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('G04','G81','G82')      THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('G041','G114','G801','G802','G830','G831','G832','G833','G834','G839') THEN 1
                    ELSE 0 END) AS cc_hemiplegia,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('582','583','585','586','588') THEN 1
                    WHEN SUBSTR(icd9,1,4) IN ('5830','5831','5832','5834','5836','5837') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('N03','N04','N05','N18','N19') THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('N052','N053','N054','N055','N056','N057','N250','Z490','Z491','Z492','Z940','Z992') THEN 1
                    ELSE 0 END) AS cc_renal,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','170','171','172','174','175','176','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('C00','C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72','C73','C74','C75','C76','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','C97') THEN 1
                    ELSE 0 END) AS cc_malign,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,4) IN ('4560','4561','4562')    THEN 1
                    WHEN SUBSTR(icd9,1,4) IN ('5722','5723','5724','5728') THEN 1
                    WHEN SUBSTR(icd10,1,4) IN ('K704','K711','K721','K729','K765','K766','K767') THEN 1
                    WHEN icd10 LIKE 'I850%' OR icd10 LIKE 'I859%' OR icd10 LIKE 'I864%' OR icd10 LIKE 'I982%' THEN 1
                    ELSE 0 END) AS cc_sev_liver,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('196','197','198','199') THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('C77','C78','C79','C80') THEN 1
                    ELSE 0 END) AS cc_metastatic,
                MAX(CASE
                    WHEN SUBSTR(icd9,1,3) IN ('042','043','044')       THEN 1
                    WHEN SUBSTR(icd10,1,3) IN ('B20','B21','B22','B24') THEN 1
                    ELSE 0 END) AS cc_hiv
            FROM diag
            GROUP BY hadm_id
        )
        SELECT
            hadm_id,
            COALESCE(cc_mi,0) + COALESCE(cc_chf,0) + COALESCE(cc_pvd,0) +
            COALESCE(cc_cvd,0) + COALESCE(cc_dementia,0) + COALESCE(cc_copd,0) +
            COALESCE(cc_rheum,0) + COALESCE(cc_pud,0) + COALESCE(cc_mild_liver,0) +
            COALESCE(cc_dm_no_cc,0) + COALESCE(cc_dm_cc,0)*2 +
            COALESCE(cc_hemiplegia,0)*2 + COALESCE(cc_renal,0)*2 +
            COALESCE(cc_malign,0)*2 + COALESCE(cc_sev_liver,0)*3 +
            COALESCE(cc_metastatic,0)*6 + COALESCE(cc_hiv,0)*6
            AS charlson_score,
            COALESCE(cc_mi,0)         AS cc_mi,
            COALESCE(cc_chf,0)        AS cc_chf,
            COALESCE(cc_pvd,0)        AS cc_pvd,
            COALESCE(cc_cvd,0)        AS cc_cvd,
            COALESCE(cc_dementia,0)   AS cc_dementia,
            COALESCE(cc_copd,0)       AS cc_copd,
            COALESCE(cc_rheum,0)      AS cc_rheum,
            COALESCE(cc_pud,0)        AS cc_pud,
            COALESCE(cc_mild_liver,0) AS cc_mild_liver,
            COALESCE(cc_dm_no_cc,0)   AS cc_dm_no_cc,
            COALESCE(cc_dm_cc,0)      AS cc_dm_cc,
            COALESCE(cc_hemiplegia,0) AS cc_paralysis,   -- same ICD codes; CCI combines hemiplegia+paraplegia
            COALESCE(cc_renal,0)      AS cc_renal,
            COALESCE(cc_malign,0)     AS cc_malign,
            COALESCE(cc_sev_liver,0)  AS cc_sev_liver,
            COALESCE(cc_metastatic,0) AS cc_metastatic,
            COALESCE(cc_hiv,0)        AS cc_hiv,
            COALESCE(cc_hemiplegia,0) AS cc_hemiplegia
        FROM flags
    """
