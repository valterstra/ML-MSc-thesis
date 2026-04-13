"""
PostgreSQL SQL queries for the sepsis dataset extraction pipeline.
Faithful translation of ai_clinician/data_extraction/sql/queries.py (BigQuery)
to standard PostgreSQL syntax.

Changes from original:
  - Table names: `physionet-data.mimic_X.Y` -> mimiciv_X.Y
                 `physionet-data.mimic_core.X` -> mimiciv_hosp.X  (core merged into hosp in MIMIC-IV PG)
  - UNIX_SECONDS(TIMESTAMP(x))         -> EXTRACT(EPOCH FROM x)::bigint
  - DATETIME_DIFF(a, b, second)        -> EXTRACT(EPOCH FROM (a - b))::bigint
  - TIMESTAMP_DIFF(a, b, HOUR)         -> EXTRACT(EPOCH FROM (a - b))::bigint / 3600
  - CAST(bool AS int)                  -> (bool)::integer
  - ROUND(CAST(x as numeric), 3)       -> ROUND(x::numeric, 3)
  - first_careunit: full MIMIC-IV names mapped to numeric unit codes
  - mechvent table: corrected to mimiciv_icu.chartevents (was mimiciv_hosp.chartevents in original)
Everything else (codes, thresholds, logic) is identical.
"""

# ---------------------------------------------------------------------------
# Item ID lists — identical to original queries.py
# ---------------------------------------------------------------------------

ANTIBIOTIC_GSN_CODES = (
    '002542','002543','007371','008873','008877','008879','008880','008935',
    '008941','008942','008943','008944','008983','008984','008990','008991',
    '008992','008995','008996','008998','009043','009046','009065','009066',
    '009136','009137','009162','009164','009165','009171','009182','009189',
    '009213','009214','009218','009219','009221','009226','009227','009235',
    '009242','009263','009273','009284','009298','009299','009310','009322',
    '009323','009326','009327','009339','009346','009351','009354','009362',
    '009394','009395','009396','009509','009510','009511','009544','009585',
    '009591','009592','009630','013023','013645','013723','013724','013725',
    '014182','014500','015979','016368','016373','016408','016931','016932',
    '016949','018636','018637','018766','019283','021187','021205','021735',
    '021871','023372','023989','024095','024194','024668','025080','026721',
    '027252','027465','027470','029325','029927','029928','037042','039551',
    '039806','040819','041798','043350','043879','044143','045131','045132',
    '046771','047797','048077','048262','048266','048292','049835','050442',
    '050443','051932','052050','060365','066295','067471',
)

CHARTEVENT_CODES = (
    226707, 581, 198, 228096, 211, 220179, 220181, 8368, 220210, 220277, 3655,
    223761, 220074, 492, 491, 8448, 116, 626, 467, 223835, 190, 470, 220339,
    224686, 224687, 224697, 224695, 224696, 226730, 580, 220045, 225309, 220052,
    8441, 3337, 646, 223762, 678, 113, 1372, 3420, 471, 506, 224684, 450, 444,
    535, 543, 224639, 6701, 225312, 225310, 224422, 834, 1366, 160, 223834, 505,
    684, 448, 226512, 6, 224322, 8555, 618, 228368, 727, 227287, 224700, 224421,
    445, 227243, 6702, 8440, 3603, 228177, 194, 3083, 224167, 443, 615, 224691,
    2566, 51, 52, 654, 455, 456, 3050, 681, 2311, 220059, 220061, 220060, 226732,
)

CULTURE_CODES = (
    6035,3333,938,941,942,4855,6043,2929,225401,225437,225444,225451,225454,
    225814,225816,225817,225818,225722,225723,225724,225725,225726,225727,
    225728,225729,225730,225731,225732,225733,227726,70006,70011,70012,70013,
    70014,70016,70024,70037,70041,225734,225735,225736,225768,70055,70057,70060,
    70063,70075,70083,226131,80220,
)

INPUTEVENT_CODES = (
    225158,225943,226089,225168,225828,225823,220862,220970,220864,225159,
    220995,225170,225825,227533,225161,227531,225171,225827,225941,225823,
    225825,225941,225825,228341,225827,30018,30021,30015,30296,30020,30066,
    30001,30030,30060,30005,30321,30061,30009,30179,30190,30143,30160,
    30008,30168,30186,30211,30353,30159,30007,30185,30063,30094,30352,30014,
    30011,30210,46493,45399,46516,40850,30176,30161,30381,30315,42742,30180,
    46087,41491,30004,42698,42244,
)

LABS_CE_CODES = (
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
    227039, 220235, 226062, 227036,
)

LABS_LE_CODES = (
    50971,50822,50824,50806,50931,51081,50885,51003,51222,50810,51301,50983,
    50902,50809,51006,50912,50960,50893,50808,50804,50878,50861,51464,50883,
    50976,50862,51002,50889,50811,51221,51279,51300,51265,51275,51274,51237,
    50820,50821,50818,50802,50813,50882,50803,52167,52166,52165,52923,51624,
    52647,
)

MECHVENT_MEASUREMENT_CODES = (
    445, 448, 449, 450, 1340, 1486, 1600, 224687,         # minute volume
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686, # tidal volume
    218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, # resp pressure
    221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, # insp pressure
    543,                                                   # plateau pressure
    5865, 5866, 224707, 224709, 224705, 224706,            # APRV pressure
    60, 437, 505, 506, 686, 220339, 224700,                # PEEP
    3459,                                                  # high pressure relief
    501, 502, 503, 224702,                                 # PCV
    223, 667, 668, 669, 670, 671, 672,                    # TCPCV
    157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810, # ETT
    224701,                                                # PSVlevel
)

MECHVENT_CODES = (640, 720, 467) + MECHVENT_MEASUREMENT_CODES

PREADM_FLUID_CODES = (
    30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
    226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
    227071,227072,
)

UO_CODES = (
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405,
    40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489, 226584,
    226563, 226564, 226565, 226557, 226558,
)

VASO_CODES = (
    30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289,
    222315, 221662, 30043, 30307,
)

COMORBIDITY_FIELDS = [
    'congestive_heart_failure', 'cardiac_arrhythmias', 'valvular_disease',
    'pulmonary_circulation', 'peripheral_vascular', 'hypertension', 'paralysis',
    'other_neurological', 'chronic_pulmonary', 'diabetes_uncomplicated',
    'diabetes_complicated', 'hypothyroidism', 'renal_failure', 'liver_disease',
    'peptic_ulcer', 'aids', 'lymphoma', 'metastatic_cancer', 'solid_tumor',
    'rheumatoid_arthritis', 'coagulopathy', 'obesity', 'weight_loss',
    'fluid_electrolyte', 'blood_loss_anemia', 'deficiency_anemias', 'alcohol_abuse',
    'drug_abuse', 'psychoses', 'depression',
]

# ---------------------------------------------------------------------------
# Query functions — PostgreSQL syntax
# ---------------------------------------------------------------------------

def abx():
    """Antibiotic prescriptions joined to ICU stays."""
    return """
        SELECT
            p.hadm_id,
            i.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM p.starttime)::bigint AS startdate,
            EXTRACT(EPOCH FROM p.stoptime)::bigint  AS enddate,
            p.gsn, p.ndc, p.dose_val_rx AS dose_val, p.dose_unit_rx AS dose_unit, p.route
        FROM mimiciv_hosp.prescriptions p
        LEFT JOIN mimiciv_icu.icustays i ON p.hadm_id = i.hadm_id
        WHERE p.gsn IN {gsn}
        ORDER BY p.hadm_id, i.stay_id
    """.format(gsn=repr(ANTIBIOTIC_GSN_CODES))


def ce(min_stay, max_stay):
    """
    Chart events (vitals, ventilator settings, code status, etc.).
    Queried in batches over stay_id ranges for memory efficiency.
    """
    return """
        SELECT DISTINCT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            CASE
                WHEN lower(value) = 'none'                                            THEN 0
                WHEN lower(value) = 'ventilator'                                      THEN 1
                WHEN lower(value) IN ('cannula','nasal cannula','high flow nasal cannula') THEN 2
                WHEN lower(value) = 'face tent'                                       THEN 3
                WHEN lower(value) = 'aerosol-cool'                                    THEN 4
                WHEN lower(value) = 'trach mask'                                      THEN 5
                WHEN lower(value) = 'hi flow neb'                                     THEN 6
                WHEN lower(value) = 'non-rebreather'                                  THEN 7
                WHEN lower(value) = ''                                                 THEN 8
                WHEN lower(value) = 'venti mask'                                      THEN 9
                WHEN lower(value) = 'medium conc mask'                                THEN 10
                ELSE valuenum
            END AS valuenum
        FROM mimiciv_icu.chartevents
        WHERE stay_id >= {min_stay}
          AND stay_id < {max_stay}
          AND value IS NOT NULL
          AND itemid IN {codes}
        ORDER BY stay_id, charttime
    """.format(min_stay=min_stay, max_stay=max_stay, codes=repr(CHARTEVENT_CODES))


def culture():
    """Culture events from chartevents (blood/urine/CSF/sputum cultures)."""
    return """
        SELECT
            subject_id, hadm_id,
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid
        FROM mimiciv_icu.chartevents
        WHERE itemid IN {codes}
        ORDER BY subject_id, hadm_id, charttime
    """.format(codes=repr(CULTURE_CODES))


def elixhauser():
    """
    Elixhauser-Quan comorbidity flags (30 conditions) from ICD-9/ICD-10 codes.
    Adapted from ai_clinician/data_extraction/sql/elixhauser.sql — only table
    name changed (mimic_core -> mimic_hosp).
    """
    return """
        WITH diag AS (
            SELECT
                hadm_id,
                CASE WHEN icd_version = 9  THEN icd_code ELSE NULL END AS icd9_code,
                CASE WHEN icd_version = 10 THEN icd_code ELSE NULL END AS icd10_code
            FROM mimiciv_hosp_typed.diagnoses_icd
        ),
        com AS (
            SELECT
                ad.hadm_id,
                MAX(CASE
                    WHEN icd9_code IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('4254','4255','4257','4258','4259') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('428') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','P290') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I43','I50') THEN 1
                    ELSE 0 END) AS chf,
                MAX(CASE
                    WHEN icd9_code IN ('42613','42610','42612','99601','99604') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I441','I442','I443','I4566','I459','R000','R001','R008','T821','Z450','Z950') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I47','I48','I49') THEN 1
                    ELSE 0 END) AS arrhy,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('0932','7463','7464','7465','7466','V422','V433') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('394','395','396','397','424') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('A520','I091','I098','Q230','Q231','Q232','Q233','Z952','Z953','Z954') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I05','I06','I07','I08','I34','I35','I36','I37','I38','I39') THEN 1
                    ELSE 0 END) AS valve,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('4150','4151','4170','4178','4179') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('416') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I280','I288','I289') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I26','I27') THEN 1
                    ELSE 0 END) AS pulmcirc,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('440','441') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I70','I71') THEN 1
                    ELSE 0 END) AS perivasc,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,3)  IN ('401') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I10') THEN 1
                    ELSE 0 END) AS htn,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,3)  IN ('402','403','404','405') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('I11','I12','I13','I15') THEN 1
                    ELSE 0 END) AS htncx,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('3341','3440','3441','3442','3443','3444','3445','3446','3449') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('342','343') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('G041','G114','G801','G802','G830','G831','G832','G833','G834','G839') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('G81','G82') THEN 1
                    ELSE 0 END) AS para,
                MAX(CASE
                    WHEN icd9_code IN ('33392') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('334','335','340','341','345') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('G254','G255','G312','G318','G319','G931','G934','R470') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('G10','G11','G12','G13','G20','G21','G22','G32','G35','G36','G37','G40','G41','R56') THEN 1
                    ELSE 0 END) AS neuro,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('4168','4169','5064','5081','5088') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('490','491','492','493','494','495','496','500','501','502','503','504','505') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I278','I279','J684','J701','J703') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67') THEN 1
                    ELSE 0 END) AS chrnlung,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2500','2501','2502','2503') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('E100','E101','E109','E110','E111','E119','E120','E121','E129','E130','E131','E139','E140','E141','E149') THEN 1
                    ELSE 0 END) AS dm,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2504','2505','2506','2507','2508','2509') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('E102','E103','E104','E105','E106','E107','E108','E112','E113','E114','E115','E116','E117','E118','E122','E123','E124','E125','E126','E127','E128','E132','E133','E134','E135','E136','E137','E138','E142','E143','E144','E145','E146','E147','E148') THEN 1
                    ELSE 0 END) AS dmcx,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2409','2461','2468') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('243','244') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('E890') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('E0','E1','E2','E3') THEN 1
                    ELSE 0 END) AS hypothy,
                MAX(CASE
                    WHEN icd9_code IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('5880','V420','V451') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('585','586','V56') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I120','I131','N250','Z490','Z491','Z492','Z940','Z992') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('N18','N19') THEN 1
                    ELSE 0 END) AS renlfail,
                MAX(CASE
                    WHEN icd9_code IN ('07022','07023','07032','07033','07044','07054') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('570','571') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('I864','I982','K711','K713','K714','K715','K717','K760','K762','K763','K764','K765','K766','K767','K768','K769','Z944') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('B18','I85','K70','K72','K73','K74') THEN 1
                    ELSE 0 END) AS liver,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('5317','5319','5327','5329','5337','5339','5347','5349') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('K257','K259','K267','K269','K277','K279','K287','K289') THEN 1
                    ELSE 0 END) AS ulcer,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,3)  IN ('042','043','044') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('B20','B21','B22','B24') THEN 1
                    ELSE 0 END) AS aids,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2030','2386') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('200','201','202') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('C900','C902') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('C81','C82','C83','C84','C85','C88','C96') THEN 1
                    ELSE 0 END) AS lymph,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,3) IN ('196','197','198','199') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('C77','C78','C79','C80') THEN 1
                    ELSE 0 END) AS mets,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,3) IN (
                        '140','141','142','143','144','145','146','147','148','149','150','151','152',
                        '153','154','155','156','157','158','159','160','161','162','163','164','165',
                        '166','167','168','169','170','171','172','174','175','176','177','178','179',
                        '180','181','182','183','184','185','186','187','188','189','190','191','192',
                        '193','194','195'
                    ) THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN (
                        'C0','C1','C2','C3','C4','C5','C6','C7','C8','C9',
                        'C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20',
                        'C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C34',
                        'C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49',
                        'C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61',
                        'C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72',
                        'C73','C74','C75','C76','C97'
                    ) THEN 1
                    ELSE 0 END) AS tumor,
                MAX(CASE
                    WHEN icd9_code IN ('72889','72930') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('446','714','720','725') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('L940','L941','L943','M120','M123','M310','M311','M312','M313','M461','M468','M469') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('M05','M06','M08','M30','M32','M33','M34','M35','M45') THEN 1
                    ELSE 0 END) AS arth,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2871','2873','2874','2875') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('286') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('D691','D693','D694','D695','D696') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('D65','D66','D67','D68') THEN 1
                    ELSE 0 END) AS coag,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2780') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('E66') THEN 1
                    ELSE 0 END) AS obese,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('7832','7994') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('260','261','262','263') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('R634','R64') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('E40','E41','E42','E43','E44','E45','E46') THEN 1
                    ELSE 0 END) AS wghtloss,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2536') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('276') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('E222') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('E86','E87') THEN 1
                    ELSE 0 END) AS lytes,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2800') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('D500') THEN 1
                    ELSE 0 END) AS bldloss,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2801','2808','2809') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('281') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('D508','D509') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('D51','D52','D53') THEN 1
                    ELSE 0 END) AS anemdef,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4) IN ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('980') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('F10','E52','G621','I426','K292','K700','K703','K709','Z502','Z714','Z721') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('T51') THEN 1
                    ELSE 0 END) AS alcohol,
                MAX(CASE
                    WHEN icd9_code IN ('V6542') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('3052','3053','3054','3055','3056','3057','3058','3059') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('292','304') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('Z715','Z722') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('F11','F12','F13','F14','F15','F16','F18','F19') THEN 1
                    ELSE 0 END) AS drug,
                MAX(CASE
                    WHEN icd9_code IN ('29604','29614','29644','29654') THEN 1
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2938') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('295','297','298') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('F302','F312','F315') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('F20','F22','F23','F24','F25','F28','F29') THEN 1
                    ELSE 0 END) AS psych,
                MAX(CASE
                    WHEN SUBSTR(icd9_code,1,4)  IN ('2962','2963','2965','3004') THEN 1
                    WHEN SUBSTR(icd9_code,1,3)  IN ('309','311') THEN 1
                    WHEN SUBSTR(icd10_code,1,4) IN ('F204','F313','F314','F315','F341','F412','F432') THEN 1
                    WHEN SUBSTR(icd10_code,1,3) IN ('F32','F33') THEN 1
                    ELSE 0 END) AS depress
            FROM mimiciv_hosp_typed.admissions ad
            LEFT JOIN diag ON ad.hadm_id = diag.hadm_id
            GROUP BY ad.hadm_id
        )
        SELECT
            adm.hadm_id,
            chf   AS congestive_heart_failure,
            arrhy AS cardiac_arrhythmias,
            valve AS valvular_disease,
            pulmcirc AS pulmonary_circulation,
            perivasc AS peripheral_vascular,
            CASE WHEN htn = 1 OR htncx = 1 THEN 1 ELSE 0 END AS hypertension,
            para  AS paralysis,
            neuro AS other_neurological,
            chrnlung AS chronic_pulmonary,
            CASE WHEN dmcx = 1 THEN 0 WHEN dm = 1 THEN 1 ELSE 0 END AS diabetes_uncomplicated,
            dmcx  AS diabetes_complicated,
            hypothy AS hypothyroidism,
            renlfail AS renal_failure,
            liver AS liver_disease,
            ulcer AS peptic_ulcer,
            aids  AS aids,
            lymph AS lymphoma,
            mets  AS metastatic_cancer,
            CASE WHEN mets = 1 THEN 0 WHEN tumor = 1 THEN 1 ELSE 0 END AS solid_tumor,
            arth  AS rheumatoid_arthritis,
            coag  AS coagulopathy,
            obese AS obesity,
            wghtloss AS weight_loss,
            lytes AS fluid_electrolyte,
            bldloss  AS blood_loss_anemia,
            anemdef  AS deficiency_anemias,
            alcohol  AS alcohol_abuse,
            drug  AS drug_abuse,
            psych AS psychoses,
            depress AS depression
        FROM mimiciv_hosp_typed.admissions adm
        LEFT JOIN com ON adm.hadm_id = com.hadm_id
        ORDER BY adm.hadm_id
    """


def demog():
    """
    Demographic information: admission/discharge times, ICU stay times,
    age, gender, mortality, Elixhauser score, and 30-day readmission flag.
    NOTE: The Elixhauser score is summed from the comorbidity flags computed
    separately and joined here. In MIMIC-IV PG, mimic_core -> mimiciv_hosp.
    MIMIC-IV first_careunit values are full strings; mapped to numeric codes
    matching the original pipeline (MICU=1, SICU=2, TSICU=3, CVICU=4, NICU=5, CCU=6).
    readmit_30d: 1 if the patient was admitted again within 30 days of discharge.
    """
    return """
        SELECT
            ad.subject_id,
            ad.hadm_id,
            i.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM ad.admittime)::bigint AS admittime,
            EXTRACT(EPOCH FROM ad.dischtime)::bigint AS dischtime,
            ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
            CASE
                WHEN i.first_careunit IN ('Neuro Surgical Intensive Care Unit (Neuro SICU)','Neuro Intermediate','Neuro Stepdown') THEN 5
                WHEN i.first_careunit IN ('Surgical Intensive Care Unit (SICU)') THEN 2
                WHEN i.first_careunit IN ('Cardiac Vascular Intensive Care Unit (CVICU)') THEN 4
                WHEN i.first_careunit IN ('Coronary Care Unit (CCU)') THEN 6
                WHEN i.first_careunit IN ('Medical Intensive Care Unit (MICU)','Medical/Surgical Intensive Care Unit (MICU/SICU)') THEN 1
                WHEN i.first_careunit IN ('Trauma SICU (TSICU)') THEN 3
            END AS unit,
            EXTRACT(EPOCH FROM i.intime)::bigint  AS intime,
            EXTRACT(EPOCH FROM i.outtime)::bigint AS outtime,
            i.los,
            (EXTRACT(YEAR FROM i.intime) - p.anchor_year + p.anchor_age)::integer AS age,
            (p.anchor_year - p.anchor_age) AS dob,
            EXTRACT(EPOCH FROM p.dod)::bigint AS dod,
            (p.dod IS NOT NULL)::integer AS expire_flag,
            CASE WHEN p.gender = 'M' THEN 1 WHEN p.gender = 'F' THEN 2 END AS gender,
            (EXTRACT(EPOCH FROM (p.dod - ad.dischtime)) <= 24*3600 AND p.dod IS NOT NULL)::integer AS morta_hosp,
            (EXTRACT(EPOCH FROM (p.dod - i.intime))    <= 90*24*3600 AND p.dod IS NOT NULL)::integer AS morta_90,
            elix.elixhauser,
            (EXISTS (
                SELECT 1 FROM mimiciv_hosp_typed.admissions next_ad
                WHERE next_ad.subject_id = ad.subject_id
                  AND next_ad.hadm_id != ad.hadm_id
                  AND next_ad.admittime > ad.dischtime
                  AND next_ad.admittime <= ad.dischtime + INTERVAL '30 days'
            ))::integer AS readmit_30d
        FROM mimiciv_hosp_typed.admissions ad
        JOIN mimiciv_icu.icustays   i    ON ad.hadm_id    = i.hadm_id
        JOIN mimiciv_hosp_typed.patients  p    ON ad.subject_id = p.subject_id
        LEFT JOIN (
            SELECT hadm_id,
                congestive_heart_failure + cardiac_arrhythmias + valvular_disease +
                pulmonary_circulation + peripheral_vascular + hypertension + paralysis +
                other_neurological + chronic_pulmonary + diabetes_uncomplicated +
                diabetes_complicated + hypothyroidism + renal_failure + liver_disease +
                peptic_ulcer + aids + lymphoma + metastatic_cancer + solid_tumor +
                rheumatoid_arthritis + coagulopathy + obesity + weight_loss +
                fluid_electrolyte + blood_loss_anemia + deficiency_anemias +
                alcohol_abuse + drug_abuse + psychoses + depression AS elixhauser
            FROM elixhauser_flags
        ) elix ON ad.hadm_id = elix.hadm_id
        ORDER BY ad.subject_id ASC, i.intime ASC
    """
    # Note: elixhauser_flags is a temporary table created by step_01_extract.py
    # before running this query. See step_01_extract.py for details.


def fluid_mv():
    """
    Real-time IV fluid input from inputevents (Metavision).
    Tonicity-corrected equivalent volume (tev) computed per item.
    """
    return """
        WITH t1 AS (
            SELECT
                stay_id AS icustay_id,
                EXTRACT(EPOCH FROM starttime)::bigint AS starttime,
                EXTRACT(EPOCH FROM endtime)::bigint   AS endtime,
                itemid,
                amount,
                rate,
                CASE
                    WHEN itemid IN (30176,30315)                                              THEN amount * 0.25
                    WHEN itemid IN (30161)                                                    THEN amount * 0.3
                    WHEN itemid IN (30020,30015,225823,30321,30186,30211,30353,42742,42244,225159) THEN amount * 0.5
                    WHEN itemid IN (227531)                                                   THEN amount * 2.75
                    WHEN itemid IN (30143,225161)                                             THEN amount * 3
                    WHEN itemid IN (30009,220862)                                             THEN amount * 5
                    WHEN itemid IN (30030,220995,227533)                                      THEN amount * 6.66
                    WHEN itemid IN (228341)                                                   THEN amount * 8
                    ELSE amount
                END AS tev
            FROM mimiciv_icu.inputevents
            WHERE stay_id IS NOT NULL
              AND amount IS NOT NULL
              AND itemid IN {items}
        )
        SELECT
            icustay_id,
            starttime,
            endtime,
            itemid,
            ROUND(amount::numeric, 3) AS amount,
            ROUND(rate::numeric,   3) AS rate,
            ROUND(tev::numeric,    3) AS tev
        FROM t1
        ORDER BY icustay_id, starttime, itemid
    """.format(items=repr(INPUTEVENT_CODES))


def labs_ce():
    """Lab values extracted from chartevents (ICU-charted labs)."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            valuenum
        FROM mimiciv_icu.chartevents
        WHERE valuenum IS NOT NULL
          AND stay_id IS NOT NULL
          AND itemid IN {codes}
        ORDER BY stay_id, charttime, itemid
    """.format(codes=repr(LABS_CE_CODES))


def labs_le():
    """Lab values extracted from labevents (hospital lab system)."""
    return """
        SELECT
            xx.icustay_id,
            EXTRACT(EPOCH FROM f.charttime)::bigint AS charttime,
            f.itemid,
            f.valuenum
        FROM (
            SELECT hadm_id, stay_id AS icustay_id, intime, outtime
            FROM mimiciv_icu.icustays
            GROUP BY hadm_id, stay_id, intime, outtime
        ) AS xx
        INNER JOIN mimiciv_hosp_typed.labevents f ON f.hadm_id = xx.hadm_id
        WHERE EXTRACT(EPOCH FROM (f.charttime - xx.intime))  >= 24*3600
          AND EXTRACT(EPOCH FROM (xx.outtime - f.charttime)) >= 24*3600
          AND f.itemid IN {codes}
          AND f.valuenum IS NOT NULL
        ORDER BY f.hadm_id, xx.icustay_id, f.charttime, f.itemid
    """.format(codes=repr(LABS_LE_CODES))


def mechvent():
    """
    Mechanical ventilation flags derived from chartevents.
    Supplemented in MIMIC-IV by mechvent_pe() from procedureevents.
    NOTE: corrected from mimiciv_hosp.chartevents to mimiciv_icu.chartevents
    (the original had a typo for the MIMIC-IV table name).
    """
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            MAX(CASE
                WHEN itemid IS NULL OR value IS NULL THEN 0
                WHEN itemid = 720 AND value != 'Other/Remarks' THEN 1
                WHEN itemid = 467 AND value = 'Ventilator'     THEN 1
                WHEN itemid IN {measurement_codes}              THEN 1
                ELSE 0
            END) AS mechvent,
            MAX(CASE
                WHEN itemid IS NULL OR value IS NULL THEN 0
                WHEN itemid = 640 AND value IN ('Extubated','Self Extubation') THEN 1
                ELSE 0
            END) AS extubated,
            MAX(CASE
                WHEN itemid IS NULL OR value IS NULL THEN 0
                WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
                ELSE 0
            END) AS selfextubated
        FROM mimiciv_icu.chartevents
        WHERE value IS NOT NULL
          AND itemid IN {codes}
        GROUP BY stay_id, charttime
    """.format(
        codes=repr(MECHVENT_CODES),
        measurement_codes=repr(MECHVENT_MEASUREMENT_CODES),
    )


def mechvent_pe():
    """Mechanical ventilation from procedureevents (MIMIC-IV only)."""
    return """
        SELECT
            subject_id,
            hadm_id,
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM endtime)::bigint   AS endtime,
            CASE WHEN itemid IN (225792,225794,224385,225433) THEN 1 ELSE 0 END AS mechvent,
            CASE WHEN itemid IN (227194,227712,225477,225468) THEN 1 ELSE 0 END AS extubated,
            CASE WHEN itemid = 225468                         THEN 1 ELSE 0 END AS selfextubated,
            itemid,
            CASE
                WHEN valueuom = 'hour' THEN value * 60
                WHEN valueuom = 'min'  THEN value
                WHEN valueuom = 'day'  THEN value * 60 * 24
                ELSE value
            END AS value
        FROM mimiciv_icu.procedureevents
        WHERE itemid IN (225792,225794,227194,227712,224385,225433,225468,225477)
    """


def microbio():
    """All microbiology events (cultures, sensitivities)."""
    return """
        SELECT
            m.subject_id,
            m.hadm_id,
            i.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM m.charttime)::bigint  AS charttime,
            EXTRACT(EPOCH FROM m.chartdate::timestamp)::bigint AS chartdate,
            NULL::integer AS org_itemid,
            NULL::integer AS spec_itemid,
            NULL::integer AS ab_itemid,
            m.interpretation
        FROM mimiciv_hosp.microbiologyevents m
        LEFT JOIN mimiciv_icu.icustays i
            ON m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id
    """


def preadm_fluid():
    """Pre-admission fluid totals from inputevents."""
    return """
        WITH mv AS (
            SELECT ie.stay_id AS icustay_id, SUM(ie.amount) AS sum
            FROM mimiciv_icu.inputevents ie
            JOIN mimiciv_icu.d_items ci ON ie.itemid = ci.itemid
            WHERE ie.itemid IN {codes}
            GROUP BY ie.stay_id
        )
        SELECT
            pt.stay_id AS icustay_id,
            COALESCE(mv.sum, NULL) AS input_preadm
        FROM mimiciv_icu.icustays pt
        LEFT JOIN mv ON mv.icustay_id = pt.stay_id
        ORDER BY pt.stay_id
    """.format(codes=repr(PREADM_FLUID_CODES))


def preadm_uo():
    """
    Pre-admission urine output.
    Itemid 40060 = Pre-Admission Urine (CareVue); 226633 = Pre-Admission Urine (MetaVision).
    226633 exists in MIMIC-IV, so this query is not MIMIC-III-only.
    datediff_minutes = intime - charttime in minutes (positive = how far before ICU admit).
    Identical logic to AI-Clinician preadm_uo().
    """
    return """
        SELECT DISTINCT
            oe.stay_id AS icustay_id,
            EXTRACT(EPOCH FROM oe.charttime)::bigint AS charttime,
            oe.itemid,
            oe.value,
            EXTRACT(EPOCH FROM (ic.intime - oe.charttime))::bigint / 60 AS datediff_minutes
        FROM mimiciv_icu.outputevents oe
        JOIN mimiciv_icu.icustays ic ON oe.stay_id = ic.stay_id
        WHERE oe.itemid IN (40060, 226633)
        ORDER BY icustay_id, charttime, itemid
    """


def uo():
    """Urine output events."""
    return """
        SELECT
            stay_id AS icustay_id,
            EXTRACT(EPOCH FROM charttime)::bigint AS charttime,
            itemid,
            value
        FROM mimiciv_icu.outputevents
        WHERE itemid IN {codes}
          AND value > 0
          AND value IS NOT NULL
        ORDER BY stay_id, charttime
    """.format(codes=repr(UO_CODES))


def vaso_mv():
    """
    Vasopressor infusions, normalized to noradrenaline-equivalent dose
    in mcg/kg/min (assuming 80 kg body weight when unknown).
    Conversion factors identical to original AI-Clinician pipeline.
    """
    return """
        SELECT
            stay_id AS icustay_id,
            itemid,
            EXTRACT(EPOCH FROM starttime)::bigint AS starttime,
            EXTRACT(EPOCH FROM endtime)::bigint   AS endtime,
            CASE
                -- Noradrenaline (itemids 30120, 221906, 30047)
                WHEN itemid IN (30120,221906,30047) AND rateuom = 'mcg/kg/min' THEN rate
                WHEN itemid IN (30120,221906,30047) AND rateuom = 'mcg/min'    THEN rate / 80.0
                -- Epinephrine (itemids 30119, 221289)
                WHEN itemid IN (30119,221289)       AND rateuom = 'mcg/kg/min' THEN rate
                WHEN itemid IN (30119,221289)       AND rateuom = 'mcg/min'    THEN rate / 80.0
                -- Vasopressin (itemids 30051, 222315) — units/hour -> convert to NE equiv
                WHEN itemid IN (30051,222315)       AND rate > 0.2             THEN rate * 5.0 / 60.0
                WHEN itemid IN (30051,222315)       AND rateuom = 'units/min'  THEN rate * 5.0
                WHEN itemid IN (30051,222315)       AND rateuom = 'units/hour' THEN rate * 5.0 / 60.0
                -- Phenylephrine (itemids 30128, 221749, 30127)
                WHEN itemid IN (30128,221749,30127) AND rateuom = 'mcg/kg/min' THEN rate * 0.45
                WHEN itemid IN (30128,221749,30127) AND rateuom = 'mcg/min'    THEN rate * 0.45 / 80.0
                -- Dopamine (itemids 221662, 30043, 30307)
                WHEN itemid IN (221662,30043,30307) AND rateuom = 'mcg/kg/min' THEN rate * 0.01
                WHEN itemid IN (221662,30043,30307) AND rateuom = 'mcg/min'    THEN rate * 0.01 / 80.0
                ELSE NULL
            END AS ratestd
        FROM mimiciv_icu.inputevents
        WHERE itemid IN {codes}
          AND rate IS NOT NULL
          AND statusdescription != 'Rewritten'
        ORDER BY stay_id, starttime
    """.format(codes=repr(VASO_CODES))
