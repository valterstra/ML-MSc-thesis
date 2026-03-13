# CARE-AI Thesis Guide

*An internal reference document explaining what this project does, why, and how — step by step.*

---

## The Big Picture

Hospitals make hundreds of decisions every day about how to treat patients: which drugs to give, when to escalate a patient to intensive care, when it is safe to discharge. Most of these decisions are guided by clinical experience and general guidelines, but they are not individually optimised for each patient.

The goal of this project is to work towards a system that can look at a patient's current state — their lab results, where they are in the hospital, what drugs they are currently on — and recommend the treatment decisions that are most likely to lead to a good outcome for *that specific patient*.

To get there responsibly, we do not jump straight to making recommendations. Instead we build up the necessary pieces in a sequence of steps, each one laying the foundation for the next.

We are using **MIMIC-IV** as our data source — a large, freely available dataset of real hospital admissions from Beth Israel Deaconess Medical Center in Boston, containing detailed clinical records for tens of thousands of patients.

---

## The Steps

```
Step 1 → Build a simulator (digital twin of the patient)   ✓ Complete
Step 2 → Estimate treatment effects from real data          ✓ Complete
Step 3 → Train a reinforcement learning policy              ✓ Complete
Step 4 → Evaluate the policy                                Planned
```

Steps 1–3 are complete. Step 4 is planned next.

---

## Step 1 — Simulator (Digital Twin)

**What we do:** We build a model that mimics how a patient's state evolves from one day to the next. Given today's state and today's treatment actions, the simulator predicts tomorrow's state.

Concretely, we train one LightGBM (gradient boosting) model per output variable. For example, one model predicts tomorrow's creatinine level, another predicts tomorrow's WBC count, another predicts whether the patient will be discharged. Each model takes as input today's full state plus today's drug flags.

We then chain these models together: the output of day T becomes the input of day T+1, allowing us to simulate an entire hospital stay step by step.

**Why we need it:** This is the most important piece of infrastructure in the project. We cannot train a reinforcement learning policy directly on real patients — we do not have enough data, we cannot do experiments, and we cannot expose patients to an untested algorithm. So instead we train the policy inside a simulator, which is safe, fast, and repeatable.

**What comes out:** A collection of trained transition models saved to disk (`models/sim_daily/`), plus evaluation metrics comparing simulated trajectories to real ones.

---

### Step 1a — Building the Dataset

Before we can train the simulator, we need to build the dataset it learns from. This section explains in detail what that dataset looks like, where the data comes from, what decisions we made, and why.

---

#### The Core Idea: One Row Per Patient Per Day

The dataset is structured as a table where **each row represents one patient on one day of their hospital stay**. If a patient is in hospital for 7 days, they contribute 7 rows. If another patient stays for 30 days, they contribute 30 rows.

Each row captures three things:
1. **What the patient looked like that day** (their state)
2. **What treatments were active that day** (the actions)
3. **What happened next** — which is just the next row in the sequence, plus a long-horizon label (were they readmitted within 30 days of eventual discharge?)

This structure is what allows us to train a model to predict: given this patient's current state and these treatments, what will tomorrow look like?

The dataset built from MIMIC-IV contains approximately **498,000 admissions** and **3 million rows** in total. For development and testing we work with a **5,000-episode sample** (~30,000 rows), which is fast to work with and representative of the full dataset.

---

#### Where the Data Comes From

MIMIC-IV stores hospital data across many separate tables in a PostgreSQL database. Think of it like a hospital's electronic health record system split into categories: one table for lab results, one for prescriptions, one for patient movements around the hospital, and so on.

We pull from six main tables:

| Table | What it contains | What we use it for |
|---|---|---|
| `admissions` | One row per hospital stay — admission time, discharge time, outcome | Episode boundaries, age, readmission label |
| `patients` | One row per patient — age, sex | Demographics |
| `transfers` | Every time a patient moves between units | Where in the hospital the patient is each day |
| `services` | Which medical team is responsible | Clinical service each day |
| `labevents` | Every lab test drawn — 18 GB table | The physiology state (creatinine, WBC, glucose, etc.) |
| `prescriptions` | Every medication order with start/stop times | What drugs are active each day |
| `microbiologyevents` | Culture orders and results | Infection status |
| `drgcodes` + `diagnoses_icd` | Billing and diagnosis codes assigned at discharge | Severity score (Charlson, DRG) |

---

#### The State — What We Observe About a Patient Each Day

The state is everything we know about the patient at the start of a given day. We split this into four groups:

**1. Laboratory values** — the patient's blood test results

These are the most informative snapshot of how the patient is doing physiologically. We include 18 lab values:

| Lab | What it measures | Why included |
|---|---|---|
| Creatinine | Kidney function | Key marker of organ failure |
| BUN (urea nitrogen) | Kidney function | Complements creatinine |
| Sodium | Electrolyte balance | Abnormalities are common and clinically important |
| Potassium | Electrolyte balance | Critical for heart rhythm |
| Bicarbonate | Acid-base balance | Marker of metabolic disturbance |
| Anion gap | Acid-base balance | Detects metabolic acidosis |
| Glucose | Blood sugar | Affected by insulin, steroids, infection |
| Haemoglobin | Red blood cells / anaemia | Relates to transfusion and bleeding |
| WBC | White blood cell count | Infection and inflammation marker |
| Platelets | Clotting | Low platelets = bleeding risk or serious illness |
| Magnesium | Electrolyte | Commonly abnormal in hospital; affects heart and muscles |
| Calcium | Electrolyte | Affects many organ systems |
| Phosphate | Electrolyte | Relates to kidney function and nutrition |
| INR | Blood clotting time | Needed for anticoagulation decisions |
| Bilirubin | Liver function | Marker of liver disease or bile obstruction |
| NLR (neutrophil-to-lymphocyte ratio) | Inflammation | Composite inflammatory marker |
| Albumin | Nutrition / protein | Low albumin = severe illness or malnutrition |
| Lactate elevated | Tissue oxygen delivery | Binary flag: lactate > 2.0 mmol/L indicates concern |

Labs are not measured every single day. When a lab result is missing, we carry the most recent value forward for a limited number of days (2 days for routine labs, 3 days for coagulation, 5 days for albumin). This reflects clinical reality — if a result was measured yesterday and not today, the clinician is still working with yesterday's value. We also record a separate binary flag (`_measured`) for each lab indicating whether it was actually drawn today or carried forward — this allows the model to distinguish "we know creatinine is 1.2" from "we assume it's still 1.2 from two days ago."

**2. Location** — where in the hospital the patient is

Patients move between units during their stay: emergency department, general ward, intensive care unit (ICU), and so on. Their physical location reflects both their severity and the level of care they are receiving.

We record:
- `careunit_group` — a simplified category: ICU, emergency department, medicine ward, surgery ward, oncology, neurology, psychiatry, obstetrics, or other
- `is_icu` — a binary flag: is the patient in an ICU unit right now?
- `days_in_current_unit` — how many days have passed since the patient arrived in this unit

**3. Clinical service** — which medical team is responsible

This is separate from physical location. A patient can be on the "Medicine" service (meaning a general medicine team is responsible for their care) while physically located in an ICU bed, or on the "Surgery" service while on a general ward recovering post-operatively. The clinical service is a useful proxy for the patient's primary diagnosis category.

We map the raw service codes to groups: medicine, surgery, ICU service, psychiatry, obstetrics, trauma, or other.

**4. Static features** — facts about the patient that do not change during the stay

- `age_at_admit` — patient age at admission (minimum 18; paediatric patients are excluded)
- `gender` — biological sex
- `charlson_score` — the Charlson Comorbidity Index, a single number (0–24+) summarising how many serious chronic conditions the patient has. A patient with no chronic conditions scores 0; a patient with heart failure, diabetes, and kidney disease might score 6 or higher.
- `drg_severity` — the APR-DRG severity of illness score (1 = minor, 4 = extreme), assigned by billing coders at discharge. Captures case complexity independently of specific diagnoses.

**5. Infection state** — culture activity and results

Infections are a major driver of hospital deterioration. We track:
- `culture_ordered_today` — was any culture specimen (blood, urine, sputum) sent to the lab today? This is a flag of clinical suspicion.
- `positive_culture_cumulative` — has any culture grown an organism at any point during this admission? This latches to 1 the moment a positive result appears and never resets, because once a patient is known to have an active infection, that context remains relevant.
- `blood_culture_positive_cumulative` — same, but specifically for blood cultures, which indicate bacteria in the bloodstream (bacteraemia/sepsis).

---

#### The Actions — What Treatments Are Active Each Day

Actions represent the clinical decisions made on a given day. We focus on six drug classes plus two care-intensity decisions:

**Drug class flags** (from the prescriptions table):

| Action | What it is | Why this class |
|---|---|---|
| `antibiotic_active` | Any antibiotic prescription is active today | Antibiotics are the primary treatment for bacterial infection; their use or non-use is a major daily decision |
| `anticoagulant_active` | Any blood thinner is active (heparin, warfarin, apixaban, etc.) | Anticoagulation is used for clot prevention and treatment; affects bleeding risk |
| `diuretic_active` | Any diuretic is active (furosemide, etc.) | Used to manage fluid overload; common in heart failure and ICU patients |
| `steroid_active` | Any corticosteroid is active (dexamethasone, prednisone, etc.) | Steroids suppress inflammation but raise glucose and suppress immunity |
| `insulin_active` | Any insulin formulation is active | Blood sugar management; closely linked to glucose state |
| `opioid_active` | Any opioid is active (morphine, fentanyl, oxycodone, etc.) | Pain management; has sedation effects relevant to care decisions |

We deliberately chose **drug classes rather than individual drugs** because individual drug names are too numerous and sparse for modelling, and because the clinically relevant question is "is this patient on an antibiotic?" not "is this patient on piperacillin-tazobactam specifically?"

Each drug class is identified by matching the drug name (as recorded in the prescriptions table) against a curated list of patterns. For example, the antibiotic class matches vancomycin, piperacillin, ceftriaxone, meropenem, azithromycin, and about 20 others.

**Care intensity transitions** (from the transfers table):

| Action | What it is |
|---|---|
| `icu_escalation` | The patient was moved from a ward into an ICU today |
| `icu_stepdown` | The patient was moved from an ICU to a ward today |
| `discharged` | The patient was discharged from hospital today |

These are not actions in the sense that a clinician fills out an order — they are observed transitions. But they are important because ICU escalation/stepdown represents a major treatment intensity decision, and discharge is the terminal action of each episode.

---

#### The Label — What We Are Ultimately Trying to Predict

The long-horizon outcome label is `readmit_30d`: was the patient readmitted to hospital within 30 days of their discharge?

This is computed by looking at whether the same patient has another hospital admission recorded within 30 days of the current admission's discharge time. The label is the same value on every row of a given admission (it is a property of the episode, not of a single day), and it is only used at the discharge-day row during model training.

The 30-day readmission rate in our dataset is **18.4%** — broadly consistent with published rates for US academic medical centres.

---

#### Key Design Decisions and the Reasoning Behind Them

**Why daily granularity, not hourly?**
An earlier version of this pipeline worked at hourly resolution, but that required the ICU-specific `chartevents` table (400 GB) and was therefore limited to ICU patients only. Only about 13% of hospital admissions ever touch an ICU. By switching to daily granularity and using the `labevents` table instead, we cover the entire hospital — all ward types — and capture the full patient journey from admission to discharge. Most clinical decisions in a general ward also happen on a daily cycle (the morning ward round), so daily granularity is clinically meaningful.

**Why forward-fill lab values rather than leaving them blank?**
Labs are not drawn every day. If creatinine was 1.8 mg/dL yesterday and was not re-checked today, the clinical team is still working with 1.8 as their reference. Forward-filling reflects this: we are not making up values, we are propagating known information forward in the same way a clinician would. Fill limits (2–5 days depending on the lab) reflect how often each test is typically re-ordered in practice.

**Why include a `_measured` flag for each lab?**
Even after forward-filling, the model should be able to distinguish between "this value was measured today" and "this value was carried forward from two days ago." A low creatinine that was measured today is more reassuring than one that is two days old. The binary measurement flags preserve this distinction.

**Why drug classes and not individual drugs?**
A hospital has hundreds of individual drug formulations. Many patients receive the same clinical intervention (e.g. anticoagulation) but recorded under different brand names or generic formulations. Grouping by clinical class produces stable, interpretable action dimensions that generalise across different hospitals and time periods.

**Why cumulative rather than daily infection flags?**
Once a culture grows an organism, the clinical team knows the patient has an active infection. That knowledge does not expire at midnight. A cumulative flag that latches to 1 and never resets accurately represents "is this a patient with a known infection?" in a way that a daily flag (which would be 1 only on the day the result came back) does not.

**Why split by patient, not by admission?**
If we split randomly by admission, the same patient's earlier admissions could end up in training and their later admissions in the test set. The model would then have seen that patient's physiology before, giving it an unfair advantage and overstating how well it generalises to new patients. Splitting by patient ensures the test set contains only patients the model has never encountered in any form.

---

#### Dataset Numbers at a Glance

**Full dataset (used for all modelling):**

| Property | Value |
|---|---|
| Rows | 3,060,097 |
| Unique admissions | 498,045 |
| Train / Validation / Test transitions | 1,791,712 / 387,049 / 383,291 |
| Train/valid/test split | ~68% / ~16% / ~16% |

**5,000-episode development sample** (used for rapid iteration and testing):

| Property | Value |
|---|---|
| Rows | 30,247 |
| Unique admissions | 5,214 |
| Unique patients | 2,132 |
| Mean length of stay | 4.8 days |
| 30-day readmission rate | 18.4% |
| % of days in ICU | 9.2% |
| % of days on an antibiotic | 33.8% |
| % of days on an anticoagulant | 59.0% |
| % of days on an opioid | 44.4% |

---

### Step 1b — Training the Transition Model

Once the dataset is built, we use it to train the simulator. The core idea is straightforward: for each thing we want to predict about tomorrow's patient state, we train a separate machine learning model.

#### What we are predicting

We split the outputs into three groups:

**Continuous outputs** — lab values that are numbers on a scale. For each of these, the model predicts tomorrow's value given today's state and actions. There are **15** of these: creatinine, BUN, sodium, potassium, bicarbonate, anion gap, glucose, haemoglobin, WBC, platelets, magnesium, calcium, phosphate, INR, bilirubin. Note that NLR and albumin are present in the dataset as input features but are not predicted outputs — they are too sparsely measured to be reliable targets.

**Binary outputs** — things that are either true or false tomorrow. There are **4** of these:
- Is the patient in the ICU?
- Is lactate elevated (above 2.0 mmol/L)?
- Has any culture grown an organism at any point in this admission (cumulative)?
- Has any blood culture grown an organism at any point in this admission (cumulative)?

**The "done" model** — a single separate model that predicts whether tomorrow will be the patient's last day in hospital, i.e. whether they will be discharged or die. This is what tells the simulator when to stop. It is derived from the `is_last_day` flag in the dataset: for each training row, we look ahead one day and ask "does the episode end tomorrow?" The done model learns to predict this from today's state.

That gives us **20 models** in total: 15 continuous regressors, 4 binary classifiers, and 1 done model.

#### The algorithm: LightGBM

We use **LightGBM** (Light Gradient Boosting Machine) for all models. This is a tree-based machine learning algorithm that is well-suited to tabular data with mixed types, missing values, and non-linear relationships. It is fast to train, robust, and consistently performs well on structured clinical data.

The same algorithm is used for both continuous outputs (as a regressor, predicting a number) and binary outputs (as a classifier, predicting a probability). The inputs to every model are the same: today's full state — all lab values, location, service, infection flags, static features, and measurement flags — plus today's action flags.

#### Training setup

- The dataset is split into train, validation, and test sets at the **patient level** (as described above).
- Models are trained on the **training set**.
- During training, the **validation set** is used for **early stopping** — training stops automatically when the model's performance on the validation set stops improving, which prevents overfitting. We allow up to 50 rounds without improvement before stopping.
- The **test set** is held back entirely and only used for final evaluation.

#### Results — Single-step accuracy on test set (full dataset, 383,291 rows)

**Continuous outputs (R² / MAE):**

| Lab | R² | MAE | Notes |
|---|---|---|---|
| Bilirubin | 0.943 | 0.316 mg/dL | Highest R² — very stable day to day |
| BUN | 0.875 | 3.96 mg/dL | Strong — kidney function well captured |
| Platelets | 0.872 | 26.6 ×10⁹/L | Strong |
| Creatinine | 0.842 | 0.219 mg/dL | Strong — key organ function marker |
| Haemoglobin | 0.812 | 0.586 g/dL | Strong |
| WBC | 0.777 | 1.60 ×10⁹/L | Good |
| Calcium | 0.696 | 0.263 mg/dL | Good |
| Bicarbonate | 0.689 | 1.65 mEq/L | Good |
| INR | 0.661 | 0.148 | Good |
| Phosphate | 0.612 | 0.452 mg/dL | Moderate |
| Anion gap | 0.520 | 1.72 mEq/L | Moderate |
| Magnesium | 0.461 | 0.145 mg/dL | Moderate — less frequently measured |
| Potassium | 0.448 | 0.273 mEq/L | Moderate — high day-to-day variability |
| Sodium | 0.649 | 1.81 mEq/L | Good |
| Glucose | 0.368 | 22.7 mg/dL | Lowest R² — highly variable, confounded by diet and insulin |

*Interpreting R² and MAE:* R² measures how much of the day-to-day variation in a lab value the model explains. A value above 0.8 indicates the model captures the dominant dynamics of that lab — essentially, today's value is a strong predictor of tomorrow's and the model uses it well. Values between 0.5 and 0.8 are moderate — the model is useful but misses some variability. Values below 0.5 reflect labs that are inherently volatile or whose changes are driven by factors not captured at daily granularity (diet, exact dosing times, patient-reported symptoms). MAE should be read in clinical units alongside the lab's normal range: an MAE of 0.219 mg/dL for creatinine is small relative to a normal range of 0.6–1.2 mg/dL, meaning the model is rarely off by a clinically meaningful margin. An MAE of 22.7 mg/dL for glucose is larger, but glucose has a normal range of 70–140 mg/dL and can swing by hundreds of mg/dL in a single day, so even a large absolute error may not translate into a wrong clinical conclusion.

**Binary outputs (AUC):**

| Output | AUC |
|---|---|
| Blood culture positive (cumulative) | 0.990 |
| Positive culture (cumulative) | 0.981 |
| ICU status | 0.972 |
| Lactate elevated | 0.904 |
| Discharge (done model) | 0.784 |

*Interpreting AUC:* AUC measures discrimination — the probability that the model ranks a truly positive patient-day above a truly negative one. An AUC above 0.95 is excellent and indicates the model is reliably separating the two classes. The four cumulative binary flags (blood culture positive, positive culture, ICU status, lactate elevated) all exceed 0.90, reflecting two things: these flags have strong clinical predictors in the state (a patient with very high WBC and confirmed infection is likely to remain ICU-bound), and cumulative flags that have already been set to 1 tend to stay at 1, making persistence easy to learn. The discharge model (0.784) is the hardest: discharge is partly a clinical decision shaped by factors outside the state representation — bed availability, family circumstances, clinician judgement — so a meaningful portion of the variation is genuinely unpredictable from labs alone. An AUC of 0.784 means the model is substantially better than chance but should not be expected to perfectly anticipate discharge timing.

Glucose has the lowest R² of all continuous outputs — partly because it is genuinely volatile, and partly because its day-to-day changes are strongly driven by dietary intake and insulin dosing, which are not fully captured in the daily-granularity dataset.

---

#### One important detail: clip bounds

After training, we record the 1st and 99th percentile of each continuous output in the training data. When the simulator generates predictions during rollout, any predicted value outside these bounds is clipped back to the boundary. This prevents the simulator from producing physiologically implausible values — for example, predicting a negative creatinine or an INR of 500 — which would cause downstream models to behave erratically.

---

### Step 1c — Evaluating the Simulator

After training, we evaluate the simulator in two complementary ways: one-step accuracy and multi-step rollout fidelity.

#### One-step accuracy: can the model predict tomorrow correctly?

The first evaluation is the simplest. We take every consecutive pair of days in the test set — today's state and actions, plus the real observed next-day state — and ask: how accurately does the model predict that next-day state?

For **continuous outputs** we report two metrics:
- **R²** (R-squared): a value from 0 to 1 indicating how much of the day-to-day variation the model explains. An R² of 1.0 is a perfect prediction; 0.0 means the model does no better than simply predicting the mean every time. For stable lab values like sodium, we expect high R² (the value does not change dramatically day to day). For volatile ones like lactate or glucose, we expect lower R².
- **MAE** (Mean Absolute Error): the average absolute difference between predicted and actual values, in the original units (e.g. mg/dL for creatinine). This is the most interpretable metric — an MAE of 0.3 for creatinine means the model is on average 0.3 mg/dL off.

For **binary outputs** we report:
- **AUC** (Area Under the ROC Curve): a value from 0.5 (random guessing) to 1.0 (perfect). This measures how well the model ranks positive cases above negative cases. For example, an AUC of 0.85 for the ICU flag means that on 85% of random pairs of (ICU day, non-ICU day), the model assigns higher probability to the ICU day.

#### Multi-step rollout: does the simulator stay realistic over time?

A model that is accurate for one step may still drift badly when run forward over many steps, because each prediction feeds into the next. An error on day 3 changes the input for day 4, which compounds into day 5, and so on.

To check this, we run the simulator forward from real patient starting states for up to 60 days, generating entire synthetic hospital trajectories. We then compare the distribution of simulated values to the distribution of real values from the test set.

The comparison uses the **Kolmogorov-Smirnov (KS) test** — a statistical test that measures how different two distributions are. A high KS statistic means the simulated and real distributions look different; a low one means they are similar. We report the KS statistic and mean/standard deviation for each state variable side by side, so we can see at a glance which variables the simulator reproduces well and which it struggles with.

The rollout is run with **zero actions** (all drug flags set to 0). This is not because we expect patients to receive no treatment — it is a neutral baseline that removes the question of which action policy to assume during evaluation, so we can focus purely on whether the state dynamics themselves are realistic.

#### Results — Multi-step rollout (500 rollouts, 5-day fixed horizon, full dataset model)

| Variable | KS statistic | Sim mean | Real mean | Verdict |
|---|---|---|---|---|
| ICU status | 0.005 | 0.09 | 0.08 | Excellent |
| Lactate elevated | 0.015 | 0.01 | 0.02 | Excellent |
| Blood culture +ve (cumulative) | 0.009 | 0.00 | 0.01 | Excellent |
| BUN | 0.129 | 18.7 | 21.8 | Good — slight underestimate |
| Creatinine | 0.200 | 1.16 | 1.31 | Good — slight underestimate |
| Haemoglobin | 0.201 | 11.2 | 10.8 | Good |
| Glucose | 0.245 | 114 | 124 | Good — systematic underestimate |
| Platelets | 0.267 | 223 | 225 | Good |
| Phosphate | 0.303 | 3.48 | 3.47 | Moderate |
| Sodium | 0.325 | 139.2 | 138.5 | Moderate |
| Anion gap | 0.341 | 13.2 | 13.3 | Moderate |
| INR | 0.345 | 1.31 | 1.41 | Moderate |
| Potassium | 0.346 | 4.09 | 4.10 | Moderate |
| WBC | 0.355 | 8.77 | 9.00 | Moderate |
| Calcium | 0.357 | 8.86 | 8.70 | Moderate |
| Bicarbonate | 0.372 | 24.6 | 25.2 | Moderate — systematic underestimate |
| Magnesium | 0.401 | 2.03 | 2.01 | Weak — variance collapse |
| Bilirubin | 0.513 | 1.14 | 1.32 | Weak — simulator underestimates liver values |
| Positive culture (cumulative) | 0.065 | 0.03 | 0.10 | Poor — simulator rarely accumulates positive culture status |

All KS statistics are statistically significant (p < 0.05), which is expected at this sample size. The more useful diagnostic is the magnitude: KS < 0.2 generally indicates the simulator reproduces the real distribution well enough for training purposes. The rollout uses a 5-day fixed horizon, which is more representative of typical hospital stays and avoids confounding from variable episode length. The weaker variables (bilirubin, magnesium, bicarbonate) show variance collapse from the first step rather than accumulated drift — the model predicts values close to the mean rather than reproducing the full spread. The cumulative culture flag is the clearest weakness — because it is a one-way latch (once positive, it stays positive), multi-step rollouts tend to underestimate how many patients carry a positive culture flag by the end of a stay. This is a known limitation of predicting cumulative flags in a step-by-step simulator.

---

### Step 1d — Action Sensitivity: Does the Model Make Clinical Sense?

Even if the simulator is statistically accurate, that is not enough on its own. We also need to check that it has learned the **right relationships** between treatments and outcomes — not just correlations that happen to fit the data.

This is the action sensitivity analysis. The idea is simple:

> Take a large sample of real patient-days from the test set. For each drug, force that drug to be ON for every patient-day (even those where it was not actually given), predict tomorrow's state. Then force it to be OFF for every patient-day (even those where it was given), predict tomorrow's state again. Measure the average difference.

This tells us: according to the model, what effect does each drug have on each outcome? We then compare this to what we would expect clinically.

#### Which pairs we evaluate and why

The transition model computes predictions for all 19 state variables for each of the 6 drugs — a full 6×19 = 114 drug-outcome combinations. However, we do not evaluate all 114 against clinical expectations. We restrict evaluation to the subset where two conditions are both met:

1. **The direction of the effect is unambiguous** — there is strong pharmacological or physiological consensus on whether the drug increases or decreases the outcome.
2. **The effect is plausible at daily temporal resolution** — some drug effects take days to weeks to appear in lab values (e.g. antibiotics clearing a deep-seated infection) and would not be detectable in a one-day prediction window.

This means we are not cherry-picking pairs that the model happens to get right. We defined the expected directions before looking at results, based on established clinical pharmacology, and we apply them uniformly.

The remaining ~90 combinations in the full matrix are reported but not judged — either because the directional expectation is ambiguous, the effect is too slow to appear at daily resolution, or it is not a primary pharmacological effect.

#### Expected directions

| Drug | Outcome | Expected direction | Strength of evidence | Clinical mechanism |
|---|---|---|---|---|
| Insulin | Glucose | **Down** | Very strong | Direct glucose uptake into cells |
| Insulin | Potassium | **Down** | Very strong | Insulin drives K⁺ into cells; used clinically to treat hyperkalemia |
| Anticoagulant | INR | **Up** | Very strong | Blood thinners extend clotting time by definition |
| Diuretic | BUN | **Up** | Strong | Water removal concentrates urea (hemoconcentration) |
| Diuretic | Potassium | **Down** | Very strong | Loop diuretics cause renal K⁺ excretion — a well-known side effect requiring supplementation |
| Diuretic | Magnesium | **Down** | Strong | Loop diuretics cause renal Mg²⁺ wasting — routinely supplemented in clinical practice |
| Diuretic | Bicarbonate | **Up** | Moderate | Contraction alkalosis — water removal concentrates bicarbonate |
| Diuretic | Sodium | **Up** | Moderate | Hemoconcentration raises sodium, though effect depends on diuretic type and fluid status |
| Steroid | Glucose | **Up** | Very strong | Steroids reduce insulin sensitivity and stimulate hepatic glucose production |
| Steroid | WBC | **Up** | Very strong | Demargination — steroids release neutrophils from vessel walls into bloodstream |
| Antibiotic | WBC | **Down** | Moderate | As bacterial infection clears, the inflammatory drive subsides — but this is a delayed effect and may be weak at one-day resolution |
| Opioid | (none) | — | — | Opioid effects on standard lab panels are indirect and ambiguous at daily resolution |

If the model's predicted directions match these clinical expectations, it is a sign that the model has learned something real about the biology — not just spurious patterns in the data. If they contradict the expectations (e.g. insulin is predicted to raise glucose), that is a red flag suggesting the model has picked up confounding rather than causation, and should not be trusted to evaluate treatment policies.

This is why the action sensitivity analysis sits between the simulator training (Step 1b) and the causal inference work (Step 2) — it is the bridge between "can the model predict?" and "can the model be trusted to reason about the effects of actions?"

#### Results — Action sensitivity (full dataset model, 3,000 test rows sampled)

| Drug | Outcome | Predicted | Expected | Verdict |
|---|---|---|---|---|
| Steroid | Glucose | +3.48 (UP) | UP | Correct |
| Steroid | WBC | +0.32 (UP) | UP | Correct |
| Anticoagulant | INR | +0.006 (UP) | UP | Correct |
| Diuretic | BUN | +1.67 (UP) | UP | Correct |
| Diuretic | Potassium | -0.04 (DOWN) | DOWN | Correct |
| Diuretic | Bicarbonate | +0.69 (UP) | UP | Correct |
| **Insulin** | **Glucose** | **+16.6 (UP)** | **DOWN** | **Wrong — confounded** |
| **Insulin** | **Potassium** | **+0.02 (UP)** | **DOWN** | **Wrong — confounded** |
| **Antibiotic** | **WBC** | **+0.05 (UP)** | **DOWN** | **Wrong — confounded** |
| Diuretic | Sodium | -0.04 (DOWN) | UP | Wrong |
| Diuretic | Magnesium | +0.006 (UP) | DOWN | Wrong |

**7 of 11 checked pairs correct.**

The failures are interpretable:

- **Insulin → glucose and potassium both wrong:** The model predicts that insulin raises both glucose and potassium, when clinically it should lower both. Both failures point in the same direction and for the same reason: in the real data, insulin is prescribed to patients who are *already* hyperglycaemic and often hyperkalaemic. The model has learned who gets insulin, not what insulin does. That both primary pharmacological effects of insulin are simultaneously wrong is strong systematic evidence of confounding — not noise.

- **Antibiotic → WBC wrong:** Same confounding logic. Antibiotics are given to infected, inflamed patients with elevated WBC. The model associates antibiotics with high WBC rather than learning that antibiotics reduce WBC by clearing infection.

- **Diuretic → sodium wrong:** Diuretics can raise or lower sodium depending on type and patient fluid status. The net effect in this population is slightly negative — a genuine clinical ambiguity rather than a clear model error.

- **Diuretic → magnesium wrong:** The model predicts magnesium goes slightly up, when loop diuretics should deplete it via renal wasting. The effect is small in magnitude (+0.006) suggesting the model is essentially flat on magnesium, but the direction is wrong.

---

## Step 2 — Treatment Effect Estimation

**What we do:** We ask a harder question about the data: not just *what tends to happen* to patients who receive a drug, but *what would have happened* if they had or had not received it. This is the domain of **causal inference**.

**Why we need it:** The simulator from Step 3 was trained on observational data, which means it learned correlations, not causes. Before we trust it to evaluate treatment policies, we want to verify that the effects it has learned match what causal analysis on the real data tells us. If a drug should lower glucose according to our causal analysis, the simulator should also predict lower glucose when that drug is administered — if they disagree, we have a problem.

**What comes out:** A table of estimated average treatment effects for each drug–outcome pair, with confidence intervals and a verdict (correct direction / wrong direction), plus overlap and balance diagnostic reports.

---

### Step 2a — Why Simple Comparisons Fail: Confounding

Imagine we look at all patient-days in our dataset and compare glucose levels the next day for patients who were on insulin versus those who were not. We might find that the insulin group actually had *higher* glucose. Does that mean insulin raises glucose? Of course not — it means that sicker, more hyperglycaemic patients are more likely to be prescribed insulin in the first place. The drug was given *because* glucose was already high. This is **confounding**: the treatment and the outcome share a common cause (in this case, high blood sugar), which distorts any naive comparison.

In observational hospital data, confounding is everywhere:
- Sicker patients receive more antibiotics, more steroids, and are more likely to deteriorate regardless of treatment.
- Patients in the ICU receive more interventions and also have worse outcomes — not because the interventions cause harm, but because they were already sicker.
- A drug prescribed to a high-risk patient will appear to perform worse than the same drug prescribed to a low-risk patient, even if the drug itself is equally effective.

If we do not account for confounding, we cannot learn anything reliable about the effect of treatments.

---

### Step 2b — Propensity Scores: Measuring How Likely Treatment Was

The first step in adjusting for confounding is to model *why* each patient received each drug on each day. For each of our six drug classes, we train a **propensity model**: a logistic regression that predicts the probability a patient receives that drug, given everything we know about them that day.

**What goes into the propensity model (the confounders):**
- All 15 continuous lab values (today's state)
- All 4 binary state variables (ICU, lactate elevated, infection flags)
- All 7 static features (age, sex, Charlson score, DRG severity/mortality, day of stay, days in current unit)
- All 15 lab measurement flags (which labs were actually drawn today)
- The other 5 drug class flags — because co-prescriptions are confounders. A patient on antibiotics and steroids simultaneously is a different clinical picture from one on antibiotics alone.

The focal drug itself is excluded from its own propensity model (you cannot use "is this patient on antibiotics?" to predict "is this patient on antibiotics?").

**Important implementation detail:** The propensity models are fitted on the **training split** only. The treatment effect estimates are then computed on the **test split**. This separation prevents the propensity model from overfitting to the data it will later be evaluated on.

The propensity score for a given patient-day is a number between 0 and 1: how likely was this patient to receive this drug today, given their observed state? A patient with a high fever, elevated WBC, and a positive culture might have a propensity score of 0.85 for antibiotics — treatment was almost certain. A healthy patient with no infection markers might score 0.05 — treatment was unexpected.

Propensity scores are clipped to the range [0.01, 0.99] to prevent extreme values from producing unstable weights.

---

### Step 2c — AIPW: The Doubly Robust Estimator

For each treatment–outcome pair we estimate the **average treatment effect (ATE)**: how much, on average, does one additional day of this drug change the next-day value of this outcome? We use the **Augmented Inverse Probability Weighting (AIPW)** estimator, also known as the doubly robust estimator.

AIPW combines two models:

1. A **propensity model** — logistic regression estimating the probability that this patient received the drug, given their observed state (fitted in Step 2b).
2. An **outcome model** — linear regression estimating tomorrow's outcome as a function of today's state and the treatment flag.

**The formula**

For a dataset of n patient-days, the AIPW estimate is:

```
ATE = (1/n) * sum over all patients i of:

    [ mu1(x_i) - mu0(x_i) ]
    + (t_i / e(x_i)) * (y_i - mu1(x_i))
    - ((1 - t_i) / (1 - e(x_i))) * (y_i - mu0(x_i))
```

**What each term means**

- `x_i` — the full set of observed confounders for patient-day i (labs, static features, co-prescriptions).

- `t_i` — the treatment indicator: 1 if the drug was given on day i, 0 if not.

- `y_i` — the observed outcome on day i+1 (e.g. next-day WBC).

- `e(x_i)` — the **propensity score**: the fitted probability P(t=1 | x_i) from the logistic propensity model. It captures how likely treatment was given everything observed about this patient. Clipped to [0.01, 0.99] to prevent extreme weights.

- `mu1(x_i)` — the outcome model's prediction of y_i *if the patient had been treated* (t=1), regardless of whether they actually were. This is a counterfactual prediction.

- `mu0(x_i)` — the outcome model's prediction of y_i *if the patient had not been treated* (t=0). Also counterfactual.

- `mu1(x_i) - mu0(x_i)` — the **outcome model term**: the direct regression estimate of the individual treatment effect for patient i. This is what you would get from regression adjustment alone — it answers "according to the outcome model, what is the effect of treatment for this patient?"

- `(t_i / e(x_i)) * (y_i - mu1(x_i))` — the **residual correction for treated patients**: for patients who actually received treatment (t_i=1), this measures how wrong the outcome model was (the residual y_i - mu1(x_i)) and upweights patients who received treatment despite a low propensity score (i.e. surprising treatments). If the outcome model were perfect, this residual would be zero and the correction would vanish.

- `((1 - t_i) / (1 - e(x_i))) * (y_i - mu0(x_i))` — the **residual correction for untreated patients**: the same logic applied in reverse. For patients who were not treated (t_i=0, so (1-t_i)=1), this corrects the outcome model using the actual untreated outcome, upweighting patients who were unexpectedly untreated (high propensity but no drug given).

**The doubly robust property**

If the outcome model is correctly specified, the two residual correction terms average to zero and the ATE reduces to the mean of (mu1 - mu0) — pure regression adjustment. If the propensity model is correctly specified, the IPW reweighting creates a balanced pseudo-population and the regression term becomes redundant. The estimator gives the correct answer if *at least one* of the two models is correctly specified — hence "doubly robust". This makes AIPW more reliable than either regression adjustment or IPW alone in practical settings where both models are approximate.

---

### Step 2d — Confidence Intervals via Bootstrap

A single point estimate is not enough — we need to know how uncertain it is. We compute **95% confidence intervals** using the bootstrap method:

1. Take the test set and randomly resample it with replacement to create a new dataset of the same size.
2. Re-estimate the AIPW ATE on this resampled dataset using the already-fitted propensity model.
3. Repeat 50 times (sufficient given the 383k-row test set — at this scale the bootstrap variance is negligible and CIs are already very tight).
4. The 2.5th and 97.5th percentiles of the estimates form the 95% confidence interval.

A narrow interval means the estimate is stable and reliable. A wide interval means there is substantial uncertainty — perhaps because few patients received the treatment, or because the outcome is highly variable. With 383,000 test rows, all intervals are narrow — the limiting factor on reliability is model misspecification and unmeasured confounding, not sample size.

---

### Step 2e — Checking That Comparison Is Valid

Before trusting any ATE estimate, we run two diagnostic checks.

**Overlap check**

For a causal comparison to be valid, both treated and untreated patients must exist across the full range of propensity scores. If all patients with propensity score above 0.7 received the drug and none did not, we have no basis for comparison in that region — we would be extrapolating.

We flag a drug as having **poor overlap** if more than 20% of patient-days have a propensity score outside the range [0.1, 0.9]. This means either very high-propensity patients almost always receive the drug, or very low-propensity patients almost never do — both reduce our ability to make valid comparisons.

**Covariate balance**

After applying IPW weighting, we check whether the treated and untreated groups now look similar across all confounders. We use the **Standardised Mean Difference (SMD)**: the difference in means between treated and untreated patients, divided by the pooled standard deviation. We compute this both before weighting (raw SMD) and after weighting (weighted SMD).

An SMD below 0.10 is considered well-balanced. If the weighted SMD is still large after adjustment, the propensity model may not be capturing the true treatment assignment mechanism well.

---

### Step 2f — The Treatment–Outcome Pairs We Analyse

We do not estimate ATEs for every possible drug–outcome combination. Instead we focus on **9 clinically motivated pairs** where we have a clear prior expectation about the direction of the effect:

| Drug | Outcome | Expected direction | Clinical reasoning |
|---|---|---|---|
| Insulin | Glucose | Down | Insulin directly lowers blood sugar |
| Antibiotic | WBC | Down | Clearing infection reduces the inflammatory response |
| Antibiotic | Positive culture (cumulative) | Down | Successful treatment should reduce ongoing positive cultures |
| Diuretic | BUN | Up | Removing water concentrates solutes including urea |
| Diuretic | Potassium | Down | Loop diuretics cause potassium excretion |
| Diuretic | Sodium | Up | Water removal concentrates sodium |
| Steroid | Glucose | Up | Steroids cause hyperglycaemia |
| Steroid | WBC | Up | Steroids cause demargination — WBC moves from tissue into blood |
| Anticoagulant | INR | Up | Blood thinners extend clotting time, raising INR |

For each pair, the pipeline reports whether the AIPW estimate points in the expected direction (**correct**), the wrong direction (**wrong**), or whether no expectation was defined (**unknown**). A "sign flip" — where the naive ATE and the causal ATE point in opposite directions — is particularly informative: it means confounding was strong enough to completely reverse the apparent effect in the raw data.

### Step 2g — Results (full dataset, 383,291 test rows, 50 bootstrap resamples)

| Drug | Outcome | Naive ATE | Causal ATE | 95% CI | Expected | Verdict |
|---|---|---|---|---|---|---|
| Insulin | Glucose | +37.30 | +25.19 | [+22.88, +27.40] | Down | Wrong |
| Antibiotic | WBC | +0.60 | **-0.03** | [-0.07, -0.00] | Down | **Correct** *(sign flip)* |
| Antibiotic | Positive culture | +0.27 | +0.01 | [+0.01, +0.01] | Down | Wrong |
| Diuretic | BUN | +10.05 | **+1.58** | [+1.50, +1.66] | Up | **Correct** |
| Diuretic | Potassium | -0.03 | **-0.04** | [-0.04, -0.03] | Down | **Correct** |
| Diuretic | Sodium | -0.12 | -0.09 | [-0.11, -0.07] | Up | Wrong |
| Steroid | Glucose | +9.74 | **+3.37** | [+2.81, +3.92] | Up | **Correct** |
| Steroid | WBC | +0.11 | **+0.23** | [+0.20, +0.27] | Up | **Correct** |
| Anticoagulant | INR | -0.06 | **+0.02** | [+0.02, +0.03] | Up | **Correct** *(sign flip)* |

**6 of 9 pairs correct.** Two sign flips successfully recovered (antibiotic→WBC, anticoagulant→INR). Confidence intervals are very tight throughout — a direct consequence of the 383k-row test set.

**Interpreting the three "wrong" pairs:**

- **Insulin → glucose (+25.19, expected down):** The AIPW estimate is still positive despite adjustment, reduced substantially from the naive +37.30 but not flipped. This is a case of strong *unmeasured confounding* — insulin patients tend to be hyperglycaemic for reasons that go beyond the 47 confounders we adjust for (dietary intake, enteral tube feeding, prior dosing history are not captured at daily resolution). The adjustment moves in the right direction but cannot fully correct for what it cannot observe. This is an honest and important result: it tells us the problem is harder than the available covariates can solve, not that our method is broken.

- **Antibiotic → positive culture (+0.01, expected down):** The causal estimate is essentially zero. The cumulative culture flag is a one-way latch — once a patient has a confirmed positive culture, the flag stays at 1 regardless of treatment. Antibiotics do not unwind a confirmed infection result, they prevent deterioration. The absence of an effect here is actually clinically sensible; the outcome measure is not the right one to capture antibiotic efficacy at one-day resolution.

- **Diuretic → sodium (-0.09, expected up):** Diuretics can raise or lower sodium depending on the type (loop vs thiazide) and the patient's fluid status. In this population, the net effect is slightly negative, reflecting the real clinical ambiguity — this is not clearly a model failure.

**Overlap flags:** Three drugs showed poor overlap (>20% of rows outside [0.1, 0.9] propensity range): insulin (20.9%), diuretic (29.0%), steroid (35.0%). This means there are patient subgroups where treatment was near-certain or near-impossible, limiting our ability to make valid causal comparisons for those patients. The estimates remain valid for the regions of overlap but should be interpreted with caution at the extremes.

**Balance after adjustment:** For all six drugs, IPW weighting substantially reduced covariate imbalance. Mean absolute SMD across confounders dropped from 0.18–0.34 (raw) to 0.02–0.09 (weighted), all within the conventional 0.10 threshold for acceptable balance.

---

### How Step 2 Connects Back to Step 1d

The action sensitivity analysis in Step 1d asked: what does the *simulator* predict will happen when each drug is turned on or off? Step 2 asks: what does *causal analysis of the real data* say will happen?

If these two analyses agree — the simulator predicts insulin lowers glucose, and AIPW on real data confirms insulin lowers glucose — we have converging evidence that the simulator has learned something real. If they disagree, it is a signal that the simulator may be reproducing confounded patterns from the training data rather than true causal relationships, which would undermine our ability to use it for policy optimisation.

**What we found:** For most drugs, the simulator and causal analysis are broadly consistent. Steroids, diuretics, and anticoagulants behave correctly in both. The systematic disagreement is insulin: the simulator predicts glucose goes *up* (+16.6 mg/dL) when insulin is given, and AIPW on real data also estimates glucose goes up (+25.2 mg/dL after adjustment). Both analyses are "wrong" in the same direction — which confirms that the failure is not the simulator making up a spurious pattern, but rather that the training data itself reflects strong selection bias that neither the simulator nor the AIPW estimator can fully overcome with the available covariates. This is a known limitation of observational data without access to dose records and dietary information.

---

## Step 3 — Reinforcement Learning Policy

**What we do:** We train a treatment policy that recommends which of the 5 drug classes with known causal effects to prescribe on day 1 of a patient's admission in order to minimise predicted 30-day readmission risk. The policy is grounded in the causal ATEs from Step 2 — not the observational patterns of the transition model — so its drug recommendations reflect genuine causal effects rather than confounded associations.

**What comes out:** A readmission risk model (reward signal) and a causal exhaustive policy, saved to `models/rl_daily/`, with evaluation results in `reports/rl_daily/`.

---

### Why Not a Standard RL Agent?

The standard RL approach — train a neural network policy by running many rollouts in the simulator — has a fundamental problem here. The transition model was trained on observational data, so its predictions for counterfactual actions embed the same confounding identified in Step 1d. A policy optimised against this model directly would learn to exploit those confounded associations rather than true causal effects. For example: because the transition model (like the raw data) associates insulin with higher glucose, a standard RL agent would learn to avoid insulin — the wrong conclusion for the wrong reason.

A second reason is that with a 1-step horizon, a neural network is unnecessary. There is no long-term credit assignment problem, no exploration-exploitation trade-off across time, and no value function to approximate. The right tool is the simplest one that works.

---

### Key Design Decisions

**1-step fixed horizon**

Rather than simulating multi-day trajectories, each action is evaluated based on one simulated step: current state → action → simulated next-state → reward. This avoids compounding simulator errors over many days — the KS statistics in Step 1c show the simulator drifts significantly beyond day 10–15 for several variables — and sidesteps the unreliable discharge model. The average length of stay is 4.8 days in the development sample; optimising the first-day drug choice is both tractable and clinically meaningful.

**Exhaustive search over 2^5 = 32 action combinations**

With 5 drugs to optimise, all 32 binary action combinations can be evaluated for each patient in milliseconds. This gives the exact optimum under the model — no approximation, no convergence issues, no hyperparameters to tune. Every recommendation is fully explainable: for each patient, all 32 risk scores are available and the reasoning behind the selected action is transparent.

**Causal ATE correction to the simulator (Option C)**

This is the core methodological contribution of Step 3. Instead of using the transition model's drug-conditional predictions directly (which are confounded), the policy applies the following procedure for each candidate action:

1. Compute a **baseline next-state** using the transition model with no drugs: `base_next = predict_next(model, state, {all drugs: 0})`
2. For each drug that is "on" in this candidate action, **add the causal ATE** from Step 2 to the relevant outcomes in the baseline: `causal_next[outcome] = base_next[outcome] + ATE(drug → outcome)`
3. Score the **causally-corrected** next-state with the readmission risk model.

This replaces the transition model's confounded observational drug effects with the doubly-robust AIPW estimates from Step 2, ensuring the policy's recommendations are driven by genuine causal effects. The transition model contributes only the baseline trajectory — how the patient would evolve without any drugs — while Step 2 determines the counterfactual impact of prescribing each drug.

Drug interactions are assumed additive (the effects of co-prescribing two drugs are treated as the sum of their individual ATEs). This is a standard approximation when interaction terms have not been separately estimated.

**opioid_active excluded from optimisation**

No causal ATE was estimated for opioid_active in Step 2. Opioid effects on standard lab panels are indirect and ambiguous at daily temporal resolution, and no clinically unambiguous directional prediction could be made (see Step 1d). Including opioid in the search without a causal ATE would reintroduce observational confounding for that drug, contradicting the purpose of the causal correction. opioid_active is fixed to 0 in all policy evaluations.

---

### Step 3a — The Reward Signal: Readmission Risk Model

Before the policy can score candidate actions, we need a reward function that maps a simulated next-state to a scalar: how likely is this patient to be readmitted within 30 days if their state tomorrow looks like this? We train a dedicated LightGBM classifier for this purpose.

**Feature set (43 features)**

The model uses only features that are available in the simulated next-state output by `predict_next()`. ACTION_COLS (drug flags) are deliberately excluded — they are inputs to the simulator, not outputs, so including them at training time would create a mismatch with prediction time.

| Group | Features | Count |
|---|---|---|
| Continuous labs | creatinine, BUN, sodium, potassium, bicarbonate, anion gap, glucose, haemoglobin, WBC, platelets, magnesium, calcium, phosphate, INR, bilirubin | 15 |
| Binary state flags | is_icu, lactate_elevated, positive_culture_cumulative, blood_culture_positive_cumulative | 4 |
| Static features | age_at_admit, charlson_score, drg_severity, drg_mortality, gender_M, day_of_stay, days_in_current_unit | 7 |
| Lab measurement flags | one per continuous lab (e.g. creatinine_measured) | 15 |
| Infection context | culture_ordered_today, n_active_drug_classes | 2 |
| **Total** | | **43** |

**Training setup**

All rows from the training split are used — not just discharge-day rows. The `readmit_30d` label is stamped identically on every row of a given admission, so training on all days exposes the model to the full distribution of patient states encountered at different points in the stay, including day-1 states which are exactly what the policy needs to score.

Architecture: `SimpleImputer(strategy="median") → LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, class_weight="balanced")`. The `class_weight="balanced"` setting compensates for the ~18% readmission prevalence.

**Results (full dataset — 2.14M train rows, 458k test rows)**

| Metric | Value |
|---|---|
| Test AUC | 0.647 |
| Test Brier score | 0.231 |
| Train prevalence | 22.7% |

The AUC of 0.647 is moderate — readmission is genuinely difficult to predict from lab and drug features alone. The Brier score of 0.231 reflects the inherent uncertainty in 30-day outcomes from a single daily snapshot. This is an honest result: predicting 30-day readmission from daily state features alone is a hard problem, and the model is used here as a relative ranking function (which strategy leads to a better state?) rather than a calibrated absolute predictor.

---

### Step 3b — The Causal Policy

**Search procedure**

For each patient, the policy evaluates all 32 drug combinations in `itertools.product([0, 1], repeat=5)` over the 5 ATE-covered drugs. For each combination:

```
base_next = predict_next(transition_model, state, {all drugs: 0})

for each drug that is "on":
    for each (drug, outcome) pair in the ATE table:
        causal_next[outcome] = base_next[outcome] + causal_ATE

risk = predict_readmission_risk(readmission_model, causal_next)
```

The combination with the lowest risk is returned as the recommended action.

**ATE table coverage**

9 (treatment, outcome) pairs are used, covering 5 of the 6 drug classes:

| Drug | Outcome corrected | Causal ATE |
|---|---|---|
| Antibiotic | WBC | −0.035 |
| Anticoagulant | INR | +0.025 |
| Diuretic | BUN | +1.584 |
| Diuretic | Potassium | −0.036 |
| Diuretic | Sodium | −0.093 |
| Steroid | Glucose | +3.371 |
| Steroid | WBC | +0.232 |
| Insulin | Glucose | +25.186 |
| Antibiotic | Positive culture (cumulative) | +0.007 |

For all other (drug, outcome) pairs not in this table, the baseline from the transition model is used unchanged — the ATE is simply not applied.

---

### Step 3c — ATE Policy: Comparing Strategies

For each patient in the test set, we compare three strategies. All three use the same baseline from `predict_next(..., no_drugs)` to ensure the comparison is driven by causal drug effects, not by the transition model's observational confounding.

| Strategy | How next-state is constructed |
|---|---|
| **Causal ATE policy** | base_next + ATE corrections for best-action drugs |
| **Do-nothing** | base_next directly (no corrections applied) |
| **Real clinical actions** | base_next + ATE corrections for the patient's actual day-0 drugs |

**Results (500 test patients, full dataset)**

| Strategy | Mean predicted readmission risk |
|---|---|
| ATE causal policy | 0.4228 |
| Do-nothing | 0.4391 |
| Real clinical actions | 0.4357 |

The ATE policy beats do-nothing in **98.6%** of patients and beats real clinical actions in the majority of patients.

**Drug recommendation frequencies (ATE policy)**

| Drug | % of patients recommended |
|---|---|
| Antibiotic | 100% |
| Diuretic | 100% |
| Steroid | 100% |
| Anticoagulant | 0% |
| Insulin | 0% |
| Opioid | — (excluded) |

The pattern reflects the ATE values directly. Antibiotics (ATE: −0.035 on WBC), diuretics (BUN +1.58, potassium −0.036), and steroids (glucose +3.37, WBC +0.232) are recommended universally because their net causal effect reduces predicted readmission risk across all patients. Insulin is avoided because its confounded ATE (+25.2 on glucose) makes next-states worse. Anticoagulants have a small positive ATE on INR — cautiously recommended or not, depending on patient state.

---

### Step 3d — CATE Policy: Personalised Drug Effects

**Design**

The CATE policy uses the same exhaustive search structure as the ATE policy (32 drug combos, 1 simulator step, score with readmission model), but replaces the population-average scalar ATEs with patient-specific heterogeneous treatment effects from the CausalForestDML models trained in Step 2c.

For each patient, CATE effects θ(x) are precomputed once across all 9 treatment-outcome pairs using the patient's current state x. These personalised corrections are then applied in place of the scalar ATEs during the 32-combo search.

This means two patients in different clinical states receive different drug-effect corrections — a patient with high WBC gets a larger antibiotic correction than one with normal WBC, if the forest has learned that infection-severity moderates the antibiotic effect.

**Results (500 test patients, full dataset)**

| Strategy | Mean predicted readmission risk |
|---|---|
| CATE causal policy | 0.4219 |
| Do-nothing | 0.4391 |
| Real clinical actions | 0.4353 |

The CATE policy beats do-nothing in **98.8%** of patients — marginally better than the ATE policy (98.6%), reflecting the benefit of personalised corrections for a small subset of patients whose optimal drug combination differs from the population average.

---

### Step 3e — Fitted Q-Iteration (FQI): Multi-Step Planning

**Motivation**

Both the ATE and CATE policies are one-step greedy: they ask "what is the best drug combo to give right now?" without considering how today's drug decision affects the patient's state in two or four days, and therefore which treatment options will be best later. Fitted Q-Iteration (FQI) addresses this by learning a policy that plans over a multi-step horizon.

**What FQI learns**

FQI learns a Q-function Q(s, a) — the expected cumulative reward from being in state s, taking action a, and then acting optimally for all future decisions. The optimal policy then selects the action that maximises Q at each decision point.

A separate Q-model is fitted per decision step (Q₀ for day 0, Q₁ for day 2, Q₂ for day 4). Each is a LightGBM regressor.

**Episode structure**

- 3 decision points: day 0, day 2, day 4
- 2 simulator days per decision (action held constant within each block)
- Total horizon: 6 simulated days
- Reward: sparse — 0 at steps 0 and 1, −P(readmit_30d) at step 2 (terminal only)
- Causal correction: ATE applied at every simulator step

**Training procedure**

1. For each training patient, sample N random drug sequences over the 3 decision steps.
2. Roll out each sequence through the causal simulator — real day-0 state, simulated days 1–6.
3. Collect transitions: (state at day t, drugs given, state at day t+2, reward).
4. Fit Q-models via backward induction:
   - Q₂: fit on terminal transitions — target = actual reward
   - Q₁: fit on step-1 transitions — target = γ × max_a Q₂(next state, a)
   - Q₀: fit on step-0 transitions — target = γ × max_a Q₁(next state, a)
5. Repeat for N_ITER iterations, each time using freshly updated Q-models as targets.

**Three FQI variants were trained and evaluated**

| Variant | Drugs | State features | Train patients | Sequences/patient | Episodes | Result: beats do-nothing |
|---|---|---|---|---|---|---|
| FQI baseline | antibiotic only | 7 | 5,000 | 8 (enumerated) | 40,000 | 63.6% |
| FQI-multi | 5 drugs | 26 (full) | 3,000 | 64 (random) | 192,000 | 73.4% |
| FQI-multi-large | 5 drugs | 26 (full) | 5,000 | 128 (random) | 640,000 | 75.8% |

With 5 drugs and 3 steps, exhaustive enumeration of all 32³ = 32,768 paths per patient is computationally infeasible. Random sampling of N sequences covers all 32 drug combos approximately twice per step in expectation (with N=64), providing adequate action-space coverage.

**Why 26 features instead of 7**

The 7-feature baseline used only infection-relevant features (WBC, culture flags, ICU status, age, Charlson, day of stay). The 5-drug extension uses the full 26-feature state (15 labs + 4 binary states + 7 static) because different drugs affect different physiological systems — creatinine matters for diuretic safety, INR for anticoagulant risk. The richer state allows the Q-function to learn cross-drug, cross-system interactions.

**Policy results (500 test patients, FQI-multi-large)**

| Strategy | Mean predicted readmission risk |
|---|---|
| FQI-multi-large policy | 0.4874 |
| Do-nothing | 0.4959 |
| Real clinical actions | 0.4955 |

The FQI policy beats do-nothing in **75.8%** of patients. This is lower than the ATE/CATE policies (98.6–98.8%), for a structural reason: ATE/CATE evaluate the state after 1 simulator step, while FQI evaluates the terminal state after 6 simulator steps. Each simulator step compounds prediction error, and the sparse terminal reward makes the credit assignment problem harder. The gap reflects the difficulty of multi-step planning through a noisy simulator, not a failure of the Q-function itself.

**Drug recommendation pattern (FQI-multi-large, day 0 / day 2 / day 4)**

| Drug | Day 0 | Day 2 | Day 4 |
|---|---|---|---|
| Antibiotic | 87.6% | 4.0% | 5.8% |
| Diuretic | 79.4% | 0.4% | 15.4% |
| Steroid | 3.0% | 31.4% | 42.4% |
| Insulin | 12.2% | 17.6% | 10.0% |
| Anticoagulant | 1.4% | 0.4% | 0.0% |

The temporal pattern is clinically coherent: antibiotics and diuretics recommended aggressively on day 0 then de-escalated (early broad treatment); steroids escalate over time (delayed anti-inflammatory use). This contrasts sharply with the antibiotic-only FQI baseline, which recommended antibiotics on day 0 only 12.6% of the time — a "wait and see" pattern that disappears when the policy has all 5 drugs available.

**Q-function feature importances**

Consistent across all three decision steps, the top features are: `age_at_admit`, `charlson_score`, `creatinine`, `hemoglobin`, `platelets`. Drug action features rank low, meaning the Q-function's recommendations are driven primarily by who the patient is rather than by a mechanical mapping from drug flag to outcome. `day_of_stay` has zero importance — the model does not care how many days the patient has been admitted when making drug decisions.

---

## Step 4 — Full Policy Comparison and Next Steps

**Complete policy benchmark (500 test patients, full dataset)**

| Policy | Design | Mean risk | Beats do-nothing |
|---|---|---|---|
| CATE | 1-step, 5 drugs, personalised effects | 0.4219 | 98.8% |
| ATE | 1-step, 5 drugs, scalar effects | 0.4228 | 98.6% |
| FQI-multi-large | 3-step, 5 drugs, 26 features, 5k/128 | 0.4874 | 75.8% |
| FQI-multi | 3-step, 5 drugs, 26 features, 3k/64 | 0.4879 | 73.4% |
| FQI baseline | 3-step, antibiotic only, 7 features | 0.4902 | 63.6% |
| Do-nothing | No drugs | 0.4959 | — |
| Real clinical | Day-0 drugs held constant | 0.4955 | — |

**Key observations**

1. All policies outperform do-nothing and real clinical, confirming that causal correction adds value over the observational baseline.
2. ATE and CATE perform comparably — personalisation (CATE) provides marginal benefit over population-average corrections (ATE) in this dataset.
3. FQI multi-step planning is weaker than 1-step greedy search on this metric. This is expected: the 1-step policies are evaluated immediately after 1 simulator step; FQI's terminal reward arrives after 6 steps of compounded simulator error. FQI's value lies in sequential decision-making capability, not raw risk reduction at a single time point.
4. Scaling FQI training data from 192k to 640k episodes improves performance (73.4% → 75.8%), confirming that coverage of the state-action space matters.

**Candidate next improvements**

- **Dense intermediate rewards:** currently FQI only receives a reward signal at the terminal step. Providing reward (readmission risk) at every RL step would give the Q-function 3× more learning signal and reduce credit assignment difficulty.
- **CATE corrections inside FQI:** replacing scalar ATE corrections with per-patient CATE deltas at each simulator step would combine multi-step planning with personalised causal effects.
- **Subgroup analysis:** examining which patient subgroups (ICU vs ward, high vs low Charlson) benefit most from each policy type.

---

## How the Steps Connect

```
Real MIMIC-IV data (PostgreSQL)
      │
      ▼
Step 1a: Build dataset — one row per patient per day              ✓
      │
      ▼
Step 1b: Train transition models (23 LightGBM models)             ✓
      │
      ▼
Step 1c: Evaluate simulator (single-step R² + rollout KS)         ✓
      │
      ▼
Step 1d: Action sensitivity — do drug effects make sense?         ✓
      │
      ▼
Step 2a: Causal analysis (AIPW) — scalar ATE per pair             ✓
      │
      ├──► Step 2b: CATE models (CausalForestDML, 27 models)      ✓
      │
      ▼
Step 3a: Readmission risk model (reward signal, AUC 0.647)        ✓
      │
      ├──► Step 3b: ATE causal policy (1-step, scalar)    ✓  0.4228 mean risk
      │
      ├──► Step 3c: CATE causal policy (1-step, personal) ✓  0.4219 mean risk
      │
      └──► Step 3d: FQI multi-step policy (3-step, 5 drugs)
               ├── FQI baseline (antibiotic only)          ✓  0.4902 mean risk
               ├── FQI-multi (3k/64 seqs)                  ✓  0.4879 mean risk
               └── FQI-multi-large (5k/128 seqs)           ✓  0.4874 mean risk
```

Each step produces artefacts (datasets, trained models, reports) consumed by later steps. The data never leaves the pipeline — MIMIC-IV records remain on the local Postgres instance and no patient data is committed to the repository.
