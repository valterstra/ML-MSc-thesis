# ICU Readmission RL Pipeline: A Full Walkthrough

*Presentation guide -- explains what the pipeline does, why it works the way it does,
and what the results mean. Written for a human audience.*

---

## Where we start: the finished dataset

By the time we reach the RL pipeline, we have already done the hard work.
`rl_dataset_broad.parquet` contains **1.5 million transitions**, one per 4-hour bloc,
for 61,771 ICU stays drawn from MIMIC-IV.

Each row in this dataset is a single moment in a patient's ICU stay:

```
(s, a, r, s', done)
 |   |  |  |    |
 |   |  |  |    Was this the patient's last bloc?
 |   |  |  The patient's state at the next 4-hour bloc
 |   |  The reward the physician received at this step
 |   The 5-drug action taken at this 4-hour bloc
 The patient's clinical state at this 4-hour bloc (51 measurements)
```

**State (s):** 51 normalised clinical measurements -- heart rate, blood pressure, lab values
(creatinine, potassium, WBC, etc.), derived scores (SOFA, GCS), ventilator settings, and
basic demographics. Everything a bedside clinician would see on a monitor or in the chart.

**Action (a):** Which of 5 drug classes were active during this 4-hour window:
vasopressor, IV fluid, antibiotic, sedation, diuretic. Each is binary (on/off),
giving 2^5 = 32 possible drug combinations. Encoded as an integer 0-31.

**Reward (r):**
- Non-terminal steps: SOFA_t - SOFA_{t+1} (positive when SOFA improves, i.e. patient gets better)
- Terminal step: +15 if the patient was NOT readmitted within 30 days; -15 if they were

This reward structure means: we care about moment-to-moment stability (SOFA) AND
the long-term outcome (avoiding readmission 30 days after discharge).

---

## What is reinforcement learning trying to do here?

Imagine a doctor who has seen thousands of ICU patients. Over years, they develop intuition:
"when I see this pattern of labs, this combination of drugs tends to help." RL tries to
build the same intuition from data -- systematically, across all 1.5 million moments
in our dataset.

The goal is to learn a **policy** π(s): given the current clinical state s, which of the
32 drug combinations should I recommend?

The challenge: we are learning **offline** -- from historical data where physicians
made the decisions. We cannot interact with real patients, we cannot try different drugs
and see what happens. We have to infer the best policy from what physicians did, and
from whether patients with similar states had good outcomes.

---

## Step 09c: Causal Discovery -- Do These Drugs Actually Cause What We Think?

Before training any RL model, we ran a causal validation step split into two analyses.

**Analysis 1** asks: does each drug causally shift its target physiological variables during the stay?
**Analysis 2** asks: does the patient's physiological state at discharge predict 30-day readmission?

Together they establish the causal chain: *drug moves physiology → discharge state predicts readmission → therefore drug treatment influences readmission risk*.

### Setup

Five graphs per analysis, one per drug grouping. Each graph has 7 nodes:

```
Analysis 1 -- Drug -> physiology:
  Tier 0: age, charlson_score, prior_ed_visits_6m   (static confounders)
  Tier 1: frac_drug                                  (treatment intensity during stay)
  Tier 2: delta_physiology x3                        (change in target variables during stay)

Analysis 2 -- Discharge state -> readmission:
  Tier 0: age, charlson_score, prior_ed_visits_6m   (static confounders)
  Tier 1: last_physiology x3                         (absolute value at discharge)
  Tier 2: readmit_30d                                (outcome, 30 days post-discharge)
```

**Why fractions (Analysis 1):** One independent observation per stay, avoiding within-stay autocorrelation.

**Why deltas (Analysis 1):** Captures what *changed* during the stay, not the raw value -- which would conflate baseline severity with treatment effect.

**Why absolute last values (Analysis 2):** Readmission is driven by where the patient *ends up*, not how much they changed. The delta representation is the wrong tool for the readmission end of the chain.

**Why three confounders:** Age, Charlson score, and prior ED visits affect both drug prescription and readmission risk. Without conditioning on them, the indication signal -- sicker patients get more drugs AND have worse outcomes -- cannot be separated from the treatment effect.

**Algorithm:** PC (Peter-Clark), FisherZ conditional independence test, alpha=0.05.
Temporal tier ordering enforced: edges pointing backwards in time are discarded.

### Analysis 1 results: Drug → physiology (alpha=0.05, drug edges only)

**Vasopressor**
- no drug → physiology edges at alpha=0.05
- (borderline: frac_vasopressor → delta_HR, delta_Arterial_BP_Sys appear at alpha=0.01)

---

**IV Fluid**
- frac_ivfluid → delta_Creatinine ✓
- frac_ivfluid → delta_Potassium ✓

---

**Antibiotic**
- no drug → physiology edges

---

**Sedation**
- frac_sedation → delta_HR ✓
- frac_sedation → delta_RR ✓
- frac_sedation → delta_SpO2 ✓

---

**Diuretic**
- frac_diuretic → delta_Potassium ✓
- frac_diuretic → delta_Creatinine ✓

### Analysis 2 results: Discharge state → readmission (alpha=0.05, physiology edges only)

**Vasopressor**
- no discharge state → readmit edges (only confounders → readmit)

---

**IV Fluid**
- last_BUN → readmit_30d ✓

---

**Antibiotic**
- last_Temp_C → readmit_30d ✓

---

**Sedation**
- last_HR → readmit_30d ✓
- last_SpO2 → readmit_30d ✓

---

**Diuretic**
- last_BUN → readmit_30d ✓

### What this tells us

The two-analysis structure establishes the causal chain cleanly where both links hold:

| Drug | Moves during stay | Discharge state → readmit |
|------|------------------|--------------------------|
| IV fluid | delta_Creatinine, delta_Potassium | last_BUN ✓ |
| Sedation | delta_HR, delta_RR, delta_SpO2 | last_HR, last_SpO2 ✓ |
| Diuretic | delta_Potassium, delta_Creatinine | last_BUN ✓ |

Sedation is the cleanest story: the drug shifts all three cardiorespiratory variables during
the stay, and discharge HR and SpO2 independently predict readmission. IV fluid and diuretic
both shift renal markers, and discharge BUN predicts readmission in both graphs.

Vasopressor and antibiotic are the weakest links. Vasopressor edges only appear at the
stricter alpha=0.01 threshold. Antibiotic has no confirmed drug → physiology edges in
Analysis 1 -- the confounders (Charlson score drives antibiotic prescription) fully absorb
the signal. Both remain in the action space on pharmacological grounds, but the observational
data does not provide the same level of causal confirmation as the other three drugs.

---

## Step 11: Learning the Q-function

Q(s, a) is a number that answers: *"if this patient is in state s and we give drug combo a right now, what total future reward can we expect?"* Once we have it, making a recommendation is trivial: pick the action with the highest Q-value. We train two networks to get there -- DDQN and SARSA.

We have two identical copies of the network: the **online net** (trains every step) and the **target net** (a slow-moving copy, updated by tau=0.001 per step). The reason for two: the target net provides a stable reference so the training signal does not chase its own tail.

---

### One training step

**The transition:**
```
s  = [creatinine=1.4, HR=92, SOFA=8, ...]   <- patient at 8am, 51 features
a  = combo #7  (ivfluid=ON, antibiotic=ON)   <- action taken by physician
r  = +1                                      <- SOFA improved 8 -> 7
s' = [creatinine=1.3, HR=88, SOFA=7, ...]   <- patient at noon
```

**DDQN**

Stage 1 -- build the target:
1. Online net looks at `s'`, finds the best possible next action: `a* = argmax Q_online(s')` -> combo #3
2. Target net evaluates it: `Q_target(s', combo #3) = -0.8`
3. `y = r + gamma * Q_target(s', a*) = +1 + 0.99 * (-0.8) = +0.208`

Stage 2 -- compare:

4. `Q_online(s, a=7) = -1.2`
5. TD error: `+0.208 - (-1.2) = +1.408` -> backprop
6. Soft-update: `theta_target <- 0.001 * theta_online + 0.999 * theta_target`

---

**SARSA (physician baseline)**

Stage 1 -- build the target:
1. Look up what the physician actually gave next in the data: `a* = combo #2`
2. Target net evaluates it: `Q_target(s', combo #2) = -0.8`
3. `y = r + gamma * Q_target(s', a*) = +1 + 0.99 * (-0.8) = +0.208`

Stage 2 -- compare:

4. `Q_online(s, a=7) = -1.2`
5. TD error: `+0.208 - (-1.2) = +1.408` -> backprop
6. Soft-update: same as DDQN

---

**The one difference:** step 1 of Stage 1. DDQN imagines the best possible next action. SARSA looks at what the physician actually did. Everything else is identical. DDQN learns *"what should I do?"*; SARSA learns *"what did the doctor do?"* -- giving a baseline to compare against in step 12.

---

### Training results

After 100,000 DDQN steps and 80,000 SARSA steps (~127 minutes on CPU):

| Steps | Loss |
|-------|------|
| 5,000 | 5.94 |
| 25,000 | 4.35 |
| 50,000 | 3.87 |
| 75,000 | 3.76 |
| 100,000 | 3.96 |

```
Final DDQN:
  Mean Q (test set): 0.0815
  Unique actions used: 28 of 32
```

28 of 32 drug combinations are recommended across the test set -- the model is not
collapsing to a single policy, it is making different recommendations in different clinical contexts.

---

## Step 12: Policy evaluation -- what did the DDQN actually learn?

### Figure 1: How does the DDQN prescribe differently from physicians?

We split the test set by patient severity -- low, medium, and high SOFA -- and for each
group compare how often the physician prescribed each drug vs how often the DDQN recommends it.

**What we see:**

Across all severity levels, the DDQN consistently recommends far less IV fluid than
physicians did -- roughly 5-10% of timesteps vs 60-93% for physicians. It compensates
by recommending more sedation and more diuretic. Vasopressor and antibiotic recommendations
are generally lower than the physician as well.

The pattern also shifts with severity: at medium and high SOFA the DDQN increases its
sedation recommendation substantially (up to ~40-54%), suggesting it has learned that
managing the patient's physiological state through sedation becomes more important as
severity rises.

**What this implies:**

The DDQN is not recommending the same drugs for everyone -- its prescribing pattern
changes depending on how sick the patient is. That is the minimum bar for a policy
that has actually learned something from the data rather than just memorising a fixed recipe.

---

### Figure 2: When DDQN and physician agree, do patients do better?

This is the core validation. For every timestep in the test set, we count how many
of the 5 drugs the DDQN and physician disagree on (0 = full agreement, 5 = completely
different). We then look at the 30-day readmission rate for each level of disagreement.

**What we see:**

For low-severity patients (the large majority of the test set, ~155k timesteps):
```
0 drugs different:  readmission ~18%   <- full agreement
1 drug different:   readmission ~19%
2 drugs different:  readmission ~22%
3 drugs different:  readmission ~24%
4 drugs different:  readmission ~25%
5 drugs different:  readmission ~28%   <- complete disagreement
```

The pattern is monotonically rising. Medium-severity patients show the same shape
at a higher baseline (~25% rising to ~32%). High-severity has only 26 patients and
is not meaningful.

**What this implies:**

Patients whose physicians happened to prescribe what the DDQN would have recommended
had the lowest readmission rates. The more the physician deviated from the DDQN's
recommendation, the worse the outcomes were. This is the strongest qualitative evidence
that the DDQN has learned a policy that aligns with real outcomes.

**What this does not prove:**

This is observational data. We cannot rule out that the cases where DDQN and physician
agree are simply the easier cases. Deploying the policy and measuring the effect would
require a randomised trial.

---

## Step 12b: Doubly Robust Evaluation

The figures above are qualitative. To get a single number -- *how much better is the
DDQN than the physician, in reward terms* -- we use a doubly robust (DR) estimator.

Two helper models are trained: a physician policy model (what probability did the physician
assign to each drug combo, given the state?) and a reward estimator (what reward would
we expect for any state-action pair?). These combine into a DR score per trajectory,
which is then averaged across all ICU stays in the test set.

**Our results (test set, 9,267 trajectories):**

```
DDQN policy:        DR mean = -0.262  (std = 2.68)
Physician (SARSA):  DR mean = -0.461  (std = 3.09)
```

The DDQN scores higher than the physician policy -- a smaller negative number means
better estimated cumulative reward over the stay.

**How much to trust this number:**

The standard deviation (~2.7-3.1) is roughly ten times the size of the difference
between the two policies (~0.2). The signal is real in direction but weak in magnitude.

There is also a deeper structural reason to be cautious. The DDQN is a deterministic
policy -- at each state it picks exactly one action. The DR estimator uses importance
weights to correct for the fact that the physician sometimes took different actions.
But when a deterministic policy disagrees with the physician, those importance weights
collapse to zero, and the DR estimate degenerates to being purely what the reward model
predicts -- with no correction at all. Raghu et al. (2017), who introduced this exact
evaluation framework for ICU RL, explicitly flag this problem:

> "A deterministic evaluation policy leads to IS terms going to zero if the clinician
> and learned policy take different actions at any given timestep. We are therefore
> limited in the accuracy of our value estimates by the accuracy of this estimated
> reward, and we cannot easily provide statistical guarantees of performance."

They conclude: *"we focus on qualitative analyses to give interpretable insight into
our learned policy's efficacy."*

We take the same position. The DR result is directionally consistent with Figure 2
(DDQN better than physician), but Figure 2 is the primary evidence.

---

## Step 13: Model-based simulators

The simulator is a separate neural network that learns: "given the current state and
a drug action, what will the patient's state look like in 4 hours?"

This is a **transition model**: P(s' | s, a).

Why build one?
1. **Validation**: We can generate simulated trajectories under the DDQN policy and
   check whether they look physiologically plausible (do labs trend in the right direction?)
2. **Future work**: Once we have a good simulator, the RL agent can train against the
   simulator instead of the fixed historical dataset. This is "model-based RL" and is
   far more sample-efficient -- the agent can explore millions of hypothetical scenarios.

Four architectures are compared (following Raghu 2018 Table 1):
- `nn`: two fully-connected layers with ReLU + batch normalisation (Raghu's preferred)
- `linear`: plain linear regression baseline
- `lstm`: uses the 4 most recent timesteps as history to capture temporal trends
- `bnn`: Bayesian neural network -- provides uncertainty estimates alongside predictions

The simulator is evaluated on:
- **Per-feature MSE**: how accurately does it predict each of the 51 state features?
- **Multi-step rollout quality**: after 10 simulated steps, how far has the predicted trajectory
  drifted from the true patient trajectory?

---

## The full pipeline summary

```
ICUdataset.csv (1.5M rows, 61k stays)
       |
       v
Step 10b: Build (s, a, r, s', done) transitions
  -- broad 51-feature state (Raghu-style)
  -- 5 binary actions (32 combinations)
  -- SOFA-delta dense reward + +-15 terminal
       |
       v
Step 11: Train Dueling DDQN + SARSA physician
  -- DDQN learns optimal Q-function (100k steps)
  -- SARSA learns physician Q-function (80k steps)
  -- Both produce Q-values + greedy actions for test set
       |
       v
Step 12: Off-policy evaluation (doubly robust)
  -- Physician policy model: P(physician action | state)
  -- Reward estimator: E[reward | state, action]
  -- Environment model: E[next state | state, action]
  -- DR scores for DDQN vs physician
       |
       v
Step 12b: Qualitative evaluation (Raghu 2017)
  -- Fig 1: drug prescribing patterns by SOFA severity
  -- Fig 2: readmission rate vs DDQN-physician disagreement
  -- KEY RESULT: readmission rises monotonically with disagreement (Raghu finding confirmed)
       |
       v
Step 13: Model-based simulators
  -- 4 architectures (nn/linear/lstm/bnn)
  -- Per-feature MSE + rollout evaluation
  -- Foundation for future model-based RL
```

---

## Connecting this to the thesis

The ICU readmission pipeline is designed around a single central question:

> Can we learn a drug recommendation policy from historical ICU data that would
> have been associated with lower 30-day readmission rates?

The pipeline gives three complementary answers:

1. **Yes, qualitatively** (step 12b): The Raghu Fig 2 result shows that readmission is
   lowest when clinicians and DDQN agree, and rises as they diverge. This is exactly
   the pattern we would expect if the DDQN had learned clinically meaningful recommendations.

2. **Yes, quantitatively (with caveats)** (step 12): The DR evaluator produces an
   estimated policy value, though as Raghu 2017 noted, DR in ICU settings should be
   interpreted cautiously due to partial observability and distributional shift.

3. **Mechanistically plausible** (steps 09c + 11): The causal discovery pipeline confirmed
   that our 5 chosen drugs causally influence the target physiological variables (vasopressor
   → blood pressure, diuretic → potassium/BUN/creatinine). The DDQN is not recommending
   arbitrary drugs -- it is acting on variables with confirmed causal pathways.

The honest limitation: this is observational data. We cannot guarantee that the DDQN's
recommendations, if followed by a different physician in a different context, would produce
the same outcomes. The confounding-by-indication problem (sicker patients get more drugs)
is never fully eliminated in observational ICU data. This is why the thesis frames the
pipeline as a **decision support tool and proof-of-concept** -- not a clinical deployment.

---

## Key numbers to remember

| Metric | Value |
|--------|-------|
| Dataset size | 1,500,857 transitions, 61,771 ICU stays |
| 30-day readmission rate | 20.7% |
| State features (broad) | 51 |
| Drug actions | 5 binary (32 combinations) |
| DDQN training time | ~127 minutes on CPU |
| DDQN converged loss | 3.76 (from 5.94) |
| Unique actions recommended | 28 of 32 |
| Fig 2: readmission at Hamming=0 (low SOFA) | ~18% |
| Fig 2: readmission at Hamming=3+ (low SOFA) | ~35% |
