# CARE-Sim Transformer Landscape Review

Date: 2026-04-06

Purpose: document the current external landscape for transformer-based simulators/world models relevant to the ICU readmission CARE-Sim pipeline, and compare that landscape to the current Step 14-16 implementation in this repository.

---

## 1. Scope

This note focuses first on transformer-based methods because that was the recommended direction for the simulator. The review is centered on the actual CARE-AI problem:

- build a patient simulator / world model from longitudinal clinical data
- condition future trajectories on interventions
- use the simulator as an environment for planning or reinforcement learning
- keep causality and treatment-confounding concerns in scope

The review is therefore not a generic survey of healthcare transformers. It is specifically about models that are relevant to:

- patient trajectory simulation
- intervention-conditioned forecasting
- offline RL / model-based RL
- counterfactual or deconfounded sequence modeling

---

## 2. Current CARE-Sim Position in This Repo

The current simulator stack is:

- Step 14: train CARE-Sim transformer world model
- Step 15: evaluate one-step and rollout fidelity
- Step 16: use CARE-Sim as the environment for planner and DDQN control

Relevant local files:

- `src/careai/icu_readmit/caresim/model.py`
- `src/careai/icu_readmit/caresim/simulator.py`
- `scripts/icu_readmit/step_14_caresim_train.py`
- `reports/icu_readmit/caresim/caresim_summary.json`
- `reports/icu_readmit/caresim_control/step_16_summary.json`
- `models/icu_readmit/caresim/run_meta.json`

### What CARE-Sim actually is

CARE-Sim is a causal-mask-capable GPT-style transformer world model over ICU trajectories:

- input per step: state and action
- model attends over the full patient history
- outputs:
  - next state
  - reward
  - terminal probability

The simulator environment wraps the trained ensemble and allows:

- seeding with a real patient history
- stepping forward under proposed actions
- short-horizon rollout for policy search and evaluation

### Important implementation finding

The current saved Step 14 model was trained with:

- `use_causal_constraints = false`
- `causal_constraints = false`

This means the currently deployed CARE-Sim is the unconstrained predictive transformer baseline, not the FCI-masked causal variant already supported in code.

### Current empirical status

From `reports/icu_readmit/caresim/caresim_summary.json`:

- one-step validation next-state MSE: about `0.0830`
- one-step test next-state MSE: about `0.0834`
- reward MAE: about `1.87`
- terminal accuracy: about `0.956`
- rollout test step-5 state MSE: about `0.176`

From `reports/icu_readmit/caresim_control/step_16_summary.json`:

- planner is the best controller
- DDQN beats random and repeat-last
- DDQN remains partly action-collapsed

Interpretation:

- CARE-Sim is usable as a short-horizon control environment
- current evidence supports predictive usefulness
- current evidence does not yet establish strong causal credibility of action effects

---

## 3. Main Conclusions

### 3.1 High-level conclusion

The transformer choice is defensible.

There is no obvious off-the-shelf transformer model in the current literature that cleanly dominates the current CARE-Sim design for:

- tabular ICU time series
- action-conditioned state transition modeling
- short-horizon model-based control

However, there are stronger methods than current CARE-Sim on the causal counterfactual side.

### 3.2 Most important conclusion

The main risk in the thesis is not:

- "CARE-Sim used the wrong model family"

The main risk is:

- "CARE-Sim is currently a predictive simulator, but not yet a strongly causal counterfactual simulator"

### 3.3 Practical thesis stance

The current system supports the following claim well:

- we built a short-horizon transformer world model for ICU control and used it successfully as a simulator for planning / RL

The current system does not yet fully support the stronger claim:

- we built a causally credible patient simulator for counterfactual intervention evaluation

That stronger claim would require additional causal structure, validation, or both.

---

## 4. Landscape Structure

The literature separates into three partly overlapping families.

### Family A: Transformer world models / simulators for RL

These models are closest to CARE-Sim architecturally.

Typical goal:

- learn state transition and reward dynamics
- generate synthetic trajectories
- plan or train RL agents in the learned environment

Strengths:

- strong sequence modeling
- natural fit for non-Markov history dependence
- often better long-horizon behavior than one-step feedforward baselines

Weaknesses:

- usually weak on causal validity
- often developed in robotics or control rather than healthcare

### Family B: Causal / counterfactual sequence models under time-varying treatment

These are closest to the thesis's causal ambition.

Typical goal:

- estimate outcomes under hypothetical treatment sequences
- correct for time-varying confounding
- support individualized treatment reasoning

Strengths:

- explicitly targets observational bias and treatment assignment bias
- more aligned with "what would happen under a different intervention sequence?"

Weaknesses:

- often designed for counterfactual prediction, not full RL environments
- may not provide a practical gym-like simulator interface
- often evaluate on semi-synthetic benchmarks rather than direct policy control

### Family C: Clinical transformers that are relevant but not true simulators

These include:

- healthcare forecasting models
- readmission prediction transformers
- medical sequence foundation models
- policy-only offline RL transformers

They are useful as architectural references, but they are not direct replacements for CARE-Sim.

---

## 5. Most Relevant Transformer-Side Models

### 5.1 Trajectory Transformer

Why it matters:

- foundational transformer world-model paper for offline RL
- treats trajectories as a sequence modeling problem
- uses transformers as long-horizon dynamics models and planners

Relevance to CARE-Sim:

- very strong conceptual ancestor of the current Step 14 design
- CARE-Sim already explicitly follows the same basic idea
- difference: CARE-Sim uses continuous linear embeddings instead of discretized tokenization

Takeaway:

- this strongly validates the architectural direction
- it is not clinical and not causal, but it is the correct family of reference

### 5.2 Bootstrapped Transformer (BooT / BooTORL)

Why it matters:

- extends transformer-based offline RL by self-generating more data
- public code exists

Relevance to CARE-Sim:

- useful for thinking about synthetic trajectory augmentation
- less important than Trajectory Transformer as a core comparator

Takeaway:

- relevant as a sequence-modeling RL reference
- not healthcare-specific and not causal

### 5.3 OTTO / World Transformers

Why it matters:

- transformer ensemble world models for offline RL
- explicitly addresses long-horizon synthetic trajectory generation
- adds uncertainty-based evaluation / correction of generated trajectories

Relevance to CARE-Sim:

- highly relevant on the world-model reliability side
- close to CARE-Sim's ensemble logic
- offers a more serious uncertainty-aware augmentation philosophy than the current Step 16 planner penalty

Takeaway:

- not clinical, but one of the most useful methodological comparators for improving simulator reliability

### 5.4 Causal Transformer for Estimating Counterfactual Outcomes

Why it matters:

- one of the central transformer papers for longitudinal counterfactual prediction under time-varying treatments
- public code exists
- compares against CRN, RMSN, G-Net, MSMs

Relevance to CARE-Sim:

- this is the most important causal-transformer comparator
- much closer to the causal ambition of CARE-AI than standard world-model RL papers
- less a full RL environment and more a counterfactual prediction engine

Takeaway:

- essential baseline for thesis positioning
- if CARE-Sim is defended causally, it should be discussed relative to this line of work

### 5.5 G-Transformer

Why it matters:

- probably the closest literature match to the intended thesis concept
- transformer-based
- supports g-computation
- designed for dynamic treatment regimes
- applied to semi-synthetic and MIMIC-based medical settings

Relevance to CARE-Sim:

- more causally serious than current CARE-Sim
- closer to intervention-conditioned counterfactual simulation
- still not exactly the same as a fully interactive RL environment, but conceptually very close

Takeaway:

- likely the single most important "closest external comparator" to cite against the current simulator vision

### 5.6 Disentangled Causal Transformer (DCT)

Why it matters:

- recent transformer-based approach for longitudinal causal inference
- explicitly tries to disentangle confounders, instruments, and outcome factors inside attention

Relevance to CARE-Sim:

- directly relevant to the causal weakness of predictive simulators
- interesting for future upgrades if a stronger causal representation is needed

Takeaway:

- promising direction
- too new to treat as the stable benchmark, but highly relevant as future-facing research

### 5.7 Continuous-Time Decision Transformer (CTDT)

Why it matters:

- transformer-based healthcare offline RL for irregularly timed decisions
- directly motivated by medical settings

Relevance to CARE-Sim:

- policy model, not a full dynamics simulator
- useful for thinking about irregular time and decision timing

Takeaway:

- important healthcare transformer RL reference
- not a direct simulator substitute

### 5.8 Foresight

Why it matters:

- generative transformer for patient timelines
- supports future event forecasting and multi-step generation

Relevance to CARE-Sim:

- sequence-generative healthcare transformer
- but focused on concept timelines and forecasting rather than action-conditioned physiological simulation

Takeaway:

- good evidence that generative transformer modeling of patient trajectories is legitimate
- not a true action-conditioned ICU simulator

### 5.9 TrajGPT / TimelyGPT / related healthcare sequence transformers

Why they matter:

- show continuing momentum in transformer modeling of healthcare time series
- provide ideas for irregular sampling, representation learning, and temporal structure

Relevance to CARE-Sim:

- useful architectural references
- not full intervention simulators

Takeaway:

- good peripheral literature, not primary comparators

---

## 6. Models Outside the Transformer Core That Still Matter

These are worth tracking because the strongest challenge to CARE-Sim may eventually come from outside the transformer family.

### 6.1 medDreamer

Why it matters:

- explicitly medical
- world model plus RL framing
- directly aligned with treatment recommendation from EHR trajectories

Relevance:

- probably one of the strongest broader healthcare comparators
- especially important because it treats the world model and downstream policy jointly

Caveat:

- not enough stable implementation evidence was found in this pass to treat it as an immediately reusable benchmark

### 6.2 Medical World Model (MeWM)

Why it matters:

- strong 2025 "medical world model" paper with public code
- genuine simulator-based treatment planning

Relevance:

- important proof that "medical world model" is now an active research direction

Caveat:

- imaging-based tumor progression setting
- much less directly relevant than tabular ICU trajectory simulation

### 6.3 Mamba-CDSP

Why it matters:

- counterfactual prediction over time using state-space models
- explicitly argues that transformer-based causal methods suffer from horizon scaling and confounding accumulation problems

Relevance:

- very important if the search broadens beyond transformers
- conceptually attacks one of CARE-Sim's likely weak points: long-horizon reliability under sequential counterfactual prediction

Takeaway:

- if CARE-Sim needs a next-generation alternative, state-space causal sequence models may be the strongest candidate family

---

## 7. Comparison: CARE-Sim vs the Literature

### 7.1 Where CARE-Sim is strong

CARE-Sim compares well on:

- practical implementation quality
- clean integration into a real ICU pipeline
- action-conditioned next-state modeling
- explicit simulator environment interface
- ensemble uncertainty
- direct use for planning and DDQN

In other words, CARE-Sim is ahead of many healthcare transformer papers in terms of being an actual usable control stack, not just a modeling paper.

### 7.2 Where CARE-Sim is weaker

CARE-Sim is weaker on:

- explicit adjustment for time-varying confounding
- causal interpretation of action effects
- counterfactual validity under distribution shift
- formal causal evaluation beyond predictive fidelity

This is the main gap relative to Causal Transformer, G-Transformer, and related counterfactual sequence models.

### 7.3 The most accurate characterization

Current CARE-Sim is best described as:

- a predictive transformer world model with some causal prior infrastructure available in code

It is not best described as:

- a fully causally identified simulator

### 7.4 Important subtlety

The code already contains a causal extension path:

- an optional FCI-masked action residual layer

That means the project is not causally naive in design. But the trained model currently used in Step 14-16 did not activate that mechanism.

This matters for thesis framing:

- the repository contains causal simulator ideas
- the reported simulator results currently come from the unconstrained baseline

---

## 8. What Is Defensible Right Now

### Defensible now

- CARE-Sim is a competent transformer world model for short-horizon ICU trajectory simulation.
- CARE-Sim is good enough to support planner- and DDQN-based control experiments.
- The Step 14-16 pipeline is a valid implementation of model-based control in a learned clinical environment.

### Not fully defensible yet

- CARE-Sim gives causally trustworthy counterfactual rollouts under arbitrary interventions.
- The current simulator can be treated as patient "ground truth" in a strong causal sense.

That second statement is too strong unless stronger causal structure or validation is added.

---

## 9. Recommended Next Steps

If the goal is to improve the current simulator rather than replace it, the highest-value steps are:

### 9.1 Train and evaluate the causal-mask CARE-Sim variant

This is the most immediate next experiment because the code path already exists.

Question to answer:

- does the FCI-masked action residual improve counterfactual plausibility without materially harming predictive performance?

### 9.2 Add a more causal evaluation layer

Current evaluation is mostly:

- one-step predictive fidelity
- short rollout drift

That should be extended with tests that ask:

- do intervention effects behave in clinically and causally plausible directions?
- do counterfactual rollouts preserve known treatment-response structure?

### 9.3 Treat uncertainty more seriously in control

Current planner uses:

- an uncertainty penalty in rollout scoring

The broader model-based RL literature suggests stronger approaches:

- uncertainty-aware truncation
- uncertainty-based rollout correction
- more conservative use of long simulated trajectories

### 9.4 Be precise in thesis wording

The safest language is:

- CARE-Sim is a short-horizon predictive world model with causal priors / causal constraints available

Avoid claiming:

- fully causal simulator

unless stronger evidence is added.

### 9.5 If a more ambitious redesign is needed

The most promising research upgrade directions appear to be:

- causal-transformer style counterfactual sequence models
- G-computation-based transformer simulation
- state-space causal models such as Mamba-CDSP

---

## 10. Final Assessment

The current CARE-Sim direction is scientifically reasonable and technically credible.

The external literature does not suggest that the project made a poor architectural choice by using a transformer world model. On the contrary, the transformer direction is well supported by both the general world-model literature and several recent clinical sequence-modeling papers.

The main strategic issue is not whether CARE-Sim should be transformer-based.

The main strategic issue is whether the current simulator is sufficiently causal for the thesis claims attached to it.

My conclusion is:

- for short-horizon simulator-based planning and RL, the current CARE-Sim is defensible
- for stronger counterfactual / causal simulator claims, the current unconstrained model is not yet enough
- the most logical immediate upgrade is to activate and test the causal constraint path already present in the codebase

---

## 11. External References

The following references were most useful in this review.

### Core world-model / transformer RL references

- Trajectory Transformer project page: <https://trajectory-transformer.github.io/>
- Bootstrapped Transformer for Offline RL code: <https://github.com/microsoft/BooTORL>
- OTTO / Offline Trajectory Optimization for Offline Reinforcement Learning: <https://renzhaochun.github.io/assets/pdf/2404.10393v2.pdf>

### Core causal / counterfactual sequence references

- Causal Transformer for Estimating Counterfactual Outcomes: <https://proceedings.mlr.press/v162/melnychuk22a>
- Causal Transformer code: <https://github.com/Valentyn1997/CausalTransformer>
- G-Transformer: <https://proceedings.mlr.press/v252/xiong24a.html>
- G-Transformer open access mirror: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12113242/>
- Deep Longitudinal TMLE with Temporal-Difference Heterogeneous Transformer: <https://arxiv.org/abs/2404.04399>
- Disentangled Causal Transformer (under review): <https://openreview.net/pdf/801ef25b71c5177cf8e141209982a12a8a126474.pdf>

### Healthcare transformer references

- Continuous-Time Decision Transformer for Healthcare Applications: <https://proceedings.mlr.press/v206/zhang23i/zhang23i.pdf>
- Foresight: GPT for Modelling of Patient Timelines using EHRs: <https://arxiv.org/abs/2212.08072>
- TrajGPT: <https://openreview.net/pdf?id=pjRHvP6WcU>
- TimelyGPT: <https://link.springer.com/article/10.1007/s13755-025-00384-0>

### Broader / adjacent medical world-model references

- Medical World Model paper / code: <https://github.com/scott-yjyang/MeWM>
- Mamba-CDSP / counterfactual prediction over time with state-space models: <https://haoxuanli-pku.github.io/papers/ICLR%2025%20-%20Effective%20and%20Efficient%20Time-Varying%20Counterfactual%20Prediction%20with%20State-Space%20Models.pdf>

