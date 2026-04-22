
Focused literature scan: causal-transformer-style models for sequential decision-making with structural masking

Context
You are modeling ICU readmission and clinical trajectories with a small GPT-style transformer world model over ICU time steps. At each time step t you have:
- state_t in R^9
- action_t in R^5

Current structure:
1. Token construction:
   e_t = W_s state_t + W_a action_t + pos_embed(t) + time_embed(t)
2. A causal transformer over the sequence:
   h_1, ..., h_T = Transformer(e_1, ..., e_T)
3. Base next-state prediction:
   base_next_t = head_state(h_t)
4. A separately learned masked action residual:
   resid_t = (W_causal ⊙ M) action_t
5. Final next-state prediction:
   next_t = base_next_t + resid_t

Where M is a fixed binary action-to-state mask from prior causal discovery or domain structure, W_causal is learned, forbidden edges are permanently zeroed out, the residual is applied directly from action_t to next-state prediction, and the transformer still learns temporal dynamics normally.

Executive conclusion
The overall architecture looks novel-ish rather than standard. The transformer-over-trajectories part is common. The idea of separating base dynamics from an action-effect component is also common. What looks uncommon is the exact combination of:
- an autoregressive transformer world model,
- plus a separate hard-masked action-to-state residual,
- where the mask comes from prior causal discovery or domain knowledge,
- and forbidden action-state edges are permanently zeroed out.

In this scan, I did not find a close match that clearly implements the full pattern in one model. I did find several neighboring lines that are highly relevant:
- models that do causal discovery first and then inject a hard mask into a transition model,
- transformers that inject a known DAG as a hard attention mask,
- healthcare counterfactual transformers over treatment trajectories,
- world models that learn sparse or hard causal interaction structure,
- and decomposed transition models with explicit action-effect modules.

Section A. Most similar papers with code

1) Counterfactual Data Augmentation using Locally Factored Dynamics
Year / venue: 2020, NeurIPS 2020
Summary:
This paper starts from the idea that many dynamical systems can be described by locally sparse mechanisms. It defines local causal models and introduces a mask-based mechanism that determines which components of the next state depend on which components of the current state and action. That mask is treated explicitly as the adjacency matrix of a local causal graph. The model then uses this structure to construct causally valid counterfactual transitions by stitching together independent subcomponents from observed transitions.
How it is similar to your setup:
The closest part is the explicit binary mask M(s,a) with shape (|S|+|A|) x |S| that encodes which state and action components affect which next-state components. That is very close in spirit to your structural action-to-state pathway with hard zeros.
How it differs:
Their mask is typically learned and can be state/action dependent rather than injected as a fixed prior graph. Their transformer variant is not the same as a GPT-style autoregressive world model over long trajectories. They do not add a separate linear masked residual on top of a transformer baseline in the exact way you do.
Uses:
- transformer: partial
- causal graph prior: no
- hard mask: yes
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 4.5 / 5

2) DAG-aware Transformer for Causal Effect Estimation
Year / venue: 2024, workshop / arXiv line
Summary:
This work takes a known causal DAG and injects it directly into self-attention by using a DAG-based attention mask. Forbidden edges are assigned negative infinity before the softmax, which yields exact zeros in attention. The transformer is then used as a flexible backbone for several causal effect estimators.
How it is similar:
This is the cleanest example I found of “inject a known causal graph as a hard mask into a transformer.” The mask is explicit, binary, and structurally meaningful.
How it differs:
It is mainly for causal effect estimation rather than world modeling or next-state prediction. The mask constrains feature-to-feature attention, not a dedicated action-to-state residual channel inside a transition model.
Uses:
- transformer: yes
- causal graph prior: yes
- hard mask: yes
- action/intervention nodes: yes
- next-state / transition prediction: no
- healthcare data: indirect / partially relevant
Similarity score: 4.0 / 5

3) Learning Causal Dynamics Models in Object-Oriented Environments
Year / venue: 2024, ICML 2024
Summary:
This paper sits in the causal dynamics model literature and focuses on learning causal structure and transitions in object-oriented MDPs. It proposes an object-oriented causal dynamics model that shares causal mechanisms within object classes and improves generalization across environments with variable numbers of objects.
How it is similar:
It is a transition model with explicit causal structure between state variables and actions. The central idea is to bind structure to a parameterized dynamics model rather than rely on a fully unconstrained black box.
How it differs:
It is not a transformer trajectory model, not in healthcare, and not built around a separate additive masked action residual on top of a transformer.
Uses:
- transformer: no
- causal graph prior: partial
- hard mask: partial / unclear in your exact sense
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 3.5 / 5

4) Causal Dynamics Learning for Task-Independent State Abstraction
Year / venue: 2022, ICML 2022
Summary:
This paper argues that dense dynamics models often pick up unnecessary dependencies between state dimensions and action, which hurts generalization. It introduces a causal dynamics model to remove such unnecessary dependencies and then uses that structure to derive better state abstractions for downstream RL.
How it is similar:
The motivation is very close to yours: remove spurious action-to-state dependencies in the transition model and improve generalization by enforcing structure.
How it differs:
It is not transformer-based, and the structural component is part of a broader causal abstraction framework rather than an explicit additive masked residual attached to a sequential transformer.
Uses:
- transformer: no
- causal graph prior: partial / no
- hard mask: unclear
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 3.0 / 5

5) MoCoDA: Model-based Counterfactual Data Augmentation
Year / venue: 2022, NeurIPS 2022
Summary:
MoCoDA extends the locally factored dynamics idea to model-based counterfactual data augmentation for offline RL. It uses structured, locally sparse dynamics models to generate controlled augmented support over state-action space and then produces counterfactual transitions that help offline RL agents generalize better out of distribution.
How it is similar:
It operationalizes causal structure in the transition model as an explicit factorization or masking idea. That is close to your use of structural priors in a simulator.
How it differs:
It is not a transformer world model over trajectories and is not specifically about a separate hard-masked action residual layered on top of a temporal sequence model.
Uses:
- transformer: no
- causal graph prior: partial
- hard mask: partial
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 2.5 / 5

6) Causal Transformer for Estimating Counterfactual Outcomes
Year / venue: 2022, arXiv / healthcare counterfactual sequence modeling line
Summary:
This is a transformer-based model for time-varying treatment settings, designed to estimate counterfactual outcomes over time under different intervention sequences. In practice it is one of the closest sequence-modeling analogues to an ICU trajectory simulator because it explicitly models longitudinal treatment and outcome dynamics.
How it is similar:
It uses a transformer over treatment trajectories and is directly focused on counterfactual “what-if” reasoning over time, which is very close to the overall clinical use case.
How it differs:
It does not appear to inject a prior causal graph as a hard structural mask, and it does not use the exact additive action-to-next-state residual pattern that your model uses.
Uses:
- transformer: yes
- causal graph prior: no
- hard mask: no
- action/intervention nodes: yes
- next-state / transition prediction: partial
- healthcare data: yes / highly relevant
Similarity score: 3.0 / 5

7) OP3 / Entity Abstraction in Visual Model-Based Reinforcement Learning
Year / venue: 2019-2020 line
Summary:
OP3 is an entity-centric latent world model for visual RL settings. A relevant design pattern in this line is to decompose transition dynamics into separate components such as self-dynamics, action effects, and interactions across entities, and then combine them into a next-state prediction.
How it is similar:
The decomposition “base dynamics plus action effect plus interactions” is conceptually similar to your base_next plus action residual split.
How it differs:
It is not transformer-based, not graph-masked from prior causal discovery, and not in healthcare.
Uses:
- transformer: no
- causal graph prior: no
- hard mask: no
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 2.0 / 5

Section B. Most relevant papers without clearly available code

1) Offline Reinforcement Learning with Causal Structured World Models (FOCUS)
Year / venue: 2022, arXiv
Summary:
FOCUS is the closest conceptual match on the structural side. It first learns causal structure from offline data, using a discovery stage that produces a binary causal structure mask matrix G. It then injects that mask into the transition model so that forbidden inputs are zeroed out for each next-state dimension.
How it is similar:
This is the clearest hit on “causal discovery first, then inject a hard mask into the dynamics network.” In particular, it masks state and action inputs per next-state dimension, which is extremely close to your “action/state to next-state under hard structural constraints” logic.
How it differs:
It is not transformer-based. The mask is applied to the full input of per-dimension predictors rather than as a separate additive masked action residual on top of a temporal transformer.
Uses:
- transformer: no
- causal graph prior: yes
- hard mask: yes
- action/intervention nodes: yes
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 4.5 / 5

2) SPARTAN: A Sparse Transformer World Model Attending to What Matters
Year / venue: 2025, NeurIPS 2025
Summary:
SPARTAN is a transformer world model that explicitly tries to learn sparse local causal graphs among tokens or entities using hard attention and sparsity regularization. It treats attention as information flow in a graph and encourages the model to eliminate spurious edges rather than relying on dense attention.
How it is similar:
It is very close at the level of “transformer world model plus masked causal structure in the transition process.”
How it differs:
The graph is learned rather than injected from a prior discovered clinical graph. It is also more object-centric than vector state/action modeling.
Uses:
- transformer: yes
- causal graph prior: no
- hard mask: yes
- action/intervention nodes: partial
- next-state / transition prediction: yes
- healthcare data: no
Similarity score: 4.0 / 5

3) CAIFormer: A Causal Informed Transformer for Multivariate Time Series Forecasting
Year / venue: 2026 submission line
Summary:
This model first builds an SCM from observational data and then partitions historical information for each target into endogenous, direct causal, collider-causal, and spurious segments. It uses transformer-style blocks over these segments and excludes spurious information from the final prediction.
How it is similar:
It follows the same pipeline idea of “causal structure first, then inject that structure into the network architecture.” It also composes the final prediction from structured subcomponents rather than relying on unrestricted all-to-all modeling.
How it differs:
It is forecasting rather than state-action transition modeling, and there is no explicit action-to-state residual layer.
Uses:
- transformer: yes
- causal graph prior: yes
- hard mask: partial
- action/intervention nodes: no
- next-state / transition prediction: no
- healthcare data: no
Similarity score: 3.0 / 5

4) Invariant Action Effect Model for Reinforcement Learning (IAEM)
Year / venue: 2022, AAAI 2022
Summary:
IAEM formulates action effect as a residual in representation space between neighboring states and learns an invariance principle for that effect. The key value for your use case is that it gives a more principled motivation for isolating action effect as a separate module.
How it is similar:
It is directly relevant to the “base dynamics plus action residual” design pattern.
How it differs:
It is not a transformer, not graph-masked, and not based on prior causal discovery.
Uses:
- transformer: no
- causal graph prior: no
- hard mask: no
- action/intervention nodes: yes
- next-state / transition prediction: partial
- healthcare data: no
Similarity score: 3.0 / 5

5) SCOT: Improved Temporal Counterfactual Estimation with Self-Supervised Learning
Year / venue: 2024 submission line
Summary:
SCOT is a self-supervised counterfactual transformer for treatment-outcome sequences. It combines temporal attention with feature-wise attention and uses a component-level contrastive objective suited to longitudinal interventions.
How it is similar:
It is highly relevant on the healthcare sequence-modeling side because it is a transformer over treatment trajectories with counterfactual intent.
How it differs:
It does not inject a prior structural graph as a hard mask over action-to-state effects.
Uses:
- transformer: yes
- causal graph prior: no
- hard mask: no
- action/intervention nodes: yes
- next-state / transition prediction: partial
- healthcare data: yes / relevant
Similarity score: 2.0 / 5

Section C. Conceptual buckets

Bucket 1. Causal structure first, then hard-mask a transition model
What this bucket does:
These papers learn or specify causal structure and then inject it into a transition model so that forbidden dependencies are removed by construction.
How close it is:
Very close to your structural component.
Contains upgrade ideas:
Yes. This is the best bucket for replacing an ad hoc residual with a more principled masked transition factorization.

Bucket 2. Transformer plus explicit DAG prior
What this bucket does:
These models take a known DAG and use it to constrain transformer attention or block structure.
How close it is:
Close to your idea of injecting prior structure into a transformer, but usually not in an MDP transition setting.
Contains upgrade ideas:
Yes. A very plausible upgrade is to encode your prior graph inside attention rather than only in a separate residual path.

Bucket 3. Healthcare counterfactual transformers over treatment trajectories
What this bucket does:
These models use transformers to model longitudinal treatment and outcome trajectories under counterfactual reasoning.
How close it is:
Close to your clinical use case and sequence backbone, but usually not close to your hard structural mask.
Contains upgrade ideas:
Yes. These papers may offer better confounding handling, balancing, or treatment-effect estimation machinery.

Bucket 4. Sparse or hard-attention transformer world models
What this bucket does:
These models learn sparse or hard interaction structure inside a transformer world model.
How close it is:
Very close on the world-model side, though the graph is often learned rather than supplied.
Contains upgrade ideas:
Yes. Especially useful if your fixed causal graph is noisy or incomplete and you want controlled flexibility.

Bucket 5. Decomposed transitions with explicit action-effect modules
What this bucket does:
These papers separate base dynamics from action effects, sometimes with interaction modules layered on top.
How close it is:
Moderately close because it matches your base_next plus action residual decomposition.
Contains upgrade ideas:
Yes. Good source of more principled action-effect parameterizations.

Section D. Direct answer about novelty

Bottom line:
Your architecture is not standard, but it is also not coming out of nowhere. It looks like a custom combination of existing ideas.

What is common:
- transformer sequence modeling over trajectories,
- next-step or multi-step world modeling,
- separating action effect from baseline dynamics,
- using causal or sparse structure to improve transition models.

What looks custom or ad hoc:
- the exact additive form next_t = base_next_t + (W_causal ⊙ M) action_t,
- where M is externally estimated beforehand,
- where action-to-state forbidden edges are hard-zeroed permanently,
- while the transformer remains otherwise unrestricted and handles temporal modeling separately.

Did I find a close prior example of “transformer + hard masked action residual from prior causal discovery”?
No clear direct match in this scan.
The closest partial matches are:
- FOCUS for “causal discovery first, then hard-mask the transition model,”
- DAG-aware Transformer for “inject a known DAG as a hard mask into a transformer,”
- CoDA / locally factored dynamics for “explicit binary action-state to next-state masking,”
- and Causal Transformer / SCOT for “healthcare treatment trajectory transformers.”
But I did not find a paper that obviously combines all of those pieces in the same exact form.

Section E. Upgrade ideas

1) Replace the additive linear residual with a FOCUS-style masked per-variable transition head
Why it may be better:
More principled SCM-like factorization. Each next-state dimension only sees allowed parents by construction.
Difficulty:
Medium
Code exists:
Partial / conceptual reference exists

2) Put the graph prior inside attention, not only in the residual
Why it may be better:
This would make structural constraints end-to-end throughout the transformer rather than only in a final shortcut path.
Difficulty:
High
Code exists:
Yes, related code exists in DAG-aware Transformer

3) Keep the hard mask, but make the action effect state-dependent
Idea:
x_{t+1} = f_theta(h_t) + g_phi(h_t) (W ⊙ M) a_t
Why it may be better:
Treatment effects in ICU are usually context-dependent. This keeps your mask while allowing effect magnitude to vary by patient state.
Difficulty:
Low to medium
Code exists:
Not directly as a matched repo, but easy to implement

4) Use a prior-plus-slack graph instead of a fully fixed graph
Why it may be better:
Pre-estimated graphs are often wrong in some places. A sparse correction mechanism could preserve interpretability while improving fit.
Difficulty:
Medium to high
Code exists:
Partial inspiration from sparse graph-learning transformer world models

5) Let the mask be state-dependent but anchored to the prior graph
Why it may be better:
Some interventions only affect certain variables in some regimes. A dynamic mask is more realistic clinically.
Difficulty:
High
Code exists:
Related ideas exist in locally factored dynamics / CoDA

6) Use an explicit neural SCM factorization
Why it may be better:
Instead of one head plus one residual, define one function per next-state variable using only its allowed parents.
Difficulty:
Medium
Code exists:
Not exact, but strongly supported by FOCUS-style design

7) Borrow confounding-robust training ideas from healthcare counterfactual transformers
Why it may be better:
Clinical sequence data is confounded. Structural masking alone does not solve that. This is likely one of the most meaningful upgrades for real ICU data.
Difficulty:
High
Code exists:
Yes, related healthcare counterfactual transformer repos exist

8) Factor the state into semantically meaningful modules and assign action effects per module
Why it may be better:
This can improve interpretability and make the structural prior cleaner and more clinically plausible.
Difficulty:
Medium
Code exists:
Related object-centric world model code exists

9) Move to a continuous-time structured model
Why it may be better:
ICU data is often irregularly sampled. A continuous-time graph-based model may be more principled than discrete step embeddings.
Difficulty:
High
Code exists:
Varies by specific implementation

10) Improve the causal discovery stage itself and propagate uncertainty into the mask
Why it may be better:
If the mask is wrong, the whole architecture inherits that error. Using edge confidence or posterior uncertainty would make the system more robust.
Difficulty:
Medium to high
Code exists:
Varies by discovery method

Practical takeaways
If you want the closest literature anchor for the exact structural logic, use FOCUS.
If you want the closest transformer anchor for injecting a prior graph, use DAG-aware Transformer.
If you want the closest healthcare transformer anchor, use Causal Transformer for counterfactual outcomes.
If you want the closest “masked causal transformer world model” neighbor, use SPARTAN.
If you want the closest explicit action-state to next-state masking formulation, use locally factored dynamics / CoDA.

Recommended positioning sentence
A fair claim would be:
“Our model combines a standard autoregressive transformer backbone for temporal trajectory modeling with a hard-masked action-to-state residual pathway derived from prior causal structure. While related ideas appear separately in causal transition models, DAG-masked transformers, and healthcare counterfactual transformers, we did not identify a close prior architecture that combines these elements in the same form.”

End of report.
