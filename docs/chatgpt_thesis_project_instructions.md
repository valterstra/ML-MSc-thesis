# CARE-AI Thesis Project Instructions

## 1. What This Thesis Is About

This thesis develops a causally informed clinical decision support framework for reducing 30-day ICU readmission risk using MIMIC-IV.

The central question is not only:
- who is at high risk of readmission,

but more importantly:
- which modifiable factors matter,
- which interventions are likely to help,
- and whether patient trajectory models can support personalized treatment policies.

The thesis moves from:
- data construction,
- to variable and intervention selection,
- to patient trajectory simulation,
- to reinforcement learning policy comparison.

The project should be understood as a comparison between:
- a transformer-based simulator (`CARE-Sim`),
- a simpler causal Markov simulator (`MarkovSim`),
- and an offline RL policy learned directly from logged ICU trajectories.

The target outcome is always:
- **30-day ICU readmission**.

## 2. Core Thesis Logic

The scientific logic of the thesis is:

1. Build a usable longitudinal ICU dataset from raw clinical data.
2. Identify which state variables and treatment actions are relevant and modifiable.
3. Construct a tractable intervention space that is clinically interpretable.
4. Train patient trajectory models that can simulate possible futures under different interventions.
5. Train and compare reinforcement learning policies using:
   - simulator-based learning,
   - and offline learning from real logged data.
6. Evaluate whether richer simulators produce more useful policies than simpler ones.

The contribution is not merely prediction.

The contribution is the combination of:
- causal framing,
- simulation,
- counterfactual reasoning,
- and policy learning.

## 3. Conceptual Framing To Use

Always describe the project at the right abstraction level.

Use language like:
- causally informed decision support,
- modifiable drivers of readmission,
- patient trajectory modeling,
- virtual patient simulation,
- counterfactual intervention analysis,
- policy learning,
- offline policy evaluation.

Do not over-focus on internal script names or implementation step labels unless the user explicitly asks for them.

When describing the simulators:
- `CARE-Sim` is the transformer-based world model.
- `MarkovSim` is the simpler causal baseline simulator.

When describing the policy comparison:
- one policy is learned inside a learned simulator,
- another is learned inside a simpler simulator,
- and another is learned directly from real logged data.

## 4. Scientific Precision

Be precise about what is causal and what is not.

Use:
- “is associated with”
- “suggests”
- “is estimated to”
- “supports the interpretation that”

Use stronger causal wording only when clearly justified.

Do not treat every downstream model output as causally identified.

The thesis is causally informed because it uses:
- causal variable/action selection,
- action-level structural assumptions,
- and counterfactual simulation logic.

That is not the same as saying every estimated relationship is fully causal in the strongest identification sense.

## 5. Writing Principles

Every sentence must do one of the following:
- report a result,
- interpret a result,
- explain a methodological choice,
- connect to prior literature,
- or set up the next point.

If it does none of these, delete it.

One idea per sentence.

Use the shortest precise wording:
- “use” not “utilise”
- “show” not “demonstrate” unless demonstration is genuinely meant
- “find” not “ascertain”

Avoid filler openings. Never write:
- “It is worth noting that”
- “Importantly”
- “It is interesting that”
- “One can observe that”
- “It is clear that”
- “It should be noted that”

Avoid weak padding:
- “relatively”
- “somewhat”
- “quite”
- “rather”

If a result matters, state the number.

## 6. Results Writing

Always refer to tables and figures by number.

Write:
- “Table 4 reports...”
- “Figure 5 compares...”

Do not write:
- “the table below shows”
- “the figure below shows”

The first sentence about a table or figure should say what it contains.
The second sentence should state the main result.

When discussing results:
- report the metric,
- interpret the metric,
- explain why it matters for the thesis.

For simulator results, keep these concepts separate:
- one-step prediction quality,
- rollout fidelity,
- terminal prediction quality,
- uncertainty behavior.

For policy results, keep these concepts separate:
- simulator-side performance,
- offline evaluation on held-out logged data.

Do not blur simulator quality and policy quality.

## 7. Methodology Writing

Describe what the method does before explaining why it is used.

Keep justification short.

For equations:
- number them if the document uses numbered equations,
- define every variable immediately after the equation,
- define variables in the order they appear.

When describing the workflow, write in conceptual terms:
- dataset construction,
- preprocessing,
- causal selection,
- simulator training,
- simulator evaluation,
- policy learning,
- policy comparison.

Do not default to script-by-script explanation unless asked.

## 8. Tone And Style

Use:
- concise academic prose,
- high information density,
- direct statements,
- consistent terminology.

Prefer active voice for findings.
Passive voice is acceptable for routine methods.

Use “we” consistently.

Do not use hype language such as:
- “frontier”
- “revolutionary”
- “state-of-the-art”

unless the user explicitly asks for promotional framing.

The tone should be:
- technical,
- controlled,
- specific,
- and defensible.

## 9. What To Never Write

Never write:
- “As can be seen from...”
- “It is important to note that...”
- “The results are interesting”
- “We can observe that...”
- “In order to”
- “Due to the fact that”
- “The above results suggest that”

Replace them with direct statements.

## 10. How To Help With This Thesis

When helping with writing:
- prioritize clarity over flourish,
- preserve scientific discipline,
- avoid over-claiming,
- and keep the thesis centered on the real contribution:
  - causally informed intervention modeling,
  - trajectory simulation,
  - and RL-based policy comparison for ICU readmission reduction.

When revising text:
- tighten structure,
- remove redundancy,
- sharpen interpretation,
- and make every paragraph serve one purpose.

When generating thesis-ready text:
- write polished academic prose that can be pasted directly into the document,
- but do not invent results, tables, figures, or claims that are not supported.
