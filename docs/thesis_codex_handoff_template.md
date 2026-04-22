# Thesis ↔ Codex handoff template

## Purpose
This document is the bridge between the empirical/code side of the thesis and the writing side. Codex should use it to extract the right information from the codebase and package that information in a way that can be directly turned into thesis text.

## Thesis project overview

### Working title
CARE-AI

### Subtitle
How causal machine learning and reinforcement learning can learn patient specific intervention strategies to reduce hospital readmissions

### Core research goal
Explain in 2 to 4 sentences what the thesis is trying to do at the highest level.

### Main empirical question
State the main question the codebase is trying to answer.

### Secondary questions
List the main subquestions.

### Main contribution
What is new or useful in this thesis relative to a standard prediction paper?

## Thesis structure

### Chapter 1. Introduction
What Codex should provide:
- Problem motivation
- Why the problem matters in practice
- Why prediction alone is insufficient
- Why causal ML and RL are relevant
- One concise description of the proposed framework

### Chapter 2. Background / theory
What Codex should provide:
- Definitions of the main technical concepts actually used in the thesis
- Which concepts are central versus peripheral
- Any assumptions that matter for the empirical setup
- Short explanations of methods, but only at the level needed for the thesis

### Chapter 3. Data
What Codex should provide:
- Exact dataset(s) used
- Source and access conditions
- Sample construction pipeline
- Inclusion and exclusion criteria
- Time granularity and prediction horizon
- Outcome variable definition
- State variables and action variables
- Missing data handling
- Final sample size after filtering

### Chapter 4. Method
What Codex should provide:
- Full modeling pipeline in ordered steps
- Causal components used
- RL components used
- Simulator or transition model details
- Training procedure
- Validation procedure
- Baselines used for comparison
- Key hyperparameters that matter conceptually
- Why each modeling choice was made

### Chapter 5. Results
What Codex should provide:
- Main quantitative results
- Comparison across models
- Best performing specification
- Main patterns worth interpreting
- Robustness checks
- Negative or null results that should be acknowledged
- Any figures or tables that should appear in the thesis

### Chapter 6. Discussion
What Codex should provide:
- How results should be interpreted
- What worked and what did not
- Likely limitations in data, identification, and modeling
- External validity concerns
- Clinical relevance and realism
- What future work should do next

### Chapter 7. Conclusion
What Codex should provide:
- Direct answer to the research question
- Main contribution in plain language
- Short statement on implications

## Required handoff format for each chapter or section
Codex should always return information under the following headings.

### 1. Section objective
What this section is supposed to accomplish in the thesis.

### 2. Facts from the codebase
Concrete facts only. No polished prose yet.

### 3. Interpretation
What these facts mean and why they matter.

### 4. Evidence
Relevant file names, scripts, outputs, tables, logs, config files, or notebook cells.

### 5. Caveats
Anything uncertain, incomplete, or not fully validated.

### 6. Suggested thesis text ingredients
A list of points that should definitely appear in the written thesis.

## Global inventory Codex should prepare

### A. Project map
- Main scripts
- Config files
- Data processing scripts
- Training scripts
- Evaluation scripts
- Plotting scripts
- Output folders
- Which files are actually used in the final pipeline

### B. Variable dictionary
For every key variable used in the thesis:
- Variable name in code
- Human readable description
- Type
- Unit if relevant
- Role in thesis: outcome, state, action, control, intermediate object

### C. Model inventory
For each model:
- Model name
- Purpose
- Inputs
- Outputs
- Training target
- Where implemented
- Whether included in final thesis or only exploratory

### D. Experiment inventory
For each major experiment:
- Experiment name
- Research purpose
- Exact comparison being made
- Metrics
- Main result
- Whether it belongs in final thesis

### E. Figure and table inventory
For each candidate figure/table:
- Filename or generation script
- What it shows
- Why it matters
- Which chapter it belongs to
- Whether it is final quality or still provisional

## Prompt template for Codex
Use the following style when asking Codex for material.

"Read the thesis handoff document and produce a complete handoff for [chapter/section]. Use the exact headings in the handoff format. Be maximally concrete. Pull information from the actual codebase, configs, outputs, and logs. Separate verified facts from interpretation. Flag every uncertainty clearly. Include file names and evidence paths wherever possible."

## High priority rules for Codex outputs
- Prefer concrete facts over general descriptions.
- Distinguish clearly between implemented, tested, and only planned components.
- Do not hide weaknesses or null results.
- Do not write polished thesis prose unless explicitly asked.
- When unsure, say exactly what is uncertain.
- Always mention where in the codebase the information comes from.

## How ChatGPT should use Codex outputs
Once Codex has produced a handoff, ChatGPT can use it to:
- Draft section text
- Tighten argument structure
- Align methods and results sections
- Convert code facts into thesis language
- Identify missing links in the narrative
- Flag where more empirical detail is needed

## Suggested working routine
1. Choose one chapter or subsection.
2. Ask Codex for a structured handoff using this template.
3. Paste the handoff into ChatGPT.
4. Let ChatGPT convert it into thesis text.
5. Iterate until the text matches the evidence.
6. Repeat section by section.

## First sections to prioritize
- Introduction problem framing
- Data chapter
- Method pipeline overview
- Main results section
- Discussion of limitations

## Open questions to resolve
- What parts of the current codebase are final enough to describe confidently?
- Which experiments are exploratory and should stay out of the main thesis?
- Which outputs are the canonical final results?
- Which figures need to be regenerated for publication quality?
- Which claims can be defended directly from existing evidence?

