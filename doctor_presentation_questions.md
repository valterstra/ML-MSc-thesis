# Doctor Presentation

## Slide 1: The Clinical Problem

Hospital readmission within 30 days of discharge is common, costly, and disruptive for patients. Many of these readmissions are at least partially preventable. The broad question driving this project is: given everything we observe about a patient during the hospital stay, can we learn a treatment strategy that leaves patients in a state where they are less likely to come back?

The motivation is to move from prediction alone toward decision support. In other words, not only to identify patients who are high risk, but to ask whether there are treatment decisions during the stay that might meaningfully shift that risk.

---

## Slide 2: What We Are Building

In broad terms, we are building two things. First, a virtual patient model, or simulator, that tries to capture how a patient's condition evolves over time under different treatment decisions. Second, a decision model that uses that simulator to learn which treatment choices are more likely to lead to a better patient trajectory.

The reason we need both parts is that the clinical question is not only who is high risk, but what may happen if we choose one treatment path rather than another. The simulator gives us a way to test short what-if scenarios, and the decision model tries to use that information to recommend better actions.

So in simple terms, the simulator is the part that says what may happen next, and the policy model is the part that says what action seems best to take now.

---

## Slide 3: The Data

We use MIMIC, which is a large hospital dataset from Beth Israel Deaconess in Boston. For this project, we focus on patients who survived to discharge and did not die within 30 days after discharge. That gives us a very large dataset to work with.

We structure each patient stay as a timeline. Along that timeline, we record what the patient looks like at a given point in time, such as laboratory values, vital signs, and patient background information, and we also record which treatments were given around that same time.

The reason for this timeline structure is that this is not just a one-time prediction problem. We are trying to model how patients change over time, how treatments are sequenced, and whether different choices during the stay may leave the patient in a different state at discharge.

---

## Slide 4: Why We Focus On Causality

At this point, the main question becomes: out of all the information in the hospital data, which parts are actually useful for decision-making? Not every variable that predicts readmission is something we can act on.

So the first thing we do is a feature selection step. In simple terms, we use a machine learning approach to rank which patient-state variables seem most informative for later readmission risk. That gives us a first shortlist of variables that appear important.

But prediction alone is not enough for our purpose. A variable can be highly predictive and still not be something that treatment can meaningfully influence. So on top of that ranking step, we run causal discovery. The goal there is to identify which of these patient-state variables are plausibly linked to treatment decisions in a way that could actually matter clinically.

Because causal discovery on observational data can be unstable, we do not trust a single run. Instead, we repeat it many times on different random subsamples and random subsets of variables. We only keep links that appear consistently across many runs. That robustness step is important, because otherwise it would be too easy to over-interpret noise.

So this slide is really about the logic of the pipeline: first narrow the state space to variables that seem important, then narrow again to variables and treatment links that are plausibly actionable.

---

## Slide 5: How We Select States And Actions

This slide should explain the two concrete screening tasks we run after the broad pipeline in Slide 4.

On the first side, we screen patient states. Here the question is: which end-state variables, meaning what the patient looks like toward the end of the stay, are most informative for later 30-day readmission? In simple terms, we rank candidate patient variables by how strongly they help predict readmission. This is how we identify a shortlist of candidate discharge-risk states.

On the second side, we screen treatment-state relationships. Here the question is: for a given treatment, does being more exposed to that treatment over the stay appear to shift the patient's physiology? Concretely, we summarize each treatment by the fraction of time blocks in which it was active, and we summarize each state by its change over the stay, meaning last value minus first value. Then we run causal discovery on those treatment-fraction and state-delta variables, while adjusting for static confounders and overall length of stay.

Because we do not trust any one run, we repeat these causal-discovery analyses many times on different random subsamples and different random subsets of variables. We then keep only the states and action-state links that remain stable across repeated runs.

The important point is that we cast a fairly wide net before narrowing it down. On the state side, this includes variables like hemoglobin, BUN, creatinine, phosphate, chloride, heart rate, blood pressure, oxygenation measures, lactate, shock index, and composite clinical scores. On the action side, this includes treatment categories such as antibiotics, anticoagulants, diuretics, insulin, opioids, sedatives, vasopressors, IV fluids, transfusions, electrolyte repletion, and mechanical ventilation.

So this slide is not yet about the final selected variables. It is about the selection procedure itself: first identify candidate end states linked to readmission, then identify which treatments seem able to move those states, and only after that decide what to carry into the final model.

Suggested on-slide bullets:
- Rank candidate end-state variables by how well they predict 30-day readmission
- For each treatment, test whether fraction active over the stay shifts state deltas
- Repeat causal discovery over many random samples and variable subsets
- Carry forward only robust states and robust treatment-state links

---

## Slide 6: From Selected Variables To A Virtual Patient Model

Once we have selected the states and actions, we use them to build the working model. The first part is a virtual patient model that predicts how the patient's condition may evolve over time under different treatment decisions. That gives us a simulator of short patient trajectories.

On top of that, we train a decision policy that learns which actions tend to lead to better trajectories. I do not want to go into the technical details here, but the main point is that the earlier selection step directly defines the model we build afterward.

But the main point for today is not the algorithm itself. The main point is whether this whole setup makes clinical sense. Are these the right variables? Are these realistic treatment decisions? Do the links we use look plausible? And what important parts of the real clinical decision process are still missing? That is what we want to discuss in the next part.

---

---

# Questions for Doctors

## Timing & Drug Administration

1. For drugs like antibiotics and diuretics, how often does the treatment decision actually change in practice: is it reviewed each shift or primarily at morning rounds? We're modeling decisions at 4-hour resolution for all drugs.
2. Who makes the vasopressor titration decision, the bedside nurse reacting to the monitor, or the team at rounds?
3. When a concerning lab result comes in, say rising creatinine, how quickly does that translate to a drug change: immediately at the bedside, or does it wait for the next rounds?
4. For most ICU patients, how often are labs like BUN and creatinine actually redrawn per day, once, twice, or more?
5. Which of these drug decisions follow standing protocols where the action is essentially automatic given a lab value, and which involve genuine clinical judgment?

---

## State Variables: What's Clinically Actionable

- When deciding whether a patient is ready to leave the ICU, what are you actually looking at: labs, vitals, a combination? Is there a checklist or more of a gestalt clinical judgment?
- Are there specific thresholds you use, values below which you'd hold a patient regardless of everything else looking fine?

1. Which lab or vital values do you explicitly check off before discharge? Are there any single values that would hold a patient even if everything else looks good?
2. For chronic patients, say someone with known CKD, do you target normalization of creatinine, or return to their personal baseline? How do you handle "abnormal but normal for them"?
3. Of the values we're tracking, hemoglobin, BUN, creatinine, heart rate, shock index, which can you realistically move with drugs during an ICU stay, and which are more a reflection of disease trajectory regardless of treatment?
4. In your experience, what's the most common reason a patient bounces back within 30 days: a borderline lab at discharge, a social or support failure, or something not detectable from the data at all?
5. Do you use composite scores like SOFA, which scores 6 organ systems, respiration, coagulation, liver, cardiovascular, CNS, and renal, each 0 to 4 with a total of 0 to 24, or individual values when making the discharge decision? How much weight does a composite like that carry vs. the individual numbers?
6. Most patients leave the ICU to a general ward before going home. Does readmission risk get meaningfully reset at that transition, or is the ICU discharge state still the dominant predictor?
7. Shock index, heart rate divided by systolic BP, appears as one of our strongest readmission predictors. Is that something you actually track clinically, or would you more naturally look at HR and BP separately?
8. How much of the discharge decision is non-physiological: home situation, support network, follow-up availability? And do those factors predict readmission as strongly as clinical markers?

---

## Discharge Destination

1. When choosing between home, home with services, or institutional care, what are the top two or three factors driving that decision?
2. Is discharge destination primarily a clinical decision, or heavily constrained by non-clinical factors like bed availability, insurance, or what the family can manage?
3. If a patient goes to a nursing home rather than home, does that always mean they're clinically more fragile, or are there patients in institutional care mainly for social reasons?
4. Are there patients discharged home who you'd privately consider higher risk, but home was chosen because rehab wasn't available or the patient refused?
5. Our model recommends home discharge much more aggressively than physicians actually practice, roughly 42% vs. 24%. Does an aggressive push toward home have any clinical basis: is there evidence ICU patients do better recovering at home when medically stable enough?
6. Do patients discharged to nursing homes or rehab have higher 30-day readmission than those going home, or the opposite, because they have more supervision?
7. Is there a discharge destination you associate specifically with avoidable readmissions, patients who came back within a month and you felt the destination choice contributed?
8. What happens at discharge that we can't see in ICU data: follow-up appointments, medication reconciliation, patient education, that meaningfully affects whether they come back?
9. In the Stockholm system, is there a more structured discharge pathway, mandatory follow-up within 7 days for example, that might make readmission dynamics different from what we see in US data?

---

## Actions Not Captured

1. Our action space covers drug interventions grouped into broad categories: antibiotics, anticoagulants, diuretics, insulin, opioids, sedatives, vasopressors, IV fluids, transfusions, electrolyte repletion, and mechanical ventilation, plus the discharge destination decision. What meaningful actions fall completely outside this list: physiotherapy, nutritional management, dialysis, procedures like bronchoscopy, palliative care conversations, or post-discharge follow-up scheduling? Which of those do you think most strongly affects whether a patient comes back within 30 days?
2. What patient characteristics not in routine lab or vital data, frailty, functional status, cognitive baseline, social support, would most improve a readmission prediction model if we had them?
3. Is there a class of patient where you know from the moment of ICU admission that readmission is almost certain, and if so, what defines that group?

---

## Readmission Outcome

1. When a patient is readmitted within 30 days, what fraction of those cases do you think are genuinely avoidable vs. expected given the underlying disease?
2. Is 30-day readmission the most clinically meaningful window, or would 7-day or 90-day tell a more actionable story?
3. Our dataset covers only the ICU portion of the stay. In the ICU we get very granular data: 4-hour time blocks, continuous vitals, detailed drug records. Once a patient moves to the general ward, we lose almost all of that resolution. How do you see the division between ICU and ward care in terms of what's driving a patient toward or away from readmission, and are there decisions made on the ward that matter as much as or more than what happened in the ICU?
