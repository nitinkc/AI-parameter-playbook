# AI Math Foundations Roadmap (for IT Professionals)

This roadmap designs a tutorial-style learning track that teaches AI math **incrementally**, in both plain English and practical math, with Python notebook labs.

## Why this track

- Many AI terms are used daily (token, probability, entropy, regression, standard deviation, determinism) but are not always taught with practical intuition.
- This roadmap bridges that gap for IT practitioners who need to **use AI systems confidently**, not become theoreticians.
- The sequence is designed so each concept builds naturally on the previous one.

## Audience and teaching goals

- Primary audience: IT professionals, architects, developers, analysts, product/ops teams.
- Assumption: mixed math confidence.
- Goal: explain each concept in three layers:
  1. Plain-English mental model
  2. Minimal math formula and interpretation
  3. Practical AI use case and notebook exercise

## Running real-world story (single unifying example)

Use one continuous scenario across all modules:

**IT Service Desk Copilot**

- Input: user tickets/emails/chat messages
- Tasks: classify intent, summarize issue, predict escalation risk, generate reply draft
- Why this works: it naturally uses tokenization, probabilities, uncertainty, regression/classification, variance, and deterministic vs stochastic settings.

## Learning path structure (incremental)

## Phase 0 - Orientation and notation

- What AI models output (scores/logits, probabilities, tokens)
- Quick notation guide used in all modules
- Notebook: mini warm-up with arrays, softmax, and plots

## Phase 1 - Language units and probability basics

### Module 1: Tokenization and vocabulary
- Plain English: what a token is and why token count affects cost, latency, and context limits
- Math: sequence length, token budget constraints
- IT example: service ticket length and truncation risk
- Notebook: split text into pseudo-tokens, compare token counts by message type

### Module 2: Probability refresher for AI
- Plain English: probability as model confidence, not truth
- Math: PMF, normalization, expected value
- IT example: intent routing probabilities
- Notebook: convert raw scores to probabilities, verify sum to 1

### Module 3: Logits and softmax
- Plain English: logits are relative preference scores
- Math: softmax and temperature scaling
- IT example: candidate intents for a ticket
- Notebook: visualize logits to probabilities at different temperatures

## Phase 2 - Uncertainty, variability, and control

### Module 4: Entropy and uncertainty
- Plain English: entropy is spread/uncertainty of choices
- Math: Shannon entropy over discrete distributions
- IT example: when classifier is uncertain, route to human
- Notebook: entropy vs peaked/flat distributions

### Module 5: Standard deviation, variance, and run-to-run stability
- Plain English: how much outcomes vary across repeated runs
- Math: mean, variance, standard deviation
- IT example: response variability in production prompts
- Notebook: run repeated generations/simulations and compute variability metrics

### Module 6: Deterministic vs stochastic behavior
- Plain English: same input can produce same or different outputs depending on settings
- Math: argmax vs sampling; role of random seed
- IT example: compliance-safe deterministic routing vs creative drafting
- Notebook: compare seeded and unseeded runs with temperature/top_p changes

## Phase 3 - Prediction models used in AI systems

### Module 7: Regression (practical refresher)
- Plain English: predict a number (e.g., time-to-resolution)
- Math: linear regression, residuals, MAE/RMSE
- IT example: predict SLA breach risk score or resolution hours
- Notebook: fit simple regression and interpret coefficients

### Module 8: Classification and calibration
- Plain English: predict categories with confidence
- Math: logistic function, cross-entropy, decision threshold
- IT example: incident type classification
- Notebook: precision/recall tradeoffs via threshold tuning

### Module 9: Correlation vs causation in AI ops
- Plain English: avoid false operational conclusions
- Math: correlation coefficients and confounding basics
- IT example: "long prompts cause incidents" vs hidden variables
- Notebook: synthetic data with confounders

## Phase 4 - LLM sampling and decoding controls

### Module 10: Temperature, top-p, top-k, penalties
- Plain English: creativity/reliability knobs
- Math: filtered distributions and renormalization
- IT example: same ticket summary under safe vs creative profiles
- Notebook: use your existing local sampling simulator + plots

### Module 11: Confidence-aware guardrails
- Plain English: when to abstain, escalate, or request clarification
- Math: thresholding and uncertainty bands
- IT example: auto-close vs route-to-human policy
- Notebook: policy simulation over confidence scores

## Phase 5 - Measurement and production decisions

### Module 12: Evaluation metrics that matter in enterprise AI
- Plain English: tie math metrics to business outcomes
- Math: aggregate metrics, confidence intervals, drift indicators
- IT example: rollout dashboard for IT copilot
- Notebook: compute and visualize KPI trends over runs

## Deliverables by content type

For each module:

- 1 concise markdown lesson (`math + plain English + IT example`)
- 1 Jupyter notebook (`hands-on mini lab`)
- 1 quick checklist (`what to look for, common pitfalls`)
- 1 "from math to decision" section (`how this impacts settings/governance`)

## Proposed folder layout

```text
# Docs track
/docs/math-foundations/
  overview.md
  notation.md
  01-tokenization.md
  02-probability.md
  03-logits-softmax.md
  04-entropy.md
  05-variance-stddev.md
  06-determinism.md
  07-regression.md
  08-classification-calibration.md
  09-correlation-causation.md
  10-sampling-controls.md
  11-guardrails-thresholds.md
  12-evaluation-in-production.md

# Notebook labs
/experiments/local/notebooks/math-foundations/
  00_warmup.ipynb
  01_tokenization.ipynb
  02_probability.ipynb
  03_logits_softmax.ipynb
  04_entropy.ipynb
  05_variance_stddev.ipynb
  06_determinism.ipynb
  07_regression.ipynb
  08_classification_calibration.ipynb
  09_correlation_causation.ipynb
  10_sampling_controls.ipynb
  11_guardrails_thresholds.ipynb
  12_eval_in_production.ipynb
```

## Teaching style rules (important)

- Start each module with a 5-line plain-English story.
- Use only one new formula at a time.
- Keep symbols consistent across modules.
- Every formula must answer: "what decision does this help me make?"
- Provide one pitfall and one anti-pitfall per module.

## Example of concept chaining (one mini thread)

1. Ticket text gets tokenized (token budget constraints)
2. Model outputs logits for intents (scores)
3. Softmax converts logits to probabilities
4. Entropy indicates confidence spread
5. Threshold policy routes low-confidence tickets to human
6. Track variance over weeks to detect instability
7. Use regression to estimate handling time and staffing impact

This single thread ties token, logit, probability, entropy, determinism, stddev, regression, and operational decisions together.

## Scope note

This roadmap is intentionally focused on **practical AI math literacy** for enterprise IT workflows. If desired, this can be split into a separate tutorial site later, but it also fits naturally inside this repository as a dedicated foundations track.

## Suggested execution plan (phased build)

- Sprint 1: `overview + modules 1-4 + notebooks 00-04`
- Sprint 2: `modules 5-8 + notebooks 05-08`
- Sprint 3: `modules 9-12 + notebooks 09-12`
- Sprint 4: polish, examples, assessments, and cross-links to existing parameter docs

## Definition of done (for each module)

- Concept understood by non-math readers in < 10 minutes
- Formula explained in plain language and variables defined
- Notebook runnable end-to-end
- Includes one enterprise IT scenario and one decision checklist
- Cross-linked to relevant pages in this site

## Next step

If you want, I can start implementing **Sprint 1** immediately in this repo:

1. Create `overview.md` + modules 1-4
2. Create notebooks 00-04 with runnable cells
3. Add nav entries under a new "Math Foundations" section
4. Keep your existing parameter content intact and cross-linked

--8<-- "_abbreviations.md"


