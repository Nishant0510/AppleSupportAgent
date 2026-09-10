# Apple Support Triage and Grounded Reply System — Evaluation Report

## 1. Problem framing

This project evaluates a support-triage pipeline for Apple-related social-media requests. The system assigns one primary intent, derives whether specialist escalation is needed, retrieves relevant historical resolution patterns, and generates a grounded support reply. The evaluation keeps a frozen golden set separate from development/retrieval material and evaluates both routing and reply quality.

**Evaluation data.** The frozen evaluation file contains 173 judged examples. A 30-example stratified human-review subset was independently scored on the same six 1–5 reply-quality dimensions: groundedness, on-brand tone, actionability, escalation handling, safety/correctness, and overall quality. The small sample size means the agreement results are diagnostic rather than a precise population estimate.

## 2. Results

### 2.1 Automated routing results (n = 173)

| Task | Metric | Result |
| --- | --- | ---: |
| Intent classification | Accuracy | 68.8% |
| Intent classification | Macro F1 | 67.5% |
| Intent classification | Weighted F1 | 69.8% |
| Escalation detection | Precision | 81.5% |
| Escalation detection | Recall | 72.1% |
| Escalation detection | F1 | 76.5% |
| Escalation detection | Accuracy | 84.4% |

The required trivial baseline is an always-majority intent classifier. On this evaluation set, the majority label is `software_update_performance` (71/173), yielding **41.0% accuracy**. The production classifier therefore improves accuracy by **27.8 percentage points** over this baseline. Add the measured simple baseline result (for example, TF-IDF + logistic regression) here before submitting the final report.

### 2.2 LLM-judged reply quality (n = 173)

| Dimension | Mean score (1–5) |
| --- | ---: |
| Groundedness | 4.66 |
| On-brand tone | 3.79 |
| Actionability | 3.38 |
| Escalation handling | 4.25 |
| Safety/correctness | 4.88 |
| Overall | 3.63 |

The evaluator assigns no overall score of 1, only one score of 2, 62 scores of 3, and 110 scores of 4. This compressed distribution is important when interpreting the mean.

### 2.3 Human–LLM judge agreement (n = 30)

| Dimension | Exact agreement | Within 1 point | Linear Cohen’s κ | Mean absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Groundedness | 50.0% | 90.0% | 0.144 | 0.63 |
| On-brand tone | 70.0% | 100.0% | 0.000 | 0.30 |
| Actionability | 50.0% | 100.0% | 0.242 | 0.50 |
| Escalation handling | 33.3% | 73.3% | -0.024 | 0.93 |
| Safety/correctness | 80.0% | 96.7% | 0.125 | 0.27 |
| Overall | 66.7% | 100.0% | 0.353 | 0.33 |

Exact agreement is only one view: ratings are ordinal, so within-one-point agreement and absolute error are also reported. Kappa is low partly because scores are concentrated in a narrow 3–5 range, which reduces the amount of variation available for chance-corrected agreement.

## 3. Five findings

1. **The intent model is useful but not ready to be summarized as “high accuracy.”** It reaches 68.8% accuracy and 67.5% macro F1, clearly above the 41.0% majority baseline, but still misclassifies 54 of 173 examples.
2. **Performance problems are over-routed to “feature/how-to.”** The most frequent confusion is `software_update_performance → feature_how_to` (15 cases). Ambiguous issue reports without device/context also drive confusion between `other_unclear`, hardware, and software labels.
3. **Escalation decisions are precision-oriented but miss meaningful risk cases.** Precision is 81.5%, whereas recall is 72.1%. This is safer than indiscriminate escalation, but approximately 28% of golden escalation cases are missed and should be reviewed by risk type.
4. **Replies are generally safe and grounded according to the LLM judge, but are less actionable.** The judge’s strongest dimension is safety/correctness (4.88/5); actionability is the weakest (3.38/5). Replies frequently ask for a device and OS version but omit a tailored next step.
5. **Human review supports approximate overall alignment, not blind reliance on the judge.** Overall scores are always within one point and have κ = 0.353, but groundedness and escalation handling have low κ (0.144 and -0.024). Use the LLM judge for scalable monitoring and sample human audit, not as the sole quality gate.

## 4. Interesting human–judge disagreements

There are no overall-score gaps of two or more points. The meaningful disagreements occur at the dimension level:

| Case | Dimension | Human vs. judge | Why it matters |
| --- | --- | --- | --- |
| Unplayable playlist | Groundedness | 1 vs. 4 | The reply treats a music-availability problem as possible data loss and recommends backups. The human score flags a mismatched remedy that the judge largely accepts. |
| New iPhone will not charge | Safety/correctness | 2 vs. 5 | The reply gives “Safe Mode” instructions that are not appropriate for iPhone. This is a concrete factual/safety miss that the LLM judge failed to detect. |
| iPhone 8 carrier activation | Groundedness | 2 vs. 4 | The reply includes an unsupported `support.apple.com/one-time-payment` path and fails to directly address carrier compatibility. |
| Eight non-escalated cases | Escalation handling | 5 vs. 3 | The human reviewer treated “no escalation required” as correctly handled; the judge scored these replies down for not explicitly describing escalation. The rubric should distinguish *correct non-escalation* from *insufficient escalation follow-through*. |

## 5. The misleading headline number

The headline “**84.4% escalation accuracy**” is not a sufficient safety claim. Non-escalation is the majority class (112/173, 64.7%), so a model can appear accurate while missing risky cases. The more decision-relevant result is **72.1% escalation recall**: 17 of 61 escalation-labelled examples were not escalated. Similarly, the 3.63/5 average LLM-judged overall reply score conceals a compressed judge distribution and low agreement on escalation handling. Report accuracy and mean score only alongside class balance, recall/F1, score distribution, and human–judge agreement.

## 6. Decision log

1. Adopt one primary support intent plus separate routing flags instead of mixing intent, tone, and language in one label.
2. Freeze the evaluation set before tuning prompts, retrieval, or few-shot examples.
3. Keep golden examples out of retrieval and prompt demonstrations to avoid evaluation leakage.
4. Use the 173 evaluated examples as the current frozen set; state the exact count rather than claiming a nominal target size.
5. Compare against an always-majority baseline (41.0% accuracy) so classifier performance is interpretable.
6. Add the already-planned simple baseline result before final submission; do not substitute a narrative comparison.
7. Measure both macro F1 and accuracy for intent because class frequencies are uneven.
8. Prioritize escalation recall alongside precision and accuracy because false negatives can carry disproportionate user risk.
9. Keep escalation reasons inspectable at the individual-case level.
10. Use retrieval-grounded reply generation rather than an unconstrained reply generator.
11. Evaluate reply quality across six dimensions rather than only with an overall score.
12. Report exact agreement, within-one agreement, weighted kappa, and mean absolute difference for the human audit.
13. Treat the LLM judge as a scalable evaluator with periodic human audits; do not use it as a sole correctness oracle.
14. Update the reply rubric to explicitly reward correct non-escalation and to penalize unsupported product-specific instructions/links.
15. Add targeted regression tests for playlist availability, iPhone charging, carrier activation, and security/phishing cases.

## 7. Limitations and next steps

- The human-agreement subset has 30 examples, so it is too small to make fine-grained reliability claims.
- Some gold labels and model classifications are imperfect on ambiguous, short social posts; qualitative review should inform taxonomy/prompt revisions.
- The current LLM judge is over-permissive on some concrete factual errors. Expand the human-audited set and add product-specific correctness checks.
- Before release, report the measured simple baseline and rerun the same evaluation after targeted fixes to check for regressions.
