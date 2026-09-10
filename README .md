# Apple Support Reply Assistant

An end-to-end, retrieval-grounded support-assistance pipeline for Apple-style customer-support threads. The system classifies the customer's primary intent, retrieves similar historically resolved cases, drafts a grounded reply with Ollama's `qwen2.5:7b` model, and makes an explicit escalation recommendation with an inspectable reason.

The project is designed to be evaluated as a complete system—not only by headline intent accuracy, but also against meaningful baselines, grounded-reply quality, and escalation behavior.

## What it does

1. **Classifies intent** from the incoming customer message using a few-shot LLM classifier.
2. **Retrieves precedent** by finding semantically similar, historically resolved support threads within the predicted intent.
3. **Generates a grounded reply** using the retrieved agent replies as evidence and style guidance.
4. **Decides whether to escalate** through transparent rules (and, where configured, model judgment), always returning a machine-readable reason string.

## Intent taxonomy

Each conversation receives one primary intent:

| Intent | Typical topics |
| --- | --- |
| `account_access_security` | Apple Account, password, verification, account recovery, suspicious activity |
| `billing_subscription_purchase` | charges, refunds, subscriptions, purchases, payment methods |
| `device_hardware_repair` | physical damage, battery, display, repair or service |
| `device_software_performance` | updates, crashes, setup, slowness, apps, iCloud sync |
| `orders_shipping_trade_in` | order status, delivery, cancellation, trade-in |
| `warranty_coverage` | AppleCare, coverage, eligibility, repair pricing |
| `technical_support_other` | technical issues outside the categories above |
| `product_information` | product capabilities, compatibility, availability |
| `feedback_complaint` | feedback and dissatisfaction not requiring another primary category |
| `other` | unsupported, unclear, or out-of-scope requests |

Routing flags such as `non_english`, `multiple_issues`, and `not_a_request` are tracked separately from the primary intent where available.

## Repository layout

```text
.
├── baseline                 #                 
├── data/
│   ├── golden_set.csv               # Frozen 200-row evaluation set
│   ├── development_300.csv          # Development/retrieval corpus
│   ├── development_300_with_replies.csv
│   ├── golden_eval_judged.csv       # LLM-judged generated-reply evaluation
│   └── human_review_sample.csv      # Human-review subset
├── Judge/
│   ├── build_human_review_sample.py #Script for Reviewing LLM sample by human
│   ├── compute agreement          # computing agreement of Human and LLM replies
│   ├── human_review_tool          #human grading of generated replies
│   └── judge_human_agreement_merged      # merged judge_Human-review subset
├── Labelling                    # Dataset labeling workflow
│   ├── golden_set_candidates(labelled).csv               # Frozen 200-row set labelled
│   ├── golden_set_candidates.csv          # Frozen 200-row set labelled
│   ├── label_tool.py
│   └── labeling_guide      # labeling guide
├── Preprocessing             
│   ├── Assignment.ipynb               #chosing of company and data cleaning
│   ├── taxonomy_induction_sample.csv          #Samples for taxonomy identification
│   └── thread_openers.csv
├── src                              #main sourcefile
│   ├── classifier.py               #classifier for intent
│   ├── escalation.py               #escalation logic 
│   ├── pipeline.py                 #main pipeline
│   └── Replygenerator.py            # for generating Replies
├── Tests                           #individual test for each element
│   ├── classifier.py               
│   └── Replygenerator.py           
│   
├── Eval.ipynb                     #Evaluation of Results Notebook
├── final_report_draft.md          #Final Evaluation Report
├── README.md                      #Readme file      
└── Requirements.txt
```

> Dataset filenames may differ slightly in your local copy. Update paths in the scripts or command examples to match your data directory.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Qwen 2.5 7B model:

```bash
ollama pull qwen2.5:7b
```

Install Python dependencies

```bash
pip install -r reuirements.txt
```

## Quick start

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

Run the Pipeline on a customer message:

```bash
cd src
python pipeline.py "Question"
```

Generate a retrieval-grounded reply:


The reply-generation workflow expects a development corpus containing, at minimum:

| Column | Purpose |
| --- | --- |
| `opening_text` | Customer's opening message used for matching |
| `label_intent` | Primary intent label |
| `agent_reply` | Historically resolved agent reply used as grounding context |

When creating `development_300_with_replies.csv`, store the generated response in the `agent_reply` column so it can be used by the retriever.

## Pipeline design

### Intent classification

The classifier uses **few-shot prompting with `qwen2.5:7b`** rather than a separately fine-tuned model. This keeps the system simple to update when the taxonomy changes and works well for a modest labeled corpus. The trade-off is higher latency and less deterministic behavior than a trained TF-IDF/linear classifier; therefore, the project evaluates against both trivial and simple non-LLM baselines.

### Retrieval-grounded reply generation

For each predicted intent, the retriever embeds the incoming message and retrieves the nearest resolved threads from that intent. The generator receives those examples and is instructed to use only their supported guidance, avoid inventing policy, and state uncertainty or next steps clearly. This reduces unsupported free-form responses and allows reviewers to inspect the source cases behind a reply.

### Escalation policy

Every decision returns both an action and a reason, for example:

```json
{
  "escalate": true,
  "reason": "account_access_security: possible unauthorized access requires secure-account recovery handling"
}
```

Typical escalation triggers include suspected fraud or account compromise, safety risk, legal/privacy requests, irreversible billing disputes, repeated failed troubleshooting, and requests needing account-specific verification. Rules are intentionally explicit so they can be reviewed and tested independently of reply quality.

## Evaluation

Evaluate the frozen golden set separately from the development corpus.

- **Classification:** accuracy, macro F1, and per-intent results.
- **Baselines:** majority-class or keyword baseline, plus TF-IDF + logistic regression / nearest-neighbor baseline.
- **Retrieval and generation:** groundedness, tone, actionability, escalation handling, safety/correctness, and overall quality.
- **Escalation:** precision, recall, F1, and a review of false positives/negatives with their attached reasons.
- **Human agreement:** exact agreement, within-one agreement, Cohen's kappa, and mean absolute difference against the LLM judge on the shared review subset.

Keep the 200-row golden set frozen. Use the 300-row development set for prompt iteration, retrieval tuning, and error analysis only.

## Expected output shape

An end-to-end prediction should preserve the audit trail:

```json
{
  "classification": {
    "intent": "connectivity",
    "is_multi_issue": false,
    "is_non_english": false,
    "is_support_request": true,
    "confidence": "high",
    "parse_error": false,
    "raw_response": "{\"intent\":\"connectivity\",\"is_multi_issue\":false,\"is_non_english\":false,\"is_support_request\":true,\"confidence\":\"high\"}"
  },
  "escalation": {
    "escalate": false,
    "reason": "auto_handle: no high-risk trigger",
    "reason_codes": []
  },
  "reply": "Try turning Wi-Fi off and back on, restarting your device, and installing the latest software update. If it persists, let us know your device model, software version, and any network details. 📱💡",
  "grounding_sources": [
    {
      "source_id": "856115",
      "opening_text": "@AppleSupport Since I updated to the iOS 11.0.3 my iPhone 7+ is disconnecting on its own from my cellular data, and the battery life is bad",
      "agent_reply": "Sorry you’re having trouble connecting. Try turning Wi-Fi or Bluetooth off and back on, restarting the device, and installing the latest software update. If it persists, tell us your device model, software version, and the network/accessory involved.",
      "label_intent": "connectivity",
      "similarity": 0.12373575956271399,
      "retrieval_scope": "same_intent"
    },
    {
      "source_id": "742292",
      "opening_text": ".@115858 Why is the new #iOS11 update causing my wifi to turn on constantly? cc @116333",
      "agent_reply": "Sorry you’re having trouble connecting. Try turning Wi-Fi or Bluetooth off and back on, restarting the device, and installing the latest software update. If it persists, tell us your device model, software version, and the network/accessory involved.",
      "label_intent": "connectivity",
      "similarity": 0.022469154228776544,
      "retrieval_scope": "same_intent"
    },
    {
      "source_id": "96995",
      "opening_text": "@115858 @AppleSupport the Bluetooth function on my iPhone is super frustrating! I accidentallypressed forget device and now my iPhone is forgetting my wireless headphone forever! I did all reset but still can’t find my device! so frustrating!",
      "agent_reply": "Sorry you’re having trouble connecting. Try turning Wi-Fi or Bluetooth off and back on, restarting the device, and installing the latest software update. If it persists, tell us your device model, software version, and the network/accessory involved.",
      "label_intent": "connectivity",
      "similarity": 0.022327149244021263,
      "retrieval_scope": "same_intent"
    }
  ]
}
```

## Responsible-use notes

- This is a decision-support prototype, not an autonomous customer-support system.
- Do not place customer credentials, account numbers, payment details, or other sensitive data in prompts, logs, or evaluation files.
- Treat account-security, payment, safety, privacy, and legal matters conservatively: escalate when the system lacks a verified, safe resolution path.
- Inspect retrieved examples and escalation reasons during evaluation; an apparently fluent reply is not sufficient evidence of correctness.

## Future improvements

- Add calibrated confidence thresholds and an `uncertain` route.
- Persist retrieval embeddings to reduce startup time.
- Add intent-specific retrieval quality metrics and citation display in the UI.
- Build regression tests from known classifier, retrieval, and escalation failures.
- Compare few-shot Qwen with a fine-tuned lightweight classifier once more labeled data is available.

## Decision log

| #  | Decision                                                                                                              | Rationale                                                                                                                                                                       |
| -- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Selected **AppleSupport** as the target brand.                                                                        | It ranked #2 in both raw tweet volume and distinct-thread count, providing sufficient data for a meaningful taxonomy, development set, and frozen golden evaluation set.        |
| 2  | Validated reply quality before building the pipeline.                                                                 | A manual review of 15 sampled replies showed a mix of DM-deflection and genuine public troubleshooting, including specific steps, follow-up questions, and visible resolutions. |
| 3  | Used historical AppleSupport replies as grounding material.                                                           | The replies reflect real resolution patterns, not only generic triage phrasing or link-dump responses.                                                                          |
| 4  | Used ten primary intents.                                                                                             | This keeps labels focused on the customer’s main need and avoids mixing intent with emotion, complexity, or routing status.                                                     |
| 5  | Treated `multiple_issues`, `non_english`, and `not_a_request` as routing flags rather than intents.                   | A thread can have one primary customer need while also requiring special handling.                                                                                              |
| 6  | Reserved 200 human-reviewed examples as a frozen golden evaluation set.                                               | Keeping evaluation data separate prevents prompt, retrieval, and threshold tuning from inflating final results.                                                                 |
| 7  | Used the remaining 300 examples as the development set.                                                               | The development set supports taxonomy iteration, prompt design, retrieval tuning, and error analysis.                                                                           |
| 8  | Used macro-F1 alongside accuracy.                                                                                     | Intent labels are imbalanced, so macro-F1 prevents large classes from dominating the headline metric.                                                                           |
| 9  | Used a majority-class classifier as the trivial baseline.                                                             | This establishes the minimum performance level that a real classifier must beat.                                                                                                |
| 10 | Used TF-IDF with logistic regression as the simple baseline.                                                          | It provides a fast, interpretable non-LLM comparison for judging whether the LLM adds meaningful value.                                                                         |
| 11 | Used a few-shot Qwen 2.5 7B classifier through Ollama.                                                                | Few-shot prompting is easier to revise when the taxonomy changes; the trade-off is greater latency and less deterministic behavior than a fine-tuned classifier.                |
| 12 | Embedded customer opening messages for retrieval.                                                                     | Incoming queries are customer-authored, so matching them against similar historical customer problems is more appropriate than embedding agent replies alone.                   |
| 13 | Filtered retrieval by predicted intent before similarity ranking.                                                     | This reduces semantically similar but operationally irrelevant examples from other issue categories.                                                                            |
| 14 | Excluded golden-set threads from the retrieval index.                                                                 | This prevents retrieval leakage, where evaluation cases could be answered using their own historical resolution.                                                                |
| 15 | Used deterministic escalation rules for explicit safety signals and attached an inspectable reason to every decision. | Account-security, fraud, safety, privacy, legal, and irreversible billing cases should not depend solely on generative-model judgment.                                          |
| 16 | Tuned prompts, retrieval settings, and confidence thresholds only on development data.                                | The frozen golden set remains a valid final evaluation benchmark.                                                                                                               |
| 17 | Cached model outputs during evaluation.                                                                               | Caching improves reproducibility, reduces Ollama inference time, and makes error analysis easier.                                                                               |
| 18 | Measured human–LLM-judge agreement.                                                                                   | Judge scores are more credible when reported alongside exact agreement, within-one agreement, Cohen’s kappa, and mean absolute difference against human reviews.                |


## Why the headline number can be misleading

The reported intent-classification and reply-quality results should be interpreted as **offline evaluation results**, not as proof that the system is ready for autonomous production use.

* **The golden set is intentionally stratified.** It ensures meaningful representation of important intents, but it does not reflect the natural distribution of production support traffic. A score on this set may differ from performance on live, unbalanced traffic.

* **The golden set was created by one primary reviewer.** Although spot-checking and review improve quality, individual labeling decisions can introduce interpretation bias, especially for ambiguous or multi-issue conversations.

* **The Twitter conversations are from 2017.** Current Apple products, policies, procedures, support channels, and customer expectations may have changed. A historically appropriate resolution may no longer be valid today.

* **LLM-judge scores do not prove customer satisfaction.** They measure whether a draft appears grounded, actionable, safe, and on-brand according to an evaluation rubric; they do not measure whether a real customer’s issue was resolved or whether the customer was satisfied.

* **The generator and judge may share blind spots.** If related LLMs are used for generation and judging, they may reward the same fluent but unsupported reasoning patterns or fail to identify the same errors.

* **Auto-handle coverage depends on the safety threshold.** A stricter escalation threshold can improve safety but sends more cases to human agents; a looser threshold increases automation coverage but raises the risk of incorrectly handling sensitive cases.

* **Historical similarity is not proof of current validity.** Retrieved threads provide useful precedent and wording patterns, but similarity does not guarantee that an old resolution is still accurate, policy-compliant, or appropriate for the current case.

* **Reply evaluation measures drafts, not real-world outcomes.** The project evaluates generated text before it reaches a customer. It does not measure delivery success, follow-up behavior, issue resolution rate, customer satisfaction, or the impact of human-agent intervention.

Therefore, the headline metric is best understood as evidence that the approach is promising under a controlled offline setup. It should be complemented by human review, current policy validation, and carefully monitored live testing before operational deployment.

## Failure analysis

This section treats failures as system-level patterns rather than isolated mistakes. The most important risk is not simply lower intent accuracy: it is cases where the system appears confident, retrieves plausible-looking precedent, and then produces an unsafe or unsupported draft.

### Finding 1: Escalation policy is severity-blind within intent categories

The escalation policy relies too heavily on the predicted primary intent. This means two requests assigned to the same intent can receive the same handling even when one contains materially higher risk.

For example, `account_access_security` includes both routine password-reset questions and potential account-compromise reports. Similarly, `billing_subscription_purchase` includes straightforward subscription questions as well as disputed or potentially unauthorized charges. Treating intent as a proxy for severity causes the policy to miss the distinction.

**Failure mode:** a high-risk request is classified correctly but not escalated because the intent itself is considered auto-handleable.

**Why it matters:** this creates escalation false negatives—the most dangerous error direction. A plausible standard reply may be generated for a customer reporting suspicious account activity, fraud, privacy risk, or an irreversible billing problem.

**Hypothesis:** the policy needs severity features independent of intent, such as explicit language about unauthorized access, fraud, account takeover, safety, legal action, personal data, repeated failure, or financial loss.

**Fix:** apply a severity-first rule layer before the normal intent-based policy. If an explicit high-risk signal is present, escalate regardless of predicted intent and record the triggering phrase in `escalation_reason`.

---

### Finding 2: Question-phrased bug reports get misclassified as `feature_how_to`

Bug reports often appear as questions: “How do I make this work?”, “Why won’t this feature open?”, or “Can I use this after updating?” The question form and feature vocabulary can cause the classifier to treat a malfunction as a request for instructions.

**Failure mode:** `device_software_performance` or technical-failure cases are predicted as `feature_how_to`.

**Why it matters:** the retrieved examples then contain usage explanations instead of troubleshooting steps. The generated reply may explain intended functionality without acknowledging that the feature is failing.

**Hypothesis:** the classifier overweights interrogative phrasing (`how`, `can`, `why`) and underweights failure signals such as `won't`, `stopped`, `crashes`, `after update`, `error`, and repeated unsuccessful attempts.

**Fix:** add contrastive few-shot examples that pair feature questions with feature failures. Add a diagnostic check before finalizing the intent: if a question also contains failure language, prefer a troubleshooting intent or lower confidence and escalate to clarification.

---

### Finding 3: `other_unclear` both over- and under-triggers

`other_unclear` serves two incompatible roles: a safe fallback for genuinely underspecified messages and a catch-all for cases the model cannot confidently map to the taxonomy.

**Over-triggering:** messages with enough technical, billing, or account context are sent to `other_unclear` because the wording is informal, abbreviated, or multi-issue.

**Under-triggering:** vague messages are forced into a specific intent because they contain a single misleading keyword.

**Why it matters:** over-triggering reduces automation coverage and makes the taxonomy appear less useful. Under-triggering is more serious because it produces overconfident retrieval and replies for requests that should have received a clarification question or human review.

**Hypothesis:** the current classifier lacks an explicit uncertainty decision separate from choosing an intent.

**Fix:** add a confidence threshold and a structured `clarification_required` route. Use `other_unclear` only when the message truly lacks enough information after applying routing flags; otherwise retain the best intent and request the missing detail.

---

### Finding 4: `billing_subscription_purchase` had only two training examples

The billing class has only two development examples, making both classification and retrieval unreliable.

**Failure mode:** the classifier has too little evidence to learn the variety of billing language, while retrieval repeatedly returns the same one or two precedents.

**Why it matters:** billing includes high-impact subcases such as refunds, duplicate charges, subscriptions, purchase disputes, and payment failures. Sparse coverage can make a model look accurate on the aggregate while failing on operationally important billing variations.

**Hypothesis:** apparent billing performance is dominated by lexical overlap with the two available examples rather than generalization.

**Fix:** collect and label additional billing threads before treating this intent as production-ready. Include examples covering subscription cancellation, refund status, duplicate charges, unauthorized charges, payment-method failure, purchase delivery, and regional pricing. Until then, use a lower-confidence threshold and conservative escalation for billing requests.

---

### Finding 5: Reply generator injects ungrounded factual content, especially URLs

The retrieval-grounded generator still introduces factual claims and URLs that do not appear in the retrieved historical examples.

**Failure mode:** the generated reply looks authoritative but adds unsupported links, policy claims, product-specific instructions, or service recommendations.

**Why it matters:** retrieval grounding is intended to constrain the model to known precedents. Invented URLs or facts undermine that guarantee and are especially risky because Twitter-era support information may no longer be current.

**Hypothesis:** the prompt encourages helpfulness more strongly than evidence fidelity. The model fills perceived gaps with learned background knowledge instead of explicitly stating that the retrieved evidence is insufficient.

**Fix:** enforce an evidence-bound generation policy:

1. Provide source thread IDs and retrieved text to the generator.
2. Instruct the model to include a claim only when supported by the retrieved examples.
3. Prohibit URLs unless they appear verbatim in retrieved context or come from a verified allowlist.
4. Add a post-generation validator that flags URLs, unsupported factual assertions, and product-policy claims absent from the retrieval context.
5. Escalate or ask a clarification question when retrieval evidence is weak instead of generating a polished but speculative answer.

### Cross-cutting conclusion

These failures are connected. An uncertain or misclassified message can retrieve the wrong precedent; the wrong precedent can make an unsupported reply sound convincing; and an escalation policy tied too closely to intent can fail to intervene. The next iteration should therefore prioritize:

1. Severity-aware escalation before automation coverage.
2. More representative development data for sparse intents, especially billing.
3. Explicit uncertainty and clarification routing.
4. Evidence-constrained generation and URL validation.
5. Evaluation slices for high-risk cases, question-phrased bug reports, and low-data intents—not only aggregate scores.
