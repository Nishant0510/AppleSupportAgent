# AppleSupport Intent Taxonomy & Labeling Guide (v1)

**Scope:** Customer-initiated tweets to @AppleSupport, Sep–Dec 2017 window
(the only period with meaningful volume in this dataset — see decision log).

**Unit of labeling:** the customer's *opening* message in a thread (the first
inbound turn). Later turns in the same thread are not separately intent-labeled
in v1 — documented limitation, see report.

---

## Label fields

Every row has one `label_intent` plus three independent routing flags:

| Field | Values | Meaning |
| --- | --- | --- |
| `label_intent` | one taxonomy label below | The single underlying problem to address first. |
| `is_multi_issue` | `true` / `false` | The message contains two or more distinct support problems. |
| `is_non_english` | `true` / `false` | The opening message is not English. |
| `is_support_request` | `true` / `false` | The message is a genuine customer support request. |

Frustration is **tone, not intent**. When a concrete underlying problem is visible, label that problem even if the customer is angry.

---

## Step 0 — Routing flags (apply BEFORE intent labeling)

**Is the message in English?**
- No → set `is_non_english = true`, `is_support_request = true`, and use
  `label_intent = out_of_scope`; escalation = `ESCALATE` (reason:
  "non-English input, out of scope for v1 auto-handling").
- Yes → set `is_non_english = false` and proceed.

**Is this actually a customer support request?** (vs. Apple's own marketing
tweet caught in a reply chain, a retweet, or pure off-topic chatter with no
support content)
- No → set `is_support_request = false`, `is_multi_issue = false`, and use
  `label_intent = out_of_scope`. Exclude it from the golden set.
- Yes → set `is_support_request = true` and proceed.

For English support requests, set `is_multi_issue = true` only when two or more
distinct problems are present. Still choose the **one primary intent** that is
most central, urgent, or actionable. Do not use the flag merely because the
customer uses several emotional phrases.

---

## Intent categories

### 1. `account_access_security`
Locked out of account, activation errors, suspicious email/phishing check.
**Examples:**
- "I have been locked out of my account for 5 days and now it says 23 days."
- "Hi is this a scam email? Assuming yes"
**Default escalation:** ESCALATE (security-sensitive, identity verification
needed — not safely auto-handleable).

### 2. `billing_subscription_purchase`
Charges, refunds, subscriptions, purchase problems, or missing purchased
content.
**Examples:**
- "why do I have to pay 10€ for extra iCloud storage I never authorized"
**Default escalation:** ESCALATE (money-sensitive).

### 3. `connectivity`
Wi-Fi, Bluetooth, cellular, SIM, hotspot, network, or call-quality issues.

### 4. `data_backup_loss`
Missing photos or files, backups, deleted data, iCloud data, or restore
failures. A missing purchased item belongs to
`billing_subscription_purchase` instead.

### 5. `hardware_physical`
Physical device failure not clearly tied to a software update; includes
injury claims.
**Examples:**
- "Apple Watch burned my wrist pretty bad?? Help!"
- "the adapter for my sons iphone7 has failed to work"
**Default escalation:** ESCALATE (liability/safety-sensitive, especially
injury claims — always escalate injury regardless of anything else).

### 6. `software_update_performance`
Crashes, freezing, battery drain, slowness, bugs, or post-update failures.
An update mention is helpful but not required when the issue is clearly
software behavior rather than physical damage.

### 7. `feature_how_to`
No malfunction; customer doesn't know how to do something.
**Examples:**
- "how long is it safe to charge a mac/iphone/ipad/ipod for?"
- "Can I downgrade my iPhone 6s to iOS 10.3.3?"

### 8. `service_complaint`
Complaint about a *prior* support interaction — case left unresolved, long
wait, unhelpful rep, broken promise ("said 48 hours, been 3 days").
**Examples:**
- "AppleSupport said i'd get an email about my issue within 48 hours, 3 days ago"
**Default escalation:** ESCALATE (a prior automated/first-line attempt
already failed this customer — don't repeat it).

### 9. `other_unclear`
A genuine English support request with too little information to identify an
underlying problem. Use an empathetic clarification question. Do not use this
for anger alone when an underlying issue can be identified.

### 10. `out_of_scope`
Non-English v1 input, advertisements, unrelated content, retweets, or other
non-support material. Use the routing flags to record *why* it is out of scope.

---

## Escalation label (independent of intent, but intent sets the default)

Every example gets:
- `label_intent` (exactly one)
- `is_multi_issue`, `is_non_english`, and `is_support_request`
- `escalate: true/false`
- `reason:` one sentence, plain language (e.g., "injury claim — safety
  liability, not resolvable by scripted troubleshooting")

Escalation is not purely a function of intent — override the default when:
- Message shows self-harm/emotional crisis language → always escalate
  (out of scope for this project's auto-handling regardless of category)
- Customer explicitly asks for a human / says automation already failed them
- Legal threats, press/media mentions, public-figure accounts

When `is_multi_issue = true`, consider escalation if a single automated reply
would likely overlook an important issue; the flag is a routing signal, not an
intent category.

---

## Multi-labeler note
If more than one person labels, resolve disagreements by discussion, not
majority vote alone — log *why* the disagreement happened (ambiguous
taxonomy boundary vs. genuine reading-comprehension difference). This becomes
useful evidence for LLM-judge agreement analysis later.
