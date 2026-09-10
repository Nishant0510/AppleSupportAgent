"""
Builds a stratified sample for the human-vs-LLM-judge agreement check.

Design: sample ~30 replies (a reasonable size for one person to hand-grade
carefully in under an hour) stratified across:
  - escalate=True and escalate=False (both need coverage)
  - a few different intents, not just the dominant one
  - deliberately over-sample cases likely to be judge-disagreement-prone:
    replies containing a spelled-out URL (Finding 5) and hardware_physical
    replies (safety-sensitive, highest stakes for judge reliability)

This is run AFTER llm_judge.py has produced golden_eval_judged.csv, so the
human reviewer can blind-grade the same set the judge scored, then we
compute agreement.
"""
import pandas as pd

df = pd.read_csv("golden_eval_judged.csv")

# Ensure coverage of interesting/high-stakes cases
has_url = df["generated_reply"].str.contains(r"https?://|\.com", regex=True, na=False)
is_hardware = df["llm_intent"] == "hardware_physical"
is_escalate = df["llm_escalate_derived"] == True

priority = df[has_url | is_hardware].sample(frac=1, random_state=11).head(12)
remaining_pool = df.drop(priority.index)

# fill the rest with a stratified-ish random sample balancing escalate T/F
esc_true = remaining_pool[remaining_pool["llm_escalate_derived"] == True].sample(
    n=min(9, (remaining_pool["llm_escalate_derived"] == True).sum()), random_state=11
)
esc_false = remaining_pool[remaining_pool["llm_escalate_derived"] == False].sample(
    n=min(9, (remaining_pool["llm_escalate_derived"] == False).sum()), random_state=11
)

review_sample = pd.concat([priority, esc_true, esc_false]).drop_duplicates(subset="thread_id")
review_sample = review_sample.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle order

print(f"Review sample size: {len(review_sample)}")
print(review_sample["llm_intent"].value_counts())
print()
print("escalate True/False split:", review_sample["llm_escalate_derived"].value_counts().to_dict())

# Human reviewer fills these in BLIND to the judge's scores -- keep only the
# inputs needed to grade, not the judge output, to avoid anchoring bias.
human_review = review_sample[
    ["thread_id", "opening_text", "llm_intent", "llm_escalate_derived", "generated_reply"]
].copy()
for dim in ["grounded", "on_brand_tone", "actionable", "escalation_handling", "safety_correctness", "overall"]:
    human_review[f"human_{dim}"] = ""

human_review.to_csv("human_review_sample.csv", index=False)
print("\nSaved blind review sheet to human_review_sample.csv")
print("(judge scores withheld from this file on purpose -- grade blind, then we merge & compare)")
