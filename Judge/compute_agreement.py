"""
Computes agreement between the LLM judge (data/golden_eval_judged.csv) and
the human reviewer (data/human_review_sample.csv, filled in via
human_review_tool.py), on the shared subset of thread_ids.

Reports, per dimension:
  - exact agreement rate
  - agreement within 1 point (a common looser standard for 1-5 rubrics)
  - Cohen's kappa (chance-corrected agreement)
  - mean absolute difference
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

DIMS = ["grounded", "on_brand_tone", "actionable", "escalation_handling", "safety_correctness", "overall"]


def main():
    judged = pd.read_csv("golden_eval_judged.csv")
    human = pd.read_csv("human_review_sample.csv")

    human_graded = human[human["human_overall"] != ""].copy()
    if len(human_graded) == 0:
        print("No human grades found yet -- run human_review_tool.py first.")
        return

    merged = human_graded.merge(judged, on="thread_id", suffixes=("", "_judged"))
    print(f"Comparing on {len(merged)} jointly-graded examples.\n")

    print(f"{'Dimension':<22}{'Exact %':<10}{'Within-1 %':<12}{'Kappa':<10}{'Mean |diff|':<12}")
    print("-" * 66)
    for dim in DIMS:
        h = merged[f"human_{dim}"].astype(int)
        j = merged[f"judge_{dim}"].astype(int)

        exact = (h == j).mean() * 100
        within1 = (abs(h - j) <= 1).mean() * 100
        mean_diff = abs(h - j).mean()
        try:
            kappa = cohen_kappa_score(h, j, weights="linear")
        except Exception:
            kappa = float("nan")

        print(f"{dim:<22}{exact:<10.1f}{within1:<12.1f}{kappa:<10.3f}{mean_diff:<12.2f}")

    merged.to_csv("judge_human_agreement_merged.csv", index=False)
    print("\nSaved merged comparison to judge_human_agreement_merged.csv")

    # flag the biggest disagreements for qualitative review
    merged["overall_diff"] = abs(merged["human_overall"].astype(int) - merged["judge_overall"].astype(int))
    big_diffs = merged[merged["overall_diff"] >= 2].sort_values("overall_diff", ascending=False)
    print(f"\nCases with overall disagreement >= 2 points: {len(big_diffs)}")
    for _, row in big_diffs.iterrows():
        print(f"  human={row['human_overall']} judge={row['judge_overall']} | {row['opening_text'][:80]}")
        print(f"    reply: {row['generated_reply'][:100]}")


if __name__ == "__main__":
    main()
