#!/usr/bin/env python3
"""
Terminal tool for human grading of generated replies, blind to the LLM
judge's scores (data/human_review_sample.csv has no judge columns).

Usage:
    python3 human_review_tool.py

Scores each reply 1-5 on the same 5 dimensions the LLM judge uses, plus
overall. Saves incrementally, resumable.
"""
import pandas as pd
import os
import sys

CSV_PATH = "data/human_review_sample.csv"
DIMS = ["grounded", "on_brand_tone", "actionable", "escalation_handling", "safety_correctness", "overall"]

DIM_PROMPTS = {
    "grounded": "Does the reply avoid inventing specific facts/URLs beyond safe generic advice?",
    "on_brand_tone": "Is the tone concise, empathetic, professional?",
    "actionable": "Does it give the customer something concrete to do next?",
    "escalation_handling": "Does escalation get handled correctly (deflect if True, help if False)?",
    "safety_correctness": "Is it free of wrong or unsafe advice? (most important dimension)",
    "overall": "Overall quality 1-5",
}


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run build_human_review_sample.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    for idx in df.index:
        if df.at[idx, "human_overall"] != "":
            continue

        clear()
        done = (df["human_overall"] != "").sum()
        print(f"=== [{done}/{len(df)} graded] ===\n")
        print(f"Intent: {df.at[idx,'llm_intent']}   Escalate: {df.at[idx,'llm_escalate_derived']}\n")
        print(f"CUSTOMER: {df.at[idx, 'opening_text']}\n")
        print(f"REPLY: {df.at[idx, 'generated_reply']}\n")
        print("-" * 60)

        scores = {}
        for dim in DIMS:
            while True:
                val = input(f"{dim} [{DIM_PROMPTS[dim]}] (1-5, or 'q' to save&quit): ").strip()
                if val.lower() == "q":
                    df.to_csv(CSV_PATH, index=False)
                    print("Saved. Exiting.")
                    return
                if val in ("1", "2", "3", "4", "5"):
                    scores[dim] = val
                    break
                print("  enter 1-5")

        for dim, val in scores.items():
            df.at[idx, f"human_{dim}"] = val
        df.to_csv(CSV_PATH, index=False)

    print("\nAll rows graded!")


if __name__ == "__main__":
    main()
