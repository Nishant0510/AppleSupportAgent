#!/usr/bin/env python3
"""
Terminal labeling tool for the AppleSupport golden eval set.

Usage:
    python3 label_tool.py

Reads data/golden_set_candidates.csv, shows one message at a time,
prompts for one primary intent + routing flags + escalation label + reason, saves incrementally
after every single label (so you can Ctrl+C any time without losing work
and resume later exactly where you left off).

Keyboard shortcuts for intent (type the number):
    1  account_access_security
    2  billing_subscription_purchase
    3  connectivity
    4  data_backup_loss
    5  hardware_physical
    6  software_update_performance
    7  feature_how_to
    8  service_complaint
    9  other_unclear
    n  non-English v1 input (out_of_scope, auto-escalate)
    x  non-support / ad / unrelated (out_of_scope, exclude)
    s  skip for now (leave unlabeled, revisit later)
    q  save and quit
"""
import pandas as pd
import os
import sys

CSV_PATH = "data/golden_set_candidates.csv"

INTENT_MAP = {
    "1": "account_access_security",
    "2": "billing_subscription_purchase",
    "3": "connectivity",
    "4": "data_backup_loss",
    "5": "hardware_physical",
    "6": "software_update_performance",
    "7": "feature_how_to",
    "8": "service_complaint",
    "9": "other_unclear",
}

# defaults per intent — labeler can override when prompted
DEFAULT_ESCALATE = {
    "account_access_security": True,
    "billing_subscription_purchase": True,
    "connectivity": False,
    "data_backup_loss": False,
    "hardware_physical": True,
    "software_update_performance": False,
    "feature_how_to": False,
    "service_complaint": True,
    "other_unclear": False,
}

DEFAULT_REASON = {
    "account_access_security": "security-sensitive, needs identity verification",
    "billing_subscription_purchase": "money-sensitive",
    "connectivity": "routine troubleshooting, safe to auto-handle",
    "data_backup_loss": "routine triage, safe to auto-handle unless severe",
    "hardware_physical": "liability or safety-sensitive",
    "software_update_performance": "routine troubleshooting, safe to auto-handle",
    "feature_how_to": "informational, safe to auto-handle",
    "service_complaint": "prior support attempt already failed customer",
    "other_unclear": "insufficient detail; ask a clarifying question",
}

LABEL_COLUMNS = [
    "label_intent", "is_multi_issue", "is_non_english", "is_support_request",
    "label_escalate", "label_reason",
]


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def load():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run from the apple_support_agent/ directory.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    for column in LABEL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def save(df):
    df.to_csv(CSV_PATH, index=False)


def progress(df):
    total = len(df)
    done = (df["label_intent"] != "").sum()
    return done, total


def main():
    df = load()
    done, total = progress(df)
    print(f"Loaded {total} candidates, {done} already labeled.\n")

    for idx in df.index:
        if df.at[idx, "label_intent"] != "":
            continue  # already labeled, skip

        clear()
        done, total = progress(df)
        print(f"=== [{done}/{total} labeled] ===\n")
        print(f"Month: {df.at[idx, 'month']}   Thread: {df.at[idx, 'thread_id']}   n_turns: {df.at[idx, 'n_turns']}\n")
        print(f"  {df.at[idx, 'opening_text']}\n")
        print("Intent categories:")
        for k, v in INTENT_MAP.items():
            print(f"  {k:>3}  {v}")
        print("    n  non-English v1 input (out_of_scope, auto-escalate)")
        print("    x  non-support / ad / unrelated (out_of_scope, exclude)")
        print("    s  skip for now")
        print("    q  save & quit")

        choice = input("\n> Intent: ").strip().lower()

        if choice == "q":
            save(df)
            done, total = progress(df)
            print(f"\nSaved. {done}/{total} labeled so far.")
            return
        if choice == "s":
            continue
        if choice == "x":
            df.at[idx, "label_intent"] = "out_of_scope"
            df.at[idx, "is_multi_issue"] = "False"
            df.at[idx, "is_non_english"] = "False"
            df.at[idx, "is_support_request"] = "False"
            df.at[idx, "label_escalate"] = ""
            df.at[idx, "label_reason"] = "excluded: not a support request"
            save(df)
            continue
        if choice == "n":
            df.at[idx, "label_intent"] = "out_of_scope"
            df.at[idx, "is_multi_issue"] = "False"
            df.at[idx, "is_non_english"] = "True"
            df.at[idx, "is_support_request"] = "True"
            df.at[idx, "label_escalate"] = "True"
            df.at[idx, "label_reason"] = "non-English input, out of scope for v1 auto-handling"
            save(df)
            continue
        if choice not in INTENT_MAP:
            print("Unrecognized input, treating as skip.")
            input("Press enter to continue...")
            continue

        intent = INTENT_MAP[choice]
        default_esc = DEFAULT_ESCALATE[intent]
        default_reason = DEFAULT_REASON[intent]

        multi_in = input("> Multiple distinct issues? [default: False] (y/n/enter=default): ").strip().lower()
        is_multi_issue = multi_in in ("y", "yes", "true")

        esc_in = input(
            f"> Escalate? [default: {default_esc}] (y/n/enter=default): "
        ).strip().lower()
        if esc_in == "":
            escalate = default_esc
        elif esc_in in ("y", "yes", "true"):
            escalate = True
        else:
            escalate = False

        reason_in = input(f"> Reason [default: \"{default_reason}\"]: ").strip()
        reason = reason_in if reason_in else default_reason

        df.at[idx, "label_intent"] = intent
        df.at[idx, "is_multi_issue"] = str(is_multi_issue)
        df.at[idx, "is_non_english"] = "False"
        df.at[idx, "is_support_request"] = "True"
        df.at[idx, "label_escalate"] = str(escalate)
        df.at[idx, "label_reason"] = reason
        save(df)

    print("\nAll rows labeled!")
    save(df)


if __name__ == "__main__":
    main()
