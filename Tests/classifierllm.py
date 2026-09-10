"""
LLM few-shot intent classifier, using a local Ollama model.

Design decisions (see decision log):
  - Few-shot examples are drawn from development_300.csv, capped at 3 per
    category, so the dominant software_update_performance class doesn't
    crowd out the prompt and bias the model toward over-predicting it.
  - billing_subscription_purchase has only 2 dev examples total (both used).
  - out_of_scope (non-English) and service_complaint are NOT classified here
    -- they're handled by pre-classification routing gates
    (see labeling_guide_v2.md). This module assumes input has already
    passed those gates.
  - The model is prompted to return strict JSON so parsing is reliable and
    we can distinguish "model refused/malformed" from "model chose a label"
    when scoring -- important for honest failure analysis later.

Requires: a local Ollama server running with the target model pulled.
    ollama pull llama3.1:8b
    ollama serve   (usually auto-running after install)
"""
import json
import re
import time
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

INTENTS = [
    "software_update_performance",
    "feature_how_to",
    "connectivity",
    "hardware_physical",
    "account_access_security",
    "data_backup_loss",
    "billing_subscription_purchase",
    "other_unclear",
]

INTENT_DEFS = {
    "software_update_performance": "Battery drain, freezing, restarts, slowness, or UI glitches, tied to an update.",
    "feature_how_to": "No malfunction -- customer doesn't know how to do something, or asks an informational question.",
    "connectivity": "WiFi, Bluetooth, cellular, or hotspot connection issues.",
    "hardware_physical": "Physical device failure, damage, or injury (e.g. shock, burn, cracked casing).",
    "account_access_security": "Locked out, activation errors, phishing/scam concerns, login/verification failures.",
    "data_backup_loss": "Missing photos, disappearing purchased content, failed backups/restores.",
    "billing_subscription_purchase": "Unauthorized or incorrect charges, duplicate purchases, missing purchased content tied to payment.",
    "other_unclear": "Vague complaint with no specific, actionable issue identifiable.",
}

DEFAULT_ESCALATE = {
    "software_update_performance": False,
    "feature_how_to": False,
    "connectivity": False,
    "hardware_physical": True,
    "account_access_security": True,
    "data_backup_loss": True,
    "billing_subscription_purchase": True,
    "other_unclear": False,
}


def build_few_shot_examples(dev_df: pd.DataFrame, per_class_cap: int = 3) -> str:
    scoped = dev_df[
        (dev_df["is_support_request"] == True)
        & (dev_df["is_non_english"] == False)
        & (~dev_df["label_intent"].isin(["out_of_scope", "service_complaint"]))
    ]
    blocks = []
    for intent in INTENTS:
        sub = scoped[scoped["label_intent"] == intent].head(per_class_cap)
        for _, row in sub.iterrows():
            blocks.append(f'Message: "{row["opening_text"]}"\nIntent: {intent}')
    return "\n\n".join(blocks)


def build_prompt(message: str, few_shot_block: str) -> str:
    intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFS.items())
    return f"""You are classifying customer support tweets sent to @AppleSupport into exactly one intent category.

Categories:
{intent_list}

Examples:
{few_shot_block}

Now classify this new message. Respond with ONLY a JSON object, no other text:
{{"intent": "<one of the category names above>", "is_multi_issue": <true/false>, "confidence": "<low/medium/high>"}}

Message: "{message}"
JSON:"""


def call_ollama(prompt: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1)


def parse_response(raw: str) -> dict:
    """Extract the JSON object from the model's response, tolerating minor
    formatting noise (e.g. markdown code fences, leading/trailing text)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"intent": None, "is_multi_issue": None, "confidence": None, "parse_error": True, "raw": raw}
    try:
        parsed = json.loads(match.group(0))
        parsed["parse_error"] = False
        if parsed.get("intent") not in INTENTS:
            parsed["parse_error"] = True  # model invented a label outside our taxonomy
        return parsed
    except json.JSONDecodeError:
        return {"intent": None, "is_multi_issue": None, "confidence": None, "parse_error": True, "raw": raw}


def classify(message: str, few_shot_block: str) -> dict:
    prompt = build_prompt(message, few_shot_block)
    raw = call_ollama(prompt)
    result = parse_response(raw)
    result["raw_response"] = raw
    return result


def classify_batch(df: pd.DataFrame, few_shot_block: str, text_col: str = "opening_text") -> pd.DataFrame:
    results = []
    for i, row in df.iterrows():
        r = classify(row[text_col], few_shot_block)
        results.append(r)
        if (len(results)) % 10 == 0:
            print(f"  classified {len(results)}/{len(df)}...")
    out = df.copy()
    out["llm_intent"] = [r.get("intent") for r in results]
    out["llm_is_multi_issue"] = [r.get("is_multi_issue") for r in results]
    out["llm_confidence"] = [r.get("confidence") for r in results]
    out["llm_parse_error"] = [r.get("parse_error") for r in results]
    out["llm_raw_response"] = [r.get("raw_response") for r in results]
    return out


if __name__ == "__main__":
    dev = pd.read_csv("development_300.csv")
    golden = pd.read_csv("golden_eval_200.csv")

    few_shot_block = build_few_shot_examples(dev)
    print("Few-shot prompt block:\n")
    print(few_shot_block[:1000], "...\n")

    # scope golden the same way baselines.py did
    scoped_golden = golden[
        (golden["is_support_request"] == True)
        & (golden["is_non_english"] == False)
        & (~golden["label_intent"].isin(["out_of_scope", "service_complaint"]))
    ].copy()

    print(f"\nClassifying {len(scoped_golden)} golden examples with {MODEL} via Ollama...\n")
    result_df = classify_batch(scoped_golden, few_shot_block)
    result_df.to_csv("golden_eval_with_llm_predictions.csv", index=False)
    print("\nSaved to golden_eval_with_llm_predictions.csv")

    n_errors = result_df["llm_parse_error"].sum()
    print(f"Parse errors: {n_errors}/{len(result_df)}")