"""Few-shot Apple support intent and routing classifier using Ollama qwen2.5:7b.

Few-shot prompting is preferred to fine-tuning: the 300-row development set is
too small for a reliable fine-tune. The frozen golden set is never used here.
"""
from __future__ import annotations
import json, re, time
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Any
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
INTENTS = ["account_access_security", "billing_subscription_purchase", "connectivity", "data_backup_loss", "hardware_physical", "software_update_performance", "feature_how_to", "service_complaint", "other_unclear", "out_of_scope"]
INTENT_DEFS = {
 "account_access_security":"Apple Account, password, verification, activation, scam, or account recovery.", "billing_subscription_purchase":"Charges, subscriptions, purchases, refunds, or payment-linked missing content.", "connectivity":"Wi-Fi, Bluetooth, cellular, hotspot, or connection problems.", "data_backup_loss":"Lost/missing data, backup, restore, iCloud sync, or recovery.", "hardware_physical":"Physical damage, battery swelling, heat, shock, injury, or device hardware failure.", "software_update_performance":"Update-related bug, freeze, restart, battery drain, slowness, or UI problem.", "feature_how_to":"How-to or informational question without a malfunction.", "service_complaint":"Complaint about Apple service, staff, repair, or support experience as the main issue.", "other_unclear":"A support request whose issue is too vague to classify more specifically.", "out_of_scope":"Not a support request or a message that cannot be handled in English."}

def build_few_shot_examples(dev_df: pd.DataFrame, per_class_cap: int = 3) -> str:
    missing = {"opening_text", "label_intent"} - set(dev_df.columns)
    if missing: raise ValueError(f"Development data is missing required columns: {sorted(missing)}")
    blocks = []
    for intent in INTENTS:
        for _, row in dev_df[dev_df.label_intent == intent].head(per_class_cap).iterrows():
            blocks.append(f'Message: {json.dumps(str(row.opening_text))}\nOutput: {{"intent":"{intent}"}}')
    return "\n\n".join(blocks)

def build_prompt(message: str, examples: str) -> str:
    labels = "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFS.items())
    return f"""Classify one message to @AppleSupport. Use exactly one primary intent and three independent routing flags.
Primary intents:\n{labels}
Flags: is_multi_issue is true only for two distinct issues; is_non_english is true if not primarily English; is_support_request is true only for an Apple support request. Tone is not an intent.
Few-shot examples:\n{examples}
Return ONLY valid JSON: {{"intent":"one label above","is_multi_issue":true,"is_non_english":false,"is_support_request":true,"confidence":"low|medium|high"}}
Message: {json.dumps(message)}"""

def call_ollama(prompt: str, retries: int = 2) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            payload = json.dumps({"model":MODEL,"prompt":prompt,"stream":False,"format":"json","options":{"temperature":0}}).encode("utf-8")
            with urlopen(Request(OLLAMA_URL, data=payload, headers={"Content-Type":"application/json"}), timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))["response"]
        except (URLError, OSError, KeyError, ValueError) as error:
            last_error = error
            if attempt < retries: time.sleep(attempt + 1)
    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts: {last_error}")

def parse_response(raw: str) -> dict[str, Any]:
    empty = {"intent":None,"is_multi_issue":None,"is_non_english":None,"is_support_request":None,"confidence":None,"parse_error":True}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match: return empty
    try: parsed = json.loads(match.group(0))
    except json.JSONDecodeError: return empty
    valid = parsed.get("intent") in INTENTS and parsed.get("confidence") in {"low","medium","high"}
    valid = valid and all(isinstance(parsed.get(flag), bool) for flag in ("is_multi_issue","is_non_english","is_support_request"))
    return {**parsed, "parse_error":False} if valid else {**empty, "raw_parsed":parsed}

def classify(message: str, few_shot_block: str) -> dict[str, Any]:
    raw = call_ollama(build_prompt(message, few_shot_block)); return {**parse_response(raw), "raw_response":raw}

def classify_batch(df: pd.DataFrame, few_shot_block: str, text_col: str = "opening_text") -> pd.DataFrame:
    if text_col not in df: raise ValueError(f"Input is missing text column: {text_col}")
    results = [classify(str(text), few_shot_block) for text in df[text_col].fillna("")]
    output = df.copy()
    for field in ("intent","is_multi_issue","is_non_english","is_support_request","confidence","parse_error","raw_response"):
        output[f"llm_{field}"] = [result.get(field) for result in results]
    return output

if __name__ == "__main__":
    dev, golden = pd.read_csv("development_300.csv"), pd.read_csv("data/golden_eval_200.csv")
    result = classify_batch(golden, build_few_shot_examples(dev))
    result.to_csv("data/golden_eval_with_llm_predictions.csv", index=False)
    print(f"Saved {len(result)} predictions to data/golden_eval_with_llm_predictions.csv")
