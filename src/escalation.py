"""Inspectable, conservative escalation policy for the Apple support pipeline."""
from __future__ import annotations
from typing import Any
ALWAYS_ESCALATE = {"account_access_security","billing_subscription_purchase","data_backup_loss","hardware_physical","service_complaint"}
RISK_TERMS = {"safety_risk":("fire","smoke","burn","shock","swollen","exploded","injury","overheat"),"security_risk":("hacked","phishing","scam","fraud","unauthorized","stolen"),"legal_or_regulatory":("lawyer","legal","sue","lawsuit","police","regulator"),"urgent_data_loss":("lost all","deleted all","cannot recover","irreplaceable")}
def decide_escalation(message: str, intent: str, confidence: str = "medium", parse_error: bool = False, is_multi_issue: bool = False) -> dict[str,Any]:
    text, reasons = message.lower(), []
    if parse_error: reasons.append("classifier_parse_error")
    if confidence == "low": reasons.append("low_classifier_confidence")
    if intent in ALWAYS_ESCALATE: reasons.append(f"high_risk_intent:{intent}")
    for label,terms in RISK_TERMS.items():
        hits = sorted({term for term in terms if term in text})
        if hits: reasons.append(f"{label}:"+",".join(hits))
    if is_multi_issue and intent == "other_unclear": reasons.append("multi_issue_unclear_routing")
    return {"escalate":bool(reasons),"reason":"; ".join(reasons) if reasons else "auto_handle: no high-risk trigger","reason_codes":reasons}
