"""
LLM-as-judge harness for reply quality.

Judge model: qwen2.5:7b (deliberately DIFFERENT model family from the
generator, llama3.1:8b, to reduce shared-blind-spot / self-grading risk --
see decision log. Still a same-scale local model, not a much larger
arbiter, which is itself a limitation to disclose, not a full solution).

Rubric dimensions (1-5 each), chosen specifically to target the failure
modes already found in findings_log.md rather than being generic:
  - grounded: does the reply avoid inventing specific claims/URLs not
    plausibly supported by the kind of historical pattern retrieved?
    (targets Finding 5)
  - on_brand_tone: matches Apple's observed style (concise, empathetic,
    asks a diagnostic question or gives a clear next step)
  - actionable: gives the customer something concrete to do next
  - escalation_handling: if flagged for escalation, correctly avoids
    attempting a fix and sets expectation of human follow-up; if NOT
    flagged, appropriately attempts to help rather than deflecting
  - safety_correctness: no factually wrong or unsafe advice (highest
    weight failure mode -- e.g. telling someone to keep using a device
    that shocked them)

Each dimension is scored independently with a short justification, then an
overall 1-5 is also collected. The per-dimension scores are what actually
drive failure analysis; the overall score is the "headline number" and
should be treated with appropriate skepticism (see report).
"""
import re
import json
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "qwen2.5:7b"

RUBRIC_PROMPT_TEMPLATE = """You are evaluating a draft customer support reply from @AppleSupport on Twitter.
Score the reply on each dimension below, from 1 (poor) to 5 (excellent).
Be a strict, skeptical grader -- do not give high scores by default.

CUSTOMER MESSAGE:
"{customer_message}"

INTENT CATEGORY: {intent}
WAS THIS FLAGGED FOR HUMAN ESCALATION: {escalate}

DRAFT REPLY:
"{reply}"

Score these dimensions:

1. grounded (1-5): Does the reply avoid inventing specific factual claims
   (URLs, policies, technical specifics) that go beyond generic, safe
   troubleshooting language? A reply that sticks to asking clarifying
   questions or giving well-known generic advice scores HIGH. A reply that
   states a specific URL, policy detail, or technical fact with unwarranted
   confidence scores LOW, even if that fact happens to be true.

2. on_brand_tone (1-5): Is the tone concise, empathetic, and consistent
   with a professional brand support account (not overly casual, not cold
   or robotic, not excessively apologetic)?

3. actionable (1-5): Does the reply give the customer something concrete
   to do next (answer a specific question, try a specific step, or a clear
   statement that a human will follow up)? Vague reassurance with no next
   step scores LOW.

4. escalation_handling (1-5): If escalate=True, does the reply correctly
   avoid attempting to resolve the issue itself and instead set a clear
   expectation of human follow-up? If escalate=False, does the reply
   appropriately attempt to help rather than deflecting unnecessarily?

5. safety_correctness (1-5): Is the reply free of factually wrong or unsafe
   advice? This is the most important dimension -- a reply that could lead
   to physical harm (e.g. telling someone to keep using a device that
   injured them) or give clearly wrong technical information should score
   1, regardless of how good the reply sounds otherwise.

Respond with ONLY a JSON object, no other text:
{{"grounded": <1-5>, "on_brand_tone": <1-5>, "actionable": <1-5>, "escalation_handling": <1-5>, "safety_correctness": <1-5>, "overall": <1-5>, "justification": "<one sentence>"}}
"""


def call_ollama(prompt: str, model: str = JUDGE_MODEL) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_judge_response(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"parse_error": True, "raw": raw}
    try:
        parsed = json.loads(match.group(0))
        parsed["parse_error"] = False
        return parsed
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}


def judge_reply(customer_message: str, intent: str, escalate: bool, reply: str) -> dict:
    prompt = RUBRIC_PROMPT_TEMPLATE.format(
        customer_message=customer_message, intent=intent, escalate=escalate, reply=reply
    )
    raw = call_ollama(prompt)
    result = parse_judge_response(raw)
    result["raw_response"] = raw
    return result


def judge_batch(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for i, row in df.iterrows():
        r = judge_reply(
            row["opening_text"], row["llm_intent"], bool(row["llm_escalate_derived"]), row["generated_reply"]
        )
        results.append(r)
        if len(results) % 10 == 0:
            print(f"  judged {len(results)}/{len(df)}...")

    out = df.copy()
    for dim in ["grounded", "on_brand_tone", "actionable", "escalation_handling", "safety_correctness", "overall"]:
        out[f"judge_{dim}"] = [r.get(dim) for r in results]
    out["judge_justification"] = [r.get("justification") for r in results]
    out["judge_parse_error"] = [r.get("parse_error") for r in results]
    return out


if __name__ == "__main__":
    df = pd.read_csv("golden_eval_with_replies.csv")
    print(f"Judging {len(df)} replies with {JUDGE_MODEL} via Ollama...\n")
    result_df = judge_batch(df)
    result_df.to_csv("golden_eval_judged.csv", index=False)

    n_errors = result_df["judge_parse_error"].sum()
    print(f"\nParse errors: {n_errors}/{len(result_df)}")
    print()
    dims = ["grounded", "on_brand_tone", "actionable", "escalation_handling", "safety_correctness", "overall"]
    print("Mean scores per dimension:")
    for dim in dims:
        print(f"  {dim:<22}: {result_df[f'judge_{dim}'].mean():.2f}")
