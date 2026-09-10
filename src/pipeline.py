"""End-to-end classify -> escalation -> retrieve -> generate pipeline."""
from __future__ import annotations
import json
import pandas as pd
from classifierllm import build_few_shot_examples, classify
from escalation import decide_escalation
from Replygenerator import ReplyRetriever, generate_reply
class SupportPipeline:
    def __init__(self, development_path="development_300.csv", reply_corpus_path="development_300_with_replies.csv"):
        self.few_shot = build_few_shot_examples(pd.read_csv(development_path)); self.retriever = ReplyRetriever(reply_corpus_path)
    def run(self, message: str) -> dict:
        classification = classify(message,self.few_shot)
        policy = decide_escalation(message,classification.get("intent") or "other_unclear",classification.get("confidence") or "low",classification.get("parse_error",True),classification.get("is_multi_issue",False))
        generation = generate_reply(message,classification.get("intent") or "other_unclear",policy["escalate"],self.retriever,policy["reason"])
        return {"classification":classification,"escalation":policy,"reply":generation.get("reply"),"grounding_sources":generation.get("grounding_sources",[])}
if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("message"); args=parser.parse_args(); print(json.dumps(SupportPipeline().run(args.message),indent=2,ensure_ascii=False))
