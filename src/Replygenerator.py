"""Retrieval-grounded Apple Support reply generation using Ollama qwen2.5:7b."""
from __future__ import annotations
import json
from urllib.request import Request, urlopen
from typing import Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
OLLAMA_URL, MODEL, CORPUS_PATH = "http://localhost:11434/api/generate", "qwen2.5:7b", "development_300_with_replies.csv"
def _as_bool(value: Any) -> bool: return str(value).strip().lower() in {"true","1","yes"}

class ReplyRetriever:
    """Development-only retrieval; golden data stays held out."""
    def __init__(self, corpus_path: str = CORPUS_PATH):
        corpus = pd.read_csv(corpus_path); missing = {"opening_text","agent_reply","label_intent"} - set(corpus.columns)
        if missing: raise ValueError(f"Reply corpus is missing columns: {sorted(missing)}")
        if "is_support_request" in corpus: corpus = corpus[corpus.is_support_request.map(_as_bool)]
        if "is_non_english" in corpus: corpus = corpus[~corpus.is_non_english.map(_as_bool)]
        self.corpus = corpus.dropna(subset=["opening_text","agent_reply"]).query("label_intent != 'out_of_scope'").reset_index(drop=True)
        self.corpus["source_id"] = self.corpus.get("thread_id", self.corpus.index).astype(str)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus.opening_text)
    def retrieve(self, message: str, intent: str, k: int = 3, min_same_intent: int = 2) -> pd.DataFrame:
        pool = self.corpus[self.corpus.label_intent == intent]; fallback = len(pool) < min_same_intent
        if fallback: pool = self.corpus
        indices = pool.index.to_numpy(); sims = cosine_similarity(self.vectorizer.transform([message]), self.matrix[indices]).ravel(); chosen = sims.argsort()[::-1][:min(k,len(indices))]
        result = self.corpus.loc[indices[chosen],["source_id","opening_text","agent_reply","label_intent"]].copy(); result["similarity"] = sims[chosen]; result["retrieval_scope"] = "global_fallback" if fallback else "same_intent"; return result.reset_index(drop=True)

def build_reply_prompt(message: str, intent: str, escalate: bool, reason: str, retrieved: pd.DataFrame) -> str:
    evidence = "\n\n".join(f"Example {i+1} (source {r.source_id}, similarity {r.similarity:.3f}):\nCustomer: {json.dumps(r.opening_text)}\nHistorical agent reply: {json.dumps(r.agent_reply)}" for i,r in enumerate(retrieved.itertuples()))
    policy = "A human specialist will handle this. Acknowledge and direct the customer to DM; do not diagnose, promise outcomes, or request sensitive data publicly." if escalate else "Give one concise next step. Use a DM invitation when supported by the evidence. Do not invent policies or troubleshooting absent from the message and evidence."
    return f"""Write a concise @AppleSupport reply (maximum 280 characters). Ground it in the retrieved historical examples: match their style and only use support actions consistent with them.
Intent: {intent}\nEscalation: {escalate}\nReason: {reason}\nPolicy: {policy}
Retrieved evidence:\n{evidence}
Customer message: {json.dumps(message)}
Return only the reply text."""
def call_ollama(prompt: str) -> str:
    payload = json.dumps({"model":MODEL,"prompt":prompt,"stream":False,"options":{"temperature":0.2}}).encode("utf-8")
    with urlopen(Request(OLLAMA_URL, data=payload, headers={"Content-Type":"application/json"}), timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))["response"].strip().strip('"')[:280]
def generate_reply(message: str, intent: str, escalate: bool, retriever: ReplyRetriever, escalation_reason: str = "", k: int = 3) -> dict[str,Any]:
    if intent == "out_of_scope": return {"reply":None,"grounding_sources":[],"note":"No reply drafted for out-of-scope input."}
    retrieved = retriever.retrieve(message,intent,k); prompt = build_reply_prompt(message,intent,escalate,escalation_reason,retrieved)
    return {"reply":call_ollama(prompt),"grounding_sources":retrieved.to_dict(orient="records"),"prompt":prompt}
