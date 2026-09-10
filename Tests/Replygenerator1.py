"""
Retrieval-grounded reply generator.

Design (see decision log):
  - Retrieval corpus: development_300_with_replies.csv (never golden --
    golden stays held out for eval only).
  - Retrieval method: TF-IDF cosine similarity over opening_text, restricted
    to same-intent examples first (falls back to global top-k if the
    predicted intent has too few examples, e.g. billing_subscription_purchase
    with only 2).
  - The generator is explicitly allowed to produce a DM-handoff-style reply
    when that's what similar real historical cases did -- this is scored as
    a correct, on-brand output, not a fallback/failure (see decision log:
    "DM-handoff as valid output").
  - service_complaint is excluded entirely -- escalate-only, no reply drafted
    (see labeling_guide_v2.md).
"""
import re
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

CORPUS_PATH = "development_300_with_replies.csv"


class ReplyRetriever:
    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus = pd.read_csv(corpus_path)
        # only retrieve from real, English, non-service_complaint examples
        self.corpus = self.corpus[
            (self.corpus["is_support_request"] == True)
            & (self.corpus["is_non_english"] == False)
            & (self.corpus["label_intent"] != "service_complaint")
        ].reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus["opening_text"])

    def retrieve(self, message: str, intent: str, k: int = 3, min_same_intent: int = 2) -> pd.DataFrame:
        same_intent = self.corpus[self.corpus["label_intent"] == intent]
        pool = same_intent if len(same_intent) >= min_same_intent else self.corpus

        pool_idx = pool.index
        query_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(query_vec, self.tfidf_matrix[pool_idx]).flatten()

        top_k_local = sims.argsort()[::-1][:k]
        top_k_global_idx = pool_idx[top_k_local]

        result = self.corpus.loc[top_k_global_idx].copy()
        result["similarity"] = sims[top_k_local]
        return result[["opening_text", "agent_reply", "label_intent", "similarity"]]


def build_reply_prompt(message: str, intent: str, escalate: bool, retrieved: pd.DataFrame) -> str:
    examples_block = "\n\n".join(
        f'Customer: "{row.opening_text}"\nApple Support replied: "{row.agent_reply}"'
        for row in retrieved.itertuples()
    )

    escalate_note = (
        "This message has been flagged for HUMAN ESCALATION. Your reply should "
        "acknowledge the issue empathetically and let the customer know a specialist "
        "will follow up -- do not attempt to resolve the issue yourself."
        if escalate
        else "This message can be auto-handled. Draft a helpful reply."
    )

    return f"""You are drafting a reply as @AppleSupport on Twitter, in Apple's actual observed
support style. Below are real examples of how Apple has replied to similar
past messages -- match this style and level of detail. If the historical
examples mostly redirect to DM, that is a legitimate pattern to follow, not
something to avoid.

{escalate_note}

Historical examples (intent: {intent}):
{examples_block}

Now draft a reply to this new message. Keep it under 280 characters, in
Apple's voice (helpful, concise, asks for specifics or device/iOS version
when relevant). Respond with ONLY the reply text, no other commentary.

Customer: "{message}"
Apple Support reply:"""


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate_reply(message: str, intent: str, escalate: bool, retriever: ReplyRetriever, k: int = 3) -> dict:
    if intent == "service_complaint":
        return {
            "reply": None,
            "retrieved_examples": [],
            "note": "service_complaint is escalate-only by design; no reply generated.",
        }
    retrieved = retriever.retrieve(message, intent, k=k)
    prompt = build_reply_prompt(message, intent, escalate, retrieved)
    reply_text = call_ollama(prompt)
    return {
        "reply": reply_text,
        "retrieved_examples": retrieved.to_dict(orient="records"),
        "prompt": prompt,
    }


if __name__ == "__main__":
    retriever = ReplyRetriever()
    print(f"Retrieval corpus size (after filtering): {len(retriever.corpus)}")
    print()

    golden = pd.read_csv("golden_eval_final_scored.csv")
    # skip service_complaint (no reply generated) and out_of_scope (not classified here)
    scoped = golden[~golden["label_intent"].isin(["service_complaint", "out_of_scope"])].copy()

    print(f"Generating replies for {len(scoped)} golden examples...\n")
    replies = []
    for i, row in scoped.iterrows():
        result = generate_reply(
            row["opening_text"], row["llm_intent"], bool(row["llm_escalate_derived"]), retriever
        )
        replies.append(result["reply"])
        if len(replies) % 10 == 0:
            print(f"  generated {len(replies)}/{len(scoped)}...")

    scoped["generated_reply"] = replies
    scoped.to_csv("golden_eval_with_replies.csv", index=False)
    print("\nSaved to golden_eval_with_replies.csv")