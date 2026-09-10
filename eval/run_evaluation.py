import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.pipeline import run


INPUT_FILE = Path("golden_eval_200.csv")
OUTPUT_FILE = Path("final_predictions.jsonl")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in tqdm(
        gold.iterrows(),
        total=len(gold),
        desc="Running pipeline"
    ):
        prediction = run(row["opening_text"])

        # Ensure the output is a normal dictionary before adding evaluation fields.
        prediction = dict(prediction)

        prediction.update(
            {
                "thread_id": row["thread_id"],
                "opening_text": row["opening_text"],
                "gold_intent": row["label_intent"],
                "gold_escalate": row["label_escalate"],
                "gold_reason": row["label_reason"],
            }
        )

        results.append(prediction)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    print(f"Saved {len(results)} predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()