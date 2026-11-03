"""Run 10 test questions against POST /ask and classify every wrong answer."""
import time

from fastapi.testclient import TestClient

import rag
from main import app

# >>> EDIT THESE 6 to match YOUR manual: (question, keyword that must appear in the correct answer)
ANSWERABLE = [
    ("How do I reset the device to factory settings?", "reset"),
    ("What is the warranty period?", "year"),
    ("How long does the battery take to charge fully?", "hour"),
    ("What should I do if the device will not turn on?", "power"),
    ("How do I update the firmware?", "update"),
    ("What does the blinking red light mean?", "red"),
]

# These must return "I don't know." (not in any product manual)
UNANSWERABLE = [
    "Who is the CEO of the company?",
    "What is the price of the product in Japan?",
    "Does this product work underwater at 100 metres?",
    "What is the capital of Australia?",
]

IDK = "I don't know."


def main():
    wrong = []
    with TestClient(app) as client:
        for kind, items in (("ANSWERABLE", ANSWERABLE), ("UNANSWERABLE", UNANSWERABLE)):
            for item in items:
                q, keyword = item if kind == "ANSWERABLE" else (item, None)
                r = client.post("/ask", json={"question": q}).json()
                answer = r["answer"]
                chunks = rag.retrieve(q)
                context = " ".join(c["text"] for c in chunks).lower()

                if kind == "ANSWERABLE":
                    ok = answer != IDK and keyword.lower() in answer.lower()
                    cause = ("generation failure (answer was in retrieved chunks)"
                             if keyword.lower() in context else
                             "retrieval failure (answer not in top-3 chunks)")
                else:
                    ok = answer == IDK
                    cause = "generation failure (model answered despite no support)"

                print(f"\n[{kind}] {q}\n  -> {r}\n  retrieved pages: {[c['page'] for c in chunks]}"
                      f"\n  {'PASS' if ok else 'FAIL: ' + cause}")
                if not ok:
                    wrong.append((q, cause))
                time.sleep(10)  # stay under the free-tier 15 requests/min limit

    print(f"\n==== {10 - len(wrong)}/10 correct ====")
    for q, cause in wrong:
        print(f"WRONG: {q}  ->  {cause}")


if __name__ == "__main__":
    main()