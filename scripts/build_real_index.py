import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


DATA_DIR = Path(".")
QUESTIONS_FILE = DATA_DIR / "Questions.csv"
ANSWERS_FILE = DATA_DIR / "Answers.csv"
OUTPUT_FILE = DATA_DIR / "data" / "processed_qa.jsonl"


def clean_html(text: str) -> str:
    soup = BeautifulSoup(str(text), "html.parser")
    for code in soup.find_all("code"):
        code.string = f"\n```\n{code.get_text()}\n```\n"
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def build_dataset_file():
    if not QUESTIONS_FILE.exists() or not ANSWERS_FILE.exists():
        raise FileNotFoundError("Expected Questions.csv and Answers.csv in the project root.")

    questions = pd.read_csv(QUESTIONS_FILE, encoding="latin-1")
    answers = pd.read_csv(ANSWERS_FILE, encoding="latin-1")

    questions = questions[questions["Score"] >= 5].copy()
    answers = answers[answers["Score"] >= 3].copy()

    best_answers = (
        answers.sort_values("Score", ascending=False)
        .groupby("ParentId", as_index=False)
        .first()
    )

    merged = questions.merge(
        best_answers[["ParentId", "Body", "Score"]],
        left_on="Id",
        right_on="ParentId",
        how="inner",
        suffixes=("_question", "_answer"),
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            question_text = clean_html(str(row.get("Title", "")) + " " + str(row.get("Body_question", "")))
            answer_text = clean_html(str(row.get("Body_answer", "")))
            item = {
                "id": str(row.get("Id", "")),
                "title": str(row.get("Title", "")),
                "question": question_text[:2000],
                "answer": answer_text[:3000],
                "score": int(row.get("Score_answer", 0)),
            }
            f.write(json.dumps(item) + "\n")
            written += 1

    print(f"Wrote {written} processed QA records to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset_file()
