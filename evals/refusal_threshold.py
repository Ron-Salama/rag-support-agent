"""Choose the refusal threshold (ragagent.rag.answer.MIN_SCORE) from real retrieval scores.

    python -m evals.refusal_threshold

For each question we look only at the BEST chunk's similarity score (retrieval only, no LLM
calls, so it is free). Questions the documents answer should score high, off-topic questions
low; the threshold belongs in the gap between the two groups.

The third group is the hard case: questions that SOUND like they belong here (similar words)
but that the documents do not answer. Their scores overlap the answerable ones, so no
threshold can catch them - that is the job of gate 2 (the LLM says found=false).

The questions avoid the fields scored in Part 1 on purpose. Re-run this after changing the
chunking or the embedding model: the scores shift, so the threshold has to be re-checked.
This is a small probe (20 questions); the Part 4 golden set should replace it.
"""
from ragagent.rag.answer import MIN_SCORE
from ragagent.rag.retrieve import search

ANSWERABLE = [  # each checked by hand: the answer is printed in at least one document
    "What is the occupancy rate of the apartment complex?",
    "In what year was the apartment complex built?",
    "How many stories do the buildings have?",
    "What late charge applies if a loan payment is late?",
    "Can the borrower prepay the note, and is there a prepayment premium?",
    "What quarterly dividend did the company declare?",
    "What services did the law firm bill for?",
    "Who is the guarantor of the loan?",
    "What is the condition of the building's roof?",
    "What zoning applies to the property?",
]
OFF_TOPIC = [  # nothing to do with these documents: gate 1 (the score) should refuse
    "What is the capital of Australia?",
    "What is a good recipe for chocolate cake?",
    "Who won the 2022 football World Cup?",
    "What is the CEO's favorite color?",
    "How do I reset my email password?",
    "What is the weather forecast for tomorrow?",
    "What is the boiling point of water in Fahrenheit?",
    "Which programming language is best for beginners?",
]
ON_TOPIC_TRAPS = [  # sound related, but not answered by the documents: gate 2 (the LLM) must refuse
    "How many vacation days do employees get per year?",
    "What was Tesla's revenue in 2025?",
]


def best_scores(questions: list[str]) -> list[tuple[float, str, str]]:
    out = []
    for q in questions:
        top = search(q, k=1)[0]
        out.append((top.score, top.chunk_id, q))
    return sorted(out, reverse=True)


def main():
    groups = {"ANSWERABLE": ANSWERABLE, "OFF-TOPIC": OFF_TOPIC, "ON-TOPIC TRAPS": ON_TOPIC_TRAPS}
    lows_highs = {}
    for name, questions in groups.items():
        scores = best_scores(questions)
        print(f"\n{name}")
        for score, chunk_id, q in scores:
            gate = "passes gate 1" if score >= MIN_SCORE else "REFUSED by gate 1"
            print(f"  {score:.3f}  {gate:<18} {chunk_id:<32} {q}")
        lows_highs[name] = (scores[-1][0], scores[0][0])
        print(f"  range {scores[-1][0]:.3f} - {scores[0][0]:.3f}")
    lowest_answerable, highest_off_topic = lows_highs["ANSWERABLE"][0], lows_highs["OFF-TOPIC"][1]
    print(f"\nGap between groups: highest off-topic {highest_off_topic:.3f} < lowest answerable {lowest_answerable:.3f}"
          if highest_off_topic < lowest_answerable else
          f"\nNO GAP: an off-topic question ({highest_off_topic:.3f}) beats an answerable one ({lowest_answerable:.3f})")
    print(f"Current MIN_SCORE = {MIN_SCORE}")


if __name__ == "__main__":
    main()
