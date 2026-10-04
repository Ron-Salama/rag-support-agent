"""LLM-as-judge: a second model grades the system's answers, so a whole golden set can be scored
without a person reading every answer. Used by evals/answer_eval.py; the prompt lives here so it
can be reused (and so evals/judge_agreement.py checks the SAME judge that produced the numbers).

    python -m evals.judge --selftest     # prompt building + reply parsing with a fake LLM (no API calls)

Why a judge? "Correct" for free text is not string equality: "Net 30 days" and "payment is due
within 30 days" are the same answer. Comparing MEANING needs a model.
Why not trust it blindly? It is an instrument, and instruments can be off. Check it against
your own verdicts on a sample (evals/judge_agreement.py) before believing its percentages.

It answers two separate questions, because they fail separately:
  correct   does the answer say what the reference says? (numbers, names, dates)
  faithful  is EVERY claim in the answer backed by the sources it cites?
An answer can be correct but unfaithful (right number, but from memory or a lucky guess) or
faithful but wrong (it quoted a real clause - from the WRONG document).
For a trap the reference says "the documents do not contain this", so correct = the system said
so instead of inventing an answer.

Choices (DECISIONS.md C15): the judge is a DIFFERENT model than the one answering (its own free
quota, and models tend to favour text like their own); yes/no verdicts instead of a 1-5 score
(easy to compare with a person and to turn into a rate); "reasoning" comes FIRST in the JSON so
the model writes its comparison before it commits to a verdict.
"""
import sys

from pydantic import BaseModel, Field, ValidationError

from ragagent import config

JUDGE_SYSTEM = """You grade answers produced by a question-answering system over financial documents.
You get: the question, a REFERENCE saying what a correct response contains, the system's answer,
and the source passages the answer cites ([S1], [S2], ...).

Decide two things independently:
- correct: true if the answer agrees with the reference on every fact that matters (numbers, names,
  dates, yes/no). Different wording or extra true detail is fine. A missing or different key fact is not.
- faithful: true if EVERY factual claim in the answer is stated in the cited passages. Judge only
  against the passages - not against the reference and not against your own knowledge. A claim the
  passages do not state is unsupported, even if it happens to be true. With no passages, the answer
  is faithful only if it makes no factual claim.

Write your reasoning first (1-3 short sentences), then the two verdicts."""


class Verdict(BaseModel):
    """The JSON shape the judge must return. Field order matters: reasoning is written first."""
    reasoning: str = Field(description="1-3 sentences comparing the answer with the reference and with the passages.")
    correct: bool = Field(description="The answer agrees with the reference on every fact that matters.")
    faithful: bool = Field(description="Every factual claim in the answer is stated in the cited passages.")


def reference(item: dict) -> str:
    """What a correct response must contain - from the golden item."""
    if item["type"] == "trap":
        return (f"The documents do NOT contain the answer to this question ({item['why_unanswerable']}). "
                "A correct response says the information is not available in the documents. A response that "
                "gives a specific value or otherwise claims to answer the question is NOT correct.")
    return f"A correct answer says: {item['answer']}\n(The document says: \"{item['quote']}\")"


def source_name(s: dict) -> str:
    """How the answer refers to a source: [S2] in plain RAG, [loan_x p.2] (its source_id) in the agent."""
    return s.get("source_id") or f"S{s['n']}"


def build_prompt(item: dict, answer_text: str, sources: list[dict]) -> str:
    """sources = the CITED passages: [{"n" or "source_id", "doc_id", "page", "text"}], named like the answer's marks."""
    passages = "\n\n".join(f"[{source_name(s)}] ({s['doc_id']}, page {s['page']})\n{s['text']}" for s in sources)
    return (f"Question: {item['question']}\n\nREFERENCE: {reference(item)}\n\n"
            f"System's answer: {answer_text}\n\nCited passages:\n{passages or '(none)'}")


def judge(llm, item: dict, answer_text: str, sources: list[dict]) -> Verdict | None:
    """One judge call. None if the judge's reply is not valid JSON (counted as a judge error, not a verdict)."""
    raw = llm.generate_json(JUDGE_SYSTEM, build_prompt(item, answer_text, sources), Verdict)
    try:
        return Verdict.model_validate_json(raw)
    except ValidationError:
        return None


def get_judge_llm():
    """The judge model: GEMINI_JUDGE_MODEL on Gemini (a separate ~500/day quota); on Ollama the same local model."""
    from ragagent import llm

    if config.LLM_PROVIDER == "gemini":
        return llm.GeminiLLM(config.GEMINI_JUDGE_MODEL)
    return llm.get_llm()


# ----------------------------------------------------------------- self-test with a fake LLM
def selftest():
    from ragagent.extract import FakeLLM   # plays back scripted answers, counts calls

    class RecordingLLM(FakeLLM):
        """Also remembers the last prompt, so we can check what the judge was shown."""
        def generate_json(self, system, user, schema):
            self.user = user
            return super().generate_json(system, user, schema)

    item = {"id": "a1", "type": "answerable", "question": "What are the payment terms?",
            "answer": "Net 30 days", "quote": "Payment Terms: Net 30 days"}
    trap = {"id": "t1", "type": "trap", "question": "What is the warranty?", "why_unanswerable": "no warranty printed"}
    sources = [{"n": 2, "doc_id": "inv_x", "page": 1, "text": "Payment Terms: Net 30 days"}]
    ok_reply = Verdict(reasoning="Matches.", correct=True, faithful=True).model_dump_json()

    fake = RecordingLLM([ok_reply])
    v = judge(fake, item, "Net 30 days [S2].", sources)
    fake_trap = RecordingLLM([ok_reply])
    judge(fake_trap, trap, "Not in the documents.", [])
    fake_agent = RecordingLLM([ok_reply])
    judge(fake_agent, item, "Net 30 days [inv_x p.1].", [{**sources[0], "n": None, "source_id": "inv_x p.1"}])
    cases = [
        ("valid reply -> Verdict", v is not None and v.correct and v.faithful and fake.calls == 1),
        ("prompt has reference + quote", "Net 30 days" in fake.user and "Payment Terms: Net 30 days\")" in fake.user),
        ("sources keep their [S2] number", "[S2] (inv_x, page 1)" in fake.user),
        ("agent page citations named by source_id", "[inv_x p.1] (inv_x, page 1)" in fake_agent.user),
        ("trap prompt says NOT answerable", "do NOT contain" in fake_trap.user and "(none)" in fake_trap.user),
        ("invalid JSON -> None", judge(RecordingLLM(["The answer looks right to me."]), item, "x", []) is None),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(ok for _, ok in cases)}/{len(cases)} passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(__doc__)
