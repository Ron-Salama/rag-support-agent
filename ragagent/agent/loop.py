"""The agent loop: the model decides which tools to call, our code runs them, until it answers.

    python -m ragagent.agent.loop --selftest     # test the agent loop with a scripted fake LLM (no API calls)
    python -m ragagent.agent.loop "question"     # run the worker alone (no verifier) and print its steps

The flow (one "step" = one LLM call):

    messages = [question]
        │
        ├──> llm.chat(SYSTEM, messages, tool specs) ──> reply
        │        reply has no tool calls?   yes ──> that text is the FINAL ANSWER, stop ("answered")
        │                                   no
        │    run every tool it asked for:   unknown tool / bad arguments / tool crashed
        │                                   ──> the ERROR goes back to the model as the result (never crash)
        │    append the model's turn + one "tool" message per result, go round again
        │
        └── after max_steps LLM calls without a final answer ──> stop ("step_limit"), safe answer

Why a step limit? A model can get stuck (searching the same thing forever). Every step costs an
LLM call - real money on a paid API, free-tier quota here - so the loop must always end.
Why feed errors back instead of raising? The model can often fix its own mistake ("unknown
doc_id - call list_documents") on the next step. A crash would throw the whole question away.
"""
import json
import sys
from dataclasses import dataclass, field

from ragagent.agent.tools import Tool, sources_in
from ragagent.llm import ChatReply, LLMError, ToolCall

MAX_STEPS = 6   # LLM calls per question, at most - see DECISIONS.md
NOT_FOUND = "I could not find this in the documents."
COULD_NOT_COMPLETE = "I could not complete this question (the agent stopped before it had an answer)."

SYSTEM = """You answer questions about a small set of public financial documents (loan notes, appraisals,
financial statements, invoices). You have tools to search and read them.
Rules:
- Always use the tools to find evidence before you answer. If a search finds nothing relevant, search again
  with other words or read the page around a promising hit with get_page.
- Use ONLY what the tools returned. Never use outside knowledge and never guess.
- Cite the source of every fact right after it, using its source_id in square brackets, e.g. [loan_note_jshighland p.2].
- If the tools do not give you the answer, reply exactly: I could not find this in the documents.
- If documents disagree, say which document says what, citing each.
- Answer in 1-4 short sentences. Copy numbers exactly as printed."""


@dataclass
class AgentResult:
    answer: str
    steps: list[dict]                  # trace: {"step", "tool", "args", "result"} for every tool call
    stopped_reason: str                # "answered" | "step_limit" | "llm_error"
    sources_used: list[str] = field(default_factory=list)   # source_ids the tools returned = what the model saw
    error: str | None = None           # the LLMError message when stopped_reason is "llm_error"


# ===================================================================================
# Core loop - kept small and framework-free on purpose; offline self-test:
#   python -m ragagent.agent.loop --selftest
# ===================================================================================
def run_agent(llm, question: str, tools: dict[str, Tool], max_steps: int = MAX_STEPS) -> AgentResult:
    """Answer `question` by letting the model call `tools` for at most `max_steps` LLM calls.

    Returns an AgentResult whose stopped_reason is "answered" (a reply with no tool calls is the final
    answer), "step_limit" (COULD_NOT_COMPLETE after max_steps calls) or "llm_error" (an LLMError, with
    its message in `error`). Unknown tools, bad arguments and tool exceptions never raise: they go back
    to the model as {"error": ...} results. Every tool call is recorded in `steps`. DECISIONS.md C11.
    """
    messages = [{"role": "user", "content": question}]
    specs = [tool.spec for tool in tools.values()]
    steps = []
    for step in range(1, max_steps + 1):
        try:
            reply = llm.chat(SYSTEM, messages, specs)
        except LLMError as e:  # bad key, quota used up, server down after retries: stop, but don't crash
            return AgentResult(COULD_NOT_COMPLETE, steps, "llm_error", sources_in(steps), error=str(e))
        if not reply.tool_calls:  # no tool requested = this is the final answer
            return AgentResult((reply.text or "").strip(), steps, "answered", sources_in(steps))

        messages.append(reply.as_message())
        for call in reply.tool_calls:
            tool = tools.get(call.name)
            try:
                if tool is None:
                    result = {"error": f"unknown tool {call.name!r} - the tools are: {', '.join(tools)}"}
                else:
                    result = tool.fn(**call.args)
            except Exception as e:  # noqa: BLE001 - wrong arguments (TypeError) or a bug in the tool: tell the model
                result = {"error": f"{type(e).__name__}: {e}"}
            steps.append({"step": step, "tool": call.name, "args": call.args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})

    return AgentResult(COULD_NOT_COMPLETE, steps, "step_limit", sources_in(steps))


# ----------------------------------------------------------------- self-test with a scripted fake LLM
class FakeChatLLM:
    """Plays back scripted ChatReplys (or raises a scripted exception) and records every request."""
    model = "fake"

    def __init__(self, replies: list):
        self.replies, self.requests = list(replies), []

    def chat(self, system, messages, tools):
        self.requests.append([dict(m) for m in messages])   # a copy: the loop keeps appending to the same list
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fake_tools() -> dict[str, Tool]:
    """A search tool that always finds page 2 of 'loan_x', and a tool that always crashes."""
    def search_docs(query: str, doc_id: str | None = None, k: int = 5) -> dict:
        return {"query": query, "sources": [{"source_id": "loan_x p.2", "doc_id": "loan_x", "page": 2,
                                             "text": "A late charge of 4% of the overdue payment applies."}]}

    def broken(**kwargs) -> dict:
        raise RuntimeError("disk on fire")

    params = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    return {"search_docs": Tool("search_docs", "Search the documents.", params, search_docs),
            "broken": Tool("broken", "Always fails.", {"type": "object", "properties": {}}, broken)}


def tool_call(name: str, args: dict, id_: str = "c1") -> ChatReply:
    return ChatReply(text=None, tool_calls=[ToolCall(id_, name, args)])


def final(text: str) -> ChatReply:
    return ChatReply(text=text)


def selftest():
    search = tool_call("search_docs", {"query": "late charge"})
    answer = final("The late charge is 4% of the overdue payment [loan_x p.2].")
    two_calls = ChatReply(text=None, tool_calls=[ToolCall("a", "search_docs", {"query": "late charge"}),
                                                 ToolCall("b", "broken", {})])

    def history_ok(fake: FakeChatLLM) -> bool:   # 2nd request = question, model's turn, then the tool result
        roles = [m["role"] for m in fake.requests[1]]
        tool_msg = fake.requests[1][2]
        return roles == ["user", "assistant", "tool"] and tool_msg["tool_call_id"] == "c1" and \
            tool_msg["name"] == "search_docs" and "4%" in tool_msg["content"]

    cases = [  # name, scripted replies, max_steps, check(result, fake) -> bool
        ("direct answer, no tools", [final(NOT_FOUND)], 6,
         lambda r, f: r.stopped_reason == "answered" and r.answer == NOT_FOUND and r.steps == [] and len(f.requests) == 1),
        ("one tool call, then answer", [search, answer], 6,
         lambda r, f: r.stopped_reason == "answered" and "4%" in r.answer and len(r.steps) == 1
         and r.sources_used == ["loan_x p.2"] and history_ok(f)),
        ("unknown tool -> error fed back, recovers", [tool_call("delete_everything", {}), answer], 6,
         lambda r, f: r.stopped_reason == "answered" and "unknown tool" in r.steps[0]["result"]["error"]
         and "unknown tool" in f.requests[1][-1]["content"]),
        ("bad arguments -> error fed back, recovers", [tool_call("search_docs", {"qeury": "typo"}), answer], 6,
         lambda r, f: r.stopped_reason == "answered" and r.steps[0]["result"]["error"].startswith("TypeError")),
        ("tool raises -> error fed back, recovers", [tool_call("broken", {}), answer], 6,
         lambda r, f: r.stopped_reason == "answered" and "disk on fire" in r.steps[0]["result"]["error"]),
        # Gemini may ask for several tools in ONE turn: run them all, one "tool" message per call, same step number
        ("two tool calls in one turn -> both run", [two_calls, answer], 6,
         lambda r, f: r.stopped_reason == "answered" and [s["step"] for s in r.steps] == [1, 1]
         and [m["role"] for m in f.requests[1]] == ["user", "assistant", "tool", "tool"]
         and [m["tool_call_id"] for m in f.requests[1][2:]] == ["a", "b"] and "error" in r.steps[1]["result"]),
        ("endless tool calls -> step_limit", [search] * 10, 3,
         lambda r, f: r.stopped_reason == "step_limit" and r.answer == COULD_NOT_COMPLETE
         and len(f.requests) == 3 and [s["step"] for s in r.steps] == [1, 2, 3] and r.sources_used == ["loan_x p.2"]),
        ("LLM fails -> llm_error, no crash", [search, LLMError("daily quota used up")], 6,
         lambda r, f: r.stopped_reason == "llm_error" and "quota" in r.error and len(r.steps) == 1),
    ]
    passed = 0
    for name, replies, max_steps, check in cases:
        fake = FakeChatLLM(replies)
        try:
            r = run_agent(fake, "What late charge applies?", fake_tools(), max_steps=max_steps)
            ok, got = check(r, fake), f"stopped={r.stopped_reason} steps={len(r.steps)} answer={r.answer[:40]!r}"
        except Exception as e:  # noqa: BLE001 - show any crash as a failed case
            ok, got = False, f"crashed: {type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<42} | {got}")
    print(f"\n{passed}/{len(cases)} passed")


def show(result: AgentResult):
    for s in result.steps:
        found = s["result"].get("error") or ", ".join(x["source_id"] for x in s["result"].get("sources", [])) \
            or s["result"].get("source_id", "")
        print(f"  step {s['step']}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)}) -> {found}")
    print(f"\n[{result.stopped_reason}] {result.answer}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")   # don't crash on '§' etc. when output is piped (see retrieve.py)
    args = sys.argv[1:]
    if args == ["--selftest"]:
        selftest()
    elif args:
        from ragagent.agent.tools import registry
        from ragagent.llm import get_llm
        show(run_agent(get_llm(), " ".join(args), registry()))
    else:
        print(__doc__)
