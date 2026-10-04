"""Talking to an LLM: one small interface, two interchangeable backends.

The rest of the project only ever calls:

    llm = get_llm()
    raw_json_text = llm.generate_json(system_prompt, user_prompt, SomePydanticModel)   # Parts 1, 2, 4
    reply = llm.chat(system_prompt, messages, tools)                                  # Part 3: tool calling

and never knows which provider is behind it. Moving to AWS Bedrock (Claude) later means
writing one more class with the same methods - nothing else changes.

    python -m ragagent.llm --selftest    # checks the chat() translations with fake clients (no API calls)

TOOL CALLING (chat). The model never runs anything itself: it only ASKS for a tool call
("call search_docs with query='late charge'"). Our code runs the tool and sends the result
back in the next request. Every provider has its own message format for this, so the agent
uses ONE neutral format and each class below translates it:

    tools    = [{"name": "search_docs", "description": "...", "parameters": {JSON schema of the arguments}}]
    messages = [
        {"role": "user", "content": "What late charge applies?"},
        {"role": "assistant", "content": "", "raw": <the provider's own copy of this turn>,   # = reply.as_message()
         "tool_calls": [{"id": "c1", "name": "search_docs", "args": {"query": "late charge"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search_docs", "content": '{"sources": [...]}'},
    ]
    reply = llm.chat(system, messages, tools)  ->  ChatReply(text, tool_calls=[ToolCall(id, name, args)], raw)
    A reply with no tool_calls is the model's final answer.

  - Gemini: an assistant turn can carry hidden "thought signatures" that must come back exactly
    as they were, so "raw" keeps Gemini's own Content object and we send THAT back unchanged.
    Tool results go back as FunctionResponse parts with the same id and name as the call.
  - Ollama: tools=..., the calls are in the reply's message.tool_calls, results go back as
    role "tool" messages. Ollama gives calls no id (ToolCall.id = ""): results match by order + name.
  - AWS Bedrock's Converse API has the same shape, 1:1:
        tools[i]                 -> toolConfig.tools[i].toolSpec {name, description, inputSchema: {json: parameters}}
        system                   -> system=[{"text": system}]
        assistant tool_calls[i]  -> content block {"toolUse": {toolUseId: id, name, input: args}}
        role "tool" message      -> content block {"toolResult": {toolUseId: tool_call_id, content: [{"json": ...}]}},
                                    all results of one turn together in ONE "user" message (same as Gemini)
        reply.tool_calls empty   <- stopReason "end_turn"   (calls present <- stopReason "tool_use")
    so a BedrockLLM class would be one more chat() translation; the agent loop would not change.

This file also handles the FIRST kind of retry: transport problems (rate limit 429,
server overloaded 503). Those are "the phone line was busy" errors - wait, then call
again, waiting longer each time (exponential backoff). The SECOND kind of retry - "the
model answered, but the answer is wrong/invalid" - lives in extract.py, because only
that code knows what a valid answer looks like.
"""
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel

from ragagent import config

CACHE_DIR = config.ROOT / ".cache" / "llm"
_last_call: dict[str, float] = {}


def cached(key_parts: dict, call) -> str:
    """Same exact request twice -> answer from disk, no API call.

    Re-running an eval after changing only the scoring code then costs zero quota.
    Change the prompt, the model or the document and the key changes, so it calls for real.
    Turn off with LLM_CACHE=0 in .env (e.g. to measure how much answers vary run to run).
    """
    if not config.LLM_CACHE:
        return call()
    key = hashlib.sha256(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()[:32]
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    text = call()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": key_parts, "text": text}), encoding="utf-8")
    return text


def pace(model: str, per_minute: int):
    """Free tier allows ~15 requests/minute: wait so calls are at least 60/per_minute seconds apart."""
    wait = _last_call.get(model, 0.0) + 60.0 / per_minute - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[model] = time.time()


class LLMError(Exception):
    """Permanent failure: bad key, bad request, daily quota used up. Retrying won't help."""


class TransientLLMError(LLMError):
    """Temporary failure: rate limit / overload / timeout. Worth waiting and retrying."""


def with_backoff(call, attempts: int = 5, base_seconds: float = 2.0):
    """Run call(); on a TransientLLMError wait 2s, 4s, 8s, ... (+ a little randomness) and retry."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except TransientLLMError as e:
            if attempt == attempts:
                raise
            wait = base_seconds * 2 ** (attempt - 1) + random.random()
            print(f"  [llm] {e} -> waiting {wait:.0f}s, then retry {attempt + 1}/{attempts}")
            time.sleep(wait)


# ----------------------------------------------------------------- tool calling: the neutral types
@dataclass
class ToolCall:
    id: str          # the provider's id for this call ("" if it gives none); the result must carry it back
    name: str        # which tool the model wants
    args: dict       # the arguments, already parsed from JSON into a dict


@dataclass
class ChatReply:
    text: str | None                                   # the model's words (None if it only asked for tools)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None                                 # the provider's own copy of this turn (Gemini: types.Content)

    def as_message(self) -> dict:
        """This reply as an assistant message for the history - keeps raw, so Gemini gets its turn back unchanged."""
        return {"role": "assistant", "content": self.text or "",
                "tool_calls": [asdict(c) for c in self.tool_calls], "raw": self.raw}


def chat_cache_key(provider: str, model: str, system: str, messages: list[dict], tools: list[dict]) -> dict:
    """What identifies a chat request. "raw" is left out: it is the provider's copy of an assistant
    turn whose text and tool calls are already in the key (and Gemini's object is not plain JSON)."""
    return {"provider": provider, "model": model, "system": system, "tools": tools,
            "messages": [{k: v for k, v in m.items() if k != "raw"} for m in messages]}


def _as_dict(text: str) -> dict:
    """A tool result (JSON text) as a dict - Gemini wants FunctionResponse.response to be an object."""
    try:
        value = json.loads(text)
    except ValueError:
        return {"output": text}
    return value if isinstance(value, dict) else {"output": value}


def _gemini_error(e) -> LLMError:
    """Sort a Gemini APIError into permanent (LLMError) or worth waiting for (TransientLLMError)."""
    msg = f"{e} {e.details}"
    if e.code == 429 and "PerDay" in msg:
        return LLMError("Gemini daily free quota used up - try tomorrow or LLM_PROVIDER=ollama")
    if e.code in (429, 500, 503, 504):
        return TransientLLMError(f"Gemini {e.code} {e.status}")
    return LLMError(msg)


def to_gemini(messages: list[dict]) -> list:
    """Neutral messages -> Gemini Contents (roles "user" and "model"; tool results are "user" turns)."""
    from google.genai import types

    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
        elif m["role"] == "assistant":
            if isinstance(m.get("raw"), types.Content):
                contents.append(m["raw"])  # exactly what Gemini sent us, thought signatures included
                continue
            parts = [types.Part.from_text(text=m["content"])] if m.get("content") else []
            parts += [types.Part(function_call=types.FunctionCall(id=c["id"] or None, name=c["name"], args=c["args"]))
                      for c in m.get("tool_calls", [])]
            contents.append(types.Content(role="model", parts=parts))
        elif m["role"] == "tool":
            part = types.Part(function_response=types.FunctionResponse(
                id=m.get("tool_call_id") or None, name=m["name"], response=_as_dict(m["content"])))
            previous = contents[-1] if contents else None
            if previous is not None and previous.role == "user" and previous.parts[0].function_response:
                previous.parts.append(part)  # all answers to one turn's calls go back in ONE Content
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return contents


def to_ollama(messages: list[dict]) -> list[dict]:
    """Neutral messages -> Ollama's chat messages (the system prompt is added by the caller)."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": m.get("content", ""),
                        "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}}
                                       for c in m.get("tool_calls", [])]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


class GeminiLLM:
    def __init__(self, model: str = config.GEMINI_MODEL, client=None):
        from google import genai

        # genai.Client() reads GEMINI_API_KEY from the environment (.env); tests pass a fake client instead.
        # With NO key (e.g. `docker run` without --env-file .env) it raises ValueError: we turn that into
        # LLMError, so the API answers 503 "LLM unavailable" with this hint instead of crashing with a 500.
        try:
            self.client = client or genai.Client()
        except ValueError as e:
            raise LLMError(f"no Gemini API key - put GEMINI_API_KEY in .env (Docker: docker run --env-file .env ...). {e}") from e
        self.model = model

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> str:
        import httpx
        from google.genai import errors, types

        # No temperature here: Gemini 3.5+ ignores it (and Google says future models will reject it).
        # So the same input can give slightly different answers - which is exactly why we validate
        # every answer and measure accuracy over many documents instead of trusting one run.
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),  # "structured output": model must fill this shape
            # We never let the SDK run tools for us; the agent loop (Part 3, agent/loop.py) does that in our own code.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        def call() -> str:
            pace(self.model, config.GEMINI_RPM)
            try:
                resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
            except errors.APIError as e:
                raise _gemini_error(e) from e
            except httpx.TransportError as e:   # no internet / timeout: not an APIError, but also "line busy"
                raise TransientLLMError(f"Gemini not reachable ({type(e).__name__})") from e
            return resp.text or ""

        key = {"provider": "gemini", "model": self.model, "system": system, "user": user,
               "schema": schema.model_json_schema()}
        return cached(key, lambda: with_backoff(call))

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> ChatReply:
        """One tool-calling turn (see the module docstring for the message format)."""
        import httpx
        from google.genai import errors, types

        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=t["parameters"]) for t in tools]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            # The SDK could run Python functions for us; we switch that off - the agent loop is ours.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        def call() -> str:
            pace(self.model, config.GEMINI_RPM)
            try:
                resp = self.client.models.generate_content(model=self.model, contents=to_gemini(messages), config=cfg)
            except errors.APIError as e:
                raise _gemini_error(e) from e
            except httpx.TransportError as e:   # no internet / timeout: not an APIError, but also "line busy"
                raise TransientLLMError(f"Gemini not reachable ({type(e).__name__})") from e
            content = resp.candidates[0].content if resp.candidates else None
            parts = (content.parts or []) if content else []
            text = "".join(p.text for p in parts if p.text and not p.thought) or None   # skip "thinking" text
            calls = [{"id": p.function_call.id or "", "name": p.function_call.name, "args": dict(p.function_call.args or {})}
                     for p in parts if p.function_call]
            if not text and not calls:  # blocked / malformed reply: fail loudly, and don't cache it
                reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
                raise LLMError(f"Gemini returned an empty reply (finish_reason={reason})")
            # Returned as JSON text so cached() can store it; the Content keeps its thought signatures (base64)
            return json.dumps({"text": text, "tool_calls": calls, "raw": content.model_dump(mode="json", exclude_none=True)})

        key = chat_cache_key("gemini", self.model, system, messages, tools)
        out = json.loads(cached(key, lambda: with_backoff(call)))
        return ChatReply(out["text"], [ToolCall(**c) for c in out["tool_calls"]], types.Content.model_validate(out["raw"]))


class OllamaLLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, client=None):
        import ollama

        # ollama.Client() talks to the local Ollama server on http://localhost:11434; tests pass a fake
        self.client = client or ollama.Client()
        self.model = model

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> str:
        import ollama

        def call() -> str:
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    format=schema.model_json_schema(),  # structured output, same idea as Gemini's
                    options={"temperature": 0, "num_ctx": 8192},  # context window; longer input is CUT silently
                    think=False,
                )
            except ollama.ResponseError as e:
                raise LLMError(str(e)) from e
            except ConnectionError as e:
                raise TransientLLMError("Ollama not reachable - is it running?") from e
            return resp.message.content or ""

        key = {"provider": "ollama", "model": self.model, "system": system, "user": user,
               "schema": schema.model_json_schema()}
        return cached(key, lambda: with_backoff(call))

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> ChatReply:
        """One tool-calling turn (see the module docstring for the message format)."""
        import ollama

        def call() -> str:
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system}] + to_ollama(messages),
                    tools=[{"type": "function", "function": t} for t in tools] or None,
                    options={"temperature": 0, "num_ctx": 8192},  # tool results make prompts long: see generate_json
                    think=False,
                )
            except ollama.ResponseError as e:
                raise LLMError(str(e)) from e
            except ConnectionError as e:
                raise TransientLLMError("Ollama not reachable - is it running?") from e
            msg = resp.message
            calls = [{"id": "", "name": tc.function.name, "args": dict(tc.function.arguments or {})}
                     for tc in msg.tool_calls or []]
            if not msg.content and not calls:
                raise LLMError("Ollama returned an empty reply")
            return json.dumps({"text": msg.content or None, "tool_calls": calls,
                               "raw": msg.model_dump(mode="json", exclude_none=True)})

        key = chat_cache_key("ollama", self.model, system, messages, tools)
        out = json.loads(cached(key, lambda: with_backoff(call)))
        return ChatReply(out["text"], [ToolCall(**c) for c in out["tool_calls"]], out["raw"])


def get_llm(provider: str | None = None):
    provider = provider or config.LLM_PROVIDER
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (use 'gemini' or 'ollama')")


# ----------------------------------------------------------------- self-test with fake clients (no API calls)
class _FakeClient:
    """Stands in for genai.Client AND ollama.Client: plays back scripted responses, records each request."""

    def __init__(self, responses: list):
        self.responses, self.requests = list(responses), []
        self.models = self   # Gemini code calls client.models.generate_content(...)

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):   # a scripted failure, e.g. no internet
            raise response
        return response

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def selftest():
    import tempfile
    from pathlib import Path

    import ollama
    from google.genai import types

    global CACHE_DIR
    tools = [{"name": "search_docs", "description": "Search.",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
    question = {"role": "user", "content": "What late charge applies?"}
    call_part = types.Part(function_call=types.FunctionCall(id="c1", name="search_docs", args={"query": "late charge"}),
                           thought_signature=b"secret-signature")
    asks_tool = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[call_part]), finish_reason="STOP")])
    answers = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text="It is 4% [loan_x p.2].")]))])
    empty = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="MALFORMED_FUNCTION_CALL")])

    def gemini_round_trip() -> bool:
        fake = _FakeClient([asks_tool, answers])
        llm = GeminiLLM("fake-gemini", client=fake)
        first = llm.chat("system", [question], tools)
        history = [question, first.as_message(),
                   {"role": "tool", "tool_call_id": "c1", "name": "search_docs", "content": '{"sources": []}'}]
        second = llm.chat("system", history, tools)
        sent = fake.requests[1]["contents"]
        return (first.tool_calls == [ToolCall("c1", "search_docs", {"query": "late charge"})] and first.text is None
                and sent[1].parts[0].thought_signature == b"secret-signature"         # raw turn sent back unchanged
                and sent[2].role == "user" and sent[2].parts[0].function_response.id == "c1"
                and sent[2].parts[0].function_response.response == {"sources": []}
                and second.text == "It is 4% [loan_x p.2]." and not second.tool_calls)

    def gemini_parallel_results() -> bool:   # two calls in one turn -> both results in ONE Content
        msgs = [question, {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "a", "name": "t1", "args": {}}, {"id": "b", "name": "t2", "args": {}}]},
                {"role": "tool", "tool_call_id": "a", "name": "t1", "content": '{"x": 1}'},
                {"role": "tool", "tool_call_id": "b", "name": "t2", "content": "not json"}]
        c = to_gemini(msgs)
        return (len(c) == 3 and c[1].role == "model" and [p.function_call.id for p in c[1].parts] == ["a", "b"]
                and [p.function_response.name for p in c[2].parts] == ["t1", "t2"]
                and c[2].parts[1].function_response.response == {"output": "not json"})

    def cache_replays() -> bool:   # same request twice -> the fake client is asked only once
        fake = _FakeClient([asks_tool])
        llm = GeminiLLM("fake-gemini-cache", client=fake)
        a, b = llm.chat("system", [question], tools), llm.chat("system", [question], tools)
        return len(fake.requests) == 1 and a.tool_calls == b.tool_calls and b.raw.parts[0].thought_signature == b"secret-signature"

    def empty_reply_not_cached() -> bool:   # raises, and is NOT replayed from the cache next time
        llm = GeminiLLM("fake-gemini-empty", client=_FakeClient([empty, answers]))
        try:
            llm.chat("system", [question], tools)
            return False
        except LLMError as e:
            return "MALFORMED_FUNCTION_CALL" in str(e) and llm.chat("system", [question], tools).text is not None

    def no_internet_retried() -> bool:   # a network error is "line busy" too: wait (~2 s), call again
        import httpx

        fake = _FakeClient([httpx.ConnectError("no route to host"), answers])
        reply = GeminiLLM("fake-gemini-offline", client=fake).chat("system", [question], tools)
        return len(fake.requests) == 2 and reply.text == "It is 4% [loan_x p.2]."

    def ollama_round_trip() -> bool:
        tc = ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name="search_docs", arguments={"query": "x"}))
        fake = _FakeClient([ollama.ChatResponse(message=ollama.Message(role="assistant", content="", tool_calls=[tc]))])
        reply = OllamaLLM("fake-ollama", client=fake).chat("system", [question], tools)
        sent = fake.requests[0]
        back = to_ollama([reply.as_message(), {"role": "tool", "tool_call_id": "", "name": "search_docs", "content": "{}"}])
        return (reply.tool_calls == [ToolCall("", "search_docs", {"query": "x"})]
                and sent["messages"][0] == {"role": "system", "content": "system"}
                and sent["tools"][0]["function"]["name"] == "search_docs"
                and back[0]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
                and back[1] == {"role": "tool", "content": "{}", "tool_name": "search_docs"})

    def errors_sorted() -> bool:   # daily quota -> permanent (stop); per-minute 429 / 503 -> wait and retry
        from google.genai import errors

        per_day = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                                                     "details": [{"violations": [{"quotaId": "RequestsPerDayPerModel"}]}]}})
        per_minute = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down"}})
        overloaded = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}})
        bad_key = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad key"}})
        kinds = [type(_gemini_error(e)) for e in (per_day, per_minute, overloaded, bad_key)]
        return kinds == [LLMError, TransientLLMError, TransientLLMError, LLMError]

    cases = [("gemini: errors sorted into stop / retry", errors_sorted),
             ("gemini: tool call, raw turn + result sent back", gemini_round_trip),
             ("gemini: parallel results share one Content", gemini_parallel_results),
             ("gemini: identical request replayed from cache", cache_replays),
             ("gemini: empty reply -> LLMError, not cached", empty_reply_not_cached),
             ("gemini: no internet -> wait + retry, no crash", no_internet_retried),
             ("ollama: tool call + message translation", ollama_round_trip)]
    saved = CACHE_DIR, config.LLM_CACHE, config.GEMINI_RPM
    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        CACHE_DIR, config.LLM_CACHE, config.GEMINI_RPM = Path(tmp), True, 100_000   # no real cache, no pacing
        try:
            for name, check in cases:
                try:
                    ok, got = check(), ""
                except Exception as e:  # noqa: BLE001 - show any crash as a failed case
                    ok, got = False, f"crashed: {type(e).__name__}: {e}"
                passed += ok
                print(f"{'PASS' if ok else 'FAIL'}  {name:<48} {got}")
        finally:
            CACHE_DIR, config.LLM_CACHE, config.GEMINI_RPM = saved
    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(__doc__)
