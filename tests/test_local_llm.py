"""Layers 2 and 3 with a local LLM (generator "ollama"), against a stub OpenAI-compatible server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _Stub(BaseHTTPRequestHandler):
    replies: list[str] = []
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append(body)
        text = _Stub.replies.pop(0) if _Stub.replies else "[]"
        out = json.dumps({"model": body["model"], "choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def ollama(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.delenv("KB_LLM_MODEL", raising=False)
    _Stub.replies, _Stub.seen = [], []
    yield _Stub
    srv.shutdown()


def test_call_llm_ollama(ollama):
    from kanoon_bridge.rag.generate import LLM_GENERATORS, call_llm

    ollama.replies = ["hello"]
    assert "ollama" in LLM_GENERATORS
    assert call_llm("ollama", "hi") == "hello"
    assert ollama.seen[0]["model"] == "llama3.2:3b" and ollama.seen[0]["temperature"] == 0


def test_llm_planner_uses_ollama(ollama):
    from kanoon_bridge.agent.plan import LLMPlanner
    from kanoon_bridge.schema import AnalyzedQuery, Query

    ollama.replies = ['["murder punishment BNS 103", "IPC 302 murder"]']
    p = LLMPlanner(generator="ollama").plan(Query("punishment for murder"), AnalyzedQuery(Query("punishment for murder")))
    kinds = [s.kind for s in p.subqueries]
    assert kinds.count("llm") == 2 and len(ollama.seen) == 1


def test_rag_v2_with_ollama_regenerates_and_verifies(ollama):
    """A wrong first answer is verified, regenerated once with the problems listed, then checked."""
    from kanoon_bridge.rag.answer import _answer_v2
    from kanoon_bridge.rag.chunker import Chunk
    from kanoon_bridge.schema import AnalyzedQuery, Code, Document, DocType, Paragraph, Query
    from kanoon_bridge.config import load_config

    murder = "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."
    docs = {"bns:103": Document("bns:103", DocType.STATUTE, title="Punishment for murder", code=Code.BNS, section="103",
                                meta={"ref": "bns:103"}, paragraphs=[Paragraph(murder, "statute", 0)])}
    chunks = [Chunk(1, "bns:103", "statute", murder, 1, 1.0, "BNS Section 103 - Punishment for murder")]
    ollama.replies = ["Murder is punishable with imprisonment up to 10 years. [1]",
                      "Murder is punishable with death or imprisonment for life, and fine. [1]"]
    q = Query("What is the punishment for murder?")
    aq = AnalyzedQuery(q, code_in_force=Code.BNS)
    ans = _answer_v2(q, aq, chunks, docs, load_config().rag, None, None, None, "ollama")
    assert len(ollama.seen) == 2 and "problems" in ollama.seen[1]["messages"][1]["content"]
    assert ans.generator == "ollama" and "10 years" not in ans.text
    assert all(c.verdict == "supported" for c in ans.claims)
