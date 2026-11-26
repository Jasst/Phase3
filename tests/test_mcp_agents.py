# -*- coding: utf-8 -*-
"""Для других агентов: карта нейронов, журнал действий и описание сервера — по настоящему
MCP-клиенту по HTTP, с проверкой изоляции между пользователями."""
import asyncio
import json
import socket
import tempfile
import threading
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")

from pulsenet.unified_mcp.server import Settings, create_app  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    uvicorn = pytest.importorskip("uvicorn")
    cfg = Settings(data_dir=tempfile.mkdtemp(), owner_token="OWNERTOKEN", owner_id="owner")
    app = create_app(cfg)
    alice = app.state.tokens.issue("alice")
    bob = app.state.tokens.issue("bob")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}/mcp", "alice": alice, "bob": bob}
    srv.should_exit = True


def _call(url, token, name, args=None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def run():
        async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                res = await sess.call_tool(name, args or {})
                return json.loads(res.content[0].text)

    return asyncio.run(run())


def test_new_tools_are_registered(server):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def names():
        async with streamablehttp_client(server["url"], headers={"Authorization": f"Bearer {server['alice']}"}) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                return {t.name for t in (await sess.list_tools()).tools}

    got = asyncio.run(names())
    assert {"triplex_graph", "activity_recent", "pulsenet_guide", "web_search", "web_fetch"} <= got


def test_guide_describes_server_and_state(server):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    g = _call(server["url"], server["alice"], "pulsenet_guide")
    assert g["user"] == "alice"
    assert "triplex_search" in g["tools"] and "web_search" in g["tools"]
    assert "state" in g and g["state"]["docs"] == 0

    async def registered():
        async with streamablehttp_client(server["url"],
                                         headers={"Authorization": f"Bearer {server['alice']}"}) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                return {t.name for t in (await sess.list_tools()).tools}

    names = asyncio.run(registered())
    described = json.dumps(g["tools"], ensure_ascii=False)
    for name in sorted(names):
        assert name in described, f"pulsenet_guide не описывает инструмент {name}"
    assert g["tools_count"] == len(names) == 31

    workflow = "\n".join(g["workflow"])
    assert "triplex_feedback(fact_ids" in workflow
    assert "triplex_feedback(path" not in workflow
    assert "thought_id" in workflow


def test_activity_records_operations_per_user(server):
    _call(server["url"], server["bob"], "triplex_ingest", {"text": "Боб любит чай"})
    _call(server["url"], server["bob"], "triplex_search", {"query": "чай"})
    log = _call(server["url"], server["bob"], "activity_recent", {"limit": 10})
    tools = [i["tool"] for i in log["items"]]
    assert "triplex_ingest" in tools and "triplex_search" in tools
    assert all(i["ok"] in (True, False) for i in log["items"])
    # журнал Алисы не содержит действий Боба
    alice_log = _call(server["url"], server["alice"], "activity_recent", {"limit": 10})
    assert all("чай" not in i["summary"] for i in alice_log["items"])


def test_graph_is_shared_between_agents_and_shows_author(server):
    """Память общая для ваших агентов: факт, записанный одним, виден другому, автор — в поле agent."""
    doc = _call(server["url"], server["alice"], "triplex_ingest",
                {"text": "Алиса работает в Lab. Lab находится в Москве."})
    assert doc.get("doc_id")
    _call(server["url"], server["alice"], "triplex_assert",
          {"s": "Алиса", "r": "работает в", "o": "Lab", "doc_id": doc["doc_id"], "span": [0, 20]})
    def has_alice(graph):
        # id сущности может быть e:..., человекочитаемое имя лежит в label.
        return any(n.get("id") == "Алиса" or n.get("label") == "Алиса" for n in graph["nodes"])

    g = _call(server["url"], server["alice"], "triplex_graph", {"max_edges": 50})
    assert has_alice(g)
    assert g["edges"] and g["growth"]
    other = _call(server["url"], server["bob"], "triplex_graph", {"max_edges": 50})
    assert has_alice(other)
    facts = _call(server["url"], server["bob"], "triplex_facts", {"limit": 50})["facts"]
    mine = [f for f in facts if f["text"].startswith("Алиса")]
    assert mine and mine[0]["agent"] == "alice"


def test_link_and_facts_tools(server):
    _call(server["url"], server["alice"], "triplex_ingest", {"text": "Lab is a lab. Nel likes tea."})
    res = _call(server["url"], server["alice"], "triplex_link",
                {"entity_id": "несуществующая", "alias": "X"})
    assert res["status"] == "error"
