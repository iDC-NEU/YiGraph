"""Runtime dispatcher tests against a real local MCP HTTP fixture.

Run: python -m unittest discover -s tests -p test_web_search.py -v
Only MCP, HTTPX and PyYAML are needed. Unrelated graph/model/database services
are isolated at import/initialization; engine dispatch and MCP transport are real.
The reasoner is scripted, so these tests do not exercise model inference.
"""

import asyncio
import importlib
import json
import os
import tempfile
import threading
import time
import types
import unittest
from uuid import UUID
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from aag.config.engine_config import load_config_from_yaml
from aag.expert_search_engine.web_search import DEFAULT_URL, USER_AGENT


def load_engine_classes():
    """Load the production dispatcher without optional GPU/database packages."""
    dependencies = {
        "sympy": "N",
        "aag.reasoner.model_deployment": "Reasoner",
        "aag.computing_engine.computing_engine": "ComputingEngine",
        "aag.expert_search_engine.search": "ExpertSearchEngine",
        "aag.data_pipeline.data_transformer.dataset_manager": "DatasetManager",
        "aag.rag_engine.vector_rag": "VectorRAG",
        "aag.rag_engine.graph_rag": "GraphRAG",
    }
    modules = {}
    for name, attribute in dependencies.items():
        module = types.ModuleType(name)
        setattr(module, attribute, Mock())
        modules[name] = module
    with patch.dict("sys.modules", modules):
        scheduler = importlib.import_module("aag.engine.scheduler").Scheduler
        engine = importlib.import_module("aag.engine.aag_engine").AAGEngine
    return engine, scheduler


AAGEngine, Scheduler = load_engine_classes()
QueryType = Scheduler.execute.__globals__["QueryType"]
SOURCE = "NetworkX PageRank reference https://networkx.org/documentation/stable/reference/algorithms/link_analysis.html"


class MCPHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), request))
        method = request["method"]
        if "id" not in request:
            self.send_response(202)
            self.end_headers()
            return
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in self.server.tools]}
        elif method == "tools/call":
            self.server.called.set()
            time.sleep(self.server.delay)
            result = {"content": [{"type": "text", "text": self.server.content}],
                      "isError": self.server.tool_error}
            if self.server.empty:
                result["content"] = []
            if self.server.echo_input:
                result["content"] = [{"type": "text", "text": json.dumps(request["params"]["arguments"])}]
        else:
            raise AssertionError(method)
        status = self.server.http_status if method == "tools/call" else 200
        payload = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
        if self.server.malformed and method == "tools/call":
            payload = b"not valid JSON"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Expected when testing timeout/cancellation.


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/mcp"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.requests = []
        self.server.called = threading.Event()
        self.server.delay = 0
        self.server.content = SOURCE
        self.server.tool_error = False
        self.server.empty = False
        self.server.http_status = 200
        self.server.malformed = False
        self.server.echo_input = False
        self.server.tools = ["web_search", "web_fetch"]

    def engine(self, settings=None):
        # Exercise the production YAML loader too; old files need no new section.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "engine.yaml"
            path.write_text(json.dumps({"retrieval": {"web_search": settings if settings is not None else {}}}))
            config = load_config_from_yaml(path)
        def initialize(scheduler):
            scheduler.reasoner = Mock()
            scheduler.reasoner.generate_response.side_effect = lambda prompt: "Summary: " + prompt.split("<web_sources>\n")[1].split("\n</web_sources>")[0]
            scheduler.router = Mock()
            scheduler.dataset_manager = Mock()
        with patch.object(Scheduler, "_initialize_components", initialize):
            return AAGEngine(config)

    def calls(self):
        return [request[2]["params"] for request in self.server.requests
                if request[2]["method"] == "tools/call"]

    async def test_unconfigured_default_uses_anonymous_parallel_and_consumes_sources(self):
        engine = self.engine()
        self.assertEqual(engine.config.retrieval.web_search, {})
        from aag.expert_search_engine import web_search
        transport = web_search.streamablehttp_client
        seen = []
        def redirect_fixture(url, **kwargs):
            seen.append(url)
            return transport(self.url, **kwargs)
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "ambient-must-not-be-used"}), \
                patch.object(web_search, "streamablehttp_client", redirect_fixture):
            answer = await engine.run("/web NetworkX PageRank convergence reference")
        self.assertEqual(seen, [DEFAULT_URL])
        self.assertIn(SOURCE, answer)
        engine.scheduler.router.route.assert_not_called()
        engine.scheduler.reasoner.generate_response.assert_called_once()
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assert_session_id(calls[0]["arguments"].pop("session_id"))
        self.assertEqual(calls, [{"name": "web_search", "arguments": {
            "objective": "NetworkX PageRank convergence reference",
            "search_queries": ["NetworkX PageRank convergence reference"]}}])
        self.assert_wire_headers()

    def assert_session_id(self, value):
        self.assertEqual(str(UUID(value)), value)
        self.assertEqual(UUID(value).version, 4)

    def assert_wire_headers(self, key=None):
        methods = {request[2]["method"] for request in self.server.requests}
        self.assertTrue({"initialize", "tools/list", "tools/call"} <= methods)
        for path, headers, _ in self.server.requests:
            headers = {k.lower(): v for k, v in headers.items()}
            self.assertEqual(path, "/mcp")
            self.assertEqual(headers["user-agent"], USER_AGENT)
            self.assertEqual(headers.get("authorization"), f"Bearer {key}" if key else None)

    async def test_saved_endpoint_and_credentials_fetch_url_and_ignore_dataset(self):
        engine = self.engine({"url": self.url, "provider": "parallel", "api_key": "explicit-key"})
        engine.scheduler.current_dataset = ["private dataset must not be sent"]
        engine.scheduler.current_dataset_name = "private-name"
        url = "https://networkx.org/documentation/stable/?version=3#pagerank"
        answer = await engine.run("/fetch " + url, mode="interact")
        self.assertIn(SOURCE, answer)
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assert_session_id(calls[0]["arguments"].pop("session_id"))
        self.assertEqual(calls, [{"name": "web_fetch", "arguments": {"urls": [url]}}])
        self.assertNotIn("private", json.dumps(self.server.requests))
        self.assert_wire_headers("explicit-key")

    async def test_disabled_and_invalid_settings_never_access_network(self):
        for settings in [{"enabled": False}, {"provider": "unknown"}, {"timeout": 0},
                         {"timeout": True}, {"timeout": float("inf")},
                         {"enabled": "false"}, {"api_key": 42}, {"url": "file:///tmp/socket"},
                         {"url": 42}, {"url": []}, {"url": None}, False, []]:
            with self.subTest(settings=settings):
                engine = self.engine(settings)
                answer = await engine.run("/web graph algorithm references")
                self.assertTrue(answer.startswith(("⚠️", "❌")))
                engine.scheduler.reasoner.generate_response.assert_not_called()
        self.assertEqual(self.server.requests, [])

    async def test_empty_commands_and_invalid_fetch_urls_never_access_network(self):
        engine = self.engine({"url": self.url})
        for text in ["/web", "/fetch", "/fetch file:///etc/passwd", "/fetch https://a.test https://b.test",
                     "/fetch https://user:password@example.com", "/fetch javascript:alert(1)"]:
            with self.subTest(text=text):
                self.assertTrue((await engine.run(text)).startswith(("⚠️", "❌")))
        self.assertEqual(self.server.requests, [])

    async def test_ordinary_queries_keep_original_router_and_dataset_guard(self):
        engine = self.engine({"url": self.url})
        self.assertIn("No dataset selected", await engine.run("explain graph algorithms"))
        engine.scheduler.router.route.assert_called_once_with(query="explain graph algorithms")
        engine.scheduler.current_dataset = [object()]
        engine.scheduler.current_dataset_name = "local"
        engine.scheduler.router.route.return_value.query_type = QueryType.GENERAL
        engine.scheduler.reasoner.general_query_response.return_value = "local answer"
        self.assertEqual(await engine.run("explain PageRank"), "local answer")
        self.assertEqual(self.server.requests, [])

    async def test_provider_errors_empty_results_and_missing_tools_do_not_synthesize(self):
        for kind in ["tool", "http", "empty", "missing", "malformed"]:
            with self.subTest(kind=kind):
                self.server.tool_error = kind == "tool"
                self.server.http_status = 503 if kind == "http" else 200
                self.server.empty = kind == "empty"
                self.server.malformed = kind == "malformed"
                self.server.tools = [] if kind == "missing" else ["web_search", "web_fetch"]
                engine = self.engine({"url": self.url, "timeout": 2})
                answer = await engine.run("/web graph algorithm references")
                self.assertTrue(answer.startswith(("⚠️", "❌")))
                engine.scheduler.reasoner.generate_response.assert_not_called()

    async def test_timeout_releases_transport_without_synthesis(self):
        self.server.delay = 2
        engine = self.engine({"url": self.url, "timeout": 1})
        answer = await engine.run("/web graph algorithm references")
        self.assertTrue(answer.startswith("❌"))
        self.assertTrue(self.server.called.is_set())
        engine.scheduler.reasoner.generate_response.assert_not_called()

    async def test_cancellation_propagates_without_synthesis(self):
        self.server.delay = 0.5
        engine = self.engine({"url": self.url, "timeout": 5})
        task = asyncio.create_task(engine.run("/web graph algorithm references"))
        self.assertTrue(await asyncio.to_thread(self.server.called.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        engine.scheduler.reasoner.generate_response.assert_not_called()

    async def test_concurrent_commands_do_not_share_request_context(self):
        self.server.echo_input = True
        engine = self.engine({"url": self.url})
        first, second = await asyncio.gather(
            engine.run("/web pagerank convergence documentation"),
            engine.run("/fetch https://example.org/communities"),
        )
        self.assertIn("pagerank convergence documentation", first)
        self.assertNotIn("communities", first)
        self.assertIn("https://example.org/communities", second)
        self.assertNotIn("pagerank", second)
        ids = [call["arguments"]["session_id"] for call in self.calls()]
        self.assertEqual(len(set(ids)), 2)
        for value in ids:
            self.assert_session_id(value)


if __name__ == "__main__":
    unittest.main()
