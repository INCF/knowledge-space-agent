import asyncio
import sys
import types
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def _install_dependency_stubs():
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.END = "__end__"

    class StateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_node(self, *_args, **_kwargs):
            pass

        def set_entry_point(self, *_args, **_kwargs):
            pass

        def add_edge(self, *_args, **_kwargs):
            pass

        def compile(self):
            class Graph:
                async def ainvoke(self, state):
                    return {**state, "final_response": "ok"}

            return Graph()

    graph_module.StateGraph = StateGraph
    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.graph = graph_module
    sys.modules.setdefault("langgraph", langgraph_module)
    sys.modules.setdefault("langgraph.graph", graph_module)

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_types_module = types.ModuleType("google.genai.types")
    genai_module.Client = lambda *_args, **_kwargs: object()
    genai_module.types = genai_types_module
    google_module.genai = genai_module
    sys.modules.setdefault("google", google_module)
    sys.modules.setdefault("google.genai", genai_module)
    sys.modules.setdefault("google.genai.types", genai_types_module)

    search_module = types.ModuleType("ks_search_tool")
    search_module.general_search = lambda *_args, **_kwargs: []
    search_module.general_search_async = lambda *_args, **_kwargs: []
    search_module.global_fuzzy_keyword_search = lambda *_args, **_kwargs: []
    sys.modules.setdefault("ks_search_tool", search_module)

    retrieval_module = types.ModuleType("retrieval")
    retrieval_module.get_retriever = lambda *_args, **_kwargs: None
    sys.modules.setdefault("retrieval", retrieval_module)


_install_dependency_stubs()
import agents  # noqa: E402


class BlockingGraph:
    def __init__(self, response_text):
        self.response_text = response_text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, state):
        self.started.set()
        await self.release.wait()
        return {
            **state,
            "all_results": ["dataset"],
            "effective_query": state["query"],
            "final_response": self.response_text,
            "intents": [agents.QueryIntent.DATA_DISCOVERY.value],
            "keywords": ["dataset"],
        }


class BlockingSynthesis:
    def __init__(self, response_text):
        self.response_text = response_text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, *_args, **_kwargs):
        self.started.set()
        await self.release.wait()
        return self.response_text


def test_active_session_is_not_evicted_before_response_is_recorded(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_COUNT", "1")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")

    async def run():
        assistant = agents.NeuroscienceAssistant()
        assistant.graph = BlockingGraph("held response")

        task = asyncio.create_task(assistant.handle_chat("active", "find data"))
        await assistant.graph.started.wait()

        assistant._ensure_session("newer")
        assert "active" in assistant._session_last_seen
        assert "newer" in assistant._session_last_seen

        assistant.graph.release.set()
        assert await task == "held response"

        assert assistant.chat_history["active"] == [
            "User: find data",
            "Assistant: held response",
        ]
        assert assistant.session_memory["active"]["last_text"] == "held response"
        assert "active" not in assistant._session_active_counts
        assert "newer" not in assistant._session_last_seen
        assert "newer" not in assistant.chat_history

    asyncio.run(run())


def test_active_more_session_is_not_evicted_before_page_is_recorded(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_COUNT", "1")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")

    async def run():
        assistant = agents.NeuroscienceAssistant()
        synthesis = BlockingSynthesis("second page")
        monkeypatch.setattr(agents, "call_gemini_for_final_synthesis", synthesis)

        assistant._ensure_session("active")
        assistant.session_memory["active"] = {
            "all_results": ["first", "second"],
            "page": 1,
            "page_size": 1,
            "effective_query": "find data",
            "intents": [agents.QueryIntent.DATA_DISCOVERY.value],
            "last_text": "first page",
        }

        task = asyncio.create_task(assistant.handle_chat("active", "more"))
        await synthesis.started.wait()

        assistant._ensure_session("newer")
        assert "active" in assistant._session_last_seen
        assert "newer" in assistant._session_last_seen

        synthesis.release.set()
        assert await task == "second page"

        assert assistant.session_memory["active"]["page"] == 2
        assert assistant.session_memory["active"]["last_text"].endswith("second page")
        assert assistant.chat_history["active"] == [
            "User: more",
            "Assistant: second page",
        ]
        assert "active" not in assistant._session_active_counts
        assert "newer" not in assistant._session_last_seen
        assert "newer" not in assistant.chat_history

    asyncio.run(run())


def test_active_session_is_not_dropped_by_ttl_eviction_until_released(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_COUNT", "2")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "10")

    assistant = agents.NeuroscienceAssistant()
    assistant._ensure_session("active")
    assistant._session_last_seen["active"] = 100.0
    assistant._reserve_session("active")

    assistant._evict_expired_sessions(now=111.0)
    assert "active" in assistant._session_last_seen
    assert "active" in assistant.chat_history

    monkeypatch.setattr(agents, "monotonic", lambda: 111.0)
    assistant._release_session("active")
    assert "active" not in assistant._session_last_seen
    assert "active" not in assistant.chat_history
    assert "active" not in assistant._session_active_counts


def test_reset_invalidates_in_flight_session_response(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_COUNT", "2")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")

    async def run():
        assistant = agents.NeuroscienceAssistant()
        assistant.graph = BlockingGraph("stale response")

        task = asyncio.create_task(assistant.handle_chat("session", "find data"))
        await assistant.graph.started.wait()
        assistant.reset_session("session")

        assistant.graph.release.set()
        assert await task == "stale response"

        assert "session" not in assistant.chat_history
        assert "session" not in assistant.session_memory
        assert "session" not in assistant._session_last_seen
        assert "session" not in assistant._session_active_counts

    asyncio.run(run())
