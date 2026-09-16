from types import SimpleNamespace
from unittest.mock import MagicMock

from app.rag.agent import RAGAgent
from app.rag.rate_limiter import RateLimiter
from app.rag.vectorstore import RetrievedChunk


def _function_call_response(name: str, args: dict):
    call = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(function_call=call)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=None)


def _text_response(text: str):
    part = SimpleNamespace(function_call=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)


def test_agent_answers_directly_when_model_skips_tool_call():
    client = MagicMock()
    client.models.generate_content.return_value = _text_response("Paris is the capital.")
    search_tool = MagicMock()

    agent = RAGAgent(client, ["test-model"], search_tool, max_tool_calls=3, rate_limiter=RateLimiter(0))
    stream, chunks = agent.answer("What is the capital of France?", history=[])

    assert list(stream) == ["Paris is the capital."]
    assert chunks == []
    search_tool.run.assert_not_called()


def test_agent_calls_tool_then_answers():
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _function_call_response("search_documents", {"query": "capital of France"}),
        _text_response("Paris. [source: geo.txt]"),
    ]
    chunk = RetrievedChunk(id="geo.txt::0", text="Paris is the capital of France.", source="geo.txt", chunk_index=0, similarity=0.9)
    search_tool = MagicMock()
    search_tool.run.return_value = ({"results": [{"source": "geo.txt", "text": chunk.text}]}, [chunk])

    agent = RAGAgent(client, ["test-model"], search_tool, max_tool_calls=3, rate_limiter=RateLimiter(0))
    stream, chunks = agent.answer("What is the capital of France?", history=[])

    assert list(stream) == ["Paris. [source: geo.txt]"]
    assert chunks == [chunk]
    search_tool.run.assert_called_once_with("capital of France")


def test_agent_forces_final_answer_after_max_tool_calls():
    client = MagicMock()
    client.models.generate_content.return_value = _function_call_response(
        "search_documents", {"query": "anything"}
    )
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="forced answer")]
    )
    search_tool = MagicMock()
    search_tool.run.return_value = ({"results": []}, [])

    agent = RAGAgent(client, ["test-model"], search_tool, max_tool_calls=2, rate_limiter=RateLimiter(0))
    stream, chunks = agent.answer("Loop forever?", history=[])

    assert list(stream) == ["forced answer"]
    assert client.models.generate_content.call_count == 2
    client.models.generate_content_stream.assert_called_once()
