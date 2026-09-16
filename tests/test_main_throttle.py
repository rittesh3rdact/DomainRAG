"""Integration-level check that /api/chat rejects excess requests before
they ever reach the model, using a monkeypatched agent/store so no network
call happens even for the request(s) that pass the throttle."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import app.main as main_module


def test_chat_endpoint_throttles_excess_requests():
    main_module._agent = MagicMock(answer=MagicMock(return_value=(iter(["ok"]), [])))
    main_module._store = MagicMock(count=MagicMock(return_value=1))
    main_module._chat_throttle.__init__(1)  # only 1 request/minute allowed

    client = TestClient(main_module.app)

    first = client.post("/api/chat", json={"message": "hello"})
    second = client.post("/api/chat", json={"message": "hello again"})

    assert first.status_code == 200
    assert second.status_code == 429
