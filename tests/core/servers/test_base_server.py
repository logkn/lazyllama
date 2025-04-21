import pytest
import aiohttp

from src.core.servers.base_server import BaseServer, ServerStatus
from src.core.models.alias import Alias, AliasModel, Backend
import src.core.servers.base_server as base_server


def test_server_status_enum_values():
    # Ensure all expected statuses are present with correct values
    expected = ["starting", "running", "stopping", "stopped"]
    values = [member.value for member in ServerStatus]
    assert values == expected


class DummyServer(BaseServer):
    """Minimal BaseServer subclass for testing wait_until_ready."""
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def check_running(self) -> bool:
        return False


class DummyStartServer(BaseServer):
    """Server subclass to test start_and_wait behavior."""
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self.calls = []

    async def start(self) -> None:
        # status should be STARTING when start() is called
        assert self.status == ServerStatus.STARTING
        self.calls.append("start")

    async def stop(self) -> None:
        pass

    async def check_running(self) -> bool:
        return False

    async def wait_until_ready(self, timeout: float = 10.0) -> None:
        # status remains STARTING until readiness is confirmed
        assert self.status == ServerStatus.STARTING
        self.calls.append("wait_ready")


class DummyStopServer(BaseServer):
    """Server subclass to test stop_and_wait behavior."""
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self.calls = []
        # start assumed to have left server running
        self.running = True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        # status should be STOPPING when stop() is called
        assert self.status == ServerStatus.STOPPING
        self.calls.append("stop")
        self.running = False

    async def check_running(self) -> bool:
        self.calls.append("check_running")
        return self.running


@pytest.mark.asyncio
async def test_start_and_wait_sets_status_and_order():
    alias = Alias(
        name="test", model=AliasModel(model_name="m", backend=Backend.OLLAMA),
        n_ctx=1, command_params=[]
    )
    server = DummyStartServer(alias, port=1234)
    # Initially stopped
    assert server.status == ServerStatus.STOPPED
    await server.start_and_wait()
    # Verify call order and final status
    assert server.calls == ["start", "wait_ready"]
    assert server.status == ServerStatus.RUNNING


@pytest.mark.asyncio
async def test_stop_and_wait_sets_status_and_order():
    alias = Alias(
        name="test", model=AliasModel(model_name="m", backend=Backend.OLLAMA),
        n_ctx=1, command_params=[]
    )
    server = DummyStopServer(alias, port=1234)
    # Simulate running state
    server.status = ServerStatus.RUNNING
    await server.stop_and_wait()
    # Verify call order and final status
    assert server.calls == ["stop", "check_running"]
    assert server.status == ServerStatus.STOPPED


@pytest.mark.asyncio
async def test_wait_until_ready_success(monkeypatch):
    """wait_until_ready should return when HTTP 200 is encountered."""
    class DummyResponse:
        def __init__(self):
            self.status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url):
            return DummyResponse()

    # Patch ClientSession to use dummy that immediately returns 200
    monkeypatch.setattr(base_server.aiohttp, "ClientSession", DummySession)

    alias = Alias(
        name="test", model=AliasModel(model_name="m", backend=Backend.OLLAMA),
        n_ctx=1, command_params=[]
    )
    server = DummyServer(alias, port=1234)
    # Should complete without raising
    await server.wait_until_ready(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_timeout(monkeypatch):
    """wait_until_ready should timeout if HTTP errors continue."""
    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url):
            # Simulate persistent connection error
            raise aiohttp.ClientError()

    # Patch ClientSession to always error
    monkeypatch.setattr(base_server.aiohttp, "ClientSession", DummySession)

    alias = Alias(
        name="test", model=AliasModel(model_name="m", backend=Backend.OLLAMA),
        n_ctx=1, command_params=[]
    )
    server = DummyServer(alias, port=1234)
    with pytest.raises(TimeoutError, match="Server did not become ready in time."):
        await server.wait_until_ready(timeout=1.0)