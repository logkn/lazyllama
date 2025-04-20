import asyncio
import pytest

from src.core.models.alias import Alias, AliasModel, Backend
from src.core.server_manager import ServerManager
from src.core.servers.base_server import BaseServer, ServerStatus


class MockServer(BaseServer):
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self._is_running = False

    async def start(self) -> None:
        await asyncio.sleep(0.01)
        self._is_running = True

    async def stop(self) -> None:
        self._is_running = False  # ✅ clear running state first
        print("Mock set self._is_running to False")
        await asyncio.sleep(0.01)

    async def check_running(self) -> bool:
        return self._is_running

    async def wait_until_ready(self, timeout: float = 10.0) -> None:
        await asyncio.sleep(0.01)  # simulate readiness


@pytest.mark.asyncio
async def test_server_manager_lifecycle():
    manager = ServerManager()

    manager.total_ram_mb = 10_000  # plenty
    manager.total_vram_mb = 10_000  # plenty

    # Patch with mock server factory
    def create_mock_server(self, alias: Alias, port: int) -> BaseServer:
        return MockServer(alias, port)

    def dummy_resource_model(self, alias: Alias) -> tuple[float, float, float, float]:
        return (100.0, 0.1, 200.0, 0.2)  # Simple linear model

    manager.create_server = create_mock_server.__get__(manager)
    manager.get_or_measure_resource_model = dummy_resource_model.__get__(manager)

    model_spec = AliasModel(model_name="llama2", backend=Backend.OLLAMA)

    alias_a = Alias(name="a", model=model_spec, n_ctx=2048, command_params=["--foo"])
    alias_b = Alias(name="b", model=model_spec, n_ctx=4096, command_params=["--foo"])

    # Start first server
    s1 = await manager.get_or_start_server(alias_a)
    assert s1.status == ServerStatus.RUNNING

    # Should reuse for alias A
    s2 = await manager.get_or_start_server(alias_a)
    assert s2 is s1

    # Should not reuse for alias B (higher n_ctx)
    s3 = await manager.get_or_start_server(alias_b)
    assert s3 is not s1
    assert s3.status == ServerStatus.RUNNING

    # Check internal state
    assert len(manager.running_servers) >= 2


class EvictableMockServer(BaseServer):
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self._is_running = False

    async def start(self) -> None:
        self._is_running = True
        await asyncio.sleep(0.01)

    async def stop(self) -> None:
        self._is_running = False  # ✅ immediately mark as not running
        await asyncio.sleep(0.01)

    async def check_running(self) -> bool:
        return self._is_running

    async def wait_until_ready(self, timeout: float = 10.0) -> None:
        await asyncio.sleep(0.01)

    def get_resource_model(self) -> tuple[float, float, float, float]:
        # Not used in this test; stubbed for interface completeness
        return (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_eviction_impossible_due_to_insufficient_resources():
    manager = ServerManager()

    # Tight total resources
    manager.total_ram_mb = 300.0
    manager.total_vram_mb = 1000.0  # instead of 800.0

    # Patch mocks
    def create_mock_server(self, alias: Alias, port: int) -> BaseServer:
        return EvictableMockServer(alias, port)

    def dummy_resource_model(self, alias: Alias) -> tuple[float, float, float, float]:
        return (100.0, 0.1, 100.0, 0.2)  # Linear growth

    manager.create_server = create_mock_server.__get__(manager)
    manager.get_or_measure_resource_model = dummy_resource_model.__get__(manager)

    model_spec = AliasModel(model_name="llama2", backend=Backend.OLLAMA)

    # Small server — consumes ~200 RAM and ~300 VRAM
    alias_a = Alias(name="a", model=model_spec, n_ctx=1000, command_params=["--x"])
    await manager.get_or_start_server(alias_a)

    # Large server — requires more than total capacity, even if A is evicted
    alias_b = Alias(name="b", model=model_spec, n_ctx=3000, command_params=["--x"])

    with pytest.raises(
        RuntimeError,
        match="No combination of servers can satisfy resource requirements",
    ):
        await manager.get_or_start_server(alias_b)
