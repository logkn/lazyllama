import pytest
import asyncio

from src.core.models.alias import Alias, AliasModel, Backend
from src.core.server_manager import ServerManager
from src.core.servers.base_server import BaseServer, ServerStatus


class EvictableMockServer(BaseServer):
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self._is_running = False

    async def start(self) -> None:
        await asyncio.sleep(0.01)
        self._is_running = True

    async def stop(self) -> None:
        await asyncio.sleep(0.01)
        self._is_running = False

    async def check_running(self) -> bool:
        return self._is_running

    async def wait_until_ready(self, timeout: float = 10.0) -> None:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_server_eviction_under_pressure():
    manager = ServerManager()

    # Patch with tight resource limits
    manager.total_ram_mb = 400.0
    manager.total_vram_mb = 800.0

    # Patch with mocks
    def create_mock_server(self, alias: Alias, port: int) -> BaseServer:
        return EvictableMockServer(alias, port)

    def dummy_resource_model(self, alias: Alias) -> tuple[float, float, float, float]:
        return (100.0, 0.1, 200.0, 0.2)  # RAM, RAM/ctx, VRAM, VRAM/ctx

    manager.create_server = create_mock_server.__get__(manager)
    manager.get_or_measure_resource_model = dummy_resource_model.__get__(manager)

    model_spec = AliasModel(model_name="llama2", backend=Backend.OLLAMA)

    # First server takes nearly all VRAM
    alias_a = Alias(
        name="a", model=model_spec, n_ctx=3000, command_params=["--foo"]
    )  # Uses ~400 VRAM
    alias_b = Alias(
        name="b", model=model_spec, n_ctx=3500, command_params=["--foo"]
    )  # Uses ~500 VRAM

    s1 = await manager.get_or_start_server(alias_a)
    assert s1.status == ServerStatus.RUNNING
    assert len(manager.running_servers) == 1

    # This should force eviction of s1 to start s2
    s2 = await manager.get_or_start_server(alias_b)
    assert s2.status == ServerStatus.RUNNING
    assert s2 is not s1
    assert len(manager.running_servers) == 1  # s1 evicted

    # Ensure s1 really stopped
    assert not await s1.check_running()
