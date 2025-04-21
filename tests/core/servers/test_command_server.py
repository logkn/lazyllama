import asyncio
import pytest

from src.core.servers.command_server import CommandServer
from src.core.models.alias import Alias, AliasModel, Backend


class DummyProcess:
    """Simulate a subprocess with immediate wait completion."""
    def __init__(self):
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False

    def terminate(self):
        self.terminate_called = True
        # Simulate normal termination
        self.returncode = 0

    def kill(self):
        self.kill_called = True
        # Simulate kill return code
        self.returncode = -9

    async def wait(self):
        self.wait_called = True
        return self.returncode


class DummyProcessKill:
    """Simulate a subprocess that hangs until killed."""
    def __init__(self):
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False
        self._event = asyncio.Event()

    def terminate(self):
        self.terminate_called = True
        # Do not set returncode to simulate hang

    def kill(self):
        self.kill_called = True
        # Set returncode and wake waiters
        self.returncode = -9
        self._event.set()

    async def wait(self):
        self.wait_called = True
        # Block until kill is called
        if not self.kill_called:
            await self._event.wait()
        return self.returncode


class FakeCommandServer(CommandServer):
    """Concrete CommandServer for testing build_command, start, and stop."""
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self.command_args = None

    def build_command(self, port: int, alias: Alias) -> str:
        # Record arguments and return a dummy shell command
        self.command_args = (port, alias)
        return f"run_server --port={port} --alias={alias.name}"

    async def check_running(self) -> bool:
        # Not used in start/stop tests
        return False


@pytest.mark.asyncio
async def test_start_spawns_process(monkeypatch):
    alias = Alias(
        name="test", model=AliasModel(model_name="m", backend=Backend.LLAMACPP),
        n_ctx=1, command_params=[]
    )
    server = FakeCommandServer(alias, port=4242)
    dummy = DummyProcess()
    calls = []

    async def fake_create(cmd, stdout, stderr):  # noqa: ARG002
        calls.append((cmd, stdout, stderr))
        return dummy

    # Patch subprocess creation
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create)

    await server.start()
    # build_command should have been called with correct args
    assert server.command_args == (4242, alias)
    # subprocess should be spawned with correct command and DEVNULLs
    assert len(calls) == 1
    cmd, out, err = calls[0]
    assert cmd == "run_server --port=4242 --alias=test"
    assert out == asyncio.subprocess.DEVNULL
    assert err == asyncio.subprocess.DEVNULL
    # server.process should be set to dummy
    assert server.process is dummy


@pytest.mark.asyncio
async def test_start_idempotent(monkeypatch):
    alias = Alias(
        name="test2", model=AliasModel(model_name="m2", backend=Backend.OLLAMA),
        n_ctx=2, command_params=[]
    )
    server = FakeCommandServer(alias, port=1111)
    dummy = DummyProcess()
    calls = []

    async def fake_create(cmd, stdout, stderr):  # noqa: ARG002
        calls.append(cmd)
        return dummy

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create)

    # First start should spawn, second should be no-op
    await server.start()
    await server.start()
    assert calls == ["run_server --port=1111 --alias=test2"]


@pytest.mark.asyncio
async def test_stop_normal_flow():
    alias = Alias(
        name="stoptest", model=AliasModel(model_name="m3", backend=Backend.LLAMACPP),
        n_ctx=3, command_params=[]
    )
    server = FakeCommandServer(alias, port=2020)
    dummy = DummyProcess()
    server.process = dummy

    # Stop should terminate and wait without killing
    await server.stop()
    assert dummy.terminate_called is True
    assert dummy.wait_called is True
    assert dummy.kill_called is False
    assert server.process is None


@pytest.mark.asyncio
async def test_stop_kill_on_timeout(monkeypatch):
    alias = Alias(
        name="killtest", model=AliasModel(model_name="m4", backend=Backend.OLLAMA),
        n_ctx=4, command_params=[]
    )
    server = FakeCommandServer(alias, port=3030)
    dummy = DummyProcessKill()
    server.process = dummy

    # Simulate immediate timeout on first wait
    async def fake_wait_for(coro, timeout):  # noqa: ARG002
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    # Stop should terminate, then kill, then wait
    await server.stop()
    assert dummy.terminate_called is True
    assert dummy.kill_called is True
    assert dummy.wait_called is True
    assert server.process is None


@pytest.mark.asyncio
async def test_stop_no_process_or_already_exited():
    alias = Alias(
        name="noproc", model=AliasModel(model_name="m5", backend=Backend.LLAMACPP),
        n_ctx=5, command_params=[]
    )
    server = FakeCommandServer(alias, port=4040)
    # Case: no process
    server.process = None
    await server.stop()
    assert server.process is None

    # Case: process has already exited
    dummy = DummyProcess()
    dummy.returncode = 1
    server.process = dummy
    # Should not call terminate or wait or kill
    await server.stop()
    assert dummy.terminate_called is False
    assert dummy.wait_called is False
    assert dummy.kill_called is False
    assert server.process is None