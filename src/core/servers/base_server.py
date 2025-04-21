from abc import ABC, abstractmethod
import asyncio
from enum import StrEnum
import aiohttp

from src.core.models.alias import Alias


class ServerStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class BaseServer(ABC):
    def __init__(self, alias: Alias, port: int):
        self.alias: Alias = alias
        self.port: int = port
        self.status: ServerStatus = ServerStatus.STOPPED

    @abstractmethod
    async def start(self) -> None:
        """
        Start the server process.

        This method is responsible for initiating the backend server process
        (e.g. via subprocess or system API). It must:
          - Launch the process that will respond to requests on `http://localhost:{self.port}/v1`.
          - Ensure the process is detached, non-blocking, and long-running.
          - Not return until the launch command has been successfully dispatched.
          - *Not* block waiting for readiness — readiness is handled separately by `wait_until_ready`.

        This method should update any necessary internal state such as tracking the subprocess object.
        The process must bind to `localhost:{self.port}` and respond to `/v1/models` when ready.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the server process and release all resources.

        This method must:
          - Terminate any background process(es) spawned by `start()`, including children if applicable.
          - Wait for the process to exit cleanly (use `terminate`, fallback to `kill` if necessary).
          - Handle zombie or orphan processes.
          - Be safe to call even if the server is already stopped or partially exited.
          - Not raise if the process is already gone — just ensure it is fully stopped.

        If you are using subprocesses, you are responsible for managing the entire subtree and cleaning up.
        If using Docker or a daemon, this method should stop that container or process reliably.
        """
        ...

    @abstractmethod
    async def check_running(self) -> bool:
        """
        Return whether the server process is still alive and expected to be responsive.

        This method must:
          - Return True if the underlying process (or container, etc.) is running.
          - Return False if it has exited, crashed, or was never started.
          - Work even if called immediately after `start()` or during shutdown.

        It is *not* required to check the network port or HTTP health, only process-level liveness.
        If you are using subprocesses, check returncode or PID validity.
        This is used during graceful shutdown polling in `stop_and_wait`.
        """
        ...

    async def wait_until_ready(self, timeout: float = 10.0) -> None:
        async with aiohttp.ClientSession() as session:
            for _ in range(int(timeout / 0.5)):
                try:
                    async with session.get(f"http://localhost:{self.port}/models") as r:
                        if r.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.5)
            raise TimeoutError("Server did not become ready in time.")

    async def start_and_wait(self) -> None:
        self.status = ServerStatus.STARTING
        await self.start()
        await self.wait_until_ready()
        self.status = ServerStatus.RUNNING

    async def stop_and_wait(self) -> None:
        self.status = ServerStatus.STOPPING
        await self.stop()
        while await self.check_running():
            await asyncio.sleep(0.01)
        self.status = ServerStatus.STOPPED
