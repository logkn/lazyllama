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
        """Run the command to start the server."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Run the command to stop the server."""
        ...

    @abstractmethod
    async def check_running(self) -> bool:
        """Check if the server is still running."""
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
