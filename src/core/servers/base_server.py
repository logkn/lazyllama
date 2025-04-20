from abc import ABC, abstractmethod
import asyncio
from enum import StrEnum


class ServerStatus(StrEnum):
    """
    Enum representing the status of a server.

    - STARTING: The command to start the server has been sent.
    - RUNNING: The server is running, and requests can be sent.
    - STOPPING: The command to stop the server has been sent.
    - STOPPED: The server has been stopped, and all resources freed.
    """

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class BaseServer(ABC):
    status: ServerStatus = ServerStatus.STOPPED

    @abstractmethod
    async def start(self) -> None:
        """
        Run the command to start the server.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Run the command to stop the server.
        """
        pass

    @abstractmethod
    async def check_running(self) -> bool:
        """
        Check if the server is running.
        Returns True if the server is running, False otherwise.
        """
        pass

    async def start_and_wait(self) -> None:
        """
        Start the server and wait until it is running.
        """
        self.status = ServerStatus.STARTING
        await self.start()
        while not await self.check_running():
            await asyncio.sleep(0.5)
        self.status = ServerStatus.RUNNING

    async def stop_and_wait(self) -> None:
        """
        Stop the server and wait until it is stopped.
        """
        self.status = ServerStatus.STOPPING
        await self.stop()
        while await self.check_running():
            await asyncio.sleep(0.5)
        self.status = ServerStatus.STOPPED
