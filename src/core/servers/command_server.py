import asyncio
from abc import ABC, abstractmethod
from typing import override

from src.core.models.alias import Alias

from .base_server import BaseServer


class CommandServer(BaseServer, ABC):
    process: asyncio.subprocess.Process | None = None

    @abstractmethod
    def build_command(self, port: int, alias: Alias) -> str:
        """
        Return the full shell command to launch the server.
        This should interpolate `port` and use any fields from `alias`.
        """
        ...

    @override
    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return  # Already running

        command = self.build_command(self.port, self.alias)
        self.process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    @override
    async def stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None

    @override
    async def check_running(self) -> bool:
        return self.process is not None and self.process.returncode is None
