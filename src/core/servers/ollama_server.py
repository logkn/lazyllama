import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import override

import aiohttp
import psutil

from src.core.models.alias import Alias
from src.core.servers.command_server import CommandServer


class OllamaServer(CommandServer):
    def __init__(self, alias: Alias, port: int):
        """
        Initialize the OllamaServer.

        Set up model alias and port, create a temporary directory for Modelfile,
        and initialize process and PID tracking attributes.
        """
        super().__init__(alias, port)
        self._process: asyncio.subprocess.Process | None = None
        self._tag = f"{alias.model.model_name}-ctx{alias.n_ctx}-{port}"
        self._modelfile_dir = tempfile.mkdtemp()
        self._ollama_pid: int | None = None

    @override
    def build_command(self, port: int, alias: Alias) -> str:
        return f"OLLAMA_HOST=localhost:{port} ollama serve"

    def _modelfile_path(self) -> Path:
        """
        Return the full path to the Modelfile in the temporary directory.
        """
        return Path(self._modelfile_dir) / "Modelfile"

    def _generate_modelfile(self) -> None:
        """
        Generate the Modelfile for the Ollama model.

        Write the base model and context size parameters to the Modelfile.
        """
        base_model = self.alias.model.model_name
        with open(self._modelfile_path(), "w") as f:
            f.write(f"FROM {base_model}\n")
            f.write(f"PARAMETER n_ctx {self.alias.n_ctx}\n")

    async def _wait_for_api_ready(self) -> None:
        """
        Poll the Ollama API status endpoint until it's ready.

        Repeatedly GET /v1/status on localhost until a 200 response is received,
        or raise TimeoutError after a timeout (~15s).
        """
        url = f"http://localhost:{self.port}/v1/status"
        async with aiohttp.ClientSession() as session:
            for _ in range(30):  # ~15s max
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("Ollama API did not become ready")

    async def _ollama_create(self) -> None:
        """
        Create a new Ollama model instance from the Modelfile.

        Generates the Modelfile and runs `ollama create` with the model tag.
        """
        self._generate_modelfile()
        cmd = f"ollama create {self._tag} -f {self._modelfile_path()}"
        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()

    async def _load_model_via_dummy_request(self) -> None:
        """
        Load the Ollama model into memory via a dummy API request.

        Sends a non-streaming chat completion request and polls until the model
        responds successfully, or raise TimeoutError after a timeout (~10s).
        """
        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = {
            "model": self._tag,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            for _ in range(20):  # ~10s
                try:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("Model did not load via dummy request")

    @override
    async def start(self) -> None:
        # Step 1: Start ollama serve
        cmd = self.build_command(self.port, self.alias)
        self._process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._ollama_pid = self._process.pid

        # Step 2: Wait until API is ready
        await self._wait_for_api_ready()

        # Step 3: Create the temp model
        await self._ollama_create()

        # Step 4: Load it into memory
        await self._load_model_via_dummy_request()

    @override
    async def stop(self) -> None:
        if self._ollama_pid is not None:
            try:
                parent = psutil.Process(self._ollama_pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
            self._ollama_pid = None
        self._process = None
        shutil.rmtree(self._modelfile_dir, ignore_errors=True)

    @override
    async def check_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.returncode is None
