import asyncio
from pathlib import Path
from typing import Optional, override

from src.core.models.alias import Alias

from .command_server import CommandServer


class LlamaCppServer(CommandServer):
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self._process: Optional[asyncio.subprocess.Process] = None

    @override
    def build_command(self, port: int, alias: Alias) -> str:
        """
        Construct the full shell command to launch llama.cpp's server binary
        with the appropriate model and context size.
        """
        model_path = Path(f"/models/{alias.model.model_name}.gguf")
        args = [
            "./server",
            f"--model {model_path}",
            f"--ctx-size {alias.n_ctx}",
            f"--port {port}",
        ]
        if alias.command_params:
            args.extend(alias.command_params)

        return " ".join(args)
