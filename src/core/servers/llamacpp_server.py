import asyncio
from pathlib import Path
from typing import override

from src.core.config.config import GlobalConfig
from src.core.models.alias import Alias

from .command_server import CommandServer


class LlamaCppServer(CommandServer):
    def __init__(self, alias: Alias, port: int):
        super().__init__(alias, port)
        self._process: asyncio.subprocess.Process | None = None

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """
        Resolve the model path based on the model name.
        """
        # if model_name is a full path ending in .gguf, use it directly
        if model_name.endswith(".gguf"):
            return model_name

        global_config = GlobalConfig.load()
        if not global_config:
            raise ValueError(
                "A llamacpp model was aliased by filename, but no global config was found."
            )

        # prepend the model name with the llamacpp directory
        llamacpp_dir = global_config.llamacpp_model_dir
        if not llamacpp_dir:
            raise ValueError(
                "A llamacpp model was aliased by filename, but no llamacpp directory was found in the global config."
            )

        if not model_name.endswith(".gguf"):
            model_name += ".gguf"

        # check if the model file exists
        model_path = Path(llamacpp_dir) / model_name

        if not model_path.exists():
            raise ValueError(
                f"The model file {model_path} does not exist. Please check the model name and path."
            )

        return str(model_path)

    @override
    def build_command(self, port: int, alias: Alias) -> str:
        """
        Construct the full shell command to launch llama.cpp's server binary
        with the appropriate model and context size.
        """

        model_path = self._resolve_model_path(alias.model.model_name)
        args = [
            "llama-server",
            f"--model {model_path}",
            f"--ctx-size {alias.n_ctx}",
            f"--port {port}",
        ]
        if alias.command_params:
            args.extend(alias.command_params)

        return " ".join(args)
