from enum import StrEnum, auto
from pydantic import BaseModel


class Backend(StrEnum):
    llama_cpp = auto()
    ollama = auto()


class AliasModel(BaseModel):
    model_name: str
    backend: Backend


class Alias(BaseModel):
    name: str
    model: AliasModel
    n_ctx: int = 4096
    command_params: list[str]
