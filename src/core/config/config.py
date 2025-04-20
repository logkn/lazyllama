from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Self, override
from pydantic import BaseModel

from src.constants import GLOBAL_CONFIG_PATH
from src.core.models.alias import Alias, AliasModel, Backend


class BaseConfig(BaseModel, meta=ABCMeta):
    aliases: dict[str, "AliasConfig"] = {}

    def _get_aliases(self) -> list[Alias]:
        return [alias.to_alias(name) for name, alias in self.aliases.items()]

    @classmethod
    def get_aliases(cls) -> list[Alias]:
        instance = cls.load()
        if not instance:
            return []
        return instance._get_aliases()

    @classmethod
    def _from_yaml(cls, path: str | Path) -> Self | None:
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            return None

        import yaml

        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def load(cls) -> Self | None:
        return cls._from_yaml(cls.path())

    @staticmethod
    @abstractmethod
    def path() -> str | Path:
        raise NotImplementedError


class GlobalConfig(BaseConfig):
    llamacpp_model_dir: str | None = None

    @staticmethod
    @override
    def path() -> str | Path:
        return GLOBAL_CONFIG_PATH


class ProjectConfig(BaseConfig):
    @staticmethod
    @override
    def path() -> str | Path:
        return Path.cwd() / ".lazyllama.yaml"


class AliasConfig(BaseModel):
    model_name: str
    backend: Backend
    n_ctx: int = 4096
    command_params: list[str] = []

    def to_alias(self, alias_name: str) -> Alias:
        return Alias(
            name=alias_name,
            model=AliasModel(
                model_name=self.model_name,
                backend=self.backend,
            ),
            n_ctx=self.n_ctx,
            command_params=self.command_params,
        )
