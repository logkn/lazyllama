from abc import abstractmethod
from pathlib import Path
from typing import Any, Self, override
from pydantic import BaseModel, Field

from src.constants import GLOBAL_CONFIG_PATH
from src.core.models.alias import Alias, AliasModel, Backend


class BaseConfig(BaseModel):
    aliases: dict[str, "AliasConfig"] = Field(
        default_factory=dict,
        description="A dictionary of aliases, where the key is the alias name and the value is an AliasConfig object.",
    )

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
        
    def save(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
            
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        
        # Add a custom representer for enum values
        def enum_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.value))
        
        # Register enum representer
        yaml.SafeDumper.add_representer(Backend, enum_representer)
        
        # Convert model to dict and handle enum values
        data = self.model_dump()
        
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

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
