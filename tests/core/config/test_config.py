import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from src.core.models.alias import Alias, Backend
from src.core.models.alias import Alias, Backend, AliasModel
from src.core.alias_manager import AliasManager, AliasAlreadyExistsError
from src.core.config.config import (
    BaseConfig,
    GlobalConfig,
    ProjectConfig,
    AliasConfig,
)


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary YAML file with test data"""
    test_data = {
        "llamacpp_model_dir": "/tmp/models",
        "aliases": {
            "test-alias": {
                "model_name": "mixtral-8x7b-32768",
                "backend": "llamacpp",
                "n_ctx": 4096,
                "command_params": ["--param1", "--param2"],
            }
        },
    }

    temp_file = tmp_path / ".lazyllama.yaml"
    with open(temp_file, "w") as f:
        for key, value in test_data.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    return temp_file


def test_global_config_load(temp_yaml_file: Path) -> None:
    """Test loading global config from file"""
    with patch("src.core.config.config.GLOBAL_CONFIG_PATH", new=temp_yaml_file):
        config = GlobalConfig.load()

    assert isinstance(config, GlobalConfig)
    assert config.llamacpp_model_dir == "/tmp/models"
    assert len(config.aliases) == 1


def test_global_config_load_nonexistent() -> None:
    """Test loading global config when file doesn't exist"""
    with patch("src.core.config.config.GLOBAL_CONFIG_PATH", new=Path("/nonexistent")):
        config = GlobalConfig.load()

    assert config is None


def test_project_config_load(temp_yaml_file: Path) -> None:
    """Test loading project config from current directory"""
    with patch("src.core.config.config.Path.cwd", return_value=temp_yaml_file.parent):
        config = ProjectConfig.load()

    assert isinstance(config, ProjectConfig)
    assert len(config.aliases) == 1


def test_project_config_load_nonexistent() -> None:
    """Test loading project config when file doesn't exist"""
    with patch("src.core.config.config.Path.cwd", return_value=Path("/nonexistent")):
        config = ProjectConfig.load()

    assert config is None


def test_alias_config_to_alias() -> None:
    """Test conversion from AliasConfig to Alias model"""
    alias_config = AliasConfig(
        model_name="mixtral-8x7b-32768",
        backend=Backend.LLAMACPP,
        n_ctx=4096,
        command_params=["--param1", "--param2"],
    )

    alias = alias_config.to_alias("test-alias")

    assert isinstance(alias, Alias)
    assert alias.name == "test-alias"
    assert alias.model.model_name == "mixtral-8x7b-32768"
    assert alias.model.backend == Backend.LLAMACPP
    assert alias.n_ctx == 4096
    assert alias.command_params == ["--param1", "--param2"]


def test_project_config_get_aliases(temp_yaml_file: Path) -> None:
    """Test getting aliases from project config"""
    with patch("src.core.config.config.Path.cwd", return_value=temp_yaml_file.parent):
        aliases = ProjectConfig.get_aliases()

    assert len(aliases) == 1
    assert isinstance(aliases[0], Alias)


def test_base_config_get_aliases_no_config() -> None:
    """Test getting aliases when no config exists"""
    with patch("src.core.config.config.Path.exists", return_value=False):
        aliases = GlobalConfig.get_aliases()

    assert len(aliases) == 0

def test_add_duplicate_alias_different_config(alias_manager: AliasManager, alias: Alias):
    """Test adding duplicate alias with different configuration"""
    alias_manager.add_alias(alias)
    with pytest.raises(AliasAlreadyExistsError) as e:
        alias_manager.add_alias(Alias(name=alias.name, model=AliasModel(model_name="different_model", backend=Backend.LLAMACPP)))
    
    assert f"Alias '{alias.name}' already exists with different configuration:" in str(e.value)
    assert "Existing alias" in str(e.value)
    assert "New alias" in str(e.value)
