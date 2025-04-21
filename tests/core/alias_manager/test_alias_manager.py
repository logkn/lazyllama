import pytest
from src.core.models.alias import Alias, Backend
from src.core.config.config import GlobalConfig, ProjectConfig, AliasConfig
from src.core.alias_manager import AliasManager, AliasAlreadyExistsError


@pytest.fixture
def alias_manager():
    return AliasManager()


@pytest.fixture
def sample_alias():
    from src.core.models.alias import AliasModel
    return Alias(
        name="test-alias",
        model=AliasModel(
            model_name="test-alias-model",
            backend=Backend.LLAMACPP
        )
    )


@pytest.fixture
def different_config_alias():
    from src.core.models.alias import AliasModel
    return Alias(
        name="test-alias",
        model=AliasModel(
            model_name="test-alias-model",
            backend=Backend.OLLAMA
        )
    )


def test_add_alias_same_name_and_config(alias_manager, sample_alias):
    alias_manager.add_alias(sample_alias)
    # Should not raise an error when adding same alias again
    alias_manager.add_alias(sample_alias)


def test_add_alias_same_name_different_config(
    alias_manager, sample_alias, different_config_alias
):
    alias_manager.add_alias(sample_alias)
    with pytest.raises(AliasAlreadyExistsError) as exc_info:
        alias_manager.add_alias(different_config_alias)
    
    assert sample_alias.name in str(exc_info.value)


def test_load_starting_aliases(alias_manager, tmp_path, sample_alias):
    from unittest.mock import patch
    
    # Create temporary config files
    global_config = GlobalConfig()
    project_config = ProjectConfig()
    
    # Add sample alias to both configs
    global_config.aliases["global-alias"] = AliasConfig(
        model_name="llama2",
        backend=Backend.LLAMACPP
    )
    project_config.aliases["project-alias"] = AliasConfig(
        model_name="gpt4", 
        backend=Backend.OLLAMA
    )
    
    # Save configs to temporary files
    global_config_path = tmp_path / "global_config.yaml"
    project_config_path = tmp_path / "project_config.yaml"
    
    global_config.save(global_config_path)
    project_config.save(project_config_path)
    
    # Patch the path methods to use our temporary files
    def mock_global_path():
        return global_config_path
        
    def mock_project_path():
        return project_config_path
    
    # Reset alias manager and reload aliases with patched config paths
    alias_manager._aliases.clear()
    
    with patch.object(GlobalConfig, 'path', mock_global_path):
        with patch.object(ProjectConfig, 'path', mock_project_path):
            alias_manager._load_starting_aliases()
    
    assert "global-alias" in alias_manager._aliases
    assert "project-alias" in alias_manager._aliases


def test_get_alias(alias_manager, sample_alias):
    alias_manager.add_alias(sample_alias)
    retrieved_alias = alias_manager.get_alias(sample_alias.name)
    
    assert retrieved_alias == sample_alias
