import os
from pathlib import Path

if os.getenv("DEV") is not None:
    _global_config = Path("dev/global_config.yaml")
    _resource_cache_path = Path("dev/resource_measurements.json")
else:
    _global_config = Path.home() / ".config" / "lazyllama" / "config.yaml"
    _resource_cache_path = (
        Path.home() / ".cache" / "lazyllama" / "resource_measurements.json"
    )

GLOBAL_CONFIG_PATH = _global_config
RESOURCE_CACHE_PATH = _resource_cache_path
