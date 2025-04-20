import os
from pathlib import Path

if os.getenv("DEV") is not None:
    _global_config = Path("dev/global_config.yaml")
else:
    _global_config = Path.home() / ".config" / "lazyllama" / "config.yaml"

GLOBAL_CONFIG_PATH = _global_config
