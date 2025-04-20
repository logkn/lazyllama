from src.core.models.alias import Alias
from src.core.config.config import GlobalConfig, ProjectConfig


class AliasAlreadyExistsError(Exception):
    def __init__(self, new_alias: Alias, existing_alias: Alias):
        message = (
            f"Alias '{new_alias.name}' already exists with different configuration:\n"
        )
        message += f"Existing alias ({existing_alias.backend}): {existing_alias}\n"
        message += f"New alias ({new_alias.backend}): {new_alias}"
        super().__init__(message)


class AliasManager:
    def __init__(self):
        self._aliases: dict[str, Alias] = {}
        self._load_starting_aliases()

    def add_alias(self, alias: Alias) -> None:
        if alias.name in self._aliases:
            existing_alias = self._aliases[alias.name]
            if existing_alias != alias:
                raise AliasAlreadyExistsError(alias, existing_alias)
            return
        self._aliases[alias.name] = alias

    def _add_aliases(self, aliases: list[Alias]) -> None:
        for alias in aliases:
            self.add_alias(alias)

    def _load_starting_aliases(self):
        global_aliases = GlobalConfig.get_aliases()
        project_aliases = ProjectConfig.get_aliases()
        self._add_aliases(global_aliases)
        self._add_aliases(project_aliases)

    def get_alias(self, name: str) -> Alias | None:
        return self._aliases.get(name)
