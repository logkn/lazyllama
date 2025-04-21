import asyncio
from typing import Any
import lazyllama


class AliasNotFoundError(Exception):
    """Custom exception for alias not found errors."""

    def __init__(self, alias: str):
        super().__init__(f"Alias '{alias}' not found.")


class LazyLlama(lazyllama.OpenAI):
    def __init__(self):
        super().__init__()
        self.chat.completions.create = self._patched_create
        self.beta.chat.completions.parse = self._patched_beta_parse

    def _patched_create(self, *args: Any, model: str, **kwargs: Any) -> Any:
        alias = lazyllama.ALIAS_MANAGER.get_alias(model)
        if not alias:
            raise AliasNotFoundError(model)

        server = asyncio.run(lazyllama.SERVER_MANAGER.get_or_start_server(alias))
        base_url = f"http://localhost:{server.port}/v1"

        proxy = lazyllama.OpenAI(base_url=base_url)
        return proxy.chat.completions.create(*args, model=model, **kwargs)

    def _patched_beta_parse(self, *args: Any, model: str, **kwargs: Any) -> Any:
        alias = lazyllama.ALIAS_MANAGER.get_alias(model)
        if not alias:
            raise AliasNotFoundError(model)
        server = asyncio.run(lazyllama.SERVER_MANAGER.get_or_start_server(alias))
        base_url = f"http://localhost:{server.port}/v1"

        proxy = lazyllama.OpenAI(base_url=base_url)
        return proxy.beta.chat.completions.parse(*args, model=model, **kwargs)


class AsyncLazyLlama(lazyllama.AsyncOpenAI):
    def __init__(self):
        super().__init__()
        self.chat.completions.create = self._patched_create
        self.beta.chat.completions.parse = self._patched_beta_parse

    async def _patched_create(self, *args: Any, model: str, **kwargs: Any) -> Any:
        alias = lazyllama.ALIAS_MANAGER.get_alias(model)
        if not alias:
            raise AliasNotFoundError(model)
        server = await lazyllama.SERVER_MANAGER.get_or_start_server(alias)
        base_url = f"http://localhost:{server.port}/v1"

        proxy = lazyllama.AsyncOpenAI(base_url=base_url)
        return await proxy.chat.completions.create(*args, model=model, **kwargs)

    async def _patched_beta_parse(self, *args: Any, model: str, **kwargs: Any) -> Any:
        alias = lazyllama.ALIAS_MANAGER.get_alias(model)
        if not alias:
            raise AliasNotFoundError(model)
        server = await lazyllama.SERVER_MANAGER.get_or_start_server(alias)
        base_url = f"http://localhost:{server.port}/v1"

        proxy = lazyllama.AsyncOpenAI(base_url=base_url)
        return await proxy.beta.chat.completions.parse(*args, model=model, **kwargs)
