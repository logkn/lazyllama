import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.client.lazyllama import AliasNotFoundError, AsyncLazyLlama, LazyLlama


@pytest.fixture
def alias():
    class DummyAlias:
        name = "mistral"
        model = "mistral"
        n_ctx = 2048
        command_params = []

    return DummyAlias()


@pytest.mark.asyncio
async def test_async_create_forwards_to_correct_server(alias):
    with (
        patch("lazyllama.ALIAS_MANAGER.get_alias", return_value=alias),
        patch(
            "lazyllama.SERVER_MANAGER.get_or_start_server", new_callable=AsyncMock
        ) as mock_get_server,
        patch("lazyllama.AsyncOpenAI") as mock_openai,
    ):
        mock_get_server.return_value.port = 1234
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value={"mock": "response"}
        )

        client = AsyncLazyLlama()
        response = await client.chat.completions.create(
            model="mistral", messages=[{"role": "user", "content": "Hi"}]
        )

        assert response == {"mock": "response"}
        mock_openai.assert_called_with(base_url="http://localhost:1234/v1")
        mock_instance.chat.completions.create.assert_awaited_once()


def test_sync_create_forwards_to_correct_server(alias):
    with (
        patch("lazyllama.ALIAS_MANAGER.get_alias", return_value=alias),
        patch(
            "lazyllama.SERVER_MANAGER.get_or_start_server",
            return_value=MagicMock(port=4321),
        ),
        patch("lazyllama.OpenAI") as mock_openai,
    ):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = {"mock": "response"}

        client = LazyLlama()
        response = client.chat.completions.create(
            model="mistral", messages=[{"role": "user", "content": "Hi"}]
        )

        assert response == {"mock": "response"}
        mock_openai.assert_called_with(base_url="http://localhost:4321/v1")
        mock_instance.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_async_raises_for_missing_alias():
    with patch("lazyllama.ALIAS_MANAGER.get_alias", return_value=None):
        client = AsyncLazyLlama()
        with pytest.raises(AliasNotFoundError):
            await client.chat.completions.create(model="unknown", messages=[])


def test_sync_raises_for_missing_alias():
    with patch("lazyllama.ALIAS_MANAGER.get_alias", return_value=None):
        client = LazyLlama()
        with pytest.raises(AliasNotFoundError):
            client.chat.completions.create(model="unknown", messages=[])
