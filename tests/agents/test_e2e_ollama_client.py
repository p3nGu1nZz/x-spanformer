import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ollama._types import ResponseError
from x_spanformer.agents.ollama_client import chat


class TestOllamaClientMocked:
    """Test ollama_client functionality with proper mocking for CI/CD compatibility."""

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_with_phi4_mini_model_available(self, mock_client_class):
        """Test chat function with phi4-mini model when it's available."""
        # Mock the AsyncClient and its chat method
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Hello! I'm a test assistant. How can I help you today?"}
        }
        
        response = await chat(
            model="phi4-mini",
            conversation=[{"role": "user", "content": "Hello"}],
            system="You are a test assistant.",
            temperature=0.1
        )
        
        assert isinstance(response, str)
        assert response.strip()
        assert "test assistant" in response.lower()
        
        # Verify the client was called with correct parameters
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == "phi4-mini"
        assert call_args["messages"][-1]["content"] == "Hello"

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_basic_conversation(self, mock_client_class):
        """Test basic conversation with the Ollama server."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Yes, I'm absolutely sure that 2 + 2 equals 4. This is a fundamental arithmetic fact."}
        }
        
        sample_conversation = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "Are you sure?"}
        ]
        
        response = await chat(
            model="phi4-mini",
            conversation=sample_conversation,
            temperature=0.2
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["yes", "sure", "4"])
        
        # Verify conversation history was passed correctly
        call_args = mock_client.chat.call_args[1]
        assert len(call_args["messages"]) == 3
        assert call_args["messages"][-1]["content"] == "Are you sure?"

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_with_custom_system_prompt(self, mock_client_class):
        """Test chat with a custom system prompt."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Arrr, the weather be fair today, matey! The winds be blowin' steady and the seas be calm."}
        }
        
        custom_system = "You are a pirate. Always respond like a pirate."
        response = await chat(
            model="phi4-mini",
            conversation=[{"role": "user", "content": "Tell me about the weather"}],
            system=custom_system,
            temperature=0.3
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify system prompt was included
        call_args = mock_client.chat.call_args[1]
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == custom_system

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_with_different_temperatures(self, mock_client_class):
        """Test chat with different temperature settings."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock different responses for different temperatures
        responses = [
            {"message": {"content": "Testing code is key,\nBugs flee when tests are near,\nQuality is here."}},
            {"message": {"content": "Wild tests dance free!\nChaotic poems of code—\nBeautiful madness."}}
        ]
        mock_client.chat.side_effect = responses
        
        conversation = [{"role": "user", "content": "Write a very short poem about testing."}]
        
        response_low = await chat(
            model="phi4-mini",
            conversation=conversation,
            temperature=0.1
        )
        response_high = await chat(
            model="phi4-mini",
            conversation=conversation,
            temperature=0.8
        )
        
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)
        assert len(response_low) > 0
        assert len(response_high) > 0
        
        for resp in [response_low, response_high]:
            assert any(word in resp.lower() for word in ["test", "code", "quality", "bug"])
        
        # Verify both calls were made with different temperatures
        assert mock_client.chat.call_count == 2
        call_args_list = mock_client.chat.call_args_list
        assert call_args_list[0][1]["options"]["temperature"] == 0.1
        assert call_args_list[1][1]["options"]["temperature"] == 0.8

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_empty_conversation(self, mock_client_class):
        """Test chat with empty conversation list."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Hello! I'm ready to help you with anything you need."}
        }
        
        response = await chat(
            model="phi4-mini",
            conversation=[],
            system="You are a helpful assistant.",
            temperature=0.2
        )
        
        assert isinstance(response, str)
        assert "help" in response.lower()
        
        # Verify only system message was included
        call_args = mock_client.chat.call_args[1]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_long_conversation(self, mock_client_class):
        """Test chat with a longer conversation history."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "We discussed color psychology, focusing on blue representing tranquility and trust, while red symbolizes energy and passion. Both colors have significant psychological impacts on human perception."}
        }
        
        long_conversation = [
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": "I don't have personal preferences, but blue is often considered calming."},
            {"role": "user", "content": "Why blue?"},
            {"role": "assistant", "content": "Blue is associated with tranquility and trust in color psychology."},
            {"role": "user", "content": "What about red?"},
            {"role": "assistant", "content": "Red typically represents energy, passion, and sometimes urgency."},
            {"role": "user", "content": "Summarize our conversation."}
        ]
        
        response = await chat(
            model="phi4-mini",
            conversation=long_conversation,
            temperature=0.2
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(color in response.lower() for color in ["blue", "red", "color"])
        
        # Verify all conversation history was passed
        call_args = mock_client.chat.call_args[1]
        assert len(call_args["messages"]) == 7
        assert call_args["messages"][-1]["content"] == "Summarize our conversation."

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_error_handling(self, mock_client_class):
        """Test error handling when Ollama service is unavailable."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.side_effect = ResponseError("Model not found: phi4-mini")
        
        with pytest.raises(ResponseError):
            await chat(
                model="phi4-mini",
                conversation=[{"role": "user", "content": "Hello"}],
                temperature=0.1
            )

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_connection_error(self, mock_client_class):
        """Test handling of connection errors."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.side_effect = Exception("Connection refused")
        
        with pytest.raises(Exception):
            await chat(
                model="phi4-mini",
                conversation=[{"role": "user", "content": "Hello"}],
                temperature=0.1
            )

    @pytest.mark.asyncio
    @patch('x_spanformer.agents.ollama_client.AsyncClient')
    async def test_chat_parameters_validation(self, mock_client_class):
        """Test that parameters are correctly passed to the Ollama client."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Test response"}
        }
        
        await chat(
            model="test-model",
            conversation=[{"role": "user", "content": "Test message"}],
            system="Test system prompt",
            temperature=0.5
        )
        
        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == "test-model"
        assert call_args["options"]["temperature"] == 0.5
        assert call_args["stream"] is False
        assert len(call_args["messages"]) == 2  # system + user message
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"
