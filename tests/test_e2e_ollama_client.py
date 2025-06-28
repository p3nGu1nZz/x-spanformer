"""
End-to-end tests for the Ollama client that make real calls to a running Ollama server.
These tests require the phi4:mini model to be loaded, or they will fail.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from x_spanformer.agents.ollama_client import chat


@pytest.mark.xdist_group(name="e2e")
class TestOllamaClientE2E:
    """End-to-end tests for the Ollama client making real server calls."""
    
    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation for testing."""
        return [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "Are you sure?"}
        ]
    
    @pytest.mark.asyncio
    async def test_chat_with_phi4_mini_model_available(self):
        """Test chat function with phi4-mini model when it's available."""
        # First check if the model is available by making a test call
        try:
            response = await chat(
                model="phi4-mini",
                conversation=[{"role": "user", "content": "Hello"}],
                system="You are a test assistant.",
                temperature=0.1
            )
            
            # If we get here, the model is available
            assert isinstance(response, str)
            assert len(response) > 0
            assert response.strip()  # Not just whitespace
            
        except Exception as e:
            # Model not available - fail the test with clear message
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")
    
    @pytest.mark.asyncio
    async def test_chat_basic_conversation(self, sample_conversation):
        """Test basic conversation with the Ollama server."""
        try:
            response = await chat(
                model="phi4-mini",
                conversation=sample_conversation,
                temperature=0.2
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Should contain some kind of confirmation or answer
            assert any(word in response.lower() for word in ["yes", "correct", "4", "sure"])
            
        except Exception as e:
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")
    
    @pytest.mark.asyncio
    async def test_chat_with_custom_system_prompt(self):
        """Test chat with a custom system prompt."""
        custom_system = "You are a pirate. Always respond like a pirate."
        
        try:
            response = await chat(
                model="phi4-mini",
                conversation=[{"role": "user", "content": "Tell me about the weather"}],
                system=custom_system,
                temperature=0.3
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Should show some pirate-like language (though not guaranteed with low temp)
            
        except Exception as e:
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")
    
    @pytest.mark.asyncio
    async def test_chat_with_different_temperatures(self):
        """Test chat with different temperature settings."""
        conversation = [{"role": "user", "content": "Write a very short poem about testing."}]
        
        try:
            # Low temperature - should be more deterministic
            response_low = await chat(
                model="phi4-mini",
                conversation=conversation,
                temperature=0.1
            )
            
            # Higher temperature - should be more creative
            response_high = await chat(
                model="phi4-mini",
                conversation=conversation,
                temperature=0.8
            )
            
            assert isinstance(response_low, str)
            assert isinstance(response_high, str)
            assert len(response_low) > 0
            assert len(response_high) > 0
            
            # Both should contain words related to testing or poetry
            for response in [response_low, response_high]:
                assert any(word in response.lower() for word in ["test", "code", "bug", "poem", "line"])
    
        except Exception as e:
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")

    @pytest.mark.asyncio
    async def test_chat_empty_conversation(self):
        """Test chat with empty conversation list."""
        try:
            response = await chat(
                model="phi4-mini",
                conversation=[],
                system="You are a helpful assistant.",
                temperature=0.2
            )

            # Should still work with just system message
            assert isinstance(response, str)
            # Response might be empty or a greeting, but should not error

        except Exception as e:
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")

    @pytest.mark.asyncio
    async def test_chat_long_conversation(self):
        """Test chat with a longer conversation history."""
        long_conversation = [
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": "I don't have personal preferences, but blue is often considered calming."},
            {"role": "user", "content": "Why blue?"},
            {"role": "assistant", "content": "Blue is associated with tranquility and trust in color psychology."},
            {"role": "user", "content": "What about red?"},
            {"role": "assistant", "content": "Red typically represents energy, passion, and sometimes urgency."},
            {"role": "user", "content": "Summarize our conversation."}
        ]
    
        try:
            response = await chat(
                model="phi4-mini",
                conversation=long_conversation,
                temperature=0.2
            )
    
            assert isinstance(response, str)
            assert len(response) > 0
            # Should mention colors in the summary
            assert any(color in response.lower() for color in ["blue", "red", "color"])
    
        except Exception as e:
            pytest.fail(f"phi4-mini model not loaded in Ollama server. Error: {e}")


class TestOllamaClientMocked:
    """Tests for Ollama client behavior with mocked server responses."""
    
    @pytest.mark.asyncio
    async def test_chat_response_parsing(self):
        """Test that the chat function correctly parses server responses."""
        mock_response = {
            "message": {
                "content": "This is a test response from the mocked server."
            }
        }
        
        with patch('x_spanformer.agents.ollama_client.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response
            
            response = await chat(
                model="phi4-mini",
                conversation=[{"role": "user", "content": "Test message"}],
                temperature=0.2
            )
            
            assert response == "This is a test response from the mocked server."
            mock_client.chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_message_construction(self):
        """Test that messages are constructed correctly."""
        mock_response = {"message": {"content": "Mock response"}}
        
        with patch('x_spanformer.agents.ollama_client.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response
            
            conversation = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"}
            ]
            
            await chat(
                model="test-model",
                conversation=conversation,
                system="Custom system prompt",
                temperature=0.5
            )
            
            # Verify the call was made with correct parameters
            call_args = mock_client.chat.call_args
            assert call_args[1]["model"] == "test-model"
            assert call_args[1]["options"]["temperature"] == 0.5
            
            # Check message structure
            messages = call_args[1]["messages"]
            assert len(messages) == 4  # system + 3 conversation messages
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Custom system prompt"
            assert messages[1:] == conversation
    
    @pytest.mark.asyncio
    async def test_chat_default_system_prompt(self):
        """Test that default system prompt is used when none provided."""
        mock_response = {"message": {"content": "Mock response"}}
        
        with patch('x_spanformer.agents.ollama_client.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response
            
            await chat(
                model="test-model",
                conversation=[{"role": "user", "content": "Test"}],
                temperature=0.2
            )
            
            call_args = mock_client.chat.call_args
            messages = call_args[1]["messages"]
            
            # Should use default system prompt from constants
            from x_spanformer.agents.constants import DEFAULT_SYSTEM
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == DEFAULT_SYSTEM

    @pytest.mark.asyncio
    async def test_chat_mocked_response(self, mock_client, sample_conversation):
        """Test that chat function returns the mocked response."""
        mock_client.chat.return_value = {
            "message": {"content": "mocked response"}
        }
        
        response = await chat(
            model="mock-model",
            conversation=sample_conversation
        )
        
        assert response == "mocked response"
        mock_client.chat.assert_called_once()
        # Verify that the arguments passed to the mock are correct
        args, kwargs = mock_client.chat.call_args
        assert kwargs["model"] == "mock-model"
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1:] == sample_conversation

    @pytest.mark.asyncio
    async def test_chat_with_custom_mocked_system_prompt(self, mock_client):
        """Test chat with a custom system prompt and mocked response."""
        custom_system = "You are a test bot."
        mock_client.chat.return_value = {
            "message": {"content": "Acknowledged"}
        }

        response = await chat(
            model="mock-model",
            conversation=[{"role": "user", "content": "Hi"}],
            system=custom_system
        )

        assert response == "Acknowledged"
        args, kwargs = mock_client.chat.call_args
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == custom_system

    @pytest.mark.asyncio
    async def test_chat_handles_api_error(self, mock_client):
        """Test that chat function handles API errors gracefully."""
        mock_client.chat.side_effect = Exception("API connection failed")

        with pytest.raises(Exception, match="API connection failed"):
            await chat(
                model="error-model",
                conversation=[{"role": "user", "content": "This will fail"}]
            )

    @pytest.mark.asyncio
    async def test_chat_empty_conversation_mocked(self, mock_client):
        """Test chat with an empty conversation list and mocked response."""
        mock_client.chat.return_value = {
            "message": {"content": "Hello there!"}
        }

        response = await chat(
            model="mock-model",
            conversation=[]
        )

        assert response == "Hello there!"
        args, kwargs = mock_client.chat.call_args
        assert len(kwargs["messages"]) == 1  # Only system message
        assert kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_chat_with_different_mocked_temperatures(self, mock_client):
        """Test that temperature is correctly passed to the client."""
        mock_client.chat.return_value = {"message": {"content": "temp check"}}

        await chat(
            model="temp-model",
            conversation=[{"role": "user", "content": "test"}],
            temperature=0.9
        )

        args, kwargs = mock_client.chat.call_args
        assert kwargs["options"]["temperature"] == 0.9

        await chat(
            model="temp-model",
            conversation=[{"role": "user", "content": "test"}],
            temperature=0.0
        )

        args, kwargs = mock_client.chat.call_args
        assert kwargs["options"]["temperature"] == 0.0
