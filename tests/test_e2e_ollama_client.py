import pytest
from ollama._types import ResponseError
from x_spanformer.agents.ollama_client import chat

@ pytest.mark.asyncio
async def test_chat_with_phi4_mini_model_available():
    """Test chat function with phi4-mini model when it's available."""
    try:
        response = await chat(
            model="phi4-mini",
            conversation=[{"role": "user", "content": "Hello"}],
            system="You are a test assistant.",
            temperature=0.1
        )
        assert isinstance(response, str)
        assert response.strip()
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")

@ pytest.mark.asyncio
async def test_chat_basic_conversation():
    """Test basic conversation with the Ollama server."""
    sample_conversation = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user", "content": "Are you sure?"}
    ]
    try:
        response = await chat(
            model="phi4-mini",
            conversation=sample_conversation,
            temperature=0.2
        )
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["yes", "correct", "4", "sure"])
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")

@ pytest.mark.asyncio
async def test_chat_with_custom_system_prompt():
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
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")

@ pytest.mark.asyncio
async def test_chat_with_different_temperatures():
    """Test chat with different temperature settings."""
    conversation = [{"role": "user", "content": "Write a very short poem about testing."}]
    try:
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
            assert any(word in resp.lower() for word in ["test", "poem", "line"])
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")

@ pytest.mark.asyncio
async def test_chat_empty_conversation():
    """Test chat with empty conversation list."""
    try:
        response = await chat(
            model="phi4-mini",
            conversation=[],
            system="You are a helpful assistant.",
            temperature=0.2
        )
        assert isinstance(response, str)
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")

@ pytest.mark.asyncio
async def test_chat_long_conversation():
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
        assert any(color in response.lower() for color in ["blue", "red", "color"])
    except ResponseError as e:
        pytest.skip(f"phi4-mini model not available: {e}")
