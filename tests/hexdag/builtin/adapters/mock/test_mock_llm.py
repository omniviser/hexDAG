"""Tests for MockLLM implementation."""

import pytest

from hexdag.builtin.adapters.mock.mock_llm import MockLLM


class TestMockLLM:
    """Test cases for MockLLM."""

    @pytest.mark.asyncio
    async def test_default_response(self):
        """Test that MockLLM returns the default response."""
        mock_llm = MockLLM()
        response = await mock_llm.aresponse([{"role": "user", "content": "Hello"}])
        assert response == '{"result": "Mock response"}'

    @pytest.mark.asyncio
    async def test_single_custom_response(self):
        """Test MockLLM with a single custom response."""
        mock_llm = MockLLM(responses="Custom response")
        response = await mock_llm.aresponse([{"role": "user", "content": "Test"}])
        assert response == "Custom response"

    @pytest.mark.asyncio
    async def test_multiple_responses(self):
        """Test MockLLM with multiple responses."""
        responses = ["First response", "Second response", "Third response"]
        mock_llm = MockLLM(responses=responses)

        response1 = await mock_llm.aresponse([{"role": "user", "content": "Test 1"}])
        response2 = await mock_llm.aresponse([{"role": "user", "content": "Test 2"}])
        response3 = await mock_llm.aresponse([{"role": "user", "content": "Test 3"}])

        assert response1 == "First response"
        assert response2 == "Second response"
        assert response3 == "Third response"

    @pytest.mark.asyncio
    async def test_exhausted_responses_repeat_last(self):
        """Test that exhausted responses repeat the last one."""
        responses = ["First", "Second"]
        mock_llm = MockLLM(responses=responses)

        # Use up all responses
        await mock_llm.aresponse([{"role": "user", "content": "Test 1"}])
        await mock_llm.aresponse([{"role": "user", "content": "Test 2"}])

        # Should repeat the last response
        response = await mock_llm.aresponse([{"role": "user", "content": "Test 3"}])
        assert response == "Second"

    @pytest.mark.asyncio
    async def test_last_messages_tracking(self):
        """Test that last_messages is tracked for testing."""
        mock_llm = MockLLM(responses="Test response")
        messages = [{"role": "user", "content": "Test message"}]

        await mock_llm.aresponse(messages)
        assert mock_llm.last_messages == messages

    def test_reset_functionality(self):
        """Test that reset clears the mock state."""
        mock_llm = MockLLM(responses=["First", "Second"])

        # Make a call
        import asyncio

        asyncio.run(mock_llm.aresponse([{"role": "user", "content": "Test"}]))

        assert mock_llm.call_count == 1
        assert mock_llm.last_messages is not None

        # Reset
        mock_llm.reset()
        assert mock_llm.call_count == 0
        assert mock_llm.last_messages is None

    @pytest.mark.asyncio
    async def test_responses_parameter_override(self):
        """Test that responses parameter overrides config."""
        # Test that responses can be passed directly
        mock_llm = MockLLM(responses=["Direct response 1", "Direct response 2"])

        response1 = await mock_llm.aresponse([{"role": "user", "content": "Test"}])
        response2 = await mock_llm.aresponse([{"role": "user", "content": "Test"}])

        assert response1 == "Direct response 1"
        assert response2 == "Direct response 2"
