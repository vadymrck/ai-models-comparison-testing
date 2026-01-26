"""
Error handling tests.
Verify graceful handling of rate limits, invalid models, malformed requests.
"""

import time
from unittest.mock import Mock

import pytest
from config import OPENAI_MODEL
from helpers import call_with_delay
from openai import BadRequestError, NotFoundError, RateLimitError


def test_handles_rate_limit_error():
    """Gracefully handle rate limit errors with retry logic"""

    mock_client = Mock()
    mock_success_response = Mock()
    mock_success_response.choices = [Mock()]
    mock_success_response.choices[0].message.content = "positive"

    mock_response = Mock()
    mock_response.status_code = 429
    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    mock_client.chat.completions.create.side_effect = [
        rate_limit_error,
        mock_success_response,
    ]

    start_time = time.time()

    result = call_with_delay(
        mock_client, model=OPENAI_MODEL, messages=[{"role": "user", "content": "Test"}]
    )

    elapsed = time.time() - start_time

    assert result == mock_success_response, "Should return success after retry"
    assert elapsed >= 1.0, f"Should wait 1s before retry, waited {elapsed:.2f}s"
    assert mock_client.chat.completions.create.call_count == 2, "Should call API twice"


def test_handles_invalid_model_name(openai_client):
    """Handle invalid model name gracefully with clear error"""

    invalid_model = "gpt-99-ultra-super-model-that-does-not-exist"

    with pytest.raises(NotFoundError) as exc_info:
        openai_client.chat.completions.create(
            model=invalid_model,
            messages=[{"role": "user", "content": "Test"}],
            temperature=0,
        )

    error_str = str(exc_info.value).lower()
    assert (
        "model" in error_str
        or "not found" in error_str
        or "does not exist" in error_str
    ), f"Error message should mention model issue: {exc_info.value}"


def test_handles_missing_messages_parameter(openai_client):
    """Handle missing required 'messages' parameter"""

    with pytest.raises((BadRequestError, TypeError)) as exc_info:
        openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
        )

    error_str = str(exc_info.value).lower()
    assert (
        "messages" in error_str or "required" in error_str
    ), f"Error should mention missing 'messages' parameter: {exc_info.value}"


def test_handles_invalid_parameter_type(openai_client):
    """Handle invalid parameter types with clear errors"""

    with pytest.raises((BadRequestError, TypeError, ValueError)) as exc_info:
        openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            temperature="very-hot",
        )

    error_str = str(exc_info.value).lower()
    assert (
        "temperature" in error_str or "type" in error_str or "invalid" in error_str
    ), f"Error should mention temperature or type issue: {exc_info.value}"


def test_handles_invalid_message_role(openai_client):
    """Handle invalid message roles with clear errors"""

    invalid_role = "superadmin"

    with pytest.raises(BadRequestError) as exc_info:
        openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": invalid_role, "content": "Test"}],
            temperature=0,
        )

    error_str = str(exc_info.value).lower()
    assert (
        "role" in error_str or "invalid" in error_str
    ), f"Error should mention invalid role: {exc_info.value}"
