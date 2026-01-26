"""
Authentication tests.
Verify proper handling of invalid and missing API keys.
"""

import pytest
from config import OPENAI_MODEL
from openai import AuthenticationError, OpenAI


def test_invalid_api_key():
    """Reject invalid API keys with clear authentication error"""

    fake_api_key = "sk-fake-invalid-key-12345678901234567890"
    invalid_client = OpenAI(api_key=fake_api_key)

    with pytest.raises(AuthenticationError) as exc_info:
        invalid_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            temperature=0,
        )

    error_str = str(exc_info.value).lower()
    assert (
        "api" in error_str
        or "key" in error_str
        or "auth" in error_str
        or "401" in error_str
    ), f"Error should mention authentication issue: {exc_info.value}"


def test_missing_api_key():
    """Handle missing API key configuration with clear error"""

    invalid_client = OpenAI(api_key="")

    with pytest.raises(AuthenticationError) as exc_info:
        invalid_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            temperature=0,
        )

    error_str = str(exc_info.value).lower()
    assert (
        "api" in error_str
        or "key" in error_str
        or "auth" in error_str
        or "401" in error_str
        or "provide" in error_str
    ), f"Error should mention missing API key: {exc_info.value}"
