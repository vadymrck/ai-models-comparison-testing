"""
Production-readiness tests:
- System prompt effectiveness
- Streaming responses
- Cross-model consistency on clear cases
"""

from config import ANTHROPIC_MODEL, OPENAI_MODEL
from helpers import call_with_delay, classify_cases, format_failures


def test_system_prompt_effectiveness(openai_client):
    """System prompt overrides default behavior"""

    test_text = "This product is amazing!"

    # Without system prompt
    response_default = call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Classify as: positive, negative, or neutral\n\n{test_text}\n\nSentiment:",
            }
        ],
        temperature=0,
    )

    # With strict system prompt
    response_system = call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment classifier. Respond ONLY with one word: positive, negative, or neutral. No explanations, no punctuation.",
            },
            {"role": "user", "content": test_text},
        ],
        temperature=0,
    )

    default_response = response_default.choices[0].message.content.strip()
    system_response = response_system.choices[0].message.content.strip()

    default_word_count = len(default_response.split())
    system_word_count = len(system_response.split())

    # Assert system prompt is more effective (shorter, cleaner)
    assert (
        system_word_count <= default_word_count
    ), f"System prompt produced longer response: {system_word_count} vs {default_word_count} words"

    assert (
        system_word_count == 1
    ), f"System prompt should enforce single word, got: '{system_response}'"

    assert system_response.lower() in [
        "positive",
        "negative",
        "neutral",
    ], f"Invalid format: '{system_response}'"


def test_streaming_response(openai_client):
    """Streaming responses work correctly"""

    test_text = "This is an excellent product that exceeded all my expectations!"

    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment classifier. Respond ONLY with one word: positive, negative, or neutral.",
            },
            {"role": "user", "content": test_text},
        ],
        temperature=0,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    full_response = "".join(chunks).strip().lower()

    assert len(chunks) >= 1, "Streaming should produce at least one chunk"
    assert full_response in [
        "positive",
        "negative",
        "neutral",
    ], f"Invalid classification: '{full_response}'"


def test_cross_model_agreement_on_clear_cases(openai_client, anthropic_client):
    """All models should agree on obvious cases"""

    clear_cases = [
        {"text": "Absolutely perfect! Best product ever!", "expected": "positive"},
        {"text": "Terrible! Completely broken and useless!", "expected": "negative"},
        {"text": "Love it! Highly recommend to everyone!", "expected": "positive"},
        {"text": "Awful! Worst purchase of my life!", "expected": "negative"},
    ]

    _, gpt_failures = classify_cases(openai_client, OPENAI_MODEL, clear_cases)
    _, claude_failures = classify_cases(
        anthropic_client, ANTHROPIC_MODEL, clear_cases, provider="anthropic"
    )

    all_failures = gpt_failures + claude_failures
    assert (
        len(all_failures) == 0
    ), f"Clear cases failed:\n{format_failures(all_failures)}"
