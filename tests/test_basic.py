"""
Basic model behavior tests:
- Response presence and correctness
- Determinism at zero temperature
- Temperature effect on creativity
- Token limit enforcement
"""

from config import OPENAI_MODEL
from helpers import collect_responses


def test_model_responds(openai_client):
    """Verify model returns a response"""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
    )
    answer = response.choices[0].message.content
    assert answer is not None, "Model returned None"
    assert len(answer) > 0, "Model returned empty string"
    assert "hello" in answer.lower(), f"Expected 'hello', got: {answer}"


def test_determinism_at_zero_temperature(openai_client):
    """Same input at temp=0 gives same output"""

    prompt = "What is 5 + 3? Answer with only the number."
    answers = collect_responses(openai_client, OPENAI_MODEL, [prompt] * 3)

    assert answers[0] == answers[1] == answers[2], f"Answers differ: {answers}"
    assert "8" in answers[0], f"Wrong math answer: {answers[0]}"


def test_temperature_affects_creativity(openai_client):
    """Higher temperature gives varied responses"""

    prompt = "Describe a sunset in exactly 5 words"

    # Low temperature (deterministic)
    response_low = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer_low = response_low.choices[0].message.content

    # High temperature (creative)
    response_high = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.8,
    )
    answer_high = response_high.choices[0].message.content

    print(f"\n  Temp=0.0: {answer_low}")
    print(f"   Temp=1.8: {answer_high}")

    # They should be different (not guaranteed, but very likely)
    assert len(answer_low) > 0, "Low temp response empty"
    assert len(answer_high) > 0, "High temp response empty"
    assert answer_low != answer_high, "High and low temperature responses should differ"


def test_max_tokens_limit(openai_client):
    """Model respects token limits"""

    max_tokens = 10  # Very short limit
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Write a long story about a dragon"}],
        max_tokens=max_tokens,
        temperature=0,
    )

    answer = response.choices[0].message.content
    word_count = len(answer.split())

    print(f"\n  Response: '{answer}'")
    print(f"  Word count: {word_count}")

    # With max_tokens, response should be very brief
    assert word_count < 15, f"Response too long: {word_count} words"
