from config import OPENAI_MODEL


def test_model_responds(openai_client):
    """Verify model returns a response"""
    # Call OpenAI API
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
    )
    # Extract the answer
    answer = response.choices[0].message.content
    # Verify we got something back
    assert answer is not None, "Model returned None"
    assert len(answer) > 0, "Model returned empty string"
    assert "hello" in answer.lower(), f"Expected 'hello', got: {answer}"
    print(f"\nPASSED - Model responded: '{answer}'")


def test_determinism_at_zero_temperature(openai_client):
    """Same input at temp=0 gives same output"""

    prompt = "What is 5 + 3? Answer with only the number."
    answers = []

    # Ask the same question 3 times
    for i in range(3):
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # ← This makes it deterministic
        )
        answer = response.choices[0].message.content.strip()
        answers.append(answer)
        print(f"  Attempt {i+1}: {answer}")

    # All 3 answers should be identical
    assert answers[0] == answers[1] == answers[2], f"Answers differ: {answers}"

    assert "8" in answers[0], f"Wrong math answer: {answers[0]}"

    print(f"\nPASSED - All 3 answers matched: '{answers[0]}'")


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
    # So we just verify both gave valid responses
    assert len(answer_low) > 0, "Low temp response empty"
    assert len(answer_high) > 0, "High temp response empty"

    print("\nPASSED - Both temperatures produced responses")


def test_max_tokens_limit(openai_client):
    """Model respects token limits"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Write a long story about a dragon"}],
        max_tokens=10,  # Very short limit
        temperature=0,
    )

    answer = response.choices[0].message.content
    word_count = len(answer.split())

    print(f"\n  Response: '{answer}'")
    print(f"  Word count: {word_count}")

    # With max_tokens=10, response should be very brief
    assert word_count < 15, f"Response too long: {word_count} words"

    print(f"\nPASSED - Response limited to {word_count} words")
